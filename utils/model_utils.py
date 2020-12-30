
import logging
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import math
import copy
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i == j else (len(j) + 1 if i.endswith('.' + j) else 0) for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)

    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def skip_prefix(state_dict, prefixs):
    if isinstance(prefixs, (tuple, list)) and len(prefixs):
        return OrderedDict({k: v for k, v in state_dict.items() if not any(k.startswith(prefix) for prefix in prefixs)})
    else:
        return state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    loaded_state_dict = skip_prefix(loaded_state_dict, prefixs=[])
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)


class CheckPointer(object):
    def __init__(
            self,
            model,
            solver=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
            mode='full',  # 'full'-the whole model, 'state-dict'-only model state-dict
            device='cpu'
    ):
        self.model = model
        self.solver = solver
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.mode = mode
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return
        model = self.solver.ema.model if hasattr(self.solver, 'ema') else self.model

        data = {"model": model.state_dict() if self.mode == 'state-dict' else model}
        if self.solver is not None:
            state_dict = self.solver.state_dict()
            state_dict['solver_name'] = self.solver.solver_name
            data["solver"] = state_dict
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pt".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True, load_solver=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        ckpt = self._load_file(f)
        # load model
        if self.mode == 'full':
            ckpt['model'] = ckpt['model'].float().state_dict()  # to FP32, filter
        self._load_model(ckpt)

        if self.solver:
            self.solver.build_optim_and_scheduler()
            if "solver" in ckpt and load_solver:
                solver_name = self.solver.solver_name
                self.logger.info("Loading solver from {}".format(f))
                cp = ckpt["solver"]
                solver_name_old = 'Adam'
                if 'solver_name' in cp:
                    solver_name_old = cp['solver_name']
                if solver_name == solver_name_old:
                    self.solver.load_state_dict(cp)

        return ckpt

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        fm = (lambda storage, loc: storage) if self.device.type == 'cpu' else (lambda storage, loc: storage.cuda(self.device))
        return torch.load(f, map_location=fm)

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint["model"])


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform model on different device from model if set
        if device:
            self.model.to(device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.model.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.model.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.model, k, getattr(model, k))