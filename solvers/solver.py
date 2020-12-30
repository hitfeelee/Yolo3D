import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import numpy as np
import math
from utils import model_utils


class Solver(object):
    def __init__(self, model, cfg, max_steps_burn_in=1000, apex=None):
        super(Solver, self).__init__()
        self.model = model
        self.cfg = cfg
        self.max_steps_burn_in = max_steps_burn_in
        self.apex = apex
        self.nbs = 64  # nominal batch size
        # accumulate = max(round(self.nbs / cfg['batch_size']), 1)  # accumulate loss before optimizing
        # cfg['weight_decay'] *= (cfg['batch_size'] * accumulate / self.nbs)  # scale weight_decay
        # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        # for k, v in model.named_parameters():
        #     if len(cfg['include_scopes']):
        #         if not any([k.startswith(s) for s in cfg['include_scopes']]):
        #             continue
        #     if len(cfg['exclude_scopes']):
        #         if any([k.startswith(s) for s in cfg['exclude_scopes']]):
        #             print('remove : %s' % k)
        #             continue
        #     if v.requires_grad:
        #         if '.bias' in k:
        #             pg2.append(v)  # biases
        #         elif '.weight' in k and '.bn' not in k:
        #             pg1.append(v)  # apply weight decay
        #         else:
        #             pg0.append(v)  # all else
        # params = []
        # params += [{'params': pg0}] if len(pg0) else []
        # params += [{'params': pg1, 'weight_decay': cfg['weight_decay']}] if len(pg1) else []
        # params += [{'params': pg2}] if len(pg2) else []
        # assert len(params) > 0 , 'the num of optimized params must be > 0'
        #
        # optimizer = optim.Adam(params, lr=cfg['lr0']) if cfg['adam'] else \
        #     optim.SGD(params, lr=cfg['lr0'], momentum=cfg['momentum'], nesterov=True)
        # # optimizer.add_param_group({'params': pg1, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
        # # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        # print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        # del pg0, pg1, pg2
        #
        # # Mixed precision training https://github.com/NVIDIA/apex
        # if self.apex is not None:
        #     model, optimizer = self.apex.initialize(model, optimizer, opt_level='O1', verbosity=0)
        #
        # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # self._lr_scheduler_lf = lambda x: (((1 + math.cos(
        #     x * math.pi / cfg['epochs'])) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_scheduler_lf)
        #
        # # Exponential moving average
        # ema = model_utils.ModelEMA(model)
        #
        self._lr_scheduler_lf = None
        self._optimizer = None
        self._scheduler = None
        self._ema = None
        self._global_step = 0

    @property
    def solver_name(self):
        return type(self._optimizer).__name__ + '_' + type(self._scheduler).__name__

    @property
    def ema(self):
        return self._ema

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def global_step(self):
        return self._global_step

    @property
    def learn_rate(self):
        return self._optimizer.param_groups[0]['lr']

    def load_state_dict(self, state_dict: dict):
        self._global_step = state_dict.pop('global_step') \
            if 'global_step' in state_dict else self._global_step
        if 'optimizer' in state_dict:
            cp = state_dict.pop("optimizer")
            self.optimizer.load_state_dict(cp)
        if "scheduler" in state_dict:
            cp = state_dict.pop("scheduler")
            cp = {'{}'.format(k): cp[k] for k in cp if k in ['last_epoch']}
            cp['lr_lambdas'] = [None] * len(self._optimizer.param_groups)
            self._scheduler._step_count = cp['last_epoch']
            self.scheduler.load_state_dict(cp)
            self._optimizer._step_count = self._global_step

            self.optimizer.step()
            self.scheduler.step()

    def build_optim_and_scheduler(self):
        accumulate = max(round(self.nbs / self.cfg['batch_size']), 1)  # accumulate loss before optimizing
        self.cfg['weight_decay'] *= (self.cfg['batch_size'] * accumulate / self.nbs)  # scale weight_decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if len(self.cfg['include_scopes']):
                if not any([k.startswith(s) for s in self.cfg['include_scopes']]):
                    print('solver remove params: %s' % k)
                    continue
            if len(self.cfg['exclude_scopes']):
                if any([k.startswith(s) for s in self.cfg['exclude_scopes']]):
                    print('solver remove params: %s' % k)
                    continue
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        params = []
        params += [{'params': pg0}] if len(pg0) else []
        params += [{'params': pg1, 'weight_decay': self.cfg['weight_decay']}] if len(pg1) else []
        params += [{'params': pg2}] if len(pg2) else []
        assert len(params) > 0, 'the num of optimized params must be > 0'

        optimizer = optim.Adam(params, lr=self.cfg['lr0']) if self.cfg['adam'] else \
            optim.SGD(params, lr=self.cfg['lr0'], momentum=self.cfg['momentum'], nesterov=True)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Mixed precision training https://github.com/NVIDIA/apex
        if self.apex is not None:
            model, optimizer = self.apex.initialize(self.model, optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self._lr_scheduler_lf = lambda x: (((1 + math.cos(
            x * math.pi / self.cfg['epochs'])) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_scheduler_lf)

        # Exponential moving average
        ema = model_utils.ModelEMA(self.model)

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._ema = ema
        self._global_step = 0

    def state_dict(self):
        state = {
            'global_step': self._global_step,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': {
                'last_epoch': self._scheduler.last_epoch
            }
        }
        return state

    def update(self, epoch):
        # Burn-in
        if self._global_step <= self.max_steps_burn_in:
            xi = [0, self.max_steps_burn_in]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            for j, x in enumerate(self._optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(self._global_step, xi, [0.1 if j == 2 else
                                                            0.0, x['initial_lr'] * self._lr_scheduler_lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(self._global_step, xi, [0.9, self.cfg['momentum']])

    def optimizer_step(self, loss):
        xi = [0, self.max_steps_burn_in]  # x interp
        accumulate = max(1, np.interp(self._global_step, xi, [1, self.nbs / self.cfg['batch_size']]).round())
        # Backward
        if self.apex is not None:
            with self.apex.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Optimize
        if self._global_step % accumulate == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._ema.update(self.model)
        self._global_step += 1

    def scheduler_step(self):
        # Scheduler
        self._scheduler.step()
