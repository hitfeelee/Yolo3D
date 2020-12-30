import argparse
import torch.distributed as dist
import torch.nn.functional as F
from solvers.solver import Solver
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import yaml
import numpy as np
import random
import time
import tqdm
import math
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import utils
from utils import torch_utils
from utils import model_utils
from utils import anchor_utils
from utils import visual_utils

from datasets.dataset_reader import create_dataloader
from preprocess.data_preprocess import TrainAugmentation, TestTransform
from models import losses
from utils.ParamList import ParamList
wdir = 'weights' + os.sep  # weights dir
os.makedirs(wdir, exist_ok=True)
last = 'model3d_5m_last_transconv_11_25'
best = 'model3d_5m_best_transconv_11_25'

config_path = './configs/train_config.yaml'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed


def train(config):
    utils.init_seeds(1)
    results_file = os.path.join(config['logdir'], 'results.txt')
    # Remove previous results
    for f in glob.glob(os.path.join(config['logdir'], 'train_batch*.jpg')) + glob.glob(results_file):
        os.remove(f)

    epochs = config['epochs']  # 300
    batch_size = config['batch_size']  # 64
    weights = config['weights']  # initial training weights
    imgsz, imgsz_test = config['img_size']
    strides = config['detect_strides']
    num_classes = config['num_classes']
    if config['only_3d']:
        config['notest'] = True
        config['include_scopes'] = ['model.24.bbox3d_headers']
        config['giou'] = 0.
        config['obj'] = 0.
        config['cls'] = 0.
    elif config['only_2d']:
        config['exclude_scopes'] = ['model.24.bbox3d_headers']
        config['conf'] = 0.
        config['orient'] = 0.
        config['dim'] = 0.

    config['cls'] *= num_classes / 80.  # scale coco-tuned config['cls'] to current dataset
    gs = int(max(strides))

    # dataset
    with open(config['data']) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    dataset_path = data_dict['dataset_path']

    # Trainloader
    test_cfg = {}
    test_cfg.update(config)
    dataloader, dataset = create_dataloader(dataset_path,
                                            config,
                                            transform=TrainAugmentation(cfg['img_size'][0], mean=config['brg_mean']),
                                            is_training=True)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < num_classes, \
        'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, num_classes, config['cfg'])

    # Testloader
    test_cfg['is_rect'] = True
    test_cfg['is_mosaic'] = False
    testloader = create_dataloader(dataset_path,
                                   test_cfg,
                                   transform=TestTransform(cfg['img_size'][0], mean=config['brg_mean']),
                                   is_training=False,
                                   split='test')[0]

    # Create model
    model = Model(config).to(device)
    nb = len(dataloader)  # number of batches
    max_step_burn_in = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    solver = Solver(model, config, max_steps_burn_in=max_step_burn_in, apex=amp)
    losser = losses.YoloLoss(model)
    # Load Model
    start_epoch, best_fitness = 0, 0.0
    checkpointer = model_utils.CheckPointer(model, solver, save_dir='./weights', save_to_disk=True, device=device)
    if weights.endswith('.pt'):  # pytorch format
        ckpt = checkpointer.load(weights, use_latest=False, load_solver=(not config['resume']))
        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt
        if not config['resume']:
            start_epoch = ckpt['epoch'] + 1
        best_fitness = ckpt['best_fitness']
        del ckpt
    else:
        solver.build_optim_and_scheduler()

    if tb_writer:
        # Class frequency
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        visual_utils.plot_labels(labels, config['logdir'])
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    if not config['noautoanchor']:
        anchor_utils.check_anchors(dataset, model=model, thr=config['anchor_t'], imgsz=imgsz)

    # Start training
    t0 = time.time()
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(7, device=device)  # mean losses
        print(('\n' + '%10s' * 12) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'conf', 'orient', 'dim', 'total',
                                      'targets', 'img_size', 'lr'))
        pbar = tqdm.tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            targets.delete_by_mask()
            targets.to_float32()
            targ = ParamList(targets.size, True)
            targ.copy_from(targets)
            img_id = targets.get_field('img_id')
            classes = targets.get_field('class')
            bboxes = targets.get_field('bbox')

            targets = torch.cat([img_id.unsqueeze(-1), classes.unsqueeze(-1), bboxes], dim=-1)
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 1.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            solver.update(epoch)
            # Multi-scale
            if config['multi_scale']:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            # loss, loss_items = losses.calc_loss(pred, targets.to(device), model)
            loss, loss_items = losser(pred, targ)
            # print(loss_items)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            solver.optimizer_step(loss)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 10) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1], solver.learn_rate)
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = os.path.join(config['logdir'], 'train_batch%g.jpg' % ni)  # filename
                result = visual_utils.plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ============================================================================================
        solver.scheduler_step()
        # mAP
        solver.ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not config['notest'] or final_epoch:  # Calculate mAP
            results, maps, times = test.test(config['data'],
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and config['data'].endswith(os.sep + 'kitti.yaml'),
                                             model=solver.ema.model,
                                             logdir=config['logdir'],
                                             dataloader=testloader)

        # Write
        with open(os.path.join(results_file), 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = utils.fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not config['nosave']) or final_epoch
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read()}

            # Save last, best and delete
            checkpointer.save(last, **ckpt)
            if (best_fitness == fi) and not final_epoch:
                checkpointer.save(best, **ckpt)
            del ckpt

        # end epoch =================================================================================================
    # end training
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cfg', type=str, default='models/configs/yolo3d_5s.yaml', help='*.yaml path')
    parser.add_argument('--data', type=str, default='data/coco_tl.yaml', help='*.data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank')
    parser.add_argument('--exclude-scopes', nargs='+', type=str, default=[],
                        help='do not train the params in exclude_scopes')
    parser.add_argument('--include-scopes', nargs='+', type=str, default=[],
                        help='only train the params in include_scopes')
    parser.add_argument('--logdir', type=str, default='./runs', help='do not train the params in exclude_scopes')
    parser.add_argument('--is-mosaic', action='store_true', help='load image by applying mosaic')
    parser.add_argument('--is-rect', action='store_true', help='resize image apply rect mode not square mode')
    parser.add_argument('--only-3d', action='store_true', help='only train 3d')
    parser.add_argument('--only-2d', action='store_true', help='only train 2d, that is, excluding 3d')
    opt = parser.parse_args()
    # opt.weights = last if opt.resume else opt.weights
    opt.cfg = utils.check_file(opt.cfg)  # check file
    opt.data = utils.check_file(opt.data)  # check file
    print(opt)
    cfg = None
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg.update(opt.__dict__)

    with open(opt.cfg) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)  # model config
        cfg.update(model_cfg)

    # dataset
    with open(cfg['data']) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)  # data config
        cfg.update(data_cfg)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # Train
    tb_writer = SummaryWriter(comment=opt.name)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # assert cfg is None, 'Please check config for training!'

    train(cfg)

