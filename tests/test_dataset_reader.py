import argparse
from utils import utils
from utils import visual_utils
import yaml
from datasets.dataset_reader import DatasetReader
import os
import torch
from preprocess.data_preprocess import TrainAugmentation
import random
import cv2
import numpy as np
from utils import visual_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data', type=str, default='./datasets/configs/kitti.yaml', help='*.yaml path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--is-mosaic', action='store_true', help='load image by applying mosaic')
    parser.add_argument('--is-rect', action='store_true', help='resize image apply rect mode not square mode')
    opt = parser.parse_args()
    opt.data = utils.check_file(opt.data)  # check file
    cfg = {}
    cfg.update(opt.__dict__)
    # dataset
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        cfg.update(data_dict)
    dataset_path = data_dict['dataset_path']

    brg_mean = data_dict['brg_mean']
    dr = DatasetReader('./datasets/data/kitti',  cfg, TrainAugmentation(cfg['img_size'][0], mean=brg_mean))

    batch_size = min(1, len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # dataloader = torch.utils.data.DataLoader(dr,
    #                                          batch_size=batch_size,
    #                                          num_workers=1,
    #                                          pin_memory=True,
    #                                          collate_fn=DatasetReader.collate_fn)
    names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for img, target, _, _ in dr:
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        src = np.copy(img)
        src3d = np.copy(img)
        src = visual_utils.cv_draw_bboxes_2d(src, target, names)
        # src3d = visual_utils.cv_draw_bboxes_3d(src3d, target, names)
        concat_img = np.concatenate([src, src3d], axis=0)
        # src = cv2.rectangle(src, (0, 0), (100, 100), (255, 0, 0), thickness=-1)
        cv2.imshow('test transform', concat_img)

        key = cv2.waitKey(1000)
        if key == 'q':
            break

