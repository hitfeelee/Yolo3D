import argparse

# from utils.datasets import *
# from utils.utils import *
import cv2
import os
from pathlib import Path
import glob
# from alfred.dl.torch.common import device
import shutil
import torch
import time
import torchvision
import random
import numpy as np
from utils import metrics_utils
from datasets.kitti import LoadStreams
from utils.torch_utils import select_device
from models.yolo import Model
import pickle
from datasets.dataset_reader import DatasetReader
from preprocess.data_preprocess import TestTransform
from postprocess import postprocess
from models.enconder_decoder import Coder
import yaml
from utils import visual_utils
import onnxruntime


def detect(cfg):
    # Initialize

    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = torch.load(opt.weights, map_location=device)['model']
    model.to(device).eval()

    if half:
        model.half()  # to FP16
    else:
        model.to(torch.float32)
    if device.type == 'cpu':
        model.to(torch.float32)

    session = onnxruntime.InferenceSession(cfg['onnx'])
    # 2. Get input/output name
    input_name = session.get_inputs()[0].name  # 'image'
    output_name = session.get_outputs()[0].name  # 'boxes'
    print('onnx input name: {}, output name: {}'.format(input_name, output_name))

    # Set Dataloader
    dataset_path = cfg['dataset_path']
    dataset = DatasetReader(dataset_path, cfg, augment=TestTransform(cfg['img_size'][0], mean=cfg['brg_mean']),
                            is_training=False, split='test')
    # Get names and colors
    names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # encoder_decoder = Coder(cfg['dim_ref'])
    encoder_decoder = model.model[-1].encoder_decoder
    # Run inference
    t0 = time.time()
    videowriter = None
    if cfg['write_video']:
        videowriter = cv2.VideoWriter('res.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 1, (1242, 750))
        max = 1000
        cnt = 0
    for img, targets, path, _ in dataset:
        src = cv2.imread(path)
        ori_img = np.copy(img)
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t_st = time.time()
        # pred = model(img)[0]
        # onnxruntime result
        pred_onnx = session.run([output_name], {input_name: [ori_img]})[0]
        pred_onnx = torch.tensor(pred_onnx).to(device)
        bi = targets.get_field('img_id')
        K = targets.get_field('K')
        Ks = []
        for i in np.unique(bi):
            indices = i == bi
            Ks.append(K[indices][None, 0])
        Ks = np.concatenate(Ks, axis=0)

        pred = postprocess.decode_pred_logits_onnx(pred_onnx, (img.shape[3], img.shape[2]),
                                              [(src.shape[1], src.shape[0])], Ks, encoder_decoder)
        # postprocess.apply_batch_nms3d(pred)
        t_end = time.time()
        # print('pred after nms:', len(pred), pred[0].shape)

        src3d = np.copy(src)
        birdview = np.zeros((2*src.shape[0], src.shape[0], 3), dtype=np.uint8)
        if pred[0] is not None:
            src = visual_utils.cv_draw_bboxes_2d(src, pred[0], names)
            src3d = visual_utils.cv_draw_bboxes_3d(src3d, pred[0], names)
            birdview = visual_utils.cv_draw_bbox3d_birdview(birdview, pred[0], color=(255, 0, 0))
            birdview = visual_utils.cv_draw_bbox3d_birdview(birdview, targets, color=(0, 0, 255))
        concat_img = np.concatenate([src, src3d], axis=0)
        concat_img = np.concatenate([concat_img, birdview], axis=1)
        cv2.imshow('test transform', concat_img)
        if cfg['write_video']:
            if cnt < max:
                concat_img = cv2.resize(concat_img, (1242, 750))
                # concat_img = concat_img[:, :, ::-1]
                videowriter.write(concat_img)
            cnt += 1

        print('the inference time of model is ', t_end - t_st)
        if cv2.waitKey(1000) == ord('q'):
            break
    if cfg['write_video']:
        videowriter.release()
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov4.5s.pt', help='model.pt path')
    parser.add_argument('--onnx', type=str, default='weights/yolov4.5s.onnx', help='model.onnx path')
    parser.add_argument('--data', type=str, default='./datasets/configs/kitti.yaml', help='*.yaml path')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size',  nargs='+', type=int, default=[640, 640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--is-mosaic', action='store_true', help='load image by applying mosaic')
    parser.add_argument('--is-rect', action='store_true', help='resize image apply rect mode not square mode')
    parser.add_argument('--write-video', action='store_true', help='write detect result to video')
    opt = parser.parse_args()
    # opt.img_size = check_img_size(opt.img_size)
    print(opt)
    cfg = opt.__dict__
    # dataset
    with open(cfg['data']) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)  # data config
        cfg.update(data_cfg)
    with torch.no_grad():
        detect(cfg)
