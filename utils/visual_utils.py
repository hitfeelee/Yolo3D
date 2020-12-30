import numpy as np
import cv2
from postprocess import postprocess
from enum import Enum
from pathlib import Path
import glob
import random
import torch
import matplotlib.pyplot as plt
import os
import math
from utils import data_utils
from copy import copy
from utils import utils
import matplotlib

cv2.setNumThreads(0)


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    DINGXIANG = (204, 164, 227)


KITTI_COLOR_MAP = (
    cv_colors.RED.value,
    cv_colors.GREEN.value,
    cv_colors.BLUE.value,
    cv_colors.PURPLE.value,
    cv_colors.ORANGE.value,
    cv_colors.MINT.value,
    cv_colors.YELLOW.value,
    cv_colors.DINGXIANG.value
)


def getColorMap():
    colormap = [[255, 255, 255]]
    for i in range(3 * 9 - 1):
        if i % 9 == 0:
            continue
        k = i // 9
        m = i % 9
        color = [255, 255, 255]
        color[k] = (color[k] >> m)
        colormap.append(color)
    return colormap


def cv_draw_bboxes_2d(image, bboxes_2d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes_2d_array = bboxes_2d.numpy()
    bboxes = bboxes_2d_array.get_field('bbox')
    classes = bboxes_2d_array.get_field('class').astype(np.int)
    scores = bboxes_2d_array.get_field('score') if bboxes_2d_array.has_field('score') else np.ones_like(classes)

    for cls, score, bbox in zip(classes, scores, bboxes):
        color = color_map[cls]
        label = '{}:{:.2f}'.format(label_map[cls] if label_map is not None else cls, score)
        image = plot_one_box(bbox, image, color=color, label=label, line_thickness=1)

    return image


def cv_draw_bboxes_3d(img, bboxes_3d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes_3d_array = bboxes_3d.numpy()
    classes = bboxes_3d_array.get_field('class').astype(np.int)
    N = len(classes)
    scores = bboxes_3d_array.get_field('score') if bboxes_3d_array.has_field('score') else np.ones((N,), dtype=np.int)
    locations = bboxes_3d_array.get_field('location')
    Rys = bboxes_3d_array.get_field('Ry')
    dimensions = bboxes_3d_array.get_field('dimension')
    Ks = bboxes_3d_array.get_field('K')

    for cls, loc, Ry, dim, score, K in zip(classes, locations, Rys, dimensions, scores, Ks):
        label = label_map[cls] if label_map is not None else cls
        cv_draw_bbox_3d(img, K.reshape((3, -1)), Ry, dim, loc, label, score, color_map[cls])
    return img


def cv_draw_bbox_3d(img, proj_matrix, ry, dimension, center, cls, score, color, thickness=1):
    tl = thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    R = postprocess.rotation_matrix(ry)

    corners = postprocess.create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    box_3d = []
    for corner in corners:
        point = postprocess.project_3d_pt(corner, proj_matrix)
        box_3d.append(point)

    # TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0], box_3d[2][1]), color, tl)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0], box_3d[6][1]), color, tl)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0], box_3d[4][1]), color, tl)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0], box_3d[6][1]), color, tl)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0], box_3d[3][1]), color, tl)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0], box_3d[5][1]), color, tl)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0], box_3d[3][1]), color, tl)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0], box_3d[5][1]), color, tl)

    for i in range(0, 7, 2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i + 1][0], box_3d[i + 1][1]), color, tl)

    front_mark = np.array([[box_3d[0][0], box_3d[0][1]],
                           [box_3d[1][0], box_3d[1][1]],
                           [box_3d[3][0], box_3d[3][1]],
                           [box_3d[2][0], box_3d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))

    label = '{}:({:.2f},{:.2f},{:.2f})'.format(cls,center[0], center[1], center[2])
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    box_3d = np.array(box_3d).min(axis=0)
    c1 = (box_3d[0], box_3d[1])
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def cv_draw_bbox3d_birdview(img, bboxes_3d, scaleX=0.2, scaleY=0.2, color=(255, 0, 0)):
    if scaleY is None:
        scaleY = scaleX

    bboxes_3d_array = bboxes_3d.numpy()
    classes = bboxes_3d_array.get_field('class')
    locations = bboxes_3d_array.get_field('location')
    Rys = bboxes_3d_array.get_field('Ry')
    dimensions = bboxes_3d_array.get_field('dimension')

    for cls, loc, Ry, dim in zip(classes, locations, Rys, dimensions):
        if cls not in [0]:
            continue
        cv_draw_bbox_birdview(img, Ry, dim, loc, scaleX, scaleY, color)
    return img


def cv_draw_bbox_birdview(img, ry, dim, loc, scaleX=0.2, scaleY=0.2, color=(255, 0, 0)):
    h, w, _ = img.shape
    offsetX = w / 2
    offsetY = h
    R = postprocess.rotation_matrix(ry)
    corners = postprocess.create_birdview_corners(dim, location=loc, R=R)
    corners = np.array(corners)
    # transform to pixel
    rr = np.array([1./scaleX, -1./scaleY])
    tt = np.array([offsetX, offsetY])
    corners = corners[:, 0::2] * rr + tt
    index = [0, 1, 3, 2, 0]
    for i in range(len(index) - 1):
        i0 = index[i]
        i1 = index[i+1]
        cv2.line(img, (int(corners[i0][0]), int(corners[i0][1])), (int(corners[i1][0]), int(corners[i1][1])), color,
                 thickness=1, lineType=cv2.LINE_AA)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h): return tuple(
        int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = data_utils.xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            # check for confidence presence (gt vs pred)
            conf = None if gt else image_targets[:, 6]

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label,
                                 color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w,
                                                   block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(
            mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(
        scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig('LR.png', dpi=200)


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = data_utils.xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' %
                   (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


# from utils.utils import *; plot_study_txt()
def plot_study_txt(f='study.txt', x=None):
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in ['coco_study/study_coco_yolov5%s.txt' % x for x in ['s', 'm', 'l', 'x']]:
        y = np.loadtxt(f, dtype=np.float32, usecols=[
                       0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95',
             't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        for i in range(7):
            ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
            ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, :j], y[3, :j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=Path(f).stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [33.5, 39.1, 42.5, 45.9, 49., 50.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid()
    ax2.set_xlim(0, 30)
    ax2.set_ylim(28, 50)
    ax2.set_yticks(np.arange(30, 55, 5))
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig('study_mAP_latency.png', dpi=300)
    plt.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_labels(labels, out_dir='./'):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classees, boxes

    def hist2d(x, y, n=100):
        xedges, yedges = np.linspace(
            x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
        hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
        return np.log(hist[xidx, yidx])

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=int(c.max() + 1))
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(os.path.join(out_dir, 'labels.png'), dpi=200)
    plt.close()


# from utils.utils import *; plot_evolution_results(hyp)
def plot_evolution_results(hyp):
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = utils.fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(12, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={
                  'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)


# from utils.utils import *; plot_results_overlay()
def plot_results_overlay(start=0, stop=0):
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5',
         'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(
            f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


# from utils.utils import *; plot_results()
def plot_results(start=0, stop=0, bucket='', id=(), labels=()):
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov5#reproduce-our-training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' %
                 (bucket, x) for x in id]
    else:
        files = glob.glob('results*.txt') + \
            glob.glob('../../Downloads/results*.txt')
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(
                f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else Path(f).stem
                ax[i].plot(x, y, marker='.', label=label,
                           linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig('results.png', dpi=200)