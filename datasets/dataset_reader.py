from torch.utils.data import Dataset
import numpy as np
import random
import os
import cv2
import torch
from utils.ParamList import ParamList
from preprocess import transforms


class DatasetReader(Dataset):
    def __init__(self, root, config, augment=None, is_training=True, split='train'):
        super(DatasetReader, self).__init__()
        self._split = 'train' if is_training else split
        self._root = root
        self._config = config
        self._augment = augment
        self.is_training = is_training

        self._img_size = [config['img_size'][0]] * 2
        self._is_mosaic = config['is_mosaic'] if 'is_mosaic' in config else False
        self._is_rect = config['is_rect'] if 'is_rect' in config else False
        with open(os.path.join(root, 'ImageSets', '{}.txt'.format(self._split))) as f:
            self._image_files = f.read().splitlines()
            self._image_files = sorted(self._image_files)

        label_file = os.path.join(root, 'cache', 'label_{}.npy'.format(self._split))  # saved labels in *.npy file
        self._labels = np.load(label_file, allow_pickle=True)
        k_file = os.path.join(root, 'cache', 'k_{}.npy'.format(self._split))  # saved labels in *.npy file
        self._K = np.load(k_file, allow_pickle=True)
        assert len(self._image_files) == len(self._labels) == len(self._K), 'Do not match labels and images'

        sp = os.path.join(root, 'cache', 'shape_{}.npy'.format(self._split))  # shapefile path
        s = np.load(sp, allow_pickle=True)
        s = np.array(s).astype(dtype=np.int)
        self.__shapes = s
        if self._is_rect:
            m = s.max(axis=1)
            r = self._img_size[0] / m
            ns = r.reshape(-1, 1) * s
            ns_max = ns.max(axis=0)
            ns_max = np.ceil(ns_max / 32).astype(np.int) * 32
            self._img_size = ns_max

    @property
    def labels(self):
        return self._labels

    @property
    def shapes(self):
        return self.__shapes

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        indices = [index]
        if self._is_mosaic and self.is_training:
            indices += [random.randint(0, len(self._labels) - 1) for _ in range(3)]  # 3 additional image indices
        images = []
        targets = []
        transform = transforms.Compose([
            transforms.ImageTo(np.float32),
            transforms.Normalize(),
            transforms.ToPercentCoords(),
            transforms.ToXYWH(),
            transforms.ToTensor(),
            transforms.ToNCHW()
        ])
        for i, idx in enumerate(indices):
            img = self._load_image(idx)
            K = self._K[idx]
            # K = self._load_calib_param(idx)
            N = len(self._labels[idx])
            target = ParamList((img.shape[1], img.shape[0]))
            target.add_field('img_id', np.zeros((N,), dtype=np.int))
            target.add_field('class', self._labels[idx][:, 0].copy())
            target.add_field('bbox', self._labels[idx][:, 1:5].copy())
            target.add_field('dimension', self._labels[idx][:, 5:8].copy())
            target.add_field('alpha', self._labels[idx][:, 8].copy())
            target.add_field('Ry', self._labels[idx][:, 9].copy())
            target.add_field('location', self._labels[idx][:, -3:].copy())
            target.add_field('mask', np.ones((N,), dtype=np.int))
            target.add_field('K', np.repeat(K.copy().reshape(1, 9), repeats=N, axis=0))
            if self._augment is not None:
                img, target = self._augment(img, targets=target, **self._config)
            images.append(img)
            targets.append(target)
        if self._is_mosaic and self.is_training:
            img, target = self._apply_mosaic(images, targets)
        else:
            img, target = self._apply_padding(images, targets)

        # Convert
        # img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        img, target = transform(img, targets=target)
        path = os.path.join(self._root, 'training', 'image_2/{}.png'.format(self._image_files[index]))
        # print('K: ', target.get_field('K'))
        # print('shape: ', self.__shapes[index])
        # print('')
        return img, target, path, self.__shapes[index]

    def _load_image(self, index):
        path = os.path.join(self._root, 'training', 'image_2/', '{}.png'.format(self._image_files[index]))
        img = cv2.imread(path)  # BGR
        return img

    def _load_calib_param(self, index):
        path = os.path.join(self._root, 'training', 'calib/', '{}.txt'.format(self._image_files[index]))
        with open(path) as f:
            K = [line.split()[1:] for line in f.read().splitlines() if line.startswith('P2:')]
            assert len(K) > 0, 'P2 is not included in %s' % self._image_files[index]
            return np.array(K[0], dtype=np.float32)

    def _apply_mosaic(self, images, targets):
        assert len(images) == 4 and len(targets) == 4
        sw, sh = self._img_size
        c = images[0].shape[2]
        sum_rgb = np.zeros([images[0].ndim, ])
        for img in images:
            sum_rgb += np.array(cv2.mean(img))[:3]
        mean_rgb = sum_rgb / len(images)
        img4 = np.full((sh * 2, sw * 2, c), mean_rgb, dtype=np.uint8)  # base image with 4 tiles
        offsets = [(0, 0), (sw, 0), (0, sh), (sw, sh)]
        target4 = ParamList((sw, sh))
        for i, img, target in zip(range(4), images, targets):
            h, w, _ = img.shape
            pad_w = int(sw - w) // 2
            pad_h = int(sh - h) // 2
            y_st = pad_h + offsets[i][1]
            x_st = pad_w + offsets[i][0]
            img4[y_st:y_st + h, x_st:x_st + w] = img
            bbox = target.get_field('bbox')
            bbox[:, 0::2] += x_st
            bbox[:, 1::2] += y_st
            target.update_field('bbox', bbox)
            # np.clip(bbox[:, 0::2], 0, 2 * sw, out=bbox[:, 0::2])  # use with random_affine
            # np.clip(bbox[:, 1::2], 0, 2 * sh, out=bbox[:, 1::2])
            target4.merge(target)

        raff = transforms.RandomAffine2D()

        param = {
            'border': (-sh//2, -sw//2)
        }
        param.update(self._config)
        return raff(img4, target4, **param)

    def _apply_padding(self, images, targets):
        img = images[0]
        sw, sh = self._img_size
        target = targets[0]
        h, w, c = img.shape
        mean_rgb = np.array(cv2.mean(img))[:3]
        nimg = np.full((sh, sw, c), mean_rgb, dtype=np.uint8)
        pad_w = int(sw - w) // 2
        pad_h = int(sh - h) // 2
        bbox = target.get_field('bbox')
        bbox[:, 0::2] += pad_w
        bbox[:, 1::2] += pad_h
        target.update_field('bbox', bbox)
        nimg[pad_h:pad_h + h, pad_w:pad_w + w] = img
        if target.has_field('K'):
            K = target.get_field('K')
            K[:, 2] += pad_w
            K[:, 5] += pad_h
            target.update_field("K", K)

        return nimg, target

    @staticmethod
    def collate_fn(batch):
        img, target, path, shape = zip(*batch)  # transposed
        ntarget = ParamList((None, None))
        for i, t in enumerate(target):
            id = t.get_field('img_id')
            id[:, ] = i
            t.update_field('img_id', id)
            ntarget.merge(t)
        ntarget.to_tensor()
        return torch.stack(img, 0), ntarget, path, shape


def create_dataloader(path, cfg, transform=None, is_training=False, split='train'):
    dr = DatasetReader(path, cfg, augment=transform, is_training=is_training, split=split)
    batch_size = min(cfg['batch_size'], len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])  # number of workers
    data_loader = torch.utils.data.DataLoader(dr,
                                              batch_size=batch_size,
                                              num_workers=nw,
                                              pin_memory=True,
                                              shuffle=True,
                                              collate_fn=DatasetReader.collate_fn)
    return data_loader, dr


if __name__ == '__main__':
    dr = DatasetReader('./datasets/data/kitti', None)

    batch_size = min(2, len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dr,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=DatasetReader.collate_fn)
    for b_img, b_target in dataloader:
        print(dr)
