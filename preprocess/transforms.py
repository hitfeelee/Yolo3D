
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from utils import data_utils
import math

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of bboxes.  The jaccard overlap
    is simply the intersection over union of two bboxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding bboxes, Shape: [num_bboxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, targets=None, **kwargs):
        for t in self.transforms:
            img, targets = t(img, targets=targets, **kwargs)
        return img, targets


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, targets=None):
        return self.lambd(img, targets)


class ConvertFromInts(object):
    def __call__(self, image, targets=None, **kwargs):
        return image.astype(np.float32), targets


class ToFloat32(object):
    def __call__(self, image, targets=None, **kwargs):
        return image.astype(np.float32), targets.to_float32()


class ToFloat16(object):
    def __call__(self, image, targets=None, **kwargs):
        return image.astype(np.float16), targets.to_float16()


class ImageTo(object):
    def __init__(self, dtype=np.uint8):
        self._dtype = dtype

    def __call__(self, image, targets=None, **kwargs):
        return image.astype(self._dtype), targets


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, targets=None, **kwargs):
        dtype = image.dtype
        mean = cv2.mean(image)
        image = cv2.subtract(image.astype(np.int), mean)
        image = np.clip(image, a_min=0, a_max=255).astype(dtype)
        # image = image.astype(np.int)
        # image = image - self.mean
        # image = np.clip(image, a_min=0, a_max=255)
        return image.astype(dtype), targets


class Normalize(object):
    def __call__(self, image, targets=None, **kwargs):
        r_max = np.amax(image[..., 0])
        g_max = np.amax(image[..., 1])
        b_max = np.amax(image[..., 2])
        return image / np.array([r_max, g_max, b_max]), targets


class InvNormalize(object):
    def __call__(self, image, targets=None, **kwargs):
        return image * 255, targets


class ToXYXY(object):
    def __call__(self, image, targets=None, **kwargs):
        if targets.has_field('bbox'):
            bboxes = targets.get_field("bbox")
            bboxes = data_utils.xywh2xyxy(bboxes)
            targets.update_field('bbox', bboxes)
        return image, targets


class ToXYWH(object):
    def __call__(self, image, targets=None, **kwargs):
        if targets.has_field('bbox'):
            bboxes = targets.get_field("bbox")
            bboxes = data_utils.xyxy2xywh(bboxes)
            targets.update_field('bbox', bboxes)
        return image, targets


class ToAbsoluteCoords(object):
    def __call__(self, image, targets=None, **kwargs):
        height, width, channels = image.shape
        if targets.has_field('bbox'):
            bboxes = targets.get_field("bbox")
            bboxes[:, 0::2] *= width
            bboxes[:, 1::2] *= height
        if targets.has_field('K'):
            K = targets.get_field('K')
            K[:, :3] *= width
            K[:, 3:6] *= height
            targets.update_field("K", K)
        return image, targets


class ToPercentCoords(object):
    def __call__(self, image, targets=None, **kwargs):
        height, width, channels = image.shape
        if targets is None:
            return image, targets
        if targets.has_field('bbox'):
            bboxes = targets.get_field("bbox")
            bboxes[:, 0::2] /= width
            bboxes[:, 1::2] /= height
        if targets.has_field('K'):
            K = targets.get_field('K')
            K[:, :3] /= width
            K[:, 3:6] /= height
            targets.update_field("K", K)

        return image, targets


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, targets=None, **kwargs):
        im = image.copy()
        im, _ = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, _ = distort(im)
        return self.rand_light_noise(im, targets=targets, **kwargs)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, targets=None, **kwargs):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper, (1,)).astype(np.float32)

        return image, targets


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, targets=None, **kwargs):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta, (1,)).astype(np.float32)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, targets


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, targets=None, **kwargs):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, targets


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, targets=None, **kwargs):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, targets


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, targets=None, **kwargs):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper, (1,)).astype(np.float32)
            image *= alpha
        return image, targets


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, targets=None, **kwargs):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta, (1,)).astype(np.float32)
            image += delta
        return image, targets


class ToCV2Image(object):
    def __call__(self, tensor, targets=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), targets


class ToTorchTensor(object):
    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, image, targets=None, **kwargs):
        if isinstance(image, list):
            image = [self.toTensor(src) for src in image]
        else:
            image = self.toTensor(image)
        return image, targets


class ToTensor(object):
    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, image, targets=None, **kwargs):
        timg = torch.from_numpy(image.astype(np.float32))
        return timg, targets


class ToNCHW(object):
    def __call__(self, image, targets=None, **kwargs):
        return image.permute(2, 0, 1), targets


class RandomAffine(object):
    def __init__(self, mean, range=0.5, offset=0.5):
        self.range = range
        self.offset = offset
        self.mean = mean

    def __call__(self, image, targets=None, **kwargs):
        h, w, _ = image.shape

        if random.randint(2):
            mean = cv2.mean(image)
            scale = (2 * random.random() - 1.) * self.range + 1.
            base_offset = (np.array([w, h], dtype=np.float32) - np.array([w, h], dtype=np.float32) * scale) / 2.

            offset = (2 * random.random_sample((2,)) - 1) * self.offset * np.abs(base_offset) + base_offset
            affineMat = np.eye(3)
            affineMat[:2, :2] *= scale
            affineMat[:2, 2] = offset
            image = cv2.warpAffine(image, affineMat[:2, :], dsize=(w, h), borderValue=mean)
            if targets is None:
                return image, targets
            bboxes = targets.get_field('bbox')
            bboxes *= scale
            bboxes[:, 0::2] += offset[0]
            bboxes[:, 1::2] += offset[1]
            if targets.has_field('K'):
                K = targets.get_field('K')
                K[:, :3] *= scale
                K[:, 3:6] *= scale
                K[:, 2] += offset[0]
                K[:, 5] += offset[1]
                targets.update_field("K", K)

        if targets.has_field('mask'):
            bboxes = targets.get_field('bbox')
            center_x = bboxes[:, 0::2].sum(axis=1) * 0.5
            center_y = bboxes[:, 1::2].sum(axis=1) * 0.5
            # w_h = bboxes[:, 2:] - bboxes[:, :2]
            # low_xy = -0.25 * w_h
            # if isinstance(low_xy, torch.Tensor):
            #     high_xy = -1. * low_xy + torch.tensor([[w, h]]).type_as(low_xy)
            # elif isinstance(low_xy, np.ndarray):
            #     high_xy = -1. * low_xy + np.array([[w, h]])
            # else:
            #     raise Exception('bboxes type not in [np.ndarrray, torch.Tensor]')
            # index1 = (center_x >= low_xy[:, 0]) & (center_x < high_xy[:, 0])
            # index2 = (center_y >= low_xy[:, 1]) & (center_y < high_xy[:, 1])
            index1 = (center_x < 0) | (center_x >= w)
            index2 = (center_y < 0) | (center_y >= h)
            index = (index1 | index2)
            masks = targets.get_field('mask')
            masks[index] = 0
        return image, targets


class RandomAffine2D(object):
    def __call__(self, image, targets=None, **kwargs):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]
        h, w, _ = image.shape
        degrees = kwargs['degrees'] if 'degrees' in kwargs else 0.
        translate = kwargs['translate'] if 'translate' in kwargs else 0.
        scale = kwargs['scale'] if 'scale' in kwargs else 0.5
        shear = kwargs['shear'] if 'shear' in kwargs else 0.0
        border = kwargs['border'] if 'border' in kwargs else (-h//4, -w//4)
        height = image.shape[0] + border[0] * 2  # shape(h,w,c)
        width = image.shape[1] + border[1] * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 - scale/2)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(image.shape[1] / 2, image.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * image.shape[1] + border[1]  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * image.shape[0] + border[0]  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        if targets.has_field('mask'):
            bboxes = np.copy(targets.get_field('bbox'))
            n = len(bboxes)
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # reject warped points outside of image
            # xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            # xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)
            mask = targets.get_field('mask')
            mask[np.bitwise_not(i)] = 0
            bboxes[i] = xy[i]
            center_x = bboxes[:, 0::2].sum(axis=1) * 0.5
            center_y = bboxes[:, 1::2].sum(axis=1) * 0.5
            index1 = (center_x < 0) | (center_x >= width)
            index2 = (center_y < 0) | (center_y >= height)
            index = (index1 | index2)
            mask[index] = 0
            targets.update_field('mask', mask)
            targets.update_field('bbox', bboxes)

        return image, targets


class RandomMirror(object):
    def __call__(self, image, targets=None, **kwargs):
        _, width, _ = image.shape
        if random.randint(2):
            image = np.fliplr(image)
            if targets is None:
                return image, targets
            bboxes = targets.get_field('bbox')
            bboxes[:, 0::2] = width - bboxes[:, [2, 0]]
            if targets.has_field('K'):
                K = targets.get_field('K')
                K[:, 2] = width - K[:, 2] - 1
                targets.update_field('K', K)
            if targets.has_field('alpha'):
                alphas = targets.get_field('alpha')
                idx_pos = alphas >= 0
                idx_neg = alphas < 0
                alphas[idx_pos] = -1. * alphas[idx_pos] + np.pi
                alphas[idx_neg] = -1. * alphas[idx_neg] - np.pi
            if targets.has_field('Ry'):
                Ry = targets.get_field('Ry')
                idx_pos = Ry >= 0
                idx_neg = Ry < 0
                Ry[idx_pos] = -1. * Ry[idx_pos] + np.pi
                Ry[idx_neg] = -1. * Ry[idx_neg] - np.pi
            if targets.has_field('location'):
                loc = targets.get_field('location')
                loc[:, 0] *= -1

        return image, targets


class Resize(object):
    def __init__(self, size=640):
        self.size = size

    def __call__(self, image, targets=None, **kwargs):
        h, w, _ = image.shape
        if isinstance(self.size, (tuple, list)):
            size = self.size
        else:
            rate = self.size / max(h, w)
            size = (int(w*rate), int(h*rate))
        if isinstance(image, list):
            image = [cv2.resize(src=src, dsize=size, interpolation=cv2.INTER_LINEAR) for src in image]
        else:
            image = cv2.resize(src=image, dsize=size, interpolation=cv2.INTER_LINEAR)
        return image, targets


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image