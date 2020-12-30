from .transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=[0, 0, 0]):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ToPercentCoords(),
            ImageTo(np.float32),
            Resize(self.size),
            PhotometricDistort(),
            ToAbsoluteCoords(),
            # RandomAffine(self.mean),
            RandomMirror(),
            # SubtractMeans(self.mean),
            ImageTo(np.uint8)
        ])

    def __call__(self, img, targets=None, **kwargs):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.augment(img, targets=targets, **kwargs)


class TestTransform:
    def __init__(self, size, mean=0.0):
        self.mean = mean
        self.size = size
        self.transform = Compose([
            ToPercentCoords(),
            ImageTo(np.float32),
            Resize(self.size),
            ToAbsoluteCoords(),
            # SubtractMeans(self.mean),
            ImageTo(np.uint8)
        ])

    def __call__(self, img, targets=None, **kwargs):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.transform(img, targets=targets, **kwargs)


class PredictionTransform:
    def __init__(self, size, mean=0.0):
        self.mean = mean
        self.size = size
        self.transform = Compose([
            ToPercentCoords(),
            ImageTo(np.float32),
            Resize(self.size),
            ToAbsoluteCoords(),
            SubtractMeans(self.mean),
            ImageTo(np.uint8)
        ])

    def __call__(self, img, targets=None, **kwargs):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.transform(img, targets=targets, **kwargs)
