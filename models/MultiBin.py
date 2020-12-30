
import numpy as np
import torch
import torch.nn as nn


class MultiBin(object):
    def __init__(self, bins=2, overlap=0.1, device=torch.device('cpu')):
        super(MultiBin, self).__init__()
        self._bin_num = bins
        self._bin_interval = 2 * np.pi / self._bin_num
        self._angle_bins = np.zeros(self._bin_num, dtype=np.float32)
        for i in range(1, self._bin_num):
            self._angle_bins[i] = i * self._bin_interval
        self._angle_bins += self._bin_interval / 2  # center of the bin

        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self._bin_ranges = []
        for i in range(0, bins):
            self._bin_ranges.append([(i * self._bin_interval - overlap) % (2 * np.pi), \
                                    (i * self._bin_interval + self._bin_interval + overlap) % (2 * np.pi)])
        self._bin_ranges = np.array(self._bin_ranges, dtype=np.float32)
        self.toTensor(device)

    @property
    def bin_num(self):
        return self._bin_num

    @property
    def bin_interval(self):
        return self._bin_interval

    @property
    def angle_bins(self):
        return self._angle_bins

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self._bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    @torch.no_grad()
    def get_bins(self, angles):

        bin_idxs = []
        bin_ranges = self._bin_ranges[:, 1] - self._bin_ranges[:, 0]
        bin_ranges = torch.where(bin_ranges > 0, bin_ranges, bin_ranges + 2*np.pi)
        bin_angles = angles[:, None] - self._bin_ranges[:, 0]
        bin_angles = torch.where(bin_angles > 0, bin_angles, bin_angles + 2 * np.pi)
        condition = bin_angles - bin_ranges
        # bin_idxs = torch.where(condition < 0) // torch version > 1.1.0
        bin_idxs = torch.nonzero(condition < 0, as_tuple=False)

        return bin_idxs[:, 0], bin_idxs[:, 1]

    @torch.no_grad()
    def get_bin_bench_angle(self, bin_index):
        return self._angle_bins[bin_index]

    @torch.no_grad()
    def get_bins_bench_angle(self, bin_index):
        # batch_size = list(bin_index.size())[0]
        # angle_bins = [copy.deepcopy(self._angle_bins[None, :]) for _ in range(batch_size)]
        # angle_bins = torch.cat(angle_bins, dim=0)
        return self._angle_bins[bin_index]

    @torch.no_grad()
    def toTensor(self, device):
        self._angle_bins = torch.from_numpy(self._angle_bins).to(device)
        self._bin_ranges = torch.from_numpy(self._bin_ranges).to(device)

    def to(self, device):
        self._angle_bins = self._angle_bins.to(device)
        self._bin_ranges = self._bin_ranges.to(device)

