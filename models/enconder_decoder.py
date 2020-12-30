

import torch
import torch.nn as nn
import numpy as np
from models.MultiBin import MultiBin


class Coder(object):
    def __init__(self, dim_ref):
        super(Coder, self).__init__()
        self.multibin = MultiBin(2, 0.1)
        self.dim_ref = torch.tensor(dim_ref, dtype=torch.float32)

    @torch.no_grad()
    def encode_bbox(self, gt_bboxes, offsets):
        gt_ij = (gt_bboxes[:, :2] - offsets).long()
        gt_i, gt_j = gt_ij.T  # grid xy indices
        gt_bboxes[:, :2] -= gt_ij
        return gt_bboxes, gt_i, gt_j

    @torch.no_grad()
    def decode_bbox(self, pred_bboxes):
        pass

    @torch.no_grad()
    def encode_orient(self, gt_alphas, device=torch.device('cpu')):
        self.multibin.to(device)
        num = list(gt_alphas.size())[0]
        # alpha is [-pi..pi], shift it to be [0..2pi]
        Orientation = torch.zeros((num, self.multibin.bin_num * 2), dtype=torch.float32, device=device)
        Confidence = torch.zeros((num, self.multibin.bin_num,), dtype=torch.long, device=device)
        alphas = gt_alphas + np.pi
        alphas = alphas.to(device)
        bin_idxs = self.multibin.get_bins(alphas)
        bin_ben_angles = self.multibin.get_bins_bench_angle(bin_idxs[1])
        angle_diff = alphas[bin_idxs[0]] - bin_ben_angles
        Confidence[bin_idxs] = 1
        Orientation[bin_idxs[0], bin_idxs[1]*self.multibin.bin_num] = torch.cos(angle_diff).to(torch.float32)
        Orientation[bin_idxs[0], bin_idxs[1]*self.multibin.bin_num + 1] = torch.sin(angle_diff).to(torch.float32)
        return Orientation, Confidence

    @torch.no_grad()
    def decode_orient(self, pred_alphas, pred_bin_confs):
        self.multibin.to(pred_alphas.device)
        batch_size, bins = pred_bin_confs.size()
        if batch_size <= 0:
            return torch.zeros_like(pred_alphas[:, :1])
        argmax = torch.argmax(pred_bin_confs, dim=1)
        indexes_cos = (argmax * bins).long()
        indexes_sin = (argmax * bins + 1).long()
        batch_ids = torch.arange(batch_size).to(pred_bin_confs.device)
        # extract just the important bin
        alpha = torch.atan2(pred_alphas[batch_ids, indexes_sin], pred_alphas[batch_ids, indexes_cos])
        alpha += self.multibin.get_bin_bench_angle(argmax)
        # alpha is [0..2pi], shift it to be [-pi..pi]
        alpha -= np.pi
        i_pos = alpha > np.pi
        i_neg = alpha < -np.pi
        alpha[i_pos] -= 2*np.pi
        alpha[i_neg] += 2*np.pi
        return alpha

    @torch.no_grad()
    def encode_dimension(self, gt_dimensions, gt_classes, device=torch.device('cpu')):
        self.dim_ref = self.dim_ref.to(device)
        gt_dimensions = gt_dimensions.to(device)
        dim_refs = self.dim_ref[gt_classes.long()]
        return gt_dimensions/dim_refs - 1

    @torch.no_grad()
    def decode_dimension(self, pred_dimension_offsets, pred_classes):
        self.dim_ref = self.dim_ref.to(pred_classes.device)
        dim_refs = self.dim_ref[pred_classes.long()]

        return (pred_dimension_offsets + 1) * dim_refs
