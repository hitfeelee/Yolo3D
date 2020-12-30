import torch
import torch.nn as nn
from utils import metrics_utils
from utils import ParamList
import numpy as np
from torch.nn import functional as F


class YoloLoss(object):
    def __init__(self, model):
        __model = model.module if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
            else model
        self._config = __model.md
        self._model_list = __model.model
        self._reduction = 'sum'  # Loss reduction (sum or mean)
        self._model = __model
        pass

    def __match_targets(self, pred_logits, targets):
        device = pred_logits[0].device
        detect = self._model_list[-1]  # Detect() module
        md = self._config
        t_bboxes = targets.get_field('bbox').clone().to(torch.float32)
        na, nt = detect.num_anchors, t_bboxes.shape[0]  # number of anchors, targets
        anchors = detect.anchors.clone().to(device)
        anchors = anchors / detect.in_strides.to(device).view(-1, 1, 1)
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=device).float()  # overlap offsets
        matched_indices_anchors = []
        matched_indices_targets = []
        matched_offsets = []
        for i in range(detect.num_layers):
            anchors_layer = anchors[i]
            gain = torch.tensor(pred_logits[i].shape, device=device)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            i_a, i_t, offsets = [], [], 0
            a, t = [], t_bboxes * gain
            if nt:
                r = t[None, :, -2:] / anchors_layer[:, None]  # wh ratio
                i_a, i_t = torch.nonzero(torch.max(r, 1. / r).max(2)[0] < md['anchor_t'], as_tuple=True)  # compare
                match_t = t[i_t]
                # overlaps
                gxy = match_t[:, :2]  # grid xy
                z = torch.zeros_like(gxy)
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                i_a = torch.cat((i_a, i_a[j], i_a[k], i_a[l], i_a[m]), 0)
                i_t = torch.cat((i_t, i_t[j], i_t[k], i_t[l], i_t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g
            matched_indices_anchors.append(i_a)
            matched_indices_targets.append(i_t)
            matched_offsets.append(offsets)
        return matched_indices_anchors, matched_indices_targets, matched_offsets

    def _build_targets(self, pred_logits, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        device = pred_logits[0].device
        detect = self._model_list[-1]  # Detect() module
        targets = targets.to(device).to(torch.float32)
        anchors = detect.anchors.clone().to(device)
        anchors = anchors / detect.in_strides.to(device).view(-1, 1, 1)

        matched_anch, matched_t, matched_off = self.__match_targets(pred_logits, targets)

        positives = []
        matched_targets = []
        # offset = 0
        for i in range(detect.num_layers):
            bs, na, ny, nx, ne = pred_logits[i].shape
            steps = torch.tensor([na * ny * nx, ny * nx, nx], device=device).long()
            i_a = matched_anch[i]
            i_t = matched_t[i]
            offsets = matched_off[i]
            anchors_layer = anchors[i]
            gain = torch.tensor(pred_logits[i].shape, device=device)[[3, 2, 3, 2]]  # xyxy gain

            t_bboxes_layer = targets.get_field('bbox').clone() * gain
            t_bboxes_layer = t_bboxes_layer[i_t]
            t_b_layer = targets.get_field('img_id').clone()[i_t].long()
            t_c_layer = targets.get_field('class').clone()[i_t].long()
            t_dim_layer = targets.get_field('dimension').clone()[i_t]
            t_alpha_layer = targets.get_field('alpha').clone()[i_t]
            t_mask_layer = targets.get_field('mask').clone()[i_t]
            t_bboxes_layer, t_i_layer, t_j_layer = detect.encoder_decoder.encode_bbox(t_bboxes_layer, offsets)
            t_alpha_layer, t_bin_conf = detect.encoder_decoder.encode_orient(t_alpha_layer, device)
            t_dim_layer = detect.encoder_decoder.encode_dimension(t_dim_layer, t_c_layer, device)
            # Append
            target_per_layer = ParamList.ParamList(targets.size, is_training=targets.is_training)
            target_per_layer.add_field('anchor', anchors_layer[i_a])
            target_per_layer.add_field('class', t_c_layer)
            target_per_layer.add_field('bbox', t_bboxes_layer)
            target_per_layer.add_field('dimension', t_dim_layer)
            target_per_layer.add_field('alpha', t_alpha_layer)
            target_per_layer.add_field('bin_conf', t_bin_conf)
            target_per_layer.add_field('mask', t_mask_layer)

            pred_logits[i] = pred_logits[i].view(-1, ne)
            valid = t_b_layer * steps[0] + i_a * steps[1] + t_j_layer * steps[2] + t_i_layer
            positives.append(valid)
            matched_targets.append(target_per_layer)

        return pred_logits, positives, matched_targets

    def _calc_bbox2d_losses(self, pred_logits, positives, targets):
        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # Define criteria
        loss_cls, loss_bbox, loss_obj = torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0]), \
                                        torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0]), \
                                        torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0])
        cls_pw = torch.tensor([self._config['cls_pw']], dtype=torch.float32, device=pred_logits[0].device)
        obj_pw = torch.tensor([self._config['obj_pw']], dtype=torch.float32, device=pred_logits[0].device)
        _BCEcls = nn.BCEWithLogitsLoss(pos_weight=cls_pw, reduction=self._reduction)
        _BCEobj = nn.BCEWithLogitsLoss(pos_weight=obj_pw, reduction=self._reduction)
        # focal loss
        g = self._config['fl_gamma']  # focal loss gamma
        if g > 0:
            _BCEcls, _BCEobj = FocalLoss(_BCEcls, g), FocalLoss(_BCEobj, g)

        cp, cn = smooth_BCE(eps=0.0)
        detect = self._model_list[-1]  # Detect() module
        num_valid = 0
        for i in range(detect.num_layers):
            positives_layer = positives[i]
            pred_logits_layer = pred_logits[i]
            anchors = targets[i].get_field('anchor')
            t_bboxes = targets[i].get_field('bbox')
            t_classes = targets[i].get_field('class')
            num_target = len(positives_layer)
            pred_target = torch.zeros_like(pred_logits_layer[:, 4])
            if num_target:
                num_valid += num_target
                pos_pred_logits_layer = pred_logits_layer[positives_layer]
                pxy = pos_pred_logits_layer[:, :2].sigmoid() * 2. - 0.5
                pwh = (pos_pred_logits_layer[:, 2:4].sigmoid() * 2) ** 2 * anchors
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou bbox loss
                giou = metrics_utils.bbox_iou(pbox.t(), t_bboxes, x1y1x2y2=False, GIoU=True)
                loss_bbox += (1.0 - giou).sum() if self._reduction == 'sum' else (1.0 - giou).mean()  # giou loss
                # obj score loss

                pred_target[positives_layer] = (1.0 - self._config['gr']) + \
                                               self._config['gr'] * giou.detach().clamp(0).type(
                    pred_target.dtype)  # giou ratio

                # class loss
                t = torch.full_like(pos_pred_logits_layer[:, 5:], cn)  # targets
                t[range(num_target), t_classes] = cp
                loss_cls += _BCEcls(pos_pred_logits_layer[:, 5:], t)  # BCE
            loss_obj += _BCEobj(pred_logits_layer[:, 4], pred_target)

        loss_bbox *= self._config['giou']
        loss_obj *= self._config['obj']
        loss_cls *= self._config['cls']

        if self._reduction == 'sum':
            num_valid = max(num_valid, 1)
            g = 3.0  # loss gain
            loss_obj *= (g / (num_valid * self._config['batch_size']))
            loss_cls *= (g / (num_valid * self._config['num_classes']))
            loss_bbox *= (g / num_valid)
        return loss_bbox, loss_obj, loss_cls

    def _calc_bbox3d_losses(self, pred_logits, positives, targets):
        '''

        :param pred_logits: list[shape [N, output]], output: [bin_conf{0:bin_num} orient_off{bin_num:bin_num * 2} dim_off{-3:}]
        :param positives: list[...]
        :param targets:list[...]
        :return:
        '''
        loss_conf, loss_orient, loss_dim = torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0]), \
                                           torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0]), \
                                           torch.zeros([1]).type_as(pred_logits[0]).to(pred_logits[0])
        cls_pw = torch.tensor([self._config['cls_pw']], dtype=torch.float32, device=pred_logits[0].device)
        _CEconf = nn.CrossEntropyLoss(reduction=self._reduction)
        _L1dim = nn.SmoothL1Loss(reduction=self._reduction)
        detect = self._model_list[-1]  # Detect() module
        num_valid = 0
        bin_num = detect.encoder_decoder.multibin.bin_num
        for i in range(detect.num_layers):
            positives_layer = positives[i]
            pred_logits_layer = pred_logits[i]
            pos_pred_logits_layer = pred_logits_layer[positives_layer]
            num_target = len(positives_layer)
            if num_target:
                num_valid += num_target
                # conf loss
                gt_bin_conf = targets[i].get_field('bin_conf')
                gt_bin_conf_label = torch.argmax(gt_bin_conf, dim=-1)
                loss_conf += _CEconf(pos_pred_logits_layer[:, 0:bin_num], gt_bin_conf_label)  # BCE

                # orient loss
                pred_orient = pos_pred_logits_layer[:, bin_num:bin_num + bin_num * 2]
                pred_orient = F.normalize(pred_orient.view(-1, bin_num, 2), dim=-1).view(-1, bin_num * 2)
                loss_orient += orientation_loss(pred_orient,
                                                targets[i].get_field('alpha'),
                                                gt_bin_conf,
                                                reduction=self._reduction)
                # dim loss
                pred_dim = torch.sigmoid(pos_pred_logits_layer[:, -3:]) * 2 - 1.

                loss_dim += smooth_l1_loss(pred_dim,
                                           targets[i].get_field('dimension'),
                                           beta=0.1,
                                           reduction=self._reduction)
        loss_conf *= self._config['conf']
        loss_orient *= self._config['orient']
        loss_dim *= self._config['dim']
        bs = self._config['batch_size']
        if self._reduction == 'sum':
            num_valid = max(num_valid, 1)
            g = 3.0  # loss gain
            loss_conf *= (g / (bs*num_valid))
            loss_orient *= (g / (bs*num_valid))
            loss_dim *= (g / (bs*num_valid))
        return loss_conf, loss_orient, loss_dim

    def __call__(self, pred_logits, targets):
        targets.delete_by_mask()
        pred_logits, positives, targets = self._build_targets(pred_logits, targets)
        num_bbox2d_outputs = self._model_list[-1].num_bbox_header_outputs
        pred2d_logits = [p[:, :num_bbox2d_outputs] for p in pred_logits]
        pred3d_logits = [p[:, num_bbox2d_outputs:] for p in pred_logits]
        loss_bbox, loss_obj, loss_cls = self._calc_bbox2d_losses(pred2d_logits, positives, targets)
        loss_conf, loss_orient, loss_dim = self._calc_bbox3d_losses(pred3d_logits, positives, targets)
        loss = loss_bbox + loss_obj + loss_cls + loss_conf + loss_orient + loss_dim
        return loss * self._config['batch_size'], \
               torch.tensor([loss_bbox, loss_obj, loss_cls, loss_conf, loss_orient, loss_dim, loss],
                            device=loss.device).detach()


def orientation_loss(pred_orient, gt_orient, gt_conf, reduction='sum'):
    '''

    :param pred_orient: shape(N, bin_num*2)
    :param gt_orient: shape(N, bin_num*2)
    :param gt_conf: shape(N, bin_num)
    :param reduction: 'sum' , 'mean'
    :return:
    '''
    batch_size, bins = gt_conf.size()
    indexes = torch.argmax(gt_conf, dim=1)
    indexes_cos = (indexes * bins).long()
    indexes_sin = (indexes * bins + 1).long()
    batch_ids = torch.arange(batch_size)
    # extract just the important bin

    theta_diff = torch.atan2(gt_orient[batch_ids, indexes_sin], gt_orient[batch_ids, indexes_cos])
    estimated_theta_diff = torch.atan2(pred_orient[batch_ids, indexes_sin], pred_orient[batch_ids, indexes_cos])
    loss = -1. * torch.cos(theta_diff - estimated_theta_diff).mean() + 1 if reduction == 'mean' else \
        -1. * torch.cos(theta_diff - estimated_theta_diff).sum() + len(theta_diff)
    return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.model[-1]  # Detect() module
    md = model.md
    na, nt = det.num_anchors, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]],
                       device=targets.device).float()  # overlap offsets
    # anchor tensor, same as .repeat_interleave(nt)
    at = torch.arange(na).view(na, 1).repeat(1, nt)
    anches = det.anchors.clone().view(det.num_layers, -1, 2) / det.in_strides.view(-1, 1, 1).to(targets.device)
    # anches_grid = det.anchor_grid.clone().view(na, -1, 2)/det.in_strides.view(-1, 1, 1).to(anches.device)
    style = 'rect4'
    for i in range(det.num_layers):
        anchors = anches[i]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(
                r, 1. / r).max(2)[0] < md['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat(
                    (t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat(
                    (t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat(
                    (z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def calc_loss(p, targets, model):
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    md = model.module if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
        else model
    tcls, tbox, indices, anchors = build_targets(p, targets, md)  # targets
    h = md.md  # configure
    red = 'sum'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            # giou(prediction, target)
            giou = metrics_utils.bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 -
                                                             giou).mean()  # giou loss

            # Obj
            tobj[b, a, gj, gi] = (1.0 - h['gr']) + h['gr'] * \
                                 giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if h['num_classes'] > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    bs = tobj.shape[0]  # batch size
    if red == 'sum':
        g = 3.0  # loss gain
        nt = max(nt, 1)
        lobj *= (g / nt / bs)
        lcls *= (g / nt / h['num_classes'])
        lbox *= (g / nt)

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def smooth_l1_loss(input, target, beta: float, reduction: str = "none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:

                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss