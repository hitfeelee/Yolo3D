import time
import torch
import torchvision
from utils import data_utils
from utils import metrics_utils
import numpy as np
import torch.nn as nn
from utils import ParamList
from preprocess import transforms


def apply_nms(prediction, num_classes=8, conf_thres=0.1, iou_thres=0.6, merge=False, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    xc = prediction[..., 4] > conf_thres  # candidates
    redundant = True  # require redundant detections
    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = num_classes > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+num_classes] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = data_utils.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = torch.nonzero(x[:, 5:5+num_classes] > conf_thres, as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, 5+num_classes:]), 1)
        else:  # best class only
            conf, j = x[:, 5:5+num_classes].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 5+num_classes:]), 1)[
                conf.view(-1) > conf_thres]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = metrics_utils.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        x = x[i]
        output[xi] = x
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def apply_nms_onnx(prediction, num_classes=8, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Compute conf
        x[:, 5:5+num_classes] = x[:, 5:5+num_classes] * x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = data_utils.xywh2xyxy(x[:, :4])

        # Detections matrix n x (7 + bin_num + bin_num*2 + 3)(bi, xyxy, conf, cls, 3d properties \
        # (bin_num + bin_num*2 + 3))
        i, j = torch.nonzero(x[:, 5:5 + num_classes] > conf_thres, as_tuple=False).t()
        bi = torch.zeros_like(j) + xi
        x = torch.cat((bi[:, None].float(), box[i], x[i, j + 5, None], j[:, None].float(), x[i, 5 + num_classes:]), 1)

        # Batched NMS
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, 1:5] + c, x[:, 5]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        # limit detections
        i = i[:max_det]
        x = x[i]
        output.append(x)

    return torch.cat(output, dim=0)


def decode_pred_logits(preds, image_size, src_size, Ks=None, coder=None):
    res = [None] * len(preds)
    Ks = Ks if isinstance(Ks, torch.Tensor) else torch.tensor(Ks).to(preds[0])
    for i, det in enumerate(preds):  # detections per image
        if det is None:
            continue

        r = ParamList.ParamList(image_size)
        K = Ks[i]
        K[:3] *= image_size[0]
        K[3:6] *= image_size[1]
        bbox = det[:, :4]
        size = src_size[i]
        scale = max(image_size) / max(src_size[i])
        pad_w = (size[0] * scale - image_size[0]) // 2
        pad_h = (size[1] * scale - image_size[1]) // 2
        bbox[:, 0::2] += pad_w
        bbox[:, 1::2] += pad_h
        K[2] += pad_w
        K[5] += pad_h
        bbox /= scale
        K[:6] /= scale
        NK = torch.zeros((3, 4)).type_as(K).to(K.device)
        NK[:, :3] = K.view(3, -1)

        score = det[:, 4]
        classes = det[:, 5]
        bin_num = (det.shape[1] - 6 - 3) // 3
        alpha = coder.decode_orient(det[:, 6+bin_num:6+3*bin_num], det[:, 6:6+bin_num])
        dim = coder.decode_dimension(det[:, -3:], classes)
        centers_x = bbox[:, 0::2].mean(dim=1)
        ray = torch.atan((centers_x - K[2]) / K[0])
        Ry = alpha + ray
        num = len(det)
        locs = []
        r.add_field('bbox', bbox)
        r.add_field('class', classes)
        r.add_field('score', score)
        r.add_field('alpha', alpha)
        r.add_field('Ry', Ry)
        r.add_field('dimension', dim)
        K = K[None, :].repeat(num, 1)
        r.add_field('K', K)
        for j in range(num):
            loc, _ = calc_regressed_bbox_3d(alpha.cpu().numpy()[j],
                                            theta_ray=ray.cpu().numpy()[j],
                                            dimension=dim.cpu().numpy()[j],
                                            bboxes=bbox.cpu().numpy()[j],
                                            proj_matrix=NK.cpu().numpy())
            locs.append(np.array(loc)[None, :])
        locs = np.concatenate(locs, axis=0)
        locs = torch.from_numpy(locs).to(bbox.device)
        r.add_field('location', locs)
        res[i] = r
    return res


def decode_pred_logits_onnx(preds, image_size, src_size, Ks, coder, batch_size=1):
    res = [None] * batch_size
    Ks = Ks if isinstance(Ks, torch.Tensor) else torch.tensor(Ks).to(preds[0])

    for i in range(batch_size):  # detections per image
        indices = i == preds[:, 0]
        det = preds[indices][:, 1:]
        if det.shape[0] <= 0:
            continue
        r = ParamList.ParamList(image_size)
        K = Ks[i]
        K[:3] *= image_size[0]
        K[3:6] *= image_size[1]
        bbox = det[:, :4]
        size = src_size[i]
        scale = max(image_size) / max(src_size[i])
        pad_w = (size[0] * scale - image_size[0]) // 2
        pad_h = (size[1] * scale - image_size[1]) // 2
        bbox[:, 0::2] += pad_w
        bbox[:, 1::2] += pad_h
        K[2] += pad_w
        K[5] += pad_h
        bbox /= scale
        K[:6] /= scale
        NK = torch.zeros((3, 4)).type_as(K).to(K.device)
        NK[:, :3] = K.view(3, -1)

        score = det[:, 4]
        classes = det[:, 5]
        bin_num = (det.shape[1] - 6 - 3) // 3
        alpha = coder.decode_orient(det[:, 6 + bin_num:6 + 3 * bin_num], det[:, 6:6 + bin_num])
        dim = coder.decode_dimension(det[:, -3:], classes)
        centers_x = bbox[:, 0::2].mean(dim=1)
        ray = torch.atan((centers_x - K[2]) / K[0])
        Ry = alpha + ray
        num = len(det)
        locs = []
        r.add_field('bbox', bbox)
        r.add_field('class', classes)
        r.add_field('score', score)
        r.add_field('alpha', alpha)
        r.add_field('Ry', Ry)
        r.add_field('dimension', dim)
        K = K[None, :].repeat(num, 1)
        r.add_field('K', K)
        for j in range(num):
            loc, _ = calc_regressed_bbox_3d(alpha.cpu().numpy()[j],
                                            theta_ray=ray.cpu().numpy()[j],
                                            dimension=dim.cpu().numpy()[j],
                                            bboxes=bbox.cpu().numpy()[j],
                                            proj_matrix=NK.cpu().numpy())
            locs.append(np.array(loc)[None, :])
        locs = np.concatenate(locs, axis=0)
        locs = torch.from_numpy(locs).to(bbox.device)
        r.add_field('location', locs)
        res[i] = r
    return res

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = data_utils.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = torch.nonzero(x[:, 5:] > conf_thres, as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = metrics_utils.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])

    return Ry.reshape([3, 3])


# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2  # L
    dy = dimension[0] / 2  # H
    dz = dimension[1] / 2  # W

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                x_corners.append(dx * i)
                y_corners.append(dy * j)
                z_corners.append(dz * k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i, loc in enumerate(location):
            corners[i, :] = corners[i, :] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])

    return final_corners


def create_birdview_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2  # L
    dy = dimension[0] / 2  # H
    dz = dimension[1] / 2  # W

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for k in [1, -1]:
            x_corners.append(dx * i)
            y_corners.append(dy)
            z_corners.append(dz * k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i, loc in enumerate(location):
            corners[i, :] = corners[i, :] + loc

    final_corners = []
    for i in range(4):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])

    return final_corners

# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, proj_matrix):
    point = np.array(pt)
    if proj_matrix.shape == (3, 4):
        point = np.append(point, 1)

    point = np.dot(proj_matrix, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2] / (point[2] + 0.0001)
    point = point.astype(np.int16)

    return point


def calc_regressed_bbox_3d(alpha, theta_ray, dimension, bboxes, proj_matrix):
    # global orientation
    orient = alpha + theta_ray
    R = rotation_matrix(orient)

    # format 2d corners
    xmin = bboxes[0]
    ymin = bboxes[1]
    xmax = bboxes[2]
    ymax = bboxes[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2  # L
    dy = dimension[0] / 2  # H
    dz = dimension[1] / 2  # W

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1, 1):
        left_constraints.append([left_mult * dx, i * dy, -switch_mult * dz])
    for i in (-1, 1):
        right_constraints.append([right_mult * dx, i * dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1, 1):
        for j in (-1, 1):
            top_constraints.append([i * dx, -dy, j * dz])
    for i in (-1, 1):
        for j in (-1, 1):
            bottom_constraints.append([i * dx, dy, j * dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4, 4])
    # 1's down diagonal
    for i in range(0, 4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4, 3], dtype=np.float)
        b = np.zeros([4, 1])

        indicies = [0, 1, 0, 1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3, 3] = RX.reshape(3)

            M = np.dot(proj_matrix, M)

            A[row, :] = M[index, :3] - box_corners[row] * M[2, :3]
            b[row] = box_corners[row] * M[2, 3] - M[index, 3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1  # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X


def batched_3d_nms(locations, dimensions, scores, rotys, batch_ids, iou_threshold=0.25):
    """
    Select best objects by the position constraint of 3d object
    """
    loc_earths = locations[:, 0::2]
    keeps = []
    for id in torch.unique(batch_ids).cpu().tolist():
        index = (batch_ids == id).nonzero(as_tuple=False).view(-1)
        keep = torch.ones_like(index)
        mask = 1 - torch.eye(len(index.cpu().tolist())).to(loc_earths.device)
        mask = mask.bool()
        loc = loc_earths[index]
        dim = dimensions[index] # h, w, l
        score = scores[index]
        score = score.unsqueeze(-1) - score.unsqueeze(0)
        roty = rotys[index]
        det_loc = loc.view(-1, 1, 2) - loc.unsqueeze(dim=0)
        det_loc = det_loc.pow_(2.)
        det_loc = torch.sqrt_(det_loc[:, :, 0] + det_loc[:, :, 1])
        # det_roty = roty.view(-1, 1, 1) - roty.unsqueeze(dim=0)
        # r_idx = det_roty > np.pi
        # det_roty[r_idx] = det_roty[r_idx] - np.pi
        # r_idx = det_roty < -np.pi
        # det_roty[r_idx] = det_roty[r_idx] + np.pi
        dim1 = dim.view(-1, 1, 3)
        dim2 = dim.unsqueeze(dim=0)
        dim_cond1 = (dim1[:, :, 1] + dim2[:, :, 1])/2.
        dim_cond2 = (dim1[:, :, 1] + dim2[:, :, 2])/2.
        dim_cond3 = (dim1[:, :, 2] + dim2[:, :, 1])/2.
        dim_cond4 = (dim1[:, :, 2] + dim2[:, :, 2])/2.
        dim_cond = (dim_cond1 + dim_cond2 + dim_cond3 + dim_cond4)/4.
        remove = (det_loc < dim_cond)
        remove = (remove & mask)
        remove_idx = (score[remove] >= 0)
        if remove_idx.sum() == 0:
            keeps.append(keep.bool())
            continue
        remove = remove.nonzero()
        remove = remove[remove_idx]
        remove = remove[:, 1].unique()
        keep[remove] = 0
        keeps.append(keep.bool())

    return torch.cat(keeps, dim=0) if len(keeps) != 0 else torch.tensor([], dtype=torch.bool, device=loc_earths.device)


def nms3d(objs):
    """
        Select best objects by the position constraint of 3d object
        """
    locations = objs.get_field('location')
    dimensions = objs.get_field('dimension')
    scores = objs.get_field('score')
    rotys = objs.get_field('Ry')
    loc_earths = locations[:, 0::2]
    keep = torch.ones_like(scores)
    mask = 1 - torch.eye(len(scores)).to(loc_earths.device)
    mask = mask.bool()
    loc = loc_earths.clone()
    dim = dimensions.clone()  # h, w, l
    score = scores.clone()
    score = score.unsqueeze(-1) - score.unsqueeze(0)
    roty = rotys.clone()
    det_loc = loc.view(-1, 1, 2) - loc.unsqueeze(dim=0)
    det_loc = det_loc.pow_(2.)
    det_loc = torch.sqrt_(det_loc[:, :, 0] + det_loc[:, :, 1])
    # det_roty = roty.view(-1, 1, 1) - roty.unsqueeze(dim=0)
    # r_idx = det_roty > np.pi
    # det_roty[r_idx] = det_roty[r_idx] - np.pi
    # r_idx = det_roty < -np.pi
    # det_roty[r_idx] = det_roty[r_idx] + np.pi
    dim1 = dim.view(-1, 1, 3)
    dim2 = dim.unsqueeze(dim=0)
    dim_cond1 = (dim1[:, :, 1] + dim2[:, :, 1]) / 2.
    dim_cond2 = (dim1[:, :, 1] + dim2[:, :, 2]) / 2.
    dim_cond3 = (dim1[:, :, 2] + dim2[:, :, 1]) / 2.
    dim_cond4 = (dim1[:, :, 2] + dim2[:, :, 2]) / 2.
    dim_cond = (dim_cond1 + dim_cond2 + dim_cond3 + dim_cond4) / 4.
    remove = (det_loc < dim_cond)
    remove = (remove & mask)
    remove_idx = (score[remove] >= 0)
    if remove_idx.sum() != 0:
        remove = remove.nonzero()
        remove = remove[remove_idx]
        remove = remove[:, 1].unique()
        keep[remove] = 0
    keep = keep.bool()
    objs.add_field('mask', keep, to_tensor=True)
    objs.delete_by_mask()
    return objs


def apply_batch_nms3d(batch_objs):
    for i, objs in enumerate(batch_objs):
        objs = nms3d(objs)
        batch_objs[i] = objs