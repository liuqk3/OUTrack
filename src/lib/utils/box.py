import numpy as np
import math
import torch
from .image import gaussian_radius

def iou(boxes, iou_type=1):
    """
        boxes: 2D array, [N, 4], (x1, y1, x2, y2)
        iou_type: int, can be 1 or 2
    """
    aboxes = boxes.copy()
    bboxes = boxes.copy()
    return iou_ab(aboxes, bboxes, iou_type=iou_type)

def iou_ab(abox, bbox, format='tlbr', iou_type=1):
    """
        aboxes: 2D array, [N, 4], (x1, y1, x2, y2)
        nboxes: 2D array, [m, 4], (x1, y1, x2, y2)
    """
    if abox.size == 0 or bbox.size == 0:
        return np.zeros((abox.shape[0], bbox.shape[0]))
    aboxes = abox.copy()
    bboxes = bbox.copy()

    if format == 'tlwh':
        aboxes[:, 2] += aboxes[:, 0]
        aboxes[:, 3] += aboxes[:, 1]
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

    N = aboxes.shape[0]
    M = bboxes.shape[0]
    aboxes = aboxes.reshape(N, 1, 4)
    bboxes = bboxes.reshape(1, M, 4)

    x1 = np.where(aboxes[:, :, 0] > bboxes[:, :, 0], aboxes[:, :, 0], bboxes[:, :, 0])
    y1 = np.where(aboxes[:, :, 1] > bboxes[:, :, 1], aboxes[:, :, 1], bboxes[:, :, 1])
    x2 = np.where(aboxes[:, :, 2] < bboxes[:, :, 2], aboxes[:, :, 2], bboxes[:, :, 2])
    y2 = np.where(aboxes[:, :, 3] < bboxes[:, :, 3], aboxes[:, :, 3], bboxes[:, :, 3])
    
    idx1 = (x2 > x1) * (y2 > y1)
    area = (x2 - x1) * (y2 - y1) # [N, M]

    aarea = (aboxes[:, :, 2] - aboxes[:, :, 0]) * (aboxes[:, :, 3] - aboxes[:, :, 1]) # [N, M]
    idx2 = (aboxes[:, :, 2] > aboxes[:, :, 0]) * (aboxes[:, :, 3] > aboxes[:, :, 1]) * (aarea > 0)
    aarea[aarea == 0] = np.abs(aarea.max()) + 1

    barea = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1]) # [N, M]
    idx3 = (bboxes[:, :, 2] > bboxes[:, :, 0]) * (bboxes[:, :, 3] > bboxes[:, :, 1]) * (barea > 0) # [N, M]
    barea[barea == 0] = np.abs(barea.max()) + 1

    if iou_type == 1:
        _iou = area / np.where(aarea < barea, aarea, barea) # frac{A n B}{min(A, B)}
    elif iou_type == 2:
        _iou = area / np.where(aarea > barea, aarea, barea) # frac{A n B}{max(A, B)}
    elif iou_type == 3:
        _iou = area / (aarea + barea - area) # frac{A n B}{A u B}
    elif iou_type == 4:
        _iou = area / aarea # frac{A n B}{A}
    else:
        raise ValueError('Unknown type of iou calculation {}'.format(iou_type))
    
    _iou = _iou - np.eye(N, M)
    idx = idx1 * idx2 * idx3
    _iou[~idx] = 0
    return _iou



def occlusion_boxes(boxes, iou_type=1, iou_thr=0.5):
    """
    Get the overlap boxes between given two sets of boxes
    Args:
        boxes: 2D array, [N, 4], (x1, y1, x2, y2)
        iou_type: int, can be 1 or 2
        iou_thr: float
    """
    aboxes = boxes.copy()
    bboxes = boxes.copy()

    N = aboxes.shape[0]
    M = bboxes.shape[0]
    aboxes = aboxes.reshape(N, 1, 4)
    bboxes = bboxes.reshape(1, M, 4)

    x1 = np.where(aboxes[:, :, [0]] > bboxes[:, :, [0]], aboxes[:, :, [0]], bboxes[:, :, [0]])
    y1 = np.where(aboxes[:, :, [1]] > bboxes[:, :, [1]], aboxes[:, :, [1]], bboxes[:, :, [1]])
    x2 = np.where(aboxes[:, :, [2]] < bboxes[:, :, [2]], aboxes[:, :, [2]], bboxes[:, :, [2]])
    y2 = np.where(aboxes[:, :, [3]] < bboxes[:, :, [3]], aboxes[:, :, [3]], bboxes[:, :, [3]])
    
    idx1 = (x2 > x1) * (y2 > y1)
    area = (x2 - x1) * (y2 - y1) # [N, M, 1]
    aarea = (aboxes[:, :, [2]] - aboxes[:, :, [0]]) * (aboxes[:, :, [3]] - aboxes[:, :, [1]]) # [N, M, 1]
    idx2 = (aboxes[:, :, [2]] > aboxes[:, :, [0]]) * (aboxes[:, :, [3]] > aboxes[:, :, [1]]) * (aarea > 0)
    aarea[aarea == 0] = np.abs(aarea.max()) + 1

    barea = (bboxes[:, :, [2]] - bboxes[:, :, [0]]) * (bboxes[:, :, [3]] - bboxes[:, :, [1]]) # [N, M, 1]
    idx3 = (bboxes[:, :, [2]] > bboxes[:, :, [0]]) * (bboxes[:, :, [3]] > bboxes[:, :, [1]]) * (barea > 0) # [N, M, 1]
    barea[barea == 0] = np.abs(barea.max()) + 1

    if iou_type == 1:
        _iou = area / np.where(aarea < barea, aarea, barea) # frac{A n B}{min(A, B)}
    elif iou_type == 2:
        _iou = area / np.where(aarea > barea, aarea, barea) # frac{A n B}{max(A, B)}
    elif iou_type == 3:
        _iou = area / (aarea + bboxes) # frac{A n B}{A u B}
    else:
        raise ValueError('Unknown type of iou calculation {}'.format(iou_type))
    _iou = _iou - np.eye(N, M).reshape(N, M, 1)
    idx4 = _iou > iou_thr
    idx = idx1 * idx2 * idx3 * idx4

    overlap_boxes = np.concatenate((x1, y1, x2, y2), axis=2) # [N, M, 4]
    overlap_boxes = overlap_boxes.reshape(-1, 4)
    idx = idx.reshape(-1)
    overlap_boxes = overlap_boxes[idx, :]

    return overlap_boxes


def occlusion_box_inv(abox, bbox, occ_ct):
    """get the occluded bboxe based on the known abox and occlusion center
    
    Args:
        abox: array with shape (4,), (x1, y1, x2, y2)
        bbox: array with shape (4,), (x1, y1, x2, y2). Need to be updated
        occ_ct: array with shape (2,), (cx, cy)
    

    Note that the occlusion center should be be the center of overlapped box, 
    which means: 

    cx = (max(ax1, bx1) + min(ax2, bx2)) / 2
    cy = (max(ay1, by1) + min(ay2, by2)) / 2

    And we assume that the height and width of bbox is reliable, what we need
    to do is updata the center of bbox, i.e (bx1, by1) or (bx2, by2)
    """

    assert ((abox[0] < occ_ct[0]) * (occ_ct[0] < abox[2]) * \
            (abox[1] < occ_ct[1]) * (occ_ct[1] < abox[3]) * \
            (bbox[0] < occ_ct[0]) * (occ_ct[0] < bbox[2]) * \
            (bbox[1] < occ_ct[1]) * (occ_ct[1] < bbox[3])).sum() > 0

    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]

    ''' cx = (max(ax1, bx1) + min(ax2, bx1 + bw)) / 2, bx1 need to be find'''
    bx1 = []
    # case 1: bx1 <= ax1 and bx1+bw <= ax2: --> cx = (ax1 + bx1 + bw)/2
    bx1_tmp = 2 * occ_ct[0] - abox[0] - bw
    if bx1_tmp <= abox[0] and bx1_tmp + bw <= abox[2]:
        bx1.append(bx1_tmp.copy())
    # case 2: bx1 <= ax1 and bx1 + bw > ax2 --> cx = (ax1 + ax2), no solution
    # case 3: bx1 > ax1 and bx1 + bw <= ax2 --> cx = (bx1 + bx1 + bw) / 2
    bx1_tmp = (2 * occ_ct[0] - bw) / 2
    if bx1_tmp > abox[0] and bx1_tmp + bw <= abox[2]:
        bx1.append(bx1_tmp.copy())
    # case 4: bx1 > ax1 and bx1 + bw > ax2 --> cx = (bx1 + ax2) / 2
    bx1_tmp = occ_ct[0] * 2 - abox[2]
    if bx1_tmp > abox[0] and bx1_tmp + bw > abox[2]:
        bx1.append(bx1_tmp.copy())

    ''' cy = (max(ay1, by1) + min(ay2, by1 + bh)) / 2, by1 need to be find'''
    by1 = []
    # case 1: by1 <= ay1 and by1+bh <= ay2: --> cy = (ay1 + by1 + bh)/2
    by1_tmp = 2 * occ_ct[1] - abox[1] - bh
    if by1_tmp <= abox[1] and by1_tmp + bh <= abox[3]:
        by1.append(by1_tmp.copy())
    # case 2: by1 <= ay1 and by1 + bh > ay2 --> cy = (ay1 + ay2), no solution
    # case 3: by1 > ay1 and by1 + bh <= ay2 --> cy = (by1 + by1 + bh) / 2
    by1_tmp = (2 * occ_ct[1] - bh) / 2
    if by1_tmp > abox[1] and by1_tmp + bh <= abox[3]:
        by1.append(by1_tmp.copy())
    # case 4: by1 > ay1 and by1 + bh > ay2 --> cy = (by1 + ay2) / 2
    by1_tmp = occ_ct[1] * 2 - abox[3]
    if by1_tmp > abox[1] and by1_tmp + bh > abox[3]:
        by1.append(by1_tmp.copy())

    if len(bx1) > 0:
        bx1 = sum(bx1) / len(bx1)
    else:
        bx1 = bbox[0]
    if len(by1) > 0:
        by1 = sum(by1) / len(by1)
    else:
        by1 = bbox[1]
    
    bbox_new = np.array([bx1, by1, bx1+bw, by1+bh])

    return bbox_new


def check_occ_center(abox, bbox, occ_ct, thr=0.7, return_prob=False):
    """
    Check the valid of occlusion center

    Args:
        abox: [4], (x1, y1, x2, y2)
        bbox: [4], (x1, y1, x2, y2)
        occ_ct: [2], (x, y)

    """
    x1 = max(abox[0], bbox[0])
    y1 = max(abox[1], bbox[1])
    x2 = min(abox[2], bbox[2])
    y2 = min(abox[3], bbox[3])

    if x2 <= x1 or y2 <= y1:
        if return_prob:
            return False, 0
        else:
            return False

    cx, cy = (x1+x2)/2, (y1+y2)/2
    radius = gaussian_radius((math.ceil(x2-x1), math.ceil(y2-y1)))
    prob = math.exp(-((occ_ct[0]-cx)**2 + (occ_ct[1]-cy)**2)/(2 * radius**2))
    
    if return_prob:
        return prob >= thr, prob
    else:
        return prob >= thr



############################################################
#                functions for GSM
#############################################################

def encode_boxes(boxes, im_shape, encode=True, dim_position=64, wave_length=1000, normalize=False, quantify=-1):
    """ modified from PositionalEmbedding in:
    Args:
        boxes: [bs, num_nodes, 4] or [num_nodes, 4]
        im_shape: 2D tensor, [bs, 2] or [2], the size of image is represented as [width, height]
        encode: bool, whether to encode the box
        dim_position: int, the dimension for position embedding
        wave_length: the wave length for the position embedding
        normalize: bool, whether to normalize the embedded features
        quantify: int, if it is > 0, it will be used to quantify the position of objects

    """
    batch = boxes.dim() > 2
    if not batch:
        boxes = boxes.unsqueeze(dim=0)
        im_shape = im_shape.unsqueeze(dim=0)

    if quantify > 1:
        boxes = boxes // quantify
    # in this case, the last 2 dims of input data is num_samples and 4.
    # we compute the pairwise relative postion embedings for each box
    if boxes.dim() == 3: # [bs, num_sample, 4]
        # in this case, the boxes should be tlbr: [x1, y1, x2, y2]
        device = boxes.device

        bs, num_sample, pos_dim = boxes.size(0), boxes.size(1), boxes.size(2) # pos_dim should be 4

        x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=2) # each has the size [bs, num_sample, 1]

        # handle some invalid box
        x_max[x_max<x_min] = x_min[x_max<x_min]
        y_max[y_max<y_min] = y_min[y_max<y_min]

        cx_a = (x_min + x_max) * 0.5 # [bs, num_sample_a, 1]
        cy_a = (y_min + y_max) * 0.5 # [bs, num_sample_a, 1]
        w_a = (x_max - x_min) + 1. # [bs, num_sample_a, 1]
        h_a = (y_max - y_min) + 1. # [bs, num_sample_a, 1]

        cx_b = cx_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        cy_b = cy_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        w_b = w_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]
        h_b = h_a.view(bs, 1, num_sample) # [bs, 1, num_sample_b]

        delta_x = ((cx_b - cx_a) / w_a).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        delta_y = ((cy_b - cy_a) / h_a).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        delta_w = torch.log(w_b / w_a).unsqueeze(dim=-1)  # [bs, num_sample_a, num_sample_b, 1]
        delta_h = torch.log(h_b / h_a).unsqueeze(dim=-1)  # [bs, num_sample_a, num_sample_b, 1]

        relative_pos = torch.cat((delta_x, delta_y, delta_w, delta_h), dim=-1) # [bs, num_sample_a, num_sample_b, 4]
        # if im_shape is not None:
        im_shape = im_shape.unsqueeze(dim=-1) # [bs, 2, 1]
        im_width, im_height = torch.chunk(im_shape, 2, dim=1) # each has the size [bs, 1, 1]
        x = ((cx_b - cx_a) / im_width).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        y = ((cy_b - cy_a) / im_height).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        # w = ((w_b + w_a) / (2 * im_width)).unsqueeze(dim=-1) - 0.5 # [bs, num_sample_a, num_sample_b, 1]
        # h = ((h_b + h_a) / (2 * im_height)).unsqueeze(dim=-1) - 0.5 # [bs, num_sample_a. num_sample_b, 1]
        w = ((w_b - w_a) / im_width).unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 1]
        h = ((h_b - h_a) / im_height).unsqueeze(dim=-1) # [bs, num_sample_a. num_sample_b, 1]

        relative_pos = torch.cat((relative_pos, x, y, w, h), dim=-1) # [bs, num_sample_a, num_sample_b, 8]

        if not encode:
            embedding = relative_pos
        else:
            position_mat = relative_pos # [bs, num_sample_a, num_sample_b, 8]
            pos_dim = position_mat.size(-1)
            feat_range = torch.arange(dim_position / (2*pos_dim)).to(device) # [self.dim_position / 16]
            dim_mat = feat_range / (dim_position / (2*pos_dim))
            dim_mat = 1. / (torch.pow(wave_length, dim_mat)) # [self.dim_position / 16]

            dim_mat = dim_mat.view(1, 1, 1, 1, -1) # [1, 1, 1, 1, self.dim_position / 16]
            # position_mat = position_mat.view(bs, num_sample, num_sample, pos_dim, -1) # [bs, num_sample_a, num_sample_b, 4, 1]
            position_mat = position_mat.unsqueeze(dim=-1) # [bs, num_sample_a, num_sample_b, 8, 1]
            position_mat = 100. * position_mat # [bs, num_sample_a, num_sample_b, 8, 1]

            mul_mat = position_mat * dim_mat # [bs, num_sample_a, num_sample_b, 8, dim_position / 16]
            mul_mat = mul_mat.view(bs, num_sample, num_sample, -1) # [bs, num_sample_a, num_sample_b, dim_position / 2]
            sin_mat = torch.sin(mul_mat)# [bs, num_sample_a, num_sample_b, dim_position / 2]
            cos_mat = torch.cos(mul_mat)# [bs, num_sample_a, num_sample_b, dim_position / 2]
            embedding = torch.cat((sin_mat, cos_mat), -1)# [bs, num_sample_a, num_sample_b, dim_position]

        if normalize:
            embedding = embedding / torch.clamp(torch.norm(embedding, dim=-1, p=2, keepdim=True), 1e-6)

    else:
        raise ValueError("Invalid input of boxes.")
    if not batch: # 2D tensor, [num_boxes, 4]
        embedding = embedding.squeeze(dim=0)

    return relative_pos, embedding


def inverse_encode_boxes(boxes, relative_pos, im_shape=None, quantify=-1):
    """ This function get the anchor boxes from the boxes of neighbors and relative position:
    Args:
        boxes: [bs, neighbor_k, 4] or [neighbor_k]
        relative_pos: [bs, neighbor_k, 8] or [neighbor_k, 8]
        im_shape: 2D tensor, [bs, 2], [width, height]
        quantify: int, if it is > 0, it will be used to quantify the position of objects

    """
    batch = boxes.dim() > 2
    if not batch:
        boxes = boxes.unsqueeze(dim=0)
        relative_pos = relative_pos.unsqueeze(dim=0)
        if im_shape is not None:
            im_shape = im_shape.unsqueeze(dim=0)

    if quantify > 1:
        boxes = boxes // quantify

    # in this case, the last 2 dims of input data is num_samples and 4.
    # we try to get the anchor box based on the boxes of neighbors and relative position
    if boxes.dim() == 3:

        delta_x, delta_y, delta_w, delta_h, x, y, w, h = torch.chunk(relative_pos, 8, dim=2) # each has the size [bs, neighbor_k, 1]

        x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=2)
        # handle some invalid box
        x_max[x_max<x_min] = x_min[x_max<x_min]
        y_max[y_max<y_min] = y_min[y_max<y_min]

        cx_n = (x_min + x_max) * 0.5 # [bs, neighbor_k, 1]
        cy_n = (y_min + y_max) * 0.5 # [bs, neighbor_k, 1]
        w_n = (x_max - x_min) + 1. # [bs, neighbor_k, 1]
        h_n = (y_max - y_min) + 1. # [bs, neighbor_k, 1]

        # get the size of anchors based on the first 4 elements of relative position
        w_a = w_n / torch.exp(delta_w) # [bs, neighbor_k, 1]
        h_a = h_n / torch.exp(delta_h)
        cx_a = cx_n - delta_x * w_a
        cy_a = cy_n - delta_y * h_a

        x_amax = cx_a + 0.5 * w_a
        x_amin = cx_a - 0.5 * w_a
        y_amax = cy_a + 0.5 * h_a
        y_amin = cy_a - 0.5 * h_a

        box_a_1 = torch.cat((x_amin, y_amin, x_amax, y_amax), dim=-1) # [bs, neighbor_k, 4]

        if im_shape is not None:
            # get the size of anchors based on the last 4 elements of relative position
            im_shape = im_shape.unsqueeze(dim=-1) # [bs, 2, 1]
            im_width, im_height = torch.chunk(im_shape, 2, dim=1) # each has the size [bs, 1, 1]
            cx_a = cx_n - x * im_width
            cy_a = cy_n - y * im_height
            w_a = w_n - w * im_width
            h_a = h_n - h * im_height

            x_amax = cx_a + 0.5 * w_a
            x_amin = cx_a - 0.5 * w_a
            y_amax = cy_a + 0.5 * h_a
            y_amin = cy_a - 0.5 * h_a

            box_a_2 = torch.cat((x_amin, y_amin, x_amax, y_amax), dim=-1) # [bs, neighbor_k, 4]

            box_a = (box_a_1 + box_a_2) / 2
        else:
            box_a = box_a_1

    else:
        raise ValueError("Invalid input of boxes.")

    box_a = box_a * quantify
    box_a = box_a.mean(dim=1, keepdim=True) # [bs, 1, 4]

    if not batch:
        box_a = box_a.squeeze(dim=0)

    return box_a


def box_center_dist(tlbr):
    """This function compute the Euclidean distance between the center coordinates of boxes.
    Args:
        tlbr: 2D ([N, 4]) or 3D tensor ([bs, N, 4])
    """

    if tlbr.dim() == 2:
        num_box = tlbr.size(1)
        center = (tlbr[:, 0:2] + tlbr[:, 2:4])/2 # [num_box, 2 ]
        dist = center.view(num_box, 1, 2) - center.view(1, num_box, 2) # [num_box, num_box, 2]
        dist = dist * dist
        dist = dist.sum(dim=-1) # [num_box, num_box]
    elif tlbr.dim() == 3:
        bs, num_box = tlbr.size(0), tlbr.size(1)
        center = (tlbr[:, :, 0:2] + tlbr[:, :, 2:4]) / 2 # [bs, num_box, 2]
        dist = center.view(bs, num_box, 1, 2) - center.view(bs, 1, num_box, 2) # [bs, num_box, num_box, 2]
        dist = dist * dist
        dist = dist.sum(dim=-1) # [bs, num_box, num_box]
    else:
        raise NotImplementedError

    return dist


def jitter_boxes(boxes, iou_thr=0.8, up_or_low=None, region=None):
    """Jitter some boxes.
    Args:
        boxes: 2D array, [N, 4], [x1, y1, w, h]
        iou_thr: the threshold to generate random boxes
        up_or_low: str, 'up' or 'low'
        region: [x1, y1, x2, y2]， the region the cantain all boxes

    """
    jit_boxes = boxes.copy() # x1, y1, w, h
    for i in range(jit_boxes.shape[0]):
        if (jit_boxes[i] > 0).sum() > 0: # if this box if not padded zero boxes
            jit_boxes[i, 0:4] = jitter_a_box(jit_boxes[i, 0:4].copy(), iou_thr=iou_thr, up_or_low=up_or_low, region=region)
    return jit_boxes


def jitter_a_box(one_box, iou_thr=None, up_or_low=None, region=None):
    """
    This function jitter a box
    :param box: [x1, y1, w, h]
    :param iou_thr: the overlap threshold
    :param up_or_low: string, 'up' or 'low'
    :param region: [x1, y1, x2, y2], the region that contain all boxes
    :return:
    """

    # get the std
    # 1.96 is the interval of probability 95% (i.e. 0.5 * (1 + erf(1.96/sqrt(2))) = 0.975)
    std_xy = one_box[2: 4] / (2 * 1.96)
    std_wh = 10 * np.tanh(np.log10(one_box[2:4]))
    std = np.concatenate((std_xy, std_wh), axis=0)

    if up_or_low == 'up':
        jit_boxes = np.random.normal(loc=one_box, scale=std, size=(1000, 4))
        if region is not None:
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] - 1

        overlap = iou(one_box, jit_boxes)
        index = overlap >= iou_thr
        index = np.nonzero(index)[0]

    elif up_or_low == 'low':
        jit_boxes = np.random.normal(loc=one_box, scale=std, size=(1000, 4))
        if region is not None:
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=region[0], a_max=region[2] - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=region[1], a_max=region[3] - 1)
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] - 1

        overlap = iou(one_box, jit_boxes)
        index = (overlap <= iou_thr) & (overlap >= 0)
        index = np.nonzero(index)[0]

    else:
        raise NotImplementedError

    if index.shape[0] > 0:
        choose_index = index[np.random.choice(range(index.shape[0]))]
        choose_box = jit_boxes[choose_index]
    else:
        choose_box = one_box

    return choose_box



if __name__ == '__main__':
    abox = np.array([20, 20, 40, 40])
    bbox = np.array([30, 30, 50, 50])
    occ_ct = np.array([36, 37])
