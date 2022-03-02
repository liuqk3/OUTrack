import cv2
import numpy as np
import scipy
import lap
import torch
import itertools
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time
from collections import deque

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious_ = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious_.size == 0:
        return ious_

    ious_ = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious_


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# def embedding_distance(tracks, detections, metric='cosine', embed_type=None):
#     """
#     :param tracks: list[STrack]
#     :param detections: list[BaseTrack]
#     :param metric:
#     :return: cost_matrix np.ndarray
#     """

#     cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
#     if cost_matrix.size == 0:
#         return cost_matrix
#     det_features = np.asarray([track.curr_feat_norm for track in detections], dtype=np.float)
#     #for i, track in enumerate(tracks):
#         #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat_norm.reshape(1,-1), det_features, metric))
#     track_features = np.asarray([track.smooth_feat_norm for track in tracks], dtype=np.float)
#     cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
#     return cost_matrix


def embedding_distance(tracks, detections, metric='cosine', embed_type='momentum'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :param embed_type: str, 'momentum', 'latest', 'all_mean', 'all_min'
    :return: cost_matrix np.ndarray
    """
    if metric != 'cosine':
        raise NotImplementedError('Metric {} does not implemented!'.format(metric))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = []
    det_cls = []
    track_cls = []
    for det in detections:
        det_features.append(det.curr_feat_norm)
        det_cls.append(det.class_id)
    det_features = np.array(det_features, dtype=np.float)
    det_cls = np.array(det_cls, dtype=np.int).reshape(1, -1)
    
    for track_idx in range(len(tracks)):
        track = tracks[track_idx]
        track_cls.append(track.class_id)
        if 'momentum' in embed_type:
            track_feat = track.smooth_feat_norm.reshape(1, -1)
        elif 'latest' in embed_type:
            track_feat = track.curr_feat_norm.reshape(1, -1)
        elif 'all' in embed_type:
            if isinstance(track.features, deque):
                track_feat = list(track.features)[:track.num_features]
            else:
                track_feat = track.features[:track.num_features]
            if isinstance(track_feat, list):
                track_feat = np.array(track_feat)
            track_feat = track_feat.reshape(track.num_features, -1)
            track_feat = track_feat  / np.linalg.norm(track_feat, axis=1, keepdims=True)
        else:
            raise ValueError('Unknown type of embeding type {}'.format(embed_type))
        cost_matrix_tmp = cdist(track_feat, det_features, metric) # [num_faet, num_dim]

        if 'min' in embed_type: 
            cost_matrix_tmp = np.min(cost_matrix_tmp, axis=0)
        else:
            cost_matrix_tmp = np.mean(cost_matrix_tmp, axis=0)

        cost_matrix[track_idx] = cost_matrix_tmp  # Nomalized features, [num_track, ]
    
    track_cls = np.array(track_cls, dtype=np.int).reshape(-1, 1)
    cost_matrix = np.maximum(0.0, cost_matrix)
    cost_matrix[track_cls != det_cls] = np.inf
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


############################################################################################
#                               functions for GSM
#############################################################################################
def set_neighbors(gsm, tracks, im):
    """This function try to get the neighbors for each track (detection).

    Args:
        tracks: list of STrack. It can be the list of tracklets or detections, which
            is indicated by tracks_type
        gsm: GraphSimilarity model
        im: origin image
    """
    num_track = len(tracks)
    if num_track <= 0:
        return
    
    ''' get relative pos features '''
    tlbr = [t.tlbr for t in tracks] 
    tlbr = torch.Tensor(tlbr).to(gsm.device).unsqueeze(dim=0) # [bs, num_track, 4]

    im_shape = [im.shape[1], im.shape[0]]
    im_shape = torch.Tensor(im_shape).to(gsm.device).unsqueeze(dim=0) # [bs, 2]

    # relative_pos: [bs, num_track, num_track, pos_dim]
    # pos_feat: [bs, num_track, num_track, pos_dim_out]
    # anchor_pos_feat: [bs, num_track, pos_dim_out]
    with torch.no_grad():
         relative_pos, pos_feat, anchor_pos_feat = gsm.graph_match.get_pos_feat(box=tlbr, im_shape=im_shape)

    ''' pick up neighbors based on the distanc between boxes '''    
    feat = gsm.get_reid_feature(im, [t.tlbr for t in tracks]).unsqueeze(dim=0) # [bs, num_track, dim]
    # how many neighbors to use
    neighbor_k = gsm.graph_match.get_neighbor_k(num_nodes_list=[num_track])

    if neighbor_k >= 1: # if there is a neighbor
        with torch.no_grad():
            # feat_nei: [bs, num_track, neighbor_k, feat_dim]
            # pos_feat_nei: [bs, num_track, neighbor_k, pos_dim_out]
            # weight_nei: [bs, num_track, neighbor_k]
            feat_nei, pos_feat_nei, _, _, _, weight_nei, _, _ = \
                gsm.graph_match.pick_up_neighbors(feat=feat, pos_feat=pos_feat,
                                                  relative_pos=relative_pos, box=tlbr,
                                                  neighbor_k=neighbor_k)

    else:
         feat_nei, pos_feat_nei, weight_nei = None, None, None

    # move to cpu so that some GPU memory can be saved.
    cpu_device = torch.device('cpu')
    feat = feat.to(cpu_device) if feat is not None else None
    anchor_pos_feat = anchor_pos_feat.to(cpu_device) if anchor_pos_feat is not None else None
    feat_nei = feat_nei.to(cpu_device) if feat_nei is not None else None
    pos_feat_nei = pos_feat_nei.to(cpu_device) if pos_feat_nei is not None else None
    weight_nei = weight_nei.to(cpu_device) if weight_nei is not None else None

    ''' set neighbors for each track '''
    for t_idx in range(len(tracks)):
        neighbor_tmp = {}
        neighbor_tmp['app_feat_nei'] = feat_nei[0, t_idx] if feat_nei is not None else None# [neighbor_k, feat_dim]
        neighbor_tmp['pos_feat_nei'] = pos_feat_nei[0, t_idx] if pos_feat_nei is not None else None # []
        neighbor_tmp['weight_nei'] = weight_nei[0, t_idx] if weight_nei is not None else None # [neighbor_k]
        neighbor_tmp['app_feat_anchor'] = feat[0, t_idx, :] # [app_feat_dim]
        neighbor_tmp['pos_feat_anchor'] = anchor_pos_feat[0, t_idx, :] if anchor_pos_feat is not None else None # [pos_dim_out]
 
        tracks[t_idx].update_neighbors(neighbor_tmp)

def _prepare_gsm_data(tracks, neighbor_k, device=None):
    """Prepare the data from tracks, so that can be input
    to GSM model to get the score.
    
    Args:
        tracks: list of STrack.
        neighbor_k: the number of neighbors
        device: torch.device
    """
    app_feat_anchor = [] # need to be a tensor with the size of [bs, num_track, feat_dim]
    pos_feat_anchor = [] # need to be a tensor with the size of [bs, num_track, pos_dim]
    if neighbor_k > 0:
        app_feat_nei = [] # need to be a tensor with the size of [bs, num_track, neighbor_k, feat_dim]
        pos_feat_nei = [] # need to be a tensor with the size of [bs, num_track, neighbor_k, pos_dim]
        weight_nei = [] # need to be a tensor with the size of [bs, num_track, neighbor_k]
    else:
        app_feat_nei, pos_feat_nei, weight_nei = None, None, None

    for t in tracks:
        app_feat_anchor.append(t.curr_neighbor['app_feat_anchor'])
        
        if t.curr_neighbor['pos_feat_anchor'] is not None:
            pos_feat_anchor.append(t.curr_neighbor['pos_feat_anchor'])
        else:
            pos_feat_anchor = None
        
        if neighbor_k > 0:
            app_feat_nei.append(t.curr_neighbor['app_feat_nei'][0:neighbor_k]) # [neighbor_k, feat_dim]

            pos_feat_nei.append(t.curr_neighbor['pos_feat_nei'][0:neighbor_k]) # [neighbor_k, pos_feat_dim]

            weight_nei.append(t.curr_neighbor['weight_nei'][0:neighbor_k]) # [neighbor_k]

    app_feat_anchor = torch.stack(app_feat_anchor).unsqueeze(dim=0).to(device)
    pos_feat_anchor = torch.stack(pos_feat_anchor).unsqueeze(dim=0).to(device) if pos_feat_anchor is not None else None
    app_feat_nei = torch.stack(app_feat_nei).unsqueeze(dim=0).to(device) if app_feat_nei is not None else None
    pos_feat_nei = torch.stack(pos_feat_nei).unsqueeze(dim=0).to(device) if pos_feat_nei is not None else None
    weight_nei = torch.stack(weight_nei).unsqueeze(dim=0).to(device) if weight_nei is not None else None

    return app_feat_anchor, pos_feat_anchor, app_feat_nei, pos_feat_nei, weight_nei


def graph_distance(gsm, tracks, detections):
    
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float)

    '''we first need to get the number of neighbors'''
    neighbor_k = 1e3
    for t in itertools.chain(tracks, detections):
        app_feat_nei_tmp = t.curr_neighbor['app_feat_nei']  # None or a 2D tensor with the size [neighbor_k, feat_dim]
        if app_feat_nei_tmp is None:
            neighbor_k = 0
            break
        else:
            neighbor_k = min(neighbor_k, app_feat_nei_tmp.size(0))
    
    # app_feat: [bs, num_track, feat_dim]
    # pos_feat: [bs, num_track, pos_feat_dim]
    # app_feat_nei: [bs, num_track, neighbor_k, feat_dim]
    # pos_feat_nei: [bs, num_track, neighbor_k, feat_dim]
    # weight_nei: [bs, num_tracks, neighbor_k]
    app_feat_t, pos_feat_t, app_feat_nei_t, pos_feat_nei_t, weight_nei_t = _prepare_gsm_data(tracks=tracks, neighbor_k=neighbor_k, device=gsm.device)
    app_feat_d, pos_feat_d, app_feat_nei_d, pos_feat_nei_d, weight_nei_d = _prepare_gsm_data(tracks=detections, neighbor_k=neighbor_k, device=gsm.device)

    ''' forward to get the final score'''
    with torch.no_grad():
        # ==== get the score in a loop, so that it will not out of memory ====
        total_score = []
        batch_track = int(2000 / app_feat_d.size(1))
        # batch_track = 40
        num_track = app_feat_t.size(1)
        # print('num tracks, num detections: {}, {}'.format(num_track, app_feat_d.size(1)))
        loops = num_track // batch_track
        if batch_track * loops < num_track:
            loops += 1

        for l in range(loops):
            idx1 = l * batch_track
            idx2 = min((l+1)*batch_track, num_track)

            app_feat_t_tmp = app_feat_t[:, idx1:idx2, :]
            pos_feat_t_tmp = pos_feat_t[:, idx1:idx2, :] if pos_feat_t is not None else None
            app_feat_nei_t_tmp = app_feat_nei_t[:, idx1:idx2, :, :] if app_feat_nei_t is not None else None
            pos_feat_nei_t_tmp = pos_feat_nei_t[:, idx1:idx2, :, :] if pos_feat_nei_t is not None else None
            weight_nei_t_tmp = weight_nei_t[:, idx1:idx2, :] if weight_nei_t is not None else None

            score_tmp = gsm.graph_match.get_score(feat1=app_feat_t_tmp,
                                                  pos_feat1=pos_feat_t_tmp,
                                                  feat_nei1=app_feat_nei_t_tmp,
                                                  pos_feat_nei1=pos_feat_nei_t_tmp,
                                                  weight_nei1=weight_nei_t_tmp,
                                                  feat2=app_feat_d, pos_feat2=pos_feat_d, feat_nei2=app_feat_nei_d,
                                                  pos_feat_nei2=pos_feat_nei_d, weight_nei2=weight_nei_d) # tuple
            total_score.append(score_tmp[0])
        score = torch.cat(total_score, dim=1) # [bs, num_track, num_det]

        # ==== get the score in one forward ====
        # score = self.model.graph_match.get_score(feat1=app_feat_t, pos_feat1=pos_feat_t, feat_nei1=app_feat_nei_t,
        #                                          pos_feat_nei1=pos_feat_nei_t, weight_nei1=weight_nei_t,
        #                                          feat2=app_feat_d, pos_feat2=pos_feat_d, feat_nei2=app_feat_nei_d,
        #                                          pos_feat_nei2=pos_feat_nei_d, weight_nei2=weight_nei_d) # tuple
        # # score, score_anchor, score_nei = score
        # score = score[0]# [bs, num_track, num_det]

        score = score[0] # [num_track, num_det]
        score = score.to(torch.device('cpu')).numpy()
        score = 1 - score
        return score