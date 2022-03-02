from matplotlib.pyplot import axis
import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import copy
import torch.nn.functional as F

from models.model import create_model, create_gsm, load_model
from models.decode import mot_decode, mot_occlusion_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.gmm import GMM
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process, occlusion_post_process
from utils.image import get_affine_transform, crop_image
from models.utils import _tranpose_and_gather_feat
from utils.debugger import Debugger
from utils.box import iou, iou_ab, occlusion_boxes, occlusion_box_inv, check_occ_center

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, img_shape, buffer_size=30, feat_dim=128, use_gmm=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.class_id = 1
        self.tracklet_len = 0
        self.img_shape = img_shape # height, width

        self.buffer_size = buffer_size
        self.num_features = 0
        self.smooth_feat_norm = None
        self.curr_feat_norm = None
        # self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        
        if use_gmm and temp_feat is not None:
            self.gmm = GMM(sample_dim=feat_dim, num_samples=buffer_size)
            self.features = self.gmm.init_samples()
        else:     
            self.gmm = None
            self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)
        self.curr_neighbor = {}

    def update_neighbors(self, neighbor):
        self.curr_neighbor = neighbor

    def has_neighbor(self):
        if len(list(self.curr_neighbor.keys())) == 0:
            return False 
        return True

    def update_features(self, feat):
        if feat is not None:
            self.curr_feat = feat
            feat_norm = feat / np.linalg.norm(feat)
            self.curr_feat_norm = feat_norm
            if self.smooth_feat_norm is None:
                self.smooth_feat_norm = feat_norm
            else:
                self.smooth_feat_norm = self.alpha * self.smooth_feat_norm + (1 - self.alpha) * feat_norm
            self.smooth_feat_norm /= np.linalg.norm(self.smooth_feat_norm)
            
            if self.gmm is None:
                self.features.append(feat)
            else:
                merged_sample, new_sample, merged_sample_id, new_sample_id = \
                self.gmm.update_sample_space_model(self.features, feat, self.num_features)
                if new_sample_id >= 0 :
                    self.features[new_sample_id:new_sample_id+1, :] = new_sample
                if merged_sample_id >= 0:
                    self.features[merged_sample_id:merged_sample_id+1, :] = merged_sample
            if self.num_features < self.buffer_size:
                self.num_features += 1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.update_neighbors(new_track.curr_neighbor)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
            self.update_neighbors(new_track.curr_neighbor)

    def update_box(self, tlbr):
        tlwh = tlbr.copy()
        tlwh[2:] = tlwh[2:] - tlwh[:2]
        # if self.kalman_filter is not None:
        #     self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(tlwh))
        # else:
        #     self._tlwh = tlwh
        self._tlwh = tlwh

    def output(self, clip_box=False, min_box_area=100):
        ret = {}
        tlbr = self.tlbr
        tlwh = self.tlwh
        if clip_box:
            tlbr[0::2] = np.clip(tlbr[0::2], a_min=0, a_max=self.img_shape[1]-1) # x
            tlbr[1::2] = np.clip(tlbr[1::2], a_min=0, a_max=self.img_shape[0]-1) # y
            tlwh = tlbr.copy()
            tlwh[2:4] = tlwh[2:4] - tlwh[0:2]

        if tlbr[2] <= tlbr[0] or tlbr[3] <= tlbr[1]: # boxes out of the view
            return None
        
        if self.track_id > 0: # output for tracklets
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] <= min_box_area or vertical:
                return None

        ret['bbox'] = tlbr
        ret['tlwh'] = tlwh
        ret['score'] = self.score
        ret['object_id'] = self.track_id
        ret['state'] = self.get_state()
        
        return ret


    def sample_candidate(self, num_sample=64, format='tlbr', image=None):
        """This function used to sample the candidates for lost tracks"""

        tlwh = self.tlwh

        # return tlwh[np.newaxis, :]  # [1, 4]

        height, width = self.img_shape[0], self.img_shape[1]

        np.random.seed(410)
        # get the std
        # 1.96 is the interval of probability 95% (i.e. 0.5 * (1 + erf(1.96/sqrt(2))) = 0.975)
        # std_xy = tlwh[2: 4] / (2 * 1.96)
        std = np.sqrt(tlwh[2] * tlwh[3]) / (2 * 1.96)
        std_xy = np.array([std, std])

        std_wh = np.tanh(np.log10(tlwh[2:4]))
        # std_wh = np.tanh(np.log10(tlwh[2:4]))/2
        std = np.concatenate((std_xy, std_wh), axis=0)

        if (std < 0).sum() > 0:
            jit_boxes = tlwh[np.newaxis, :]  # [1, 4]
        else:
            jit_boxes = np.random.normal(loc=tlwh, scale=std, size=(2000, 4))
            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] + jit_boxes[:, 0:2] - 1 # x1, y1, x2, y2
            jit_boxes[:, 0] = np.clip(jit_boxes[:, 0], a_min=0, a_max=width - 1)
            jit_boxes[:, 1] = np.clip(jit_boxes[:, 1], a_min=0, a_max=height - 1)
            jit_boxes[:, 2] = np.clip(jit_boxes[:, 2], a_min=0, a_max=width - 1)
            jit_boxes[:, 3] = np.clip(jit_boxes[:, 3], a_min=0, a_max=height - 1)

            jit_boxes[:, 2:4] = jit_boxes[:, 2:4] - jit_boxes[:, 0:2] + 1  # x1, y1, w, h
            index = (jit_boxes[:, 2] > 1) * (jit_boxes[:, 3] > 1) * (jit_boxes[:, 3]/jit_boxes[:, 2] < 3)
            if index.sum() > 0:
                jit_boxes = jit_boxes[index]
                overlap = iou_ab(tlwh[None, :], jit_boxes, format='tlwh', iou_type=3)[0]
                index = overlap > 0.75
                if index.sum() > 0:
                    jit_boxes = jit_boxes[index]
                    jit_boxes = jit_boxes[0:min(jit_boxes.shape[0], num_sample)]  # tlwh
                    jit_boxes = np.concatenate((jit_boxes, tlwh[np.newaxis, :]))
                else:
                    jit_boxes = tlwh[np.newaxis, :]  # [1, 4]
            else:
                jit_boxes = tlwh[np.newaxis, :]  # [1, 4]


        if format == 'tlbr':
            jit_boxes[:, 2:4] += jit_boxes[:, 0:2]
        elif format == 'tlwh':
            pass
        else:
            raise ValueError('Unknown format of boxes {}'.format(format))

        if image is not None:
            from lib.utils.visualization import plot_detections
            import matplotlib.pyplot as plt
            if isinstance(image, torch.Tensor):
                image = image.to(torch.device('cpu')).numpy()
            image = image.astype(np.uint8)
            image = plot_detections(image=image, tlbrs=jit_boxes, scores=None, color=(0, 255, 0))
            tlbr = self.tlbr()
            tlbr = tlbr[np.newaxis, :]
            image = plot_detections(image=image, tlbrs=tlbr, scores=None, color=(255, 0, 0))
            plt.clf()
            plt.imshow(image)
            plt.pause(1)

        return jit_boxes

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def wh(self):
        ret = self.tlwh[2:].copy()
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def to_center(self, clip=True): # center coordinates
        # self.img_shape = img_shape # height, width
        ret = self.tlbr
        if clip:
            ret[[0, 2]] = np.clip(ret[[0, 2]], 0, self.img_shape[1] - 1)
            ret[[1, 3]] = np.clip(ret[[1, 3]], 0, self.img_shape[0] - 1)
        ret[0] = (ret[0] + ret[2])/2
        ret[1] = (ret[1] + ret[3])/2
        ret = ret[:2]
        return ret

    def to_center_norm(self, clip=True):
        ret = self.to_center(clip=clip)
        ret[0] /= self.img_shape[1]
        ret[1] /= self.img_shape[0]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model, epoch = load_model(self.model, opt.load_model, return_epoch=True)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.epoch = epoch
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.track_id_to_occ = {}

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.frame_rate = frame_rate
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        if self.opt.debug > 0:
            self.debugger = Debugger(opt)

        if int(opt.lost_frame_range) >= 1:
            self.lost_frame_range = int(opt.lost_frame_range)
        else:
            self.lost_frame_range = max(1, int(opt.lost_frame_range * self.frame_rate))
        
        if opt.load_gsm != '':
            self.gsm = create_gsm(opt.gsm_config)
            self.gsm = load_model(self.gsm, opt.load_gsm, model_param_key='model')
            self.gsm = self.gsm.to(opt.device)
            self.gsm.device = opt.device
            self.gsm.eval()
        else:
            self.gsm = None

    def reset(self, frame_rate):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.track_id_to_occ = {}
        self.frame_rate = frame_rate
        self.buffer_size = int(frame_rate / 30.0 * self.opt.track_buffer)
        
        self.frame_id = 0
        if self.opt.debug > 0:
            self.debugger.reset()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy() # B, K, D
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def post_process_occ(self, occ_centers, meta):
        occ_classes = 1
        occ_centers = occ_centers.detach().cpu().numpy() # B, K, D
        occ_centers = occ_centers.reshape(1, -1, occ_centers.shape[2])
        occ_centers = occlusion_post_process(
            occ_centers.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], occ_classes)
        for j in range(1, occ_classes + 1):
            occ_centers[0][j] = np.array(occ_centers[0][j], dtype=np.float32).reshape(-1, 3)
        return occ_centers[0]

    def find_lost_basedon_occlusion(self, lost_tracks, tracked_tracks, occlusion=None):
        """
        Find occluded object based on iou
        """
        occluded = []
        if len(lost_tracks) == 0:
            return occluded
        if occlusion is None:
            for track in lost_tracks:
                if track.track_id in self.track_id_to_occ.keys():
                    occluded.append(track)
        else:
            if len(occlusion) > 0:
                occ_cts = np.array(occlusion[:, :2], dtype=np.float32).reshape(-1, 1, 2)
                lost_ltrb = np.array([track.tlbr for track in lost_tracks], dtype=np.float32).reshape(1, -1, 4)
                covered_lost = (lost_ltrb[:, :, 0] < occ_cts[:, :, 0]) * (occ_cts[:, :, 0] < lost_ltrb[:, :, 2]) * \
                               (lost_ltrb[:, :, 1] < occ_cts[:, :, 1]) * (occ_cts[:, :, 1] < lost_ltrb[:, :, 3]) # [num_ct, num_lost]
                
                tracked_ltrb = np.array([track.tlbr for track in tracked_tracks], dtype=np.float32).reshape(1, -1, 4)
                covered_tracked = (tracked_ltrb[:, :, 0] < occ_cts[:, :, 0]) * (occ_cts[:, :, 0] < tracked_ltrb[:, :, 2]) * \
                                  (tracked_ltrb[:, :, 1] < occ_cts[:, :, 1]) * (occ_cts[:, :, 1] < tracked_ltrb[:, :, 3]) # [num_ct, num_lost]

                occ_cts = occ_cts.reshape(-1, 2)
                tracked_ltrb = tracked_ltrb.reshape(-1, 4)
                lost_ltrb = lost_ltrb.reshape(-1, 4)
                prob_thr = 0.7
                for lost_i in range(len(lost_tracks)):
                    occ_idx = np.where(covered_lost[:, lost_i])[0]
                    valids = []
                    probs = []
                    ltrbs = []
                    occs = []
                    for occ_i in occ_idx:
                        tracked_idx = np.where(covered_tracked[occ_i, :])[0]
                        for tracked_i in tracked_idx:
                            valid, prob = check_occ_center(tracked_ltrb[tracked_i], lost_ltrb[lost_i], occ_cts[occ_i], thr=prob_thr, return_prob=True)
                            valids.append(valid)
                            probs.append(prob)
                            ltrbs.append(tracked_ltrb[tracked_i])
                            occs.append(occ_cts[occ_i])
                    # get the largest prob
                    if len(probs) > 0:
                        idx = probs.index(max(probs))
                        if valids[idx]:
                            lost_box = occlusion_box_inv(abox=ltrbs[idx], bbox=lost_ltrb[lost_i], occ_ct=occs[idx])
                            lost_tracks[lost_i].update_box(lost_box)

                        if lost_tracks[lost_i].track_id in self.track_id_to_occ.keys():
                            occluded.append(lost_tracks[lost_i])
        return occluded

    def find_lost_basedon_gsm(self, lost_tracks, support_tracks, im,
                            strict=True, threshold=0.0):
        """This function get the neighbors for detections. Once a detection is matched with a
        track, the information of neighbors will be delivered to tracks.

        Args:
            lost_tracks: tracks: list of lost STrack
            support_tracks: list of Stracks, which are used to find the lost tracks
            im_wh: ndarray, origin image, H, W, 3
            strict: bool, if true, we will perform association between true tracks and true detections.
                If False, we will do association between all (including padded)tracks and all (including
                padded) detections.
            threshold: float, the min distance used to find the lost tracks
        """
        # support_tracks = [st for st in support_tracks if st.tracks]

        num_lost = len(lost_tracks)
        num_sup = len(support_tracks)
        if num_lost == 0 or num_sup == 0:
            return []

        neighbor_k = self.gsm.graph_match.get_neighbor_k([num_sup + 1])  # plus 1 if for the candidate
        if neighbor_k < 1:
            return []

        # note that the features and neighbors of lost tracks are not updated
        # while the features and neighbors of support tracks should be updated

        # prepare im shape
        im_shape = [im.shape[1], im.shape[0]] # [width, height]
        im_shape = torch.Tensor(im_shape).to(self.gsm.device) # [2]
        im_shape = im_shape.view(1, 2) # [bs, 2]

        # get the box and app feature of support tracks
        tlbr_s = [st.tlbr for st in support_tracks]
        app_feat_s = self.gsm.get_reid_feature(im, tlbr_s) # [num_sup, app_feat_dim]
        # pad box
        if self.gsm.pad_boxes:
            raise NotImplementedError
        else:
            if not strict:
                raise ValueError('The model is trained with no padded boxes, so it only support strict assign!')
        tlbr_s = torch.Tensor(tlbr_s).to(self.gsm.device) # [num_sup, 4]

        # forward to get the app features and position features for lost tracks
        valid_lost_tracks = []
        for lt in lost_tracks:
            if not lt.has_neighbor():
                continue
            valid_lost_tracks.append(lt)

        # batch_num = 4
        # batch_num = int((15000 / (len(support_tracks) + 1)) / 65)
        batch_num = int((10000 / (len(support_tracks) + 1)) / 65)
        num_valid_lost = len(valid_lost_tracks)
        loops = num_valid_lost // batch_num
        if loops * batch_num < num_valid_lost:
            loops += 1

        occluded_trakcs = []
        for l in range(loops):
            lost_tracks_filter = valid_lost_tracks[l*batch_num:min((l+1)*batch_num, num_valid_lost)]

            num_candidate = []
            tlbr_candidate = []
            for lt in lost_tracks_filter:
                # prepare the information of lost track in current frame candidate
                candidate = lt.sample_candidate(format='tlbr', num_sample=16) # [num_candicate, 4]
                tlbr_candidate.append(candidate)

                num_can = candidate.shape[0]
                num_candidate.append(num_can)

            # get app feature
            tlbr_candidate = np.concatenate(tlbr_candidate, axis=0)
            tlbr_candidate = torch.Tensor(tlbr_candidate).to(self.gsm.device)
            with torch.no_grad():
                app_feat_can = self.gsm.get_reid_feature(im, tlbr_candidate)  # [num_candidate, feat_dim]

            # get pos feature
            tlbr_s_tmp = tlbr_s.view(1, num_sup, 4).repeat(tlbr_candidate.size(0), 1, 1) # [num_candidate, num_sup, 4]
            tlbr_can = tlbr_candidate.view(tlbr_candidate.size(0), 1, 4) # [num_candidate, 1, 4]
            tlbr_can = torch.cat((tlbr_s_tmp, tlbr_can), dim=1) # [num_candidate, num_sup+1, 4]
            # pos_feat_cur: [num_candidate, num_sup+1, num_sup+1, pos_feat_dim]
            # anchor_pos_feat_cur: [num_candidate, num_sup+1, pos_feat_dim]
            with torch.no_grad():
                #print('num_candicate: {}, num_support {}'.format(tlbr_can.size(0), tlbr_can.size(1)))
                relative_pos_can, pos_feat_can, anchor_pos_feat_can = self.gsm.graph_match.get_pos_feat(box=tlbr_can, im_shape=im_shape)

            # handle the lost tracks
            idx1 = 0
            idx2 = 0
            for i in range(len(lost_tracks_filter)):
                lt = lost_tracks_filter[i]

                num_can = num_candidate[i]
                idx2 = idx2+num_can

                # prepare neighbor information of lost track in previous frame
                app_feat_pre = lt.curr_neighbor['app_feat_anchor'] # 1D tensor, [app_feat_dim]
                anchor_pos_feat_pre = lt.curr_neighbor['pos_feat_anchor'] # 1D tensor, [pos_feat_dim]

                app_feat_nei_pre = lt.curr_neighbor['app_feat_nei'] # [neighbor_k, app_feat_dim] or None
                pos_feat_nei_pre = lt.curr_neighbor['pos_feat_nei'] # [neighbor_k, pos_feat_dim] or None
                weight_nei_pre = lt.curr_neighbor['weight_nei'] # [neighbor_k] or None
                # import pdb; pdb.set_trace()
                neighbor_k_tmp = min(neighbor_k, app_feat_nei_pre.size(0))
                if neighbor_k_tmp < 1:
                    continue

                # prepare the information of lost track in current frame candidate
                # get app feature
                app_feat_cur = app_feat_can[idx1:idx2]  # [num_candidate, feat_dim]
                app_feat_s_tmp = app_feat_s.view(1, num_sup, -1).repeat(num_can, 1, 1) # [num_candidate, num_sup, app_feat_dim]

                app_feat_cur = app_feat_cur.view(num_can, 1, -1) # [num_candidate, 1, app_feat_dim]
                app_feat_cur = torch.cat((app_feat_s_tmp, app_feat_cur), dim=1) # [num_candidate, num_sup+1, app_feat_dim]

                # get pos feature
                tlbr_cur = tlbr_can[idx1:idx2] # [num_candidate, num_sup+1, 4]
                relative_pos_cur = relative_pos_can[idx1:idx2] # [num_candidate, num_sup+1, num_sup+1, 8]
                pos_feat_cur = pos_feat_can[idx1:idx2] # [num_candidate, num_sup+1, num_sup+1, pos_feat_dim]
                anchor_pos_feat_cur = anchor_pos_feat_can[idx1:idx2] # [num_candidate, num_sup+1, pos_feat_dim]

                # app_feat_nei: [num_candidate, num_sup+1, neighbor_k, app_feat_dim]
                # pos_feat_nei: [num_candidate, num_sup+1, neighbor_k, app_feat_dim]
                # tlbr_nei: [num_candidate, num_sup+1, neighbor_k, 4]
                # weight_nei: [num_candidate, num_sup+1, neighbor_k], the weight to get the neighbors
                # feat_nei, pos_feat_nei, relative_pos_nei, box_nei, ids_nei, nei_v, nei_idx, weight_logits
                app_feat_nei_cur, pos_feat_nei_cur, _, tlbr_nei_cur, _,  weight_nei_cur, _, _ = \
                    self.gsm.graph_match.pick_up_neighbors(feat=app_feat_cur, pos_feat=pos_feat_cur,
                                                             relative_pos=relative_pos_cur,
                                                             box=tlbr_cur, neighbor_k=neighbor_k_tmp)

                # prepare the data to get the score
                app_feat_cur = app_feat_cur[:, -1:, :] # [num_candidate, 1, app_feat_dim]
                anchor_pos_feat_cur = anchor_pos_feat_cur[:, -1:, :] if anchor_pos_feat_cur is not None else None # [num_candidate, 1, pos_feat_dim]
                app_feat_nei_cur = app_feat_nei_cur[:, -1:, :, :] # [num_candidate, 1, neighbor_k, app_feat_dim]
                pos_feat_nei_cur = pos_feat_nei_cur[:, -1:, :, :] if pos_feat_nei_cur is not None else None # [num_candidate, 1, neighbor_k, pos_feat_dim]
                tlbr_nei_cur = tlbr_nei_cur[:, -1, :, :] if tlbr_nei_cur is not None else None # [num_candidate, 1, neighbor_k, 4]
                weight_nei_cur = weight_nei_cur[:, -1:, :] if weight_nei_cur is not None else None # [num_candidate, 1, neighbor_k]

                app_feat_pre = app_feat_pre.view(1, 1, -1).repeat(num_can, 1, 1).to(self.gsm.device)  # [num_candidate, 1, app_feat_dim]
                if anchor_pos_feat_pre is not None:
                    anchor_pos_feat_pre = anchor_pos_feat_pre.to(self.gsm.device)
                    anchor_pos_feat_pre = anchor_pos_feat_pre.view(1, 1, anchor_pos_feat_pre.size(-1)).repeat(num_can, 1, 1) # [num_candidate, 1, pos_feat_dim]
                if app_feat_nei_pre is not None:
                    app_feat_nei_pre = app_feat_nei_pre[0:neighbor_k_tmp].to(self.gsm.device) # [neighbor_k, app_feat_dim]
                    app_feat_nei_pre = app_feat_nei_pre.view(1, 1, neighbor_k_tmp, app_feat_nei_pre.size(-1)).repeat(num_can, 1, 1, 1)  # [num_candidate, 1, neighbor_k, app_feat_dim]
                if pos_feat_nei_pre is not None:
                    pos_feat_nei_pre = pos_feat_nei_pre[0:neighbor_k_tmp].to(self.gsm.device) # [neighbor_k, pos_feat_dim]
                    pos_feat_nei_pre = pos_feat_nei_pre.view(1, 1, neighbor_k_tmp, pos_feat_nei_pre.size(-1)).repeat(num_can, 1, 1, 1) # [num_candidate, 1, neighbor_k, pos_feat_dim]
                if weight_nei_pre is not None:
                    weight_nei_pre = weight_nei_pre[0:neighbor_k_tmp].to(self.gsm.device) # [neighbor_k]
                    weight_nei_pre = weight_nei_pre.view(1, 1, weight_nei_pre.size(-1)).repeat(num_can, 1, 1) # [num_candidate, 1, neighbor_k]
                with torch.no_grad():
                    score = self.gsm.graph_match.get_score(feat1=app_feat_pre, pos_feat1=anchor_pos_feat_pre, feat_nei1=app_feat_nei_pre,
                                                       pos_feat_nei1=pos_feat_nei_pre, weight_nei1=weight_nei_pre,
                                                       feat2=app_feat_cur, pos_feat2=anchor_pos_feat_cur, feat_nei2=app_feat_nei_cur,
                                                       pos_feat_nei2=pos_feat_nei_cur, weight_nei2=weight_nei_cur) # tuple

                score = score[0]  # [num_candidate, 1,1]
                dist = 1 - score.squeeze(-1).squeeze(-1) # num_candidate
                min_dist, min_idx = torch.topk(dist, k=1, largest=False)
                if min_dist.item() < threshold * 0.7:
                    occluded_trakcs.append(lt)
                idx1 = idx2     

        return occluded_trakcs




    def refresh_occluded_objects(self, tracks):
        N = len(tracks)
        if N == 0:
            return
        ltrb = np.array([t.tlbr for t in tracks], dtype=np.float32).reshape(N, 4)
        occ = iou(ltrb, iou_type=4) # [N, N]
        occ = occ.max(axis=1) # [N]


        occ_thr = 0.2 # 0.3 # 0.5
        border_thr = 0.05
        track_id_to_occ = {}
        for i in range(N):
            center_norm = tracks[i].to_center_norm(clip=True)
            if ((border_thr < center_norm) * (center_norm < 1 - border_thr)).sum() == 2 and occ[i] > occ_thr:
                track_id_to_occ[tracks[i].track_id] = {'occ': occ[i], 'start_frame': self.frame_id}
        
        # update the occluded tracks with a buffer 
        self.track_id_to_occ.update(track_id_to_occ)
        track_id_to_occ = {}
        for k, v in self.track_id_to_occ.items():
            if self.frame_id - v['start_frame'] <= self.lost_frame_range:
                track_id_to_occ[k] = v
        
        self.track_id_to_occ = track_id_to_occ

    def update(self, im_blob, img0, det_meta=None, img_info=None):
        # import pdb; pdb.set_trace()
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections, occlusions & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            # occlusions 
            if self.opt.occlusion:
                occ_hm = output['occlusion'].sigmoid_()
                occ_off = output['occlusion_offset'] if self.opt.occlusion_offset else None
                occ_cts = mot_occlusion_decode(occ_hm, reg=occ_off, K=4*self.opt.K) # B, K, 4, # x, y, score, class
                occ_cts = self.post_process_occ(occ_cts, meta)[1] # N, 3
                keep = occ_cts[:, 2] >= self.opt.occlusion_thres
                occ_cts = occ_cts[keep] # N, 3
            else:
                occ_hm = None
                occ_cts = None
            
            # reid features
            if not self.opt.not_reid:
                id_feature = output['id']
                id_feature = F.normalize(id_feature, dim=1)
            else:
                id_feature = None

            # tracking with private detection or do public tracking just as CenterTrack
            if self.opt.track_type in ['private_track', 'public_track']:
                identities = None
                visibility = None
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                reg = output['reg'] if self.opt.reg_offset else None
                dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
                if not self.opt.not_reid:
                    # import pdb; pdb.set_trace()
                    id_feature = _tranpose_and_gather_feat(id_feature, inds)
                    id_feature = id_feature.squeeze(0)
                    id_feature = id_feature.cpu().numpy()

                dets = self.post_process(dets, meta)
                dets = self.merge_outputs([dets])[1] # x1, y1, x2, y2, score
                inds = inds.cpu().numpy()[0] # K

                # track with public detrection like CenterTrack
                if self.opt.track_type == 'public_track':
                    if det_meta['detection'].shape[1] > 8:
                        visibility = copy.deepcopy(det_meta['detection'][:, 8])
                    pub_dets = copy.deepcopy(det_meta['detection'][:, 2:6]) # xywh
                    pub_dets[:, 2:4] = pub_dets[:, 2:4] + pub_dets[:, 0:2] # xyxy
                    # map to origin image size
                    pub_dets[:, 0] = pub_dets[:, 0] - det_meta['padw']
                    pub_dets[:, 2] = pub_dets[:, 2] - det_meta['padw']
                    pub_dets[:, 1] = pub_dets[:, 1] - det_meta['padh']
                    pub_dets[:, 3] = pub_dets[:, 3] - det_meta['padh']
                    pub_dets[:, 0:4] = pub_dets[:, 0:4] / det_meta['ratio']   
            # else: # tracking with provided detections
            elif self.opt.track_type == 'provide_track':
                hm = None
                if det_meta['detection'].shape[1] > 8:
                    visibility = copy.deepcopy(det_meta['detection'][:, 8])

                identities = copy.deepcopy(det_meta['detection'][:, 1])
                dets = copy.deepcopy(det_meta['detection'][:, 2:7]) # xywh,score, Kx5
                dets[:, 2:4] = dets[:, 0:2] + dets[:, 2:4] # ltrb,score
                if not self.opt.not_reid:
                    # pick up reid_features using box center
                    dets_out = copy.deepcopy(dets[:, 0:4])
                    dets_out = dets_out / self.opt.down_ratio
                    cts = (dets_out[:, 0:2] + dets_out[:, 2:4]) / 2
                    cts[:, 0] = np.clip(cts[:, 0], a_min=0, a_max=meta['out_width']-1)
                    cts[:, 1] = np.clip(cts[:, 1], a_min=0, a_max=meta['out_height']-1)
                    cts = np.asarray(cts, dtype=np.int64)
                    inds = cts[:, 1] * meta['out_width'] + cts[:, 0]
                    inds = torch.as_tensor(inds, dtype=torch.long).to(id_feature.device).unsqueeze(dim=0)
                    # import pdb; pdb.set_trace()
                    id_feature = _tranpose_and_gather_feat(id_feature, inds)
                    id_feature = id_feature.squeeze(0)
                    id_feature = id_feature.cpu().numpy()      

                    # # pick up reid_features using detected peak center
                    # inds = copy.deepcopy(det_meta['detection'][:, -1]) # the index, K
                    # inds = torch.as_tensor(inds, dtype=torch.long).to(id_feature.device).unsqueeze(dim=0)
                    # # import pdb; pdb.set_trace()
                    # id_feature = _tranpose_and_gather_feat(id_feature, inds)
                    # id_feature = id_feature.squeeze(0)
                    # id_feature = id_feature.cpu().numpy()   

                    inds = inds.cpu().numpy()[0]       
                
                # map to origin image size
                dets[:, 0] = dets[:, 0] - det_meta['padw']
                dets[:, 2] = dets[:, 2] - det_meta['padw']
                dets[:, 1] = dets[:, 1] - det_meta['padh']
                dets[:, 3] = dets[:, 3] - det_meta['padh']
                dets[:, 0:4] = dets[:, 0:4] / det_meta['ratio']   
            # public track like CenterTrack
            else: 
                raise ValueError("unknown type of tracking {}".format(self.opt.track_type))
                

        # perform detection
        if self.opt.only_det:
            if self.opt.debug > 0:
                self.debugger.add_image_with_bbox(img0, bbox=dets[dets[:, 4] > self.opt.conf_thres], img_id='det', bbox_type='jde_det')
                if 'video_name' and 'frame_id' in img_info:
                    self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='det')
                self.debugger.show_all_imgs()

            detections = []
            for i in range(len(dets)):
                tlwh = dets[i, 0:4].copy()
                tlwh[2:4] = tlwh[2:4] - tlwh[0:2]
                det = {
                    'bbox': dets[i, 0:4],
                    'score': dets[i, 4],
                    'tlwh': tlwh,
                    'index': -1 if inds is None else inds[i],
                    'object_id': -1 if identities is None else int(identities[i]),
                    'visibility': -1 if visibility is None else visibility[i],
                    'state': 'detection'
                }
                if not self.opt.not_reid:
                    det['feat'] = id_feature[i]
                detections.append(det)
            
            return detections

            # dets[:, 2:4] = dets[:, 2:4] - dets[:, 0:2] # K, 5
            # inds = inds[:, np.newaxis]
            # dets = np.concatenate((dets, inds), axis=1)
            # return dets

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds] if id_feature is not None else None

        remain_inds = (dets[:, 0] <= width-1) * (dets[:, 1] <= width-1) * (dets[:, 2] > 0) * (dets[:, 3] > 0)
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds] if id_feature is not None else None


        # import pdb; pdb.set_trace()
        if self.opt.debug > 0:
            if det_meta is not None:
                det_p = det_meta['detection'][:, 2:7]
                det_p[:, 2:4] = det_p[:, 0:2] + det_p[:, 2:4]
                det_p = det_p[det_p[:, -1] > self.opt.conf_thres]
                self.debugger.add_image_with_bbox(img0, bbox=det_p, img_id='det_p', bbox_type='jde_det')
                if 'video_name' and 'frame_id' in img_info:
                    self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='det_p')
            self.debugger.add_image_with_bbox(img0, bbox=dets, img_id='det', bbox_type='jde_det')
            if 'video_name' and 'frame_id' in img_info:
                self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='det')
            if hm is not None:
                self.debugger.add_image_with_heatmap(img0, heatmap=hm[0], img_id='det_hm')
                if 'video_name' and 'frame_id' in img_info:
                    self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='det_hm')
            if occ_hm is not None:
                self.debugger.add_image_with_heatmap(img0, heatmap=occ_hm[0], img_id='occ_hm')
                if 'video_name' and 'frame_id' in img_info:
                    self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='occ_hm')

        if len(dets) > 0:
            '''Detections'''
            detections = []
            for i in range(len(dets)):
                feat = id_feature[i] if id_feature is not None else None
                detections.append(STrack(STrack.tlbr_to_tlwh(dets[i, :4]), dets[i, 4], feat, 
                                         img_shape=(height, width), 
                                         buffer_size=30,
                                         use_gmm=self.opt.gmm))
        else:
            detections = []

        # TODO: set the neighbors for each detection
        if self.gsm is not None:
            matching.set_neighbors(gsm=self.gsm, tracks=detections, im=img0)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        if not self.opt.not_reid:
            dists = matching.embedding_distance(strack_pool, detections, embed_type=self.opt.reid_feat_type)
            #dists = matching.iou_distance(strack_pool, detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.reid_thres) #0.4)
        else:
            matches = []
            u_track = list(range(len(strack_pool)))
            u_detection = list(range(len(detections)))

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        if self.opt.track_type == 'public_track' and len(u_detection) > 0:
            # the following implementation is the same with CenterTrack
            pub_u_detections = []
            if len(pub_dets) > 0:
                # Public detection: only create tracks that near to provided public detections
                # just as CenterTrack
                pub_ct = (pub_dets[:, 0:2] + pub_dets[:, 2:4]) / 2
                pri_ct = np.array([detections[i].to_center(clip=False) for i in u_detection])
                dist = ((pri_ct.reshape(-1, 1, 2) - pub_ct.reshape(1, -1, 2)) ** 2).sum(axis=-1)
                for j in range(len(pub_dets)):
                    i = dist[:, j].argmin()
                    if dist[i, j] < detections[i].wh.prod():
                        dist[i, :] = 1e18
                        pub_u_detections.append(u_detection[i])
            u_detection = pub_u_detections

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]


        """step 6: find lost objects"""
        if self.gsm is not None or occ_cts is not None:
            if self.gsm is None:
                refind_stracks = self.find_lost_basedon_occlusion(self.lost_stracks, self.tracked_stracks, occlusion=occ_cts)
            else:
                refind_stracks = self.find_lost_basedon_gsm(self.lost_stracks, self.tracked_stracks, im=img0, threshold=0.9)
            # TODO: is this operation necessary?
            # output_stracks, refind_stracks = remove_duplicate_stracks(output_stracks, refind_stracks, remove_type='second')
            output_stracks.extend(refind_stracks)
            output_stracks = [t for t in output_stracks if t.state != TrackState.Removed]
            self.refresh_occluded_objects(self.tracked_stracks)

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        track_results = [t.output(clip_box=self.opt.clip_box, min_box_area=self.opt.min_box_area) for t in output_stracks]
        track_results = [r for r in track_results if r is not None]
        if self.opt.debug > 0:
            self.debugger.add_image_with_bbox(img0, bbox=track_results, img_id='tracks')
            if 'video_name' and 'frame_id' in img_info:
                self.debugger.add_text('{}:{}'.format(img_info['video_name'], img_info['frame_id']), img_id='tracks')
            self.debugger.show_all_imgs()
        return track_results


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, thr=0.15, remove_type='both'):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < thr)
    if remove_type == 'both':
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    elif remove_type == 'first':
        dupa = list()
        for p, q in zip(*pairs):
            dupa.append(p)
        resa =  [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = stracksb
    elif remove_type == 'second':
        dupb = list()
        for p, q in zip(*pairs):
            dupb.append(q)
        resb =  [t for i, t in enumerate(stracksb) if not i in dupb]
        resa = stracksa
    return resa, resb