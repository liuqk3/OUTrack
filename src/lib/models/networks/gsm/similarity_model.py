import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import numpy as np

from models.networks.gsm.naive_match import NaiveMatch
from models.networks.gsm.graph_match import GraphMatch
from models.networks.gsm.resnet import ResNetBackbone

class GraphSimilarity(nn.Module):
    def __init__(self, *, backbone_name='none', backbone_args=None,
                graphmatch_args, naivematch_args, 
                 train_part='all', pad_boxes=False,
                 match_name='GraphMatch'):
        """This class get the similarity score between detections and tracks

        Args:
            backbone_name: str, the name of backbone name
            backbone_args: dict, the args to initialize the backbone
            graphmatch_args: dict, the args to the GraphMatch module
            naivematch_args: dict, the args to the NaiveMatch module
            train_part: str, which part to train
            pad_boxes: bool, whether the inputed data is padded a track box and a det box
            match_name: str, 'GraphMatch', 'NaiveMatch', or 'GraphMatch, NaiveMatch'
        """
        super(GraphSimilarity, self).__init__()
        # pdb.set_trace()
        self.name = 'GraphSimilarity'

        self.backbone_name = backbone_name
        self.train_part = 'graph_match, naive_match' if train_part == 'all' else train_part
        self.pad_boxes = pad_boxes

        self.match_name = match_name

        if self.backbone_name == 'none':
            self.backbone = None
        elif self.backbone_name == 'ResNetBackbone':
            self.backbone = ResNetBackbone(**backbone_args)

            self.im_scale = torch.Tensor(self.backbone.im_info['scale']).view(1, -1, 1, 1)  # [1, 1, 3]
            self.im_mean = torch.Tensor(self.backbone.im_info['mean']).view(1, -1, 1, 1)  # [1, 1, 3]
            self.im_var = torch.Tensor(self.backbone.im_info['var']).view(1, -1, 1, 1)  # [1, 1, 3]

        else:
            raise NotImplementedError
        # import pdb; pdb.set_trace()
        match_names = self.match_name.split(',')
        if 'GraphMatch_v5' in match_names:
            self.graph_match = GraphMatch(**graphmatch_args)

        if 'NaiveMatch' in match_names:
            self.naive_match = NaiveMatch(**naivematch_args)

        self.train()

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            if hasattr(self, 'graph_match'):  # 'GraphMatch' in self.match_name:
                if 'graph_match' in self.train_part:
                    print('set GraphMatch trainable...')
                    self.graph_match.train()
                else:
                    for key, value in dict(self.graph_match.named_parameters()).items():
                        value.requires_grad = False
                    self.graph_match.eval()

            if hasattr(self, 'naive_match'):  # 'NaiveMatch' in self.match_name:
                if 'naive_match' in self.train_part:
                    self.naive_match.train()
                    print('set NaiveMatch trainable...')
                else:
                    for key, value in dict(self.naive_match.named_parameters()).items():
                        value.requires_grad = False
                    self.naive_match.eval()

    def _hungarian_assign(self, cost_matrix, ids1=None, ids2=None, strict=False, tool='scipy'):
        """This function perform data association based on hungarian algorithm

        Args:
            cost_matrix: 2D tensor, [n, m]
            ids1: 1D tensor, [n], the gt ids for the first frame
            ids2: 1D tensor, [m], the gt ids for the second frame. If both ids1 and ids2 are None,
                this function may be called while tracking online
            strict: bool, if true, we only try to assign true detections with true tracks.
                Otherwise, we try to assing all (including padded) detections to all
                (included padded) tracks.
        """

        if isinstance(cost_matrix, torch.Tensor):
            cost_matrix = cost_matrix.numpy()
        if isinstance(ids1, torch.Tensor):
            ids1 = ids1.numpy()
        if isinstance(ids2, torch.Tensor):
            ids2 = ids2.numpy()

        if not strict and not self.pad_boxes:
            raise ValueError(
                'There are no padded boxes in the input data, only support strict assign!')

        if ids1 is not None and ids2 is not None:
            if strict:
                # filter out empty and padded nodes
                if ids1 is not None:
                    index1 = ids1 > 0
                    ids1 = ids1[index1]
                    cost_matrix = cost_matrix[index1, :]
                if ids2 is not None:
                    index2 = ids2 > 0
                    ids2 = ids2[index2]
                    cost_matrix = cost_matrix[:, index2]
            else:
                # filter out empty nodes
                # filter out empty and padded nodes
                if ids1 is not None:
                    index1 = ids1 >= 0
                    ids1 = ids1[index1]
                    cost_matrix = cost_matrix[index1, :]
                if ids2 is not None:
                    index2 = ids2 >= 0
                    ids2 = ids2[index2]
                    cost_matrix = cost_matrix[:, index2]

                # the last row is the padded track, and the last column is padded detection.
                # in order to make sure that each node will be assigned anthoer node, we pad
                # the cost matrix into a square matrix
                num_track = cost_matrix.shape[0]
                num_det = cost_matrix.shape[1]
                if num_track > num_det:
                    # repeat the last column
                    cost_col = cost_matrix[:, -1]
                    cost_col = cost_col[:, np.newaxis]  # [num_track, 1]
                    cost_col = np.tile(cost_col, (1, num_track-num_det))
                    cost_matrix = np.concatenate(
                        (cost_matrix, cost_col), axis=1)
                    if ids2 is not None:
                        ids2 = np.concatenate(
                            (ids2, np.tile(ids2[-1], (num_track-num_det))), axis=0)
                elif num_det > num_track:
                    # repeat the last tow
                    cost_row = cost_matrix[-1, :]
                    cost_row = cost_row[np.newaxis, :]
                    cost_row = np.tile(cost_row, (num_det-num_track, 1))
                    cost_matrix = np.concatenate(
                        (cost_matrix, cost_row), axis=0)
                    if ids1 is not None:
                        ids1 = np.concatenate(
                            (ids1, np.tile(ids1[-1], (num_det-num_track))), axis=0)

        if tool == 'scipy':
            from scipy.optimize import linear_sum_assignment
            indices = linear_sum_assignment(cost_matrix)
            if isinstance(indices, tuple):
                # [num_assign, 2], the first row is the index of tracks, and the second is the index of detections
                indices = np.array(indices).transpose()
        elif tool == 'lapjv':
            import lapjv
            # pad the cost matrix to a square matrix for tool lapjv
            num_track, num_det = cost_matrix.shape
            if num_track > num_det:
                pad = np.zeros((num_track, num_track - num_det))
                pad.fill(1e5)
                cost_matrix = np.concatenate((cost_matrix, pad), axis=1)

            elif num_det > num_track:
                pad = np.zeros((num_det - num_track, num_det))
                pad.fill(1e5)
                cost_matrix = np.concatenate((cost_matrix, pad), axis=0)
            ass = lapjv.lapjv(cost_matrix)
            r_ass, c_ass = ass[0], ass[1]
            if num_track <= num_det:
                r_ass = r_ass[0:num_track]
                indices = np.arange(0, num_track)
                # [num_assign, 2]
                indices = np.array([indices, r_ass]).transpose()
            else:
                c_ass = c_ass[0:num_det]
                indices = np.arange(0, num_det)
                # [num_assign, 2]
                indices = np.array([c_ass, indices]).transpose()

        if ids1 is not None and ids2 is not None:
            ids1_tmp = ids1[indices[:, 0]]
            ids2_tmp = ids2[indices[:, 1]]

            right_num = ids1_tmp == ids2_tmp
            right_num = right_num.sum()
            assign_num = indices.shape[0]

            if not strict:  # we need to check the disappeared track and appeared detection
                ids1_true = ids1[ids1 > 0]
                ids2_true = ids2[ids2 > 0]

                # [num_true_track, num_true_det]
                mask = ids1_true[:, np.newaxis] == ids2_true[np.newaxis, :]

                # check if there are some tracks disappear
                disappear_idx = np.sum(mask, axis=1) == 0  # [num_true_track]
                ids1_disappear = ids1_true[disappear_idx]  # [num_disappear]
                # [num_assign_disappear]
                ids1_assign_disappear = ids1_tmp[ids2_tmp == 0]

                disappear_mask = ids1_assign_disappear[:,
                                                       np.newaxis] == ids1_disappear[np.newaxis, :]
                # [num_assign_disappear]
                disappear_idx = np.sum(disappear_mask, axis=1) > 0
                num_true_disappear = disappear_idx.sum()

                # check if there are som detect appear
                appear_idx = np.sum(mask, axis=0) == 0  # [num_ture_det]
                ids2_appear = ids2_true[appear_idx]
                ids2_assign_appear = ids2_tmp[ids1_tmp == 0]
                appear_mask = ids2_assign_appear[:,np.newaxis] == ids2_appear[np.newaxis, :]
                # [num_assign_disappear]
                appear_idx = np.sum(appear_mask, axis=1) > 0
                num_true_appear = appear_idx.sum()

                right_num = right_num + num_true_disappear + num_true_appear
            assert right_num <= assign_num
        else:
            assign_num, right_num = 0, 0

        return indices, assign_num, right_num

    def _get_acc(self, score, obj_id1, obj_id2):
        """get the accuracy using Hungarian algorithm

        Args:
            score: 3D tensor, [bs, num_node1, num_node2]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
        """
        if obj_id1 is None and obj_id2 is None:
            acc = torch.Tensor([0]).to(score.device)
        else:
            bs = score.size(0)
            cost = 1 - score

            assign_num = []
            right_num = []
            cost_cpu = cost.detach().cpu()
            track_id_cpu = obj_id1.cpu()
            det_id_cpu = obj_id2.cpu()
            for b in range(bs):
                _, assign_num_tmp, right_num_tmp = self._hungarian_assign(cost_matrix=cost_cpu[b], ids1=track_id_cpu[b],
                                                                          ids2=det_id_cpu[b], strict=True)
                assign_num.append(assign_num_tmp)
                right_num.append(right_num_tmp)

            assign_num = sum(assign_num)
            right_num = sum(right_num)

            acc = right_num / assign_num
            # pdb.set_trace()
            acc = torch.Tensor([acc]).to(score.device)

        return acc


    def get_reid_feature(self, image, tlbrs):
        """
            im_patch1: 4D tensor, [bs, 3, height, width]
        """
        import cv2 
        height, width = image.shape[0:2]
        crop_w, crop_h = self.backbone.im_info['size']
        # feats = []
        # for box in tlbrs:
        #     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #     x1 = max(0, x1)
        #     y1 = max(0, y1)
        #     x2 = min(x2, width)
        #     y2 = min(y2, height)
        #     if x2 > x1 and y2 > y1:
        #         im = image[y1:y2, x1:x2, :].copy()
        #         im = cv2.resize(im, (crop_w, crop_h)) # h, w, 3
        #         im = torch.Tensor(im).float()
        #         im = (im / self.im_scale - self.im_mean) / self.im_var # h x w x 3
        #         im = im.to(self.device).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
        #         feat = self.backbone(im)
        #         im_patchs.append(im)
        #     else:
        #         feat = torch.zeros(1, self.backbone.output_dim).to(self.device)
        #     feats.append(feat)
        # feats = torch.cat(feats, dim=0)
        # return feats

        im_patchs = []
        for box in tlbrs:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width)
            y2 = min(y2, height)
            if x2 > x1 and y2 > y1:
                im = image[y1:y2, x1:x2, :].copy()
                im = cv2.resize(im, (crop_w, crop_h)) # h, w, 3
                im = torch.Tensor(im).float()
            else:
                im = torch.zeros(crop_h, crop_w, 3)
            im_patchs.append(im)
        im_patchs = torch.stack(im_patchs, dim=0).permute(0, 3, 1, 2).to(self.device)# B x 3 x H x W 
        im_patchs = (im_patchs / self.im_scale.to(self.device) - self.im_mean.to(self.device)) / self.im_var.to(self.device)
        with torch.no_grad():
            feats = self.backbone(im_patchs) # bs x dim
        return feats


    def forward(self, feat1, feat2, box1=None, box2=None, obj_id1=None, obj_id2=None, im_shape=None):
        """get the similarity between the tracks and detections

        Args:

            feat1: 4D tensor, [bs, num_node1, feat_dim]
            feat2: 4D tensor, [bs, num_node2, feat_dim]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
            box1: 3D tensor, [bs, num_node1, 4], each box is presented as [x1, y1, x2, y2]
            box1: 3D tensor, [bs, num_node2, 4], each box is presented as [x1, y1, x2, y2]
            im_shape: 2D tensor, [bs, 2], [width, height]
        """
        if self.training:
            raise NotImplementedError

        loss = {}
        acc = {}

        # if we have the graph match
        if 'GraphMatch' in self.match_name:
            if self.graph_match.name in ['GraphMatch']:
                score_g, loss_g = self.graph_match(feat1=feat1, feat2=feat2, box1=box1,
                                                   box2=box2, obj_id1=obj_id1, obj_id2=obj_id2, im_shape=im_shape)
                for k in loss_g.keys():
                    loss[k + '_graph'] = loss_g[k]

                acc_g = self._get_acc(
                    score=score_g['score_g'], obj_id1=obj_id1, obj_id2=obj_id2)
                acc_ga = self._get_acc(
                    score=score_g['score_a'], obj_id1=obj_id1, obj_id2=obj_id2)
                acc['acc_g_graph'] = acc_g
                acc['acc_a_graph'] = acc_ga

        if 'NaiveMatch' in self.match_name:
            score_n, loss_n = self.naive_match(feat1=feat1, feat2=feat2, box1=box1, box2=box2,
                                               obj_id1=obj_id1, obj_id2=obj_id2, im_shape=im_shape)
            acc_n = self._get_acc(
                score=score_n['score'], obj_id1=obj_id1, obj_id2=obj_id2)
            for k in loss_n.keys():
                loss[k+'_naive'] = loss_n[k]
            acc['acc_naive'] = acc_n

        return loss, acc
