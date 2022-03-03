import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import binary_cross_entropy_loss
from utils.box import encode_boxes

class NaiveMatch(nn.Module):
    def __init__(self, n_in, loss_type='binary_cross_entropy', do_drop=0.2,
                 use_pos=False, encode_pos=False, embed_pos=True,
                 pos_dim=64, wave_length=1000, np_ratio=-1, pos_quantify=-1):
        """Initialize
        Args:
            n_in: int, input feat_dim for each node
            loss_type: str, which type of loss to use
            do_drop: dropout probability
            use_pos: bool, whether to use the positions in final classification
            pos_dim: the number of dimensions to embed the position information
            wave_length: the wavelength used to embed position information
            np_ratio: the ration between negative and positive samples to get the loss
            pos_quantify: int, used to quantify the bounding boxes
        """
        super(NaiveMatch, self).__init__()
        self.loss_type = loss_type
        self.n_in = n_in
        self.do_drop = do_drop

        self.use_pos = use_pos
        self.encode_pos = encode_pos
        self.embed_pos = embed_pos
        self.pos_dim = pos_dim
        self.pos_quantify = pos_quantify

        self.wave_length = wave_length
        self.np_ratio = np_ratio

        if self.use_pos:
            raise NotImplementedError

        if self.use_pos:
            if self.embed_pos:
                self.pos_dim_out = int(self.n_in / 2) # we set the dims of embedded position feature to a half of appearance feature
            else:
                if self.encode_pos:
                    self.pos_dim_out = self.pos_dim
                else:
                    self.pos_dim_out = 4
        else:
            self.pos_dim_out = 0

        if self.use_pos and self.embed_pos:
            n_in = self.pos_dim if self.encode_pos else 4
            self.pos_embeder = nn.Sequential(
                nn.Linear(in_features=n_in, out_features=self.pos_dim_out),
                nn.BatchNorm1d(num_features=self.pos_dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.pos_dim_out, out_features=self.pos_dim_out),
                nn.BatchNorm1d(num_features=self.pos_dim_out),
                nn.ReLU(inplace=True),
            )

        # we use a classifier to get the similarity score
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.n_in+self.use_pos*self.pos_dim_out, out_features=self.n_in),
            nn.BatchNorm1d(num_features=self.n_in),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_in, out_features=1)
        )

    def _logits2score(self, logits):
        """
        Args:
            logits: 3D tensor, [bs, num_track, num_det], the values in this tensor are not activated

        """
        if self.loss_type == 'binary_cross_entropy':
            score = torch.sigmoid(logits)
        else:
            raise RuntimeError('Unknown type of loss {}'.format(self.loss_type))
        return score

    def _get_loss(self, score, obj_id1, obj_id2):
        """get final loss

        Args:
            score: 3D tensor, [bs, num_node1, num_node2]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
        """

        bs, num_node1 = obj_id1.size()
        num_node2 = obj_id2.size(1)

        # get the loss of final classification
        obj_id1 = obj_id1.view(bs, num_node1, 1)
        obj_id2 = obj_id2.view(bs, 1, num_node2)
        gt_score = (obj_id1 == obj_id2).float()  # [bs, num_node1, num_node2]
        gt_score = gt_score * (obj_id1 > 0).float()
        gt_score = gt_score * (obj_id2 > 0).float()

        if self.loss_type == 'binary_cross_entropy':
            loss = binary_cross_entropy_loss(score=score, gt_score=gt_score, ratio=self.np_ratio)
        else:
            raise RuntimeError('Unknown type of loss {}'.format(self.loss_type))

        return loss

    def get_pos_feat(self, box, im_shape):
        """This function get the features from boxes
        Args:
            box: 3D tensor, [bs, num_node, 4], each box is presented as [x1, y1, x2, y2]
            im_shape: 2D tensor, [bs, 2]
        """
        if self.use_pos:
            bs, num_node = box.size(0), box.size(1)
            relative_pos, pos_feat = encode_boxes(boxes=box, im_shape=im_shape, dim_position=self.pos_dim,
                                                  wave_length=self.wave_length, normalize=False,
                                                  quantify=self.pos_quantify, encode=self.encode_pos)  # [bs, num_node, num_node, dim_pos]
            if self.embed_pos:
                pos_feat = self.pos_embeder(pos_feat.view(-1, pos_feat.size(-1))).view(bs, num_node, num_node, -1)  # [bs, num_node, num_node, dim_pos_out]
                if self.do_drop > 0:
                    pos_feat = F.dropout(pos_feat, p=self.do_drop, training=self.training)

            # get anchor pos feat
            anchor_pos_feat = torch.diagonal(pos_feat, dim1=1, dim2=2)  # [bs, pos_dim_out, num_node]
            anchor_pos_feat = anchor_pos_feat.permute(0, 2, 1)  # [bs, num_node, dim_pos_out]
        else:
            relative_pos = None
            pos_feat = None
            anchor_pos_feat = None
        return relative_pos, pos_feat, anchor_pos_feat

    def get_score(self, feat1, feat2, pos_feat1=None, pos_feat2=None):
        """get the similarity between the tracks and detections

        Args:

            feat1: 3D tensor, [bs, num_node1, feat_dim]
            feat2: 3D tensor, [bs, num_node2, feat_dim]
            pos_feat1: 3D tensor, [bs, num_node1, pos_dim_out]
            pos_feat2: 3D tensor, [bs, num_node2, pos_dim_out]
        """
        # classification
        bs, num_node1, _ = feat1.size()
        num_node2 = feat2.size(1)

        # append the pos embeding
        if self.use_pos:
            feat1 = torch.cat((feat1, pos_feat1), dim=-1) # [bs, num_node1, feat_dim + pos_dim_out]
            feat2 = torch.cat((feat2, pos_feat2), dim=-1) # [bs, num_node2, feat_dim + pos_dim_out]

        feat1 = feat1.view(bs, num_node1, 1, -1).repeat(1, 1, num_node2, 1)  # [bs, num_node1, num_node2, -1]
        feat2 = feat2.view(bs, 1, num_node2, -1).repeat(1, num_node1, 1, 1)  # [bs, num_node1, num_node2, -1]
        # feat = torch.cat((track_feat, det_feat), dim=3) # [bs, num_track, num_det, 2*n_out]
        feat = feat1 - feat2
        feat = feat * feat # [bs, num_track, num_det, n_out]

        logits = self.classifier(feat.view(-1, feat.size(-1))).view(bs, num_node1, num_node2)  # [bs, num_track, num_det]
        score = self._logits2score(logits)

        return score

    def forward(self, feat1, feat2, box1=None, box2=None, obj_id1=None, obj_id2=None, im_shape=None):
        """get the similarity between the tracks and detections

        Args:

            feat1: 3D tensor, [bs, num_node1, feat_dim]
            feat2: 3D tensor, [bs, num_node2, feat_dim]
            box1: 3D tensor, [bs, num_node1, 4], each box is presented as [x1, y1, x2, y2]
            box1: 3D tensor, [bs, num_node2, 4], each box is presented as [x1, y1, x2, y2]
            obj_id1: 2D tensor, [bs, num_node1], the id of track node
            obj_id2: 2D tensor, [bs, num_node2], the if of det node
            im_shape: 2D tensor, [bs, 2]
        """
        # classification
        if self.do_drop > 0:
            feat1 = F.dropout(feat1, p=self.do_drop, training=self.training)
            feat2 = F.dropout(feat2, p=self.do_drop, training=self.training)

        # get pos feature
        _, _, anchor_pos_feat1 = self.get_pos_feat(box=box1, im_shape=im_shape)  # [bs, num_node1, pos_dim_out]
        _, _, anchor_pos_feat2 = self.get_pos_feat(box=box2, im_shape=im_shape)  # [bs, num_node2, pos_dim_out]

        score = self.get_score(feat1=feat1, feat2=feat2, pos_feat1=anchor_pos_feat1, pos_feat2=anchor_pos_feat2)

        if obj_id1 is not None and obj_id2 is not None:
            loss = self._get_loss(score=score, obj_id1=obj_id1, obj_id2=obj_id2)
        else:
            loss = torch.Tensor([0]).to(score.device)

        score = {
            'score': score,
        }

        loss = {
            'loss': loss,
        }
        return score, loss
