import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import time

from models.losses import binary_cross_entropy_loss
from utils.box import encode_boxes, box_center_dist
from models.utils import association_neighbor

class GraphMatch(nn.Module):
    def __init__(self, n_in, neighbor_k=3, absorb_weight=0.5,
                 neighbor_type='learn_app', neighbor_weight_branch='none', match_neighbor=True,
                 use_pos=False, encode_pos=True, embed_pos=True, pos_quantify=-1,
                 pos_dim=64, pos_dim_out=256, wave_length=1000,
                 do_drop=0.2, loss_type='binary_cross_entropy', np_ratio=-1, train_part='all'):
        """Initialize the module

        Args:
            n_in: int, the feat dim of each node
            neighbor_k: int, the top-k neighbors
            absorb_weight: float, the weight to absorb features from neighbor
            loss_type: str, which type of loss to use
            neighbor_type: str, which type of neighbor to use, learn-based or position-based
            match_neighbor: bool, whether to match the neighbors when get the score
            neighbor_weight_branch: how to get the neighbor weight to absorb information from neighbors
            use_pos: bool, whether to use position features in the final classification
            embed_pos: whether to embed the position information by some layers
            encode_pos: whether to encode the position information by the method proposed in Transformer
            pos_dim: the number of dimensions to encode the position information, only effective when encode_pos is True
            pos_dim_out: the dimensions of the embedded pos feature
            wave_length: the wavelength used to encode position information, only effective when encode_pos is True
            do_drop: dropout probability
            np_ratio: the ration between the number of negative and positive samples
            train_part: str, indicating which parts of the model to be trained
        """
        super(GraphMatch, self).__init__()

        self.name = 'GraphMatch'
        self.n_in = n_in
        self.loss_type = loss_type
        self.np_ratio = np_ratio
        self.train_part = train_part

        # positions
        self.use_pos = use_pos
        self.encode_pos = encode_pos
        self.embed_pos = embed_pos
        self.pos_dim = pos_dim
        self.pos_quantify = pos_quantify
        self.wave_length = wave_length

        self.neighbor_k = neighbor_k
        self.absorb_weight = absorb_weight

        self.neighbor_type_list = ['learn_app_pos', 'learn_app', 'learn_pos', 'pos']
        self.neighbor_type = neighbor_type
        self.match_neighbor = match_neighbor
        assert self.neighbor_type in self.neighbor_type_list

        self.neighbor_weight_branch = neighbor_weight_branch
        assert self.neighbor_weight_branch in ['none', 'learn', 'neighbor_dist', 'score_based']
        # none: all neighbors have the same weights
        # learn: learn the weights based on the features of neighbors and anchors
        # neighbor_dist: compute the weights based on the the distance between the neighbors and anchors
        # score_based: compute the weights based on the affinity score between matched neighbors

        self.do_drop = do_drop
        # if not self.use_pos:
        #     raise ValueError('The graph match model should be trained with the postion information!')


        # self.neighbor_k = 4
        # self.neighbor_weight_branch = 'score_based'


        if ('pos' in self.neighbor_type and 'learn' in self.neighbor_type) or self.use_pos:
            if self.embed_pos:
                self.pos_dim_out = pos_dim_out #2 * self.n_in #int(self.n_in / 2) # we set the dims of embedded position feature to a half of appearance feature
            else:
                if self.encode_pos:
                    self.pos_dim_out = self.pos_dim
                else:
                    self.pos_dim_out = 8
        else:
            self.pos_dim_out = 0

        # the layers to embed position information so that the embeddings can be used to gt the neighbors
        if ('pos' in self.neighbor_type and 'learn' in self.neighbor_type) or self.use_pos:
            if self.embed_pos:
                n_in = self.pos_dim if self.encode_pos else 8
                self.pos_embeder = nn.Sequential(
                    nn.Linear(in_features=n_in, out_features=self.pos_dim_out),
                    nn.BatchNorm1d(num_features=self.pos_dim_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=self.pos_dim_out, out_features=self.pos_dim_out),
                )

        # the layers for computing the similarity between the nodes that within one frame
        if 'learn' in self.neighbor_type:
            n_in = 0
            if 'pos' in self.neighbor_type:
                n_in += self.pos_dim_out
            if 'app' in self.neighbor_type: # if learn neighbors from appearance features
                n_in += 2*self.n_in # we concate the appearance features of two nodes
            self.sim_intra_frame = nn.Sequential(
                nn.Linear(in_features=n_in, out_features=self.n_in),
                nn.BatchNorm1d(num_features=self.n_in),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.n_in, out_features=1),
            )

        # if the weight for neighbors are learned by a new branch
        if self.neighbor_weight_branch.lower() == 'learn':
            # support to learn the weight of neighbors to absorb information
            n_in = self.n_in  # appearance feature dimensions
            # if we have relative position embeddings, then use it to compute the neighbor weight
            n_in += self.pos_dim_out * self.use_pos

            n_in = n_in * 2 # we concate the differ feature between anchor node and neighbor node to get the weight of the neighbor
            self.weight_nei = nn.Sequential(
                nn.Linear(in_features=n_in, out_features=n_in),
                nn.BatchNorm1d(num_features=n_in),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=n_in, out_features=1),
            )

        # the layers for computing the similarity between the nodes that in different frames
        n_in = self.n_in + self.pos_dim_out * self.use_pos
        # n_in = self.pos_dim_out * self.use_pos
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_in, out_features=self.n_in),
            nn.BatchNorm1d(num_features=self.n_in),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_in, out_features=1),
        )

        # fix some layers
        if self.train_part == 'all':
            pass
        else:
            if hasattr(self, 'pos_embedder') and 'pos_embedder' not in self.train_part:
                for p in self.pos_embeder.parameters():
                    p.requires_grad = False
            if hasattr(self, 'sim_intra_frame') and 'sim_intra_frame' not in self.train_part:
                for p in self.sim_intra_frame.parameters():
                    p.requires_grad = False
            if hasattr(self, 'weight_nei') and 'weight_nei' not in self.train_part:
                for p in self.weight_nei.parameters():
                    p.requires_grad = False
            if hasattr(self, 'classifier') and 'classifier' not in self.train_part:
                for p in self.classifier.parameters():
                    p.requires_grad = False

    def train(self, mode=True):

        nn.Module.train(self, mode)
        # fix some layers
        if self.train_part == 'all':
            pass
        else:
            if hasattr(self, 'pos_embedder') and 'pos_embedder' not in self.train_part:
                self.pos_embeder.eval()
            if hasattr(self, 'sim_intra_frame') and 'sim_intra_frame' not in self.train_part:
                self.sim_intra_frame.eval()
            if hasattr(self, 'weight_nei') and 'weight_nei' not in self.train_part:
                self.weight_nei.eval()
            if hasattr(self, 'classifier') and 'classifier' not in self.train_part:
                self.classifier.eval()

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
        """Getting the neighbor weight loss
        Args:
            score: 3D tensor, [bs, num_node1, num_node2]
            obj_id1: 2D tensor, [bs, num_node1]
            obj_id2: 2D tensor, [bs, num_node2]
        """
        bs, num_node1 = obj_id1.size(0), obj_id1.size(1)
        num_node2 = obj_id2.size(1)

        # get the loss of final classification
        obj_id1 = obj_id1.view(bs, num_node1, 1)
        obj_id2 = obj_id2.view(bs, 1, num_node2)
        gt_score = (obj_id1 == obj_id2).float() # [bs, num_node1, num_node2]
        gt_score = gt_score * (obj_id1 > 0).float()
        gt_score = gt_score * (obj_id2 > 0).float()

        if self.loss_type == 'binary_cross_entropy':
            loss = binary_cross_entropy_loss(score=score, gt_score=gt_score, ratio=self.np_ratio)
        else:
            raise RuntimeError('Unknown type of loss {}'.format(self.loss_type))

        return loss

    def _get_neighbor_score_loss(self, score_nei,  id_nei1, id_nei2, id1, id2):
        """
        Args:
            score_nei: [bs, num_node1, num_ndoe1, neighbor_k]
            id_nei1: [bs, num_node1, neighbor_k]
            id_nei2: [bs, num_node2, neighbor_k]
            id1: [bs, num_node1]
            id2: [bs, num_node2]

        """

        if id_nei1 is not None and id_nei2 is not None:
            bs, num_node1, neighbor_k = id_nei1.size()
            num_node2 = id_nei2.size(1)

            id_tmp1 = id_nei1.view(bs, num_node1, 1, neighbor_k)
            id_tmp2 = id_nei2.view(bs, 1, num_node2, neighbor_k)
            mask_nei = (id_tmp1 == id_tmp2) * (id_tmp1 > 0) * (id_tmp2 > 0) # [bs, num_node1, num_node2, neighbor_k]

            id_tmp1 = id1.view(bs, num_node1, 1)
            id_tmp2 = id2.view(bs, 1, num_node2)
            mask_anchor = (id_tmp1 == id_tmp2) * (id_tmp1 > 0) * (id_tmp2 > 0) # [bs, num_node1, num_node2]

            mask = mask_anchor.view(bs, num_node1, num_node2, 1) * mask_nei
            mask = mask.float()

            if self.loss_type == 'binary_cross_entropy':
                loss = binary_cross_entropy_loss(score=score_nei, gt_score=mask, ratio=self.np_ratio)
            else:
                raise RuntimeError('Unknown type of loss {}'.format(self.loss_type))

        else:
            loss = score_nei.mean() * 0 #  torch.Tensor([0]).to(score_nei.device)

        return loss

    def get_neighbor_k(self, num_nodes_list):
        """This function get the number of neighbors.
        Args:
            num_nodes_list: list of int, each number is the number of nodes in a graph
        """
        # the neighbor_k should be at least the number of nodes, so that for
        # each node, the neighbors will not be all the same
        k_tmp = [n - 1 for n in num_nodes_list]
        k_tmp.append(self.neighbor_k)
        neighbor_k = min(k_tmp)
        return neighbor_k

    def pick_up_neighbors(self, feat, pos_feat, relative_pos, box, neighbor_k, ids=None):
        """Get the weight between the nodes within one frame

        Args:
            feat: 3D tensor, [bs, num_node, feat_dim]
            pos_feat: 4D tensor, [bs, num_node, num_node, pos_dim_out]
            relative_pos: 4D tensor, [bs, num_node, num_node, pos_dim]
            box: 3D tensor, [bs, num_node, 4], [x1, y1, x2, y2]
            neighbor_k: int, the neighbor k
            ids: [bs, num_node]
        """

        bs, num_node, feat_dim = feat.size()
        # assert feat_dim == self.n_in

        # (1) get neighbor idx
        if 'learn' in self.neighbor_type: # learn the k neighbors from the features
            all_feat = []
            if 'app' in self.neighbor_type:
                feat1 = feat.view(bs, num_node, 1, feat_dim).repeat(1, 1, num_node, 1) # [bs, num_node, num_node, feat_dim]
                feat2 = feat.view(bs, 1, num_node, feat_dim).repeat(1, num_node, 1, 1) # [bs, num_node, num_node, feat_dim]
                all_feat.append(feat1)
                all_feat.append(feat2)
            if 'pos' in self.neighbor_type:
                all_feat.append(pos_feat)
            # concate all features
            all_feat = torch.cat(all_feat, dim=-1) # [bs, num_node, num_node, dim_pos_out+2*n_in]
            #pdb.set_trace()

            weight_logits = self.sim_intra_frame(all_feat.view(-1, all_feat.size(-1))).view(bs, num_node, num_node) # [bs, num_node, num_node]
            weight = F.softmax(weight_logits, dim=2) # [bs, num_node, num_node]
            # weight = F.sigmoid(weight) # [bs, num_node, num_node]

            # set the diagnoal to zero, i.e. self weight is zero
            diag = torch.diagonal(weight, dim1=1, dim2=2) # [bs, num_node]
            diag = torch.diag_embed(diag, dim1=1, dim2=2) # [bs, num_node, num_node]
            weight = weight - diag
            # get the top k neighbors
            nei_v, nei_idx = torch.topk(weight, neighbor_k, dim=-1) # [bs, num_node, neighbor_k]
        elif self.neighbor_type == 'pos': # get the neighbors purely on the positions
            dist = box_center_dist(tlbr=box) # [bs, num_node, num_node]
            # set the diagonal to a large value, so that the neighbor will not include itself
            max_value = dist.max().item() + 1 # float
            diag = torch.Tensor([[max_value]]).repeat(bs, num_node).to(dist.device)
            diag = torch.diag_embed(diag, dim1=1, dim2=2) # [bs, num_node, num_node]
            dist = dist + diag
            nei_v, nei_idx = torch.topk(dist, neighbor_k, dim=-1, largest=False) # [bs, num_node, neighbor_k]

            weight_logits = None
        else:
            raise ValueError('Unknown type of neighbor type. Expected is one of {}, but found {}'.format(self.neighbor_type_list, self.neighbor_type))

        # nei_v: [bs, num_node, neighbor_k]
        # nei_idx: [bs, num_node, neighbor_k]

        # (2) pick up neighbors information: features, position embeddings, relative position, neighbor_boxes
        feat_nei = feat.view(bs, 1, num_node, feat_dim).repeat(1, num_node, 1, 1)  # [bs, num_node, num_node, feat_dim]
        nei_idx_tmp = nei_idx.unsqueeze(dim=-1).repeat(1, 1, 1, feat_nei.size(-1)) # [bs, num_node, neighbor_k, feat_dim]
        feat_nei = torch.gather(feat_nei, index=nei_idx_tmp, dim=2)  # [bs, num_node, neighbor_k, feat_dim]

        if ('learn' in self.neighbor_type and 'pos' in self.neighbor_type) or self.use_pos: # this means that there is pos feature
            nei_idx_tmp = nei_idx.unsqueeze(dim=-1).repeat(1, 1, 1, self.pos_dim_out) # [bs, num_node, neighbor_k, pos_dim_out]
            pos_feat_nei = torch.gather(pos_feat, index=nei_idx_tmp, dim=2) # [bs, num_node, neighbor_k, pos_dim_out]

            nei_idx_tmp = nei_idx.unsqueeze(dim=-1).repeat(1, 1, 1, relative_pos.size(-1))
            relative_pos_nei = torch.gather(relative_pos, index=nei_idx_tmp, dim=2)

        else:
            pos_feat_nei = None
            relative_pos_nei = None

        # box_nei = box.view(bs, num_node, 1, -1).repeat(1, 1, num_node, 1) # [bs, num_node, num_node, 4]
        box_nei = box.view(bs, 1, num_node, -1).repeat(1, num_node, 1, 1)  # [bs, num_node, num_node, 4]
        nei_idx_tmp = nei_idx.unsqueeze(dim=-1).repeat(1, 1, 1, box_nei.size(-1)) # [bs, num_node, neighbor_k, 4]
        box_nei = torch.gather(box_nei, index=nei_idx_tmp, dim=2) # [bs, num_node, neighbor_k, 4]

        if ids is not None:
            ids_nei = ids.view(bs, 1, num_node).repeat(1, num_node, 1) # [bs, num_node, num_node]
            ids_nei = torch.gather(ids_nei, index=nei_idx, dim=2)
        else:
            ids_nei = None

        return feat_nei, pos_feat_nei, relative_pos_nei, box_nei, ids_nei, nei_v, nei_idx, weight_logits

    def get_pos_feat(self, box, im_shape):
        """This function get the features from boxes
        Args:
            box: 3D tensor, [bs, num_node, 4], each box is presented as [x1, y1, x2, y2]
            im_shape: 2D tensor, [bs, 2], [width, height]
        """
        if ('learn' in self.neighbor_type and 'pos' in self.neighbor_type) or self.use_pos:
            bs, num_node = box.size(0), box.size(1)
            relative_pos, pos_feat = encode_boxes(boxes=box, im_shape=im_shape, encode=self.encode_pos,
                                                  dim_position=self.pos_dim, wave_length=self.wave_length,
                                                  normalize=False, quantify=self.pos_quantify)  # [bs, num_node, num_node, dim_pos]

            if self.embed_pos:
                pos_feat = self.pos_embeder(pos_feat.view(-1, pos_feat.size(-1))).view(bs, num_node, num_node, -1)  # [bs, num_node, num_node, dim_pos_out]
                if self.do_drop > 0:
                    pos_feat = F.dropout(pos_feat, p=self.do_drop, training=self.training)
            # pdb.set_trace()
            # get anchor pos feat
            if self.use_pos: # if use the pos features in final classification
                anchor_pos_feat = torch.diagonal(pos_feat, dim1=1, dim2=2)  # [bs, pos_dim_out, num_node]
                anchor_pos_feat = anchor_pos_feat.permute(0, 2, 1)  # [bs, num_node, dim_pos_out]
            else:
                anchor_pos_feat = None
        else:
            relative_pos = None
            pos_feat = None
            anchor_pos_feat = None

        # pdb.set_trace()
        return relative_pos, pos_feat, anchor_pos_feat

    def get_score(self, feat1, pos_feat1, feat_nei1, pos_feat_nei1, weight_nei1,
                  feat2, pos_feat2, feat_nei2, pos_feat_nei2,  weight_nei2):

        """Forward function

        Args:
            feat1: 3D tensor, [bs, num_node1, feat_dim], we treat this frame as track (previous) frame
            pos_feat1:  3D tensor, [bs, num_node1, pos_dim_out]
            feat_nei1: 4D tensor, [bs, num_node, neighbor_k, feat_dim]
            pos_feat_nei1: 4D tensor, [bs, num_node, neighbor_k, pos_dim_out]
            weight_nei1: 3D tensor, [bs, num_node, neighbor_k], the weight to get the neighbors

            feat2: 3D tensor, [bs, num_node1, feat_dim], we treat this frame as detect (current) frame
            pos_feat2: 3D tensor, [bs, num_node2, pos_dim_out]
            pos_feat_nei2: 4D tensor, [bs, num_node2, neighbor_k, pos_dim_out]
            feat_nei2: 4D tensor, [bs, num_node, neighbor_k, feat_dim]
            weight_nei1: 3D tensor, [bs, num_node2, neighbor_k], the weight to get the neighbors
        """
        bs, num_node1, feat_dim = feat1.size()
        num_node2 = feat2.size(1)

        # get the score of anchors
        if self.use_pos: # use the pos features in final classification
            feat1_tmp = torch.cat((feat1, pos_feat1), dim=-1) # [bs, num_node1, feat_dim+pos_dim_out]
            feat2_tmp = torch.cat((feat2, pos_feat2), dim=-1) # [bs, num_node2, feat_dim+pos_dim_out]
        else:
            feat1_tmp = feat1
            feat2_tmp = feat2
        # feat1_tmp = pos_feat1
        # feat2_tmp = pos_feat2
        feat1_tmp = feat1_tmp.view(bs, num_node1, 1, -1)
        feat2_tmp = feat2_tmp.view(bs, 1, num_node2, -1)
        feat = feat1_tmp - feat2_tmp  # [bs, num_node1, num_node2, -1]
        feat = torch.pow(feat, 2) #feat * feat  # [bs, num_node1, num_node2, -1]

        logits_a = self.classifier(feat.view(-1, feat.size(-1))).view(bs, num_node1, num_node2)
        score_anchor = self._logits2score(logits_a) # [bs, num_node1, num_node2]

        # get the score of neighbor
        if feat_nei1 is not None and feat_nei2 is not None: # this means there is at least one neighbor
            neighbor_k = feat_nei1.size(2)

            # 1) prepare neighbor features
            if self.use_pos:  # use the position embeddings in final classification
                feat1_tmp = torch.cat((feat_nei1, pos_feat_nei1), dim=-1)  # [bs, num_node1, neighbor_k, feat_dim+pos_dim_out]
                feat2_tmp = torch.cat((feat_nei2, pos_feat_nei2), dim=-1)  # [bs, num_node2, neighbor_k, feat_dim+pos_dim_out]
            else:  # only appearance features used in the final classification
                feat1_tmp = feat_nei1
                feat2_tmp = feat_nei2

            if self.training:
                self.match_neighbor = False
            else:
                self.match_neighbor = True
            # self.match_neighbor = False

            if not self.match_neighbor and not self.training:
                print('get score without matching neighbors!')
            # self.match_neighbor = False
            if self.match_neighbor:
                feat1_tmp = feat1_tmp.view(bs, num_node1, 1, neighbor_k, 1, -1)
                feat2_tmp = feat2_tmp.view(bs, 1, num_node2, 1, neighbor_k, -1)
                feat_nei_asso = feat1_tmp - feat2_tmp
                feat_nei_asso = torch.pow(feat_nei_asso, 2) # [bs, num_node1, num_node2, neighbor_k, neighbor_k, -1]

                logits_nei_asso = self.classifier(feat_nei_asso.view(-1, feat_nei_asso.size(-1))).view(bs, num_node1, num_node2, neighbor_k, neighbor_k)
                score_nei_asso = self._logits2score(logits_nei_asso) # [bs, num_node1, num_node2, neighbor_k, neighbor_k]

                score_nei = association_neighbor(score=score_nei_asso, tool='scipy') # [bs, num_node1, num_node2, neighbor_k]
                # TODO: leave a margin, for exaple: only use the k-1 matched neighbors among the k neighbors

            else:
                # feat1_tmp = pos_feat_nei1
                # feat2_tmp = pos_feat_nei2
                feat1_tmp = feat1_tmp.view(bs, num_node1, 1, neighbor_k, -1)
                feat2_tmp = feat2_tmp.view(bs, 1, num_node2, neighbor_k, -1)
                feat_nei = feat1_tmp - feat2_tmp
                feat_nei = torch.pow(feat_nei, 2) # feat_nei * feat_nei # [bs, num_node1, num_node2, neighbor_k, -1]

                logits_nei = self.classifier(feat_nei.view(-1, feat_nei.size(-1))).view(bs, num_node1, num_node2, neighbor_k)
                score_nei = self._logits2score(logits_nei) # [bs, num_node1, num_node2, neighbor_k]

            # absorb the score of neighbors
            # 2) get neighbor weight
            if self.neighbor_weight_branch == 'learn':
                # get the weight of neighbor based on the anchor feat and neighbor feat

                # concate the anchor feat with neighbor feat
                feat_tmp = feat.view(bs, num_node1, num_node2, 1, -1).repeat(1, 1, 1, neighbor_k, 1)
                feat_tmp = torch.cat((feat_tmp, feat_nei), dim=-1)  # [bs, num_node1, num_node2, neighbor_k, -1]
                weight_nei = self.weight_nei(feat_tmp.view(-1, feat_tmp.size(-1))).view(bs, num_node1, num_node2,
                                                                                        neighbor_k)
                # weight_nei: [bs, num_node1, num_node2, neighbor_k]
                weight_nei = weight_nei.softmax(dim=-1)
                # 3) aggregate information from neighbor
                score_n_tmp = (weight_nei * score_nei).sum(-1)  # [bs, num_node1, num_node2]

                # set the absorb weight, so the final score will be the mean of anchor and neighbor score
                self.absorb_weight = 1 - 1.0 / (neighbor_k + 1)
                score_graph = (1 - self.absorb_weight) * score_anchor + self.absorb_weight * score_n_tmp

            elif self.neighbor_weight_branch == 'neighbor_dist':
                # get the weight of neighbors based on the distance between neighbors and anchors
                raise NotImplementedError
            elif self.neighbor_weight_branch == 'score_based':
                weight_nei = score_nei.clone()
                weight_nei = weight_nei.softmax(dim=-1)

                # 3) aggregate information from neighbor
                score_n_tmp = (weight_nei * score_nei).sum(-1)  # [bs, num_node1, num_node2]

                # set the absorb weight, so the final score will be the mean of anchor and neighbor score
                self.absorb_weight = 1 - 1.0 / (neighbor_k + 1)
                score_graph = (1 - self.absorb_weight) * score_anchor + self.absorb_weight * score_n_tmp
            else:
                # all weight are set to 1. We perform global average
                weight_nei = torch.ones((bs, num_node1, num_node2, neighbor_k)).to(score_nei.device)
                weight_nei = weight_nei.softmax(dim=-1)

                # 3) aggregate information from neighbor
                score_n_tmp = (weight_nei * score_nei).sum(-1) # [bs, num_node1, num_node2]

                # set the absorb weight, so the final score will be the mean of anchor and neighbor score
                self.absorb_weight = 1 - 1.0 / (neighbor_k + 1)
                score_graph = (1 - self.absorb_weight) * score_anchor + self.absorb_weight * score_n_tmp
        else:
            score_graph = score_anchor
            score_nei = None
        return score_graph, score_anchor, score_nei#, score_nei_asso

    def forward(self, feat1, feat2, box1=None, box2=None, obj_id1=None, obj_id2=None, im_shape=None):
        """Forward function

        Args:
            feat1: 3D tensor, [bs, num_node1, feat_dim], we treat this frame as track (previous) frame
            feat2: 3D tensor, [bs, num_node1, feat_dim], we treat this frame as detect (current) frame
            box1: 3D tensor, [bs, num_node1, 4], each box is presented as [x1, y1, x2, y2]
            box1: 3D tensor, [bs, num_node2, 4], each box is presented as [x1, y1, x2, y2]
            obj_id1: 2D tensor, [bs, num_node1]
            obj_id2: 2D tensor, [bs, num_node1]
            im_shape: 2D tensor, [bs, 2], [width, height]
        """
        loss = {}
        score = {}

        if self.do_drop > 0:
            feat1 = F.dropout(feat1, p=self.do_drop, training=self.training)
            feat2 = F.dropout(feat2, p=self.do_drop, training=self.training)

        bs, num_node1, feat_dim = feat1.size()
        num_node2 = feat2.size(1)

        # pos_feat: [bs, num_node, num_node, pos_dim_out]
        # anchor_pos_feat: [bs, num_node1, pos_dim_out]
        relative_pos1, pos_feat1, anchor_pos_feat1 = self.get_pos_feat(box=box1, im_shape=im_shape)
        relative_pos2, pos_feat2, anchor_pos_feat2 = self.get_pos_feat(box=box2, im_shape=im_shape)

        # pick up the neighbor features and ids
        neighbor_k = self.get_neighbor_k(num_nodes_list=[num_node1, num_node2])
        if neighbor_k >= 1:
            # feat_nei: [bs, num_node, neighbor_k, feat_dim]
            # pos_feat_nei: [bs, num_node, neighbor_k, pos_dim_out]
            # obj_id_nei: [bs, num_node, neighbor]
            feat_nei1, pos_feat_nei1, _, _, ids_nei1, weight_nei1, _, weight1 = \
                self.pick_up_neighbors(feat=feat1, pos_feat=pos_feat1, relative_pos=relative_pos1,
                                       box=box1, neighbor_k=neighbor_k, ids=obj_id1)
            feat_nei2, pos_feat_nei2, _, _, ids_nei2, weight_nei2, _, weight2 = \
                self.pick_up_neighbors(feat=feat2, pos_feat=pos_feat2, relative_pos=relative_pos2,
                                       box=box2, neighbor_k=neighbor_k, ids=obj_id2)
        else:
            feat_nei1, pos_feat_nei1, ids_nei1, weight_nei1 = None, None, None, None
            feat_nei2, pos_feat_nei2, ids_nei2, weight_nei2 = None, None, None, None

        score_graph, score_anchor, score_nei = self.get_score(feat1=feat1, pos_feat1=anchor_pos_feat1,
                                                              feat_nei1=feat_nei1,
                                                              pos_feat_nei1=pos_feat_nei1,
                                                              weight_nei1=weight_nei1,
                                                              feat2=feat2, pos_feat2=anchor_pos_feat2,
                                                              feat_nei2=feat_nei2,
                                                              pos_feat_nei2=pos_feat_nei2,
                                                              weight_nei2=weight_nei2)
        score['score_g'] = score_graph # [bs, num_node1, num_node2]
        score['score_a'] = score_anchor # [bs, num_node1, num_node2]

        if obj_id1 is not None and obj_id2 is not None:
            loss_anchor = self._get_loss(score=score_anchor, obj_id1=obj_id1, obj_id2=obj_id2)
            loss['loss_a'] = loss_anchor

            if self.neighbor_weight_branch.lower() == 'learn':
                loss_graph = self._get_loss(score=score_graph, obj_id1=obj_id1, obj_id2=obj_id2)
                loss['loss_g'] = loss_graph

            if self.use_pos and score_nei is not None:
                loss_nei = self._get_neighbor_score_loss(score_nei=score_nei, id_nei1=ids_nei1, id_nei2=ids_nei2,
                                                         id1=obj_id1, id2=obj_id2)
                loss['loss_n'] = loss_nei

        return score, loss

