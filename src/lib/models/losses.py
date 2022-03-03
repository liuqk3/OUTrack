# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F
import math
import os
import copy

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res



def binary_cross_entropy_loss(score, gt_score, ratio=-1):
    """Get the binary cross entropy loss
    Args:
        score: multi-dim tensor, predicted score
        gt_score: multi-dim tensor, ground-truth score, 1 is positive, 0 is negative
        ratio: the ratio number of negative and positive samples, if -1, we use all negative samples
    """
    # positive samples
    margin = 1e-10
    mask = gt_score == 1
    score_pos = score[mask]  # 1D tensor
    score_pos = score_pos[score_pos > margin]
    #score_pos[score_pos < margin] = score_pos[score_pos < margin] * 0 + margin
    loss_pos = 0 - torch.log(score_pos)
    num_pos = loss_pos.size(0)

    # negative samples
    mask = gt_score == 0
    score_neg = score[mask]  # 1D tensor
    score_neg = 1 - score_neg
    score_neg = score_neg[score_neg > margin]
    #score_neg[score_neg < margin] = score_neg[score_neg < margin] * 0 + margin
    loss_neg = 0 - torch.log(score_neg)
    num_neg = loss_neg.size(0)
    if ratio > 0:
        loss_neg = loss_neg.sort(descending=True)[0]
        num_neg = min(loss_neg.size(0), int(ratio*num_pos))
        loss_neg = loss_neg[0:num_neg]

    if num_neg > 0 and num_pos > 0:
        weight_pos = num_neg / (num_pos + num_neg)
        weight_neg = num_pos / (num_pos + num_neg)
        loss = weight_pos * loss_pos.mean() + weight_neg * loss_neg.mean()
    elif num_neg == 0 and num_pos > 0:
        loss = loss_pos.mean()
    elif num_neg > 0 and num_pos == 0:
        loss = loss_neg.mean()
    else:
        loss = score.mean() * 0  # torch.Tensor([0]).to(score.device)

    return loss



class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


class CycasLoss(nn.Module):
    """
    Computes the loss for ReID learning. 
    Given two adjacent frame, we can compute the loss based 
    on cycle assistance.
    Ref. paper: CycAs: Self-supervised Cycle Association for Learning Re-identifable Descriptions
    """

    def __init__(self, margin=0.5, supervise=1):

        super(CycasLoss, self).__init__()
        self.supervise = supervise
        self.margin = margin

    def _softmax(self, x, dim=0, epsilon=0.5):
        """
        Args:
            x: 2D tensor, performae softmax along the given dim
        """
        K = x.shape[dim]
        t = math.log(K + 1) / epsilon
        x = torch.exp(t*x)
        s = torch.sum(x, dim=dim, keepdim=True)
        x = x / s
        return x 

    def activate_func(self, x, dim=0):
        return self._softmax(x, dim)

    def loss_func(self, reid_feat_0, identity_0, reid_feat_1, identity_1):
        """
        Get the cycle loss

        Args:
            reid_feat_0: 2D tensor, [N, D]
            identity_0: 1D tensor, [N]
            reid_feat_1: 2D tensor, [M, D]
            identity_1: 1D tensor, [N]
        """
        N = reid_feat_0.shape[0]
        M = reid_feat_1.shape[0]

        if N > M: # swap
            reid_feat_0, reid_feat_1 = reid_feat_1, reid_feat_0
            identity_0, identity_1 = identity_1, identity_0
            N = reid_feat_0.shape[0]
            M = reid_feat_1.shape[0]

        similarity = torch.einsum('nd,md->nm', reid_feat_0, reid_feat_1) # N, M, in [-1, 1]

        # get gt
        if self.supervise > 0:
            if self.supervise == 1:
                identity_0 = identity_0.unsqueeze(dim=1) # N, 1
                identity_1 = identity_1.unsqueeze(dim=0) # 1, M

                a_forward = self.activate_func(similarity, dim=1) # N, M
                a_backward = self.activate_func(similarity.permute(1, 0), dim=1) # M, N
                assign_cycle = torch.einsum('im,mj->ij', a_forward, a_backward) # N, N

                similarity_gt = (identity_0 == identity_1).float() * (identity_0 > 0).float() * (identity_1 > 0).float() # N, M
                assign_cycle_gt = torch.einsum('im,mj->ij', similarity_gt, similarity_gt.permute(1, 0)) # N, N

                loss = (assign_cycle - assign_cycle_gt).abs().sum()
                num_sample = assign_cycle.shape[0]
            elif self.supervise == 2:
                identity_0 = identity_0.unsqueeze(dim=1) # N, 1
                identity_1 = identity_1.unsqueeze(dim=0) # 1, M

                a_forward = self.activate_func(similarity, dim=1) # N, M
                a_backward = self.activate_func(similarity.permute(1, 0), dim=1) # M, N
                assign_cycle = torch.einsum('im,mj->ij', a_forward, a_backward) # N, N

                similarity_gt = (identity_0 == identity_1).float() * (identity_0 > 0).float() * (identity_1 > 0).float() # N, M
                assign_cycle_gt = torch.einsum('im,mj->ij', similarity_gt, similarity_gt.permute(1, 0)) # N, N

                loss = - (assign_cycle_gt * torch.log(assign_cycle + 1e-10) + (1 - assign_cycle_gt) * torch.log((1 - assign_cycle) + 1e-10)).sum()
                num_sample = assign_cycle.shape[0]

            else:
                raise NotImplementedError
        else:
            if self.supervise == 0:
                # The cycle Association in ECCV 2020 paper
                # soft max with adaptive temperature
                a = self.activate_func(similarity, dim=1) # N, M
                a_trans = self.activate_func(similarity.permute(1, 0), dim=1) # M, N
                assign_cycle = torch.einsum('im,mj->ij', a, a_trans) # N, N
                
                diag = torch.diagonal(assign_cycle) # get the diagnoal matrix 

                # get the max value in row and collom
                assign_cycle = assign_cycle - torch.diag(diag)
                max_row, _ = torch.max(assign_cycle, dim=1)
                max_col, _ = torch.max(assign_cycle, dim=0)

                loss = (F.relu(max_row - diag + self.margin) + F.relu(max_col - diag + self.margin)).sum() 
                num_sample = diag.shape[0]
            else:
                raise NotImplementedError

        return loss, num_sample


    def forward(self, feat_cur, feat_pre, mask_cur, mask_pre,  target_cur, target_pre, reid_mask_cur, reid_mask_pre, negative=None):
        """
        Computes the reid loss.
        Arguments:
            feat_cur: B, N, C
            mask_cur: B, N
            target_cur: B, N
            feat_pre: B, N, C
            mask_pre: B, N
            target_pre: B, N   
        Returns:
            reid_loss (Tensor)
        """
        # prepare identities for proposals. If trained with GT, no need to prepare identities
        loss = feat_cur.mean() * 0
        num_sample = 0

        feat_cur = F.normalize(feat_cur, dim=2)    
        feat_pre = F.normalize(feat_pre, dim=2)

        bs = feat_cur.shape[0]
        for i in range(bs):
            mask1 = mask_cur[i] # N
            mask2 = mask_pre[i] # N
            feat1 = feat_cur[i][mask1 > 0] # N', C
            feat2 = feat_pre[i][mask2 > 0] # M', C
            target1 = target_cur[i][mask1 > 0] if target_cur is not None else None
            target2 = target_pre[i][mask2 > 0] if target_pre is not None else None
            if feat1.shape[0] > 0 and feat2.shape[0] > 0:
                loss_tmp, num_sample_tmp = self.loss_func(feat1, target1, feat2, target2)
                loss = loss + loss_tmp
                num_sample += num_sample_tmp
        
        if num_sample > 0:
            loss = loss / num_sample
        
        return loss



class CycleLoss(nn.Module):
    """
    Computes the loss for ReID learning. 
    Given two adjacent frame, we can compute the loss based 
    on cycle assistance.
    """

    def __init__(self, 
                 margin=0.5, 
                 supervise=1, 
                 place_holder='zero', 
                 loss_names=[], 
                 temperature=None, # if not None, always use the given epsilon for unsupervised training
                 temperature_epsilon=0.5 # used to get the dynamic epsilon for unsupervised training
                 ):

        super(CycleLoss, self).__init__()
        self.margin = margin
        self.supervise = supervise
        self.place_holder = place_holder
        self.loss_names = loss_names

        self.temperature = temperature
        self.temperature_epsilon = temperature_epsilon
        assert self.place_holder in ['none', 'zero', 'median', 'mean']

        

    def _softmax(self, x, dim=0, t=None):
        """
        Args:
            x: 2D tensor, performae softmax along the given dim
        """
        if t == None:
          K = x.shape[dim]
          t = math.log(K + 1) / self.temperature_epsilon
        x = torch.exp(t*x)
        s = torch.sum(x, dim=dim, keepdim=True)
        x = x / s

        return x, t


    def loss_func(self, reid_feat_0, identity_0, reid_feat_1, identity_1, negative=False):
        """
        Get the cycle loss

        Args:
            reid_feat_0: 2D tensor, [N, D]
            identity_0: 1D tensor, [N]
            reid_feat_1: 2D tensor, [M, D]
            identity_1: 1D tensor, [N]
        """
        # reid_feat_0 = reid_feat_0[:3]
        # reid_feat_1 = reid_feat_1[:3]

        N = reid_feat_0.shape[0]
        M = reid_feat_1.shape[0]

        feat = torch.cat((reid_feat_0, reid_feat_1), dim=0) # N+M, D
        similarity = torch.einsum('nd,md->nm', feat, feat) # N+M, N+M, in [-1, 1]

        # set diagonal to -inf
        diag = torch.diag_embed(torch.diag(similarity) - 1e18)
        similarity = similarity + diag
        similarity_true = similarity[similarity >= -1]
        # get gt
        if self.supervise > 0:
            raise NotImplementedError
        else:
            if self.supervise == 0:
                """
                pad a zero placeholder column
                without intra frame constrain
                """
                zeros = torch.zeros(N+M).to(similarity.device) # N+M
                similarity_f = torch.cat((similarity, zeros.unsqueeze(dim=1)), dim=1) # N+M, N+M+1
                assign_f, t = self._softmax(similarity_f, dim=1, t=1) # N+M, N+M+1
                assign_b = assign_f.permute(1, 0) # N+M+1, N+M

                loss_l1 = (assign_f[:, :N+M] - assign_b[:N+M, :]).abs().sum() # N+M, N+M

                if self.margin > 0:
                  margin = self.margin
                else: # adaptive theorectical margin
                  margin = (math.exp(t) - math.exp(-t)) / ((N+M-1)*math.exp(-t) + math.exp(t))

                max_row, _ = torch.topk(assign_f, dim=1, k=2) # N+M, 2
                loss_row = F.relu(max_row[:, 1] + margin - max_row[:, 0]).sum()
                # print('l1: {}, loss_row: {}, loss_col: {}'.format(loss_l1, loss_row, loss_col))

                loss = loss_l1 + loss_row
                num_sample = N + M
            elif self.supervise == -1:
                """
                pad a zero/none/mean/median placeholder column
                with intra frame constrain
                """
                place_holder_type = 'zero' if negative else self.place_holder
                if place_holder_type in ['zero', 'mean', 'median']:
                    place_holder = torch.zeros(N+M).to(similarity.device) # N+M
                    
                    if place_holder_type == 'mean':
                        place_holder = place_holder + similarity_true.mean().detach()
                    elif place_holder_type == 'median':
                        place_holder = place_holder + similarity_true.median().detach()
                    similarity = torch.cat((similarity, place_holder.unsqueeze(dim=1)), dim=1) # N+M, N+M+1
                # print('tempture {}'.format(self.temperature))
                if self.temperature is not None:
                  assert isinstance(self.temperature, float)
                  assign, t = self._softmax(similarity, dim=1, t=self.temperature) # N+M, N+M+1
                else:
                  assign, t = self._softmax(similarity, dim=1) # N+M, N+M+1
                
                if self.margin > 0:
                  margin = self.margin
                else: # adaptive theorectical margin
                  margin = (math.exp(t) - math.exp(-t)) / ((N+M-1)*math.exp(-t) + math.exp(t))

                if not negative:
                    if  len(self.loss_names) > 0 and 'loss_intra' not in self.loss_names: # without intra loss
                        loss_intra = torch.tensor(0.0).to(similarity.device)
                    else:
                        # the similarity intra one frame, all similarity should be 0
                        loss_intra_cur = (assign[:N, :N] - torch.diag_embed(torch.diag(assign[:N, :N]))).abs().sum()
                        loss_intra_pre = (assign[N:N+M, N:N+M] - torch.diag_embed(torch.diag(assign[N:N+M, N:N+M]))).abs().sum()
                        loss_intra = loss_intra_cur + loss_intra_pre

                    if len(self.loss_names) > 0 and 'loss_consistent' not in self.loss_names: # without consistency loss
                        loss_consistent = torch.tensor(0.0).to(similarity.device)
                    else:
                        # the similarity inter two frames, it should be consistency
                        assign_f = assign[:N, N:N+M] # N, M
                        assign_b = assign[N:N+M,:N] # M, N
                        loss_consistent = (assign_f - assign_b.permute(1, 0)).abs().sum()

                    if len(self.loss_names) > 0 and 'loss_inter' not in self.loss_names: # without inter loss
                        loss_inter = torch.tensor(0.0).to(similarity.device)
                    else:
                        max_row, _ = torch.topk(assign, dim=1, k=2) # N+M, 2
                        loss_inter = F.relu(max_row[:, 1] + margin - max_row[:, 0]).sum()   

                    loss = loss_intra + loss_consistent + loss_inter
                    loss_dict = {'id_loss': loss,
                                'id_loss_intra': loss_intra,
                                'id_loss_inter': loss_inter,
                                'id_loss_cons': loss_consistent}
                    num_sample = N + M
                else:
                    loss = (assign[:N+M, :N+M] - torch.diag_embed(torch.diag(assign[:N+M, :N+M]))).abs().sum()
                    num_sample = N + M
                    loss_dict = {'id_loss': loss,
                                'id_loss_intra': torch.zeros_like(loss).detach(),
                                'id_loss_inter': torch.zeros_like(loss).detach(),
                                'id_loss_cons': torch.zeros_like(loss).detach(),}
                    if assign.shape[-1] > N+M: # padded
                        # import pdb; pdb.set_trace()
                        max_row, _ = torch.topk(assign[:, :N+M], dim=1, k=1) # N+M, 1
                        loss_inter = F.relu(max_row[:, 0] + margin - assign[:, N+M]).sum()
                        # loss = loss + loss_inter
                        loss_dict['id_loss'] = loss_dict['id_loss'] + loss_inter
                        loss_dict['id_loss_inter'] = loss_inter
            else:
                raise NotImplementedError
        
        return loss_dict, num_sample, similarity_true


    def forward(self, feat_cur, feat_pre, mask_cur, mask_pre,  target_cur, target_pre, reid_mask_cur, reid_mask_pre, negative):
        """
        Computes the reid loss.
        Arguments:
            feat_cur: B, N, C
            mask_cur: B, N
            target_cur: B, N
            feat_pre: B, N, C
            mask_pre: B, N
            target_pre: B, N   
            reid_mask_cur: B, N
            reid_mask_pre: B N
            negative: B, bool
        Returns:
            reid_loss (Tensor)
        """
        negative = negative.bool()
        # prepare identities for proposals. If trained with GT, no need to prepare identities
        loss_dict = {'id_loss': feat_cur.mean() * 0}
        num_sample = 0

        feat_cur = F.normalize(feat_cur, dim=2)    
        feat_pre = F.normalize(feat_pre, dim=2)

        bs = feat_cur.shape[0]
        similarity = []
        for i in range(bs):
            mask1 = mask_cur[i] * reid_mask_cur[i] # N
            mask2 = mask_pre[i] * reid_mask_pre[i] # N
            feat1 = feat_cur[i][mask1 > 0] # N', C
            feat2 = feat_pre[i][mask2 > 0] # M', C
            target1 = target_cur[i][mask1 > 0] if target_cur is not None else None
            target2 = target_pre[i][mask2 > 0] if target_pre is not None else None
            neg = negative[i]
            if feat1.shape[0] > 0 and feat2.shape[0] > 0:
                loss_dict_tmp, num_sample_tmp, similarity_tmp = self.loss_func(feat1, target1, feat2, target2, neg)
                for k in loss_dict_tmp:
                  if k not in loss_dict:
                    loss_dict[k] = loss_dict_tmp[k]
                  else:
                    loss_dict[k] = loss_dict[k] + loss_dict_tmp[k]
                # loss = loss + loss_tmp
                num_sample += num_sample_tmp
                similarity.append(similarity_tmp.view(-1))
        if num_sample > 0:
            # loss = loss / num_sample
            for k in loss_dict:
              loss_dict[k] = loss_dict[k] / num_sample
            similarity = torch.cat(similarity, dim=0)
        else:
            similarity = torch.as_tensor(0).to(feat_cur.device).float()
        loss_dict.update({
          "sim_mean": similarity.mean(),
          "sim_median": similarity.median(),
          "sim_max": similarity.max(),
          "sim_min": similarity.min(),
          "sim_var": similarity.var(),
          "sim_std": similarity.std(),
          "id_samples": torch.tensor(num_sample).to(similarity.device).float()
        })
        # print("num_sample: {}".format(num_sample))
        return loss_dict
