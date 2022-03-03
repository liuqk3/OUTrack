from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # B, N -> B, N, 1 -> B, N, C
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() # N C H W -> N, H, W, C
    feat = feat.view(feat.size(0), -1, feat.size(3)) # N, HW, C
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)



############################################################
#                functions for GSM
#############################################################
def association_neighbor(score, threshold=None, tool='scipy'):
    """This function association the neighbors of two anchors

    Args:
        score: [bs, num_node1, num_node2, neighbor_k, neighbor_k]
        threshold: float or None, the score that lower than this threshold will
            not be associated.
    """
    if threshold is not None:
        raise NotImplementedError


    if tool in ['scipy', 'lapjv']:
        bs, num_node1, num_node2, neighbor_k, _ = score.size()
        dist = score.view(-1, neighbor_k, neighbor_k)
        dist = 1 - dist
        dist = dist.detach().to(torch.device('cpu')).numpy()

        if tool == 'scipy':
            from scipy.optimize import linear_sum_assignment
            index1, index2 = [], []
            for i in range(dist.shape[0]):
                r_idx, c_idx = linear_sum_assignment(dist[i])
                index1.append(r_idx)
                index2.append(c_idx)

            index1 = torch.Tensor(index1).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long()
            index2 = torch.Tensor(index2).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long()

            index1 = index1.repeat(1, 1, 1, 1, neighbor_k)
            score = torch.gather(score, index=index1, dim=-2)  # [bs, num_node1, num_node2, neighbor_k, neighbor_k]
            score = torch.gather(score, index=index2, dim=-1)  # [bs, num_node1, num_node2, neighbor_k, 1]

        elif tool == 'lapjv':
            import lapjv
            ass1 = []
            for i in range(dist.shape[0]):
                # r_ass = hungarian.lap(dist[i])[0]
                r_ass = lapjv.lapjv(dist[i])[0]
                ass1.append(r_ass)

            ass1 = torch.Tensor(ass1).view(bs, num_node1, num_node2, neighbor_k, 1).to(score.device).long() #  [bs, num_node1, num_node2, neighbor_k, 1]
            score = torch.gather(score, index=ass1, dim=-1)  # [bs, num_node1, num_node2, neighbor_k, 1]

        score = score.view(bs, num_node1, num_node2, neighbor_k)  # [bs, num_node1, num_node2, neighbor_k]
    elif tool == 'min':
        score, _ = score.min(dim=-2)

    return score


def arrange_neighbor(neighbor1, neighbor2, assignment):
    """This function re-arrange the neighbor, so the neighbors will be matched

    Args:
        neighbor1: 3D tensor, [bs, num_node1, neighbor_k, dim]
        neighbor2: 3D tensor, [bs, num_node2, neighbor_k, dim]
        assignment: [bs, num_node1, num_node2, neighbor_k, 2]
    """
    index1, index2 = torch.chunk(assignment, 2, dim=-1) # [bs, num_node1, num_node2, neighbor_k, 1]
    index1 = index1.repeat(1, 1, 1, 1, neighbor1.size(-1)) # [bs, num_node1, num_node2, neighbor_k, dim]
    index2 = index2.repeat(1, 1, 1, 1, neighbor2.size(-1)) # [bs, num_node1, num_node2, neighbor_k, dim]
    neighbor1 = torch.gather(neighbor1, index=index1.long(), dim=-2)
    neighbor2 = torch.gather(neighbor2, index=index2.long(), dim=-2)

    return neighbor1, neighbor2

