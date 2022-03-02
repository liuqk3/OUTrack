from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim,  # x1, y1, x2, y2, score, class
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def occlusion_post_process(centers, c, s, h, w, num_classes):
  # centers: batch x max_dets x dim, # x, y, score, class
  # return 1-based class det dict
  ret = []
  for i in range(centers.shape[0]):
    top_preds = {}
    centers[i, :, 0:2] = transform_preds(centers[i, :, 0:2], c[i], s[i], (w, h))
    classes = centers[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        centers[i, inds, 0:2].astype(np.float32),
        centers[i, inds, 2:3].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret
