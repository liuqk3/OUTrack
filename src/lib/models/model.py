from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import math
import bisect

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.pose_dla_conv import get_pose_net as get_dla_conv

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dlaconv': get_dla_conv,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def create_gsm(args):
  from .networks.gsm.similarity_model import GraphSimilarity 
  return GraphSimilarity(**args)

def load_model(model, model_path, optimizer=None, loss_model=None, resume=False, 
               lr=None, lr_step=None, return_epoch=False, skip_load_param=[], model_param_key='state_dict'):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint[model_param_key]
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      skip = False
      for sp in skip_load_param:
        if sp in k:
          skip = True
          break
      if (state_dict[k].shape != model_state_dict[k].shape) or skip:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      # start_lr = adjust_lr(start_epoch, opt)
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if loss_model is not None and resume:
    if 'loss_state_dict' in checkpoint:
      loss_model.load_state_dict(checkpoint['loss_state_dict'])
    else:
      print('No loss state dict in checkpoint.')
  if optimizer is not None:
    if loss_model is None:
      return model, optimizer, start_epoch
    else:
      return model, optimizer, loss_model, start_epoch
  elif return_epoch:
    start_epoch = checkpoint['epoch']
    return model, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None, loss_model=None):
  print('Save model to {}'.format(path))
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  if not (loss_model is None):
    data['loss_state_dict'] = loss_model.module.state_dict() if hasattr(loss_model, 'module') else loss_model.state_dict()
  version = torch.__version__
  if version >= '1.6.0':
    torch.save(data, path, _use_new_zipfile_serialization=False)
  else:
    torch.save(data, path)


def adjust_lr(epoch, opt):
    """
    adjust lr
    epoch start from 1
    """
    lr = opt.lr
    if opt.warmup_epochs > 0 and epoch <= opt.warmup_epochs:
        lr = lr * epoch * 1.0 / opt.warmup_epochs
        print('===> Adjust LR to {} for warmup, start with {}'.format(lr, opt.lr))
    if epoch > opt.warmup_epochs:
        if opt.lr_strategy == 'step':
          if epoch - 1 in opt.lr_step:
            i = opt.lr_step.index(epoch - 1)
            i = i + 1
          else:
            i = bisect.bisect(opt.lr_step, epoch - 1)
          lr = lr * (0.1 ** i)
          print('===> Adjust LR to {} steply, start with {}, step is {}'.format(lr, opt.lr, str(opt.lr_step)))       
        elif opt.lr_strategy == 'cosine':
            lr = lr * max(0.5 * (1 + math.cos(math.pi * (epoch - 1 - opt.warmup_epochs) / (opt.num_epochs - opt.warmup_epochs))), opt.min_lr_ratio)
            print('===> Adjust LR to {} with cosine, start with {}'.format(lr, opt.lr))
        elif opt.lr_strategy == 'linear':
            lr = lr * max(1 - (epoch - 1 - opt.warmup_epochs) / (opt.num_epochs - opt.warmup_epochs), opt.min_lr_ratio)
            print('===> Adjust LR to {} linearly, start with {}, step is {}'.format(lr, opt.lr, str(opt.lr_step)))       
        else:
            raise ValueError('Unknown lr strategy: {}'.format(opt.lr_strategy))
    return lr