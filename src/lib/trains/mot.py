from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, CycasLoss, CycleLoss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer
from utils.utils import get_model_parameters_info

class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        
        if not opt.not_reid:
            self.s_det = nn.Parameter(self.opt.det_uncertainty * torch.ones(1))
            self.s_id = nn.Parameter(self.opt.id_uncertainty * torch.ones(1))           
            if self.opt.reid_loss == 'cross_entropy_loss':
                self.emb_dim = opt.reid_dim
                self.nID = opt.nID
                self.classifier = nn.Linear(self.emb_dim, self.nID)
                print('==== number of parameters in classifier =====' )
                print(get_model_parameters_info(self.classifier))
                #self.TriLoss = TripletLoss()
                self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
                self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            elif self.opt.reid_loss == 'cycas_loss':
                self.IDLoss = CycasLoss(margin=opt.reid_cycle_loss_margin, supervise=opt.reid_cycle_loss_supervise)
            elif self.opt.reid_loss == 'cycle_loss':
                self.IDLoss = CycleLoss(margin=opt.reid_cycle_loss_margin, 
                                             supervise=opt.reid_cycle_loss_supervise,
                                             place_holder=opt.reid_cycle_loss_placeholder,
                                             loss_names=opt.reid_cycle_loss_names,
                                             temperature=opt.reid_cycle_loss_temperature,
                                             temperature_epsilon=opt.reid_cycle_loss_temperature_epsilon
                                             )
            else:
                raise ValueError('Unknown type of reid loss: {}'.format(self.opt.reid_loss))

    def forward(self, outputs, batch, outputs_pre=None, batch_pre=None):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, occ_loss, occ_off_loss = 0, 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output_pre = outputs_pre[s] if outputs_pre is not None else None
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.occlusion:
                assert opt.occlusion_weight > 0
                if not opt.mse_loss:
                    output['occlusion'] = _sigmoid(output['occlusion'])
                occ_loss += self.crit(output['occlusion'], batch['occlusion']) / opt.num_stacks
                if opt.occlusion_offset:
                    occ_off_loss += self.crit_reg(output['occlusion_offset'], batch['occ_mask'],
                                          batch['occ_ind'], batch['occ_offset']) / opt.num_stacks    
            # TODO: modify this classification loss to cycle reid loss
            if not opt.not_reid:
                if opt.id_weight > 0:
                    if opt.reid_loss == 'cross_entropy_loss':
                        id_target = batch['ids'][batch['reg_mask'] > 0]
                        id_feat = _tranpose_and_gather_feat(output['id'], batch['ind']) # B, N, C
                        id_feat = id_feat[batch['reg_mask'] > 0].contiguous()
                        id_feat = self.emb_scale * F.normalize(id_feat)
                        id_output = self.classifier(id_feat).contiguous()
                        id_loss_info = self.IDLoss(id_output, id_target)
                    elif opt.reid_loss in ['cycle_loss', 'cycle_loss2']:
                        id_feat = _tranpose_and_gather_feat(output['id'], batch['ind']) # B, N, C
                        id_feat_pre = _tranpose_and_gather_feat(output_pre['id'], batch_pre['ind']) # B, N, C
                        id_loss_info = self.IDLoss(feat_cur=id_feat, 
                                                        feat_pre=id_feat_pre, 
                                                        mask_cur=batch['reg_mask'], 
                                                        mask_pre=batch_pre['reg_mask'],
                                                        target_cur=batch['ids'],
                                                        target_pre=batch_pre['ids'],
                                                        negative=batch_pre['negative'],
                                                        reid_mask_cur=batch['reid_box_mask'],
                                                        reid_mask_pre=batch_pre['reid_box_mask'])
                    else:
                        raise ValueError(opt.reid_loss)
                    if isinstance(id_loss_info, dict):
                        id_loss = id_loss + id_loss_info['id_loss'] * opt.id_weight
                    else:
                        id_loss = id_loss + id_loss_info * opt.id_weight
                        id_loss_info = {"id_loss": id_loss}
            #loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.occlusion_weight * occ_loss + opt.occlusion_off_weight * occ_off_loss
        if not self.opt.not_reid:
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss 
            if not self.opt.loss_not_plus_task_weight:
                loss = loss + (self.s_det + self.s_id)
            loss *= 0.5
            weight_dict = {"det_weight": self.s_det, "id_weight": self.s_id}
        else:
            loss = det_loss

        #print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        if not opt.not_reid:
            loss_stats.update(id_loss_info)
            loss_stats.update(weight_dict)
        if opt.occlusion:
            loss_stats['occ_loss'] = occ_loss
            if opt.occlusion_offset:
                loss_stats['occ_off_loss'] = occ_off_loss
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None, logger=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer, logger=logger)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        if not opt.not_reid:
            loss_states.append('id_loss')
        if opt.occlusion:
            loss_states.append('occ_loss')
            if opt.occlusion_offset:
                loss_states.append('occ_off_loss')
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
