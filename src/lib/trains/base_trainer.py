from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from torch import nn
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False



class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss, amp=False):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.amp = amp and AMP

    def forward(self, batch, batch_pre=None):
        if batch_pre is None:
            if self.amp:
                with autocast():
                    outputs = self.model(batch['input'])
            else:
                outputs = self.model(batch['input'])
            outputs_pre = None
        else:
            batch_size = batch['input'].shape[0]
            if self.amp:
                with autocast():
                    outputs_ = self.model(torch.cat((batch['input'], batch_pre['input']), dim=0))
            else:
                outputs_ = self.model(torch.cat((batch['input'], batch_pre['input']), dim=0))              
            outputs = []
            outputs_pre = []
            for i in range(len(outputs_)):
                outputs_tmp = {}
                outputs_pre_tmp = {}
                for k in outputs_[i].keys():
                    d = torch.split(outputs_[i][k], split_size_or_sections=batch_size)
                    outputs_tmp[k] = d[0]
                    outputs_pre_tmp[k] = d[1]
                outputs.append(outputs_tmp)
                outputs_pre.append(outputs_pre_tmp)
            
            # if self.amp:
            #     with autocast():
            #         outputs = self.model(batch['input'])  
            #         outputs_pre = self.model(batch_pre['input'])  
            # else:
            #     outputs = self.model(batch['input'])  
            #     outputs_pre = self.model(batch_pre['input'])        

        if self.amp:
            with autocast():
                loss, loss_stats = self.loss(outputs, batch, outputs_pre, batch_pre)
        else:
            loss, loss_stats = self.loss(outputs, batch, outputs_pre, batch_pre)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None, logger=None):
        self.opt = opt
        self.logger = logger
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.amp = opt.amp and AMP
        self.model_with_loss = ModleWithLoss(model, self.loss, amp=self.amp)
        if not opt.not_train_loss_para:
            self.optimizer.add_param_group({'params': self.loss.parameters()})
        if self.amp:
            self.scaler = GradScaler()

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def data_to_device(self, data):
        if data is None:
            return data
        for k in data:
            if k != 'meta' and k != 'pre_data':
                data[k] = data[k].to(device=self.opt.device, non_blocking=True)
        return data

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            if len(self.opt.train_part) != 0:
                for _, module in model_with_loss.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        train = True
                        for _, p in module.named_parameters():
                            if not p.requires_grad:
                                train = False
                        if not train:
                            module.eval()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if self.opt.debug == 3:
                continue

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            if 'pre_data' in batch:
                batch_pre = batch['pre_data']
                del batch['pre_data']
            else:
                batch_pre = None
            
            # import pdb; pdb.set_trace()
            batch = self.data_to_device(batch)
            batch_pre = self.data_to_device(batch_pre)

            output, loss, loss_stats = model_with_loss(batch, batch_pre)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}/{1}][{2}/{3}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, self.opt.num_epochs, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in loss_stats:
                if l not in avg_loss_stats:
                    avg_loss_stats[l] = AverageMeter()
            # for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    log_info = '{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)
                    print(log_info)
                    if self.logger is not None:
                        self.logger.write_training_log(log_info)
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
