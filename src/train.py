from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import json
import torch
from torch import nn
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model, adjust_lr
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from datasets.dataset.jde_json import JointDataset as JointDatasetJson
from utils.utils import get_model_parameters_info
def get_optimizer(model, opt):
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    params = []
    for key, value in model.named_parameters():
        train = True
        if len(opt.train_part) > 0:
            train = False
            for p in opt.train_part:
                if p in key:
                    train = True
                    break
        if train and value.requires_grad:
            if len(opt.train_part):
                print('Train parameter: {}'.format(key))
            params.append(value)
        else:
            value.requires_grad = False
    assert len(params)

    for name, module in model.named_children():
        train = False
        for p in module.parameters():
            if p.requires_grad:
                train = True
                break
        if train:
            module.train()
        else:
            module.eval()
            print('Fix module {}'.format(name))
    optimizer = torch.optim.Adam(params, opt.lr)

    return model, optimizer


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    annotation_paths = data_config['annotation']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    if opt.dataset == 'jde':
        raise RuntimeError("This is the original FairMOT code, and is not needed")
    elif opt.dataset == 'jde_json':
        dataset = JointDatasetJson(opt, dataset_root, trainset_paths, annotation_paths, img_size=(1088, 608), augment=True, transforms=transforms)
    else:
        raise ValueError('Unknown type of dataset {}'.format(opt.dataset))
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    
    print('========== Model parameters info ====')
    parameter_info = get_model_parameters_info(model)
    print(parameter_info)

    model, optimizer = get_optimizer(model, opt)
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer, logger=logger)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, trainer.loss, start_epoch = load_model(model, opt.load_model, optimizer=trainer.optimizer, 
                                                   loss_model=trainer.loss,
                                                   resume=opt.resume, lr=opt.lr, lr_step=opt.lr_step,
                                                   skip_load_param=opt.skip_load_param)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        lr = adjust_lr(epoch=epoch, opt=opt)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} lr {:8f} |'.format(epoch, lr))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        
        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                    epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
        saved = False
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
            saved = True

        logger.write('\n')
        # if epoch in opt.lr_step:
        #     if not saved:
        #         save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
        #                 epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
        #         saved = True
        #     lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        #     print('Drop LR to', lr)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        if len(opt.save_epoch) > 0 and epoch in opt.save_epoch and not saved:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
            saved = True
        elif len(opt.save_epoch) == 0 and epoch % 10 == 0 and not saved: # == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
            saved = True
        elif epoch >= opt.num_epochs and not saved:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                    epoch=epoch, model=model, optimizer=optimizer, loss_model=trainer.loss)
    logger.close()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    opt = opts().parse()
    main(opt)
