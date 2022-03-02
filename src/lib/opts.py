from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('--task', default='mot', help='mot')
    # self.parser.add_argument('--dataset_root', default='/home/liuqk/Dataset/jde_fairmot/', 
    #                         help='dataset root directory')
    self.parser.add_argument('--dataset', default='jde_json',
                            help='jde')
    self.parser.add_argument('--exp_id', default='default',
                             help='the exp id')
    self.parser.add_argument('--output', default='',
                             help='output dir to save the experiments output data')
    self.parser.add_argument('--test', action='store_true')
    #self.parser.add_argument('--load_model', default='../models/ctdet_coco_dla_2x.pth',
                             #help='path to pretrained model')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--skip_load_param', type=str, default='',
                             help='skip load the parameters that contain the given key')
    self.parser.add_argument('--resume', action='store_true', default=True,
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    self.parser.add_argument('--debug', type=int, default=0, 
                             help='0: no debug. ' \
                                  '1: visilualize some images'\
                                  '2: save some images'\
                                  '3: just test dataloader') 
    self.parser.add_argument('--show_images', type=str, default='',
                             help='the name to show images') 
    self.parser.add_argument('--save_images', type=str, default='',
                             help='the name to show images') 
    self.parser.add_argument('--show_border', type=int, default=50,
                             help='the borader to pad images to show') 
    self.parser.add_argument('--show_tango_color', action='store_true', default=50,
                             help='the borader to pad images to show') 
    self.parser.add_argument('--show_max_size', type=int, default=1000,
                             help='max size to show') 
    self.parser.add_argument('--pause', action='store_true', default=False,
                             help='pause') 

      
    # system
    self.parser.add_argument('--gpus', default='0, 1',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=1, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--save_epoch', type=str, default='',
                             help='Epochs to save model')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                             help='visualization threshold.')
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                  'dla_34 | hrnet_18')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '256 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.') 

    # train
    self.parser.add_argument('--clip_box', action='store_true', default=False,
                             help='clip the box to image.')
    self.parser.add_argument('--amp', action='store_true', default=False,
                             help='train with amp')
    self.parser.add_argument('--lr', type=float, default=1e-4,
                             help='learning rate for batch size 12.')
    self.parser.add_argument('--min_lr_ratio', type=float, default=0.0,
                             help='min learning rate for the given ratio. Effective for cosine and linear lr strategy')
    self.parser.add_argument('--lr_step', type=str, default='20',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--lr_strategy', type=str, default='step',
                             choices=['step', 'cosine', 'linear'],
                             help='lr adjust strategy')
    self.parser.add_argument('--num_epochs', type=int, default=30,
                             help='total training epochs.')
    self.parser.add_argument('--warmup_epochs', type=int, default=0,
                             help='epochs to warmup.')
    self.parser.add_argument('--batch_size', type=int, default=12,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=10,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--mot_vis_thr', type=float, default=-1,
                             help='the visibility threshold to filter mot bounding boxes')
    self.parser.add_argument('--frame_range', type=int, default=10,
                             help='the max frame dist for finding adjacent frames')
    self.parser.add_argument('--frame_pre', type=str, default='adjacent',
                             choices=['none', 'adjacent', 'augment'],
                             help='the max frame dist for finding adjacent frames')
    # self.parser.add_argument('--frame_pre', action='store_true', default=False,
    #                          help='get the adjacent frame')  
    self.parser.add_argument('--negative_pre', type=float, default=-1.0,
                             help='the probability to load a negative pre frame')
    self.parser.add_argument('--frame_pre_trian_det', action='store_true', default=False,
                             help='using previous frame to train the model for detection')      
    self.parser.add_argument('--not_train_loss_para', action='store_true', default=False,
                             help='not train the loss in parameters')                         
    self.parser.add_argument('--train_part', type=str, default='',
                             help='train the part of models, if empty, train all the modules')   

    # test
    self.parser.add_argument('--K', type=int, default=500,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    # tracking
    self.parser.add_argument('--half_track', default=False,
                             help='track with half video')
    self.parser.add_argument('--only_det', default=False, 
                             help='only detection without tracking')
    self.parser.add_argument('--det_dir', type=str, default='',
                             help='if provided, using the given detections to track')
    self.parser.add_argument('--track_type', type=str, default='private_track',
                             choices=['private_track', 'provide_track', 'public_track'],
                             help='whic type of tracking to perform: ' 
                              'private_track: tracking with private detection, detection and tracking are performed jointly'
                              'public_track: given the public detections (det_dir), perform tracking just like CenterTrack'
                              'provide_track: given the detections (det_dir), directly using the provided detection to track')
    self.parser.add_argument('--test_mot16', default=False, help='test mot16')
    self.parser.add_argument('--val_mot15', default=False, help='val mot15')
    self.parser.add_argument('--test_mot15', default=False, help='test mot15')
    self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
    self.parser.add_argument('--test_mot17', default=False, help='test mot17')
    self.parser.add_argument('--val_mot17', default=False, 
                             help='val mot17')
    self.parser.add_argument('--val_mot20', default=False, help='val mot20')
    self.parser.add_argument('--test_mot20', default=False, help='test mot20')
    self.parser.add_argument('--val_hie', default=False, help='val hie')
    self.parser.add_argument('--test_hie', default=False, help='test hie')
    self.parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
    self.parser.add_argument('--reid_thres', type=float, default=0.4, help='threshold for reid based data association')
    self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
    self.parser.add_argument('--occlusion_thres', type=float, default=0.1, help='confidence thresh for occlusion centers')
    self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
    self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    self.parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    self.parser.add_argument('--input_video', type=str,
                             default='../videos/MOT16-03.mp4',
                             help='path to the input video')
    self.parser.add_argument('--output_format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output_root', type=str, default='../demos', help='expected output root path')
    self.parser.add_argument('--reid_feat_type', type=str, default='momentum',
                             choices=['momentum', 'latest', 'all_min', 'all_mean'],
                             help='reid feature type used in tracklets for data association')
    self.parser.add_argument('--gmm', action='store_true',
                             help='use gmm model to update features for tracklets')
    self.parser.add_argument('--lost_frame_range', type=float, default=1,
                             help='frame window to find lost objects, if less than 1, denotes seconds, otherwise denotes frames')   
    self.parser.add_argument('--gt_type', type=str, default='',
                             help='the gt type to evaluate, used to load the modified gt')           
    # mot
    self.parser.add_argument('--data_cfg', type=str,
                             default='src/lib/cfg/mot17_half.json',
                             help='load data from cfg')
    self.parser.add_argument('--data_dir', type=str, default='')


    self.parser.add_argument('--load_gsm', default='',
                            help='load gsm model')


    # loss
    self.parser.add_argument('--det_uncertainty', type=float, default=-1.85,
                             help='uncertainty for detection')
    self.parser.add_argument('--id_uncertainty', type=float, default=-1.05,
                             help='uncertainty for reid')
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
    self.parser.add_argument('--loss_not_plus_task_weight', action='store_true', default=False,
                             help='plus the final loss with task weight')
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--not_reid', action='store_true', default=False,
                             help='not with reid branch')
    self.parser.add_argument('--id_weight', type=float, default=1,
                             help='loss weight for id')
    self.parser.add_argument('--reid_dim', type=int, default=128,
                             help='feature dim for reid')
    self.parser.add_argument('--reid_loss', type=str, default='cross_entropy_loss',
                             choices=['cross_entropy_loss', 'cycas_loss', 'cycle_loss'],
                             help='the loss function for reid')
    self.parser.add_argument('--reid_cycle_loss_supervise', type=int, default=1,
                             help='supervise the reid cycle loss for reid')
    self.parser.add_argument('--reid_cycle_loss_placeholder', type=str, default='none',
                             help='the placeholder for reid cycle loss')
    self.parser.add_argument('--reid_cycle_loss_temperature', type=float, default=None,
                             help='the temperature for softmax for unsupervised training, if none, get the dynamic tempterature')
    self.parser.add_argument('--reid_cycle_loss_temperature_epsilon', type=float, default=0.5,
                             help='the parameter to get the dynamic temperature for unsupervised training, default 0.5')
    self.parser.add_argument('--reid_cycle_loss_names', type=str, default='',
                             help='the reid loss names')
    self.parser.add_argument('--reid_cycle_loss_margin', type=float, default=0.5,
                             help='supervise the reid cycle loss for reid')
    self.parser.add_argument('--reid_area', type=str, default='',
                             help='the are in the image, the boxes locates within this area is account for reid loss, default all boxes')
    self.parser.add_argument('--ltrb', default=True,
                             help='regress left, top, right, bottom of bbox')
    self.parser.add_argument('--occlusion', action='store_true', default=False,
                             help='detect occlusions')
    self.parser.add_argument('--occlusion_iou_thr', type=float, default=0.7,
                             help='the iou to compute occlusions')
    self.parser.add_argument('--occlusion_offset', action='store_true', default=False,
                             help='regress occlusions center offset')                            
    self.parser.add_argument('--occlusion_weight', type=float, default=0.5,
                             help='loss weight for occlusion')
    self.parser.add_argument('--occlusion_off_weight', type=float, default=0.5,
                             help='loss weight for occlusion center offsets.')
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    
    if opt.debug:
      opt.gpus = '0'
      opt.batch_size = 1
      opt.num_workers = 0
    if opt.show_images != '':
      opt.show_images = opt.show_images.strip().split(',')
    else:
      opt.show_images = []
    if opt.save_images != '':
      opt.save_images = opt.save_images.strip().split(',')
    else:
      opt.save_images = []
    
    if opt.reid_loss not in ['cycas_loss', 'cycle_loss']:
      opt.frame_pre = 'none'
    if opt.reid_loss == 'cycas_loss':
      opt.negative_pre = -1
    if opt.reid_area == '':
      opt.reid_area = []
    else:
      opt.reid_area = opt.reid_area.split(',')
      opt.reid_area = [float(i) for i in opt.reid_area]
    
    if opt.skip_load_param != '':
      opt.skip_load_param = opt.skip_load_param.strip().split(',')
    else:
      opt.skip_load_param = []
    
    if opt.reid_cycle_loss_names != '':
      opt.reid_cycle_loss_names = opt.reid_cycle_loss_names.strip().split(',')
    else:
      opt.reid_cycle_loss_names = []

    if opt.track_type != 'private_track':
      assert opt.det_dir != '', 'detections should be provided'
      assert os.path.exists(opt.det_dir), '{} not exists!'.format(opt.det_dir)

    opt.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = sorted([int(i) for i in opt.lr_step.split(',')])
    # import pdb;pdb.set_trace()
    opt.save_epoch = [int(i) for i in opt.save_epoch.strip().split(',') if opt.save_epoch != '']
    opt.train_part = opt.train_part.strip().split(',')

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 256
    opt.pad = 31
    opt.num_stacks = 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    if opt.data_dir == '':
      opt.data_dir = os.path.abspath(os.path.join(opt.root_dir, 'data'))
    if opt.output != '':
      opt.exp_dir = os.path.join(opt.output, opt.task)
    else:
      opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    # import pdb;
    # pdb.set_trace()
    if opt.resume:
      model_dir = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
      # import pdb; pdb.set_trace()
      # import pdb; pdb.set_trace()
      if opt.exp_id not in opt.load_model:
        model_last = os.path.join(model_dir, 'model_last.pth')
        if os.path.exists(model_last):
          opt.load_model = model_last
          print('===> Resume training using {}'.format(model_last))
        else:
          opt.resume = False
          print('===> No last model. Training using {}'.format(opt.load_model))
    # if opt.resume and opt.load_model == '':
    #   model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
    #   opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)

    if opt.task == 'mot':
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.ltrb else 4,
                   'id': opt.reid_dim}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.occlusion:
        opt.heads.update({'occlusion': 1})
      if opt.occlusion_offset:
        opt.heads.update({'occlusion_offset': 2})
      opt.nID = dataset.nID
      opt.img_size = (1088, 608)
      #opt.img_size = (864, 480)
      #opt.img_size = (576, 320)
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'jde', 'nID': 14455},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)


    # get the config for gsm model
    if opt.load_gsm != '':
      basename = os.path.basename(opt.load_gsm)
      config_path = opt.load_gsm.replace(basename, 'config.json')
      assert os.path.exists(opt.load_gsm) and os.path.exists(config_path)
      from models.networks.gsm.config import load_config as load_gsm_config
      gsm_config = load_gsm_config(config_path)['GraphSimilarity']['init_args']
      opt.gsm_config = gsm_config
      print('Load GSM model from: {}'.format(opt.load_gsm))


    return opt
