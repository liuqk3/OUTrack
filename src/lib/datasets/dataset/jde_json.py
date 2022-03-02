import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from utils.debugger import Debugger
from utils.box import occlusion_boxes, iou_ab


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608), det_path=None, seq_name=''):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]
        self.seq_name = seq_name
        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.det_path = det_path
        if self.det_path != None:
            self.detections = np.loadtxt(self.det_path, delimiter=',') # the same with public detection in MOT
            if 'gt.txt' in self.det_path:
                index = self.detections[:, 7] == 1 # keep person
                self.detections = self.detections[index]
                # index = self.detections[:, 8] >= 0.2 # visibility
                # self.detections = self.detections[index]

        else:
            self.detections = None

        assert self.nF > 0, 'No images found in ' + path


    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, ratio, padw, padh = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        
        frame_id = int(os.path.basename(img_path).split('.')[0])
        if self.detections is not None:
            # import pdb; pdb.set_trace()
            detection = copy.deepcopy(self.detections[self.detections[:, 0]==frame_id])
            # map to resized image
            detection[:, 2:6] = detection[:, 2:6] * ratio
            detection[:, 2] = detection[:, 2] + padw
            detection[:, 3] = detection[:, 3] + padh
            detection_meta = {
                'detection': detection,
                'ratio': ratio,
                'padw': padw,
                'padh': padh
            }
        else:
            detection_meta = None
        
        img_info = {
            'im_path': img_path,
            'video_name': self.seq_name,
            'frame_id': frame_id,
        }

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_info, img, img0, None, detection_meta

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, ratio, padw, padh = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        frame_id = int(os.path.basename(img_path).split('.')[0])
        if self.detections is not None:
            
            detection = self.detections[self.detections[:, 0]==frame_id]
            # map to resized image
            detection[:, 2:6] = detection[:, 2:6] * ratio
            detection[:, 2] = detection[:, 2] + padw
            detection[:, 3] = detection[:, 3] + padh
            detection_meta = {
                'detection': detection,
                'ratio': ratio,
                'padw': padw,
                'padh': padh
            }
        else:
            detection_meta = {
                'ratio': ratio,
            }
        img_info = {
            'im_path': img_path,
            'video_name': self.seq_name,
            'frame_id': frame_id,
        }
        return img_info, img, img0, None, detection_meta

    def __len__(self):
        return self.nF  # number of files
class LoadImagesAndLabels:  # for training
    def __init__(self, opt, path, annotation_path, img_size=(1088, 608), augment=False, transforms=None):
        
        self.opt = opt
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        # self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        #                     for x in self.img_files]
        
        annotations_ = json.load(open(annotation_path, 'r'))
        annotations = {
            'images': {},
            'annotations':{}
        }
        for im_name in self.img_files:
            annotations['images'][im_name] = annotations_['images'][im_name]
            anns = []
            for ann in annotations_['annotations'][im_name]:
                anns.append([ann['label'], ann['object_id']] + ann['xywh_norm'])
            annotations['annotations'][im_name] = np.array(anns, dtype=np.float32).reshape(-1, 6)
        self.annotations = annotations

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        labels = copy.deepcopy(self.annotations['annotations'][img_path])
        return self.get_data(img_path, labels)

    def get_data(self, img_path, labels, aug_corpus=None, reid_area=None):
        """
        reid_area is used to select a region. Objects that locates in 
        the region is used to compute the reid loss. 
        """
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        if len(self.opt.reid_area) == 0: # keep all boxes to compute reid loss
            reid_box_mask = np.ones(labels.shape[0], dtype=np.int64)
        else:
            if reid_area is None:
                ratio = random.choice(self.opt.reid_area)
                rest = 1 - ratio
                choices = []
                for i in range(100):
                    r = i / 100.
                    if r < rest:
                        choices.append(r)
                    else:
                        break
                start_w = random.choice(choices)
                start_h = random.choice(choices)
                
                reid_area = np.array([[start_w, start_h, start_w + ratio, start_h + ratio]])

            # check if boxes are in the region
            # xywh -> xyxy
            norm_box = labels[:, 2:6].copy()
            norm_box[:, 2] = norm_box[:, 0] + norm_box[:, 2]
            norm_box[:, 3] = norm_box[:, 1] + norm_box[:, 3]
            iou_ = iou_ab(norm_box, reid_area, iou_type=4) # num_box, 1
            index = iou_[:, 0] > 0.2
            reid_box_mask = np.asanyarray(index, dtype=np.int64)


        # Load labels
        if labels.size > 0:
            labels0 = copy.deepcopy(labels)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, aug_corpus = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20), aug_corpus=aug_corpus)

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            if 'lr_flipped' in aug_corpus:
                lr_flipped = aug_corpus['lr_flipped']     
            else:
                lr_flip = True
                lr_flipped = lr_flip & (random.random() > 0.5)
                aug_corpus['lr_flipped'] = lr_flipped
            if lr_flipped:
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w), aug_corpus, reid_box_mask, reid_area 

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5), aug_corpus=None):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    height = img.shape[0]
    width = img.shape[1]

    if aug_corpus is None:
        border = 0  # width of added border (optional)
        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T02 = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border
        T12 = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border
        T[0, 2] = T02 # x translation (pixels)
        T[1, 2] = T12  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S01 = (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
        S10 = (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
        S[0, 1] = math.tan(S01)  # x shear (deg)
        S[1, 0] = math.tan(S10)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

        aug_corpus = {
            'angle': a,
            'scale': s,
            'translation02': T02,
            'translation12': T12,
            'shear01': S01,
            'shear10': S10
        }
    else:
        # get warp matrix for adjacent frames
        disturb = random.choice(range(5, 20, 1))
        disturb = disturb / 100
        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) * disturb + aug_corpus['angle']
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) * disturb + aug_corpus['scale']
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] * disturb + aug_corpus['translation02']  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] * disturb + aug_corpus['translation12']  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S01 = (random.random() * (shear[1] - shear[0]) * disturb) * math.pi / 180 + aug_corpus['shear01']
        S10 = (random.random() * (shear[1] - shear[0]) * disturb) * math.pi / 180 + aug_corpus['shear10']
        S[0, 1] = math.tan(S01)  # x shear (deg)
        S[1, 0] = math.tan(S10)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, aug_corpus
    else:
        return imw


def prepare_adjacent_frames(annotations, frame_range, ds_name):
    print('==> {}: prepare adjacent frames'.format(ds_name))
    videos = {}
    for im_name in annotations['images'].keys():
        im_info = annotations['images'][im_name]
        if 'video_name' not in im_info:
            continue
        video_name = im_info['video_name']
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(im_name)
    
    adjacent_frames = {}
    for video_name in videos.keys():
        im_names = videos[video_name]
        frame_ids = []
        for im_name in im_names:
            frame_id = int(annotations['images'][im_name]['frame_id'])
            frame_ids.append(frame_id)
        frame_ids = np.array(frame_ids).reshape(len(im_names), 1)
        frame_dist = np.abs(frame_ids - frame_ids.transpose())
        for idx in range(len(frame_dist)):
            one_dist = frame_dist[idx]
            valid_idices = np.where(one_dist <= frame_range)[0]
            adjacent_frames[im_names[idx]] = []
            for i in valid_idices:
                if i == idx:
                    continue
                adjacent_frames[im_names[idx]].append(im_names[i])
    
    annotations['adjacent_frames'] = adjacent_frames
    annotations['videos'] = videos
    print('==> {}: prepare adjacent frames done'.format(ds_name))
    return annotations


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, annotation_paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        if self.opt.data_dir != '':
            root = self.opt.data_dir

        if self.opt.debug:
            self.debugger = Debugger(self.opt)

        self.root = root
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.annotations = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            path = os.path.join(self.opt.project_root, 'src', path)
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [x.replace('\n', '') for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            anno_path = os.path.join(root, annotation_paths[ds])
            annotations_ = json.load(open(anno_path, 'r'))
            annotations = {
                'images': {},
                'annotations': {},
            }
            ann_count = 0
            for img_file in self.img_files[ds]:
                annotations['images'][img_file] = annotations_['images'][img_file]
                anns = []
                for idx in range(len(annotations_['annotations'][img_file])):
                    ann = annotations_['annotations'][img_file][idx]
                    if self.opt.reid_loss == 'cross_entropy_loss':
                        if self.opt.reid_cycle_loss_supervise <= 0:
                            ann_count += 1
                            ann['object_id'] = ann_count

                    if self.opt.reid_loss == 'cycas_loss':
                        if self.opt.reid_cycle_loss_supervise > 0:
                            # NOTE Is this the optimal setting for supervised reid learning
                            if not annotations['images'][img_file]['has_object_id']:
                                # this image has no object id labelled
                                if 'video_name' in annotations['images'][img_file]:
                                    # if this is a video and no object id is labelled,
                                    # we treat it as static image
                                    del annotations['images'][img_file]['video_name']
                                ann['object_id'] = idx + 1
                    if self.opt.reid_loss == 'matching_loss':
                        if self.opt.reid_cycle_loss_supervise > 0:
                            raise NotImplementedError
                    if self.opt.mot_vis_thr >= 0:
                        if 'visibility' in ann:
                            if float(ann['visibility']) < self.opt.mot_vis_thr:
                                continue
                    anns.append([ann['label'], ann['object_id']] + ann['xywh_norm'])
                annotations['annotations'][img_file] = np.array(anns, dtype=np.float32).reshape(-1, 6)
                # annotations['annotations'][img_file] = annotations_['annotations'][img_file]
            annotations = prepare_adjacent_frames(annotations, self.opt.frame_range, ds_name=ds)
            self.annotations[ds] = annotations

        for ds in self.annotations.keys():
            annos = self.annotations[ds]
            max_index = -1
            for im_name in annos['annotations'].keys():
                lb = annos['annotations'][im_name]
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def gen_target(self, imgs, labels, reid_box_mask, dataset_name, pre=False, pre_for_det=False):
        """
        For previous data, only the ind and ind mask need to be generated
        
        """
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[dataset_name]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]

        # the following are both needed for current and previous data
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        if not self.opt.not_reid:
            ids = np.zeros((self.max_objs, ), dtype=np.int64)
            reid_box_masks = np.zeros((self.max_objs, ), dtype=np.int64)

        if not pre or (pre and pre_for_det):
            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
            if self.opt.ltrb:
                wh = np.zeros((self.max_objs, 4), dtype=np.float32)
            else:
                wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        consider_ltrb = []
        consider_amodal_ltrb = []
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3] # ltrb
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3] # ltrb

            consider_amodal_ltrb.append(copy.deepcopy(bbox_amodal))
            consider_ltrb.append(copy.deepcopy(bbox_xy))

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg_mask[k] = 1
                if not self.opt.not_reid:
                    ids[k] = label[1]
                    reid_box_masks[k] = reid_box_mask[k]
                
                if not pre or (pre and pre_for_det):
                    draw_gaussian(hm[cls_id], ct_int, radius)
                    if self.opt.ltrb:
                        wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                                bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                    else:
                        wh[k] = 1. * w, 1. * h
                    # ind[k] = ct_int[1] * output_w + ct_int[0]
                    # reg_mask[k] = 1
                    # if not self.opt.not_reid:
                    #     ids[k] = label[1]
                    reg[k] = ct - ct_int
                    bbox_xys[k] = bbox_xy


        ret = {'input': imgs, 'reg_mask': reg_mask, 'ind': ind}
        if not self.opt.not_reid:
            ret['ids'] = ids
            ret['reid_box_mask'] = reid_box_masks

        if not pre or (pre and pre_for_det):
            ret.update({'hm': hm, 'wh': wh, 'reg': reg, 'bbox': bbox_xys})
            if self.opt.occlusion:
                occ_ret = self.gen_occlusion_target(output_h, output_w, consider_ltrb, consider_amodal_ltrb)
                ret.update(occ_ret)
        return ret      

    def gen_occlusion_target(self, output_h, output_w, boxes, boxes_amodal):
        """add the occlusion map based on the bounding boxes.
        Note that the boxes have already been mapped to the output size
        """
        ret = {}
        ret['occlusion'] = np.zeros((1, output_h, output_w), dtype=np.float32)
        ret['occ_ind'] = np.zeros((self.max_objs * 4, ), dtype=np.int64)
        ret['occ_mask'] = np.zeros((self.max_objs * 4, ), dtype=np.uint8)
        if self.opt.occlusion_offset:
            ret['occ_offset'] = np.zeros((self.max_objs * 4, 2), dtype=np.float32)

        if isinstance(boxes, list):
            boxes = np.array(boxes, dtype=np.float32)
        if boxes.size == 0:
            return ret
        # occ_boxes = occlusion_boxes(boxes, iou_type=1, iou_thr=0.7) # [N, 4], (x1, y1, x2, y2)
        occ_boxes = occlusion_boxes(boxes, iou_type=1, iou_thr=self.opt.occlusion_iou_thr) # [N, 4], (x1, y1, x2, y2)
        if occ_boxes.size > 0:
            occ_boxes[:, [0, 2]] = np.clip(occ_boxes[:, [0, 2]], 0, output_w - 1)
            occ_boxes[:, [1, 3]] = np.clip(occ_boxes[:, [1, 3]], 0, output_h - 1)
            if occ_boxes.shape[0] > ret['occ_ind'].shape[0]:
                print('found occlusion: {}, but the max number of objects is {}'.format(occ_boxes.shape[0], ret['occ_ind'].shape[0]))
            
            draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
            for i in range(min(occ_boxes.shape[0], ret['occ_ind'].shape[0])):
                box = occ_boxes[i]
                h, w = box[3] - box[1], box[2] - box[0]
                if h <= 0 or w <= 0:
                    continue
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(ret['occlusion'][0], ct_int, radius)

                ret['occ_ind'][i] = ct_int[1] * output_w + ct_int[0]
                ret['occ_mask'][i] = 1 # take into consideration
                if 'occ_offset' in ret:
                    ret['occ_offset'][i] = ct - ct_int
        return ret

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.img_files.keys())[i]
                start_index = c

        img_name = self.img_files[ds][files_index - start_index]
        labels = copy.deepcopy(self.annotations[ds]['annotations'][img_name])
        img_path = os.path.join(self.root, img_name)
        imgs, labels, img_path, (input_h, input_w), warp_coprus, reid_box_mask, reid_area = self.get_data(img_path, labels)
        ret = self.gen_target(imgs, copy.deepcopy(labels), reid_box_mask=reid_box_mask, dataset_name=ds)

        if self.opt.frame_pre != 'none':
            if random.random() < self.opt.negative_pre:
                negative = True # negative adjacent frame
            else:
                negative = False # positive adjacent frame         

            # for a dataset that only has one video, all frames should be treated as positive
            if len(self.annotations[ds]['videos'].keys()) == 1:
                negative = False
            if self.opt.reid_loss == 'cycle_loss':
                negative = False

            if negative:
                videos = list(self.annotations[ds]['videos'].keys())
                if len(videos) > 0: # video based dataset
                    img_info = self.annotations[ds]['images'][img_name]
                    video_name = img_info['video_name']
                    video_name_pre = random.choice(videos)
                    while video_name == video_name_pre:
                        video_name_pre = random.choice(videos)
                    img_name_pre = random.choice(self.annotations[ds]['videos'][video_name_pre])
                else: # static images based dataset
                    img_name_pre = random.choice(self.img_files[ds])
                    while img_name == img_name_pre:
                        img_name_pre = random.choice(self.img_files[ds])
            else:
                img_name_pre = img_name # get the adjacent frames by augmenting itself
                if self.opt.frame_pre == 'adjacent' and \
                    img_name in self.annotations[ds]['adjacent_frames'] and \
                    len(self.annotations[ds]['adjacent_frames'][img_name]) > 0: # get the adjacent frame by sampling
                    while img_name == img_name_pre:
                        img_name_pre = random.choice(self.annotations[ds]['adjacent_frames'][img_name])

            labels_pre = copy.deepcopy(self.annotations[ds]['annotations'][img_name_pre])
            img_path_pre = os.path.join(self.root, img_name_pre)
            imgs_pre, labels_pre, _, _, _, reid_box_mask_pre, _ = self.get_data(img_path_pre, labels_pre, aug_corpus=None if negative else warp_coprus, reid_area=reid_area)
            ret_pre = self.gen_target(imgs_pre, copy.deepcopy(labels_pre), reid_box_mask=reid_box_mask_pre, dataset_name=ds, pre=True, pre_for_det=self.opt.frame_pre_trian_det)
            ret_pre['negative'] = int(negative)
            ret['pre_data'] = ret_pre
        
        if self.opt.debug:
            self.debugger.add_image_with_bbox(imgs, labels, img_id='img', bbox_type='jde_gt,xywh,norm', img_type='RGB')
            if self.opt.frame_pre != 'none':
                self.debugger.add_image_with_bbox(imgs_pre, labels_pre, img_id='img_pre', bbox_type='jde_gt,xywh,norm', img_type='RGB')
            self.debugger.show_all_imgs()
        return ret

