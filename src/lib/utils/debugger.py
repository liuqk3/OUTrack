from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import math
import cv2
import os
import sys
import torch
import copy


class Debugger(object):
    def __init__(self, opt, num_categories=81, theme='white'):
        self.opt = opt
        self.imgs = {}
        self.theme = theme

        self.num_categories = num_categories
        self._prepare_category_colors(self.num_categories)

        self.track_color = {}

    def reset(self):
        self.imgs = {}

    def set_up(self, category_names):
        self.category_names = category_names
        self._prepare_category_colors(len(self.category_names))

    def _prepare_category_colors(self, count):
        colors = [(color_list[i]).astype(np.uint8) for i in range(count)]
        while len(colors) < count:
            colors = colors + colors[:min(len(colors), count - len(colors))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)

    def preprocess_image(self, img, img_type='BGR'):
        if isinstance(img, torch.Tensor):
            img = img.to(torch.device("cpu")).numpy()
            img = img * 255
        assert len(img.shape) == 3
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
        img = np.asarray(img, dtype=np.uint8)
        if img_type == 'RGB':
            img = img[:, :, ::-1]
        return img

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        
        pad = self.opt.show_border
        if pad != 0:
            img = cv2.copyMakeBorder(img.copy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT)

        self.imgs[img_id] = img.copy()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.8):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    def flow2hsv(self, flow, tool_type='cv2'):
        """
        This function convert the frame extracted from compressed video to rgb image.
        :param flow: 3D array, [2, H, W]
        :param tool_type: string, 'plt' or 'cv2', the tools used to show
        :return: RGB image
        """
        flow = flow.transpose(1, 2, 0)

        s = np.shape(flow)
        new_s = (s[0], s[1], 3)
        hsv = np.zeros(new_s)  # make a hsv motion vector map

        hsv[:, :, 1] = 255

        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv[:, :, 0] = ang * 180 / np.pi / 2  # direction
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude

        hsv = hsv.astype(np.uint8)  # change to uint8

        if tool_type == 'plt':
            hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # for plt
        elif tool_type == 'cv2':
            hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # for cv2

        return hsv

    def gen_colormap(self, img):
        """
        img: the predicted heatmap. 3D tensor or array
        """
        if isinstance(img, torch.Tensor):
            img = img.to(torch.device('cpu')).numpy()
        img = img.copy()
        # ignore region
        # import pdb; pdb.set_trace()
        img[img == 1] = 0.5
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        #TODO
        colors = np.array(self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        # colors = np.array([0, 0, 255], dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3) # red
        if self.theme == 'white':
            colors = 255 - colors
        if self.opt.show_tango_color:
            colors = tango_color_dark[:c].reshape(1, 1, c, 3)
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        return color_map

    def _get_rand_color(self):
        c = ((np.random.random((3)) * 0.6 + 0.2) * 255).astype(np.int32).tolist()
        return c

    def add_arrow(self, st, ed, img_id, c=(255, 0, 255)):
        cv2.arrowedLine(
            self.imgs[img_id], (int(st[0]), int(st[1])),
            (int(ed[0] + st[0]), int(ed[1] + st[1])), c, 2,
            line_type=cv2.LINE_AA, tipLength=0.3)

    def add_text(self, text, img_id, coord=(5, 5),  c=(255, 0, 255)):
        pad = self.opt.show_border
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_size = 1
        cat_size = cv2.getTextSize(text, font, font_size, thickness)[0]
        cv2.putText(self.imgs[img_id], text, (coord[0]+pad, coord[1]+pad+cat_size[1]+thickness), 
                    font, font_size, c, thickness=thickness, lineType=cv2.LINE_AA)


    def add_bbox(self, bbox, cat, conf=1, track_id=None, show_txt=True, img_id='default'):
        pad = self.opt.show_border
        bbox = np.array(bbox, dtype=np.int32) + pad

        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        if self.opt.show_tango_color:
            c = (255 - tango_color_dark[cat][0][0]).tolist()
        
        if hasattr(self, 'category_names'):
            cat = self.category_names[cat][:4]
        txt = '{}:{:.1f}'.format(cat, conf) if conf is not None else '{}'.format(cat)
        thickness = 3 #2
        fontsize = 1 # 0.5
        if track_id is not None:
            track_id = int(track_id)
            if not (track_id % 1000 in self.track_color):
                self.track_color[track_id % 1000] = self._get_rand_color()
            c = self.track_color[track_id % 1000]
            txt = '{}:{}'.format(cat, track_id) if hasattr(self, 'category_names') else '{}'.format(track_id)
        cv2.rectangle(self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, thickness)

        if show_txt:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
            cv2.rectangle(self.imgs[img_id],
                            (bbox[0], bbox[1] - cat_size[1] - thickness),
                            (bbox[0] + cat_size[0], bbox[1]), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - thickness - 1),
                        font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
 

    def add_image_with_heatmap(self, img, heatmap, img_id='default'):
        img = self.preprocess_image(img)
        hm = self.gen_colormap(heatmap)
        self.add_blend_img(back=img, fore=hm, img_id=img_id)

    def add_image_with_bbox(self, img, bbox, img_id='default', bbox_type='', img_type='BGR'):
        img = self.preprocess_image(img=img, img_type=img_type)
        im_h, im_w = img.shape[:2]
        self.add_img(img, img_id)
        
        if len(bbox) == 0:
            return

        for box in bbox:
            if isinstance(box, dict):
                ltrb = box['bbox']
                label = int(box['label']) if 'label' in box else 0
                score = box['score'] if 'score' in box else None
                object_id = int(box['object_id']) if 'object_id' in box else None
            elif isinstance(box, np.ndarray): # For jde labels
                if 'jde_gt' in bbox_type:
                    ltrb = box[2:6]
                    label = int(box[0])
                    score = None
                    object_id = int(box[1])
                elif 'jde_det' in bbox_type:
                    ltrb = box[0:4]
                    score = box[4]
                    label = 1
                    object_id = None
                else:
                    raise ValueError('Unknown type of box')
            else:
                raise ValueError('Unknown of box type')
                
            if 'xywh' in bbox_type:
                ltrb0 = copy.deepcopy(ltrb)
                ltrb[0] = ltrb0[0] - ltrb0[2]/2
                ltrb[1] = ltrb0[1] - ltrb0[3]/2
                ltrb[2] = ltrb0[0] + ltrb0[2]/2
                ltrb[3] = ltrb0[1] + ltrb0[3]/2
            if 'norm' in bbox_type:
                ltrb = [ltrb[0]*im_w, ltrb[1]*im_h, ltrb[2]*im_w, ltrb[3]*im_h]

            self.add_bbox(ltrb, label, score, object_id, img_id=img_id)


    def show_all_imgs(self, Time=0, sup_title=None):
        def _resize_image(image):
            from .image import get_size
            height, width = image.shape[0], image.shape[1]
            if max(height, width) > self.opt.show_max_size:
                min_size = min(height, width, 0.5 * self.opt.show_max_size)
                height_s, width_s = get_size(height, width, min_size, self.opt.show_max_size)
                image = cv2.resize(image, (int(width_s), int(height_s)))
            return image

        if 1:
            for i, v in self.imgs.items():
                if i in list(self.opt.show_images) or len(self.opt.show_images) == 0:
                    imshow = _resize_image(v)
                    im_title = '{}'.format(i)
                    cv2.imshow(im_title, imshow)
            if cv2.waitKey(0 if self.opt.pause else 1) == 27: # press esc to quit
                # cv2.destroyAllWindows()
                sys.exit(0)
        else:
            keys = [k for k in self.imgs.keys()]
            if len(self.opt.show_images):
                keys = [k for k in keys if k in list(self.opt.show_images)]
            nImgs = len(keys)
            nCols = math.ceil(math.sqrt(nImgs))
            nRows = math.ceil(nImgs/nCols)
            assert nCols * nRows >= nImgs
            # fig = plt.figure(figsize=(nCols * 10, nRows * 10))
            plt.cla()
            # for i, (k, v) in enumerate(self.imgs.items()):
            for i in range(len(keys)):
                k = keys[i]
                v = self.imgs[k]
                plt.subplot(nRows, nCols, i+1)
                if len(v.shape) == 3:
                    plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(v)
                plt.title(k)
                plt.axis('off')
            # plt.show()
            if sup_title is not None:
                plt.suptitle(sup_title)
            if not self.opt.pause: # 
                plt.pause(1e-16)
            else:
                plt.show()

    def show_heatmap_xyz(self, heatmap, image=None):
        """
        heatmap: 2D arraylike
        """

        h, w = heatmap.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        fig = plt.figure()
        ax = Axes3D(fig)
        Y = np.arange(0, heatmap.shape[0], 1)
        X = np.arange(0, heatmap.shape[1], 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, heatmap, cmap=plt.get_cmap('rainbow'))
        ax.contourf(X, Y, heatmap, zdir='z',offset=-0.8, cmap='rainbow')
        
        if image is not None:
            fig = plt.figure()
            plt.imshow(image)
        plt.show()

    def save_all_imgs(self, prefix='', save_dir=None):
        if save_dir is None:
            path = os.path.join(self.opt.debug_dir, "debug")
        else:
            path = save_dir
        for i, v in self.imgs.items():
            if i in list(self.opt.save_images) or len(self.opt.save_images) == 0:
                i_path = os.path.join(path, i)
                if not os.path.exists(i_path):
                    os.makedirs(i_path)
                basename = '{}.png'.format(prefix)

                cv2.imwrite(os.path.join(i_path, basename), v)



color_list = np.array(
    [0.850, 0.325, 0.098,
     1.000, 1.000, 1.000, 
     0.929, 0.694, 0.125,
     0.494, 0.184, 0.556,
     0.466, 0.674, 0.188,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500,
     0.333, 0.000, 0.500,
     0.333, 0.333, 0.500,
     0.333, 0.667, 0.500,
     0.333, 1.000, 0.500,
     0.667, 0.000, 0.500,
     0.667, 0.333, 0.500,
     0.667, 0.667, 0.500,
     0.667, 1.000, 0.500,
     1.000, 0.000, 0.500,
     1.000, 0.333, 0.500,
     1.000, 0.667, 0.500,
     1.000, 1.000, 0.500,
     0.000, 0.333, 1.000,
     0.000, 0.667, 1.000,
     0.000, 1.000, 1.000,
     0.333, 0.000, 1.000,
     0.333, 0.333, 1.000,
     0.333, 0.667, 1.000,
     0.333, 1.000, 1.000,
     0.667, 0.000, 1.000,
     0.667, 0.333, 1.000,
     0.667, 0.667, 1.000,
     0.667, 1.000, 1.000,
     1.000, 0.000, 1.000,
     1.000, 0.333, 1.000,
     1.000, 0.667, 1.000,
     0.167, 0.000, 0.000,
     0.333, 0.000, 0.000,
     0.500, 0.000, 0.000,
     0.667, 0.000, 0.000,
     0.833, 0.000, 0.000,
     1.000, 0.000, 0.000,
     0.000, 0.167, 0.000,
     0.000, 0.333, 0.000,
     0.000, 0.500, 0.000,
     0.000, 0.667, 0.000,
     0.000, 0.833, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 0.000,
     0.000, 0.000, 0.167,
     0.000, 0.000, 0.333,
     0.000, 0.000, 0.500,
     0.000, 0.000, 0.667,
     0.000, 0.000, 0.833,
     0.000, 0.000, 1.000,
     0.333, 0.000, 0.500,
     0.143, 0.143, 0.143,
     0.286, 0.286, 0.286,
     0.429, 0.429, 0.429,
     0.571, 0.571, 0.571,
     0.714, 0.714, 0.714,
     0.857, 0.857, 0.857,
     0.000, 0.447, 0.741,
     0.50, 0.5, 0
     ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


tango_color = [[252, 233,  79],  # Butter 1
               [237, 212,   0],  # Butter 2
               [196, 160,   0],  # Butter 3
               [138, 226,  52],  # Chameleon 1
               [115, 210,  22],  # Chameleon 2
               [78, 154,   6],  # Chameleon 3
               [252, 175,  62],  # Orange 1
               [245, 121,   0],  # Orange 2
               [206,  92,   0],  # Orange 3
               [114, 159, 207],  # Sky Blue 1
               [52, 101, 164],  # Sky Blue 2
               [32,  74, 135],  # Sky Blue 3
               [173, 127, 168],  # Plum 1
               [117,  80, 123],  # Plum 2
               [92,  53, 102],  # Plum 3
               [233, 185, 110],  # Chocolate 1
               [193, 125,  17],  # Chocolate 2
               [143,  89,   2],  # Chocolate 3
               [239,  41,  41],  # Scarlet Red 1
               [204,   0,   0],  # Scarlet Red 2
               [164,   0,   0],  # Scarlet Red 3
               [238, 238, 236],  # Aluminium 1
               [211, 215, 207],  # Aluminium 2
               [186, 189, 182],  # Aluminium 3
               [136, 138, 133],  # Aluminium 4
               [85,  87,  83],  # Aluminium 5
               [46,  52,  54],  # Aluminium 6
               ]
tango_color = np.array(tango_color, np.uint8).reshape((-1, 1, 1, 3))


tango_color_dark = [
    [114, 159, 207],  # Sky Blue 1
    [196, 160,   0],  # Butter 3
    [78, 154,   6],  # Chameleon 3
    [206,  92,   0],  # Orange 3
    [164,   0,   0],  # Scarlet Red 3
    [32,  74, 135],  # Sky Blue 3
    [92,  53, 102],  # Plum 3
    [143,  89,   2],  # Chocolate 3
    [85,  87,  83],  # Aluminium 5
    [186, 189, 182],  # Aluminium 3
]

tango_color_dark = np.array(tango_color_dark, np.uint8).reshape((-1, 1, 1, 3))
