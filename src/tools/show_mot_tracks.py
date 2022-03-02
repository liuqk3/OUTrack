import os
import sys
cur_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_dir, '..')) # src/lib

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
# from lib.utils.visualization import plot_tracking

MOT_INFO = {
    '2DMOT2015': {
            'train': ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
                      'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte'],
            'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
                     'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
        },
    'MOT16': {
            'train': ['MOT16-04', 'MOT16-11', 'MOT16-05', 'MOT16-13', 'MOT16-02', 'MOT16-09', 'MOT16-10'],
            'test': ['MOT16-12', 'MOT16-03', 'MOT16-01', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'],
        },
    'MOT17': {
            'train': ['MOT17-04', 'MOT17-11', 'MOT17-05', 'MOT17-13', 'MOT17-02', 'MOT17-10', 'MOT17-09'],
            'test': ['MOT17-03', 'MOT17-01', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
        },
    'MOT20': {
            'train': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
            'test': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
        },
}

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = int(1/640 * image.shape[1]) # max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def show_tracks(file_list, im_root, save_video_path=None, step=1, pad=0, box_idx=0, fps=5, tool='plt', pause=False):
    """ show mot trrack results
    file_list: a list of box file for one video
    im_root, the images need to be shown
    step: int, one of step images will be shown 
    pad: the number of pixels to pad the imges
    save_video_path: if not None, it shoule be a path of .avi format video to save the tracking results
    box_idx: the index of box file the need to be saved in the video
    fps: the fps of saved video
    
    """
    row = 1
    col = max(1, len(file_list))

    start_frame = -1
    box_list = []
    for f in file_list:
        box = np.loadtxt(fname=f, delimiter=',')
        seq_name = os.path.basename(f).replace('.txt', '')
        if seq_name not in MOT_INFO['2DMOT2015']['test'] and seq_name not in MOT_INFO['2DMOT2015']['train']:
            if 'gt' in os.path.basename(f):
                # idx = ((box[:, 7] == 1) + (box[:, 7] == 2) + (box[:, 7] == 7)) * (box[:, 8] > 0.15)
                # idx = ((box[:, 7] == 1)) * (box[:, 8] > 0.05)
                idx = (box[:, 7] == 1)
                box = box[idx]
        
        box[:, 2:4] = box[:, 2:4] + pad

        track_id = []
        for one_box in box:
            if int(one_box[1]) not in track_id:
                track_id.append(int(one_box[1]))
        print('num of objects: {}'.format(len(track_id)))

        start_frame = max(start_frame, box[:, 0].min())
        box_list.append(box)
    assert os.path.exists(im_root)
    im_name = glob.glob(os.path.join(im_root, '*.jpg'))
    num_frames = len(im_name)
    frame_id = list(range(int(start_frame), len(im_name)+1, step))

    im = cv2.imread(im_name[0])
    scale = 0.6
    size = (int(im.shape[1]*scale), int(im.shape[0]*scale)) # [w, h]
    if save_video_path is not None:
        assert save_video_path.split('.')[-1] in ['mp4', 'avi']
        video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        print('Video will be saved to {}'.format(save_video_path))
    else:
        video = None

    for fid in frame_id:
        if fid < start_frame:
            continue
        im_name = os.path.join(im_root, str(fid).zfill(6)+'.jpg')
        im = cv2.imread(im_name)
        if pad != 0:
            im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        for idx in range(len(box_list)):
            box = box_list[idx]
            frame_box = box[box[:, 0] == fid]

            tlwhs = frame_box[:, 2:6]
            ids = frame_box[:, 1]
            im_tmp = plot_tracking(image=im.copy(), tlwhs=tlwhs, obj_ids=ids, frame_id=fid)
            im_tmp = cv2.resize(im_tmp, size, interpolation=cv2.INTER_AREA) 
            if video is not None and idx == box_idx:
                video.write(im_tmp)
            
            if tool == 'plt':
                im_tmp = im_tmp[:, :, ::-1]
                if idx == 0:
                    plt.clf()

                plt.subplot(row, col, idx+1)
                plt.imshow(im_tmp)
                plt.axis('off')
            else:
                cv2.imshow(str(idx), im_tmp)
        if tool == 'plt':
            plt.pause(0.1)
        else:
            while cv2.waitKey(int(not pause)) == 27:
                import sys
                sys.exit(0)
        print('{}/{}'.format(fid, num_frames))
    if video is not None:
        video.release()

if __name__ == '__main__':

    pause = True
    step = 2
    pad = 100
    tool = 'cv2'
    box1 = '/home/liuqk/Program/python/OUTrack/exp/mot/mot17_bs8_dla34_cycle2ReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_e2e_amp/results/test_mot16_det0.35_reid0.5_momentum_occOff0.1_lostFrame1.0_e30_public/MOT16-07.txt'
    box2 = '/home/liuqk/Program/pycharm/OUTrack/exp/mot/FairMOT_results/MOT20/MOT20-08.txt'
    mot_root = '/home/liuqk/Dataset/MOT/'
    if 'MOT16' in box1:
        year = 'MOT16'
    elif 'MOT17' in box1:
        year = 'MOT17'
    elif 'MOT20' in box1:
        year = 'MOT20'
    else:
        year = '2DMOT2015'

    base_name = os.path.basename(box1)
    if 'gt' in os.path.basename(box1) or 'det' in os.path.basename(box1):
        seq_name = box1.split('/')[-3]
    else:
        seq_name = os.path.basename(box1).split('.')[0]
    

    if year == '2DMOT2015':
        if seq_name in MOT_INFO[year]['test']:
            phase = 'test'
        else:
            phase = 'train'
    else:
        if seq_name[:8] in MOT_INFO[year]['test']:
            phase = 'test'
        else:
            phase = 'train'

    im_root = os.path.join(mot_root, year, phase, seq_name, 'img1')
    save_video_path = None # box1.replace('.txt', '.avi')

    show_tracks(file_list=[box1], im_root=im_root, save_video_path=save_video_path, pad=pad, step=step, fps=10, tool=tool, pause=pause)

