import os
import numpy as np
import json
import glob
import cv2
import argparse

CWD = os.path.dirname(__file__)
IMAGE_LIST_FILE_DATA = os.path.join(CWD, 'data')


def save_ann(data, save_path, name, split):
    print('==> {}: {}, {} images'.format(name, split, len(data['images'])))
    print('==> {}: saved in {}'.format(name, save_path))
    json.dump(data, open(save_path, 'w'))


def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def read_files(root, file_path):
    with open(file_path, 'r') as f:
        files = f.readlines()
        files = [x.strip() for x in files if len(x.strip()) > 0]
        files = [os.path.join(root, x) for x in files]
        return files

def convert_mot(args, year='17'):
    if not os.path.exists(os.path.join(args.root, 'MOT{}'.format(year))):
        print(' ==== MOT{} not found ===='.format(year))
        return

    assert year in ['15', '16', '17', '20']
    splits = ['train']
    for split in splits:
        data_path = os.path.join(args.root, 'MOT{}'.format(year), 'images', 'train' if split != 'test' else 'test')
        out_path = os.path.join(args.root, 'MOT{}'.format(year), 'annotations', '{}.json'.format(split))
        mkdirs(os.path.dirname(out_path))

        out = {
            'images': {}, 
            'annotations':{}
        }
        
        if year in ['16', '17', '20']:
            seqs = [s for s in os.listdir(data_path)]
            tid_curr = 0
            tid_last = -1
            for seq in seqs:
                if year == '17':
                    if '-SDP' not in seq:
                        continue

                video_name = os.path.join('MOT{}'.format(year), 'images', 'train' if split != 'test' else 'test', seq)
                seq_info = open(os.path.join(data_path, seq, 'seqinfo.ini')).read()
                seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
                seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

                gt_txt = os.path.join(data_path, seq, 'gt', 'gt.txt')
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

                for fid, tid, x, y, w, h, mark, label, visbility in gt:
                    frame_name = os.path.join('MOT{}'.format(year), 'images', 'train' if split != 'test' else 'test', seq, 'img1', '{:06d}.jpg'.format(int(fid)))
                    if frame_name not in out['images']:
                        frame_info = {
                            'name': frame_name,
                            'video_name': video_name,
                            'frame_id': fid,
                            'has_object_id': 1,
                        }
                        out['images'][frame_name] = frame_info

                    if mark == 0 or not label == 1:
                        continue
                    fid = int(fid)
                    tid = int(tid)
                    if not tid == tid_last:
                        tid_curr += 1
                        tid_last = tid
                    x += w / 2
                    y += h / 2

                    ann = {
                        'label': 0,
                        'object_id': tid_curr,
                        'xywh_norm': [round(x / seq_width, 6), round(y / seq_height, 6), round(w / seq_width, 6), round(h / seq_height, 6)],
                        'visibility': visbility,
                    }

                    if frame_name not in out['annotations']:
                        out['annotations'][frame_name] = []
                    out['annotations'][frame_name].append(ann)

        elif year == '15':
            seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-17',
                    'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']

            tid_curr = 0
            tid_last = -1
            for seq in seqs:
                video_name = os.path.join('MOT{}'.format(year), 'images', 'train' if split != 'test' else 'test', seq)
                seq_info = open(os.path.join(data_path, seq, 'seqinfo.ini')).read()
                seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
                seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

                gt_txt = os.path.join(data_path, seq, 'gt', 'gt.txt')
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
                idx = np.lexsort(gt.T[:2, :])
                gt = gt[idx, :]

                for fid, tid, x, y, w, h, mark, _, _, _ in gt:
                    frame_name = os.path.join('MOT{}'.format(year), 'images', 'train' if split != 'test' else 'test', seq, 'img1', '{:06d}.jpg'.format(int(fid)))
                    if frame_name not in out['images']:
                        frame_info = {
                            'name': frame_name,
                            'video_name': video_name,
                            'frame_id': fid,
                            'has_object_id': 1,
                        }
                        out['images'][frame_name] = frame_info
                    
                    if mark == 0:
                        continue
                    fid = int(fid)
                    tid = int(tid)
                    if not tid == tid_last:
                        tid_curr += 1
                        tid_last = tid
                    x += w / 2
                    y += h / 2
                    
                    ann = {
                        'label': 0,
                        'object_id': tid_curr,
                        'xywh_norm': [x / seq_width, y / seq_height, w / seq_width, h / seq_height],
                    }
                    # ann = [0, tid_curr, round(x / seq_width, 6), round(y / seq_height, 6), round(w / seq_width, 6), round(h / seq_height, 6)]
                    if frame_name not in out['annotations']:
                        out['annotations'][frame_name] = []
                    out['annotations'][frame_name].append(ann)
                    
        save_ann(out, out_path, 'MOT{}'.format(year), split)


def convert_crowdhuman(args):
    if not os.path.exists(os.path.join(args.root, 'crowdhuman')):
        print(' ==== Caltech not found ====')
        return
    def load_func(fpath):
        print('fpath', fpath)
        assert os.path.exists(fpath), fpath
        with open(fpath, 'r') as fid:
            lines = fid.readlines()
        records =[json.loads(line.strip('\n')) for line in lines]
        return records

    splits = ['train', 'val']

    for split in splits:
        data_path = os.path.join(args.root, 'crowdhuman', 'images', split)
        ann_file = os.path.join(args.root, 'crowdhuman', 'annotation_{}.odgt'.format(split))
        out_path = os.path.join(args.root, 'crowdhuman', 'annotations', '{}.json'.format(split))
        mkdirs(os.path.dirname(out_path))
        out = {
            'images': {}, 
            'annotations':{},
        }
        anns_data = load_func(ann_file)

        tid_curr = 0
        for i, ann_data in enumerate(anns_data):
            if (i%2000) == 0:
                print('crowdhuamn: {}'.format(i))
            image_name = '{}.jpg'.format(ann_data['ID'])
            img_path = os.path.join(data_path, image_name)
            img_name = os.path.join('crowdhuman', 'images', split, image_name)

            im_info = {
                'name': img_name,
                'frame_id': 1,
                'has_object_id': True,
            }
            out['images'][img_name] = im_info
            out['annotations'][img_name] = []

            anns = ann_data['gtboxes']
            img = cv2.imread(
                img_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            img_height, img_width = img.shape[0:2]
            for i in range(len(anns)):
                if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and anns[i]['extra']['ignore'] == 1:
                    continue
                x, y, w, h = anns[i]['fbox']
                x += w / 2
                y += h / 2

                tid_curr += 1

                # ann = [0, tid_curr, round(x / img_width, 6), round(y / img_height, 6), round(w / img_width, 6), round(h / img_height, 6)]
                ann = {
                    'label': 0,
                    'object_id': tid_curr,
                    'xywh_norm': [round(x / img_width, 6), round(y / img_height, 6), round(w / img_width, 6), round(h / img_height, 6)],
                }
                out['annotations'][img_name].append(ann)
        save_ann(out, out_path, 'CrowdHuman', split)


def ArgParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))
    if args.root == '':
        args.root = os.path.join(args.cwd, '../../data')
    return args

if __name__ == '__main__':
    args = ArgParse()
    # convert_mot(args, year='15')
    convert_mot(args, year='16')
    convert_mot(args, year='17')
    convert_mot(args, year='20')
    convert_crowdhuman(args)