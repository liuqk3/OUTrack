import os
import numpy as np
import glob
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = os.path.abspath(os.path.join(this_dir, '..', 'lib'))
add_path(lib_path)



def refine_gt_file_based_on_visibility(mot_root, vis_thr=0.8):
    """
    Args:
        mot_root: path to MOT17 or MOT16 or MOT20
    """
    assert os.path.exists(mot_root)
    assert 'MOT15' not in mot_root, 'MOT15 has not visibility annotation!'
    seqs = [s for s in os.listdir(os.path.join(mot_root, 'images', 'train')) if not s.startswith('.')]
    seqs = [s for s in seqs if '-SDP' in s]
    save_dir = os.path.join(mot_root, 'gts')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for seq in seqs:
        gt_file = os.path.join(mot_root, 'images', 'train', seq, 'gt', 'gt.txt')
        save_path = os.path.join(save_dir, '{}_vis_{}.txt'.format(seq, vis_thr))
        gt = np.loadtxt(gt_file, delimiter=',')
        # fid, tid, x, y, w, h, mark, label, visbility
        gt_filter = gt.copy()
        index = gt_filter[:, 8] > vis_thr
        ignore = gt_filter[index]
        ignore[:, 7] = 2 # change the class label to an ignored class
        gt_filter[index] = ignore
        lines = []
        for line in gt_filter:
            line = line.tolist()
            line[0] = int(line[0])
            line[1] = int(line[1])
            line[6] = int(line[6])
            line[7] = int(line[7])
            line = [str(t) for t in line]
            line = ','.join(line)
            lines.append(line)  
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
            f.close()

def refine_gt_file_based_on_iou(mot_root, iou_thr):
    """
    Args:
        mot_root: path to MOT17 or MOT16 or MOT20
    """
    from utils.box import iou as IOU
    assert os.path.exists(mot_root)
    assert 'MOT15' not in mot_root, 'MOT15 has not visibility annotation!'
    seqs = [s for s in os.listdir(os.path.join(mot_root, 'images', 'train')) if not s.startswith('.')]
    seqs = [s for s in seqs if '-SDP' in s]
    save_dir = os.path.join(mot_root, 'gts')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for seq in seqs:
        gt_file = os.path.join(mot_root, 'images', 'train', seq, 'gt', 'gt.txt')
        save_path = os.path.join(save_dir, '{}_iou_{}.txt'.format(seq, iou_thr))
        gt = np.loadtxt(gt_file, delimiter=',')
        # fid, tid, x, y, w, h, mark, label, visbility

        gt_filter = gt.copy()
        # get the index of box

        frame_id = np.unique(gt_filter[:, 0])
        for fid in frame_id:
            index_frame = gt_filter[:, 0] == fid
            gt_frame = gt_filter[index_frame]

            index_person = gt_frame[:, 7] == 1
            gt_person = gt_frame[index_person]
            ltrb = gt_person[:, 2:6].copy()
            ltrb[:, 2:4] = ltrb[:, 2:4] + ltrb[:, 0:2] # N, 4

            iou = IOU(ltrb, iou_type=1) # N, N
            iou_max = np.max(iou, axis=1) # N
            index_ignore = iou_max <= iou_thr

            gt_ignore = gt_person[index_ignore]
            gt_ignore[:, 7] = 2
            gt_person[index_ignore] = gt_ignore

            gt_frame[index_person] = gt_person

            gt_filter[index_frame] = gt_frame

        lines = []
        for line in gt_filter:
            line = line.tolist()
            line[0] = int(line[0])
            line[1] = int(line[1])
            line[6] = int(line[6])
            line[7] = int(line[7])
            line = [str(t) for t in line]
            line = ','.join(line)
            lines.append(line)  
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
            f.close()



if __name__ == '__main__':
    mot_root = '/home/liuqk/Dataset/jde_fairmot/MOT17'
    refine_gt_file_based_on_visibility(mot_root=mot_root, vis_thr=0.5)
    # refine_gt_file_based_on_iou(mot_root=mot_root, iou_thr=0.2)