from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
import os.path as osp
import torch
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import json


# from tracker.multitracker import JDETracker
from tracker.occ_tracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
from utils.utils import write_opt
import datasets.dataset.jde_json as datasets
from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type, result_type='track'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,{vis},{ind}\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    
    results_info = {
        'total_box_count': 0,
        'object_count': 0, 
    }
    objects_ids = set([])

    with open(filename, 'w') as f:
        for res in results:
            track_id = int(res['object_id'])
            if result_type == 'track' and track_id < 0:
                continue
            frame_id = res['frame_id']
            if data_type == 'kitti': # start from 0
                frame_id -= 1
            x1, y1, w, h = res['tlwh']
            x2, y2 = x1 + w, y1 + h
            score = res['score']
            if result_type == 'track':
                ind = -1
            elif result_type == 'det':
                ind = res['index']
            else:
                raise ValueError(result_type)
            
            results_info['total_box_count'] += 1
            if track_id not in objects_ids:
                objects_ids.add(track_id)
            results_info['object_count'] = len(objects_ids)
            state = '{}_box_count'.format(res['state'])
            if state not in results_info:
                results_info[state] = 0
            results_info[state] += 1
            vis = res.get('visibility', -1)
            # import pdb; pdb.set_trace()
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score, ind=ind, vis=vis)
            if 'feat' in res:
                feat = str(res['feat'].tolist())[1:-1]
                feat = feat.replace(' ', '')
                if ' ' in feat:
                    import sys
                    sys.exit(0)
                line = line.replace('\n', ','+feat+'\n')
            f.write(line)
    
    keys = list(results_info.keys())
    for k in keys:
        if k in ['total_box_count', 'object_count']:
            continue
        k_new = k.replace('_count', '_ratio')
        results_info[k_new] = results_info[k] / max(results_info['total_box_count'], 1)

    logger.info('save results to {}'.format(filename))

    return results_info

def eval_seq(opt, tracker, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    # tracker = JDETracker(opt, frame_rate=frame_rate)
    tracker.reset(frame_rate=frame_rate)
    timer = Timer()
    results = []
    len_all = len(dataloader)
    if opt.half_track:
        start_frame = int(len_all / 2)
        frame_id = int(len_all / 2)
    else:
        start_frame = -1
        frame_id = 0
    # frame_id = int(len_all / 2)
    for i, (img_info, img, img0, img_origin, det_meta) in enumerate(dataloader):
        if start_frame > 0 and i < start_frame:
            continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        if 'frame_id' in img_info:
            assert frame_id + 1 == img_info['frame_id'], '{}, {}'.format(frame_id+1, img_info['frame_id'])
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        if opt.only_det:
            results_tmp = tracker.update(blob, img0, det_meta=det_meta, img_info=img_info)
            results_frame = []
            for ret in results_tmp:
                ret.update({'frame_id': frame_id + 1})
                results_frame.append(ret)
        else:
            results_tmp = tracker.update(blob, img0, det_meta=det_meta, img_info=img_info)
            results_frame = []
            for ret in results_tmp:
                # tlwh = ret['tlwh'] 
                # vertical = tlwh[2] / tlwh[3] > 1.6
                # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                #     ret.update({'frame_id': frame_id + 1})
                #     results_frame.append(ret)
                ret.update({'frame_id': frame_id + 1})
                results_frame.append(ret)
            timer.toc()
        # save results
        results.extend(results_frame)
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img_origin, results_frame, frame_id=frame_id, fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    results_info = write_results(result_filename, results, data_type, result_type='det' if opt.only_det else 'track')
    return frame_id, timer.average_time, timer.calls, results_info


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo', gt_type='',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    tracker = JDETracker(opt)
    if opt.only_det:
        result_root = os.path.join(opt.save_dir, 'results_det', '{}_e{}'.format(exp_name, tracker.epoch))
    else:
        if opt.track_type in ['public_track', 'private_track']: # tracking with private detection
            dir_list = ['{}_det{}'.format(exp_name, opt.conf_thres)]
            if opt.not_reid:
                dir_list.append('noid')
            else:
                dir_list.append('reid{}_{}'.format(opt.reid_thres, opt.reid_feat_type))
                if opt.gmm:
                    dir_list.append('gmm')
            if opt.occlusion:
                if opt.occlusion_offset:
                    dir_list.append('occOff{}'.format(opt.occlusion_thres))
                else:
                    dir_list.append('occ{}'.format(opt.occlusion_thres))
                dir_list.append('lostFrame{:.1f}'.format(opt.lost_frame_range))
                
            dir_list.append('e{}'.format(tracker.epoch))
            result_root = os.path.join(opt.save_dir, 'results', '_'.join(dir_list))
        else:
            dir_list = ['{}_det{}'.format(exp_name, opt.conf_thres)]
            dir_list.append('reid{}'.format(opt.reid_thres) if not opt.not_reid else 'noid')
            if opt.occlusion:
                dir_list.append('occ{}'.format(opt.occlusion_thres))
            dir_list.append('e{}'.format(tracker.epoch))
            result_root = os.path.join(opt.det_dir, opt.exp_id, '_'.join(dir_list))
        if gt_type != '':
            result_root = result_root + '_{}'.format(gt_type)
    if opt.track_type != 'private_track':
        result_root = result_root + '_{}'.format(opt.track_type.split('_')[0])

    print('==> Results will be saved in {}'.format(result_root))
    mkdir_if_missing(result_root)
    write_opt(opt, os.path.join(result_root, 'opt.txt'))
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    results_info = {}
    overall_results_info = {}
    for seq in seqs:
        # output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        output_dir = os.path.join(opt.save_dir, 'visilization', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        if opt.det_dir != '':
            if opt.det_dir.endswith('/'):
                opt.det_dir = opt.det_dir[:-1]
            if 'MOT' in opt.det_dir and (opt.det_dir.split('/')[-1] in ['train', 'test']): # load from motchallenge
                file_type = 'gt' if opt.det_dir.endswith('train') else 'det'
                det_path = os.path.join(opt.det_dir, seq, file_type, '{}.txt'.format(file_type))
            else:
                det_path = os.path.join(opt.det_dir, '{}.txt'.format(seq))
        else:
            det_path = None
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size, det_path=det_path, seq_name=seq)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc, res_info = eval_seq(opt, tracker, dataloader, data_type, result_filename,
                                        save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        # save results info
        results_info[seq] = res_info
        for k in res_info.keys():
            if not k.endswith('_count'):
                continue
            if k not in overall_results_info:
                overall_results_info[k] = 0
            overall_results_info[k] += res_info[k]
        
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type, gt_type=gt_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # save results info
    keys = list(overall_results_info.keys())
    for k in keys:
        if k in ['total_box_count', 'object_count']:
            continue
        k_new = k.replace('_count', '_ratio')
        overall_results_info[k_new] = overall_results_info[k] / max(overall_results_info['total_box_count'], 1)
    results_info['overall'] = overall_results_info
    results_info_out = os.path.join(result_root, 'results_info.json')
    json.dump(results_info, open(results_info_out, 'w'), indent=4)
    print('Results info saved in {}'.format(results_info_out))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    with open(os.path.join(result_root, 'summary_{}.txt'.format(exp_name)), 'wt') as file_w:
        file_w.write(strsummary)
        file_w.close()
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
        opt.test_track_name = 'val_mot16'
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/train')
        opt.test_track_name = 'train_mot16'
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        # seqs_str = '''MOT16-06 MOT16-07 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/test')
        opt.test_track_name = 'test_mot16'
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
        opt.test_track_name = 'test_mot15'
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        #seqs_str = '''MOT17-01-SDP
                      #MOT17-06-SDP
                      #MOT17-07-SDP
                      #MOT17-12-SDP
                      #'''
        #seqs_str = '''MOT17-07-SDP MOT17-08-SDP'''
        if opt.track_type == 'public_track':
            seqs_str = '''MOT17-01-SDP
                        MOT17-03-SDP
                        MOT17-06-SDP
                        MOT17-07-SDP
                        MOT17-08-SDP
                        MOT17-12-SDP
                        MOT17-14-SDP
                        MOT17-01-DPM
                        MOT17-03-DPM
                        MOT17-06-DPM
                        MOT17-07-DPM
                        MOT17-08-DPM
                        MOT17-12-DPM
                        MOT17-14-DPM
                        MOT17-01-FRCNN
                        MOT17-03-FRCNN
                        MOT17-06-FRCNN
                        MOT17-07-FRCNN
                        MOT17-08-FRCNN
                        MOT17-12-FRCNN
                        MOT17-14-FRCNN
                        '''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
        opt.test_track_name = 'test_mot17'
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        #seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
        opt.test_track_name = 'val_mot17'
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        #seqs_str = '''Venice-2'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
        opt.test_track_name = 'val_mot15'
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
        opt.test_track_name = 'val_mot20'
        opt.clip_box = True
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
        opt.test_track_name = 'test_mot20'
        opt.clip_box = True
    seqs = [seq.strip() for seq in seqs_str.split()]

    if opt.half_track:
        opt.test_track_name += '_half'



    # seqs = ['MOT17-02-SDP']

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.test_track_name, #'MOT17_val_jde_half_dla34_det',
         gt_type=opt.gt_type,
         show_image=False,
         save_images=False,
         save_videos=False)
