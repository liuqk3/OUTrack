import argparse
import numpy as np
import glob
import os
import json

def ArgParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='')
    args = parser.parse_args()
    # assert os.path.exists(args.result_dir)
    return args

def get_mot_result_statics(args):
    mot_seqs = {

        'train':
        ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2', 'ETH-Sunnyday', 
        'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte'
         
        'MOT16-04', 'MOT16-11', 'MOT16-05', 'MOT16-13', 'MOT16-02', 'MOT16-09', 'MOT16-10' 

        'MOT17-04-DPM', 'MOT17-11-DPM', 'MOT17-05-DPM', 'MOT17-13-DPM', 'MOT17-02-DPM', 'MOT17-10-DPM', 'MOT17-09-DPM',
        'MOT17-04-SDP', 'MOT17-11-SDP', 'MOT17-05-SDP', 'MOT17-13-SDP', 'MOT17-02-SDP', 'MOT17-10-SDP', 'MOT17-09-SDP',
        'MOT17-04-FRCNN', 'MOT17-11-FRCNN', 'MOT17-05-FRCNN', 'MOT17-13-FRCNN', 'MOT17-02-FRCNN', 'MOT17-10-FRCNN', 'MOT17-09-FRCNN',

        'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
        
        'test':
        ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 
        'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1',
         
        'MOT16-03', 'MOT16-01', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14',

        'MOT17-03-DPM', 'MOT17-01-DPM', 'MOT17-06-DPM', 'MOT17-07-DPM', 'MOT17-08-DPM', 'MOT17-12-DPM', 'MOT17-14-DPM',
        'MOT17-03-SDP', 'MOT17-01-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP', 'MOT17-14-SDP',
        'MOT17-03-FRCNN', 'MOT17-01-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN',

        'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
        }

    files = glob.glob(os.path.join(args.result_dir, '*.txt'))
    files = [f for f in files if os.path.basename(f).replace('.txt', '') in mot_seqs['test']]

    statics = {}
    overall_statics = {
        'total_box_count': 0,
        'object_count': 0,
    }
    for f in files:
        seq_name = os.path.basename(f).replace('.txt', '')
        statics_seq = {}
        result = np.loadtxt(f, delimiter=',')
        statics_seq['total_box_count'] = result.shape[0]
        overall_statics['total_box_count'] += result.shape[0]

        track_id = result[:, 1]
        track_id = np.unique(track_id)
        statics_seq['object_count'] = track_id.shape[0]
        overall_statics['object_count'] += track_id.shape[0]

        statics[seq_name] = statics_seq
    statics['overall'] = overall_statics

    save_path = os.path.join(args.result_dir, 'results_info.json')
    print('results info: \n', statics)
    json.dump(statics, open(save_path, 'w'), indent=4)
    print('statcis saved in {}'.format(save_path))


if __name__ == '__main__':
    args = ArgParse()
    args.result_dir = '/home/liuqk/Program/python/OUTrack/exp/mot/mot17_bs8_dla34_cycle2ReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_e2e_amp/results/test_mot17_det0.35_reid0.5_momentum_occOff0.1_lostFrame1.0_e30_public'
    get_mot_result_statics(args)



