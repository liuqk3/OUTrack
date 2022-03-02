import os
import glob

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
        }
}


result_dir='/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_mot17_bs12_dla34_clsReid_lr1e-4/results/test_mot17_det0.4_reid0.4_momentum_e30'
file_list = glob.glob(os.path.join(result_dir, '*.txt'))
file_list = [f for f in file_list if '-'.join(os.path.basename(f).split('-')[:2]) in MOT_INFO['MOT17']['test']]

detecotrs = ['-DPM.', '-FRCNN.', '-SDP.']

for d in detecotrs:
    if d in os.path.basename(file_list[0]):
        cur_dector = d

for f in file_list:
    for d in detecotrs:
        if d == cur_dector:
            continue

        save_path = f.replace(cur_dector, d)
        os.system('cp {} {}'.format(f, save_path))






