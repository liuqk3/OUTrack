
sequences = {
    'MOT15': {
        'train': ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
                    'TUD-Stadtmitte'],
        'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
                    'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
        'val': []
    },
    'MOT16': {
        'train': ['MOT16-04', 'MOT16-11', 'MOT16-05', 'MOT16-13', 'MOT16-02'], #, 'MOT16-10', 'MOT16-09'],
        'test': ['MOT16-12', 'MOT16-03', 'MOT16-01', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'],
        'val': ['MOT16-09', 'MOT16-10']
    },
    'MOT17': {
        'train': ['MOT17-04', 'MOT17-11', 'MOT17-05', 'MOT17-13', 'MOT17-02'],
        'test': ['MOT17-03', 'MOT17-01', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
        'val': ['MOT17-10', 'MOT17-09']
    },
    'MOT20':{
        'train':['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
        'test': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
        'val': [],
    }
}

def get_seq_info(seq_name):
    if seq_name.startswith('MOT'):
        year = seq_name[:5]
        if year == 'MOT17':
            seq_name = seq_name[0:8]
    else:
        year = 'MOT15'
    
    phase = 'train' if seq_name in sequences[year]['train'] else 'test'

    return year, phase

