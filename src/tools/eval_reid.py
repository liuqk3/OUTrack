from __future__ import print_function, absolute_import
import numpy as np
import glob
import os
import json
import time
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    # assert os.path.exists(args.result_dir)
    return args


def save_json(in_dict, save_path):
    json_str = json.dumps(in_dict, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)
    json_file.close()


def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
        # ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    res = {
        'CMC': CMC.tolist(),
        'mAP': mAP, 
    }

    return res


def prepare_data(data_dir, visibility=0.0, query_ratio=0.1):
    files = glob.glob(os.path.join(data_dir, 'MOT*.txt'))
    files = sorted(files)

    start_person_id = 0
    start_camera_id = 1

    q_pids = []
    q_cids = []
    g_pids = []
    g_cids = []
    q_feats = []
    g_feats = []

    for f in files:
        data_tmp = np.loadtxt(f, delimiter=',')
        keep = data_tmp[:,8] >= visibility
        data_tmp = data_tmp[keep]
        data_tmp[:, 1] += start_person_id
        start_person_id = data_tmp[:, 1].max() + 1

        # get camera ids
        # we set each data an unique camera id, so it will not be filtered
        cid_tmp = np.arange(data_tmp.shape[0]) + start_camera_id
        start_camera_id = cid_tmp.max() + 1
        
        unique_pids = np.unique(data_tmp[:, 1])
        for i in unique_pids:
            idx = data_tmp[:, 1] == i 
            one_cid_tmp = cid_tmp[idx]
            one_pid_tmp = data_tmp[idx][:, 1]
            one_feat_tmp = data_tmp[idx][:, -128:]
            
            num_query = int(one_cid_tmp.shape[0]*query_ratio)
            q_pids.append(one_pid_tmp[:num_query])
            g_pids.append(one_pid_tmp[num_query:])

            q_cids.append(one_cid_tmp[:num_query])
            g_cids.append(one_cid_tmp[num_query:])

            q_feats.append(one_feat_tmp[:num_query])
            g_feats.append(one_feat_tmp[num_query:])
    
    q_pids = np.concatenate(q_pids, axis=0)
    g_pids = np.concatenate(g_pids, axis=0)
    q_cids = np.concatenate(q_cids, axis=0)
    g_cids = np.concatenate(g_cids, axis=0)
    q_feats = np.concatenate(q_feats, axis=0)
    g_feats = np.concatenate(g_feats, axis=0)
    
    return q_pids, g_pids, q_cids, g_cids, q_feats, g_feats

def get_distance(feat1, feat2, metrics='cosine'):
    import torch 
    feat1 = torch.Tensor(feat1)
    feat2 = torch.Tensor(feat2)
    batch_size = 1024
    loops = (feat2.shape[0] + batch_size - 1) // batch_size
    if metrics == 'cosine':
        dist = []
        feat1 = feat1.cuda()
        feat1 = feat1 / torch.norm(feat1, dim=1, p=2, keepdim=True)
        for l in range(loops):
            start = l * batch_size
            end = min((l+1)*batch_size, feat2.shape[0])
            feat2_ = feat2[start:end].cuda()
            feat2_ = feat2_ / torch.norm(feat2_, dim=1, p=2, keepdim=True)
            dist_ = torch.einsum('md,nd->mn', feat1, feat2_)
            dist_ = 1 - dist_
            dist.append(dist_.to('cpu'))
        dist = torch.cat(dist, dim=1)
    elif metrics == 'euclidean':
        m = feat1.size(0)
        feat1 = feat1.cuda()
        dist = []
        for l in range(loops):
            start = l * batch_size
            end = min((l+1)*batch_size, feat2.shape[0])
            feat2_ = feat2[start:end].cuda()
            n = feat2_.shape[0]
            dist_ = (
                    torch.pow(feat1, 2).sum(dim=1, keepdim=True).expand(m, n)
                    + torch.pow(feat2_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            )
            dist_.addmm_(1, -2, feat1, feat2_.t())
            dist.append(dist_.to('cpu'))
        dist = torch.cat(dist, dim=1)
    else:
        raise NotImplementedError('Metric {} not implemented!'.format(metrics))
    
    dist = dist.numpy()
    return dist

def evaluate_dir(data_dir, metrics='cosine', visibility=0.0, query_ratio=0.1):
    q_pids, g_pids, q_cids, g_cids, q_feats, g_feats = prepare_data(data_dir, visibility=visibility, query_ratio=query_ratio)
    dist = get_distance(q_feats, g_feats, metrics=metrics)

    res = evaluate(dist, q_pids, g_pids, q_cids, g_cids)
    print(res['mAP'])

    save_path = os.path.join(data_dir, 'metrics_{}_vis_{}_qratio_{}.json'.format(metrics, visibility, query_ratio))
    print('Results saved to {}'.format(save_path))
    save_json(res, save_path)




if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)

    data_dir = [
        # '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs8_dla34_clsReID_lr1e-4/results_det/val_mot17_e60_provide',
        # '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs8_dla34_cycleReID1Sup0_0.5M_lr1e-4/results_det/val_mot17_e29_provide',
        '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs24_dla34_cycle2ReID1Sup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.1/results_det/val_mot17_e29_provide'
        # '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs24_dla34_cycle2ReID1Sup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.2/results_det/val_mot17_e60_provide'
        # '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs24_dla34_cycle2ReID1Sup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.5/results_det/val_mot17_e60_provide',
        # '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs24_dla34_cycle2ReID1Sup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.8/results_det/val_mot17_e60_provide'
    ]

    visibility = [0.7]
    query_ratio = [0.5]
    metrics = ['cosine']
    for dd in data_dir:
        for vis in visibility:
            for qr in query_ratio:
                for m in metrics:
                    start = time.time()
                    evaluate_dir(data_dir=dd, visibility=vis, query_ratio=qr, metrics=m)
                    end = time.time()
                    print('data_dir: {}, vis: {}, query_ratio: {}, metric: {}'.format(dd, vis, qr, m))
                    print('Consume {}s'.format(end-start))