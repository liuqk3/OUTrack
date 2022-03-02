"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Xingyi Zhou
"""

import argparse
import glob
import os
import pandas as pd
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.
Files
-----
All file content, ground truth and test files, have to comply with the
format described in 
Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/
Structure
---------
Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...
Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...
Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--gt_type', type=str, default='')
    parser.add_argument('--eval_official', action='store_true')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    parser.add_argument('--txt_path', type=str, default='', help='save the results to provied txt file')
    parser.add_argument('--eval_info', type=str, default='', help='the information about this evaluation')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            acc = mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5)
            # acc, ana = mm.utils.CLEAR_MOT_M(gts[k], tsacc, gts[k], 'iou', distth=0.5)
            accs.append(acc)
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


if __name__ == '__main__':

    args = parse_args()
    
    # args.groundtruths='/home/liuqk/Dataset/jde_fairmot/MOT17/images/train'  #'/home/liuqk/Program/python/CenterTrack/data/mot17/train'
    # args.tests='/home/liuqk/Program/python/OUTrack/exp/mot/mot17_half_bs8_dla34_cycle2ReIDSup_1_Pmean_0.5M_occOff_lr1e-4_2_trainOcc/results/val_mot17_half_det0.4_reid0.4_momentum_occOff0.1_lostFrame1.0_e30_vis_0.7'
    # args.gt_type='_halfval'
    # args.eval_official = True

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(
        level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gt_type = args.gt_type
    print('gt_type', gt_type)
    gtfiles = glob.glob(os.path.join(args.groundtruths, '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(args.tests, '*.txt')) if not os.path.basename(f).startswith('eval') and 'opt' not in os.path.basename(f) and 'summary' not in os.path.basename(f)]

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logging.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters,
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']
        }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
                       'mostly_tracked', 'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    # save sumary
    save_name = os.path.join(args.tests, 'summary.xlsx')
    writer = pd.ExcelWriter(save_name)
    summary.to_excel(writer)
    writer.save()
    summary_str = mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names)
    print(summary_str)
    with open(os.path.join(args.tests, 'summary.txt'), 'w') as ftxt:
        ftxt.write(summary_str)
        ftxt.close()

    if args.eval_official:
        metrics = mm.metrics.motchallenge_metrics + ['num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        # save summary
        save_name = os.path.join(args.tests, 'summary_official.xlsx')
        writer = pd.ExcelWriter(save_name)
        summary.to_excel(writer)
        writer.save()
        
        summary_str = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(summary_str)
        with open(os.path.join(args.tests, 'summary_official.txt'), 'w') as ftxt:
            ftxt.write(summary_str)
            ftxt.close()
        
        if args.txt_path != '':
            with open(args.txt_path, 'a') as ftxt:
                eval_info = args.eval_info if args.eval_info != '' else args.tests
                ftxt.write('\n\n\n' + eval_info + '\n')
                ftxt.write(summary_str)
                ftxt.close()

        logging.info('Completed')
