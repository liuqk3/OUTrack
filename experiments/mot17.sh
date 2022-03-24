cd src
# train
# only reid
export CUDA_VISIBLE_DEVICES=0,1 && python train.py --task mot --exp_id mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_lr1e-4 --dataset jde_json --gpus 0,1 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 1 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth'  --data_cfg '../src/lib/cfg/mot17.json' --debug 0

# reid and occlusion
export CUDA_VISIBLE_DEVICES=2,3 && python train.py --task mot --exp_id mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_e2e_amp --dataset jde_json --gpus 0,1 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 1 --occlusion --occlusion_offset --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth'  --data_cfg '../src/lib/cfg/mot17.json' --amp --debug 0

# reid, then train occlusion with reid fixed
export CUDA_VISIBLE_DEVICES=7,9 && python train.py --task mot --exp_id mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_trainOcc --dataset jde_json --gpus 0,1 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 1 --occlusion --occlusion_offset --train_part occlusion --batch_size 8 --load_model '/home/liuqk/Program/python/OUTrack/exp/mot/mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_lr1e-4/model_last.pth' --data_cfg '../src/lib/cfg/mot17.json' --debug 0

# test

export CUDA_VISIBLE_DEVICES=4 && python track.py --task mot --exp_id mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_e2e_amp --resume  --conf_thres 0.4 --test_mot17 True --occlusion --occlusion_offset --track_type public_track --det_dir /home/liuqk/Program/python/OUTrack/data/MOT17/images/test --reid_thres 0.5

export CUDA_VISIBLE_DEVICES=7 && python track.py --task mot --exp_id mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_e2e_amp --resume  --conf_thres 0.4 --test_mot16 True --occlusion --occlusion_offset --track_type public_track --det_dir /home/liuqk/Program/python/OUTrack/data/MOT16/images/test --reid_thres 0.5 --debug 0


cd ..