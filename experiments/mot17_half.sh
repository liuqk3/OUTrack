cd src

# (1) reid
export CUDA_VISIBLE_DEVICES=5,6 && python train.py --task mot --exp_id mot17_half_bs12_dla34_clsReID_lr1e-4 --dataset jde_json --gpus 0,1 --batch_size 12 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --debug 0
export CUDA_VISIBLE_DEVICES=0,1 && python train.py --task mot --exp_id mot17_half_bs8_dla34_cycleReIDSup_1_Pmean_0.5M_lr1e-4 --dataset jde_json --gpus 0,1 --batch_size 8 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --debug 0

# test
# for example
export CUDA_VISIBLE_DEVICES=8 && python track.py --task mot --exp_id mot17_half_bs8_dla34_cycleReIDSup_1_Pzero_0.5M_lr1e-4 --resume --conf_thres 0.4 --val_mot17 True --half_track True 




# (2) occlusion + reid
export CUDA_VISIBLE_DEVICES=7,8 && python train.py --task mot --exp_id mot17_half_bs8_dla34_cycleReIDSup_1_Pmean_0.5M_occOffW0.5_lr1e-4 --dataset jde_json --gpus 0,1 --batch_size 8 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --occlusion --occlusion_offset --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --debug 0


# test
export CUDA_VISIBLE_DEVICES=4 && python track.py --task mot --exp_id mot17_half_bs8_dla34_cycleReIDSup_1_Pmean_0.5M_occOffW0.5_lr1e-4 --resume  --conf_thres 0.4 --val_mot17 True --half_track True --occlusion --occlusion_offset

cd ..