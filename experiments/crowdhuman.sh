cd src

# detection
export CUDA_VISIBLE_DEVICES=0,1 && python train.py --task mot --exp_id crowdhuman_bs8_dla34_lr1e-4 --debug 0 --dataset jde_json --gpus 0,1 --not_reid --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

# detection and FairMOT classification based reid
export CUDA_VISIBLE_DEVICES=6,7 && python train.py --task mot --exp_id crowdhuman_bs8_dla34_clsReID_lr1e-4 --debug 0 --dataset jde_json --gpus 0,1 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

# detection and occlusion
export CUDA_VISIBLE_DEVICES=4,5 && python train.py --task mot --exp_id crowdhuman_bs8_dla34_occOff_lr1e-4 --debug 0 --dataset jde_json --gpus 0,1 --not_reid --batch_size 8 --occlusion --occlusion_offset --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

# detection and Cycas reid
export CUDA_VISIBLE_DEVICES=6,7 && python train.py --task mot --exp_id crowdhuman_bs8_dla34_cycasReIDSup0_0.5M_lr1e-4 --debug 0 --dataset jde_json --gpus 0,1 --reid_loss cycas_loss --reid_cycle_loss_supervise 0  --reid_cycle_loss_margin 0.5 --batch_size 8 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'

# detection and ours unsupervised reid
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 && python train.py --task mot --exp_id crowdhuman_bs24_dla34_cycleReIDSup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.2 --dataset jde_json --gpus 0,1,2,3,4,5 --lr 3e-4 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 10 --negative_pre 0.2 --batch_size 24 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json' --save_epoch '10,20,30,40,50,51,52,53,54,55,56,57,58,59' 

# detection, ours unsupervised reid and occlusion
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5 && python train.py --task mot --exp_id crowdhuman_bs24_dla34_cycleReIDSup_1Sup_1W10_Pmean_0.5M_occOff_lr3e-4_PreNeg0.2_amp --dataset jde_json --gpus 0,1,2,3,4,5 --lr 3e-4 --reid_loss cycle_loss --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 10 --occlusion --occlusion_offset --negative_pre 0.2 --batch_size 24 --load_model '../exp/mot/crowdhuman_bs24_dla34_cycle2ReID1Sup_1W10_Pmean_0.5M_occOff_lr3e-4_PreNeg0.2_amp/model_last.pth' --num_epochs 60 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'  --save_epoch '10,20,30,40,50,51,52,53,54,55,56,57,58,59' --debug 0 --amp



# test, for example
export CUDA_VISIBLE_DEVICES=0 && python track.py --task mot --exp_id crowdhuman_bs24_dla34_cycleReIDSup_1Sup_1W10_Pmean_0.5M_occOff_lr3e-4_PreNeg0.2_amp --resume --conf_thres 0.4 --val_mot17 True --half_track True --debug 0 --reid_thres 0.5 --debug 0
# or
export CUDA_VISIBLE_DEVICES=1 && python track.py --task mot --exp_id crowdhuman_bs24_dla34_cycleReIDSup_1Sup_1W10_Pmean_0.5M_occOff_lr3e-4_PreNeg0.2_amp --load_model '/home/liuqk/Program/python/OUTrack/exp/mot/crowdhuman_bs24_dla34_cycleReIDSup_1Sup_1W10_Pmean_0.5M_occOff_lr3e-4_PreNeg0.2_amp/model_55.pth' --conf_thres 0.4 --val_mot17 True --half_track True --debug 0 --reid_thres 0.5 --debug 0

cd ..