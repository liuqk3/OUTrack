cd src

export CUDA_VISIBLE_DEVICES=6,7 && python train.py --task mot --exp_id crowdhuman_mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_lr1e-4 --dataset jde_json --gpus 0,1 --reid_loss cycle_loss2 --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 1 --batch_size 8 --load_model '../exp/mot/crowdhuman_bs24_dla34_cycleReIDSup_1W10_Pmean_0.5M_lr3e-4_PreNeg0.2/model_last.pth' --data_cfg '../src/lib/cfg/mot17.json' --debug 0
export CUDA_VISIBLE_DEVICES=0,1 && python train.py --task mot --exp_id crowdhuman_mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_trainOcc --dataset jde_json --gpus 0,1 --reid_loss cycle_loss2 --reid_cycle_loss_supervise -1 --reid_cycle_loss_margin 0.5 --reid_cycle_loss_placeholder mean --id_weight 1 --occlusion --occlusion_offset --train_part occlusion --batch_size 8 --load_model '../exp/mot/crowdhuman_mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_lr1e-4/model_last.pth' --data_cfg '../src/lib/cfg/mot17.json' --debug 0



# test
export CUDA_VISIBLE_DEVICES=4 && python track.py --task mot --exp_id crowdhuman_mot17_bs8_dla34_cycle2ReIDSup_1W0.1_Pmean_0.5M_lr1e-4 --resume  --conf_thres 0.4 --test_mot17 True
export CUDA_VISIBLE_DEVICES=8 && python track.py --task mot --exp_id crowdhuman_mot17_bs8_dla34_cycleReIDSup_1W1_Pmean_0.5M_occOff_lr1e-4_trainOcc --resume  --conf_thres 0.4 --test_mot17 True --occlusion_offset --occlusion --reid_thres 0.5



cd ..