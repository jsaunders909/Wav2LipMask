#python mask_train.py --data_root  ../data/lrs2_preprocessed/ --checkpoint_dir checkpoints
python hq_wav2lip_train.py --data_root  ../data/lrs2_preprocessed/ --checkpoint_dir checkpoints --syncnet_checkpoint_path checkpoints/lipsync_expert2.pth