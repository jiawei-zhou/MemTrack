
# training
nohup python tracking/train.py --script histrack --config histrack_384_full --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1 >MemTrack_384_full.log 2>&1 &

nohup python tracking/train.py --script histrack_seq --config histrack_seq_384_full --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1 >MemTrackSeq_384_full.log 2>&1 &

