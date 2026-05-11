#!/bin/bash

MAX_TRAIN_STEPS=75000
BATCH_SIZE=4096
CHUNK=20480

echo "Starting training"

python3 main.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/dataset/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/logs \
                --gpu_id 0 \
                --max_train_steps $MAX_TRAIN_STEPS \
                --batch_size $BATCH_SIZE \
                --chunk $CHUNK \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/checkpoints > nonfed_run.txt &
    