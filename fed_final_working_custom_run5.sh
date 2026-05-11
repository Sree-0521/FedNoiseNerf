#!/bin/bash

MAX_TRAIN_STEPS=75000
BATCH_SIZE=4096
CHUNK=20480

echo "Starting server"
python3 custom_server.py > run5_slog.txt &
sleep 3 # Sleep for 3s to give the server enough time to start
echo "started server, moving to clients"

for i in `seq 0 1`; do

    if [ $i -eq 0 ]; then
        echo "Starting client $i"
        python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1v4/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1v4/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1v4/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs \
                --gpu_id $i \
                --max_train_steps $MAX_TRAIN_STEPS \
                --batch_size $BATCH_SIZE \
                --chunk $CHUNK \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints > run5_log1.txt &
    else
        echo "Starting client $i"
        python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2v4/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2v4/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2v4/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs \
                --gpu_id $i \
                --max_train_steps $MAX_TRAIN_STEPS \
                --batch_size $BATCH_SIZE \
                --chunk $CHUNK \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints > run5_log2.txt &
    fi
done
wait