#!/bin/bash

# run7: Same as run6 but with Gaussian noise added to training RGB pixels per client
# Noise is injected at dataset load time: rgb += N(0, noise_level), clipped to [0,1]
# Client 0: noise_level=0.10 (10%)
# Client 1: noise_level=0.20 (20%)
# Client 2: noise_level=0.30 (30%)
# Client 3: noise_level=0.00 (no noise — control)
MAX_TRAIN_STEPS=12000
BATCH_SIZE=4096
CHUNK=20480
PYTHON=/home/myid/sl02922/miniconda3/envs/fedsatnerf/bin/python3

echo "Starting server"
$PYTHON custom_server.py > run7_slog.txt &
sleep 3
echo "started server, moving to clients"

# Client 0 — GPU 0, 10% Gaussian noise
echo "Starting client 0 (noise=0.10)"
$PYTHON client.py --model sat-nerf \
        --exp_name JAX_068_ds1_sat-nerf \
        --root_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v4/client1v4/root_dir/crops_rpcs_ba_v2/JAX_068 \
        --img_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v4/client1v4/DFC2019/Track3-RGB-crops/JAX_068 \
        --cache_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
        --gt_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v4/client1v4/DFC2019/Track3-Truth \
        --logs_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/logs \
        --gpu_id 0 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --noise_level 0.10 \
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/checkpoints > run7_log1.txt &

# Client 1 — GPU 1, 20% Gaussian noise
echo "Starting client 1 (noise=0.20)"
$PYTHON client.py --model sat-nerf \
        --exp_name JAX_068_ds1_sat-nerf \
        --root_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v4/client2v4/root_dir/crops_rpcs_ba_v2/JAX_068 \
        --img_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v4/client2v4/DFC2019/Track3-RGB-crops/JAX_068 \
        --cache_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
        --gt_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v4/client2v4/DFC2019/Track3-Truth \
        --logs_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/logs \
        --gpu_id 1 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --noise_level 0.20 \
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/checkpoints > run7_log2.txt &

# Client 2 — GPU 0, 30% Gaussian noise
echo "Starting client 2 (noise=0.30)"
$PYTHON client.py --model sat-nerf \
        --exp_name JAX_068_ds1_sat-nerf \
        --root_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v3/client1v3/root_dir/crops_rpcs_ba_v2/JAX_068 \
        --img_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v3/client1v3/DFC2019/Track3-RGB-crops/JAX_068 \
        --cache_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/cache_v3/crops_rpcs_ba_v2/JAX_068_ds1 \
        --gt_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/client1v3/client1v3/DFC2019/Track3-Truth \
        --logs_dir /home/myid/sl02922/nerf/fedsatnerf/cl3/logs \
        --gpu_id 0 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --noise_level 0.30 \
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl3/checkpoints > run7_log3.txt &

# Client 3 — GPU 1, no noise (control)
echo "Starting client 3 (noise=0.00)"
$PYTHON client.py --model sat-nerf \
        --exp_name JAX_068_ds1_sat-nerf \
        --root_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v3/client2v3/root_dir/crops_rpcs_ba_v2/JAX_068 \
        --img_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v3/client2v3/DFC2019/Track3-RGB-crops/JAX_068 \
        --cache_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/cache_v3/crops_rpcs_ba_v2/JAX_068_ds1 \
        --gt_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/client2v3/client2v3/DFC2019/Track3-Truth \
        --logs_dir /home/myid/sl02922/nerf/fedsatnerf/cl4/logs \
        --gpu_id 1 \
        --max_train_steps $MAX_TRAIN_STEPS \
        --batch_size $BATCH_SIZE \
        --chunk $CHUNK \
        --noise_level 0.00 \
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl4/checkpoints > run7_log4.txt &

wait
