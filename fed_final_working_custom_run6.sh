#!/bin/bash

# run5 took 20 hrs with 75000 steps (2 clients, 5 rounds, fraction_fit=0.5 → 1 client/round)
# Per round = 20/5 = 4 hrs for 75000 steps
# Target 3 hrs total: 3/5 rounds = 0.6 hrs/round → 0.6/4 * 75000 ≈ 11250 steps
# Using 12000 steps for ~3 hrs total. With 4 clients (fraction_fit=0.5 → 2/round on 2 GPUs in parallel)
MAX_TRAIN_STEPS=12000
BATCH_SIZE=4096
CHUNK=20480
PYTHON=/home/myid/sl02922/miniconda3/envs/fedsatnerf/bin/python3

echo "Starting server"
$PYTHON custom_server.py > run6_slog.txt &
sleep 3 # Sleep for 3s to give the server enough time to start
echo "started server, moving to clients"

# NOTE: GPU 0 is in Exclusive_Process mode. While another job occupies it (check
# with nvidia-smi), all clients use --gpu_id 1 (Default mode, allows sharing).
# client.py sets os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) before any
# CUDA init, so each process sees only one GPU (as device 0 internally).
# The file lock /tmp/gpu_1.lock in client.py serializes training on GPU 1.
#
# When GPU 0 is free, restore clients 0 & 2 to --gpu_id 0.

# Client 0 — GPU 0, cl1/client1v4 data (10 images: 002,003,004,005,007,009,012,016,018,022)
echo "Starting client 0"
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
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl1/checkpoints > run6_log1.txt &

# Client 1 — GPU 1, cl2/client2v4 data (11 images: 001,002,006,010,011,...)
echo "Starting client 1"
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
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl2/checkpoints > run6_log2.txt &

# Client 2 — GPU 0, cl1/client1v3 data (10 images: 001,002,006,007,012,013,016,018,019,020)
echo "Starting client 2"
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
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl3/checkpoints > run6_log3.txt &

# Client 3 — GPU 1, cl2/client2v3 data (11 images: 002,003,004,005,009,...)
echo "Starting client 3"
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
        --ckpts_dir /home/myid/sl02922/nerf/fedsatnerf/cl4/checkpoints > run6_log4.txt &

wait
