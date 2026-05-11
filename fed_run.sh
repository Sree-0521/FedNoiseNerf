#!/bin/bash

echo "Starting server"
python3 server.py &
sleep 3 # Sleep for 3s to give the server enough time to start
echo "started server, moving to client"
# for i in `seq 0 1`; do
#     echo "Starting client $i"
#     python client.py &
# done

for i in `seq 0 1`; do
    echo "Starting client $i"
    python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/dataset/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/logs \
                --gpu_id $i \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/checkpoints &
done


# echo "Starting client $i"
# python3 client.py --model sat-nerf \
#                     --exp_name JAX_068_ds1_sat-nerf \
#                     --root_dir /home/myid/kg23166/adcs/satnerf/dataset/root_dir/crops_rpcs_ba_v2/JAX_068 \
#                     --img_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-RGB-crops/JAX_068 \
#                     --cache_dir /home/myid/kg23166/adcs/satnerf/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
#                     --gt_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-Truth \
#                     --logs_dir /home/myid/kg23166/adcs/satnerf/logs \
#                     --gpu_id 2 \
#                     --ckpts_dir /home/myid/kg23166/adcs/satnerf/checkpoints &


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait


# python3 main.py --model sat-nerf \
#                 --exp_name JAX_068_ds1_sat-nerf \
#                 --root_dir /home/myid/kg23166/adcs/satnerf/dataset/root_dir/crops_rpcs_ba_v2/JAX_068 \
#                 --img_dir /home/myid/kg23166/adcs/satnerf/dataset/DFC2019/Track3-RGB-crops/JAX_068 \
#                 --cache_dir /mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 \
#                 --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth \
#                 --logs_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/logs \
#                 --ckpts_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/ckpts