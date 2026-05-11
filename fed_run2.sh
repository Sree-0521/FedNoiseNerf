#!/bin/bash

echo "Starting server"
python3 server.py &
sleep 3 # Sleep for 3s to give the server enough time to start
echo "started server, moving to client"

echo "Starting client 1"
python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs \
                --gpu_id 0 \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints &

echo "Starting client 2"
python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs \
                --gpu_id 1 \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
#=======================================================================================================
# echo "Starting server"
# python3 server.py &
# sleep 3 # Sleep for 3s to give the server enough time to start
# echo "Started server, moving to clients"

# # Define an array of arguments for each client
# declare -a client_args=(
#     "--model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-Truth --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs --gpu_id 0 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints"
#     "--model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-Truth --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs --gpu_id 1 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints"
# )

# # Iterate over the array of client arguments and start the clients
# for i in "${!client_args[@]}"; do
#     echo "Starting client $i"
#     python3 client.py ${client_args[$i]} &
# done

# # Wait for all background processes to finish
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# wait
#=======================================================================================================


# echo "Starting server"
# python3 server.py &
# sleep 3 # Sleep for 3s to give the server enough time to start
# echo "Started server, moving to clients"

# # # Define an array of client arguments for each client
# # client_args=(
# #     "--model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-Truth --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs --gpu_id 0 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints",
# #     "--model sat-nerf --exp_name JAX_068_ds1_sat-nerf --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/root_dir/crops_rpcs_ba_v2/JAX_068 --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-RGB-crops/JAX_068 --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-Truth --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs --gpu_id 2 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints"
# # )

# # Start the first client
# echo "Starting client 1"
# python3 client.py --model sat-nerf \
#                 --exp_name JAX_068_ds1_sat-nerf \
#                 --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/root_dir/crops_rpcs_ba_v2/JAX_068 \
#                 --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-RGB-crops/JAX_068 \
#                 --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
#                 --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-Truth \
#                 --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs \
#                 --gpu_id 0 \
#                 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints &
# echo "Client 1 started"

# # Start the second client
# echo "Starting client 2"
# python3 client.py --model sat-nerf \
#                 --exp_name JAX_068_ds1_sat-nerf \
#                 --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/root_dir/crops_rpcs_ba_v2/JAX_068 \
#                 --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-RGB-crops/JAX_068 \
#                 --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
#                 --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-Truth \
#                 --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs \
#                 --gpu_id 1 \
#                 --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints &
# echo "Client 2 started"

# # Wait for all background processes to finish
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# wait
