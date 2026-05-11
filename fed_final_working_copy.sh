#!/bin/bash

echo "Starting server"
python3 servercopy.py &
sleep 3 # Sleep for 3s to give the server enough time to start
echo "started server, moving to clients"

for i in `seq 0 1`; do

    if [ $i -eq 0 ]; then
        echo "Starting client $i"
        python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl1/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl1/client1/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl1/logs \
                --gpu_id $i \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl1/checkpoints &
    else
        python3 client.py --model sat-nerf \
                --exp_name JAX_068_ds1_sat-nerf \
                --root_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/root_dir/crops_rpcs_ba_v2/JAX_068 \
                --img_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-RGB-crops/JAX_068 \
                --cache_dir /home/myid/kg23166/adcs/satnerf/cl2/cache/crops_rpcs_ba_v2/JAX_068_ds1 \
                --gt_dir /home/myid/kg23166/adcs/satnerf/cl2/client2/DFC2019/Track3-Truth \
                --logs_dir /home/myid/kg23166/adcs/satnerf/cl2/logs \
                --gpu_id $i \
                --ckpts_dir /home/myid/kg23166/adcs/satnerf/cl2/checkpoints &
    fi
done
wait