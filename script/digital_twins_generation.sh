#!/bin/bash

image_path_list_txt=""
digital_twins_dir=""
out_log_path=""
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

echo "Start image to digital twins"
echo "    image_path_list_txt: "$image_path_list_txt
echo "    digital_twins_dir: "$digital_twins_dir
echo "    gpu_id:"$gpu_id
echo "    out_log_path: "$out_log_path

python ./digital_twins_pipeline/digital_twins_generation.py\
 --image_path_list_txt $image_path_list_txt\
 --digital_twins_dir $digital_twins_dir\
 > $out_log_path

