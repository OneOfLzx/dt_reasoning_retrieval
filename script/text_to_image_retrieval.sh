query_json_path=""
digital_twins_dir=""
result_output_dir=""
coarsely_answer_json_path=""
log_file=""
gpu_id=0

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Start llm retrieval"
echo "    query_json_path: "$query_json_path
echo "    digital_twins_dir: "$digital_twins_dir
echo "    result_output_dir: "$result_output_dir
echo "    gpu_id:"$gpu_id
echo "    coarsely_answer_json_path:"$coarsely_answer_json_path
echo "    log_file:"$log_file

python ./retrieval_pipeline/llm_retrieval.py\
 --query_json_path $query_json_path\
 --dt_dir $digital_twins_dir\
 --output_dir $result_output_dir\
 --coarsely_answer_json_path $coarsely_answer_json_path\
 > $log_file
