query_json_path=""
dt_dir=""
output_dir=""

echo "Running check_dt_valid.py"
echo "    query_json_path: $query_json_path"
echo "    dt_dir: $dt_dir"
echo "    output_dir: $output_dir"

python ./experiment/check_dt_valid.py --query_json_path $query_json_path --dt_dir $dt_dir --output_dir $output_dir