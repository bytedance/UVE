model_name=qwen2-vl-7b
local_model_path=local_model_path   # local path to load mllms, if cannot download from huggingface, set this to local path
video_path=uve_bench_videos
anno_path=annotations
result_path="results/${model_name}"
prompt_type="full"
eval_mode=pairwise
max_num_frames=12

# Single GPU
# CUDA_VISIBLE_DEVICES=0 python3 eval_uve_bench.py \
#     --model_name $model_name \
#     --local_model_path $local_model_path \
#     --video_path $video_path \
#     --anno_path $anno_path \
#     --result_path $result_path \
#     --max_num_frames $max_num_frames \
#     --prompt_type $prompt_type \
#     --eval_mode $eval_mode

# Multi GPU
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for IDX in $(seq 0 $(($gpu_count-1))); do
    CUDA_VISIBLE_DEVICES=${IDX} python3 eval_uve_bench.py \
        --model_name $model_name \
        --local_model_path $local_model_path \
        --video_path $video_path \
        --anno_path $anno_path \
        --result_path $result_path \
        --max_num_frames $max_num_frames \
        --eval_mode $eval_mode \
        --prompt_type $prompt_type \
        --num_chunks $gpu_count \
        --chunk_idx $IDX  &
done

wait

if [ "$prompt_type" == "full" ]; then
    prompt_type='' 
fi

# Concatnate inference results from multi-gpu
python3 cat_eval_results.py \
    --result_path $result_path \
    --result_prefix "${eval_mode}${prompt_type}_frame${max_num_frames}_${gpu_count}" \
    --eval_mode $eval_mode
