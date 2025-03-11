output_path=results/vbench

torchrun --nproc_per_node=8 --standalone eval_vbench.py \
    --dimension aesthetic_quality imaging_quality dynamic_degree subject_consistency motion_smoothness overall_consistency \
    --mode=custom_input \
    --full_json_dir vbench_info.json \
    --output_path $output_path \
    --load_ckpt_from_local true

python3 report_vbench_results.py