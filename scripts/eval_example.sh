# single video, structural_correctness and dynamic_degree
CUDA_VISIBLE_DEVICES=0 python3 eval_example.py \
    --video_path1 example_videos/mochi_00002.mp4 \
    --model_name qwen2-vl-7b \
    --aspect structural_correctness,dynamic_degree \
    --eval_mode single_soft_yn 

# single video, tv_alignment
CUDA_VISIBLE_DEVICES=0 python3 eval_example.py \
    --video_path1 example_videos/mochi_00002.mp4 \
    --model_name qwen2-vl-7b \
    --video_prompt "a man wearing red hat staring at the camera" \
    --aspect tv_alignment

# video pair, structural_correctness
CUDA_VISIBLE_DEVICES=0 python3 eval_example.py \
    --video_path1 example_videos/mochi_00002.mp4 \
    --video_path2 example_videos/OpenSora1.2_00002.mp4 \
    --eval_mode pairwise \
    --model_name qwen2-vl-7b \
    --aspect structural_correctness 

# Evaluate customized aspect
CUDA_VISIBLE_DEVICES=0 python3 eval_example.py \
    --video_path1 example_videos/mochi_00002.mp4 \
    --model_name qwen2-vl-7b \
    --aspect custom \
    --eval_mode single_soft_custom \
    --custom_prompt "Is the video containing sexual or violent material?\nPlease directly answer yes or no:" \
    --pos_tokens "yes,Yes,YES" \
    --neg_tokens "no,No,NO"