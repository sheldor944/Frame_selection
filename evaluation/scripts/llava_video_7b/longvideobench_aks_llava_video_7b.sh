# base_score_path=./selected_frames/longvideobench/blip
# score_type=selected_frames
# dataset_name=longvideobench

# # python ./evaluation/change_score.py \
# python /home/hpc4090/miraj/AKS/AKS/evaluation/change_score.py\
#     --base_score_path $base_score_path \
#     --score_type $score_type \
#     --dataset_name $dataset_name 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=./checkpoints/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False,use_topk=True \
#     --tasks longvideobench_val_v \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llavavid_7b_qwen_lvb_v \
#     --output_path ./results/${score_type}



#!/bin/bash
# ===========================================================
# LongVideoBench Evaluation with LLaVA-Video-7B-Qwen2 (AKS)
# ===========================================================

# ========== CONFIGURATION ==========
# Root of AKS repository
#!/bin/bash
set -e

# ==============================
#   CONFIGURATION
# ==============================
export PYTHONPATH=$PYTHONPATH:/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT

base_score_path=./selected_frames/longvideobench/blip
score_type=selected_dbfp_longvideobench_blip_k16_alpha0.75_sup3_score_diff
dataset_name=longvideobench

# ==============================
#   STEP 1: Convert score file
# ==============================
echo "=== Step 1: Running change_score.py on ${score_type} ==="
python /home/hpc4090/miraj/AKS/AKS/evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name

# ==============================
#   STEP 2: Run evaluation
# ==============================
# echo "=== Step 2: Running evaluation with lmms_eval ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False \
#     --tasks longvideobench_val_v \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llavavid_7b_qwen_lvb_v \
#     --output_path ./results/${score_type}

echo "=== Evaluation Complete ==="
