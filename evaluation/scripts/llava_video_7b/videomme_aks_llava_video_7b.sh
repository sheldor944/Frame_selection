# base_score_path=./selected_frames/videomme/blip
# score_type=selected_frames
# dataset_name=videomme

# python ./evaluation/change_score.py \
#     --base_score_path $base_score_path \
#     --score_type $score_type \
#     --dataset_name $dataset_name 

# frame_num=64
# use_topk=True

# python ./evaluation/insert_frame_num.py \
#     --frame_num $frame_num \
#     --use_topk $use_topk

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=./checkpoints/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False \
#     --tasks videomme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llavavid_7b_qwen_lvb_v \
#     --output_path ./results/${score_type}



#!/bin/bash
#!/bin/bash
set -e

# ==============================
#   CONFIGURATION
# ==============================
export PYTHONPATH=$PYTHONPATH:/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT

base_score_path=./selected_frames/videomme/blip
score_type=selected_frames_dbfp_20_alpha_0.75_score_diff_power_law_power_2.0
dataset_name=videomme

frame_num=20
use_topk=True

# ==============================
#   STEP 1: Convert score file
# ==============================
echo "=== Step 1: Running change_score.py on ${score_type} ==="
python /home/hpc4090/miraj/AKS/AKS/evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name

# ==============================
#   STEP 2: Insert frame number
# ==============================
echo "=== Step 2: Running insert_frame_num.py ==="
python /home/hpc4090/miraj/AKS/AKS/evaluation/insert_frame_num.py \
    --frame_num $frame_num \
    --use_topk $use_topk

# ==============================
#   STEP 3: Run evaluation
# ==============================
# echo "=== Step 3: Running evaluation with lmms_eval ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False \
#     --tasks videomme \
#     --batch_size 2 \
#     --log_samples \
#     --log_samples_suffix llavavid_7b_qwen_lvb_v \
#     --output_path ./results/${score_type}

# run exactly your command, but add -X importtime and capture stderr
# CUDA_VISIBLE_DEVICES=0 PYTHONPROFILEIMPORTTIME=1 \
# accelerate launch --num_processes 1 --main_process_port 12345 \
#     -X importtime -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False \
#     --tasks videomme \
#     --batch_size 2 \
#     --log_samples \
#     --log_samples_suffix llavavid_7b_qwen_lvb_v \
#     --output_path ./results/${score_type} \
#     2> importtime.log



echo "=== Evaluation Complete ==="