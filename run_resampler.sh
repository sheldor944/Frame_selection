#!/bin/bash

# Configuration
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H)
LOG_FILE="${LOG_DIR}/resampler__r2_f2_Modified${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/resampler.pid"

# Create log directory
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Starting Dense Resampler"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Run with nohup
# nohup python -u re_sampler.py \
#     --score_threshold 70 \
#     --neighbor_radius 2 \
#     --dense_fps 8.0 \
#     --num_videos 2700 \
#     > ${LOG_FILE} 2>&1 &


nohup python -u re_sampler_modfied_checkpoint.py \
    --model_type blip \
    --batch_size 256 \
    --num_workers 32 \
    --use_fp16 \
    --dense_fps 2.0 \
    --neighbor_radius 2 \
    --device cuda:0 \
    --resume \
    > ${LOG_FILE} 2>&1 &



# Save PID
PID=$!
echo ${PID} > ${PID_FILE}

echo "Process started with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo "Kill with: kill $(cat ${PID_FILE})"
echo ""

# Create symlink to latest log
ln -sf ${LOG_FILE} ${LOG_DIR}/resampler_latest.log

echo "Quick monitor: tail -f ${LOG_DIR}/resampler_latest.log"