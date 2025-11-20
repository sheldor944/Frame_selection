#!/bin/bash

# =============== CONFIG ==================
BASE_SCORE_DIR="/home/hpc4090/miraj/AKS/AKS/all_dbfp_runs"
OUTPUT_DIR="./converted_frames"       # where final renamed files will go
DATASET_NAME="videomme"

CHANGE_SCORE_SCRIPT="/home/hpc4090/miraj/AKS/AKS/evaluation/change_score.py"
INSERT_FRAME_SCRIPT="/home/hpc4090/miraj/AKS/AKS/evaluation/insert_frame_num.py"

# ACTUAL path where include_frame_idx.json is generated
INCLUDE_OUT_FILE="/home/hpc4090/miraj/AKS/AKS/datasets/videomme/include_frame_idx.json"

mkdir -p "$OUTPUT_DIR"
# =========================================

echo "Scanning DBFP outputs in: $BASE_SCORE_DIR"

counter=0

# Loop over all DBFP .json outputs
for json_file in "${BASE_SCORE_DIR}"/*.json; do

    filename=$(basename "$json_file")
    counter=$((counter + 1))

    # Print only every 100 files
    if (( counter % 100 == 1 )); then
        echo "----------------------------------------------------------"
        echo "Processing file $counter: $filename"
        echo "----------------------------------------------------------"
    fi

    # =============================
    #   Extract frame_num from kXX
    # =============================
    frame_num=$(echo "$filename" | grep -oP 'k\K[0-9]+')

    if [[ -z "$frame_num" ]]; then
        echo "ERROR: Could not extract frame_num from $filename"
        continue
    fi

    # =============================
    #   STEP 1: change_score.py
    # =============================
    python "$CHANGE_SCORE_SCRIPT" \
        --base_score_path "$BASE_SCORE_DIR" \
        --score_type "${filename%.json}" \
        --dataset_name "$DATASET_NAME"

    # =============================
    #   STEP 2: insert_frame_num.py
    # =============================
    python "$INSERT_FRAME_SCRIPT" \
        --frame_num "$frame_num" \
        --use_topk True

    # =============================
    #   Copy & rename output file
    # =============================

    if [[ -f "$INCLUDE_OUT_FILE" ]]; then
        DEST_FILE="${OUTPUT_DIR}/${filename}"
        cp "$INCLUDE_OUT_FILE" "$DEST_FILE"
    else
        echo "ERROR: include_frame_idx.json not found at $INCLUDE_OUT_FILE"
    fi

done

echo "======================================================"
echo "ALL FILES PROCESSED"
echo "Final outputs saved in: $OUTPUT_DIR"
echo "======================================================"
