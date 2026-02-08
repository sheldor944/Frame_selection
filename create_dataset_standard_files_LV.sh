#!/bin/bash

# ==================== CONFIG ====================
BASE_SCORE_DIR="./THESIS/Thesis_TOP_K/longvideobench"
OUTPUT_DIR="./THESIS_LV_TOP_K"
DATASET_NAME="longvideobench"

CHANGE_SCORE_SCRIPT="./evaluation/change_score.py"

# Where change_score.py writes its output
INCLUDE_OUT_FILE="./datasets/longvideobench/include_frame_idx.json"

mkdir -p "$OUTPUT_DIR"
# =================================================

echo "Scanning DBFP outputs in: $BASE_SCORE_DIR"

counter=0

# Loop through all .json result files
for json_file in "${BASE_SCORE_DIR}"/*.json; do
    filename=$(basename "$json_file")
    counter=$((counter + 1))

    # Periodic status log
    if (( counter % 100 == 0 )); then
        echo "----------------------------------------------------------"
        echo "Processing file $counter: $filename"
        echo "----------------------------------------------------------"
    fi

    # Extract score_type (remove .json)
    score_type="${filename%.json}"

    # =============================
    #   STEP 1: change_score.py
    # =============================
    python3 "$CHANGE_SCORE_SCRIPT" \
        --base_score_path "$BASE_SCORE_DIR" \
        --score_type "$score_type" \
        --dataset_name "$DATASET_NAME"

    # =============================
    #   Copy the generated file
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
