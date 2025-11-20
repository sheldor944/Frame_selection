import os
import subprocess
from itertools import product

# Your DBFP script name
SCRIPT = "diffusion1.py"

# Fixed parameters (change if needed)
DATASET = "longvideobench"
FEATURE = "blip"
SCORE_PATH = "./outscores/longvideobench/blip/scores.json"
FRAME_PATH = "./outscores/longvideobench/blip/frames.json"
OUTPUT_DIR = "./all_dbfp_runs_longvideobench"

# Parameter sweeps
max_frames_list = [8, 16]
alpha_list = [ 0.75]
supp_radius_list = [1, 2, 3]
edge_weight_list = [ "temporal", "score_diff"]

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_name(m, a, r, e):
    """File name based on settings."""
    return f"dbfp_M{m}_alpha{a}_sup{r}_{e}.json"

def run_one(m, a, r, e):
    """Run one configuration of parameters."""
    output_name = build_name(m, a, r, e)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    cmd = [
        "python", SCRIPT,
        "--dataset_name", DATASET,
        "--extract_feature_model", FEATURE,
        "--score_path", SCORE_PATH,
        "--frame_path", FRAME_PATH,
        "--max_num_frames", str(m),
        "--alpha", str(a),
        "--suppression_radius", str(r),
        "--edge_weight_type", e,
        "--output_file", OUTPUT_DIR
    ]

    print("\n====================================================")
    print("Running:", " ".join(cmd))
    print("Output ->", output_path)
    print("====================================================\n")

    subprocess.run(cmd)

def main():
    combinations = list(product(max_frames_list, alpha_list, supp_radius_list, edge_weight_list))

    print(f"Total combinations: {len(combinations)}")

    for m, a, r, e in combinations:
        run_one(m, a, r, e)

if __name__ == "__main__":
    main()
