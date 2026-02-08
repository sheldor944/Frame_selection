import os
import subprocess
from itertools import product

# Your DBFP script name
SCRIPT = "optimized_diffusion_adaptive_radius.py"

# Fixed parameters (change if needed)
DATASET = "longvideobench"  # or "videomme"
FEATURE = "blip"
SCORE_PATH = f"./outscores/{DATASET}/{FEATURE}/scores.json"
FRAME_PATH = f"./outscores/{DATASET}/{FEATURE}/frames.json"
METADATA_PATH = f"./datasets/{DATASET}/metadata.json"
OUTPUT_DIR = f"./selected_frames_{DATASET}_FINAL"

# Parameter sweeps
max_frames_list = [64]
alpha_list = [ .8, .85]
edge_weight_list = ["temporal", "score_diff"]
diffusion_iterations_list = [1, 2, 3]  # None = auto (log2(N))

# Adaptive suppression radius configurations for LongVideoBench
# Each config is a dict with keys: 15, 60, 600, 3600
suppression_configs_lv = [
    {
        "name": "conservative",
        15: 2.0,
        60: 3.0,
        600: 5.0,
        3600: 8.0
    }
]

# Adaptive suppression radius configurations for VideoMME
suppression_configs_vmme = [
    {
        "name": "conservative",
        "short": 2.0,
        "medium": 3.0,
        "long": 5.0
    },
    {
        "name": "moderate",
        "short": 2.0,
        "medium": 4.0,
        "long": 6.0
    },
]

# Select appropriate configs based on dataset
if DATASET == "longvideobench":
    suppression_configs = suppression_configs_lv
else:
    suppression_configs = suppression_configs_vmme

# Optional: enable optimization
OPTIMIZE_REMAINING = True  # Set to False to disable

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_name(m, a, sup_config_name, e, diff_iters, opt):
    """
    Build filename based on settings.
    Format: selected_dbfp_dense_{dataset}_{feature}_k{frames}_alpha{alpha}_
            adaptive_{sup_config}_{edge_type}_iter{iters}_optimized.json
    """
    name = (
        f"selected_dbfp_dense_"
        f"{DATASET}_"
        f"{FEATURE}_"
        f"k{m}_"
        f"alpha{a}_"
        f"adaptive_{sup_config_name}_"
        f"{e}"
    )
    
    if diff_iters is not None:
        name += f"_iter{diff_iters}"
    else:
        name += "_iterAuto"
    
    if opt:
        name += "_optimized"
    
    name += ".json"
    return name

def run_one(m, a, sup_config, e, diff_iters):
    """Run one configuration of parameters."""
    output_name = build_name(m, a, sup_config["name"], e, diff_iters, OPTIMIZE_REMAINING)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Base command
    cmd = [
        "python3", SCRIPT,
        "--dataset_name", DATASET,
        "--extract_feature_model", FEATURE,
        "--score_path", SCORE_PATH,
        "--frame_path", FRAME_PATH,
        "--metadata_path", METADATA_PATH,
        "--max_num_frames", str(m),
        "--alpha", str(a),
        "--edge_weight_type", e,
        "--output_file", OUTPUT_DIR
    ]
    
    # Add diffusion iterations
    if diff_iters is not None:
        cmd.extend(["--diffusion_iterations", str(diff_iters)])
    
    # Add adaptive suppression radius parameters
    if DATASET == "longvideobench":
        cmd.extend([
            "--suppression_radius_15", str(sup_config[15]),
            "--suppression_radius_60", str(sup_config[60]),
            "--suppression_radius_600", str(sup_config[600]),
            "--suppression_radius_3600", str(sup_config[3600])
        ])
    else:  # videomme
        cmd.extend([
            "--suppression_radius_short", str(sup_config["short"]),
            "--suppression_radius_medium", str(sup_config["medium"]),
            "--suppression_radius_long", str(sup_config["long"])
        ])
    
    # Add optimization flag
    if not OPTIMIZE_REMAINING:
        cmd.append("--no_optimize_remaining")

    print("\n" + "="*80)
    print("Running configuration:")
    print(f"  Max Frames: {m}")
    print(f"  Alpha: {a}")
    print(f"  Suppression Config: {sup_config['name']}")
    if DATASET == "longvideobench":
        print(f"    15s → {sup_config[15]}")
        print(f"    60s → {sup_config[60]}")
        print(f"    600s → {sup_config[600]}")
        print(f"    3600s → {sup_config[3600]}")
    else:
        print(f"    short → {sup_config['short']}")
        print(f"    medium → {sup_config['medium']}")
        print(f"    long → {sup_config['long']}")
    print(f"  Edge Weight: {e}")
    print(f"  Diffusion Iters: {diff_iters if diff_iters else 'Auto'}")
    print(f"  Optimize Remaining: {OPTIMIZE_REMAINING}")
    print(f"  Output: {output_name}")
    print("="*80 + "\n")

    subprocess.run(cmd)

def main():
    combinations = list(product(
        max_frames_list, 
        alpha_list, 
        suppression_configs, 
        edge_weight_list,
        diffusion_iterations_list
    ))

    print("="*80)
    print(f"DBFP Experiment Runner - Adaptive Suppression Radius")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Feature Model: {FEATURE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Optimize Remaining: {OPTIMIZE_REMAINING}")
    print("="*80 + "\n")

    for idx, (m, a, sup_config, e, diff_iters) in enumerate(combinations, 1):
        print(f"\n[{idx}/{len(combinations)}] Starting experiment...")
        run_one(m, a, sup_config, e, diff_iters)

    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()