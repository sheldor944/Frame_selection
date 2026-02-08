import os
import subprocess
from itertools import product

# Your TMAS script name
SCRIPT = "PPTMAS_param_free.py"

# Fixed parameters
DATASET = "videomme"  # or "videomme"
FEATURE = "clip"
SCORE_PATH = f"./outscores/{DATASET}/{FEATURE}/scores.json"
FRAME_PATH = f"./outscores/{DATASET}/{FEATURE}/frames.json"
METADATA_PATH = f"./datasets/{DATASET}/metadata.json"
OUTPUT_DIR = f"./CurativeSmooth_test_{DATASET}_TMAS_CURVATURE"

# Parameter sweeps (only the ones you want to vary)
max_frames_list = [8,16,32]
curvature_method_list = ['second_derivative', 'laplacian', 'gradient_change']
curvature_smoothing_list = [0]
tmas_decay_mode_list = ['half_life', 'budget_based']

# Fixed default values (not swept)
RATIO = 1
USE_CURVATURE = True
CURVATURE_NORMALIZE = 'max'
CURVATURE_CLIP_PERCENTILE = 95.0
TMAS_MODE = 'auto'
TMAS_AUTO_SCALING = 'hybrid'
TMAS_AUTO_COVERAGE = 1.73
TMAS_DECAY_FLOOR = 0.0  # Default from argparse
OPTIMIZE_REMAINING = True

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_name(m, curv_method, curv_smooth, decay_mode):
    """Build filename based on varying parameters only."""
    name = f"selected_tmas_{DATASET}_{FEATURE}_k{m}"
    name += f"_curv{curv_method[:4]}"  # curvseco, curvlapl, curvgrad
    name += f"_norm{CURVATURE_NORMALIZE[:3]}"  # normmax
    
    if curv_smooth > 0:
        name += f"_sm{curv_smooth}"
    
    name += f"_{TMAS_MODE}"
    name += f"_{TMAS_AUTO_SCALING}_cov{TMAS_AUTO_COVERAGE}"
    name += f"_{decay_mode}"
    
    if OPTIMIZE_REMAINING:
        name += "_opt"
    
    name += ".json"
    return name

def run_one(m, curv_method, curv_smooth, decay_mode):
    """Run one configuration of parameters."""
    
    output_name = build_name(m, curv_method, curv_smooth, decay_mode)
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
        "--ratio", str(RATIO),
        "--output_file", OUTPUT_DIR,
        "--tmas_mode", TMAS_MODE,
        "--tmas_decay_mode", decay_mode,
        "--tmas_auto_scaling", TMAS_AUTO_SCALING,
        "--tmas_auto_coverage", str(TMAS_AUTO_COVERAGE),
    ]
    
    # Curvature parameters
    cmd.append("--use_curvature")
    cmd.extend([
        "--curvature_method", curv_method,
        "--curvature_normalize", CURVATURE_NORMALIZE,
        "--curvature_smoothing", str(curv_smooth),
        "--curvature_clip_percentile", str(CURVATURE_CLIP_PERCENTILE),
    ])
    
    # Optimization flag
    if OPTIMIZE_REMAINING:
        cmd.append("--optimize_remaining")

    print("\n" + "="*80)
    print("Running configuration:")
    print(f"  Max Frames: {m}")
    print(f"  Curvature Method: {curv_method}")
    print(f"  Curvature Smoothing: {curv_smooth}")
    print(f"  Decay Mode: {decay_mode}")
    print(f"  Output: {output_name}")
    print("="*80 + "\n")

    subprocess.run(cmd)

def main():
    # Generate all combinations
    all_combinations = list(product(
        max_frames_list,
        curvature_method_list,
        curvature_smoothing_list,
        tmas_decay_mode_list
    ))

    print("="*80)
    print(f"TMAS + Curvature Experiment Runner")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Feature Model: {FEATURE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nFixed Parameters:")
    print(f"  TMAS Mode: {TMAS_MODE}")
    print(f"  Auto Scaling: {TMAS_AUTO_SCALING}")
    print(f"  Auto Coverage: {TMAS_AUTO_COVERAGE}")
    print(f"  Curvature Normalize: {CURVATURE_NORMALIZE}")
    print(f"  Curvature Clip: {CURVATURE_CLIP_PERCENTILE}")
    print(f"  Optimize Remaining: {OPTIMIZE_REMAINING}")
    print(f"\nVariable Parameters:")
    print(f"  Max Frames: {max_frames_list}")
    print(f"  Curvature Methods: {curvature_method_list}")
    print(f"  Curvature Smoothing: {curvature_smoothing_list}")
    print(f"  Decay Modes: {tmas_decay_mode_list}")
    print(f"\nTotal combinations: {len(all_combinations)}")
    print("="*80 + "\n")

    for idx, (m, curv_method, curv_smooth, decay_mode) in enumerate(all_combinations, 1):
        print(f"\n[{idx}/{len(all_combinations)}] Starting experiment...")
        run_one(m, curv_method, curv_smooth, decay_mode)

    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()