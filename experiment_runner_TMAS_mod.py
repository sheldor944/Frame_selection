import os
import subprocess
from itertools import product

# Your TMAS script name
SCRIPT = "PPTMAS_param_free.py"

# ═══════════════════════════════════════════════════════════════════════
# DATASET AND FEATURE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════
DATASETS = ["longvideobench", "videomme"]
FEATURES = ["clip", "blip", "sevila"]

# Base output directory
BASE_OUTPUT_DIR = "./THESIS"

# ═══════════════════════════════════════════════════════════════════════
# PARAMETER SWEEPS
# ═══════════════════════════════════════════════════════════════════════
max_frames_list = [8, 16, 32]
curvature_method_list = ['second_derivative', 'laplacian', 'gradient_change']
curvature_smoothing_list = [0]
curvature_normalize_list = ['max']
tmas_decay_mode_list = ['half_life', 'budget_based']

# ═══════════════════════════════════════════════════════════════════════
# FIXED DEFAULT VALUES
# ═══════════════════════════════════════════════════════════════════════
RATIO = 1
USE_CURVATURE = True
CURVATURE_CLIP_PERCENTILE = 95.0
TMAS_MODE = 'auto'
TMAS_AUTO_SCALING = 'hybrid'
TMAS_AUTO_COVERAGE = 1.73
TMAS_AUTO_MIN_RADIUS = 1.0
TMAS_DECAY_FLOOR = 0.0
OPTIMIZE_REMAINING = True

def get_paths(dataset, feature):
    """
    Generate paths for a given dataset and feature combination.
    
    Directory structure:
        ./CurativeSmooth_Results/
            ├── longvideobench/
            └── videomme/
            
    All features for each dataset go in the same folder.
    Feature name is part of the filename.
    """
    return {
        'score_path': f"./outscores/{dataset}/{feature}/scores.json",
        'frame_path': f"./outscores/{dataset}/{feature}/frames.json",
        'metadata_path': f"./datasets/{dataset}/metadata.json",
        'output_dir': os.path.join(BASE_OUTPUT_DIR, dataset)
    }

def build_name(dataset, feature, m, curv_method, curv_smooth, curv_norm, decay_mode):
    """Build filename based on varying parameters."""
    name = f"selected_tmas_{dataset}_{feature}_k{m}"
    name += f"_curv{curv_method[:4]}"  # curvseco, curvlapl, curvgrad
    name += f"_norm{curv_norm[:3]}"  # normmax, normstd, normiqr
    
    if curv_smooth > 0:
        name += f"_sm{curv_smooth}"
    
    name += f"_{TMAS_MODE}"
    name += f"_{TMAS_AUTO_SCALING}_cov{TMAS_AUTO_COVERAGE}"
    name += f"_{decay_mode}"
    
    if OPTIMIZE_REMAINING:
        name += "_opt"
    
    name += ".json"
    return name

def check_paths_exist(paths):
    """Check if required input files exist."""
    missing = []
    for key in ['score_path', 'frame_path', 'metadata_path']:
        if not os.path.exists(paths[key]):
            missing.append(f"{key}: {paths[key]}")
    return missing

def run_one(dataset, feature, m, curv_method, curv_smooth, curv_norm, decay_mode):
    """Run one configuration of parameters."""
    
    # Get paths for this dataset/feature combo
    paths = get_paths(dataset, feature)
    
    # Check if paths exist
    missing = check_paths_exist(paths)
    if missing:
        print(f"\n⚠️  SKIPPING: Missing files for {dataset}/{feature}:")
        for m_file in missing:
            print(f"    - {m_file}")
        return None
    
    # Create output directory (only dataset folder)
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    output_name = build_name(dataset, feature, m, curv_method, curv_smooth, curv_norm, decay_mode)
    output_path = os.path.join(paths['output_dir'], output_name)
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"\n✓ SKIPPING: Output already exists")
        print(f"  {output_path}")
        return 'skipped'

    # Base command
    cmd = [
        "python3", SCRIPT,
        "--dataset_name", dataset,
        "--extract_feature_model", feature,
        "--score_path", paths['score_path'],
        "--frame_path", paths['frame_path'],
        "--metadata_path", paths['metadata_path'],
        "--max_num_frames", str(m),
        "--ratio", str(RATIO),
        "--output_file", paths['output_dir'],
        "--tmas_mode", TMAS_MODE,
        "--tmas_decay_mode", decay_mode,
        "--tmas_auto_scaling", TMAS_AUTO_SCALING,
        "--tmas_auto_coverage", str(TMAS_AUTO_COVERAGE),
        "--tmas_auto_min_radius", str(TMAS_AUTO_MIN_RADIUS),
        "--tmas_decay_floor", str(TMAS_DECAY_FLOOR),
    ]
    
    # Curvature parameters
    cmd.append("--use_curvature")
    cmd.extend([
        "--curvature_method", curv_method,
        "--curvature_normalize", curv_norm,
        "--curvature_smoothing", str(curv_smooth),
        "--curvature_clip_percentile", str(CURVATURE_CLIP_PERCENTILE),
    ])
    
    # Optimization flag
    if OPTIMIZE_REMAINING:
        cmd.append("--optimize_remaining")

    print("\n" + "="*80)
    print("Running configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Feature Model: {feature}")
    print(f"  Max Frames: {m}")
    print(f"  Curvature Method: {curv_method}")
    print(f"  Curvature Smoothing: {curv_smooth}")
    print(f"  Curvature Normalize: {curv_norm}")
    print(f"  Decay Mode: {decay_mode}")
    print(f"  Output Dir: {paths['output_dir']}")
    print(f"  Output File: {output_name}")
    print("="*80 + "\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ SUCCESS: {output_name}")
        print(f"   Saved to: {paths['output_dir']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {output_name}")
        print(f"   Error: {e}")
        return False

def print_directory_structure():
    """Print the expected directory structure."""
    print("\n" + "="*80)
    print("📁 OUTPUT DIRECTORY STRUCTURE:")
    print("="*80)
    print(f"{BASE_OUTPUT_DIR}/")
    for i, dataset in enumerate(DATASETS):
        connector = "└──" if i == len(DATASETS) - 1 else "├──"
        print(f"{connector} {dataset}/")
        print(f"    (All features: {', '.join(FEATURES)} stored here)")
    print("="*80 + "\n")

def count_files_in_output():
    """Count files in each output directory."""
    print("\n" + "="*80)
    print("📊 OUTPUT FILE COUNTS BY DATASET:")
    print("="*80)
    
    total_files = 0
    for dataset in DATASETS:
        output_dir = os.path.join(BASE_OUTPUT_DIR, dataset)
        print(f"\n{dataset}:")
        print(f"  Directory: {output_dir}")
        
        if os.path.exists(output_dir):
            # Count by feature
            for feature in FEATURES:
                json_files = [f for f in os.listdir(output_dir) 
                             if f.endswith('.json') 
                             and not f.endswith('_stats.json')
                             and f"_{feature}_" in f]
                stats_files = [f for f in os.listdir(output_dir) 
                              if f.endswith('_stats.json')
                              and f"_{feature}_" in f]
                num_files = len(json_files)
                total_files += num_files
                print(f"    {feature:10s}: {num_files:4d} result files, {len(stats_files):4d} stats files")
            
            # Total for this dataset
            all_json = [f for f in os.listdir(output_dir) 
                       if f.endswith('.json') and not f.endswith('_stats.json')]
            print(f"    {'TOTAL':10s}: {len(all_json):4d} result files")
        else:
            print(f"  Directory does not exist yet")
    
    print(f"\n{'GRAND TOTAL:':13s} {total_files} result files across all datasets")
    print("="*80)

def main():
    # Generate all combinations
    all_combinations = list(product(
        DATASETS,
        FEATURES,
        max_frames_list,
        curvature_method_list,
        curvature_smoothing_list,
        curvature_normalize_list,
        tmas_decay_mode_list
    ))

    print("="*80)
    print(f"TMAS + Curvature Multi-Dataset/Feature Experiment Runner")
    print("="*80)
    print(f"Datasets: {DATASETS}")
    print(f"Feature Models: {FEATURES}")
    print(f"Base Output Directory: {BASE_OUTPUT_DIR}")
    print(f"\nFixed Parameters:")
    print(f"  TMAS Mode: {TMAS_MODE}")
    print(f"  Auto Scaling: {TMAS_AUTO_SCALING}")
    print(f"  Auto Coverage: {TMAS_AUTO_COVERAGE}")
    print(f"  Auto Min Radius: {TMAS_AUTO_MIN_RADIUS}")
    print(f"  Curvature Clip: {CURVATURE_CLIP_PERCENTILE}")
    print(f"  Optimize Remaining: {OPTIMIZE_REMAINING}")
    print(f"\nVariable Parameters:")
    print(f"  Max Frames: {max_frames_list}")
    print(f"  Curvature Methods: {curvature_method_list}")
    print(f"  Curvature Smoothing: {curvature_smoothing_list}")
    print(f"  Curvature Normalize: {curvature_normalize_list}")
    print(f"  Decay Modes: {tmas_decay_mode_list}")
    print(f"\nTotal combinations: {len(all_combinations)}")
    
    # Show directory structure
    print_directory_structure()

    # Track statistics
    stats = {
        'total': len(all_combinations),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'missing_files': 0
    }

    for idx, (dataset, feature, m, curv_method, curv_smooth, curv_norm, decay_mode) in enumerate(all_combinations, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(all_combinations)}] Progress: {idx/len(all_combinations)*100:.1f}%")
        print(f"{'='*80}")
        
        result = run_one(dataset, feature, m, curv_method, curv_smooth, curv_norm, decay_mode)
        
        if result is None:
            stats['missing_files'] += 1
        elif result == 'skipped':
            stats['skipped'] += 1
        elif result is False:
            stats['failed'] += 1
        elif result is True:
            stats['success'] += 1

    # Final summary
    print("\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total configurations: {stats['total']}")
    print(f"✅ Successful runs: {stats['success']}")
    print(f"❌ Failed runs: {stats['failed']}")
    print(f"⏭️  Skipped (existing): {stats['skipped']}")
    print(f"🚫 Skipped (missing files): {stats['missing_files']}")
    print("="*80)
    
    # Count files in output directories
    count_files_in_output()
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print(f"Results organized in: {BASE_OUTPUT_DIR}")
    print("  Structure:")
    print(f"    {BASE_OUTPUT_DIR}/longvideobench/ - All features mixed")
    print(f"    {BASE_OUTPUT_DIR}/videomme/       - All features mixed")
    print("="*80)

if __name__ == "__main__":
    main()