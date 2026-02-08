import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize diffusion effects on randomly selected samples"
    )
    parser.add_argument("--score_path", type=str, default='./output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json',
                        help="Path to scores.json file")
    parser.add_argument("--frame_path", type=str, default='./output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json',
                        help="Path to frames.json file")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of random samples to select")
    parser.add_argument("--output_dir", type=str, default="./diffusion_visualizations",
                        help="Root output directory for plots")
    parser.add_argument("--alphas", type=str, default="0.7,0.85,0.95",
                        help="Comma-separated list of alpha values")
    parser.add_argument("--edge_types", type=str, default="score_diff,temporal",
                        help="Comma-separated list of edge types")
    parser.add_argument("--iterations", type=str, default="1,2,3",
                        help="Comma-separated list of diffusion iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0,1] range."""
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    else:
        return np.ones_like(scores, dtype=np.float64) * 0.5


def build_edge_weights(scores: np.ndarray, frame_ids: np.ndarray, edge_type: str) -> np.ndarray:
    """Build edge weights between neighbors."""
    N = len(scores)
    if N <= 1:
        return np.array([], dtype=np.float64)

    if edge_type == "uniform":
        return np.ones(N - 1, dtype=np.float64)

    elif edge_type == "score_diff":
        score_diffs = np.abs(np.diff(scores))
        weights = 1.0 / (score_diffs + 1e-6)
        weights = weights / weights.max()
        return weights.astype(np.float64)

    elif edge_type == "temporal":
        temporal_gaps = np.diff(frame_ids.astype(np.float64))
        weights = 1.0 / (temporal_gaps + 1.0)
        max_w = weights.max()
        if max_w > 0:
            weights = weights / max_w
        return weights.astype(np.float64)

    return np.ones(N - 1, dtype=np.float64)


def diffuse_scores(scores: np.ndarray, frame_ids: np.ndarray, 
                   alpha: float, edge_type: str, iterations: int) -> np.ndarray:
    """
    Perform 1D diffusion on scores.
    
    Args:
        scores: Normalized scores array
        frame_ids: Frame indices corresponding to scores
        alpha: Self-weight parameter (0-1)
        edge_type: Type of edge weighting ('uniform', 'score_diff', 'temporal')
        iterations: Number of diffusion iterations
    
    Returns:
        Diffused scores array
    """
    N = len(scores)
    if N <= 1 or iterations <= 0:
        return scores.copy()

    edge_weights = build_edge_weights(scores, frame_ids, edge_type)
    s = scores.copy().astype(np.float64)

    for _ in range(iterations):
        left_neighbors = np.zeros(N, dtype=np.float64)
        right_neighbors = np.zeros(N, dtype=np.float64)
        left_weights = np.zeros(N, dtype=np.float64)
        right_weights = np.zeros(N, dtype=np.float64)

        # Set up neighbors and weights
        left_neighbors[1:] = s[:-1]
        left_weights[1:] = edge_weights

        right_neighbors[:-1] = s[1:]
        right_weights[:-1] = edge_weights

        total_weights = left_weights + right_weights
        neighbor_contrib = np.zeros(N, dtype=np.float64)
        mask = total_weights > 0

        neighbor_contrib[mask] = (
            (left_neighbors[mask] * left_weights[mask] + 
             right_neighbors[mask] * right_weights[mask]) / total_weights[mask]
        )

        s = alpha * s + (1.0 - alpha) * neighbor_contrib

    return s


def plot_before_after(frame_ids: np.ndarray,
                     original_scores: np.ndarray, 
                     diffused_scores: np.ndarray,
                     sample_idx: int,
                     alpha: float,
                     edge_type: str,
                     iterations: int,
                     output_path: Path):
    """
    Create a before/after comparison plot.
    
    Args:
        frame_ids: Frame indices for x-axis
        original_scores: Original normalized scores
        diffused_scores: Diffused scores
        sample_idx: Sample index for title
        alpha: Alpha parameter used
        edge_type: Edge type used
        iterations: Number of iterations used
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Before diffusion
    axes[0].plot(frame_ids, original_scores, 'b-', linewidth=2, marker='o', markersize=5)
    axes[0].set_title(f'Sample {sample_idx} - Before Diffusion', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frame Index', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Add frame count info
    axes[0].text(0.02, 0.98, f'Frames: {len(frame_ids)}', 
                transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # After diffusion
    axes[1].plot(frame_ids, diffused_scores, 'r-', linewidth=2, marker='s', markersize=5)
    axes[1].set_title(
        f'Sample {sample_idx} - After Diffusion (α={alpha}, edge={edge_type}, iter={iterations})',
        fontsize=14, fontweight='bold'
    )
    axes[1].set_xlabel('Frame Index', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    
    # Add statistics
    smoothness_before = np.mean(np.abs(np.diff(original_scores)))
    smoothness_after = np.mean(np.abs(np.diff(diffused_scores)))
    stats_text = f'Smoothness: {smoothness_before:.4f} → {smoothness_after:.4f}'
    axes[1].text(0.02, 0.98, stats_text,
                transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def plot_overlay_comparison(frame_ids: np.ndarray,
                           original_scores: np.ndarray,
                           diffused_scores: np.ndarray,
                           sample_idx: int,
                           alpha: float,
                           edge_type: str,
                           iterations: int,
                           output_path: Path):
    """
    Create an overlay comparison plot showing both before and after.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(frame_ids, original_scores, 'b-', linewidth=2, marker='o', 
            markersize=6, label='Original', alpha=0.7)
    ax.plot(frame_ids, diffused_scores, 'r-', linewidth=2, marker='s', 
            markersize=6, label='Diffused', alpha=0.7)
    
    # Add vertical lines to show temporal gaps
    if edge_type == "temporal":
        for i in range(len(frame_ids) - 1):
            gap = frame_ids[i + 1] - frame_ids[i]
            if gap > 1:  # Show gaps larger than 1
                mid_x = (frame_ids[i] + frame_ids[i + 1]) / 2
                ax.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_title(
        f'Sample {sample_idx} - Diffusion Comparison (α={alpha}, edge={edge_type}, iter={iterations})',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Add statistics box
    smoothness_before = np.mean(np.abs(np.diff(original_scores)))
    smoothness_after = np.mean(np.abs(np.diff(diffused_scores)))
    mean_change = np.mean(np.abs(diffused_scores - original_scores))
    
    stats_text = (f'Frames: {len(frame_ids)}\n'
                 f'Smoothness: {smoothness_before:.4f} → {smoothness_after:.4f}\n'
                 f'Mean Abs Change: {mean_change:.4f}')
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def create_summary_stats(original_scores: np.ndarray,
                        diffused_scores: np.ndarray,
                        frame_ids: np.ndarray) -> dict:
    """Calculate summary statistics for the diffusion."""
    temporal_gaps = np.diff(frame_ids)
    
    return {
        'num_frames': len(original_scores),
        'frame_range': [int(frame_ids.min()), int(frame_ids.max())],
        'temporal_gaps': {
            'mean': float(np.mean(temporal_gaps)),
            'std': float(np.std(temporal_gaps)),
            'min': int(temporal_gaps.min()),
            'max': int(temporal_gaps.max()),
        },
        'original': {
            'mean': float(np.mean(original_scores)),
            'std': float(np.std(original_scores)),
            'min': float(np.min(original_scores)),
            'max': float(np.max(original_scores)),
            'smoothness': float(np.mean(np.abs(np.diff(original_scores)))),
        },
        'diffused': {
            'mean': float(np.mean(diffused_scores)),
            'std': float(np.std(diffused_scores)),
            'min': float(np.min(diffused_scores)),
            'max': float(np.max(diffused_scores)),
            'smoothness': float(np.mean(np.abs(np.diff(diffused_scores)))),
        },
        'change': {
            'mean_abs_diff': float(np.mean(np.abs(diffused_scores - original_scores))),
            'max_abs_diff': float(np.max(np.abs(diffused_scores - original_scores))),
            'smoothness_reduction': float(
                (np.mean(np.abs(np.diff(original_scores))) - 
                 np.mean(np.abs(np.diff(diffused_scores)))) / 
                np.mean(np.abs(np.diff(original_scores))) * 100
            ),
        }
    }


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse hyperparameters
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    edge_types = [et.strip() for et in args.edge_types.split(",") if et.strip()]
    iterations_list = [int(it) for it in args.iterations.split(",") if it.strip()]
    
    print("="*80)
    print("DIFFUSION VISUALIZATION PIPELINE")
    print("="*80)
    print(f"Scores Path: {args.score_path}")
    print(f"Frames Path: {args.frame_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Alphas: {alphas}")
    print(f"Edge types: {edge_types}")
    print(f"Iterations: {iterations_list}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Load JSON data
    print("\n📂 Loading data...")
    with open(args.score_path, 'r') as f:
        all_scores = json.load(f)
    
    with open(args.frame_path, 'r') as f:
        all_frames = json.load(f)
    
    print(f"✓ Loaded {len(all_scores)} score entries")
    print(f"✓ Loaded {len(all_frames)} frame entries")
    
    # Validate and filter entries
    valid_entries = []
    for idx in range(min(len(all_scores), len(all_frames))):
        scores = all_scores[idx]
        frames = all_frames[idx]
        
        if len(scores) == len(frames) and len(scores) >= 2:
            valid_entries.append((idx, scores, frames))
    
    if len(valid_entries) == 0:
        print("❌ No valid entries found (all entries are empty, mismatched, or too short)")
        return
    
    print(f"✓ Found {len(valid_entries)} valid entries (matching lengths >= 2)")
    
    # Randomly select samples
    num_samples = min(args.num_samples, len(valid_entries))
    selected_samples = random.sample(valid_entries, num_samples)
    
    print(f"\n🎲 Randomly selected {num_samples} samples")
    print(f"   Sample indices: {[idx for idx, _, _ in selected_samples]}")
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'score_path': args.score_path,
        'frame_path': args.frame_path,
        'num_samples': num_samples,
        'selected_indices': [idx for idx, _, _ in selected_samples],
        'alphas': alphas,
        'edge_types': edge_types,
        'iterations': iterations_list,
        'seed': args.seed,
        'timestamp': timestamp,
    }
    
    with open(output_root / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📁 Output directory: {output_root}")
    
    # Process each sample with all hyperparameter combinations
    all_stats = {}
    
    for sample_num, (original_idx, scores_list, frames_list) in enumerate(selected_samples, 1):
        print(f"\n{'='*80}")
        print(f"Processing Sample {sample_num}/{num_samples} (Original Index: {original_idx})")
        print(f"{'='*80}")
        
        # Convert to numpy arrays
        scores = np.array(scores_list, dtype=np.float64)
        frames = np.array(frames_list, dtype=np.int64)
        
        # Normalize scores
        scores_norm = min_max_normalize(scores)
        
        print(f"  Number of frames: {len(frames)}")
        print(f"  Frame range: [{frames.min()}, {frames.max()}]")
        print(f"  Score range (original): [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score range (normalized): [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")
        
        temporal_gaps = np.diff(frames)
        print(f"  Temporal gaps: mean={temporal_gaps.mean():.2f}, "
              f"std={temporal_gaps.std():.2f}, "
              f"min={temporal_gaps.min()}, "
              f"max={temporal_gaps.max()}")
        
        sample_stats = {}
        
        # Create sample directory
        sample_dir = output_root / f"sample_{sample_num:03d}_idx_{original_idx}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save original data
        np.save(sample_dir / 'original_scores.npy', scores)
        np.save(sample_dir / 'normalized_scores.npy', scores_norm)
        np.save(sample_dir / 'frame_ids.npy', frames)
        
        # Save as JSON too for easy inspection
        with open(sample_dir / 'data.json', 'w') as f:
            json.dump({
                'original_scores': scores_list,
                'frame_ids': frames_list,
                'normalized_scores': scores_norm.tolist()
            }, f, indent=2)
        
        # Try all hyperparameter combinations
        for edge_type in edge_types:
            for alpha in alphas:
                for iterations in iterations_list:
                    print(f"\n  🔄 Diffusing: α={alpha}, edge={edge_type}, iter={iterations}")
                    
                    # Perform diffusion
                    diffused = diffuse_scores(scores_norm, frames, alpha, edge_type, iterations)
                    
                    # Create subdirectory for this configuration
                    config_name = f"alpha_{alpha}_edge_{edge_type}_iter_{iterations}"
                    config_dir = sample_dir / config_name
                    config_dir.mkdir(exist_ok=True)
                    
                    # Save diffused scores
                    np.save(config_dir / 'diffused_scores.npy', diffused)
                    
                    # Calculate statistics
                    stats = create_summary_stats(scores_norm, diffused, frames)
                    sample_stats[config_name] = stats
                    
                    with open(config_dir / 'statistics.json', 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                    # Create plots
                    plot_before_after(
                        frames, scores_norm, diffused, sample_num, alpha, edge_type, iterations,
                        config_dir / 'before_after.png'
                    )
                    
                    plot_overlay_comparison(
                        frames, scores_norm, diffused, sample_num, alpha, edge_type, iterations,
                        config_dir / 'overlay.png'
                    )
                    
                    print(f"     Smoothness reduction: {stats['change']['smoothness_reduction']:.2f}%")
        
        # Save all stats for this sample
        all_stats[f"sample_{sample_num}_idx_{original_idx}"] = sample_stats
        
        with open(sample_dir / 'all_statistics.json', 'w') as f:
            json.dump(sample_stats, f, indent=2)
        
        print(f"  ✅ Completed sample {sample_num}")
    
    # Save global statistics
    with open(output_root / 'all_samples_statistics.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Create summary report
    print(f"\n{'='*80}")
    print("📊 GENERATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary_lines = ["DIFFUSION ANALYSIS SUMMARY", "=" * 80, ""]
    
    for sample_key, sample_data in all_stats.items():
        summary_lines.append(f"\n{sample_key}:")
        summary_lines.append("-" * 40)
        
        for config_name, stats in sample_data.items():
            summary_lines.append(f"\n  {config_name}:")
            summary_lines.append(f"    Smoothness reduction: {stats['change']['smoothness_reduction']:.2f}%")
            summary_lines.append(f"    Mean absolute change: {stats['change']['mean_abs_diff']:.4f}")
            summary_lines.append(f"    Original smoothness: {stats['original']['smoothness']:.4f}")
            summary_lines.append(f"    Diffused smoothness: {stats['diffused']['smoothness']:.4f}")
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_root / 'summary_report.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\n{'='*80}")
    print("✅ ALL PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Results saved to: {output_root}")
    print(f"   - Configuration: config.json")
    print(f"   - Global stats: all_samples_statistics.json")
    print(f"   - Summary report: summary_report.txt")
    print(f"   - {num_samples} sample directories with plots and data")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()