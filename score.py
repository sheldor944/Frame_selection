import json
import argparse
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import seaborn as sns

# ================================
#  Data Classes
# ================================
@dataclass
class VideoMetrics:
    """Stores all metrics for a single video"""
    video_id: int
    scbi: float
    score_quality: float
    coverage: float
    diversity: float
    efficiency: float
    balance: float
    num_selected: int
    num_dense: int

@dataclass
class AlgorithmResults:
    """Stores aggregate results for one algorithm/JSON"""
    name: str
    video_metrics: List[VideoMetrics]
    mean_scbi: float
    mean_score_quality: float
    mean_coverage: float
    mean_diversity: float
    mean_efficiency: float
    mean_balance: float

# ================================
#  Logging Setup
# ================================
def setup_logging(log_dir: Path):
    """Initialize logging to file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "evaluation_log.txt"
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

# ================================
#  METRIC COMPUTATIONS
# ================================
def compute_score_quality(scores: List[float], sel_idx: List[int]) -> float:
    """
    Metric 1: Score Quality (S)
    Measures how well selected frames capture high-quality frames
    """
    if len(sel_idx) == 0:
        return 0.0
    
    selected_scores = [scores[i] for i in sel_idx]
    sum_selected = sum(selected_scores)
    
    M = len(sel_idx)
    topM_scores = sorted(scores, reverse=True)[:M]
    sum_topM = sum(topM_scores)
    
    return sum_selected / (sum_topM + 1e-8)

def compute_coverage(frame_ids: List[int], sel_idx: List[int]) -> float:
    """
    Metric 2: Temporal Coverage (C)
    Measures how evenly selected frames span the video timeline
    """
    if len(sel_idx) < 2:
        return 0.0
    
    sel_ts = sorted([frame_ids[i] for i in sel_idx])
    gaps = np.diff(sel_ts)
    var_gaps = np.var(gaps)
    
    ideal_gap = (frame_ids[-1] - frame_ids[0]) / (len(sel_ts) - 1)
    ideal_gaps = [ideal_gap] * (len(sel_ts) - 1)
    var_ideal = np.var(ideal_gaps) + 1e-8
    
    return 1 - min(var_gaps / var_ideal, 1.0)

def compute_diversity(scores: List[float], sel_idx: List[int]) -> float:
    """
    Metric 3: Score Diversity (D)
    Measures variance in selected frame scores (higher = more diverse selection)
    """
    if len(sel_idx) < 2:
        return 0.0
    
    selected_scores = [scores[i] for i in sel_idx]
    score_std = np.std(selected_scores)
    all_std = np.std(scores) + 1e-8
    
    # Normalize by overall score variance
    return min(score_std / all_std, 1.0)

def compute_efficiency(num_selected: int, num_dense: int) -> float:
    """
    Metric 4: Selection Efficiency (E)
    Measures compression ratio (lower selection rate = higher efficiency)
    """
    if num_dense == 0:
        return 0.0
    
    compression_ratio = 1 - (num_selected / num_dense)
    return max(compression_ratio, 0.0)

def compute_balance(score_quality: float, coverage: float) -> float:
    """
    Metric 5: Score-Coverage Balance (B)
    Measures how balanced the algorithm is between quality and coverage
    Penalizes extreme imbalance
    """
    if score_quality == 0 or coverage == 0:
        return 0.0
    
    # Harmonic mean - penalizes imbalance more than arithmetic mean
    return 2 * (score_quality * coverage) / (score_quality + coverage)

def compute_all_metrics(
    frame_ids: List[int],
    scores: List[float],
    selected_frames: List[int],
    alpha: float = 0.5,
    video_id: int = None,
    debug: bool = False
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute all 5 metrics + SCBI
    Returns: (SCBI, S, C, D, E, B)
    """
    try:
        # Debug information
        if debug and video_id is not None:
            logging.debug(f"Video {video_id}:")
            logging.debug(f"  Dense frames type: {type(frame_ids)}, length: {len(frame_ids)}")
            logging.debug(f"  Dense frames sample: {frame_ids[:5] if len(frame_ids) > 0 else 'empty'}")
            logging.debug(f"  Selected frames type: {type(selected_frames)}, length: {len(selected_frames)}")
            logging.debug(f"  Selected frames sample: {selected_frames[:5] if len(selected_frames) > 0 else 'empty'}")
        
        # Handle empty selections
        if len(selected_frames) == 0:
            if debug:
                logging.warning(f"Video {video_id}: Empty selection")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Build timestamp to index mapping
        ts_to_idx = {t: i for i, t in enumerate(frame_ids)}
        
        # Try multiple matching strategies
        sel_idx = []
        
        # Strategy 1: Direct match (timestamps match exactly)
        sel_idx = [ts_to_idx[t] for t in selected_frames if t in ts_to_idx]
        
        # Strategy 2: If no matches, try index-based matching
        if len(sel_idx) == 0:
            # Assume selected_frames are indices, not timestamps
            sel_idx = [i for i in selected_frames if 0 <= i < len(frame_ids)]
            
            if debug and len(sel_idx) > 0:
                logging.info(f"Video {video_id}: Using index-based matching (found {len(sel_idx)} matches)")
        
        # Strategy 3: If still no matches, try finding nearest timestamps
        if len(sel_idx) == 0 and len(frame_ids) > 0:
            frame_ids_array = np.array(frame_ids)
            for sel_frame in selected_frames:
                # Find nearest timestamp
                idx = np.argmin(np.abs(frame_ids_array - sel_frame))
                if abs(frame_ids_array[idx] - sel_frame) <= 5:  # Tolerance of 5 frames
                    sel_idx.append(idx)
            
            if debug and len(sel_idx) > 0:
                logging.info(f"Video {video_id}: Using nearest-neighbor matching (found {len(sel_idx)} matches)")
        
        if len(sel_idx) == 0:
            if debug:
                logging.warning(f"Video {video_id}: No selected frames could be mapped to dense frames")
                logging.warning(f"  Dense frame range: {min(frame_ids) if frame_ids else 'N/A'} to {max(frame_ids) if frame_ids else 'N/A'}")
                logging.warning(f"  Selected frame range: {min(selected_frames) if selected_frames else 'N/A'} to {max(selected_frames) if selected_frames else 'N/A'}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Remove duplicates and sort
        sel_idx = sorted(list(set(sel_idx)))
        
        # Compute individual metrics
        S = compute_score_quality(scores, sel_idx)
        C = compute_coverage(frame_ids, sel_idx)
        D = compute_diversity(scores, sel_idx)
        E = compute_efficiency(len(sel_idx), len(frame_ids))
        B = compute_balance(S, C)
        
        # SCBI (primary composite metric)
        SCBI = alpha * S + (1 - alpha) * C
        
        if debug:
            logging.debug(f"Video {video_id} metrics: SCBI={SCBI:.3f}, S={S:.3f}, C={C:.3f}, D={D:.3f}, E={E:.3f}, B={B:.3f}")
        
        return SCBI, S, C, D, E, B
        
    except Exception as e:
        logging.error(f"Metric computation error for video {video_id}: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
# ================================
#  DATA LOADING
# ================================
def load_json(path: Path) -> any:
    """Load JSON file with error handling"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        raise

def find_json_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    """Find all JSON files in directory"""
    files = list(directory.glob(pattern))
    logging.info(f"Found {len(files)} JSON files in {directory}")
    return sorted(files)

# ================================
#  EVALUATION
# ================================
def evaluate_single_algorithm(
    selected_path: Path,
    frames: List[List[int]],
    scores: List[List[float]],
    alpha: float,
    debug_first_n: int = 3
) -> AlgorithmResults:
    """
    Evaluate one selection algorithm/JSON file
    """
    logging.info(f"Evaluating: {selected_path.name}")
    
    selected_data = load_json(selected_path)
    video_metrics_list = []
    
    # Track warnings
    warning_count = 0
    max_warnings_to_show = 10
    
    for idx, item in enumerate(tqdm(selected_data, desc=f"Processing {selected_path.stem}")):
        try:
            vid = int(item["video_id"])
            selected_frames = item["frame_idx"]
            
            # Validate video ID
            if vid >= len(frames) or vid >= len(scores):
                logging.error(f"Video ID {vid} out of range (max: {len(frames)-1})")
                continue
            
            frame_ids = frames[vid]
            score_list = scores[vid]
            
            # Enable debug for first few videos
            debug = (idx < debug_first_n)
            
            scbi, S, C, D, E, B = compute_all_metrics(
                frame_ids, score_list, selected_frames, alpha, video_id=vid, debug=debug
            )
            
            # Track zero scores
            if scbi == 0.0:
                warning_count += 1
                if warning_count <= max_warnings_to_show:
                    logging.warning(f"Video {vid}: Zero SCBI score")
            
            metrics = VideoMetrics(
                video_id=vid,
                scbi=scbi,
                score_quality=S,
                coverage=C,
                diversity=D,
                efficiency=E,
                balance=B,
                num_selected=len(selected_frames),
                num_dense=len(frame_ids)
            )
            video_metrics_list.append(metrics)
            
        except Exception as e:
            logging.error(f"Error processing video {item.get('video_id', 'unknown')}: {e}")
    
    # Log summary of warnings
    if warning_count > max_warnings_to_show:
        logging.warning(f"Total videos with zero SCBI: {warning_count} (showing first {max_warnings_to_show})")
    
    # Filter out videos with zero scores for mean calculation (optional)
    valid_metrics = [m for m in video_metrics_list if m.scbi > 0]
    
    if len(valid_metrics) == 0:
        logging.error(f"No valid metrics for {selected_path.name}!")
        # Return with zeros
        return AlgorithmResults(
            name=selected_path.stem,
            video_metrics=video_metrics_list,
            mean_scbi=0.0,
            mean_score_quality=0.0,
            mean_coverage=0.0,
            mean_diversity=0.0,
            mean_efficiency=0.0,
            mean_balance=0.0
        )
    
    logging.info(f"Valid videos: {len(valid_metrics)}/{len(video_metrics_list)}")
    
    # Compute aggregate statistics (using all videos, including zeros)
    return AlgorithmResults(
        name=selected_path.stem,
        video_metrics=video_metrics_list,
        mean_scbi=np.mean([m.scbi for m in video_metrics_list]),
        mean_score_quality=np.mean([m.score_quality for m in video_metrics_list]),
        mean_coverage=np.mean([m.coverage for m in video_metrics_list]),
        mean_diversity=np.mean([m.diversity for m in video_metrics_list]),
        mean_efficiency=np.mean([m.efficiency for m in video_metrics_list]),
        mean_balance=np.mean([m.balance for m in video_metrics_list])
    )


# ================================
#  PLOTTING
# ================================
def plot_single_algorithm_metrics(results: AlgorithmResults, out_dir: Path):
    """Create detailed plots for a single algorithm"""
    algo_dir = out_dir / results.name
    algo_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = results.video_metrics
    
    # Extract metric arrays
    scbi_vals = [m.scbi for m in metrics]
    S_vals = [m.score_quality for m in metrics]
    C_vals = [m.coverage for m in metrics]
    D_vals = [m.diversity for m in metrics]
    E_vals = [m.efficiency for m in metrics]
    B_vals = [m.balance for m in metrics]
    
    # 1. SCBI Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scbi_vals, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(results.mean_scbi, color='red', linestyle='--', 
                label=f'Mean: {results.mean_scbi:.3f}')
    plt.title(f'SCBI Distribution - {results.name}')
    plt.xlabel('SCBI Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(algo_dir / 'scbi_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. All Metrics Distributions (Subplots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Metric Distributions - {results.name}', fontsize=16)
    
    metric_data = [
        (scbi_vals, 'SCBI', results.mean_scbi),
        (S_vals, 'Score Quality (S)', results.mean_score_quality),
        (C_vals, 'Coverage (C)', results.mean_coverage),
        (D_vals, 'Diversity (D)', results.mean_diversity),
        (E_vals, 'Efficiency (E)', results.mean_efficiency),
        (B_vals, 'Balance (B)', results.mean_balance)
    ]
    
    for ax, (data, label, mean_val) in zip(axes.flat, metric_data):
        ax.hist(data, bins=25, edgecolor='black', alpha=0.7)
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.3f}')
        ax.set_title(label)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(algo_dir / 'all_metrics_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Score Quality vs Coverage Scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(S_vals, C_vals, c=scbi_vals, cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='SCBI Score')
    plt.xlabel('Score Quality (S)', fontsize=12)
    plt.ylabel('Coverage (C)', fontsize=12)
    plt.title(f'Score Quality vs Coverage - {results.name}', fontsize=14)
    plt.grid(alpha=0.3)
    
    # Add diagonal line (perfect balance)
    lims = [0, 1]
    plt.plot(lims, lims, 'r--', alpha=0.5, label='Perfect Balance')
    plt.legend()
    
    plt.savefig(algo_dir / 'score_vs_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Radar Chart for Average Metrics
    categories = ['Score\nQuality', 'Coverage', 'Diversity', 'Efficiency', 'Balance']
    values = [
        results.mean_score_quality,
        results.mean_coverage,
        results.mean_diversity,
        results.mean_efficiency,
        results.mean_balance
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label=results.name)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f'Metric Radar Chart - {results.name}', size=14, pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(algo_dir / 'radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Box Plot of All Metrics (FIX: use tick_labels instead of labels)
    plt.figure(figsize=(12, 6))
    data_to_plot = [scbi_vals, S_vals, C_vals, D_vals, E_vals, B_vals]
    tick_labels = ['SCBI', 'Score\nQuality', 'Coverage', 'Diversity', 'Efficiency', 'Balance']
    
    bp = plt.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True)  # FIXED
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    plt.title(f'Metric Box Plots - {results.name}', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(algo_dir / 'boxplot_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved plots for {results.name} in {algo_dir}")


def plot_algorithm_comparison(all_results: List[AlgorithmResults], out_dir: Path):
    """Create comparison plots across all algorithms"""
    comp_dir = out_dir / 'comparisons'
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    names = [r.name for r in all_results]
    
    # 1. Bar Chart - Mean Metrics Comparison
    metrics_dict = {
        'SCBI': [r.mean_scbi for r in all_results],
        'Score Quality': [r.mean_score_quality for r in all_results],
        'Coverage': [r.mean_coverage for r in all_results],
        'Diversity': [r.mean_diversity for r in all_results],
        'Efficiency': [r.mean_efficiency for r in all_results],
        'Balance': [r.mean_balance for r in all_results]
    }
    
    x = np.arange(len(names))
    width = 0.13
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        offset = width * (i - 2.5)
        ax.bar(x + offset, values, width, label=metric_name)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Mean Metrics Comparison Across Algorithms', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comp_dir / 'mean_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of Mean Metrics
    df = pd.DataFrame(metrics_dict, index=names)
    
    plt.figure(figsize=(10, max(8, len(names) * 0.5)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
    plt.title('Metrics Heatmap - Algorithm Comparison', fontsize=14)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.tight_layout()
    plt.savefig(comp_dir / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. SCBI Box Plot Comparison (FIXED: use tick_labels)
    plt.figure(figsize=(max(12, len(names) * 1.5), 7))
    scbi_data = [[m.scbi for m in r.video_metrics] for r in all_results]
    
    bp = plt.boxplot(scbi_data, tick_labels=names, patch_artist=True)  # FIXED
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('SCBI Distribution Comparison', fontsize=14)
    plt.ylabel('SCBI Score', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_dir / 'scbi_comparison_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Radar Chart Overlay
    categories = ['Score\nQuality', 'Coverage', 'Diversity', 'Efficiency', 'Balance']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for result, color in zip(all_results, colors):
        values = [
            result.mean_score_quality,
            result.mean_coverage,
            result.mean_diversity,
            result.mean_efficiency,
            result.mean_balance
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result.name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Algorithm Comparison - Radar Chart', size=16, pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(comp_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Ranking Table
    ranking_data = []
    for result in all_results:
        ranking_data.append({
            'Algorithm': result.name,
            'SCBI': result.mean_scbi,
            'Score Quality': result.mean_score_quality,
            'Coverage': result.mean_coverage,
            'Diversity': result.mean_diversity,
            'Efficiency': result.mean_efficiency,
            'Balance': result.mean_balance
        })
    
    df_rank = pd.DataFrame(ranking_data)
    df_rank = df_rank.sort_values('SCBI', ascending=False)
    df_rank.insert(0, 'Rank', range(1, len(df_rank) + 1))
    
    # Format values to 3 decimal places for display
    df_display = df_rank.copy()
    for col in ['SCBI', 'Score Quality', 'Coverage', 'Diversity', 'Efficiency', 'Balance']:
        df_display[col] = df_display[col].apply(lambda x: f'{x:.3f}')
    
    fig, ax = plt.subplots(figsize=(14, max(len(all_results) * 0.6 + 2, 6)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.08, 0.25] + [0.12] * 6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for j in range(len(df_display.columns)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Color code rankings
    for i in range(1, len(df_rank) + 1):
        if i == 1:
            color = '#FFD700'  # Gold
        elif i == 2:
            color = '#C0C0C0'  # Silver
        elif i == 3:
            color = '#CD7F32'  # Bronze
        else:
            color = 'white'
        
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Algorithm Rankings by SCBI', fontsize=16, pad=20, weight='bold')
    plt.savefig(comp_dir / 'ranking_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Violin Plot for SCBI Distribution
    plt.figure(figsize=(max(12, len(names) * 1.5), 7))
    
    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    for result in all_results:
        scbi_values = [m.scbi for m in result.video_metrics]
        plot_data.extend(scbi_values)
        plot_labels.extend([result.name] * len(scbi_values))
    
    df_violin = pd.DataFrame({'Algorithm': plot_labels, 'SCBI': plot_data})
    
    # import seaborn as sns
    sns.violinplot(data=df_violin, x='Algorithm', y='SCBI', palette='Set2')
    plt.title('SCBI Distribution - Violin Plot', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('SCBI Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_dir / 'scbi_violin_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Correlation Matrix between Metrics
    all_scbi = []
    all_S = []
    all_C = []
    all_D = []
    all_E = []
    all_B = []
    
    for result in all_results:
        for m in result.video_metrics:
            all_scbi.append(m.scbi)
            all_S.append(m.score_quality)
            all_C.append(m.coverage)
            all_D.append(m.diversity)
            all_E.append(m.efficiency)
            all_B.append(m.balance)
    
    df_corr = pd.DataFrame({
        'SCBI': all_scbi,
        'Score Quality': all_S,
        'Coverage': all_C,
        'Diversity': all_D,
        'Efficiency': all_E,
        'Balance': all_B
    })
    
    correlation_matrix = df_corr.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Metric Correlation Matrix (All Algorithms)', fontsize=14)
    plt.tight_layout()
    plt.savefig(comp_dir / 'metric_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Score-Coverage Scatter for All Algorithms
    plt.figure(figsize=(12, 8))
    
    for result, color in zip(all_results, plt.cm.tab10(np.linspace(0, 1, len(all_results)))):
        S_vals = [m.score_quality for m in result.video_metrics]
        C_vals = [m.coverage for m in result.video_metrics]
        plt.scatter(S_vals, C_vals, label=result.name, alpha=0.5, s=30, color=color)
    
    # Add diagonal line
    lims = [0, 1]
    plt.plot(lims, lims, 'k--', alpha=0.3, label='Perfect Balance', linewidth=2)
    
    plt.xlabel('Score Quality (S)', fontsize=12)
    plt.ylabel('Coverage (C)', fontsize=12)
    plt.title('Score Quality vs Coverage - All Algorithms', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(comp_dir / 'score_coverage_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved comparison plots in {comp_dir}")

# ================================
#  RESULT SAVING
# ================================
def save_results(all_results: List[AlgorithmResults], out_dir: Path):
    """Save detailed results to JSON and CSV"""
    results_dir = out_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save detailed per-video metrics for each algorithm
    for result in all_results:
        detailed_data = []
        for vm in result.video_metrics:
            detailed_data.append({
                'video_id': vm.video_id,
                'scbi': vm.scbi,
                'score_quality': vm.score_quality,
                'coverage': vm.coverage,
                'diversity': vm.diversity,
                'efficiency': vm.efficiency,
                'balance': vm.balance,
                'num_selected': vm.num_selected,
                'num_dense': vm.num_dense,
                'compression_ratio': vm.num_selected / vm.num_dense if vm.num_dense > 0 else 0
            })
        
        # Save to JSON
        json_path = results_dir / f'{result.name}_detailed.json'
        with open(json_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save to CSV
        csv_path = results_dir / f'{result.name}_detailed.csv'
        pd.DataFrame(detailed_data).to_csv(csv_path, index=False)
        
        logging.info(f"Saved detailed results for {result.name}")
    
    # 2. Save aggregate comparison table
    summary_data = []
    for result in all_results:
        scbi_vals = [m.scbi for m in result.video_metrics]
        summary_data.append({
            'Algorithm': result.name,
            'Mean_SCBI': result.mean_scbi,
            'Std_SCBI': np.std(scbi_vals),
            'Min_SCBI': np.min(scbi_vals),
            'Max_SCBI': np.max(scbi_vals),
            'Median_SCBI': np.median(scbi_vals),
            'Mean_Score_Quality': result.mean_score_quality,
            'Mean_Coverage': result.mean_coverage,
            'Mean_Diversity': result.mean_diversity,
            'Mean_Efficiency': result.mean_efficiency,
            'Mean_Balance': result.mean_balance,
            'Num_Videos': len(result.video_metrics)
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Mean_SCBI', ascending=False)
    
    # Save summary
    summary_json_path = results_dir / 'summary_comparison.json'
    df_summary.to_json(summary_json_path, orient='records', indent=2)
    
    summary_csv_path = results_dir / 'summary_comparison.csv'
    df_summary.to_csv(summary_csv_path, index=False)
    
    logging.info(f"Saved summary comparison to {results_dir}")
    
    # 3. Save statistical significance tests (if multiple algorithms)
    if len(all_results) > 1:
        from scipy import stats
        
        pairwise_comparisons = []
        
        for i, result1 in enumerate(all_results):
            for j, result2 in enumerate(all_results):
                if i < j:
                    scbi1 = [m.scbi for m in result1.video_metrics]
                    scbi2 = [m.scbi for m in result2.video_metrics]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scbi1, scbi2)
                    
                    # Compute effect size (Cohen's d)
                    mean_diff = np.mean(scbi1) - np.mean(scbi2)
                    pooled_std = np.sqrt((np.std(scbi1)**2 + np.std(scbi2)**2) / 2)
                    cohens_d = mean_diff / (pooled_std + 1e-8)
                    
                    pairwise_comparisons.append({
                        'Algorithm_1': result1.name,
                        'Algorithm_2': result2.name,
                        'Mean_Diff': mean_diff,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Cohens_D': cohens_d,
                        'Significant_05': 'Yes' if p_value < 0.05 else 'No'
                    })
        
        df_comparisons = pd.DataFrame(pairwise_comparisons)
        
        comp_json_path = results_dir / 'statistical_comparisons.json'
        df_comparisons.to_json(comp_json_path, orient='records', indent=2)
        
        comp_csv_path = results_dir / 'statistical_comparisons.csv'
        df_comparisons.to_csv(comp_csv_path, index=False)
        
        logging.info(f"Saved statistical comparisons to {results_dir}")
    
    return df_summary

# ================================
#  REPORT GENERATION
# ================================
def generate_text_report(all_results: List[AlgorithmResults], out_dir: Path):
    """Generate a human-readable text report"""
    report_path = out_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FRAME SELECTION ALGORITHM EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of algorithms evaluated: {len(all_results)}\n")
        f.write(f"Total videos processed: {len(all_results[0].video_metrics)}\n\n")
        
        # Rankings
        sorted_results = sorted(all_results, key=lambda x: x.mean_scbi, reverse=True)
        
        f.write("OVERALL RANKINGS (by Mean SCBI)\n")
        f.write("-"*80 + "\n")
        for rank, result in enumerate(sorted_results, 1):
            f.write(f"{rank}. {result.name}\n")
            f.write(f"   SCBI: {result.mean_scbi:.4f}\n")
            f.write(f"   Score Quality: {result.mean_score_quality:.4f}\n")
            f.write(f"   Coverage: {result.mean_coverage:.4f}\n")
            f.write(f"   Diversity: {result.mean_diversity:.4f}\n")
            f.write(f"   Efficiency: {result.mean_efficiency:.4f}\n")
            f.write(f"   Balance: {result.mean_balance:.4f}\n\n")
        
        # Detailed breakdown
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED ALGORITHM ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for result in sorted_results:
            f.write(f"Algorithm: {result.name}\n")
            f.write("-"*80 + "\n")
            
            scbi_vals = [m.scbi for m in result.video_metrics]
            
            f.write(f"SCBI Statistics:\n")
            f.write(f"  Mean:   {np.mean(scbi_vals):.4f}\n")
            f.write(f"  Median: {np.median(scbi_vals):.4f}\n")
            f.write(f"  Std:    {np.std(scbi_vals):.4f}\n")
            f.write(f"  Min:    {np.min(scbi_vals):.4f}\n")
            f.write(f"  Max:    {np.max(scbi_vals):.4f}\n\n")
            
            f.write(f"Component Metrics (Mean):\n")
            f.write(f"  Score Quality: {result.mean_score_quality:.4f}\n")
            f.write(f"  Coverage:      {result.mean_coverage:.4f}\n")
            f.write(f"  Diversity:     {result.mean_diversity:.4f}\n")
            f.write(f"  Efficiency:    {result.mean_efficiency:.4f}\n")
            f.write(f"  Balance:       {result.mean_balance:.4f}\n\n")
            
            # Best and worst performing videos
            sorted_videos = sorted(result.video_metrics, key=lambda x: x.scbi, reverse=True)
            
            f.write(f"Top 5 Videos (by SCBI):\n")
            for i, vm in enumerate(sorted_videos[:5], 1):
                f.write(f"  {i}. Video {vm.video_id}: SCBI={vm.scbi:.4f} "
                       f"(S={vm.score_quality:.3f}, C={vm.coverage:.3f})\n")
            
            f.write(f"\nBottom 5 Videos (by SCBI):\n")
            for i, vm in enumerate(sorted_videos[-5:], 1):
                f.write(f"  {i}. Video {vm.video_id}: SCBI={vm.scbi:.4f} "
                       f"(S={vm.score_quality:.3f}, C={vm.coverage:.3f})\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Metric interpretations
        f.write("METRIC INTERPRETATIONS\n")
        f.write("-"*80 + "\n")
        f.write("1. SCBI (Score-Coverage Balanced Index):\n")
        f.write("   Combined metric balancing score quality and temporal coverage.\n")
        f.write("   Higher is better. Range: [0, 1]\n\n")
        
        f.write("2. Score Quality (S):\n")
        f.write("   Ratio of selected frame scores to optimal top-M scores.\n")
        f.write("   Measures how well high-quality frames are captured.\n")
        f.write("   Higher is better. Range: [0, 1]\n\n")
        
        f.write("3. Coverage (C):\n")
        f.write("   Measures temporal distribution uniformity.\n")
        f.write("   Higher means more evenly distributed frames across video.\n")
        f.write("   Higher is better. Range: [0, 1]\n\n")
        
        f.write("4. Diversity (D):\n")
        f.write("   Variance in selected frame scores (normalized).\n")
        f.write("   Higher means more diverse quality selection.\n")
        f.write("   Context-dependent: not always better. Range: [0, 1]\n\n")
        
        f.write("5. Efficiency (E):\n")
        f.write("   Compression ratio (frames selected vs available).\n")
        f.write("   Higher means fewer frames selected (more compression).\n")
        f.write("   Higher is better for efficiency. Range: [0, 1]\n\n")
        
        f.write("6. Balance (B):\n")
        f.write("   Harmonic mean of Score Quality and Coverage.\n")
        f.write("   Penalizes algorithms that excel in only one dimension.\n")
        f.write("   Higher is better. Range: [0, 1]\n\n")
    
    logging.info(f"Generated text report: {report_path}")

# ================================
#  MAIN EVALUATION PIPELINE
# ================================
def main(args):
    # Setup directories
    plot_dir = Path(args.plot_dir)
    log_dir = Path(args.log_dir)
    
    setup_logging(log_dir)
    
    logging.info("="*80)
    logging.info("STARTING FRAME SELECTION EVALUATION")
    logging.info("="*80)
    
    # Load dense frames and scores (shared across all algorithms)
    logging.info(f"Loading dense frames from: {args.frame_path}")
    frames = load_json(Path(args.frame_path))
    
    logging.info(f"Loading dense scores from: {args.score_path}")
    scores = load_json(Path(args.score_path))
    
    logging.info(f"Loaded {len(frames)} videos with dense frames")
    
    # Find all selection JSON files
    selected_dir = Path(args.selected_dir)
    if args.selected_pattern:
        json_files = find_json_files(selected_dir, args.selected_pattern)
    else:
        json_files = find_json_files(selected_dir)
    
    if len(json_files) == 0:
        logging.error(f"No JSON files found in {selected_dir}")
        raise ValueError(f"No JSON files found in {selected_dir}")
    
    logging.info(f"Found {len(json_files)} algorithm files to evaluate")
    
    # Evaluate each algorithm
    all_results = []
    
    for json_file in json_files:
        try:
            result = evaluate_single_algorithm(
                selected_path=json_file,
                frames=frames,
                scores=scores,
                alpha=args.alpha
            )
            all_results.append(result)
            
            # Plot individual algorithm results
            plot_single_algorithm_metrics(result, plot_dir)
            # break
            
        except Exception as e:
            logging.error(f"Failed to evaluate {json_file.name}: {e}")
            continue
    
    if len(all_results) == 0:
        logging.error("No algorithms were successfully evaluated")
        raise ValueError("No algorithms were successfully evaluated")
    
    logging.info(f"Successfully evaluated {len(all_results)} algorithms")
    
    # Generate comparison plots
    if len(all_results) > 1:
        logging.info("Generating algorithm comparison plots...")
        plot_algorithm_comparison(all_results, plot_dir)
    
    # Save results
    logging.info("Saving results...")
    summary_df = save_results(all_results, plot_dir)
    
    # Generate text report
    logging.info("Generating evaluation report...")
    generate_text_report(all_results, plot_dir)
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nTop 3 Algorithms by Mean SCBI:")
    print("-"*80)
    
    sorted_results = sorted(all_results, key=lambda x: x.mean_scbi, reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. {result.name}")
        print(f"   SCBI: {result.mean_scbi:.4f}")
        print(f"   Score Quality: {result.mean_score_quality:.4f}")
        print(f"   Coverage: {result.mean_coverage:.4f}")
        print()
    
    print(f"Results saved to: {plot_dir / 'results'}")
    print(f"Plots saved to: {plot_dir}")
    print(f"Log saved to: {log_dir / 'evaluation_log.txt'}")
    print(f"Report saved to: {plot_dir / 'evaluation_report.txt'}")
    print("="*80 + "\n")
    
    logging.info("Evaluation pipeline completed successfully")

# ================================
#  ARGUMENT PARSER
# ================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Comprehensive Frame Selection Algorithm Evaluation",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Evaluate all JSON files in a directory
#   python script.py --selected_dir ./results/algorithms/
  
#   # Evaluate with custom pattern
#   python script.py --selected_dir ./results/ --selected_pattern "dbfp_*.json"
  
#   # Custom alpha for SCBI weighting
#   python script.py --selected_dir ./results/ --alpha 0.7
#         """
#     )
    
#     # Required arguments
#     parser.add_argument(
#         "--frame_path",
#         type=str,
#         required=True,
#         help="Path to dense frames JSON file"
#     )
    
#     parser.add_argument(
#         "--score_path",
#         type=str,
#         required=True,
#         help="Path to dense scores JSON file"
#     )
    
#     parser.add_argument(
#         "--selected_dir",
#         type=str,
#         required=True,
#         help="Directory containing selection algorithm JSON files"
#     )
    
#     # Optional arguments
#     parser.add_argument(
#         "--selected_pattern",
#         type=str,
#         default="*.json",
#         help="Glob pattern for JSON files (default: *.json)"
#     )
    
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.5,
#         help="SCBI weighting: alpha*S + (1-alpha)*C (default: 0.5)"
#     )
    
#     parser.add_argument(
#         "--plot_dir",
#         type=str,
#         default="./plot",
#         help="Output directory for plots and results (default: ./plot)"
#     )
    
#     parser.add_argument(
#         "--log_dir",
#         type=str,
#         default="./log",
#         help="Output directory for logs (default: ./log)"
#     )
    
#     args = parser.parse_args()
    
#     # Validate paths
#     if not Path(args.frame_path).exists():
#         raise FileNotFoundError(f"Frame path not found: {args.frame_path}")
    
#     if not Path(args.score_path).exists():
#         raise FileNotFoundError(f"Score path not found: {args.score_path}")
    
#     if not Path(args.selected_dir).exists():
#         raise FileNotFoundError(f"Selected directory not found: {args.selected_dir}")
    
#     if not 0 <= args.alpha <= 1:
#         raise ValueError(f"Alpha must be in [0, 1], got {args.alpha}")
    
#     # Run evaluation
#     main(args)

# ================================
#  ARGUMENT PARSER WITH DEFAULTS
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Frame Selection Algorithm Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python score.py
  
  # Evaluate all JSON files in a directory
  python score.py --selected_dir ./results/algorithms/
  
  # Evaluate with custom pattern
  python score.py --selected_dir ./results/ --selected_pattern "dbfp_*.json"
  
  # Custom alpha for SCBI weighting
  python score.py --selected_dir ./results/ --alpha 0.7
        """
    )
    
    # Arguments with defaults
    parser.add_argument(
        "--frame_path",
        type=str,
        default='./output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json',
        help="Path to dense frames JSON file (default: ./output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json)"
    )
    
    parser.add_argument(
        "--score_path",
        type=str,
        default='./output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json',
        help="Path to dense scores JSON file (default: ./output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json)"
    )
    
    parser.add_argument(
        "--selected_dir",
        type=str,
        default="./ALL_DATA/vmme_300_16_8_paramtune/",
        help="Directory containing selection algorithm JSON files (default: ./ALL_DATA/vmme_300_16_8_paramtune/)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--selected_pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="SCBI weighting: alpha*S + (1-alpha)*C (default: 0.5)"
    )
    
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./plot",
        help="Output directory for plots and results (default: ./plot)"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log",
        help="Output directory for logs (default: ./log)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.frame_path).exists():
        raise FileNotFoundError(f"Frame path not found: {args.frame_path}")
    
    if not Path(args.score_path).exists():
        raise FileNotFoundError(f"Score path not found: {args.score_path}")
    
    if not Path(args.selected_dir).exists():
        raise FileNotFoundError(f"Selected directory not found: {args.selected_dir}")
    
    if not 0 <= args.alpha <= 1:
        raise ValueError(f"Alpha must be in [0, 1], got {args.alpha}")
    
    # Run evaluation
    try:
        main(args)
    except Exception as e:
        logging.error(f"Fatal error in main pipeline: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print(f"Check log file for details: {Path(args.log_dir) / 'evaluation_log.txt'}")
        raise