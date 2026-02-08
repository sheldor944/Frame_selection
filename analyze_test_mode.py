import json
import argparse
import numpy as np
from scipy.stats import kurtosis, skew


# output_dense_sampling_new_LV/longvideobench/blip/frames_dense_r2_f2_ram.json
# output_dense_sampling_new_LV/longvideobench/blip/scores_dense_r2_f2_ram.json

# output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json
# output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended analysis of per-video score shapes (flat vs spiky vs dense segments)"
    )
    parser.add_argument("--score_path", type=str, default='./output_dense_sampling_new_LV/longvideobench/blip/scores_dense_r2_f2_ram.json',
                        help="Path to scores.json (list[list[float]])")
    parser.add_argument("--frame_path", type=str, default='./output_dense_sampling_new_LV/longvideobench/blip/frames_dense_r2_f2_ram.json',
                        help="Path to frames.json (list[list[int]])")
    parser.add_argument("--num_videos", type=int, default=2700,
                        help="Number of videos to analyze (default: all)")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Top-k used for some peakiness stats")
    return parser.parse_args()


# ==========================================
# CORE UTILS
# ==========================================



def compute_temporal_stats(scores: np.ndarray, frame_ids: np.ndarray):
    """
    Compute stats that respect actual temporal spacing.
    
    Args:
        scores: Score values
        frame_ids: Actual frame indices (e.g., [0, 29, 58, 62, 64, ...])
    """
    # Sort by frame_id to ensure temporal order
    sort_idx = np.argsort(frame_ids)
    frame_ids = frame_ids[sort_idx]
    scores = scores[sort_idx]
    
    # Temporal differences between consecutive sampled frames
    temporal_gaps = np.diff(frame_ids)
    
    return scores, frame_ids, temporal_gaps


def neighbor_variance_temporal(scores: np.ndarray, temporal_gaps: np.ndarray) -> float:
    """Neighbor variance weighted by temporal distance."""
    if len(scores) < 2:
        return 0.0
    
    score_diffs = np.diff(scores)
    
    # Option 1: Normalize by temporal gap (rate of change)
    rates = score_diffs / np.maximum(temporal_gaps, 1)
    return float(np.mean(rates ** 2))
    
    # Option 2: Weight by temporal gap
    # weights = temporal_gaps / temporal_gaps.sum()
    # return float(np.sum(weights * score_diffs ** 2))


def run_length_stats_temporal(scores: np.ndarray, frame_ids: np.ndarray, threshold: float) -> dict:
    """
    Run-length stats in TEMPORAL space (frame counts, not array positions).
    """
    N = len(scores)
    if N == 0:
        return {"num_runs": 0, "mean_run_frames": 0.0, "max_run_frames": 0, "temporal_coverage": 0.0}
    
    mask = scores >= threshold
    if not np.any(mask):
        return {"num_runs": 0, "mean_run_frames": 0.0, "max_run_frames": 0, "temporal_coverage": 0.0}
    
    runs_in_frames = []
    current_run_start = None
    
    for i, (fid, m) in enumerate(zip(frame_ids, mask)):
        if m:
            if current_run_start is None:
                current_run_start = fid
        else:
            if current_run_start is not None:
                # Run ended - compute temporal length
                run_length = frame_ids[i-1] - current_run_start + 1
                runs_in_frames.append(run_length)
                current_run_start = None
    
    # Handle run that extends to end
    if current_run_start is not None:
        run_length = frame_ids[-1] - current_run_start + 1
        runs_in_frames.append(run_length)
    
    if len(runs_in_frames) == 0:
        return {"num_runs": 0, "mean_run_frames": 0.0, "max_run_frames": 0, "temporal_coverage": 0.0}
    
    # Temporal coverage: what fraction of the VIDEO'S temporal span is above threshold?
    total_temporal_span = frame_ids[-1] - frame_ids[0] + 1
    covered_frames = sum(runs_in_frames)
    
    return {
        "num_runs": len(runs_in_frames),
        "mean_run_frames": float(np.mean(runs_in_frames)),
        "max_run_frames": int(np.max(runs_in_frames)),
        "temporal_coverage": float(covered_frames / total_temporal_span) if total_temporal_span > 0 else 0.0,
    }


def autocorrelation_temporal(scores: np.ndarray, frame_ids: np.ndarray, max_temporal_lag: int = 100) -> float:
    """
    Autocorrelation using temporal lag, not array position lag.
    This requires interpolation or binning.
    """
    if len(scores) < 5:
        return 0.0
    
    # Create a uniform temporal grid
    min_fid, max_fid = frame_ids[0], frame_ids[-1]
    temporal_range = max_fid - min_fid
    
    if temporal_range < max_temporal_lag:
        max_temporal_lag = temporal_range
    
    # Interpolate scores onto uniform grid
    uniform_frame_ids = np.arange(min_fid, max_fid + 1)
    uniform_scores = np.interp(uniform_frame_ids, frame_ids, scores)
    
    # Now compute autocorrelation on uniform grid
    s = uniform_scores - np.mean(uniform_scores)
    var = np.var(uniform_scores)
    
    if var < 1e-9:
        return float(len(uniform_frame_ids))
    
    for k in range(1, min(max_temporal_lag + 1, len(uniform_scores))):
        cov = np.mean(s[:-k] * s[k:])
        corr = cov / var
        if corr < 0.5:
            return float(k)
    
    return float(max_temporal_lag)


def peak_dispersion_stats_temporal(scores: np.ndarray, frame_ids: np.ndarray, k: int = 16) -> dict:
    """
    Measures temporal spread of top-k frames (in frame indices, not array positions).
    """
    N = len(scores)
    if N < 2:
        return {"dispersion_mean_frames": 0.0, "dispersion_median_frames": 0.0}
    
    k_eff = min(k, N)
    top_array_indices = np.argsort(scores)[-k_eff:]
    
    # Get TEMPORAL positions of these top frames
    top_frame_ids = frame_ids[top_array_indices]
    top_frame_ids = np.sort(top_frame_ids)
    
    temporal_diffs = np.diff(top_frame_ids)
    
    if len(temporal_diffs) == 0:
        return {"dispersion_mean_frames": 0.0, "dispersion_median_frames": 0.0}
    
    return {
        "dispersion_mean_frames": float(np.mean(temporal_diffs)),
        "dispersion_median_frames": float(np.median(temporal_diffs))
    }

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    else:
        # completely flat → set to 0.5
        return np.ones_like(scores, dtype=np.float64) * 0.5


def neighbor_variance(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    diffs = np.diff(scores)
    return float(np.mean(diffs ** 2))


def total_variation(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(scores))))


# ==========================================
# DISTRIBUTION SHAPE UTILS
# ==========================================

def shannon_entropy(scores: np.ndarray) -> float:
    """
    Entropy of scores treated as a probability distribution.
    """
    s = scores - scores.min()
    s_sum = s.sum()
    if s_sum <= 0:
        return 0.0
    p = s / s_sum
    # avoid log(0)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def gini_coefficient(scores: np.ndarray) -> float:
    """
    Standard Gini for non-negative scores.
    """
    s = scores - scores.min()
    if np.allclose(s, 0):
        return 0.0
    s_sorted = np.sort(s)
    n = len(s_sorted)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * s_sorted) / (n * np.sum(s_sorted))) - (n + 1) / n)


def topk_gap(scores: np.ndarray, k: int) -> dict:
    """
    Assumes scores are normalized. Returns:
      - gap_top1_top2
      - gap_top1_median
      - mean_topk - mean_all
    """
    N = len(scores)
    if N == 0:
        return {"gap_1_2": 0.0, "gap_1_med": 0.0, "topk_minus_all": 0.0}

    order = np.argsort(scores)
    s_sorted = scores[order]

    top1 = s_sorted[-1]
    top2 = s_sorted[-2] if N >= 2 else top1
    med = np.median(scores)

    k_eff = min(k, N)
    topk_vals = s_sorted[-k_eff:]
    mean_topk = float(np.mean(topk_vals))
    mean_all = float(np.mean(scores))

    return {
        "gap_1_2": float(top1 - top2),
        "gap_1_med": float(top1 - med),
        "topk_minus_all": float(mean_topk - mean_all),
    }


def fraction_above(scores: np.ndarray, thresholds):
    out = {}
    N = len(scores)
    if N == 0:
        for t in thresholds:
            out[t] = 0.0
        return out

    for t in thresholds:
        out[t] = float(np.mean(scores >= t))
    return out


def run_length_stats(scores: np.ndarray, threshold: float) -> dict:
    """
    Compute run-length statistics for segments where scores >= threshold.
    """
    N = len(scores)
    if N == 0:
        return {"num_runs": 0, "mean_run": 0.0, "max_run": 0, "coverage": 0.0}

    mask = scores >= threshold
    if not np.any(mask):
        return {"num_runs": 0, "mean_run": 0.0, "max_run": 0, "coverage": 0.0}

    runs = []
    current = 0
    for m in mask:
        if m:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)

    runs = np.array(runs, dtype=np.int32)
    coverage = float(mask.mean())
    return {
        "num_runs": int(len(runs)),
        "mean_run": float(runs.mean()),
        "max_run": int(runs.max()),
        "coverage": coverage,
    }


def peak_stats(scores: np.ndarray) -> dict:
    """
    Count local maxima and their density.
    """
    N = len(scores)
    if N < 3:
        return {"num_peaks": 0, "peak_density": 0.0}
    s = scores
    # strict local maxima
    peaks = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:])
    num_peaks = int(peaks.sum())
    return {
        "num_peaks": num_peaks,
        "peak_density": float(num_peaks / N),
    }


def spectral_roughness(scores: np.ndarray) -> float:
    """
    Very crude 'high-frequency energy ratio' using FFT.
    """
    N = len(scores)
    if N < 4:
        return 0.0
    s = scores - scores.mean()
    fft = np.fft.rfft(s)
    power = np.abs(fft) ** 2
    # ignore DC
    power = power[1:]
    if power.sum() == 0:
        return 0.0
    # split low vs high frequency halves
    mid = len(power) // 2
    low = power[:mid].sum()
    high = power[mid:].sum()
    return float(high / (low + high))


# ==========================================
# NEW: TEMPORAL TOPOLOGY UTILS
# ==========================================

def peak_dispersion_stats(scores: np.ndarray, k: int = 16) -> dict:
    """
    Measures how spread out the top-k frames are.
    Crucial for setting suppression_radius.
    """
    N = len(scores)
    if N < 2:
        return {"dispersion_mean": 0.0, "dispersion_median": 0.0}
    
    # Get indices of top k scores
    k_eff = min(k, N)
    # argsort gives indices of low-to-high, take last k
    top_indices = np.argsort(scores)[-k_eff:] 
    top_indices = np.sort(top_indices) # Now sorted by time
    
    # Calculate distances between consecutive top frames
    diffs = np.diff(top_indices)
    
    if len(diffs) == 0:
        return {"dispersion_mean": 0.0, "dispersion_median": 0.0}

    return {
        "dispersion_mean": float(np.mean(diffs)),
        "dispersion_median": float(np.median(diffs))
    }


def autocorrelation_decay(scores: np.ndarray, max_lag: int = 100) -> float:
    """
    Finds the lag 'k' where autocorrelation drops below 0.5.
    Proxy for 'Natural Event Duration'.
    """
    n = len(scores)
    if n < 5:
        return 0.0
    
    # Center the data
    s = scores - np.mean(scores)
    var = np.var(scores)
    if var < 1e-9:
        return float(n) # Completely flat, infinite correlation
        
    # Compute ACF for first max_lag lags
    max_lag = min(max_lag, n - 1)
    
    for k in range(1, max_lag + 1):
        # Autocorrelation at lag k
        cov = np.mean(s[:-k] * s[k:])
        corr = cov / var
        
        if corr < 0.5:
            # Linear interpolation for precision
            # prev_corr >= 0.5, current_corr < 0.5
            return float(k)
        
    return float(max_lag)


def knee_point_index(scores: np.ndarray) -> int:
    """
    Finds the 'elbow' or 'knee' in the sorted score curve.
    Uses geometric method (max distance to line connecting start and end).
    """
    n = len(scores)
    if n < 3:
        return n
        
    y = np.sort(scores)[::-1] # High to low
    x = np.arange(n)
    
    # Normalize x and y to [0,1] for geometry
    if (x.max() - x.min()) == 0 or (y.max() - y.min()) == 0:
        return n
        
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Vector from start (0, y0) to end (1, yN)
    start_point = np.array([0, y_norm[0]])
    end_point = np.array([1, y_norm[-1]])
    line_vec = end_point - start_point
    
    # Vector from start to all points
    vec_from_start = np.stack([x_norm, y_norm], axis=1) - start_point
    
    # Cross product (2D) to find distance
    cross_prod = np.abs(line_vec[0] * vec_from_start[:, 1] - line_vec[1] * vec_from_start[:, 0])
    
    return int(np.argmax(cross_prod))


def total_variation_temporal(scores: np.ndarray, temporal_gaps: np.ndarray) -> float:
    """Total variation normalized by temporal distance."""
    if len(scores) < 2:
        return 0.0
    
    score_diffs = np.abs(np.diff(scores))
    
    # Normalize by temporal gaps to get rate of change
    rates = score_diffs / np.maximum(temporal_gaps, 1)
    
    return float(np.sum(rates))


def peak_stats_temporal(scores: np.ndarray, temporal_gaps: np.ndarray) -> dict:
    """
    Count local maxima, considering temporal spacing.
    A peak is only counted if it's separated by sufficient temporal distance.
    """
    N = len(scores)
    if N < 3:
        return {"num_peaks": 0, "peak_density": 0.0}
    
    s = scores
    # Strict local maxima (same as before, array-based)
    peaks = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:])
    num_peaks = int(peaks.sum())
    
    # Temporal density: peaks per unit time
    if len(temporal_gaps) > 0:
        total_temporal_span = np.sum(temporal_gaps)
        temporal_density = num_peaks / total_temporal_span if total_temporal_span > 0 else 0.0
    else:
        temporal_density = 0.0
    
    return {
        "num_peaks": num_peaks,
        "peak_density": temporal_density,  # peaks per frame of original video
    }


def spectral_roughness_temporal(scores: np.ndarray, frame_ids: np.ndarray) -> float:
    """
    FFT-based roughness on uniformly interpolated temporal grid.
    """
    N = len(scores)
    if N < 4:
        return 0.0
    
    # Interpolate to uniform grid
    min_fid, max_fid = frame_ids[0], frame_ids[-1]
    if max_fid == min_fid:
        return 0.0
    
    # Create uniform grid with reasonable resolution
    num_uniform_points = max(N, max_fid - min_fid + 1)
    num_uniform_points = min(num_uniform_points, 10000)  # Cap for performance
    
    uniform_frame_ids = np.linspace(min_fid, max_fid, num_uniform_points)
    uniform_scores = np.interp(uniform_frame_ids, frame_ids, scores)
    
    # Now apply FFT
    s = uniform_scores - uniform_scores.mean()
    fft = np.fft.rfft(s)
    power = np.abs(fft) ** 2
    
    # Ignore DC component
    power = power[1:]
    if power.sum() == 0:
        return 0.0
    
    # Split into low vs high frequency
    mid = len(power) // 2
    low = power[:mid].sum()
    high = power[mid:].sum()
    
    return float(high / (low + high))


def neighbor_variance_temporal(scores: np.ndarray, temporal_gaps: np.ndarray) -> float:
    """
    Neighbor variance weighted by temporal distance.
    Measures rate of change per frame.
    """
    if len(scores) < 2:
        return 0.0
    
    score_diffs = np.diff(scores)
    
    # Rate of change per frame
    rates = score_diffs / np.maximum(temporal_gaps, 1)
    
    return float(np.mean(rates ** 2))


def run_length_stats_temporal(scores: np.ndarray, frame_ids: np.ndarray, threshold: float) -> dict:
    """
    Run-length statistics in temporal frame space.
    """
    N = len(scores)
    if N == 0:
        return {
            "num_runs": 0,
            "mean_run_frames": 0.0,
            "max_run_frames": 0,
            "temporal_coverage": 0.0
        }
    
    mask = scores >= threshold
    if not np.any(mask):
        return {
            "num_runs": 0,
            "mean_run_frames": 0.0,
            "max_run_frames": 0,
            "temporal_coverage": 0.0
        }
    
    runs_in_frames = []
    current_run_start_idx = None
    
    for i, m in enumerate(mask):
        if m:
            if current_run_start_idx is None:
                current_run_start_idx = i
        else:
            if current_run_start_idx is not None:
                # Run ended - compute temporal length
                run_start_frame = frame_ids[current_run_start_idx]
                run_end_frame = frame_ids[i - 1]
                run_length = run_end_frame - run_start_frame + 1
                runs_in_frames.append(run_length)
                current_run_start_idx = None
    
    # Handle run extending to end
    if current_run_start_idx is not None:
        run_start_frame = frame_ids[current_run_start_idx]
        run_end_frame = frame_ids[-1]
        run_length = run_end_frame - run_start_frame + 1
        runs_in_frames.append(run_length)
    
    if len(runs_in_frames) == 0:
        return {
            "num_runs": 0,
            "mean_run_frames": 0.0,
            "max_run_frames": 0,
            "temporal_coverage": 0.0
        }
    
    # Temporal coverage
    total_temporal_span = frame_ids[-1] - frame_ids[0] + 1
    covered_frames = sum(runs_in_frames)
    
    return {
        "num_runs": len(runs_in_frames),
        "mean_run_frames": float(np.mean(runs_in_frames)),
        "max_run_frames": int(np.max(runs_in_frames)),
        "temporal_coverage": float(covered_frames / total_temporal_span) if total_temporal_span > 0 else 0.0,
    }


def peak_dispersion_stats_temporal(scores: np.ndarray, frame_ids: np.ndarray, k: int = 16) -> dict:
    """
    Temporal dispersion of top-k scoring frames.
    """
    N = len(scores)
    if N < 2:
        return {"dispersion_mean_frames": 0.0, "dispersion_median_frames": 0.0}
    
    k_eff = min(k, N)
    top_array_indices = np.argsort(scores)[-k_eff:]
    
    # Get temporal frame IDs of top scores
    top_frame_ids = frame_ids[top_array_indices]
    top_frame_ids = np.sort(top_frame_ids)
    
    temporal_diffs = np.diff(top_frame_ids)
    
    if len(temporal_diffs) == 0:
        return {"dispersion_mean_frames": 0.0, "dispersion_median_frames": 0.0}
    
    return {
        "dispersion_mean_frames": float(np.mean(temporal_diffs)),
        "dispersion_median_frames": float(np.median(temporal_diffs))
    }


def autocorrelation_temporal(scores: np.ndarray, frame_ids: np.ndarray, max_temporal_lag: int = 100) -> float:
    """
    Autocorrelation decay on uniform temporal grid.
    """
    if len(scores) < 5:
        return 0.0
    
    min_fid, max_fid = frame_ids[0], frame_ids[-1]
    temporal_range = max_fid - min_fid
    
    if temporal_range == 0:
        return 0.0
    
    if temporal_range < max_temporal_lag:
        max_temporal_lag = int(temporal_range)
    
    # Interpolate to uniform grid
    uniform_frame_ids = np.arange(min_fid, max_fid + 1)
    uniform_scores = np.interp(uniform_frame_ids, frame_ids, scores)
    
    s = uniform_scores - np.mean(uniform_scores)
    var = np.var(uniform_scores)
    
    if var < 1e-9:
        return float(len(uniform_frame_ids))
    
    for k in range(1, min(max_temporal_lag + 1, len(uniform_scores))):
        cov = np.mean(s[:-k] * s[k:])
        corr = cov / var
        if corr < 0.5:
            return float(k)
    
    return float(max_temporal_lag)



# ==========================================
# MAIN
# ==========================================

def summarize_array(name, arr):
    arr = np.array(arr, dtype=np.float64)
    if arr.size == 0:
        print(f"{name}: no data")
        return
    print(f"{name}:")
    print(f"  mean = {arr.mean():.4f}")
    print(f"  std  = {arr.std():.4f}")
    for q in [0.25, 0.5, 0.75, 0.9]:
        val = np.quantile(arr, q)
        print(f"  q{int(q*100):2d}  = {val:.4f}")
    print("")


def main():
    args = parse_args()

    print("Loading scores from:", args.score_path)
    with open(args.score_path, "r") as f:
        all_scores = json.load(f)

    print("Loading frame_ids from:", args.frame_path)
    with open(args.frame_path, "r") as f:
        all_frame_ids = json.load(f)

    total_videos = min(len(all_scores), len(all_frame_ids))
    if args.num_videos is not None:
        total_videos = min(total_videos, args.num_videos)

    print(f"Total videos available: {len(all_scores)}, analyzing: {total_videos}\n")

    # Collections
    lengths = []
    ranges = []
    nv_baseline = []
    tv_baseline = []
    mean_scores = []
    std_scores = []

    gap_1_2_list = []
    gap_1_med_list = []
    topk_minus_all_list = []

    frac_ge_07 = []
    frac_ge_08 = []
    frac_ge_09 = []

    entropy_list = []
    gini_list = []
    kurt_list = []
    skew_list = []

    # run-length (0.7 / 0.8)
    runs07_num = []
    runs07_mean = []
    runs07_max = []
    runs07_cov = []

    runs08_num = []
    runs08_mean = []
    runs08_max = []
    runs08_cov = []

    # peaks
    num_peaks_list = []
    peak_density_list = []

    spectral_rough = []
    
    # NEW STATS
    disp_mean_list = []
    disp_med_list = []
    auto_decay_list = []
    knee_idx_list = []

    # regime flags
    flat_flags = []
    sparse_flags = []
    dense_flags = []

    for vid in range(total_videos):
        scores = np.array(all_scores[vid], dtype=np.float64)
        frame_ids = np.array(all_frame_ids[vid], dtype=np.int64)

        if scores.size == 0 or scores.size != frame_ids.size:
            continue

        # ===== STEP 1: Sort by temporal order =====
        sort_idx = np.argsort(frame_ids)
        frame_ids = frame_ids[sort_idx]
        scores = scores[sort_idx]
        
        # Compute temporal gaps between consecutive frames
        temporal_gaps = np.diff(frame_ids) if len(frame_ids) > 1 else np.array([])

        # ===== STEP 2: Basic stats (temporal-agnostic) =====
        lengths.append(scores.size)
        
        s_norm = min_max_normalize(scores)
        mean_scores.append(float(s_norm.mean()))
        std_scores.append(float(s_norm.std()))
        ranges.append(float(s_norm.max() - s_norm.min()))

        # ===== STEP 3: Temporal-AWARE stats (use frame_ids/temporal_gaps) =====
        nv = neighbor_variance_temporal(s_norm, temporal_gaps)
        nv_baseline.append(nv)
        
        tv = total_variation_temporal(s_norm, temporal_gaps)
        tv_baseline.append(tv)

        # ===== STEP 4: Distribution shape stats (temporal-agnostic, OK as-is) =====
        gaps = topk_gap(s_norm, args.top_k)
        gap_1_2_list.append(gaps["gap_1_2"])
        gap_1_med_list.append(gaps["gap_1_med"])
        topk_minus_all_list.append(gaps["topk_minus_all"])

        frac = fraction_above(s_norm, [0.7, 0.8, 0.9])
        frac_ge_07.append(frac[0.7])
        frac_ge_08.append(frac[0.8])
        frac_ge_09.append(frac[0.9])

        entropy_list.append(shannon_entropy(s_norm))
        gini_list.append(gini_coefficient(s_norm))
        kurt_list.append(float(kurtosis(s_norm, fisher=False)))
        skew_list.append(float(skew(s_norm)))

        # ===== STEP 5: Run-length stats (temporal-AWARE) =====
        r07 = run_length_stats_temporal(s_norm, frame_ids, 0.7)
        runs07_num.append(r07["num_runs"])
        runs07_mean.append(r07["mean_run_frames"])
        runs07_max.append(r07["max_run_frames"])
        runs07_cov.append(r07["temporal_coverage"])

        r08 = run_length_stats_temporal(s_norm, frame_ids, 0.8)
        runs08_num.append(r08["num_runs"])
        runs08_mean.append(r08["mean_run_frames"])
        runs08_max.append(r08["max_run_frames"])
        runs08_cov.append(r08["temporal_coverage"])

        # ===== STEP 6: Peak stats (temporal-AWARE) =====
        pstats = peak_stats_temporal(s_norm, temporal_gaps)
        num_peaks_list.append(pstats["num_peaks"])
        peak_density_list.append(pstats["peak_density"])

        # ===== STEP 7: Spectral roughness (temporal-AWARE, needs uniform grid) =====
        spectral_rough.append(spectral_roughness_temporal(s_norm, frame_ids))

        # ===== STEP 8: New temporal topology stats (temporal-AWARE) =====
        disp = peak_dispersion_stats_temporal(s_norm, frame_ids, args.top_k)
        disp_mean_list.append(disp['dispersion_mean_frames'])
        disp_med_list.append(disp['dispersion_median_frames'])
        
        ac_val = autocorrelation_temporal(s_norm, frame_ids)
        auto_decay_list.append(ac_val)
        
        kp_val = knee_point_index(s_norm)  # This one is OK, works on sorted scores
        knee_idx_list.append(kp_val)

        # ===== STEP 9: Regime flags =====
        is_flat = (ranges[-1] < 0.1) and (tv < 0.5)
        is_sparse = (gini_list[-1] > 0.6) and (gap_1_med_list[-1] > 0.3) and (frac_ge_08[-1] < 0.05)
        is_dense = (runs08_cov[-1] > 0.2) and (runs08_mean[-1] >= 5)

        flat_flags.append(is_flat)
        sparse_flags.append(is_sparse)
        dense_flags.append(is_dense)
    print("===== DATASET-WIDE SCORE SHAPE STATS =====\n")

    summarize_array("Video length (#frames)", lengths)
    summarize_array("Normalized score range (max - min)", ranges)
    summarize_array("Baseline neighbor variance", nv_baseline)
    summarize_array("Total variation", tv_baseline)
    summarize_array("Score std (normalized)", std_scores)

    summarize_array("Gap (top1 - top2)", gap_1_2_list)
    summarize_array("Gap (top1 - median)", gap_1_med_list)
    summarize_array("Mean(top-k) - Mean(all)", topk_minus_all_list)

    summarize_array("Fraction of frames with score >= 0.7", frac_ge_07)
    summarize_array("Fraction of frames with score >= 0.8", frac_ge_08)
    summarize_array("Fraction of frames with score >= 0.9", frac_ge_09)

    summarize_array("Shannon entropy of scores", entropy_list)
    summarize_array("Gini coefficient of scores", gini_list)
    summarize_array("Kurtosis (Pearson)", kurt_list)
    summarize_array("Skewness", skew_list)

    summarize_array("Num local peaks", num_peaks_list)
    summarize_array("Peak density (peaks / N)", peak_density_list)

    summarize_array("Run stats τ=0.7: num_runs", runs07_num)
    summarize_array("Run stats τ=0.7: mean_run", runs07_mean)
    summarize_array("Run stats τ=0.7: max_run", runs07_max)
    summarize_array("Run stats τ=0.7: coverage", runs07_cov)

    summarize_array("Run stats τ=0.8: num_runs", runs08_num)
    summarize_array("Run stats τ=0.8: mean_run", runs08_mean)
    summarize_array("Run stats τ=0.8: max_run", runs08_max)
    summarize_array("Run stats τ=0.8: coverage", runs08_cov)

    summarize_array("Spectral roughness (high-freq power ratio)", spectral_rough)
    
    print("\n===== TEMPORAL TOPOLOGY (NEW) =====")
    print(" (Helps select: radius, alpha, optimize_remaining)\n")
    summarize_array("Peak Dispersion (Mean dist between top-k)", disp_mean_list)
    summarize_array("Peak Dispersion (Median dist between top-k)", disp_med_list)
    summarize_array("Autocorrelation Decay (Natural Event Width)", auto_decay_list)
    summarize_array("Knee Point Index (Effective Signal Count)", knee_idx_list)

    flat_flags = np.array(flat_flags, dtype=bool)
    sparse_flags = np.array(sparse_flags, dtype=bool)
    dense_flags = np.array(dense_flags, dtype=bool)
    n = len(flat_flags)

    if n > 0:
        print("\n===== CATEGORICAL REGIMES (ROUGH HEURISTICS) =====")
        print(f"Total analyzed videos: {n}")
        print(f"  'Flat' videos:   {flat_flags.sum()} ({100*flat_flags.mean():.1f}%)")
        print(f"  'Sparse peaks':  {sparse_flags.sum()} ({100*sparse_flags.mean():.1f}%)")
        print(f"  'Dense segments':{dense_flags.sum()} ({100*dense_flags.mean():.1f}%)")
        both_sparse_dense = sparse_flags & dense_flags
        if both_sparse_dense.sum() > 0:
            print(f"    (Overlap sparse & dense flags: {both_sparse_dense.sum()} videos)")
    print("Done.")


if __name__ == "__main__":
    main()

