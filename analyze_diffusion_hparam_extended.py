import json
import argparse
import numpy as np
from scipy.stats import kurtosis, skew


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended analysis of per-video score shapes (flat vs spiky vs dense segments)"
    )
    parser.add_argument("--score_path", type=str, default='./outscores/videomme/blip/scores.json',
                        help="Path to scores.json (list[list[float]])")
    parser.add_argument("--frame_path", type=str, default='./outscores/videomme/blip/scores.json',
                        help="Path to frames.json (list[list[int]])")
    parser.add_argument("--num_videos", type=int, default=2700,
                        help="Number of videos to analyze (default: all)")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Top-k used for some peakiness stats")
    return parser.parse_args()


# ==========================================
# CORE UTILS
# ==========================================

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

        lengths.append(scores.size)

        s_norm = min_max_normalize(scores)
        mean_scores.append(float(s_norm.mean()))
        std_scores.append(float(s_norm.std()))
        ranges.append(float(s_norm.max() - s_norm.min()))

        nv = neighbor_variance(s_norm)
        nv_baseline.append(nv)
        tv = total_variation(s_norm)
        tv_baseline.append(tv)

        gaps = topk_gap(s_norm, args.top_k)
        gap_1_2_list.append(gaps["gap_1_2"])
        gap_1_med_list.append(gaps["gap_1_med"])
        topk_minus_all_list.append(gaps["topk_minus_all"])

        frac = fraction_above(s_norm, [0.7, 0.8, 0.9])
        frac_ge_07.append(frac[0.7])
        frac_ge_08.append(frac[0.8])
        frac_ge_09.append(frac[0.9])

        # distribution shape
        entropy_list.append(shannon_entropy(s_norm))
        gini_list.append(gini_coefficient(s_norm))
        kurt_list.append(float(kurtosis(s_norm, fisher=False)))  # Pearson
        skew_list.append(float(skew(s_norm)))

        # run-lengths
        r07 = run_length_stats(s_norm, 0.7)
        runs07_num.append(r07["num_runs"])
        runs07_mean.append(r07["mean_run"])
        runs07_max.append(r07["max_run"])
        runs07_cov.append(r07["coverage"])

        r08 = run_length_stats(s_norm, 0.8)
        runs08_num.append(r08["num_runs"])
        runs08_mean.append(r08["mean_run"])
        runs08_max.append(r08["max_run"])
        runs08_cov.append(r08["coverage"])

        # peaks
        pstats = peak_stats(s_norm)
        num_peaks_list.append(pstats["num_peaks"])
        peak_density_list.append(pstats["peak_density"])

        # spectral roughness
        spectral_rough.append(spectral_roughness(s_norm))

        # ===== NEW STATS =====
        disp = peak_dispersion_stats(s_norm, args.top_k)
        disp_mean_list.append(disp['dispersion_mean'])
        disp_med_list.append(disp['dispersion_median'])
        
        ac_val = autocorrelation_decay(s_norm)
        auto_decay_list.append(ac_val)
        
        kp_val = knee_point_index(s_norm)
        knee_idx_list.append(kp_val)

        # ===== Simple regime flags (heuristics) =====
        is_flat = (ranges[-1] < 0.1) and (tv < 0.5)
        # "sparse peaks": high Gini, big top1-med gap, few frames >=0.8
        is_sparse = (gini_list[-1] > 0.6) and (gap_1_med_list[-1] > 0.3) and (frac_ge_08[-1] < 0.05)
        # "dense segments": non-trivial coverage at 0.8 and long runs
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



# import json
# import argparse
# import numpy as np
# from scipy.stats import kurtosis, skew
# from scipy.interpolate import interp1d


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Temporally-correct analysis of per-video score shapes"
#     )
#     parser.add_argument("--score_path", type=str, default='./output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json',
#                         help="Path to scores.json (list[list[float]])")
#     parser.add_argument("--frame_path", type=str, default='./output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json',
#                         help="Path to frames.json (list[list[int]])")
#     parser.add_argument("--num_videos", type=int, default=2700,
#                         help="Number of videos to analyze (default: all)")
#     parser.add_argument("--top_k", type=int, default=16,
#                         help="Top-k used for some peakiness stats")
#     return parser.parse_args()


# # ==========================================
# # CORE UTILS
# # ==========================================

# def min_max_normalize(scores: np.ndarray) -> np.ndarray:
#     """Normalize scores to [0, 1] range."""
#     s_min, s_max = scores.min(), scores.max()
#     if s_max > s_min:
#         return (scores - s_min) / (s_max - s_min)
#     else:
#         return np.ones_like(scores, dtype=np.float64) * 0.5


# def resample_to_uniform_frame_grid(frame_ids: np.ndarray, scores: np.ndarray) -> tuple:
#     """
#     Resample non-uniform (frame_ids, scores) to a uniform temporal grid.
#     Uses linear interpolation between sampled points.
    
#     KEY: This resamples to EVERY FRAME in the range [min_frame, max_frame].
#     This ensures the analysis is independent of input sampling density.
    
#     Args:
#         frame_ids: Non-uniform frame indices (e.g., [0, 10, 29, 60, ...])
#         scores: Scores at those frame indices
    
#     Returns:
#         (uniform_frame_ids, interpolated_scores)
#         where uniform_frame_ids = [min_frame, min_frame+1, ..., max_frame]
#     """
#     if len(frame_ids) < 2:
#         return frame_ids, scores
    
#     # Sort by frame_id (should already be sorted, but ensure it)
#     sort_idx = np.argsort(frame_ids)
#     frame_ids = frame_ids[sort_idx]
#     scores = scores[sort_idx]
    
#     # Create uniform grid covering EVERY frame
#     start_frame = int(frame_ids[0])
#     end_frame = int(frame_ids[-1])
#     uniform_frames = np.arange(start_frame, end_frame + 1, 1)  # Every single frame
    
#     # Interpolate scores to uniform grid
#     interpolator = interp1d(frame_ids, scores, kind='linear', 
#                             bounds_error=False, fill_value='extrapolate')
#     uniform_scores = interpolator(uniform_frames)
    
#     # Clip to valid range after interpolation
#     uniform_scores = np.clip(uniform_scores, 0.0, 1.0)
    
#     return uniform_frames, uniform_scores


# # ==========================================
# # SAMPLING STATISTICS (Diagnostic)
# # ==========================================

# def sampling_statistics(frame_ids: np.ndarray) -> dict:
#     """
#     Compute statistics about the input sampling pattern.
#     This is diagnostic - helps understand if input is uniform or not.
#     """
#     N = len(frame_ids)
#     if N < 2:
#         return {
#             "num_samples": N,
#             "total_span": 0,
#             "mean_gap": 0.0,
#             "std_gap": 0.0,
#             "min_gap": 0,
#             "max_gap": 0,
#             "uniformity_ratio": 0.0,
#         }
    
#     gaps = np.diff(frame_ids)
#     total_span = int(frame_ids[-1] - frame_ids[0])
#     mean_gap = float(np.mean(gaps))
#     std_gap = float(np.std(gaps))
    
#     uniformity_ratio = std_gap / mean_gap if mean_gap > 0 else 0.0
    
#     return {
#         "num_samples": N,
#         "total_span": total_span,
#         "mean_gap": mean_gap,
#         "std_gap": std_gap,
#         "min_gap": int(np.min(gaps)),
#         "max_gap": int(np.max(gaps)),
#         "uniformity_ratio": uniformity_ratio,
#     }


# # ==========================================
# # SCORE DISTRIBUTION METRICS (Temporal-Invariant)
# # ==========================================

# def shannon_entropy(scores: np.ndarray) -> float:
#     """Entropy of scores treated as a probability distribution."""
#     s = scores - scores.min()
#     s_sum = s.sum()
#     if s_sum <= 0:
#         return 0.0
#     p = s / s_sum
#     p = p[p > 0]
#     return float(-np.sum(p * np.log(p)))


# def gini_coefficient(scores: np.ndarray) -> float:
#     """Gini coefficient for inequality in scores."""
#     s = scores - scores.min()
#     if np.allclose(s, 0):
#         return 0.0
#     s_sorted = np.sort(s)
#     n = len(s_sorted)
#     index = np.arange(1, n + 1)
#     return float((2.0 * np.sum(index * s_sorted) / (n * np.sum(s_sorted))) - (n + 1) / n)


# def topk_gaps(scores: np.ndarray, k: int) -> dict:
#     """Compute gaps between top scores."""
#     N = len(scores)
#     if N == 0:
#         return {"gap_1_2": 0.0, "gap_1_med": 0.0, "topk_minus_all": 0.0}

#     s_sorted = np.sort(scores)

#     top1 = s_sorted[-1]
#     top2 = s_sorted[-2] if N >= 2 else top1
#     med = np.median(scores)

#     k_eff = min(k, N)
#     topk_vals = s_sorted[-k_eff:]
#     mean_topk = float(np.mean(topk_vals))
#     mean_all = float(np.mean(scores))

#     return {
#         "gap_1_2": float(top1 - top2),
#         "gap_1_med": float(top1 - med),
#         "topk_minus_all": float(mean_topk - mean_all),
#     }


# def knee_point_index(scores: np.ndarray) -> int:
#     """Find the 'elbow' or 'knee' in the sorted score curve."""
#     n = len(scores)
#     if n < 3:
#         return n
        
#     y = np.sort(scores)[::-1]  # High to low
#     x = np.arange(n)
    
#     if (x.max() - x.min()) == 0 or (y.max() - y.min()) == 0:
#         return n
        
#     x_norm = (x - x.min()) / (x.max() - x.min())
#     y_norm = (y - y.min()) / (y.max() - y.min())
    
#     start_point = np.array([0, y_norm[0]])
#     end_point = np.array([1, y_norm[-1]])
#     line_vec = end_point - start_point
    
#     vec_from_start = np.stack([x_norm, y_norm], axis=1) - start_point
    
#     cross_prod = np.abs(line_vec[0] * vec_from_start[:, 1] - line_vec[1] * vec_from_start[:, 0])
    
#     return int(np.argmax(cross_prod))


# # ==========================================
# # TEMPORAL VARIATION METRICS (On Uniform Grid)
# # ==========================================

# def temporal_total_variation(scores: np.ndarray, frame_ids: np.ndarray) -> float:
#     """
#     Total variation normalized by temporal span.
#     On uniform frame grid (step=1), this is just mean absolute difference.
#     """
#     if len(scores) < 2:
#         return 0.0
    
#     total_abs_change = np.sum(np.abs(np.diff(scores)))
#     temporal_span = float(frame_ids[-1] - frame_ids[0])
    
#     if temporal_span <= 0:
#         return 0.0
    
#     return float(total_abs_change / temporal_span)


# def temporal_neighbor_variance(scores: np.ndarray) -> float:
#     """
#     Variance of score changes between consecutive frames.
#     On uniform grid (step=1), this directly measures frame-to-frame variation.
#     """
#     if len(scores) < 2:
#         return 0.0
    
#     score_diffs = np.diff(scores)
#     return float(np.var(score_diffs))


# # ==========================================
# # THRESHOLD-BASED METRICS (On Uniform Grid)
# # ==========================================

# def fraction_above_thresholds(scores: np.ndarray, frame_ids: np.ndarray,
#                                thresholds: list) -> dict:
#     """
#     Fraction of temporal duration where scores are above thresholds.
#     On uniform frame grid (step=1), each frame has equal weight = 1 frame.
#     Simple counting is correct.
#     """
#     out = {}
#     N = len(scores)
    
#     if N == 0:
#         for t in thresholds:
#             out[t] = 0.0
#         return out
    
#     # On uniform grid with step=1, simple counting works correctly
#     for t in thresholds:
#         out[t] = float(np.mean(scores >= t))
    
#     return out


# def run_length_statistics(scores: np.ndarray, frame_ids: np.ndarray, 
#                           threshold: float) -> dict:
#     """
#     Compute run-length statistics for regions above threshold.
#     On uniform frame grid (step=1), run lengths directly measure frame duration.
#     """
#     N = len(scores)
#     if N == 0:
#         return {
#             "num_runs": 0,
#             "mean_run_length": 0.0,
#             "max_run_length": 0,
#             "coverage": 0.0
#         }

#     mask = scores >= threshold
#     if not np.any(mask):
#         return {
#             "num_runs": 0,
#             "mean_run_length": 0.0,
#             "max_run_length": 0,
#             "coverage": 0.0
#         }

#     # Find run boundaries using state changes
#     # Add sentinels to detect runs at boundaries
#     padded_mask = np.concatenate(([False], mask, [False]))
#     diff = np.diff(padded_mask.astype(int))
    
#     # Run starts where diff == 1, ends where diff == -1
#     run_starts = np.where(diff == 1)[0]
#     run_ends = np.where(diff == -1)[0]
    
#     # Compute run lengths in frames
#     run_lengths = frame_ids[run_ends - 1] - frame_ids[run_starts] + 1
    
#     # Coverage: fraction of temporal span covered by runs
#     total_span = float(frame_ids[-1] - frame_ids[0] + 1)
#     total_run_frames = float(np.sum(run_lengths))
#     coverage = total_run_frames / total_span if total_span > 0 else 0.0
    
#     return {
#         "num_runs": int(len(run_lengths)),
#         "mean_run_length": float(np.mean(run_lengths)),
#         "max_run_length": int(np.max(run_lengths)),
#         "coverage": float(coverage)
#     }


# # ==========================================
# # PEAK DETECTION METRICS (On Uniform Grid)
# # ==========================================

# def peak_statistics(scores: np.ndarray, frame_ids: np.ndarray) -> dict:
#     """
#     Count local maxima and compute peak density.
#     On uniform frame grid (step=1), peak density is peaks per frame.
#     """
#     N = len(scores)
#     if N < 3:
#         return {
#             "num_peaks": 0,
#             "peak_density": 0.0
#         }
    
#     # Find local maxima (strict: higher than both neighbors)
#     peaks = (scores[1:-1] > scores[:-2]) & (scores[1:-1] > scores[2:])
#     num_peaks = int(peaks.sum())
    
#     temporal_span = float(frame_ids[-1] - frame_ids[0])
#     peak_density = num_peaks / temporal_span if temporal_span > 0 else 0.0
    
#     return {
#         "num_peaks": num_peaks,
#         "peak_density": float(peak_density)
#     }


# # ==========================================
# # SPECTRAL ANALYSIS (On Uniform Grid)
# # ==========================================

# def spectral_roughness(scores: np.ndarray) -> float:
#     """
#     Spectral roughness: ratio of high-frequency to total power.
#     Requires uniform sampling (step=1) to be meaningful.
#     """
#     N = len(scores)
#     if N < 4:
#         return 0.0
    
#     s = scores - scores.mean()
#     fft = np.fft.rfft(s)
#     power = np.abs(fft) ** 2
#     power = power[1:]  # ignore DC
    
#     if power.sum() == 0:
#         return 0.0
    
#     mid = len(power) // 2
#     low = power[:mid].sum()
#     high = power[mid:].sum()
    
#     return float(high / (low + high))


# # ==========================================
# # AUTOCORRELATION (On Uniform Grid)
# # ==========================================

# def autocorrelation_decay(scores: np.ndarray, frame_ids: np.ndarray,
#                          max_lag_frames: int = 100) -> float:
#     """
#     Find the lag (in frame units) where autocorrelation drops below 0.5.
#     Proxy for natural event duration.
#     On uniform frame grid (step=1), lag index = lag in frames.
#     """
#     n = len(scores)
#     if n < 5:
#         return 0.0
    
#     # Center the data
#     s = scores - np.mean(scores)
#     var = np.var(scores)
#     if var < 1e-9:
#         return float(frame_ids[-1] - frame_ids[0])  # Flat = infinite correlation
    
#     # On uniform grid with step=1, lag in indices = lag in frames
#     max_lag = min(max_lag_frames, n - 1)
    
#     for k in range(1, max_lag + 1):
#         cov = np.mean(s[:-k] * s[k:])
#         corr = cov / var
        
#         if corr < 0.5:
#             return float(k)  # k frames
    
#     return float(max_lag)


# # ==========================================
# # PEAK DISPERSION (On Uniform Grid)
# # ==========================================

# def peak_dispersion_statistics(scores: np.ndarray, frame_ids: np.ndarray, 
#                                k: int = 16) -> dict:
#     """
#     Measures how spread out the top-k frames are in actual frame units.
#     Critical for setting suppression_radius.
#     """
#     N = len(scores)
#     if N < 2:
#         return {
#             "dispersion_mean": 0.0,
#             "dispersion_median": 0.0,
#             "dispersion_std": 0.0
#         }
    
#     k_eff = min(k, N)
#     top_indices = np.argsort(scores)[-k_eff:]
    
#     # Sort by temporal order
#     top_frame_ids = frame_ids[top_indices]
#     top_frame_ids_sorted = np.sort(top_frame_ids)
    
#     # Frame-based dispersion
#     frame_diffs = np.diff(top_frame_ids_sorted)
    
#     if len(frame_diffs) == 0:
#         return {
#             "dispersion_mean": 0.0,
#             "dispersion_median": 0.0,
#             "dispersion_std": 0.0
#         }

#     return {
#         "dispersion_mean": float(np.mean(frame_diffs)),
#         "dispersion_median": float(np.median(frame_diffs)),
#         "dispersion_std": float(np.std(frame_diffs))
#     }


# # ==========================================
# # COMPLETE VIDEO ANALYSIS
# # ==========================================

# def analyze_video(scores_raw: np.ndarray, frame_ids_raw: np.ndarray,
#                  top_k: int = 16) -> dict:
#     """
#     Complete analysis of a single video.
    
#     KEY INSIGHT: We resample to uniform frame grid (every frame) ONCE at the start,
#     then all metrics work on that uniform representation.
#     This makes results independent of input sampling density.
    
#     Args:
#         scores_raw: Raw scores from non-uniform sampling
#         frame_ids_raw: Frame indices for those scores
#         top_k: Number of top frames for peak analysis
    
#     Returns:
#         Dictionary of all metrics
#     """
#     result = {}
    
#     # Handle empty or invalid input
#     if scores_raw.size == 0 or scores_raw.size != frame_ids_raw.size:
#         return None
    
#     # Ensure sorted by frame_id
#     sort_idx = np.argsort(frame_ids_raw)
#     frame_ids_raw = frame_ids_raw[sort_idx]
#     scores_raw = scores_raw[sort_idx]
    
#     # ===== STEP 1: Record input sampling statistics (diagnostic) =====
#     input_stats = sampling_statistics(frame_ids_raw)
#     result['input_num_samples'] = input_stats['num_samples']
#     result['input_total_span'] = input_stats['total_span']
#     result['input_mean_gap'] = input_stats['mean_gap']
#     result['input_uniformity_ratio'] = input_stats['uniformity_ratio']
    
#     # ===== STEP 2: Normalize raw scores =====
#     scores_raw_norm = min_max_normalize(scores_raw)
    
#     # ===== STEP 3: Resample to uniform frame grid =====
#     # THIS IS THE KEY STEP - all subsequent metrics work on uniform grid
#     # where each frame has equal spacing (step = 1 frame)
#     frame_ids_uniform, scores_uniform = resample_to_uniform_frame_grid(
#         frame_ids_raw, scores_raw_norm
#     )
    
#     result['uniform_num_frames'] = len(scores_uniform)
#     result['uniform_total_span'] = int(frame_ids_uniform[-1] - frame_ids_uniform[0])
    
#     # ===== STEP 4: Score distribution metrics (temporal-invariant) =====
#     result['score_mean'] = float(scores_uniform.mean())
#     result['score_std'] = float(scores_uniform.std())
#     result['score_range'] = float(scores_uniform.max() - scores_uniform.min())
#     result['score_median'] = float(np.median(scores_uniform))
    
#     result['shannon_entropy'] = shannon_entropy(scores_uniform)
#     result['gini_coefficient'] = gini_coefficient(scores_uniform)
#     result['kurtosis'] = float(kurtosis(scores_uniform, fisher=False))
#     result['skewness'] = float(skew(scores_uniform))
    
#     # Top-k gaps
#     gaps = topk_gaps(scores_uniform, top_k)
#     result['gap_top1_top2'] = gaps['gap_1_2']
#     result['gap_top1_median'] = gaps['gap_1_med']
#     result['gap_topk_all'] = gaps['topk_minus_all']
    
#     result['knee_point_index'] = knee_point_index(scores_uniform)
    
#     # ===== STEP 5: Temporal variation metrics =====
#     result['total_variation'] = temporal_total_variation(scores_uniform, frame_ids_uniform)
#     result['neighbor_variance'] = temporal_neighbor_variance(scores_uniform)
    
#     # ===== STEP 6: Threshold-based metrics =====
#     frac_above = fraction_above_thresholds(scores_uniform, frame_ids_uniform, [0.6, 0.7, 0.8, 0.9])
#     result['frac_above_0.6'] = frac_above[0.6]
#     result['frac_above_0.7'] = frac_above[0.7]
#     result['frac_above_0.8'] = frac_above[0.8]
#     result['frac_above_0.9'] = frac_above[0.9]
    
#     # ===== STEP 7: Run-length statistics =====
#     runs_07 = run_length_statistics(scores_uniform, frame_ids_uniform, 0.7)
#     result['runs_0.7_num'] = runs_07['num_runs']
#     result['runs_0.7_mean_length'] = runs_07['mean_run_length']
#     result['runs_0.7_max_length'] = runs_07['max_run_length']
#     result['runs_0.7_coverage'] = runs_07['coverage']
    
#     runs_08 = run_length_statistics(scores_uniform, frame_ids_uniform, 0.8)
#     result['runs_0.8_num'] = runs_08['num_runs']
#     result['runs_0.8_mean_length'] = runs_08['mean_run_length']
#     result['runs_0.8_max_length'] = runs_08['max_run_length']
#     result['runs_0.8_coverage'] = runs_08['coverage']
    
#     # ===== STEP 8: Peak statistics =====
#     peaks = peak_statistics(scores_uniform, frame_ids_uniform)
#     result['num_peaks'] = peaks['num_peaks']
#     result['peak_density'] = peaks['peak_density']
    
#     # ===== STEP 9: Spectral analysis =====
#     result['spectral_roughness'] = spectral_roughness(scores_uniform)
    
#     # ===== STEP 10: Autocorrelation =====
#     result['autocorr_decay'] = autocorrelation_decay(scores_uniform, frame_ids_uniform, max_lag_frames=100)
    
#     # ===== STEP 11: Peak dispersion =====
#     disp = peak_dispersion_statistics(scores_uniform, frame_ids_uniform, k=top_k)
#     result['peak_dispersion_mean'] = disp['dispersion_mean']
#     result['peak_dispersion_median'] = disp['dispersion_median']
#     result['peak_dispersion_std'] = disp['dispersion_std']
    
#     # ===== STEP 12: Regime classification (heuristic) =====
#     is_flat = (result['score_range'] < 0.1) and (result['total_variation'] < 0.001)
#     is_sparse = (result['gini_coefficient'] > 0.6) and \
#                 (result['gap_top1_median'] > 0.3) and \
#                 (result['frac_above_0.8'] < 0.05)
#     is_dense = (result['runs_0.8_coverage'] > 0.2) and \
#                (result['runs_0.8_mean_length'] >= 50)
    
#     result['regime_flat'] = is_flat
#     result['regime_sparse'] = is_sparse
#     result['regime_dense'] = is_dense
    
#     return result


# # ==========================================
# # AGGREGATION AND PRINTING
# # ==========================================

# def summarize_metric(name: str, values: list, indent: str = "  "):
#     """Print summary statistics for a metric."""
#     arr = np.array(values, dtype=np.float64)
#     if arr.size == 0:
#         print(f"{indent}{name}: no data")
#         return
    
#     print(f"{indent}{name}:")
#     print(f"{indent}  mean   = {arr.mean():.4f}")
#     print(f"{indent}  std    = {arr.std():.4f}")
#     print(f"{indent}  min    = {arr.min():.4f}")
#     for q in [0.25, 0.5, 0.75, 0.9]:
#         val = np.quantile(arr, q)
#         print(f"{indent}  q{int(q*100):02d}    = {val:.4f}")
#     print(f"{indent}  max    = {arr.max():.4f}")
#     print()

# def print_results(all_results: list, args):
#     """Print comprehensive analysis results."""
    
#     if len(all_results) == 0:
#         print("No valid videos to analyze!")
#         return
    
#     print("\n" + "=" * 80)
#     print("TEMPORALLY-INVARIANT VIDEO SCORE ANALYSIS")
#     print("=" * 80)
#     print(f"\nTotal videos analyzed: {len(all_results)}")
#     print(f"Top-k parameter: {args.top_k}")
#     print("\nKEY: All metrics computed on uniform frame grid (every frame).")
#     print("     Results are INDEPENDENT of input sampling density.")
#     print()
    
#     # Extract all metrics into lists
#     metrics = {}
#     for key in all_results[0].keys():
#         if isinstance(all_results[0][key], (int, float, np.integer, np.floating)):
#             metrics[key] = [r[key] for r in all_results]
    
#     # ===== INPUT SAMPLING DIAGNOSTICS =====
#     print("=" * 80)
#     print("INPUT SAMPLING STATISTICS (Diagnostic Only)")
#     print("=" * 80)
#     print("\nThese show the characteristics of your INPUT sampling.")
#     print("High uniformity_ratio = non-uniform input sampling.")
#     print("These do NOT affect the analysis results.\n")
    
#     summarize_metric("Number of input samples", metrics['input_num_samples'])
#     summarize_metric("Input temporal span (frames)", metrics['input_total_span'])
#     summarize_metric("Input mean gap between samples", metrics['input_mean_gap'])
#     summarize_metric("Input uniformity ratio (std/mean, 0=perfectly uniform)", 
#                     metrics['input_uniformity_ratio'])
    
#     # ===== UNIFORM GRID INFO =====
#     print("=" * 80)
#     print("UNIFORM FRAME GRID (Analysis Basis)")
#     print("=" * 80)
#     print("\nAll metrics below are computed on this uniform grid.")
#     print("Each data point represents exactly 1 frame.\n")
    
#     summarize_metric("Number of frames in uniform grid", metrics['uniform_num_frames'])
#     summarize_metric("Temporal span (frames)", metrics['uniform_total_span'])
    
#     # ===== SCORE DISTRIBUTION =====
#     print("=" * 80)
#     print("SCORE DISTRIBUTION METRICS")
#     print("=" * 80)
#     print("\nThese describe the distribution of score values.")
#     print("Temporal-invariant: only depend on score distribution.\n")
    
#     summarize_metric("Score mean", metrics['score_mean'])
#     summarize_metric("Score std", metrics['score_std'])
#     summarize_metric("Score range", metrics['score_range'])
#     summarize_metric("Score median", metrics['score_median'])
#     summarize_metric("Shannon entropy", metrics['shannon_entropy'])
#     summarize_metric("Gini coefficient", metrics['gini_coefficient'])
#     summarize_metric("Kurtosis", metrics['kurtosis'])
#     summarize_metric("Skewness", metrics['skewness'])
    
#     # ===== TOP-K GAPS =====
#     print("=" * 80)
#     print("TOP-K ANALYSIS")
#     print("=" * 80)
#     print("\nGaps between top scores indicate peakiness.\n")
    
#     summarize_metric("Gap: top1 - top2", metrics['gap_top1_top2'])
#     summarize_metric("Gap: top1 - median", metrics['gap_top1_median'])
#     summarize_metric(f"Gap: mean(top{args.top_k}) - mean(all)", metrics['gap_topk_all'])
#     summarize_metric("Knee point index", metrics['knee_point_index'])
    
#     # ===== TEMPORAL VARIATION =====
#     print("=" * 80)
#     print("TEMPORAL VARIATION METRICS")
#     print("=" * 80)
#     print("\nThese measure how much scores change over time.")
#     print("Higher values = more dynamic/spiky content.\n")
    
#     summarize_metric("Total variation (per frame)", metrics['total_variation'])
#     summarize_metric("Neighbor variance (frame-to-frame)", metrics['neighbor_variance'])
    
#     # ===== THRESHOLD ANALYSIS =====
#     print("=" * 80)
#     print("THRESHOLD-BASED METRICS")
#     print("=" * 80)
#     print("\nFraction of video duration where scores exceed thresholds.\n")
    
#     summarize_metric("Fraction above 0.6", metrics['frac_above_0.6'])
#     summarize_metric("Fraction above 0.7", metrics['frac_above_0.7'])
#     summarize_metric("Fraction above 0.8", metrics['frac_above_0.8'])
#     summarize_metric("Fraction above 0.9", metrics['frac_above_0.9'])
    
#     # ===== RUN-LENGTH ANALYSIS =====
#     print("=" * 80)
#     print("RUN-LENGTH STATISTICS (Threshold = 0.7)")
#     print("=" * 80)
#     print("\nContiguous segments where score >= 0.7.")
#     print("Run lengths measured in actual frame counts.\n")
    
#     summarize_metric("Number of runs", metrics['runs_0.7_num'])
#     summarize_metric("Mean run length (frames)", metrics['runs_0.7_mean_length'])
#     summarize_metric("Max run length (frames)", metrics['runs_0.7_max_length'])
#     summarize_metric("Coverage (fraction of video)", metrics['runs_0.7_coverage'])
    
#     print("=" * 80)
#     print("RUN-LENGTH STATISTICS (Threshold = 0.8)")
#     print("=" * 80)
#     print("\nContiguous segments where score >= 0.8.")
#     print("Run lengths measured in actual frame counts.\n")
    
#     summarize_metric("Number of runs", metrics['runs_0.8_num'])
#     summarize_metric("Mean run length (frames)", metrics['runs_0.8_mean_length'])
#     summarize_metric("Max run length (frames)", metrics['runs_0.8_max_length'])
#     summarize_metric("Coverage (fraction of video)", metrics['runs_0.8_coverage'])
    
#     # ===== PEAK ANALYSIS =====
#     print("=" * 80)
#     print("PEAK STATISTICS")
#     print("=" * 80)
#     print("\nLocal maxima in the score signal.")
#     print("Peak density = peaks per frame.\n")
    
#     summarize_metric("Number of peaks", metrics['num_peaks'])
#     summarize_metric("Peak density (peaks per frame)", metrics['peak_density'])
    
#     # ===== SPECTRAL ANALYSIS =====
#     print("=" * 80)
#     print("SPECTRAL ANALYSIS")
#     print("=" * 80)
#     print("\nFrequency domain characteristics.")
#     print("Higher roughness = more high-frequency content (rapid changes).\n")
    
#     summarize_metric("Spectral roughness (high-freq ratio)", metrics['spectral_roughness'])
    
#     # ===== TEMPORAL CHARACTERISTICS =====
#     print("=" * 80)
#     print("TEMPORAL CHARACTERISTICS")
#     print("=" * 80)
#     print("\nTemporal structure of score signal.\n")
    
#     summarize_metric("Autocorrelation decay (frames to 0.5 correlation)", 
#                     metrics['autocorr_decay'])
#     summarize_metric(f"Peak dispersion mean (frames between top-{args.top_k})", 
#                     metrics['peak_dispersion_mean'])
#     summarize_metric(f"Peak dispersion median (frames between top-{args.top_k})", 
#                     metrics['peak_dispersion_median'])
#     summarize_metric(f"Peak dispersion std (frames between top-{args.top_k})", 
#                     metrics['peak_dispersion_std'])
    
#     # ===== REGIME CLASSIFICATION =====
#     print("=" * 80)
#     print("REGIME CLASSIFICATION")
#     print("=" * 80)
#     print("\nHeuristic categorization of videos based on score patterns.\n")
    
#     flat_count = sum(r['regime_flat'] for r in all_results)
#     sparse_count = sum(r['regime_sparse'] for r in all_results)
#     dense_count = sum(r['regime_dense'] for r in all_results)
    
#     total = len(all_results)
    
#     print(f"  Flat videos:          {flat_count:5d} ({100*flat_count/total:5.1f}%)")
#     print(f"    - Criteria: range < 0.1 AND total_variation < 0.001")
#     print(f"    - Interpretation: Nearly uniform scores throughout")
#     print()
    
#     print(f"  Sparse peak videos:   {sparse_count:5d} ({100*sparse_count/total:5.1f}%)")
#     print(f"    - Criteria: gini > 0.6 AND gap_top1_median > 0.3 AND frac_above_0.8 < 0.05")
#     print(f"    - Interpretation: Few isolated high-score moments")
#     print()
    
#     print(f"  Dense segment videos: {dense_count:5d} ({100*dense_count/total:5.1f}%)")
#     print(f"    - Criteria: runs_0.8_coverage > 0.2 AND runs_0.8_mean_length >= 50")
#     print(f"    - Interpretation: Extended regions of high scores")
#     print()
    
#     overlap = sum(r['regime_sparse'] and r['regime_dense'] for r in all_results)
#     if overlap > 0:
#         print(f"  Videos classified as both sparse & dense: {overlap}")
#         print(f"    - These have both isolated peaks AND extended high regions")
#         print()
    
#     other_count = sum(not (r['regime_flat'] or r['regime_sparse'] or r['regime_dense']) 
#                      for r in all_results)
#     print(f"  Other/Mixed:          {other_count:5d} ({100*other_count/total:5.1f}%)")
#     print(f"    - Videos that don't fit clear categories")
#     print()
    
#     # ===== RECOMMENDATIONS =====
#     print("=" * 80)
#     print("RECOMMENDATIONS FOR FRAME SELECTION")
#     print("=" * 80)
#     print("\nBased on average statistics:\n")
    
#     avg_disp_median = np.median(metrics['peak_dispersion_median'])
#     avg_autocorr = np.median(metrics['autocorr_decay'])
#     avg_run_length = np.median(metrics['runs_0.8_mean_length'])
    
#     print(f"  Suppression radius suggestion: {int(avg_disp_median / 2)}-{int(avg_disp_median)} frames")
#     print(f"    - Based on median peak dispersion: {avg_disp_median:.1f} frames")
#     print()
    
#     print(f"  Natural event duration: ~{int(avg_autocorr)} frames")
#     print(f"    - Based on median autocorrelation decay")
#     print()
    
#     print(f"  Typical high-score segment length: ~{int(avg_run_length)} frames")
#     print(f"    - Based on median run length at threshold 0.8")
#     print()
    
#     sparse_pct = 100 * sparse_count / total
#     dense_pct = 100 * dense_count / total
    
#     if sparse_pct > 50:
#         print("  Dataset characteristic: SPARSE")
#         print("    - Most videos have isolated high-score moments")
#         print("    - Recommendation: Use lower suppression radius")
#         print("    - May need fewer frames per video")
#     elif dense_pct > 50:
#         print("  Dataset characteristic: DENSE")
#         print("    - Most videos have extended high-score regions")
#         print("    - Recommendation: Use higher suppression radius")
#         print("    - May need more frames to cover important segments")
#     else:
#         print("  Dataset characteristic: MIXED")
#         print("    - Videos have varying patterns")
#         print("    - Recommendation: Use adaptive strategies per video")
#     print()
    
#     print("=" * 80)
#     print("ANALYSIS COMPLETE")
#     print("=" * 80)
#     print("\n✓ All metrics are temporally-invariant")
#     print("✓ Results are independent of input sampling density")
#     print("✓ Safe to compare across different sampling strategies")
#     print()


# # ==========================================
# # MAIN
# # ==========================================

# def main():
#     args = parse_args()
    
#     print("=" * 80)
#     print("LOADING DATA")
#     print("=" * 80)
#     print(f"Scores: {args.score_path}")
#     print(f"Frames: {args.frame_path}")
    
#     with open(args.score_path, "r") as f:
#         all_scores = json.load(f)
    
#     with open(args.frame_path, "r") as f:
#         all_frame_ids = json.load(f)
    
#     total_videos = min(len(all_scores), len(all_frame_ids))
#     if args.num_videos is not None:
#         total_videos = min(total_videos, args.num_videos)
    
#     print(f"\nTotal videos available: {len(all_scores)}")
#     print(f"Videos to analyze: {total_videos}")
#     print()
    
#     # Analyze all videos
#     all_results = []
    
#     print("Processing videos...")
#     for vid in range(total_videos):
#         if vid % 100 == 0:
#             print(f"  Progress: {vid}/{total_videos} ({100*vid/total_videos:.1f}%)", end='\r')
        
#         scores = np.array(all_scores[vid], dtype=np.float64)
#         frame_ids = np.array(all_frame_ids[vid], dtype=np.int64)
        
#         result = analyze_video(scores, frame_ids, top_k=args.top_k)
        
#         if result is not None:
#             all_results.append(result)
    
#     print(f"  Progress: {total_videos}/{total_videos} (100.0%)")
#     print(f"\n✓ Successfully analyzed {len(all_results)} videos.")
    
#     # Print comprehensive results
#     print_results(all_results, args)


# if __name__ == "__main__":
#     main()