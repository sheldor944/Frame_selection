import json
import numpy as np


# ============================================================
# UTILITIES
# ============================================================

def normalize(scores):
    s = np.array(scores, dtype=np.float64)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return np.ones_like(s) * 0.5


# ============================================================
# TIME-AWARE METRICS
# ============================================================

def time_aware_total_variation(frame_ids, scores):
    """
    TV = sum |s[i+1]-s[i]| * (t[i+1]-t[i])
    Represents how much the signal changes per unit time.
    """
    t = np.array(frame_ids, float)
    s = np.array(scores, float)

    if len(s) < 2:
        return 0.0

    dt = np.diff(t)
    ds = np.abs(np.diff(s))

    return float(np.sum(ds * dt))


def time_aware_neighbor_variance(frame_ids, scores):
    """
    Weighted variance of derivative:
        NV = mean( (ds/dt)^2 )
    """
    t = np.array(frame_ids, float)
    s = np.array(scores, float)

    if len(s) < 2:
        return 0.0

    dt = np.diff(t)
    ds = np.diff(s)

    deriv = ds / np.maximum(dt, 1e-9)

    return float(np.mean(deriv ** 2))


def time_aware_run_length(frame_ids, scores, threshold):
    """
    Run lengths measured in TIME, not number of points.
    """
    t = np.array(frame_ids, float)
    s = np.array(scores, float)

    mask = s >= threshold
    N = len(s)

    runs = []
    coverage_time = 0.0

    start_t = None
    last_t = None

    for i in range(N):
        if mask[i]:
            if start_t is None:
                start_t = t[i]
            last_t = t[i]
        else:
            if start_t is not None:
                dur = last_t - start_t
                runs.append(dur)
                coverage_time += dur
                start_t = None

    if start_t is not None:
        dur = last_t - start_t
        runs.append(dur)
        coverage_time += dur

    total_time = t[-1] - t[0]
    if total_time < 1e-9:
        return {"num_runs": 0, "mean_run": 0, "max_run": 0, "coverage": 0}

    if len(runs) == 0:
        return {"num_runs": 0, "mean_run": 0, "max_run": 0, "coverage": 0}

    return {
        "num_runs": len(runs),
        "mean_run": float(np.mean(runs)),
        "max_run": float(np.max(runs)),
        "coverage": float(coverage_time / total_time),
    }


def time_aware_peak_dispersion(frame_ids, scores, k=16):
    """
    Dispersion measured in actual TIME between top-k frames.
    """
    t = np.array(frame_ids, float)
    s = np.array(scores, float)

    N = len(s)
    if N < 2:
        return {"dispersion_mean": 0.0, "dispersion_median": 0.0}

    idx = np.argsort(s)[-min(k, N):]
    idx = np.sort(idx)

    diffs = np.diff(t[idx])

    if len(diffs) == 0:
        return {"dispersion_mean": 0.0, "dispersion_median": 0.0}

    return {
        "dispersion_mean": float(np.mean(diffs)),
        "dispersion_median": float(np.median(diffs)),
    }


def time_aware_autocorrelation(frame_ids, scores, lag_frames=[5,10,20,30,50,80,120]):
    """
    Autocorrelation evaluated at *actual time lags*, using nearest neighbor matches.

    Returns:
        lag τ (in frames) where correlation drops < 0.5
    """
    t = np.array(frame_ids, float)
    s = np.array(scores, float)

    s_centered = s - np.mean(s)
    var = np.var(s)
    if var < 1e-9:
        return 9999  # completely flat

    # Build fast nearest-index lookup
    t_arr = t

    def find_nearest_idx(time):
        return int(np.argmin(np.abs(t_arr - time)))

    corr_vals = []

    for lag in lag_frames:
        vals = []
        for i in range(len(s_centered)):
            t_target = t[i] + lag
            if t_target > t[-1]:
                continue
            j = find_nearest_idx(t_target)
            vals.append(s_centered[i] * s_centered[j])

        if len(vals) == 0:
            corr_vals.append(0.0)
            continue

        corr_vals.append(np.mean(vals) / var)

    # find where corr < 0.5
    for lag, corr in zip(lag_frames, corr_vals):
        if corr < 0.5:
            return float(lag)

    return float(lag_frames[-1])


# ============================================================
# NON-TEMPORAL METRICS (Gini, Entropy same as before)
# ============================================================

def shannon_entropy(scores):
    s = np.array(scores, float)
    s = s - s.min()
    if s.sum() <= 0: return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def gini_coefficient(scores):
    s = np.array(scores, float)
    s = s - s.min()
    if np.allclose(s, 0): return 0.0
    s_sorted = np.sort(s)
    n = len(s_sorted)
    idx = np.arange(1, n + 1)
    return float((2*np.sum(idx * s_sorted)/(n*np.sum(s_sorted))) - (n+1)/n)


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_dataset_time_aware(frames_path, scores_path, top_k=16):
    frames_all = json.load(open(frames_path))
    scores_all = json.load(open(scores_path))

    assert len(frames_all) == len(scores_all)

    N = len(frames_all)
    print("Loaded videos:", N)

    stats_disp = []
    stats_auto = []
    stats_gini = []
    stats_entropy = []
    stats_tv = []
    stats_nv = []

    for vid in range(N):
        t = np.array(frames_all[vid], float)
        s = normalize(scores_all[vid])

        if len(s) < 2:
            continue

        # TIME-AWARE metrics
        disp = time_aware_peak_dispersion(t, s, top_k)['dispersion_mean']
        ac = time_aware_autocorrelation(t, s)
        tv = time_aware_total_variation(t, s)
        nv = time_aware_neighbor_variance(t, s)

        # NON-TEMPORAL
        g = gini_coefficient(s)
        e = shannon_entropy(s)

        stats_disp.append(disp)
        stats_auto.append(ac)
        stats_gini.append(g)
        stats_entropy.append(e)
        stats_tv.append(tv)
        stats_nv.append(nv)

    # Print final stats
    print("\n===== TIME-AWARE RESULTS =====")
    print("Avg Peak Dispersion (frames):", np.mean(stats_disp))
    print("Avg Autocorr Decay (frames):", np.mean(stats_auto))
    print("Avg Gini:", np.mean(stats_gini))
    print("Avg Entropy:", np.mean(stats_entropy))
    print("Avg Time-Aware Total Variation:", np.mean(stats_tv))
    print("Avg Time-Aware Neighbor Variance:", np.mean(stats_nv))


if __name__ == "__main__":
    vmme_uniform_frame='./outscores/videomme/blip/frames.json'
    vmme_uniform_score='./outscores/videomme/blip/scores.json'

    vmme_dense_frame='./output_dense_sampling_new/videomme/blip/frames_dense_r2_f2_ram.json'
    vmme_dense_frame='./output_dense_sampling_new/videomme/blip/scores_dense_r2_f2_ram.json'


    analyze_dataset_time_aware(
        frames_path=vmme_uniform_frame,
        scores_path=vmme_uniform_score,
        top_k=16
    )
