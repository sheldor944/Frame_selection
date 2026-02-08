# import json
# import argparse
# import numpy as np
# from collections import defaultdict

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Analyze diffusion hyperparams (alpha, edge_type) on frame scores"
#     )
#     parser.add_argument("--score_path", type=str, default='./output_dense_sampling_new_LV/longvideobench/blip/scores_dense_r2_f2_ram.json',
#                         help="Path to scores.json (list[list[float]])")
#     parser.add_argument("--frame_path", type=str, default='output_dense_sampling_new_LV/longvideobench/blip/frames_dense_r2_f2_ram.json',
#                         help="Path to frames.json (list[list[int]])")
#     parser.add_argument("--num_videos", type=int, default=1337,
#                         help="Number of videos to analyze (from the start)")
#     parser.add_argument("--top_k", type=int, default=16,
#                         help="Top-k frames to compare for Jaccard")
#     parser.add_argument("--alphas", type=str, default="0.6,0.7,0.8, .85, .9, .95",
#                         help="Comma-separated list of alphas, e.g. '0.6,0.7,0.8'")
#     parser.add_argument("--edge_types", type=str, default="uniform,score_diff,temporal",
#                         help="Comma-separated list: 'uniform,score_diff,temporal'")
#     parser.add_argument("--iterations", type=int, default=3,
#                         help="Number of diffusion iterations to run")
#     return parser.parse_args()


# def min_max_normalize(scores: np.ndarray) -> np.ndarray:
#     """Normalize to [0,1] like your DiffusionGraph.__init__."""
#     s_min, s_max = scores.min(), scores.max()
#     if s_max > s_min:
#         return (scores - s_min) / (s_max - s_min)
#     else:
#         # All same value -> flat 0.5
#         return np.ones_like(scores, dtype=np.float64) * 0.5


# def build_edge_weights(scores: np.ndarray,
#                        frame_ids: np.ndarray,
#                        edge_type: str) -> np.ndarray:
#     """Build edge weights between neighbors as in your DiffusionGraph."""
#     N = len(scores)
#     if N <= 1:
#         return np.array([], dtype=np.float64)

#     if edge_type == "uniform":
#         return np.ones(N - 1, dtype=np.float64)

#     elif edge_type == "score_diff":
#         # Here scores are already normalized
#         score_diffs = np.abs(np.diff(scores))
#         weights = 1.0 / (score_diffs + 1e-6)
#         weights = weights / weights.max()
#         return weights.astype(np.float64)

#     elif edge_type == "temporal":
#         temporal_gaps = np.diff(frame_ids.astype(np.float64))
#         weights = 1.0 / (temporal_gaps + 1.0)
#         # If all gaps are equal, this is effectively uniform
#         max_w = weights.max()
#         if max_w > 0:
#             weights = weights / max_w
#         return weights.astype(np.float64)

#     # Fallback
#     return np.ones(N - 1, dtype=np.float64)


# def diffuse_scores(scores: np.ndarray,
#                    frame_ids: np.ndarray,
#                    alpha: float,
#                    edge_type: str,
#                    iterations: int) -> np.ndarray:
#     """
#     Vectorized 1D diffusion, mirroring your DiffusionGraph.diffuse.
#     scores are assumed already normalized.
#     """
#     N = len(scores)
#     if N <= 1 or iterations <= 0:
#         return scores.copy()

#     edge_weights = build_edge_weights(scores, frame_ids, edge_type)
#     s = scores.copy().astype(np.float64)

#     for _ in range(iterations):
#         left_neighbors = np.zeros(N, dtype=np.float64)
#         right_neighbors = np.zeros(N, dtype=np.float64)
#         left_weights = np.zeros(N, dtype=np.float64)
#         right_weights = np.zeros(N, dtype=np.float64)

#         # neighbors & weights
#         left_neighbors[1:] = s[:-1]
#         left_weights[1:] = edge_weights

#         right_neighbors[:-1] = s[1:]
#         right_weights[:-1] = edge_weights

#         total_weights = left_weights + right_weights
#         neighbor_contrib = np.zeros(N, dtype=np.float64)
#         mask = total_weights > 0

#         neighbor_contrib[mask] = (
#             (left_neighbors[mask] * left_weights[mask]
#              + right_neighbors[mask] * right_weights[mask])
#             / total_weights[mask]
#         )

#         s = alpha * s + (1.0 - alpha) * neighbor_contrib

#     return s


# def neighbor_variance(scores: np.ndarray) -> float:
#     """Mean squared difference between neighboring frames."""
#     if len(scores) <= 1:
#         return 0.0
#     diffs = np.diff(scores)
#     return float(np.mean(diffs ** 2))


# def jaccard_top_k(orig_scores: np.ndarray,
#                   diff_scores: np.ndarray,
#                   k: int) -> float:
#     """Jaccard overlap between top-k sets before/after diffusion."""
#     N = len(orig_scores)
#     if N == 0:
#         return 1.0
#     k = min(k, N)

#     orig_top = np.argsort(orig_scores)[-k:]
#     diff_top = np.argsort(diff_scores)[-k:]

#     set_orig = set(orig_top.tolist())
#     set_diff = set(diff_top.tolist())
#     union = set_orig | set_diff
#     inter = set_orig & set_diff

#     if len(union) == 0:
#         return 1.0
#     return len(inter) / len(union)


# def score_contrast(scores: np.ndarray, k: int) -> float:
#     """
#     Mean(top-k) - Mean(others). Higher = peaks stand out more.
#     """
#     N = len(scores)
#     if N == 0:
#         return 0.0
#     k = min(k, N)
#     top_idx = np.argsort(scores)[-k:]
#     top_mask = np.zeros(N, dtype=bool)
#     top_mask[top_idx] = True

#     top_vals = scores[top_mask]
#     rest_vals = scores[~top_mask]
#     if len(rest_vals) == 0:
#         # all frames are "top"
#         return 0.0
#     return float(np.mean(top_vals) - np.mean(rest_vals))


# def main():
#     args = parse_args()

#     alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
#     edge_types = [et.strip() for et in args.edge_types.split(",") if et.strip()]

#     print("Loading scores from:", args.score_path)
#     with open(args.score_path, "r") as f:
#         all_scores = json.load(f)

#     print("Loading frame_ids from:", args.frame_path)
#     with open(args.frame_path, "r") as f:
#         all_frame_ids = json.load(f)

#     num_videos = min(len(all_scores), len(all_frame_ids), args.num_videos)
#     print(f"Total videos in files: scores={len(all_scores)}, frames={len(all_frame_ids)}")
#     print(f"Analyzing first {num_videos} videos\n")

#     # metrics[(edge_type, alpha)] = list of dicts
#     metrics = defaultdict(list)

#     for vid in range(num_videos):
#         scores = np.array(all_scores[vid], dtype=np.float64)
#         frame_ids = np.array(all_frame_ids[vid], dtype=np.int64)

#         if len(scores) != len(frame_ids):
#             # Skip weird entries
#             print(f"  ⚠️ Video {vid}: scores len {len(scores)} != frame_ids len {len(frame_ids)}, skipping.")
#             continue

#         if len(scores) < 2:
#             # Too short, nothing to smooth
#             continue

#         # Normalize as in DiffusionGraph.__init__
#         s_norm = min_max_normalize(scores)
#         nv_orig = neighbor_variance(s_norm)

#         for edge_type in edge_types:
#             for alpha in alphas:
#                 s_diff = diffuse_scores(
#                     scores=s_norm,
#                     frame_ids=frame_ids,
#                     alpha=alpha,
#                     edge_type=edge_type,
#                     iterations=args.iterations,
#                 )

#                 nv_diff = neighbor_variance(s_diff)
#                 nv_ratio = nv_diff / nv_orig if nv_orig > 0 else 1.0

#                 jacc = jaccard_top_k(s_norm, s_diff, args.top_k)
#                 contrast_orig = score_contrast(s_norm, args.top_k)
#                 contrast_diff = score_contrast(s_diff, args.top_k)

#                 metrics[(edge_type, alpha)].append({
#                     "nv_orig": nv_orig,
#                     "nv_diff": nv_diff,
#                     "nv_ratio": nv_ratio,
#                     "jaccard": jacc,
#                     "contrast_orig": contrast_orig,
#                     "contrast_diff": contrast_diff,
#                 })

#     # Summarize
#     print("\n===== SUMMARY =====")
#     print(f"Top-k = {args.top_k}, iterations = {args.iterations}")
#     print("Columns:")
#     print(" edge_type | alpha | videos | NV_ratio (mean±std) | Jaccard (mean±std) | contrast_diff - contrast_orig (mean)")

#     rows = []
#     for (edge_type, alpha), vals in metrics.items():
#         if not vals:
#             continue
#         nv_ratios = np.array([v["nv_ratio"] for v in vals])
#         jaccs = np.array([v["jaccard"] for v in vals])
#         contrast_deltas = np.array([v["contrast_diff"] - v["contrast_orig"] for v in vals])

#         rows.append((
#             edge_type,
#             alpha,
#             len(vals),
#             nv_ratios.mean(), nv_ratios.std(),
#             jaccs.mean(), jaccs.std(),
#             contrast_deltas.mean()
#         ))

#     # Sort rows by edge_type then alpha
#     rows.sort(key=lambda x: (x[0], x[1]))

#     for (edge_type, alpha, count,
#          nv_mean, nv_std,
#          j_mean, j_std,
#          c_delta_mean) in rows:
#         print(
#             f"{edge_type:10s} | {alpha:.2f} | {count:6d} | "
#             f"{nv_mean:6.3f}±{nv_std:5.3f} | "
#             f"{j_mean:6.3f}±{j_std:5.3f} | "
#             f"{c_delta_mean:+7.4f}"
#         )


# if __name__ == "__main__":
#     main()


import json
import argparse
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze diffusion hyperparams (alpha, edge_type, iterations) on frame scores"
    )
    parser.add_argument("--score_path", type=str, default='./output_dense_sampling_new_LV/longvideobench/blip/scores_dense_r2_f2_ram.json',
                        help="Path to scores.json (list[list[float]])")
    parser.add_argument("--frame_path", type=str, default='output_dense_sampling_new_LV/longvideobench/blip/frames_dense_r2_f2_ram.json',
                        help="Path to frames.json (list[list[int]])")
    parser.add_argument("--num_videos", type=int, default=1337,
                        help="Number of videos to analyze (from the start)")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Top-k frames to compare for Jaccard")
    parser.add_argument("--alphas", type=str, default="0.6,0.7,0.8,0.85,0.9,0.95",
                        help="Comma-separated list of alphas, e.g. '0.6,0.7,0.8'")
    parser.add_argument("--edge_types", type=str, default="uniform,score_diff,temporal",
                        help="Comma-separated list: 'uniform,score_diff,temporal'")
    parser.add_argument("--iterations", type=str, default="1,2,3",
                        help="Comma-separated list of diffusion iterations, e.g. '1,2,3'")
    return parser.parse_args()


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] like your DiffusionGraph.__init__."""
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    else:
        # All same value -> flat 0.5
        return np.ones_like(scores, dtype=np.float64) * 0.5


def build_edge_weights(scores: np.ndarray,
                       frame_ids: np.ndarray,
                       edge_type: str) -> np.ndarray:
    """Build edge weights between neighbors as in your DiffusionGraph."""
    N = len(scores)
    if N <= 1:
        return np.array([], dtype=np.float64)

    if edge_type == "uniform":
        return np.ones(N - 1, dtype=np.float64)

    elif edge_type == "score_diff":
        # Here scores are already normalized
        score_diffs = np.abs(np.diff(scores))
        weights = 1.0 / (score_diffs + 1e-6)
        weights = weights / weights.max()
        return weights.astype(np.float64)

    elif edge_type == "temporal":
        temporal_gaps = np.diff(frame_ids.astype(np.float64))
        weights = 1.0 / (temporal_gaps + 1.0)
        # If all gaps are equal, this is effectively uniform
        max_w = weights.max()
        if max_w > 0:
            weights = weights / max_w
        return weights.astype(np.float64)

    # Fallback
    return np.ones(N - 1, dtype=np.float64)


def diffuse_scores(scores: np.ndarray,
                   frame_ids: np.ndarray,
                   alpha: float,
                   edge_type: str,
                   iterations: int) -> np.ndarray:
    """
    Vectorized 1D diffusion, mirroring your DiffusionGraph.diffuse.
    scores are assumed already normalized.
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

        # neighbors & weights
        left_neighbors[1:] = s[:-1]
        left_weights[1:] = edge_weights

        right_neighbors[:-1] = s[1:]
        right_weights[:-1] = edge_weights

        total_weights = left_weights + right_weights
        neighbor_contrib = np.zeros(N, dtype=np.float64)
        mask = total_weights > 0

        neighbor_contrib[mask] = (
            (left_neighbors[mask] * left_weights[mask]
             + right_neighbors[mask] * right_weights[mask])
            / total_weights[mask]
        )

        s = alpha * s + (1.0 - alpha) * neighbor_contrib

    return s


def neighbor_variance(scores: np.ndarray) -> float:
    """Mean squared difference between neighboring frames."""
    if len(scores) <= 1:
        return 0.0
    diffs = np.diff(scores)
    return float(np.mean(diffs ** 2))


def jaccard_top_k(orig_scores: np.ndarray,
                  diff_scores: np.ndarray,
                  k: int) -> float:
    """Jaccard overlap between top-k sets before/after diffusion."""
    N = len(orig_scores)
    if N == 0:
        return 1.0
    k = min(k, N)

    orig_top = np.argsort(orig_scores)[-k:]
    diff_top = np.argsort(diff_scores)[-k:]

    set_orig = set(orig_top.tolist())
    set_diff = set(diff_top.tolist())
    union = set_orig | set_diff
    inter = set_orig & set_diff

    if len(union) == 0:
        return 1.0
    return len(inter) / len(union)


def score_contrast(scores: np.ndarray, k: int) -> float:
    """
    Mean(top-k) - Mean(others). Higher = peaks stand out more.
    """
    N = len(scores)
    if N == 0:
        return 0.0
    k = min(k, N)
    top_idx = np.argsort(scores)[-k:]
    top_mask = np.zeros(N, dtype=bool)
    top_mask[top_idx] = True

    top_vals = scores[top_mask]
    rest_vals = scores[~top_mask]
    if len(rest_vals) == 0:
        # all frames are "top"
        return 0.0
    return float(np.mean(top_vals) - np.mean(rest_vals))


def main():
    args = parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    edge_types = [et.strip() for et in args.edge_types.split(",") if et.strip()]
    iterations_list = [int(it) for it in args.iterations.split(",") if it.strip()]

    print("Loading scores from:", args.score_path)
    with open(args.score_path, "r") as f:
        all_scores = json.load(f)

    print("Loading frame_ids from:", args.frame_path)
    with open(args.frame_path, "r") as f:
        all_frame_ids = json.load(f)

    num_videos = min(len(all_scores), len(all_frame_ids), args.num_videos)
    print(f"Total videos in files: scores={len(all_scores)}, frames={len(all_frame_ids)}")
    print(f"Analyzing first {num_videos} videos\n")

    # metrics[(edge_type, alpha, iterations)] = list of dicts
    metrics = defaultdict(list)

    for vid in range(num_videos):
        scores = np.array(all_scores[vid], dtype=np.float64)
        frame_ids = np.array(all_frame_ids[vid], dtype=np.int64)

        if len(scores) != len(frame_ids):
            # Skip weird entries
            print(f"  ⚠️ Video {vid}: scores len {len(scores)} != frame_ids len {len(frame_ids)}, skipping.")
            continue

        if len(scores) < 2:
            # Too short, nothing to smooth
            continue

        # Normalize as in DiffusionGraph.__init__
        s_norm = min_max_normalize(scores)
        nv_orig = neighbor_variance(s_norm)

        for edge_type in edge_types:
            for alpha in alphas:
                for iterations in iterations_list:
                    s_diff = diffuse_scores(
                        scores=s_norm,
                        frame_ids=frame_ids,
                        alpha=alpha,
                        edge_type=edge_type,
                        iterations=iterations,
                    )

                    nv_diff = neighbor_variance(s_diff)
                    nv_ratio = nv_diff / nv_orig if nv_orig > 0 else 1.0

                    jacc = jaccard_top_k(s_norm, s_diff, args.top_k)
                    contrast_orig = score_contrast(s_norm, args.top_k)
                    contrast_diff = score_contrast(s_diff, args.top_k)

                    metrics[(edge_type, alpha, iterations)].append({
                        "nv_orig": nv_orig,
                        "nv_diff": nv_diff,
                        "nv_ratio": nv_ratio,
                        "jaccard": jacc,
                        "contrast_orig": contrast_orig,
                        "contrast_diff": contrast_diff,
                    })

    # Summarize
    print("\n===== SUMMARY =====")
    print(f"Top-k = {args.top_k}")
    print("Columns:")
    print(" edge_type | alpha | iters | videos | NV_ratio (mean±std) | Jaccard (mean±std) | contrast_diff - contrast_orig (mean)")

    rows = []
    for (edge_type, alpha, iterations), vals in metrics.items():
        if not vals:
            continue
        nv_ratios = np.array([v["nv_ratio"] for v in vals])
        jaccs = np.array([v["jaccard"] for v in vals])
        contrast_deltas = np.array([v["contrast_diff"] - v["contrast_orig"] for v in vals])

        rows.append((
            edge_type,
            alpha,
            iterations,
            len(vals),
            nv_ratios.mean(), nv_ratios.std(),
            jaccs.mean(), jaccs.std(),
            contrast_deltas.mean()
        ))

    # Sort rows by edge_type, alpha, then iterations
    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    for (edge_type, alpha, iterations, count,
         nv_mean, nv_std,
         j_mean, j_std,
         c_delta_mean) in rows:
        print(
            f"{edge_type:10s} | {alpha:.2f} | {iterations:5d} | {count:6d} | "
            f"{nv_mean:6.3f}±{nv_std:5.3f} | "
            f"{j_mean:6.3f}±{j_std:5.3f} | "
            f"{c_delta_mean:+7.4f}"
        )


if __name__ == "__main__":
    main()