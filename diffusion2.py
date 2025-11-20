"""
Diffusion-Based Frame Propagation (DBFP)
----------------------------------------
Drop-in replacement for AKS keyframe selection.
Implements diffusion smoothing + non-maximum suppression (NMS)
for better-balanced, globally aware keyframe sampling.
"""

import os
import json
import argparse
import numpy as np
import heapq

# ================================================================
# === Utility Functions ==========================================
# ================================================================

def _normalize_safe(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xmin, xmax = np.min(x), np.max(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

# ----------------------------------------------------------------

def _diffuse_1d(scores, alpha=0.85, K=None, weighted=True, beta=5.0, boundary="reflect"):
    """
    Perform 1D diffusion smoothing over frame scores.
    scores : np.array of shape (N,)
    alpha  : retention factor for original score
    K      : number of diffusion iterations (≈ log2(N)+2 if None)
    weighted : use weight exp(-beta*|s_i - s_j|) between neighbors
    """
    s = np.asarray(scores, dtype=float)
    N = s.shape[0]
    if K is None:
        K = max(3, int(np.log2(max(N, 2))) + 2)

    s_cur = s.copy()
    for _ in range(K):
        left = np.roll(s_cur, 1)
        right = np.roll(s_cur, -1)

        # Boundary conditions
        if boundary == "reflect":
            left[0] = s_cur[0]
            right[-1] = s_cur[-1]
        elif boundary == "edge":
            left[0] = s_cur[1] if N > 1 else s_cur[0]
            right[-1] = s_cur[-2] if N > 1 else s_cur[-1]

        if weighted:
            wl = np.exp(-beta * np.abs(s_cur - left))
            wr = np.exp(-beta * np.abs(s_cur - right))
            denom = (wl + wr + 1e-12)
            neigh = (wl * left + wr * right) / denom
        else:
            neigh = 0.5 * (left + right)

        s_cur = alpha * s_cur + (1 - alpha) * neigh

    return _normalize_safe(s_cur)

# ----------------------------------------------------------------

def _nms_temporal(sorted_indices, radius, N):
    """
    Greedy temporal non-maximum suppression.
    Removes frames too close (within radius) to higher scores.
    """
    taken = np.zeros(N, dtype=bool)
    selected = []
    for idx in sorted_indices:
        if not taken[idx]:
            selected.append(idx)
            left = max(0, idx - radius)
            right = min(N, idx + radius + 1)
            taken[left:right] = True
    selected.sort()
    return selected

# ----------------------------------------------------------------

def select_frames_diffusion(scores, fn, M,
                            alpha=0.85, K=None, radius=None,
                            weighted=True, beta=5.0, boundary="reflect"):
    """
    Main DBFP selection function.
    scores : list or np.array of raw relevance scores
    fn      : list of frame ids (same length)
    M       : number of frames to select
    Returns : list of selected frame ids (sorted)
    """
    scores = np.asarray(scores, dtype=float)
    N = len(scores)
    assert N == len(fn), "scores and fn must have same length"

    if N == 0:
        return []
    if M >= N:
        return list(fn)

    # 1. Normalize and diffuse
    s0 = _normalize_safe(scores)
    s_diff = _diffuse_1d(s0, alpha=alpha, K=K, weighted=weighted, beta=beta, boundary=boundary)
    s_final = _normalize_safe(s_diff)

    # 2. Temporal NMS selection
    if radius is None:
        radius = max(1, int(round((N / max(M, 1)) / 2)))

    order = np.argsort(-s_final)
    selected_idx = _nms_temporal(order, radius=radius, N=N)

    # adjust to exactly M frames
    if len(selected_idx) > M:
        selected_idx = sorted(selected_idx, key=lambda i: -s_final[i])[:M]
        selected_idx.sort()
    elif len(selected_idx) < M:
        used = set(selected_idx)
        for i in order:
            if i not in used:
                selected_idx.append(i)
                used.add(i)
                if len(selected_idx) == M:
                    break
        selected_idx.sort()

    return [fn[i] for i in selected_idx]

# ================================================================
# === Main Pipeline ==============================================
# ================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP Keyframe Selector')
    parser.add_argument('--dataset_name', type=str, default='longvideobench')
    parser.add_argument('--extract_feature_model', type=str, default='blip')
    parser.add_argument('--score_path', type=str, default='./outscores/longvideobench/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/longvideobench/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='./selected_frames_dbfp')
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--weighted', action='store_true', help='use weighted diffusion')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=True)
    return parser.parse_args()

# ----------------------------------------------------------------

def main(args):
    os.makedirs(os.path.join(args.output_file, args.dataset_name, args.extract_feature_model), exist_ok=True)
    out_dir = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    print(f"Loaded {len(itm_outs)} videos.")

    outs = []

    for vid, (itm_out, fn_out) in enumerate(zip(itm_outs, fn_outs)):
        scores = np.array(itm_out, dtype=float)
        fns = fn_out
        num_frames = args.max_num_frames
        ratio = args.ratio

        # Downsample by ratio if needed
        if ratio > 1:
            scores = scores[::ratio]
            fns = fns[::ratio]

        if len(scores) < num_frames:
            outs.append(fns)
            continue

        selected = select_frames_diffusion(
            scores=scores,
            fn=fns,
            M=num_frames,
            alpha=args.alpha,
            beta=args.beta,
            weighted=args.weighted,
            K=None,
            boundary="reflect"
        )
        outs.append(selected)

        if (vid + 1) % 50 == 0 or vid == 0:
            print(f"[{vid+1}/{len(itm_outs)}] Processed video with {len(scores)} frames.")

    save_path = os.path.join(out_dir, 'selected_frames.json')
    with open(save_path, 'w') as f:
        json.dump(outs, f)

    print(f"\n✅ DBFP keyframes saved to: {save_path}")

# ================================================================
# === Entry Point ================================================
# ================================================================

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
