import heapq
import json
import numpy as np
import argparse
import os
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Frame Selection Algorithms')

    parser.add_argument('--dataset_name', type=str, default='videomme', help='Dataset name (e.g., videomme, longvideobench)')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
    parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='./selected_frames')
    parser.add_argument('--method', type=str, default='random', choices=['meanstd', 'random'],
                        help='frame selection method to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    return parser.parse_args()

# ---------------- Original meanstd logic ----------------
def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]

        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            split_scores.extend([
                dict(score=score[:mid], depth=depth+1),
                dict(score=score[mid:], depth=depth+1)
            ])
            split_fn.extend([fn[:mid], fn[mid:]])
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score, all_split_fn = [], []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn


# ---------------- New RANDOM SELECTION logic ----------------
def random_selection(frames, max_num_frames, seed=42):
    """
    Select start and end frames + random frames from the middle.
    """
    random.seed(seed)

    total_frames = len(frames)
    if total_frames <= max_num_frames:
        return frames  # not enough frames to sample

    # Always include start and end
    selected = [frames[0], frames[-1]]

    # Remaining frames to sample
    remaining_frames = frames[1:3]

    num_random = max_num_frames - 2
    if num_random > 0:
        sampled = random.sample(remaining_frames, num_random)
        selected.extend(sampled)

    selected = sorted(selected)
    return selected


# ---------------- Main Driver ----------------
def main(args):
    os.makedirs(os.path.join(args.output_file, args.dataset_name, args.extract_feature_model), exist_ok=True)
    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    outs = []
    for itm_out, fn_out in zip(itm_outs, fn_outs):
        nums = int(len(itm_out) / args.ratio)
        new_score = [itm_out[num * args.ratio] for num in range(nums)]
        new_fnum = [fn_out[num * args.ratio] for num in range(nums)]

        if args.method == 'meanstd':
            num = args.max_num_frames
            if len(new_score) >= num:
                normalized_data = (new_score - np.min(new_score)) / (np.max(new_score) - np.min(new_score))
                a, b = meanstd(len(new_score), [dict(score=normalized_data, depth=0)], num, [new_fnum],
                               args.t1, args.t2, args.all_depth)
                out = []
                for s, f in zip(a, b):
                    f_num = max(1, int(num / 2 ** (s['depth'])))
                    f_num = min(f_num, len(s['score']))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
                out.sort()
                outs.append(out[:num])
            else:
                indices = np.linspace(0, len(new_fnum) - 1, args.max_num_frames, dtype=int)
                outs.append([new_fnum[i] for i in indices])
        else:  # RANDOM SELECTION
            out = random_selection(new_fnum, args.max_num_frames, seed=args.seed)
            outs.append(out)

    save_path = os.path.join(out_score_path, f'selected_frames_{args.method}.json')
    with open(save_path, 'w') as f:
        json.dump(outs, f, indent=2)
    print(f"✅ Saved selected frames to {save_path}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
