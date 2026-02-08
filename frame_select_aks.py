import heapq
import json
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='longvideobench')
    parser.add_argument('--extract_feature_model', type=str, default='clip')
    parser.add_argument('--score_path', type=str, default='./outscores/longvideobench/clip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/longvideobench/clip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=4)
    parser.add_argument('--output_file', type=str, default='./selected_frames')

    # >>> FIX 1: add output_name argument
    parser.add_argument('--output_name', type=str, default='selected_frames.json')

    return parser.parse_args()


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
            mid = len(score)//2
            split_scores.append(dict(score=score[:mid], depth=depth+1))
            split_scores.append(dict(score=score[mid:], depth=depth+1))
            split_fn.append(fn[:mid])
            split_fn.append(fn[mid:])

        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if split_scores:
        child_scores, child_fns = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        child_scores, child_fns = [], []

    return no_split_scores + child_scores, no_split_fn + child_fns


def main(args):
    outs = []

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    # Ensure output directories exist
    base_out = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(base_out, exist_ok=True)

    for itm_out, fn_out in zip(itm_outs, fn_outs):
        nums = int(len(itm_out) / args.ratio)
        score = [itm_out[i * args.ratio] for i in range(nums)]
        fn = [fn_out[i * args.ratio] for i in range(nums)]

        if len(score) >= args.max_num_frames:
            normalized = (score - np.min(score)) / (np.max(score) - np.min(score))
            segments, frame_segments = meanstd(
                len(score),
                [dict(score=normalized, depth=0)],
                args.max_num_frames,
                [fn],
                args.t1,
                args.t2,
                args.all_depth
            )

            out = []
            for s, f in zip(segments, frame_segments):
                f_num = int(args.max_num_frames / (2 ** s['depth']))
                topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                out.extend([f[t] for t in topk])

            outs.append(sorted(out))
        else:
            outs.append(fn)

    # >>> FIX 2: use args.output_name
    output_path = os.path.join(base_out, args.output_name)

    with open(output_path, 'w') as f:
        json.dump(outs, f)

    print("Saved:", output_path)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
