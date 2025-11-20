#!/usr/bin/env python3
import json
import argparse
import os
from typing import List
import heapq
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Top-K Frame Selector')

    parser.add_argument('--score_path', type=str,
                        default='./outscores/videomme/blip/scores.json',
                        help='Path to input scores JSON file')

    parser.add_argument('--frame_path', type=str,
                        default='./outscores/videomme/blip/frames.json',
                        help='Path to input frame IDs JSON file')

    parser.add_argument('--output_file', type=str,
                        default='./selected_frames_topk',
                        help='Output directory')

    parser.add_argument('--max_num_frames', type=int, default=20,
                        help='Number of frames to select (Top-K)')

    parser.add_argument('--num_videos', type=int, default=None,
                        help='Limit number of videos to process')

    return parser.parse_args()


# -----------------------------------------------------
#  TOP-K SELECTION FUNCTION
# -----------------------------------------------------
def select_topk_frames(scores: List[float], frame_ids: List[int], k: int) -> List[int]:
    """
    Select top-K highest scoring frames.

    Args:
        scores (list): List of scores
        frame_ids (list): List of frame ids
        k (int): Number of frames to pick

    Returns:
        List of selected frame IDs (sorted temporally)
    """

    if len(scores) == 0:
        return []

    # Pair scores with corresponding frame ids
    paired = list(zip(scores, frame_ids))

    # Use heap to get largest k scores
    topk = heapq.nlargest(k, paired, key=lambda x: x[0])

    # Extract only frame ids, sort by temporal order
    selected = sorted([fid for _, fid in topk])

    return selected


# -----------------------------------------------------
#  PER VIDEO PROCESSING
# -----------------------------------------------------
def process_video(scores, frame_ids, k):
    if len(scores) <= k:
        return frame_ids
    return select_topk_frames(scores, frame_ids, k)


# -----------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------
def main(args):
    print("="*60)
    print("TOP-K FRAME SELECTION")
    print("="*60)
    print(f"K = {args.max_num_frames}")
    print(f"Loading scores from: {args.score_path}")
    print(f"Loading frame ids from: {args.frame_path}")

    # Load input JSONs
    with open(args.score_path) as f:
        all_scores = json.load(f)

    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)

    # How many videos to process?
    num_videos = len(all_scores)
    if args.num_videos is not None:
        num_videos = min(args.num_videos, num_videos)
        print(f"Demo mode: Only first {num_videos} videos will be processed")

    print(f"Total videos: {len(all_scores)}")
    print(f"Processing: {num_videos}\n")

    selected_all = []

    # Loop through videos
    for i in range(num_videos):
        if num_videos <= 20:
            print(f"Processing video {i+1}/{num_videos}")
        elif (i+1) % 100 == 0:
            print(f"Processed {i+1} videos...")

        scores = all_scores[i]
        frame_ids = all_frame_ids[i]

        selected = process_video(scores, frame_ids, args.max_num_frames)
        selected_all.append(selected)

    # Save results
    os.makedirs(args.output_file, exist_ok=True)
    output_path = os.path.join(
        args.output_file,
        f"selected_frames_topk_k{args.max_num_frames}.json"
    )

    with open(output_path, "w") as f:
        json.dump(selected_all, f)

    print("\n" + "="*60)
    print("✅ TOP-K SELECTION COMPLETE")
    print(f"Saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
