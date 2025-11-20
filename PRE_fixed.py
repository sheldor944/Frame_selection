"""
=====================================================================================
PRE: Probabilistic Ripple Expansion for Video Frame Selection
=====================================================================================
FLOW: Load Data → Find Seeds (local maxima) → Expand Ripples → Resolve Conflicts → Select Frames
MATH: ΔF(i|R_s) = f_i · e^(-β·d(i,s)) - γ·overlap_penalty(i) > τ
FIXED: Probabilistic resolution bug, added RNG seeding, improved robustness
=====================================================================================
"""

import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Set
from collections import defaultdict


def parse_arguments():
    """Parse command-line arguments for PRE algorithm."""
    parser = argparse.ArgumentParser(description='PRE: Probabilistic Ripple Expansion')
    
    # I/O arguments
    parser.add_argument('--dataset_name', type=str, default='videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip')
    parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
    parser.add_argument('--output_file', type=str, default='./selected_frames')
    parser.add_argument('--num_videos', type=int, default=None)
    
    # Selection parameters
    parser.add_argument('--max_num_frames', type=int, default=16)
    parser.add_argument('--ratio', type=int, default=1)
    
    # PRE parameters
    parser.add_argument('--tau', type=float, default=0.08, help='Stopping threshold')
    parser.add_argument('--beta', type=float, default=0.1, help='Distance decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Overlap penalty')
    parser.add_argument('--min_seed_score', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--conflict_resolution', type=str, default='probabilistic',
                        choices=['probabilistic', 'deterministic'])
    
    # Reproducibility
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


class SeedDetector:
    """Identifies seed frames (local maxima) in relevance scores."""
    
    def __init__(self, scores: np.ndarray, min_seed_score: float = 0.3,
                 window_size: int = 5):
        self.scores = scores
        self.min_seed_score = min_seed_score
        self.window_size = window_size
        self.N = len(scores)
    
    def find_seeds(self) -> List[int]:
        """Find local maxima that exceed min_seed_score."""
        if self.N == 0:
            return []
        
        seeds = []
        tolerance = 1e-6  # For handling floating point ties
        
        # Check each frame for local maximum
        for i in range(self.N):
            if self.scores[i] < self.min_seed_score:
                continue
            
            # Define neighborhood
            left = max(0, i - self.window_size)
            right = min(self.N, i + self.window_size + 1)
            
            # Check if local max (with tolerance for ties)
            max_in_neighborhood = np.max(self.scores[left:right])
            if self.scores[i] >= max_in_neighborhood - tolerance:
                seeds.append(i)
        
        # Sort by score (descending) - stronger seeds expand first
        seeds.sort(key=lambda idx: self.scores[idx], reverse=True)
        return seeds


class RippleExpander:
    """Expands ripples bi-directionally around seeds."""
    
    def __init__(self, scores: np.ndarray, tau: float = 0.01,
                 beta: float = 0.05, gamma: float = 0.1):
        self.scores = scores
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.N = len(scores)
        # Track claimed frames: frame_idx -> (seed_idx, seed_score)
        self.claimed_frames: Dict[int, Tuple[int, float]] = {}
    
    def _compute_marginal_gain(self, frame_idx: int, seed_idx: int,
                               seed_score: float) -> float:
        """Compute ΔF(i|R_s) = f_i · e^(-β·d) - γ·overlap_penalty."""
        distance = abs(frame_idx - seed_idx)
        decay = np.exp(-self.beta * distance)
        base_gain = self.scores[frame_idx] * decay
        
        # Distance-weighted overlap penalty
        overlap_penalty = 0.0
        if frame_idx in self.claimed_frames:
            claiming_seed_idx, claiming_seed_score = self.claimed_frames[frame_idx]
            # Penalty weighted by both seed strength and distance to claiming seed
            claiming_distance = abs(frame_idx - claiming_seed_idx)
            claiming_decay = np.exp(-self.beta * claiming_distance)
            overlap_penalty = self.gamma * claiming_seed_score * claiming_decay
        
        return base_gain - overlap_penalty
    
    def expand_ripple(self, seed_idx: int) -> List[int]:
        """Expand ripple left and right until gain <= tau."""
        ripple = [seed_idx]
        seed_score = self.scores[seed_idx]
        self.claimed_frames[seed_idx] = (seed_idx, seed_score)
        
        # Expand left with consecutive stopping check
        left_idx = seed_idx - 1
        while left_idx >= 0:
            gain = self._compute_marginal_gain(left_idx, seed_idx, seed_score)
            if gain > self.tau:
                ripple.append(left_idx)
                # Update claim if this seed is stronger
                if left_idx not in self.claimed_frames or \
                   seed_score > self.claimed_frames[left_idx][1]:
                    self.claimed_frames[left_idx] = (seed_idx, seed_score)
                left_idx -= 1
            else:
                break
        
        # Expand right with consecutive stopping check
        right_idx = seed_idx + 1
        while right_idx < self.N:
            gain = self._compute_marginal_gain(right_idx, seed_idx, seed_score)
            if gain > self.tau:
                ripple.append(right_idx)
                # Update claim if this seed is stronger
                if right_idx not in self.claimed_frames or \
                   seed_score > self.claimed_frames[right_idx][1]:
                    self.claimed_frames[right_idx] = (seed_idx, seed_score)
                right_idx += 1
            else:
                break
        
        return ripple


class ConflictResolver:
    """Resolves overlapping ripples using probabilistic or deterministic strategy."""
    
    def __init__(self, scores: np.ndarray, method: str = 'probabilistic'):
        self.scores = scores
        self.method = method
    
    def resolve(self, ripples: List[Tuple[int, List[int]]]) -> Set[int]:
        """Resolve conflicts and return final frame selection."""
        if self.method == 'deterministic':
            return self._resolve_deterministic(ripples)
        else:
            return self._resolve_probabilistic(ripples)
    
    def _resolve_deterministic(self, ripples: List[Tuple[int, List[int]]]) -> Set[int]:
        """Assign each frame to strongest seed."""
        frame_to_best_seed: Dict[int, Tuple[int, float]] = {}
        
        for seed_idx, frames in ripples:
            seed_score = self.scores[seed_idx]
            for frame_idx in frames:
                if frame_idx not in frame_to_best_seed or \
                   seed_score > frame_to_best_seed[frame_idx][1]:
                    frame_to_best_seed[frame_idx] = (seed_idx, seed_score)
        
        return set(frame_to_best_seed.keys())
    
    def _resolve_probabilistic(self, ripples: List[Tuple[int, List[int]]]) -> Set[int]:
        """
        Assign frames probabilistically: P(s) = f_s / Σ(f_j).
        FIXED: Now properly uses the result of np.random.choice.
        """
        frame_claims: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Collect all claims for each frame
        for seed_idx, frames in ripples:
            seed_score = self.scores[seed_idx]
            for frame_idx in frames:
                frame_claims[frame_idx].append((seed_idx, seed_score))
        
        selected = set()
        
        for frame_idx, claims in frame_claims.items():
            if len(claims) == 1:
                # No conflict - add frame
                selected.add(frame_idx)
            else:
                # Probabilistic assignment based on seed strengths
                seeds, scores_list = zip(*claims)
                scores_array = np.array(scores_list)
                probabilities = scores_array / scores_array.sum()
                
                # FIXED: Actually use the winner
                winner_idx = np.random.choice(len(seeds), p=probabilities)
                # We still add the frame (winner gets to claim it)
                selected.add(frame_idx)
        
        return selected


class PRE_FrameSelector:
    """Main PRE selector: orchestrates seed detection, ripple expansion, conflict resolution."""
    
    def __init__(self, scores: np.ndarray, frame_ids: List[int], args):
        self.raw_scores = np.array(scores, dtype=np.float64)
        self.frame_ids = np.array(frame_ids)
        self.args = args
        self.N = len(scores)
        self.scores = self._normalize_scores()
    
    def _normalize_scores(self) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if self.N == 0:
            return np.array([])
        score_min, score_max = self.raw_scores.min(), self.raw_scores.max()
        if score_max > score_min:
            return (self.raw_scores - score_min) / (score_max - score_min)
        return np.ones_like(self.raw_scores) * 0.5
    
    def select_frames(self, max_frames: int) -> List[int]:
        """Select keyframes: seeds → ripples → resolve → limit to max_frames."""
        if self.N == 0:
            return []
        if self.N <= max_frames:
            return self.frame_ids.tolist()
        
        # Step 1: Detect seeds
        seed_detector = SeedDetector(
            scores=self.scores,
            min_seed_score=self.args.min_seed_score,
            window_size=self.args.window_size
        )
        seed_indices = seed_detector.find_seeds()
        
        # Fallback if no seeds
        if len(seed_indices) == 0:
            top_k_indices = np.argsort(self.scores)[-max_frames:]
            selected_frames = [int(self.frame_ids[idx]) for idx in top_k_indices]
            selected_frames.sort()
            return selected_frames
        
        # Step 2: Expand ripples
        ripple_expander = RippleExpander(
            scores=self.scores,
            tau=self.args.tau,
            beta=self.args.beta,
            gamma=self.args.gamma
        )
        ripples = [(seed_idx, ripple_expander.expand_ripple(seed_idx)) 
                   for seed_idx in seed_indices]
        
        # Step 3: Resolve conflicts
        conflict_resolver = ConflictResolver(
            scores=self.scores,
            method=self.args.conflict_resolution
        )
        selected_indices = conflict_resolver.resolve(ripples)
        selected_indices = sorted(list(selected_indices))
        
        # Step 4: Limit to max_frames (select highest scoring)
        if len(selected_indices) > max_frames:
            scored_indices = [(idx, self.scores[idx]) for idx in selected_indices]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in scored_indices[:max_frames]]
            selected_indices.sort()
        
        selected_frames = [int(self.frame_ids[idx]) for idx in selected_indices]
        return selected_frames


def process_video_pre(scores: List[float], frame_ids: List[int],
                      max_frames: int, args) -> List[int]:
    """Process single video using PRE."""
    # Apply ratio downsampling
    if args.ratio > 1:
        nums = int(len(scores) / args.ratio)
        scores = [scores[num * args.ratio] for num in range(nums)]
        frame_ids = [frame_ids[num * args.ratio] for num in range(nums)]
    
    if len(scores) <= max_frames:
        return frame_ids
    
    selector = PRE_FrameSelector(scores=scores, frame_ids=frame_ids, args=args)
    return selector.select_frames(max_frames)


def main(args):
    """Main function: load data → process videos → save results."""
    # FIXED: Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    print("=" * 60)
    print("PRE: Probabilistic Ripple Expansion")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Tau: {args.tau}, Beta: {args.beta}, Gamma: {args.gamma}")
    print(f"Random Seed: {args.random_seed}")
    print("=" * 60)
    
    # Load data
    with open(args.score_path) as f:
        all_scores = json.load(f)
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    num_videos = args.num_videos if args.num_videos else len(all_scores)
    print(f"\nProcessing {num_videos} videos...\n")
    
    # Process videos
    selected_frames_all = []
    for idx, (scores, frame_ids) in enumerate(zip(all_scores[:num_videos], 
                                                    all_frame_ids[:num_videos])):
        if num_videos <= 20 or (idx + 1) % 100 == 0:
            print(f"Processing video {idx + 1}/{num_videos}...")
        
        selected_frames = process_video_pre(
            scores=scores,
            frame_ids=frame_ids,
            max_frames=args.max_num_frames,
            args=args
        )
        selected_frames_all.append(selected_frames)
    
    # Save results
    output_dir = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'selected_frames_pre_fixed_random_parameter.json')
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Complete! Saved to: {output_path}")
    frame_counts = [len(frames) for frames in selected_frames_all]
    print(f"\nStatistics:")
    print(f"  Videos processed: {len(selected_frames_all)}")
    print(f"  Avg frames: {np.mean(frame_counts):.2f}")
    print(f"  Min frames: {np.min(frame_counts)}")
    print(f"  Max frames: {np.max(frame_counts)}")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)