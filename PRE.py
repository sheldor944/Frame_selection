"""
=====================================================================================
PROBABILISTIC RIPPLE EXPANSION (PRE) - Video Frame Selection
=====================================================================================

ALGORITHM OVERVIEW:
------------------
PRE (Probabilistic Ripple Expansion) is a frame selection algorithm inspired by 
ripple propagation in water. It selects keyframes by:

1. SEED IDENTIFICATION: Find local maxima in relevance scores (peaks/events)
2. RIPPLE EXPANSION: Expand context windows around each seed bi-directionally
3. ADAPTIVE STOPPING: Stop expansion when marginal gain falls below threshold
4. CONFLICT RESOLUTION: Resolve overlapping ripples probabilistically
5. FRAME SELECTION: Return union of all ripple regions

FLOW:
-----
Input: Frame scores, frame IDs, parameters
  ↓
Find Seeds (local maxima in scores)
  ↓
For each seed:
  ├→ Expand left (backward in time)
  │   └→ Add frame if: score * decay - overlap_penalty > threshold
  ├→ Expand right (forward in time)
  │   └→ Add frame if: score * decay - overlap_penalty > threshold
  └→ Store ripple region
  ↓
Resolve overlapping ripples (probabilistic assignment)
  ↓
Merge all ripples and sort temporally
  ↓
Output: Selected frame IDs

KEY PARAMETERS:
--------------
- tau (τ): Stopping threshold - controls when ripple stops expanding
- beta (β): Distance decay - controls how quickly gain decreases with distance
- gamma (γ): Overlap penalty - prevents redundant selections
- min_seed_score: Minimum score to be considered a seed
- window_size: For local maxima detection

MATHEMATICAL FOUNDATION:
-----------------------
Objective: F(S) = Σ(f_i) - λ Σ(e^(-α·d(i,j)))
                   i∈S      (i,j)∈S

Marginal Gain: ΔF(i|R_s) = f_i · e^(-β·d(i,s)) - γ·overlap_penalty(i)

Expansion Rule: Add frame i if ΔF(i|R_s) > τ

Probabilistic Resolution: P(s) = f_s / (f_s + f_t) for conflicting seeds s,t

EXAMPLE:
--------
Frame:  0    1    2    3    4    5    6    7
Score:  0.2  0.3  0.9  0.8  0.4  0.1  0.85 0.7
                  ↑seed           ↑seed

Ripple(2) expands: [1, 2, 3, 4]
Ripple(6) expands: [5, 6, 7]
Final selection: {1, 2, 3, 4, 5, 6, 7} - capturing events with context
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
    """
    Parse command-line arguments for PRE algorithm.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description='PRE: Probabilistic Ripple Expansion for Frame Selection'
    )
    
    # ==================== Dataset and I/O Arguments ====================
    parser.add_argument('--dataset_name', type=str, default='videomme',
                        help='Dataset name: longvideobench or videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip',
                        help='Feature extraction model: blip/clip/sevila')
    parser.add_argument('--score_path', type=str,
                        default='./outscores/videomme/blip/scores.json',
                        help='Path to input scores JSON file')
    parser.add_argument('--frame_path', type=str,
                        default='./outscores/videomme/blip/frames.json',
                        help='Path to input frame IDs JSON file')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory for selected frames')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (default: all)')
    
    # ==================== Selection Parameters ====================
    parser.add_argument('--max_num_frames', type=int, default=16,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    
    # ==================== PRE-Specific Parameters ====================
    parser.add_argument('--tau', type=float, default=0.01,
                        help='Stopping threshold for ripple expansion (0-1). '
                             'Lower values = larger context windows')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Distance decay factor. '
                             'Higher values = smaller context windows, faster decay')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Overlap penalty coefficient. '
                             'Higher values = stronger penalty for overlapping ripples')
    parser.add_argument('--min_seed_score', type=float, default=0.3,
                        help='Minimum normalized score (0-1) to be considered a seed. '
                             'Only frames above this threshold can be seeds')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size for local maxima detection. '
                             'Frame must be max within ±window_size to be a seed')
    parser.add_argument('--conflict_resolution', type=str, default='probabilistic',
                        choices=['probabilistic', 'deterministic'],
                        help='Method for resolving overlapping ripples. '
                             'probabilistic: assign based on seed strength probabilities; '
                             'deterministic: assign to strongest seed')
    
    return parser.parse_args()


class SeedDetector:
    """
    Identifies seed frames (local maxima) in the relevance score sequence.
    
    Seeds represent potential keyframes or important events in the video.
    A frame qualifies as a seed if:
    1. Its score exceeds min_seed_score threshold
    2. It's a local maximum within window_size neighborhood
    
    Mathematical Definition:
    Frame i is a seed if:
        f_i ≥ min_seed_score AND f_i = max(f_{i-w}, ..., f_{i+w})
        where w = window_size
    """
    
    def __init__(self, scores: np.ndarray, min_seed_score: float = 0.3,
                 window_size: int = 5):
        """
        Initialize seed detector.
        
        Args:
            scores: Normalized relevance scores in [0, 1] range
            min_seed_score: Minimum score threshold for seed qualification
            window_size: Radius of neighborhood for local maxima detection
        """
        self.scores = scores
        self.min_seed_score = min_seed_score
        self.window_size = window_size
        self.N = len(scores)
    
    def find_seeds(self) -> List[int]:
        """
        Find seed frames using local maxima detection.
        
        Algorithm:
        1. For each frame i:
            a. Check if score >= min_seed_score
            b. Check if score is maximum in neighborhood [i-w, i+w]
        2. Sort seeds by score (descending) for priority-based expansion
        
        Returns:
            List of seed frame indices sorted by score (highest first)
            
        Example:
            scores = [0.2, 0.3, 0.9, 0.8, 0.4, 0.1, 0.85, 0.7]
            window_size = 2, min_seed_score = 0.3
            → seeds = [2, 6] (indices of 0.9 and 0.85)
        """
        if self.N == 0:
            return []
        
        seeds = []
        
        # Check each frame for local maximum property
        for i in range(self.N):
            # Threshold check: score must exceed minimum
            if self.scores[i] < self.min_seed_score:
                continue
            
            # Define neighborhood bounds (handle edges)
            left = max(0, i - self.window_size)
            right = min(self.N, i + self.window_size + 1)
            
            # Local maximum check
            neighborhood = self.scores[left:right]
            if self.scores[i] == np.max(neighborhood):
                seeds.append(i)
        
        # Sort seeds by score (descending) - stronger seeds expand first
        # This ensures important events get priority in claiming frames
        seeds.sort(key=lambda idx: self.scores[idx], reverse=True)
        
        return seeds


class RippleExpander:
    """
    Expands ripples (context windows) around seed frames bi-directionally.
    
    Each ripple grows left and right until the marginal gain of adding a frame
    falls below the stopping threshold (tau).
    
    Marginal Gain Formula:
        ΔF(i|R_s) = f_i · e^(-β·d(i,s)) - γ·overlap_penalty(i)
        
        where:
        - f_i: relevance score of frame i
        - β: distance decay factor
        - d(i,s): temporal distance from seed s
        - γ: overlap penalty coefficient
        - overlap_penalty(i): penalty if frame i is claimed by another ripple
    
    Expansion stops when ΔF(i|R_s) ≤ τ (stopping threshold)
    """
    
    def __init__(self, scores: np.ndarray, tau: float = 0.01,
                 beta: float = 0.05, gamma: float = 0.1):
        """
        Initialize ripple expander.
        
        Args:
            scores: Normalized relevance scores [0, 1]
            tau: Stopping threshold - expansion stops when gain < tau
            beta: Distance decay factor - controls context window size
            gamma: Overlap penalty coefficient - discourages overlapping claims
        """
        self.scores = scores
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.N = len(scores)
        
        # Track which frames are already claimed by ripples
        # Maps frame_idx → seed_score of claiming seed
        self.claimed_frames: Dict[int, float] = {}
    
    def _compute_marginal_gain(self, frame_idx: int, seed_idx: int,
                               seed_score: float) -> float:
        """
        Compute marginal gain of adding a frame to a ripple.
        
        Formula: ΔF(i|R_s) = f_i · e^(-β·d(i,s)) - γ·overlap_penalty(i)
        
        Components:
        1. Base gain: f_i · e^(-β·d(i,s))
           - Frame's relevance decayed by distance from seed
           - Exponential decay ensures nearby frames contribute more
           
        2. Overlap penalty: γ · score(claiming_seed)
           - If frame is already claimed, penalize based on claiming seed strength
           - Prevents weak seeds from stealing frames from strong seeds
        
        Args:
            frame_idx: Index of candidate frame to add
            seed_idx: Index of seed frame (center of ripple)
            seed_score: Relevance score of seed
        
        Returns:
            Marginal gain value (can be negative if overlap penalty is high)
            
        Example:
            frame_idx=3, seed_idx=2, seed_score=0.9
            scores[3]=0.8, β=0.05
            distance = |3-2| = 1
            decay = e^(-0.05*1) ≈ 0.951
            base_gain = 0.8 * 0.951 ≈ 0.761
            If frame 3 not claimed: ΔF = 0.761
            If frame 3 claimed by seed with score 0.85: ΔF = 0.761 - 0.1*0.85 = 0.676
        """
        # Calculate temporal distance from seed
        distance = abs(frame_idx - seed_idx)
        
        # Base gain: frame score weighted by exponential distance decay
        decay = np.exp(-self.beta * distance)
        base_gain = self.scores[frame_idx] * decay
        
        # Overlap penalty: discourage claiming frames from stronger ripples
        overlap_penalty = 0.0
        if frame_idx in self.claimed_frames:
            claiming_seed_score = self.claimed_frames[frame_idx]
            # Penalty proportional to the strength of existing claim
            overlap_penalty = self.gamma * claiming_seed_score
        
        return base_gain - overlap_penalty
    
    def expand_ripple(self, seed_idx: int) -> List[int]:
        """
        Expand a single ripple bi-directionally from seed.
        
        Expansion Algorithm:
        1. Initialize ripple with seed frame
        2. Expand LEFT (backward in time):
           - For each frame to the left of seed:
             - Compute marginal gain ΔF
             - If ΔF > τ: add frame, update claims, continue
             - Else: stop left expansion
        3. Expand RIGHT (forward in time):
           - For each frame to the right of seed:
             - Compute marginal gain ΔF
             - If ΔF > τ: add frame, update claims, continue
             - Else: stop right expansion
        4. Return all frames in ripple
        
        Args:
            seed_idx: Index of seed frame (ripple center)
        
        Returns:
            List of frame indices in the ripple (unsorted)
            
        Example:
            Seed at index 2, scores = [0.2, 0.3, 0.9, 0.8, 0.4, 0.1]
            tau=0.01, beta=0.05, gamma=0.1
            
            Left expansion from 2:
            - Check frame 1: ΔF = 0.3 * e^(-0.05*1) ≈ 0.285 > 0.01 → ADD
            - Check frame 0: ΔF = 0.2 * e^(-0.05*2) ≈ 0.181 > 0.01 → ADD
            
            Right expansion from 2:
            - Check frame 3: ΔF = 0.8 * e^(-0.05*1) ≈ 0.761 > 0.01 → ADD
            - Check frame 4: ΔF = 0.4 * e^(-0.05*2) ≈ 0.362 > 0.01 → ADD
            - Check frame 5: ΔF = 0.1 * e^(-0.05*3) ≈ 0.086 > 0.01 → ADD
            
            Ripple = [0, 1, 2, 3, 4, 5]
        """
        ripple = [seed_idx]
        seed_score = self.scores[seed_idx]
        
        # Mark seed as claimed by this ripple
        self.claimed_frames[seed_idx] = seed_score
        
        # ==================== Expand LEFT (backward in time) ====================
        left_idx = seed_idx - 1
        while left_idx >= 0:
            # Compute marginal gain for this frame
            gain = self._compute_marginal_gain(left_idx, seed_idx, seed_score)
            
            # Check stopping condition
            if gain > self.tau:
                # Add frame to ripple
                ripple.append(left_idx)
                
                # Update claim (may overwrite weaker claims)
                if left_idx not in self.claimed_frames or \
                   seed_score > self.claimed_frames[left_idx]:
                    self.claimed_frames[left_idx] = seed_score
                
                # Move to next frame
                left_idx -= 1
            else:
                # Gain too low - stop expanding left
                break
        
        # ==================== Expand RIGHT (forward in time) ====================
        right_idx = seed_idx + 1
        while right_idx < self.N:
            # Compute marginal gain for this frame
            gain = self._compute_marginal_gain(right_idx, seed_idx, seed_score)
            
            # Check stopping condition
            if gain > self.tau:
                # Add frame to ripple
                ripple.append(right_idx)
                
                # Update claim
                if right_idx not in self.claimed_frames or \
                   seed_score > self.claimed_frames[right_idx]:
                    self.claimed_frames[right_idx] = seed_score
                
                # Move to next frame
                right_idx += 1
            else:
                # Gain too low - stop expanding right
                break
        
        return ripple


class ConflictResolver:
    """
    Resolves conflicts when multiple ripples claim the same frame.
    
    Strategies:
    1. Deterministic: Assign to strongest seed
    2. Probabilistic: Assign based on P(s) = f_s / Σ(f_j)
    """
    
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
        """Assign each frame to strongest claiming seed."""
        frame_to_best_seed: Dict[int, Tuple[int, float]] = {}
        
        for seed_idx, frames in ripples:
            seed_score = self.scores[seed_idx]
            for frame_idx in frames:
                if frame_idx not in frame_to_best_seed or \
                   seed_score > frame_to_best_seed[frame_idx][1]:
                    frame_to_best_seed[frame_idx] = (seed_idx, seed_score)
        
        return set(frame_to_best_seed.keys())
    
    def _resolve_probabilistic(self, ripples: List[Tuple[int, List[int]]]) -> Set[int]:
        """Probabilistic assignment: P(s) = score(s) / Σ scores."""
        frame_claims: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        for seed_idx, frames in ripples:
            seed_score = self.scores[seed_idx]
            for frame_idx in frames:
                frame_claims[frame_idx].append((seed_idx, seed_score))
        
        selected = set()
        for frame_idx, claims in frame_claims.items():
            if len(claims) == 1:
                selected.add(frame_idx)
            else:
                # Probabilistic assignment
                seeds, scores_list = zip(*claims)
                scores_array = np.array(scores_list)
                probabilities = scores_array / scores_array.sum()
                chosen_seed_idx = np.random.choice(len(seeds), p=probabilities)
                selected.add(frame_idx)
        
        return selected


class PRE_FrameSelector:
    """Main PRE selector: seeds → ripples → conflict resolution."""
    
    def __init__(self, scores: np.ndarray, frame_ids: List[int], args):
        self.raw_scores = np.array(scores, dtype=np.float64)
        self.frame_ids = np.array(frame_ids)
        self.args = args
        self.N = len(scores)
        self.scores = self._normalize_scores()
    
    def _normalize_scores(self) -> np.ndarray:
        """Normalize to [0, 1]."""
        if self.N == 0:
            return np.array([])
        score_min, score_max = self.raw_scores.min(), self.raw_scores.max()
        if score_max > score_min:
            return (self.raw_scores - score_min) / (score_max - score_min)
        return np.ones_like(self.raw_scores) * 0.5
    
    def select_frames(self, max_frames: int) -> List[int]:
        """Select keyframes using PRE."""
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
        
        ripples = []
        for seed_idx in seed_indices:
            ripple_frames = ripple_expander.expand_ripple(seed_idx)
            ripples.append((seed_idx, ripple_frames))
        
        # Step 3: Resolve conflicts
        conflict_resolver = ConflictResolver(
            scores=self.scores,
            method=self.args.conflict_resolution
        )
        selected_indices = conflict_resolver.resolve(ripples)
        selected_indices = sorted(list(selected_indices))
        
        # Step 4: Limit to max_frames
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
    """Main processing function."""
    print("=" * 60)
    print("PRE: Probabilistic Ripple Expansion")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Tau: {args.tau}, Beta: {args.beta}, Gamma: {args.gamma}")
    print("=" * 60)
    
    # Load data
    with open(args.score_path) as f:
        all_scores = json.load(f)
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    num_videos = args.num_videos if args.num_videos else len(all_scores)
    print(f"\nProcessing {num_videos} videos...\n")
    
    selected_frames_all = []
    for idx, (scores, frame_ids) in enumerate(zip(all_scores[:num_videos], 
                                                    all_frame_ids[:num_videos])):
        if (idx + 1) % 100 == 0:
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
    output_path = os.path.join(output_dir, 'selected_frames_pre.json')
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    print(f"\n✅ Complete! Saved to: {output_path}")
    frame_counts = [len(frames) for frames in selected_frames_all]
    print(f"Avg frames: {np.mean(frame_counts):.2f}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)