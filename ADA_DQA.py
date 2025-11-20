"""
ADA-DQ: Shot-Aware Dual-Quota Adaptive Allocation for Video Frame Selection

CORE IDEA:
----------
This algorithm extends the AKS (Adaptive K-Selection) framework by replacing its vanilla 
ADA (Adaptive Distribution Allocation) with a "Dual-Quota" mechanism that balances:

1. GLOBAL COVERAGE (via ADA's recursive splitting) - ensures temporal diversity
2. LOCAL DENSITY AWARENESS (via dual quotas) - allows multiple picks in content-rich regions

WHY IT MATTERS:
- Video content is bursty: high autocorrelation (ACF(1)≈0.66) means relevant frames cluster
- Dense shots (e.g., action sequences) may contain multiple distinct salient moments
- Pure TOP-K over-concentrates; pure uniform under-samples peaks; ADA alone enforces 
  one-per-bin too rigidly

DUAL-QUOTA MECHANISM:
---------------------
For each recursive bin B with budget m:
  1. m_min:   Reserve 1 frame if bin is "active" (has strong signal) → guarantees coverage
  2. m_bonus: Allocate extra frames proportional to local peak density ρ(B) → rewards density
  3. m_alloc = clip(m_min + m_bonus, 0, m)
  4. Pick m_alloc frames with temporal NMS (gap ≥ Δ_min) to avoid micro-redundancy
  5. Recurse on children with remaining budget m_rem = m - m_alloc

MATHEMATICAL VIEW:
------------------
Implicitly optimizes:
  Σ s_t  (relevance)  +  λ·c(I)  (coverage penalty)  +  μ·Σ_shots φ(|I ∩ shot_k|)
      ↑                      ↑                                    ↑
   scoring           ADA recursion                    density bonus (saturating)

PARAMETERS:
-----------
- theta_a:      Activity threshold (e.g., P85 of scores) - marks "active" bins
- beta:         Density bonus weight [0,1] - controls m_bonus strength
- rho_ref:      Reference density (e.g., 3 peaks/unit) - normalizes ρ(B)
- delta_min:    Minimum temporal gap (seconds) - prevents near-duplicates
- L_max:        Max recursion depth - limits tree depth
- theta_stop:   Stop threshold - if top-m mean >> bin mean, stop splitting

COMPLEXITY: O(T log T) where T = number of candidate frames
DETERMINISTIC: Yes (given fixed random seed for any tie-breaking)
"""

import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

def parse_arguments():
    parser = argparse.ArgumentParser(description='ADA-DQ: Shot-Aware Dual-Quota Frame Selection')
    
    # Dataset and I/O
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
    parser.add_argument('--max_num_frames', type=int, default=16,
                        help='Maximum number of frames to select (M)')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory for selected frames')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (default: all)')
    
    # ADA base parameters
    parser.add_argument('--L_max', type=int, default=5,
                        help='Maximum recursion depth for ADA')
    parser.add_argument('--theta_stop', type=float, default=0.3,
                        help='Stop threshold: if (top_mean - bin_mean) > theta_stop, stop splitting')
    
    # Dual-Quota (DQ) parameters
    parser.add_argument('--theta_a', type=float, default=None,
                        help='Activity threshold (default: P85 of video scores)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Density bonus weight [0,1] - controls extra allocation in dense regions')
    parser.add_argument('--rho_ref', type=float, default=3.0,
                        help='Reference peak density for normalization')
    parser.add_argument('--delta_min', type=float, default=3.0,
                        help='Minimum temporal gap (seconds) between selected frames')
    parser.add_argument('--theta_p_percentile', type=float, default=75.0,
                        help='Percentile for peak threshold (default: P75)')
    
    # Shot detection (optional - simplified version)
    parser.add_argument('--use_shot_detection', action='store_true',
                        help='Enable simple shot boundary detection')
    parser.add_argument('--shot_threshold', type=float, default=0.15,
                        help='Threshold for shot boundary detection (score gradient)')
    
    return parser.parse_args()


@dataclass
class Bin:
    """Represents a temporal bin in the recursive tree"""
    idx: np.ndarray  # Frame indices in this bin
    m: int           # Budget (number of frames to select)
    level: int       # Recursion depth


class ShotDetector:
    """
    Simple shot boundary detector based on score gradients.
    In production, use PySceneDetect or CLIP feature distances.
    """
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
    
    def detect(self, scores: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect shot boundaries using score gradient changes.
        
        Returns:
            List of (start_time, end_time) tuples for each shot
        """
        if len(scores) < 3:
            return [(times[0], times[-1])]
        
        # Compute score gradient
        grad = np.abs(np.diff(scores))
        grad_norm = grad / (grad.max() + 1e-6)
        
        # Find boundaries where gradient exceeds threshold
        boundaries = [0]
        for i in range(1, len(grad)):
            if grad_norm[i] > self.threshold:
                boundaries.append(i)
        boundaries.append(len(scores) - 1)
        
        # Convert to time ranges
        shots = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            shots.append((times[start_idx], times[end_idx]))
        
        return shots


class DensityCalculator:
    """
    Calculates local peak density for temporal bins.
    """
    def __init__(self, theta_p: float, rho_ref: float = 3.0):
        """
        Args:
            theta_p: Peak threshold (scores above this count as peaks)
            rho_ref: Reference density for normalization
        """
        self.theta_p = theta_p
        self.rho_ref = rho_ref
    
    def compute_rho(self, scores: np.ndarray, times: np.ndarray) -> float:
        """
        Compute normalized peak density: (number of peaks) / (duration in seconds)
        
        Args:
            scores: Score array for the bin
            times: Time array for the bin
        
        Returns:
            Peak density (peaks per second)
        """
        if len(scores) == 0:
            return 0.0
        
        # Count peaks
        num_peaks = np.sum(scores >= self.theta_p)
        
        # Duration in seconds
        duration = max(times[-1] - times[0], 1e-6)
        
        # Normalized density
        rho = num_peaks / duration
        
        return rho
    
    def compute_rho_shot_aware(self, scores: np.ndarray, times: np.ndarray, 
                                shots: List[Tuple[float, float]]) -> float:
        """
        Shot-aware density: count shots with at least one peak.
        
        Args:
            scores: Score array for the bin
            times: Time array for the bin
            shots: List of (start, end) time tuples
        
        Returns:
            Shot-level peak density
        """
        if len(shots) == 0:
            return self.compute_rho(scores, times)
        
        # Count shots with peaks
        active_shots = 0
        for shot_start, shot_end in shots:
            # Find frames in this shot
            mask = (times >= shot_start) & (times <= shot_end)
            shot_scores = scores[mask]
            
            if len(shot_scores) > 0 and shot_scores.max() >= self.theta_p:
                active_shots += 1
        
        # Density: active shots per total shots
        rho_shot = active_shots / max(len(shots), 1)
        
        # Scale to match time-based density
        return rho_shot * self.rho_ref


class TemporalNMSSelector:
    """
    Selects top-k frames with minimum temporal gap constraint.
    """
    def __init__(self, delta_min: float):
        """
        Args:
            delta_min: Minimum temporal gap (seconds) between selected frames
        """
        self.delta_min = delta_min
    
    def select(self, scores: np.ndarray, times: np.ndarray, 
               frame_ids: np.ndarray, k: int) -> List[int]:
        """
        Greedy selection: pick highest score, suppress neighbors, repeat.
        
        Args:
            scores: Relevance scores
            times: Time stamps (seconds)
            frame_ids: Original frame IDs
            k: Number of frames to select
        
        Returns:
            List of selected frame IDs
        """
        if len(scores) == 0 or k <= 0:
            return []
        
        # Sort by score (descending)
        order = np.argsort(scores)[::-1]
        
        selected = []
        for idx in order:
            if len(selected) >= k:
                break
            
            # Check temporal gap constraint
            current_time = times[idx]
            valid = True
            for sel_idx in selected:
                if abs(current_time - times[sel_idx]) < self.delta_min:
                    valid = False
                    break
            
            if valid:
                selected.append(idx)
        
        # Convert to frame IDs and sort temporally
        selected_frame_ids = [int(frame_ids[idx]) for idx in selected]
        
        return selected_frame_ids


class ADADQSelector:
    """
    Main ADA-DQ algorithm: Adaptive Dual-Quota frame selection.
    """
    def __init__(self, args):
        self.L_max = args.L_max
        self.theta_stop = args.theta_stop
        self.beta = args.beta
        self.rho_ref = args.rho_ref
        self.delta_min = args.delta_min
        self.theta_a = args.theta_a
        self.theta_p_percentile = args.theta_p_percentile
        self.use_shot_detection = args.use_shot_detection
        
        # Initialize components
        self.shot_detector = ShotDetector(threshold=args.shot_threshold)
        self.nms_selector = TemporalNMSSelector(delta_min=args.delta_min)
        
        # Will be set per video
        self.density_calc = None
        self.shots = []
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]"""
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            return (scores - score_min) / (score_max - score_min)
        else:
            return np.ones_like(scores) * 0.5
    
    def _is_active(self, scores: np.ndarray, times: np.ndarray) -> bool:
        """
        Check if bin is "active" (has strong signal).
        
        Criteria:
          1. Max score >= theta_a (activity threshold), OR
          2. Peak density > 0 (has at least one peak)
        """
        if len(scores) == 0:
            return False
        
        max_score = scores.max()
        rho = self.density_calc.compute_rho(scores, times)
        
        return (max_score >= self.theta_a) or (rho > 0)
    
    def _compute_m_alloc(self, scores: np.ndarray, times: np.ndarray, m: int) -> int:
        """
        Compute dual-quota allocation for this bin.
        
        Returns:
            m_alloc: Number of frames to allocate to this bin before splitting
        """
        # m_min: reserve 1 if active
        m_min = 1 if self._is_active(scores, times) else 0
        
        # m_bonus: extra allocation based on density
        rho = self.density_calc.compute_rho(scores, times)
        m_bonus = int(round(self.beta * (rho / max(self.rho_ref, 1e-6))))
        
        # Total allocation (clipped to budget)
        m_alloc = max(0, min(m, m_min + m_bonus))
        
        return m_alloc
    
    def _should_stop_splitting(self, scores: np.ndarray, m: int, level: int) -> bool:
        """
        ADA stop criteria: stop if top-m is significantly better than bin average.
        
        Returns:
            True if should stop splitting and just take top-m
        """
        if level >= self.L_max or m <= 1 or len(scores) <= m:
            return True
        
        # Check if top-m is much better than average
        bin_mean = scores.mean()
        top_m_scores = np.partition(scores, -min(m, len(scores)))[-min(m, len(scores)):]
        top_mean = top_m_scores.mean()
        
        mean_diff = top_mean - bin_mean
        
        return mean_diff >= self.theta_stop
    
    def _split_bin(self, idx: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split bin into two children at temporal midpoint.
        
        Returns:
            (left_idx, right_idx): Indices for left and right children
        """
        if len(idx) <= 1:
            return idx, np.array([], dtype=int)
        
        # Find midpoint time
        bin_times = times[idx]
        mid_time = (bin_times[0] + bin_times[-1]) / 2.0
        
        # Split at midpoint
        left_mask = bin_times <= mid_time
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        
        # Ensure non-empty children
        if len(left_idx) == 0:
            split_point = len(idx) // 2
            left_idx = idx[:split_point]
            right_idx = idx[split_point:]
        elif len(right_idx) == 0:
            split_point = len(idx) // 2
            left_idx = idx[:split_point]
            right_idx = idx[split_point:]
        
        return left_idx, right_idx
    
    def _distribute_budget(self, scores: np.ndarray, left_idx: np.ndarray, 
                          right_idx: np.ndarray, m_rem: int) -> Tuple[int, int]:
        """
        Distribute remaining budget to child bins proportional to score mass.
        
        Args:
            scores: Full score array
            left_idx: Indices for left child
            right_idx: Indices for right child
            m_rem: Remaining budget to distribute
        
        Returns:
            (m_left, m_right): Budget allocation for children
        """
        if m_rem <= 0:
            return 0, 0
        
        # Compute score mass for each child
        mass_left = scores[left_idx].sum() + 1e-6
        mass_right = scores[right_idx].sum() + 1e-6
        total_mass = mass_left + mass_right
        
        # Distribute proportionally
        m_left = int(round(m_rem * mass_left / total_mass))
        m_right = m_rem - m_left
        
        return m_left, m_right
    
    def _recurse(self, bin_idx: np.ndarray, m: int, level: int,
                 scores: np.ndarray, times: np.ndarray, 
                 frame_ids: np.ndarray, selected: set):
        """
        Recursive ADA-DQ selection.
        
        Args:
            bin_idx: Indices for current bin
            m: Budget for this bin
            level: Current recursion depth
            scores: Full normalized score array
            times: Full time array (seconds)
            frame_ids: Full frame ID array
            selected: Set of selected indices (modified in-place)
        """
        # Base cases
        if m <= 0 or len(bin_idx) == 0:
            return
        
        # Extract bin data
        bin_scores = scores[bin_idx]
        bin_times = times[bin_idx]
        bin_frame_ids = frame_ids[bin_idx]
        
        # Check ADA stop condition
        if self._should_stop_splitting(bin_scores, m, level):
            # Just take top-m with temporal NMS
            selected_ids = self.nms_selector.select(
                bin_scores, bin_times, np.arange(len(bin_idx)), m
            )
            selected.update([bin_idx[sid] for sid in selected_ids])
            return
        
        # --- DUAL-QUOTA ALLOCATION ---
        m_alloc = self._compute_m_alloc(bin_scores, bin_times, m)
        
        # Allocate m_alloc frames to this bin
        if m_alloc > 0:
            selected_ids = self.nms_selector.select(
                bin_scores, bin_times, np.arange(len(bin_idx)), m_alloc
            )
            selected.update([bin_idx[sid] for sid in selected_ids])
        
        # Remaining budget
        m_rem = m - m_alloc
        if m_rem <= 0:
            return
        
        # --- RECURSIVE SPLIT ---
        left_idx, right_idx = self._split_bin(bin_idx, times)
        
        # Distribute remaining budget
        m_left, m_right = self._distribute_budget(scores, left_idx, right_idx, m_rem)
        
        # Recurse on children
        if m_left > 0 and len(left_idx) > 0:
            self._recurse(left_idx, m_left, level + 1, 
                         scores, times, frame_ids, selected)
        
        if m_right > 0 and len(right_idx) > 0:
            self._recurse(right_idx, m_right, level + 1, 
                         scores, times, frame_ids, selected)
    
    def select_frames(self, scores: List[float], frame_ids: List[int], 
                     max_frames: int) -> List[int]:
        """
        Main entry point: select max_frames from video using ADA-DQ.
        
        Args:
            scores: List of frame relevance scores
            frame_ids: List of frame IDs
            max_frames: Maximum number of frames to select (M)
        
        Returns:
            List of selected frame IDs (sorted temporally)
        """
        # Convert to numpy arrays
        scores = np.array(scores, dtype=np.float64)
        frame_ids = np.array(frame_ids, dtype=np.int64)
        
        # Handle short videos
        if len(scores) <= max_frames:
            return sorted([int(fid) for fid in frame_ids])
        
        # Normalize scores
        scores_norm = self._normalize_scores(scores)
        
        # Create time array (assume uniform 1 fps if not provided)
        times = np.arange(len(scores), dtype=np.float64)
        
        # Set activity threshold if not provided
        if self.theta_a is None:
            self.theta_a = np.percentile(scores_norm, 85)
        
        # Compute peak threshold
        theta_p = np.percentile(scores_norm, self.theta_p_percentile)
        
        # Initialize density calculator
        self.density_calc = DensityCalculator(theta_p=theta_p, rho_ref=self.rho_ref)
        
        # Optional: detect shots
        if self.use_shot_detection:
            self.shots = self.shot_detector.detect(scores_norm, times)
        else:
            self.shots = []
        
        # Initialize recursion
        root_idx = np.arange(len(scores))
        selected = set()
        
        # Run recursive selection
        self._recurse(
            bin_idx=root_idx,
            m=max_frames,
            level=0,
            scores=scores_norm,
            times=times,
            frame_ids=frame_ids,
            selected=selected
        )
        
        # Convert to sorted frame IDs
        selected_frame_ids = sorted([int(frame_ids[idx]) for idx in selected])
        
        # Ensure we have exactly max_frames (or less if video is short)
        if len(selected_frame_ids) > max_frames:
            # Trim excess (shouldn't happen, but safeguard)
            selected_frame_ids = selected_frame_ids[:max_frames]
        elif len(selected_frame_ids) < max_frames and len(frame_ids) >= max_frames:
            # Fill remaining with uniform sampling (fallback)
            remaining = max_frames - len(selected_frame_ids)
            all_ids = set(frame_ids)
            available = sorted(all_ids - set(selected_frame_ids))
            if len(available) > 0:
                step = len(available) / remaining
                extras = [available[int(i * step)] for i in range(remaining)]
                selected_frame_ids.extend(extras)
                selected_frame_ids = sorted(selected_frame_ids)[:max_frames]
        
        return selected_frame_ids


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, args, selector: ADADQSelector) -> List[int]:
    """
    Process a single video using ADA-DQ.
    
    Args:
        scores: List of frame scores
        frame_ids: List of frame IDs
        max_frames: Maximum frames to select
        args: Argument namespace
        selector: ADADQSelector instance
    
    Returns:
        List of selected frame IDs
    """
    # Apply ratio-based downsampling if needed
    if args.ratio > 1:
        nums = int(len(scores) / args.ratio)
        scores = [scores[num * args.ratio] for num in range(nums)]
        frame_ids = [frame_ids[num * args.ratio] for num in range(nums)]
    
    # Select frames using ADA-DQ
    selected_frames = selector.select_frames(scores, frame_ids, max_frames)
    
    return selected_frames


def main(args):
    """
    Main function to process all videos using ADA-DQ.
    """
    print("=" * 60)
    print("ADA-DQ: Shot-Aware Dual-Quota Frame Selection")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Feature Model: {args.extract_feature_model}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Beta (density weight): {args.beta}")
    print(f"Delta_min (temporal gap): {args.delta_min}s")
    print(f"L_max (recursion depth): {args.L_max}")
    print(f"Theta_stop: {args.theta_stop}")
    print(f"Rho_ref (reference density): {args.rho_ref}")
    print(f"Shot detection: {args.use_shot_detection}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading scores from: {args.score_path}")
    with open(args.score_path) as f:
        all_scores = json.load(f)
    
    print(f"Loading frames from: {args.frame_path}")
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    # Determine how many videos to process
    num_videos_to_process = len(all_scores)
    if args.num_videos is not None:
        num_videos_to_process = min(args.num_videos, len(all_scores))
        print(f"\n🎯 DEMO MODE: Processing first {num_videos_to_process} videos only")
    
    print(f"Total videos loaded: {len(all_scores)}")
    print(f"Videos to process: {num_videos_to_process}\n")
    
    # Initialize selector
    selector = ADADQSelector(args)
    
    # Process each video
    selected_frames_all = []
    
    for idx, (scores, frame_ids) in enumerate(zip(all_scores[:num_videos_to_process], 
                                                    all_frame_ids[:num_videos_to_process])):
        # Show progress
        if num_videos_to_process <= 20:
            print(f"Processing video {idx + 1}/{num_videos_to_process}...")
        elif (idx + 1) % 100 == 0:
            print(f"Processing video {idx + 1}/{num_videos_to_process}...")
        
        selected_frames = process_video(
            scores=scores,
            frame_ids=frame_ids,
            max_frames=args.max_num_frames,
            args=args,
            selector=selector
        )
        
        selected_frames_all.append(selected_frames)
    
    # Save results
    output_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_model_dir = os.path.join(output_dir, args.extract_feature_model)
    os.makedirs(output_model_dir, exist_ok=True)
    
    output_filename = f'selected_frames_ada_dq_M{args.max_num_frames}_beta{args.beta}_delta{args.delta_min}.json'
    output_path = os.path.join(output_model_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete!")
    print(f"Selected frames saved to: {output_path}")
    
    # Statistics
    frame_counts = [len(frames) for frames in selected_frames_all]
    print(f"\nStatistics:")
    print(f"  Videos processed: {len(selected_frames_all)}")
    print(f"  Avg frames selected: {np.mean(frame_counts):.2f}")
    print(f"  Min frames: {np.min(frame_counts)}")
    print(f"  Max frames: {np.max(frame_counts)}")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)