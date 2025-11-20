import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

def parse_arguments():
    parser = argparse.ArgumentParser(description='ADA-DQ: Shot-Aware Dual-Quota Frame Selection')
    
    parser.add_argument('--dataset_name', type=str, default='videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip')
    parser.add_argument('--score_path', type=str, default='./outscores/videomme/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/videomme/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=20)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='./selected_frames')
    parser.add_argument('--num_videos', type=int, default=None)
    
    parser.add_argument('--L_max', type=int, default=5)
    parser.add_argument('--theta_stop', type=float, default=0.3)
    parser.add_argument('--theta_a', type=float, default=None)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--rho_ref', type=float, default=3.0)
    parser.add_argument('--delta_min', type=float, default=3.0)
    parser.add_argument('--theta_p_percentile', type=float, default=75.0)
    parser.add_argument('--use_shot_detection', action='store_true')
    parser.add_argument('--shot_threshold', type=float, default=0.15)
    
    return parser.parse_args()


@dataclass
class Bin:
    idx: np.ndarray
    m: int
    level: int


class ShotDetector:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
    
    def detect(self, scores: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
        if len(scores) < 3:
            return [(times[0], times[-1])]
        
        grad = np.abs(np.diff(scores))
        grad_norm = grad / (grad.max() + 1e-6)
        
        boundaries = [0]
        for i in range(1, len(grad)):
            if grad_norm[i] > self.threshold:
                boundaries.append(i)
        boundaries.append(len(scores) - 1)
        
        shots = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            shots.append((times[start_idx], times[end_idx]))
        
        return shots


class DensityCalculator:
    def __init__(self, theta_p: float, rho_ref: float = 3.0):
        self.theta_p = theta_p
        self.rho_ref = rho_ref
    
    def compute_rho(self, scores: np.ndarray, times: np.ndarray) -> float:
        if len(scores) == 0:
            return 0.0
        
        num_peaks = np.sum(scores >= self.theta_p)
        duration = max(times[-1] - times[0], 1e-6)
        rho = num_peaks / duration
        
        return rho
    
    def compute_rho_shot_aware(self, scores: np.ndarray, times: np.ndarray, 
                                shots: List[Tuple[float, float]]) -> float:
        if len(shots) == 0:
            return self.compute_rho(scores, times)
        
        active_shots = 0
        for shot_start, shot_end in shots:
            mask = (times >= shot_start) & (times <= shot_end)
            shot_scores = scores[mask]
            
            if len(shot_scores) > 0 and shot_scores.max() >= self.theta_p:
                active_shots += 1
        
        rho_shot = active_shots / max(len(shots), 1)
        return rho_shot * self.rho_ref


class TemporalNMSSelector:
    def __init__(self, delta_min: float):
        self.delta_min = delta_min
    
    def select(self, scores: np.ndarray, times: np.ndarray, 
               indices: np.ndarray, k: int) -> List[int]:
        if len(scores) == 0 or k <= 0:
            return []
        
        order = np.argsort(scores)[::-1]
        
        selected = []
        for idx in order:
            if len(selected) >= k:
                break
            
            current_time = times[idx]
            valid = True
            for sel_idx in selected:
                if abs(current_time - times[sel_idx]) < self.delta_min:
                    valid = False
                    break
            
            if valid:
                selected.append(int(idx))  # Convert to Python int
        
        return selected


class ADADQSelector:
    def __init__(self, args):
        self.L_max = args.L_max
        self.theta_stop = args.theta_stop
        self.beta = args.beta
        self.rho_ref = args.rho_ref
        self.delta_min = args.delta_min
        self.theta_a = args.theta_a
        self.theta_p_percentile = args.theta_p_percentile
        self.use_shot_detection = args.use_shot_detection
        
        self.shot_detector = ShotDetector(threshold=args.shot_threshold)
        self.nms_selector = TemporalNMSSelector(delta_min=args.delta_min)
        
        self.density_calc = None
        self.shots = []
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            return (scores - score_min) / (score_max - score_min)
        else:
            return np.ones_like(scores) * 0.5
    
    def _is_active(self, scores: np.ndarray, times: np.ndarray) -> bool:
        if len(scores) == 0:
            return False
        
        max_score = scores.max()
        rho = self.density_calc.compute_rho(scores, times)
        
        return (max_score >= self.theta_a) or (rho > 0)
    
    def _compute_m_alloc(self, scores: np.ndarray, times: np.ndarray, m: int) -> int:
        m_min = 1 if self._is_active(scores, times) else 0
        
        rho = self.density_calc.compute_rho(scores, times)
        m_bonus = int(round(self.beta * (rho / max(self.rho_ref, 1e-6))))
        
        m_alloc = max(0, min(m, m_min + m_bonus))
        return m_alloc
    
    def _should_stop_splitting(self, scores: np.ndarray, m: int, level: int) -> bool:
        if level >= self.L_max or m <= 1 or len(scores) <= m:
            return True
        
        bin_mean = scores.mean()
        top_m_scores = np.partition(scores, -min(m, len(scores)))[-min(m, len(scores)):]
        top_mean = top_m_scores.mean()
        
        mean_diff = top_mean - bin_mean
        return mean_diff >= self.theta_stop
    
    def _split_bin(self, idx: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(idx) <= 1:
            return idx, np.array([], dtype=int)
        
        bin_times = times[idx]
        mid_time = (bin_times[0] + bin_times[-1]) / 2.0
        
        left_mask = bin_times <= mid_time
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        
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
        if m_rem <= 0:
            return 0, 0
        
        mass_left = scores[left_idx].sum() + 1e-6
        mass_right = scores[right_idx].sum() + 1e-6
        total_mass = mass_left + mass_right
        
        m_left = int(round(m_rem * mass_left / total_mass))
        m_right = m_rem - m_left
        
        return m_left, m_right
    
    def _recurse(self, bin_idx: np.ndarray, m: int, level: int,
                 scores: np.ndarray, times: np.ndarray, 
                 frame_ids: np.ndarray, selected: set):
        if m <= 0 or len(bin_idx) == 0:
            return
        
        bin_scores = scores[bin_idx]
        bin_times = times[bin_idx]
        
        if self._should_stop_splitting(bin_scores, m, level):
            selected_local = self.nms_selector.select(
                bin_scores, bin_times, np.arange(len(bin_idx)), m
            )
            for local_idx in selected_local:
                selected.add(int(bin_idx[local_idx]))  # Convert to Python int
            return
        
        # Dual-quota allocation
        m_alloc = self._compute_m_alloc(bin_scores, bin_times, m)
        
        if m_alloc > 0:
            selected_local = self.nms_selector.select(
                bin_scores, bin_times, np.arange(len(bin_idx)), m_alloc
            )
            for local_idx in selected_local:
                selected.add(int(bin_idx[local_idx]))  # Convert to Python int
        
        m_rem = m - m_alloc
        if m_rem <= 0:
            return
        
        # Split and recurse
        left_idx, right_idx = self._split_bin(bin_idx, times)
        m_left, m_right = self._distribute_budget(scores, left_idx, right_idx, m_rem)
        
        if m_left > 0 and len(left_idx) > 0:
            self._recurse(left_idx, m_left, level + 1, 
                         scores, times, frame_ids, selected)
        
        if m_right > 0 and len(right_idx) > 0:
            self._recurse(right_idx, m_right, level + 1, 
                         scores, times, frame_ids, selected)
    
    def select_frames(self, scores: List[float], frame_ids: List[int], 
                     max_frames: int) -> List[int]:
        scores = np.array(scores, dtype=np.float64)
        frame_ids = np.array(frame_ids, dtype=np.int64)
        
        if len(scores) <= max_frames:
            return [int(fid) for fid in sorted(frame_ids)]  # Convert to Python int
        
        scores_norm = self._normalize_scores(scores)
        times = np.arange(len(scores), dtype=np.float64)
        
        if self.theta_a is None:
            self.theta_a = np.percentile(scores_norm, 85)
        
        theta_p = np.percentile(scores_norm, self.theta_p_percentile)
        self.density_calc = DensityCalculator(theta_p=theta_p, rho_ref=self.rho_ref)
        
        if self.use_shot_detection:
            self.shots = self.shot_detector.detect(scores_norm, times)
        else:
            self.shots = []
        
        root_idx = np.arange(len(scores))
        selected = set()
        
        self._recurse(
            bin_idx=root_idx,
            m=max_frames,
            level=0,
            scores=scores_norm,
            times=times,
            frame_ids=frame_ids,
            selected=selected
        )
        
        # Convert selected indices to frame IDs
        selected_frame_ids = sorted([int(frame_ids[idx]) for idx in selected])
        
        if len(selected_frame_ids) > max_frames:
            selected_frame_ids = selected_frame_ids[:max_frames]
        elif len(selected_frame_ids) < max_frames and len(frame_ids) >= max_frames:
            remaining = max_frames - len(selected_frame_ids)
            all_ids = set([int(fid) for fid in frame_ids])
            available = sorted(all_ids - set(selected_frame_ids))
            if len(available) > 0:
                step = max(1, len(available) / remaining)
                extras = [available[int(i * step)] for i in range(min(remaining, len(available)))]
                selected_frame_ids.extend(extras)
                selected_frame_ids = sorted(selected_frame_ids)[:max_frames]
        
        return selected_frame_ids


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, args, selector: ADADQSelector) -> List[int]:
    if args.ratio > 1:
        nums = int(len(scores) / args.ratio)
        scores = [scores[num * args.ratio] for num in range(nums)]
        frame_ids = [frame_ids[num * args.ratio] for num in range(nums)]
    
    selected_frames = selector.select_frames(scores, frame_ids, max_frames)
    return selected_frames


def main(args):
    print("=" * 60)
    print("ADA-DQ: Shot-Aware Dual-Quota Frame Selection")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Feature Model: {args.extract_feature_model}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Beta: {args.beta}, Delta_min: {args.delta_min}s")
    print(f"L_max: {args.L_max}, Theta_stop: {args.theta_stop}")
    print("=" * 60)
    
    with open(args.score_path) as f:
        all_scores = json.load(f)
    
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    num_videos_to_process = len(all_scores)
    if args.num_videos is not None:
        num_videos_to_process = min(args.num_videos, len(all_scores))
        print(f"\n🎯 Processing first {num_videos_to_process} videos")
    
    print(f"Total videos: {len(all_scores)}")
    print(f"Processing: {num_videos_to_process}\n")
    
    selector = ADADQSelector(args)
    selected_frames_all = []
    
    for idx, (scores, frame_ids) in enumerate(zip(all_scores[:num_videos_to_process], 
                                                    all_frame_ids[:num_videos_to_process])):
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
    print(f"Saved to: {output_path}")
    
    frame_counts = [len(frames) for frames in selected_frames_all]
    print(f"\nStatistics:")
    print(f"  Videos: {len(selected_frames_all)}")
    print(f"  Avg frames: {np.mean(frame_counts):.2f}")
    print(f"  Min/Max: {np.min(frame_counts)}/{np.max(frame_counts)}")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)