import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
import warnings

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TMASConfig:
    """Configuration for Temporal Memory-Aware Suppression"""
    mode: str
    auto_scaling: str
    auto_coverage: float
    auto_min_radius: float
    delta_strategy: str
    delta_R: float
    multiplier: float
    decay_mode: str
    custom_lambda: float
    decay_floor: float
    budget_half_life_ratio: float
    auto_max_radius: Optional[float] = None
    base_radius: Optional[float] = None


@dataclass
class PPTMASConfig:
    """Simplified PPTMAS Configuration"""
    # Peak detection
    num_scales: int = 3
    peak_threshold_percentile: float = 75.0  # Use percentile instead of std
    peak_cap_multiplier: float = 3.0         # More generous peak detection
    
    # Selection strategy
    peak_priority_weight: float = 2.0        # Boost peak scores by this factor
    min_peak_score: float = 0.6              # Only consider high-quality peaks
    
    # Revival mechanism (simplified)
    enable_revival: bool = True
    revival_start_ratio: float = 0.5         # Start revival after 50% budget used
    revival_score_boost: float = 1.5         # Multiply peak scores by this in late stage
    
    # Diversity (simplified)
    diversity_weight: float = 0.1            # Much lower than before!
    temporal_penalty_sigma: float = None     # Auto-calculate from video length
    
    # Coverage
    force_coverage: bool = True
    coverage_ratio: float = 0.2              # Reserve 20% budget for coverage
    
    # Advanced
    normalize_scores: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# BASE TMAS CALCULATOR (ORIGINAL - UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalMemorySuppressionCalculator:
    """
    Base TMAS calculator (original implementation).
    """
    
    def __init__(self, config: TMASConfig, total_frames: int, target_frames: int):
        self.config = config
        self.L = total_frames
        self.target_k = target_frames
        self.frames_selected = 0
        
        self.selection_map = {}
        self.selection_counter = 0
        
        if config.mode == 'auto':
            self.R0 = self._calculate_auto_R0()
            self.Delta_R = 0.0
        else:
            self.R0 = config.base_radius if config.base_radius else 3.0
            self.Delta_R = self._calculate_delta_R()
        
        self.lambda_decay = self._calculate_decay_rate()
        self.suppression_history = []
    
    def _calculate_auto_R0(self) -> float:
        if self.L == 0 or self.target_k == 0:
            return self.config.auto_min_radius
        
        base_spacing = self.L / self.target_k
        
        if self.config.auto_scaling == 'linear':
            R0 = base_spacing * self.config.auto_coverage
        elif self.config.auto_scaling == 'sqrt':
            R0 = np.sqrt(base_spacing) * self.config.auto_coverage
        else:
            R0 = (base_spacing ** 0.7) * self.config.auto_coverage
        
        R0 = max(self.config.auto_min_radius, R0)
        if self.config.auto_max_radius is not None:
            R0 = min(self.config.auto_max_radius, R0)
        
        return R0
    
    def _calculate_delta_R(self) -> float:
        if self.config.delta_strategy == 'fixed':
            return self.config.delta_R
        else:
            return self.R0 * self.config.multiplier
    
    def _get_delta_R_dynamic(self) -> float:
        if self.config.delta_strategy != 'adaptive':
            return self.Delta_R
        
        remaining_budget = max(1, self.target_k - self.frames_selected)
        budget_ratio = remaining_budget / self.target_k
        
        return self.R0 * self.config.multiplier * budget_ratio
    
    def _calculate_decay_rate(self) -> float:
        if self.L == 0:
            return 0.01
        
        if self.config.decay_mode == 'video_length':
            return np.log(max(2, self.L)) / self.L
        
        elif self.config.decay_mode == 'half_life':
            if self.target_k == 0:
                return 0.01
            half_life = self.L / (2 * self.target_k)
            return np.log(2) / max(1, half_life)
        
        elif self.config.decay_mode == 'quartile':
            quartile_distance = self.L / 4
            return np.log(4) / max(1, quartile_distance)
        
        elif self.config.decay_mode == 'budget_based':
            if self.target_k == 0:
                return 0.01
            half_life_selections = max(1, self.target_k * self.config.budget_half_life_ratio)
            return np.log(2) / half_life_selections
        
        else:
            return self.config.custom_lambda
    
    def get_current_radius_for_position(self, candidate_idx: int) -> float:
        """Calculate current suppression radius (base implementation)"""
        if not self.selection_map:
            if self.config.mode == 'auto':
                return self.R0
            else:
                current_delta = self._get_delta_R_dynamic()
                return self.R0 + current_delta
        
        max_radius = 0.0
        
        for selected_idx, selection_num in self.selection_map.items():
            if self.config.decay_mode == 'budget_based':
                frames_since = self.selection_counter - selection_num
                decay_distance = float(frames_since)
            else:
                decay_distance = abs(candidate_idx - selected_idx)
            
            if self.config.mode == 'auto':
                radius = self.R0 * np.exp(-self.lambda_decay * decay_distance)
            else:
                current_delta = self._get_delta_R_dynamic()
                tmas_contribution = current_delta * np.exp(-self.lambda_decay * decay_distance)
                
                if self.config.decay_floor > 0:
                    tmas_contribution = max(tmas_contribution, current_delta * self.config.decay_floor)
                
                radius = self.R0 + tmas_contribution
            
            max_radius = max(max_radius, radius)
        
        return max_radius
    
    def is_suppressed(self, candidate_idx: int) -> Tuple[bool, Optional[int]]:
        if not self.selection_map:
            return False, None
        
        effective_radius = self.get_current_radius_for_position(candidate_idx)
        
        for selected_idx in self.selection_map.keys():
            temporal_delta = abs(candidate_idx - selected_idx)
            
            if temporal_delta > 0 and temporal_delta <= effective_radius:
                self.suppression_history.append({
                    'candidate_idx': candidate_idx,
                    'suppressed_by': selected_idx,
                    'temporal_delta': temporal_delta,
                    'effective_radius': effective_radius,
                    'frames_selected': self.frames_selected
                })
                return True, selected_idx
        
        return False, None
    
    def notify_frame_selected(self, frame_idx: int):
        self.selection_counter += 1
        self.selection_map[frame_idx] = self.selection_counter
        self.frames_selected += 1
    
    def get_stats(self) -> Dict:
        stats = {
            'mode': self.config.mode,
            'decay_mode': self.config.decay_mode,
            'R0': round(self.R0, 3),
            'lambda': round(self.lambda_decay, 6),
            'total_frames': self.L,
            'target_frames': self.target_k,
            'frames_selected': self.frames_selected,
        }
        
        if self.config.mode == 'additive':
            current_delta = self._get_delta_R_dynamic()
            stats['base_radius'] = round(self.R0, 3)
            stats['delta_R'] = round(current_delta, 3)
            stats['max_radius_at_t0'] = round(self.R0 + self.Delta_R, 3)
        else:
            stats['max_radius_at_t0'] = round(self.R0, 3)
        
        if self.suppression_history:
            temporal_deltas = [s['temporal_delta'] for s in self.suppression_history]
            stats['suppression_stats'] = {
                'total_suppressions': len(self.suppression_history),
                'avg_temporal_delta': round(np.mean(temporal_deltas), 2),
                'max_temporal_delta': int(np.max(temporal_deltas)),
            }
        
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# BASE KEYFRAME SELECTOR (ORIGINAL - UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class KeyframeSelector:
    """Base keyframe selector (original implementation)"""
    
    def __init__(self, scores: np.ndarray, frame_ids: np.ndarray,
                 tmas_calculator: TemporalMemorySuppressionCalculator = None,
                 min_score_threshold: float = 0.0):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.N = len(scores)
        self.min_score_threshold = min_score_threshold
        self.tmas_calculator = tmas_calculator
        
        if self.N > 0:
            score_min, score_max = self.scores.min(), self.scores.max()
            if score_max > score_min:
                self.scores = (self.scores - score_min) / (score_max - score_min)
            else:
                self.scores = np.ones_like(self.scores) * 0.5
        
        self.valid_mask = self.scores >= min_score_threshold
        self.num_valid = np.sum(self.valid_mask)
    
    def select_keyframes(self, max_frames: int, optimize_remaining: bool = False) -> List[int]:
        if self.N == 0 or self.num_valid == 0:
            return []
        
        if self.num_valid <= max_frames:
            valid_indices = np.where(self.valid_mask)[0]
            return sorted(list(set([int(self.frame_ids[idx]) for idx in valid_indices])))
        
        selected_indices = self._sorted_tmas_selection(max_frames)
        
        if optimize_remaining and len(selected_indices) < max_frames:
            selected_indices = self._fill_remaining_slots(selected_indices, max_frames)
        
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        return sorted(list(set(selected_frame_ids)))
    
    def _sorted_tmas_selection(self, max_frames: int) -> List[int]:
        valid_indices = np.where(self.valid_mask)[0]
        
        sorted_candidates = sorted(
            [(self.scores[idx], idx) for idx in valid_indices],
            key=lambda x: (-x[0], x[1])
        )
        
        selected_indices = []
        selected_set = set()
        
        for score, idx in sorted_candidates:
            if len(selected_indices) >= max_frames:
                break
            
            if idx in selected_set:
                continue
            
            if self.tmas_calculator:
                is_suppressed, _ = self.tmas_calculator.is_suppressed(idx)
                if is_suppressed:
                    continue
            
            selected_indices.append(idx)
            selected_set.add(idx)
            
            if self.tmas_calculator:
                self.tmas_calculator.notify_frame_selected(idx)
        
        return selected_indices
    
    def _fill_remaining_slots(self, selected_indices: List[int], max_frames: int) -> List[int]:
        selected_set = set(selected_indices)
        remaining = max_frames - len(selected_indices)
        
        if remaining <= 0:
            return selected_indices
        
        valid_indices = np.where(self.valid_mask)[0]
        available = [idx for idx in valid_indices if idx not in selected_set]
        
        if not available:
            return selected_indices
        
        available_scores = [(self.scores[idx], idx) for idx in available]
        available_scores.sort(reverse=True, key=lambda x: x[0])
        
        num_to_add = min(remaining, len(available_scores))
        for i in range(num_to_add):
            selected_indices.append(available_scores[i][1])
        
        return selected_indices


# ══════════════════════════════════════════════════════════════════════════════
# FIXED PPTMAS IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class SimplifiedPeakDetector:
    """
    Robust peak detection using percentile-based thresholding.
    """
    
    def __init__(self, config: PPTMASConfig):
        self.config = config
    
    def detect_peaks(self, scores: np.ndarray, max_peaks: Optional[int] = None) -> Dict:
        """
        Detect peaks using local maxima + percentile threshold.
        """
        T = len(scores)
        
        if T < 5:
            return {
                'peak_indices': np.arange(T),
                'peak_scores': scores,
                'threshold': 0.0,
                'num_candidates': T
            }
        
        # Find local maxima with minimum prominence
        prominence = (np.max(scores) - np.min(scores)) * 0.1  # 10% of range
        peaks, properties = find_peaks(scores, prominence=prominence, distance=3)
        
        # Filter by percentile threshold
        threshold = np.percentile(scores, self.config.peak_threshold_percentile)
        
        # Keep peaks above threshold AND above minimum score
        valid_peaks = []
        for p in peaks:
            if scores[p] >= threshold and scores[p] >= self.config.min_peak_score:
                valid_peaks.append(p)
        
        valid_peaks = np.array(valid_peaks, dtype=int)
        
        # If too many peaks, take top by score
        if max_peaks is not None and len(valid_peaks) > max_peaks:
            peak_scores = scores[valid_peaks]
            top_indices = np.argsort(peak_scores)[-max_peaks:]
            valid_peaks = valid_peaks[top_indices]
        
        # Sort by position
        valid_peaks = np.sort(valid_peaks)
        
        return {
            'peak_indices': valid_peaks,
            'peak_scores': scores[valid_peaks] if len(valid_peaks) > 0 else np.array([]),
            'threshold': threshold,
            'num_candidates': len(peaks)
        }


class SimplifiedSubmodularObjective:
    """
    Simplified objective: weighted score + light temporal diversity penalty.
    
    f(S) = Σᵢ w(i)·yᵢ - λ·Σᵢ,ⱼ∈S exp(-|i-j|²/2σ²)
    
    Where w(i) = peak_weight if i is peak, else 1.0
    """
    
    def __init__(self, 
                 scores: np.ndarray, 
                 peak_indices: np.ndarray,
                 config: PPTMASConfig, 
                 target_frames: int):
        self.scores = scores
        self.T = len(scores)
        self.K = target_frames
        self.peak_set = set(peak_indices)
        
        # Weights: boost peaks
        self.weights = np.ones(self.T)
        self.weights[peak_indices] = config.peak_priority_weight
        
        # Diversity penalty sigma
        if config.temporal_penalty_sigma is None:
            self.sigma = max(3.0, self.T / (2 * self.K))  # Auto-scale
        else:
            self.sigma = config.temporal_penalty_sigma
        
        self.diversity_weight = config.diversity_weight
        
        # Selected tracking
        self.selected_indices = []
    
    def compute_marginal_gain(self, candidate_idx: int) -> float:
        """
        Compute marginal gain with lightweight diversity penalty.
        """
        # Relevance term (weighted)
        relevance = self.weights[candidate_idx] * self.scores[candidate_idx]
        
        # Diversity penalty: how close is this to existing selections?
        if len(self.selected_indices) == 0:
            penalty = 0.0
        else:
            # Compute temporal overlap with existing selections
            distances = np.abs(np.array(self.selected_indices) - candidate_idx)
            overlaps = np.exp(-distances**2 / (2 * self.sigma**2))
            penalty = np.sum(overlaps)
        
        return relevance - self.diversity_weight * penalty
    
    def update_selection(self, selected_idx: int):
        """Update after selection"""
        self.selected_indices.append(selected_idx)


class FixedPPTMASCalculator(TemporalMemorySuppressionCalculator):
    """
    Fixed PPTMAS with simplified revival.
    
    Key changes:
    1. Peaks get BOOSTED scores instead of reduced suppression
    2. Revival is score-based, not radius-based
    3. Simpler logic, fewer failure modes
    """
    
    def __init__(self,
                 tmas_config: TMASConfig,
                 pptmas_config: PPTMASConfig,
                 scores: np.ndarray,
                 peak_indices: np.ndarray,
                 total_frames: int,
                 target_frames: int):
        super().__init__(tmas_config, total_frames, target_frames)
        
        self.pptmas_config = pptmas_config
        self.scores = scores
        self.peak_set = set(peak_indices)
        
        self.revival_history = []
    
    def get_effective_score(self, candidate_idx: int) -> float:
        """
        Get effective score with revival boost.
        
        Revival: boost peak scores in late stage.
        """
        base_score = self.scores[candidate_idx]
        
        # Not a peak? Return base score
        if candidate_idx not in self.peak_set:
            return base_score
        
        # Revival mechanism: boost peaks after revival_start_ratio of budget used
        if not self.pptmas_config.enable_revival:
            return base_score
        
        progress = self.frames_selected / max(1, self.target_k)
        
        if progress < self.pptmas_config.revival_start_ratio:
            return base_score
        
        # Late stage: boost peak scores
        revival_strength = (progress - self.pptmas_config.revival_start_ratio) / \
                          (1.0 - self.pptmas_config.revival_start_ratio)
        
        boost_factor = 1.0 + revival_strength * (self.pptmas_config.revival_score_boost - 1.0)
        
        boosted_score = base_score * boost_factor
        
        if boost_factor > 1.1:
            self.revival_history.append({
                'candidate_idx': candidate_idx,
                'iteration': self.frames_selected,
                'base_score': base_score,
                'boosted_score': boosted_score,
                'boost_factor': boost_factor
            })
        
        return boosted_score
    
    def is_suppressed_peak_aware(self, candidate_idx: int) -> Tuple[bool, Optional[int], Dict]:
        """
        Check suppression with peak awareness.
        
        Peaks are LESS likely to be suppressed (check with smaller radius).
        """
        if not self.selection_map:
            return False, None, {'is_peak': candidate_idx in self.peak_set}
        
        # Base radius
        base_radius = super().get_current_radius_for_position(candidate_idx)
        
        # For peaks: use reduced radius (harder to suppress)
        is_peak = candidate_idx in self.peak_set
        if is_peak:
            effective_radius = base_radius * 0.7  # 30% reduction for peaks
        else:
            effective_radius = base_radius
        
        # Check suppression
        for selected_idx in self.selection_map.keys():
            temporal_delta = abs(candidate_idx - selected_idx)
            
            if temporal_delta > 0 and temporal_delta <= effective_radius:
                return True, selected_idx, {
                    'is_peak': is_peak,
                    'effective_radius': effective_radius,
                    'temporal_delta': temporal_delta
                }
        
        return False, None, {'is_peak': is_peak, 'effective_radius': effective_radius}
    
    def get_stats(self) -> Dict:
        """Enhanced stats with PPTMAS-specific metrics"""
        stats = super().get_stats()
        
        # PPTMAS-specific stats
        stats['pptmas'] = {
            'num_peaks': len(self.peak_set),
            'peaks_selected': sum(1 for idx in self.selection_map.keys() if idx in self.peak_set),
            'revival_events': len(self.revival_history)
        }
        
        if self.revival_history:
            boost_factors = [r['boost_factor'] for r in self.revival_history]
            stats['pptmas']['avg_boost_factor'] = round(np.mean(boost_factors), 3)
            stats['pptmas']['max_boost_factor'] = round(np.max(boost_factors), 3)
        
        return stats


class FixedPPTMASKeyframeSelector:
    """
    Fixed PPTMAS selector with proper score-based selection.
    
    Strategy:
    1. Detect peaks robustly
    2. Use SCORE + light diversity as objective (not complex log-det)
    3. Boost peak scores (especially in late stage)
    4. Add coverage grid as fallback
    """
    
    def __init__(self,
                 scores: np.ndarray,
                 frame_ids: np.ndarray,
                 pptmas_calculator: FixedPPTMASCalculator,
                 submodular_objective: SimplifiedSubmodularObjective,
                 peak_indices: np.ndarray,
                 config: PPTMASConfig):
        self.scores = scores
        self.frame_ids = frame_ids
        self.N = len(scores)
        self.pptmas_calculator = pptmas_calculator
        self.submodular_objective = submodular_objective
        self.peak_set = set(peak_indices)
        self.config = config
        
        self.selection_log = []
    
    def select_keyframes(self, max_frames: int) -> List[int]:
        """
        Select keyframes with fixed PPTMAS.
        """
        if self.N == 0:
            return []
        
        if self.N <= max_frames:
            return sorted([int(self.frame_ids[i]) for i in range(self.N)])
        
        # Phase 1: Score-based greedy selection with diversity penalty
        num_main = max_frames
        if self.config.force_coverage:
            num_main = int(max_frames * (1.0 - self.config.coverage_ratio))
        
        selected_indices = self._greedy_selection(num_main)
        
        # Phase 2: Add coverage if needed
        if self.config.force_coverage and len(selected_indices) < max_frames:
            selected_indices = self._add_coverage(selected_indices, max_frames)
        
        # Convert to frame IDs
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        return sorted(list(set(selected_frame_ids)))
    
    def _greedy_selection(self, max_frames: int) -> List[int]:
        """
        Greedy selection maximizing: score - diversity_penalty
        With peak boosting and TMAS suppression.
        """
        selected_indices = []
        selected_set = set()
        
        for iteration in range(max_frames):
            best_candidate = None
            best_gain = -np.inf
            best_info = {}
            
            # Evaluate all candidates
            for candidate_idx in range(self.N):
                if candidate_idx in selected_set:
                    continue
                
                # Check TMAS suppression (peaks have advantage)
                is_suppressed, suppressor, info = \
                    self.pptmas_calculator.is_suppressed_peak_aware(candidate_idx)
                
                if is_suppressed:
                    continue
                
                # Compute gain with simplified objective
                marginal_gain = self.submodular_objective.compute_marginal_gain(candidate_idx)
                
                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_candidate = candidate_idx
                    best_info = {
                        **info,
                        'marginal_gain': marginal_gain,
                        'raw_score': self.scores[candidate_idx],
                        'effective_score': self.pptmas_calculator.get_effective_score(candidate_idx)
                    }
            
            # Fallback: if no candidate, pick highest score among remaining
            if best_candidate is None:
                remaining = [i for i in range(self.N) if i not in selected_set]
                if not remaining:
                    break
                
                # Use EFFECTIVE scores (with revival boost)
                effective_scores = [
                    (self.pptmas_calculator.get_effective_score(i), i) 
                    for i in remaining
                ]
                effective_scores.sort(reverse=True)
                best_candidate = effective_scores[0][1]
                best_info = {
                    'fallback': True,
                    'effective_score': effective_scores[0][0]
                }
            
            # Select
            selected_indices.append(best_candidate)
            selected_set.add(best_candidate)
            
            # Update trackers
            self.pptmas_calculator.notify_frame_selected(best_candidate)
            self.submodular_objective.update_selection(best_candidate)
            
            # Log
            self.selection_log.append({
                'iteration': iteration,
                'selected_idx': best_candidate,
                'is_peak': best_candidate in self.peak_set,
                **best_info
            })
        
        return selected_indices
    
    def _add_coverage(self, selected_indices: List[int], max_frames: int) -> List[int]:
        """Add uniform coverage frames to fill budget."""
        remaining = max_frames - len(selected_indices)
        if remaining <= 0:
            return selected_indices
        
        selected_set = set(selected_indices)
        
        # Build uniform grid avoiding selected
        grid = np.linspace(0, self.N - 1, remaining * 3, dtype=int)  # 3x candidates
        candidates = [g for g in grid if g not in selected_set]
        
        # Take highest scoring grid points
        if candidates:
            candidate_scores = [(self.scores[c], c) for c in candidates]
            candidate_scores.sort(reverse=True)
            
            num_to_add = min(remaining, len(candidate_scores))
            for i in range(num_to_add):
                selected_indices.append(candidate_scores[i][1])
        
        return selected_indices
    
    def get_selection_stats(self) -> Dict:
        """Get selection statistics"""
        if not self.selection_log:
            return {}
        
        peaks_selected = sum(1 for log in self.selection_log if log.get('is_peak', False))
        
        return {
            'total_selected': len(self.selection_log),
            'peaks_selected': peaks_selected,
            'peak_ratio': peaks_selected / max(1, len(self.selection_log)),
            'avg_raw_score': np.mean([log.get('raw_score', 0) for log in self.selection_log]),
            'fallback_used': sum(1 for log in self.selection_log if log.get('fallback', False))
        }


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def process_video_pptmas(scores: List[float], 
                         frame_ids: List[int],
                         max_frames: int,
                         tmas_config: TMASConfig,
                         pptmas_config: PPTMASConfig,
                         args) -> Tuple[List[int], Dict]:
    """
    Process video with FIXED PPTMAS.
    """
    scores = np.asarray(scores, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    
    # Apply ratio sampling
    if args.ratio > 1:
        indices = np.arange(0, len(scores), args.ratio)
        scores = scores[indices]
        frame_ids = frame_ids[indices]
    
    # Handle short videos
    if len(scores) <= max_frames:
        return sorted(list(set([int(x) for x in frame_ids])))[:max_frames], {'short_video': True}
    
    # Normalize scores
    if pptmas_config.normalize_scores:
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            scores_norm = (scores - score_min) / (score_max - score_min)
        else:
            scores_norm = np.ones_like(scores) * 0.5
    else:
        scores_norm = scores
    
    # Phase 1: Robust Peak Detection
    peak_detector = SimplifiedPeakDetector(pptmas_config)
    
    max_peaks = int(pptmas_config.peak_cap_multiplier * max_frames)
    peak_result = peak_detector.detect_peaks(scores_norm, max_peaks=max_peaks)
    
    peak_indices = peak_result['peak_indices']
    
    # Phase 2: Initialize Components
    pptmas_calculator = FixedPPTMASCalculator(
        tmas_config=tmas_config,
        pptmas_config=pptmas_config,
        scores=scores_norm,
        peak_indices=peak_indices,
        total_frames=len(scores),
        target_frames=max_frames
    )
    
    submodular_objective = SimplifiedSubmodularObjective(
        scores=scores_norm,
        peak_indices=peak_indices,
        config=pptmas_config,
        target_frames=max_frames
    )
    
    # Phase 3: Selection
    selector = FixedPPTMASKeyframeSelector(
        scores=scores_norm,
        frame_ids=frame_ids,
        pptmas_calculator=pptmas_calculator,
        submodular_objective=submodular_objective,
        peak_indices=peak_indices,
        config=pptmas_config
    )
    
    selected_frames = selector.select_keyframes(max_frames=max_frames)
    
    # Statistics
    stats = pptmas_calculator.get_stats()
    stats['peak_detection'] = {
        'num_peaks_detected': int(len(peak_indices)),
        'threshold': float(peak_result['threshold']),
        'num_candidates': int(peak_result['num_candidates'])
    }
    stats['selection'] = selector.get_selection_stats()
    
    # Validation
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames!"
    assert len(selected_frames) <= max_frames, f"Too many frames: {len(selected_frames)}"
    
    return selected_frames, stats


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, tmas_config: TMASConfig, args) -> Tuple[List[int], Dict]:
    """
    Process video with standard TMAS (original implementation).
    """
    scores = np.asarray(scores, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    
    if args.ratio > 1:
        indices = np.arange(0, len(scores), args.ratio)
        scores = scores[indices]
        frame_ids = frame_ids[indices]
    
    if len(scores) <= max_frames:
        return sorted(list(set([int(x) for x in frame_ids])))[:max_frames], {'short_video': True}
    
    tmas_calculator = TemporalMemorySuppressionCalculator(
        config=tmas_config,
        total_frames=len(scores),
        target_frames=max_frames
    )
    
    selector = KeyframeSelector(
        scores=scores,
        frame_ids=frame_ids,
        tmas_calculator=tmas_calculator,
        min_score_threshold=args.min_score_threshold
    )
    
    selected_frames = selector.select_keyframes(
        max_frames=max_frames,
        optimize_remaining=args.optimize_remaining
    )
    
    stats = tmas_calculator.get_stats()
    
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames!"
    assert len(selected_frames) <= max_frames, f"Too many frames: {len(selected_frames)}"
    
    return selected_frames, stats


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_base_suppression_radius(metadata_entry: dict, args, dataset_name: str) -> float:
    """Get base suppression radius based on video category."""
    if dataset_name == 'longvideobench':
        duration_group = metadata_entry.get('duration_group', None)
        
        if duration_group == 15:
            return args.suppression_radius_15
        elif duration_group == 60:
            return args.suppression_radius_60
        elif duration_group == 600:
            return args.suppression_radius_600
        elif duration_group == 3600:
            return args.suppression_radius_3600
        else:
            return 3.0
            
    elif dataset_name == 'videomme':
        duration = metadata_entry.get('duration', None)
        
        if duration == 'short':
            return args.suppression_radius_short
        elif duration == 'medium':
            return args.suppression_radius_medium
        elif duration == 'long':
            return args.suppression_radius_long
        else:
            return 3.0
    else:
        return 3.0


def create_tmas_config(args, base_radius: Optional[float] = None) -> TMASConfig:
    """Create TMAS configuration from arguments."""
    return TMASConfig(
        mode=args.tmas_mode,
        auto_scaling=args.tmas_auto_scaling,
        auto_coverage=args.tmas_auto_coverage,
        auto_min_radius=args.tmas_auto_min_radius,
        auto_max_radius=args.tmas_auto_max_radius,
        delta_strategy=args.tmas_delta_strategy,
        delta_R=args.tmas_delta_R,
        multiplier=args.tmas_multiplier,
        base_radius=base_radius,
        decay_mode=args.tmas_decay_mode,
        custom_lambda=args.tmas_custom_lambda,
        decay_floor=args.tmas_decay_floor,
        budget_half_life_ratio=args.tmas_budget_half_life_ratio,
    )


def create_pptmas_config(args) -> PPTMASConfig:
    """Create FIXED PPTMAS configuration."""
    return PPTMASConfig(
        num_scales=3,
        peak_threshold_percentile=args.pptmas_peak_threshold_percentile,
        peak_cap_multiplier=args.pptmas_peak_cap_multiplier,
        peak_priority_weight=args.pptmas_peak_priority_weight,
        min_peak_score=args.pptmas_min_peak_score,
        enable_revival=args.pptmas_enable_revival,
        revival_start_ratio=args.pptmas_revival_start_ratio,
        revival_score_boost=args.pptmas_revival_score_boost,
        diversity_weight=args.pptmas_diversity_weight,
        force_coverage=args.pptmas_force_coverage,
        coverage_ratio=args.pptmas_coverage_ratio,
        normalize_scores=True
    )


def build_output_filename(args):
    """Build output filename."""
    if args.use_pptmas:
        name = f"selected_pptmas_{args.dataset_name}_{args.extract_feature_model}_k{args.max_num_frames}"
        name += f"_w{args.pptmas_peak_priority_weight}_d{args.pptmas_diversity_weight}"
    else:
        name = f"selected_tmas_{args.dataset_name}_{args.extract_feature_model}_k{args.max_num_frames}"
        name += f"_{args.tmas_mode}_{args.tmas_decay_mode}"
        
        if args.tmas_mode == 'auto':
            name += f"_{args.tmas_auto_scaling}_cov{args.tmas_auto_coverage}"
        elif args.tmas_mode == 'additive':
            name += f"_{args.tmas_delta_strategy}"
    
    if args.optimize_remaining:
        name += "_opt"
    
    name += ".json"
    return name


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP with Fixed PPTMAS')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='longvideobench', 
                        help='Dataset name: longvideobench or videomme')
    parser.add_argument('--extract_feature_model', type=str, default='clip', 
                        help='Feature extraction model: blip/clip/sevila')
    parser.add_argument('--score_path', type=str, 
                        default='./outscores/longvideobench/clip/scores.json',
                        help='Path to input scores JSON file')
    parser.add_argument('--frame_path', type=str, 
                        default='./outscores/longvideobench/clip/frames.json',
                        help='Path to input frame IDs JSON file')
    parser.add_argument('--metadata_path', type=str,
                        default='./datasets/longvideobench/metadata.json',
                        help='Path to metadata JSON file')
    parser.add_argument('--max_num_frames', type=int, default=32,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    
    # TMAS mode selection
    parser.add_argument('--tmas_mode', type=str, default='auto',
                        choices=['auto', 'additive'],
                        help='TMAS suppression mode')
    
    # TMAS auto mode
    parser.add_argument('--tmas_auto_scaling', type=str, default='hybrid',
                        choices=['linear', 'sqrt', 'hybrid'],
                        help='[AUTO MODE] Scaling function for R0 calculation')
    parser.add_argument('--tmas_auto_coverage', type=float, default=1.0,
                        help='[AUTO MODE] Coverage factor')
    parser.add_argument('--tmas_auto_min_radius', type=float, default=1.0,
                        help='[AUTO MODE] Minimum R0 value')
    parser.add_argument('--tmas_auto_max_radius', type=float, default=None,
                        help='[AUTO MODE] Maximum R0 value')
    
    # TMAS additive mode
    parser.add_argument('--tmas_delta_strategy', type=str, default='proportional',
                        choices=['fixed', 'proportional', 'adaptive'],
                        help='[ADDITIVE MODE] Delta_R calculation strategy')
    parser.add_argument('--tmas_delta_R', type=float, default=3.0,
                        help='[ADDITIVE MODE - fixed] Fixed Delta_R value')
    parser.add_argument('--tmas_multiplier', type=float, default=0.7,
                        help='[ADDITIVE MODE - proportional/adaptive] Multiplier')
    
    # Base radii for additive mode
    parser.add_argument('--suppression_radius_15', type=float, default=2.0)
    parser.add_argument('--suppression_radius_60', type=float, default=3.0)
    parser.add_argument('--suppression_radius_600', type=float, default=5.0)
    parser.add_argument('--suppression_radius_3600', type=float, default=8.0)
    parser.add_argument('--suppression_radius_short', type=float, default=2.0)
    parser.add_argument('--suppression_radius_medium', type=float, default=3.0)
    parser.add_argument('--suppression_radius_long', type=float, default=5.0)
    
    # TMAS decay configuration
    parser.add_argument('--tmas_decay_mode', type=str, default='half_life',
                        choices=['video_length', 'half_life', 'quartile', 'custom', 'budget_based'],
                        help='Decay rate calculation method')
    parser.add_argument('--tmas_custom_lambda', type=float, default=0.01,
                        help='[DECAY - custom] Manual decay rate')
    parser.add_argument('--tmas_budget_half_life_ratio', type=float, default=0.25,
                        help='[DECAY - budget_based] Half-life as ratio of budget')
    parser.add_argument('--tmas_decay_floor', type=float, default=0.0,
                        help='Minimum TMAS contribution')
    
    # Output
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (None = all)')
    parser.add_argument('--min_score_threshold', type=float, default=0.0,
                        help='Minimum normalized score threshold (0-1)')
    parser.add_argument('--optimize_remaining', action='store_true', default=True,
                        help='Fill remaining slots with best frames')
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIXED PPTMAS ARGUMENTS
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--use_pptmas', action='store_true', default=True,
                        help='Enable Peak-Prioritized TMAS')
    
    # Peak detection (simplified)
    parser.add_argument('--pptmas_peak_threshold_percentile', type=float, default=75.0,
                        help='Peak threshold percentile [50-90]')
    parser.add_argument('--pptmas_peak_cap_multiplier', type=float, default=3.0,
                        help='Max peaks = multiplier * K')
    parser.add_argument('--pptmas_min_peak_score', type=float, default=0.6,
                        help='Minimum score to be considered a peak')
    
    # Selection strategy
    parser.add_argument('--pptmas_peak_priority_weight', type=float, default=2.0,
                        help='Score multiplier for peaks [1.5-3.0]')
    
    # Revival
    parser.add_argument('--pptmas_enable_revival', action='store_true', default=True,
                        help='Enable late-stage peak revival')
    parser.add_argument('--pptmas_revival_start_ratio', type=float, default=0.5,
                        help='Start revival after this fraction of budget')
    parser.add_argument('--pptmas_revival_score_boost', type=float, default=1.5,
                        help='Peak score boost in late stage')
    
    # Diversity
    parser.add_argument('--pptmas_diversity_weight', type=float, default=0.1,
                        help='Diversity penalty weight [0.05-0.3]')
    
    # Coverage
    parser.add_argument('--pptmas_force_coverage', action='store_true', default=True,
                        help='Force uniform coverage')
    parser.add_argument('--pptmas_coverage_ratio', type=float, default=0.2,
                        help='Reserve this fraction for coverage grid')
    
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    """Main processing function."""
    print("=" * 80)
    if args.use_pptmas:
        print("🚀 FIXED Peak-Prioritized TMAS (PPTMAS)")
    else:
        print("🚀 Standard TMAS")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    with open(args.score_path) as f:
        all_scores = json.load(f)
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    with open(args.metadata_path) as f:
        metadata = json.load(f)
    
    num_videos = len(all_scores)
    if args.num_videos:
        num_videos = min(args.num_videos, num_videos)
    
    print(f"Processing {num_videos} videos")
    
    if args.use_pptmas:
        print("\n📊 Fixed PPTMAS Configuration:")
        print(f"  Peak threshold: {args.pptmas_peak_threshold_percentile}th percentile")
        print(f"  Peak priority weight: {args.pptmas_peak_priority_weight}x")
        print(f"  Revival enabled: {args.pptmas_enable_revival}")
        print(f"  Diversity weight: {args.pptmas_diversity_weight}")
        print(f"  Coverage ratio: {args.pptmas_coverage_ratio}")
    
    print("\n")
    
    # Create configs
    pptmas_config = create_pptmas_config(args) if args.use_pptmas else None
    
    # Process videos
    selected_frames_all = []
    all_stats = []
    
    for idx in range(num_videos):
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{num_videos}...")
        
        try:
            # Get base radius for TMAS
            if args.tmas_mode == 'additive':
                base_radius = get_base_suppression_radius(metadata[idx], args, args.dataset_name)
            else:
                base_radius = None
            
            tmas_config = create_tmas_config(args, base_radius)
            
            # Process with PPTMAS or standard TMAS
            if args.use_pptmas:
                selected_frames, stats = process_video_pptmas(
                    scores=all_scores[idx],
                    frame_ids=all_frame_ids[idx],
                    max_frames=args.max_num_frames,
                    tmas_config=tmas_config,
                    pptmas_config=pptmas_config,
                    args=args
                )
            else:
                selected_frames, stats = process_video(
                    scores=all_scores[idx],
                    frame_ids=all_frame_ids[idx],
                    max_frames=args.max_num_frames,
                    tmas_config=tmas_config,
                    args=args
                )
            
            selected_frames_all.append(selected_frames)
            all_stats.append(stats)
            
        except Exception as e:
            print(f"Error processing video {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            selected_frames_all.append([])
            all_stats.append({'error': str(e)})
    
    # Save results
    output_dir = os.path.dirname(args.output_file) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, build_output_filename(args))
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"✅ Stats saved to: {stats_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print("=" * 80)
    
    # Frame count statistics
    frame_counts = [len(f) for f in selected_frames_all if len(f) > 0]
    if frame_counts:
        print(f"\n📹 Frame Selection:")
        print(f"  Videos processed: {len(frame_counts)}")
        print(f"  Avg frames/video: {np.mean(frame_counts):.2f}")
        print(f"  Std frames/video: {np.std(frame_counts):.2f}")
        print(f"  Min/Max frames: {np.min(frame_counts)}/{np.max(frame_counts)}")
    
    # PPTMAS-specific statistics
    if args.use_pptmas:
        valid_stats = [s for s in all_stats if 'error' not in s and 'short_video' not in s]
        
        if valid_stats:
            print(f"\n🎯 PPTMAS Statistics:")
            
            # Peak detection stats
            if 'peak_detection' in valid_stats[0]:
                avg_peaks = np.mean([s['peak_detection']['num_peaks_detected'] for s in valid_stats])
                print(f"  Avg peaks detected: {avg_peaks:.1f}")
            
            # Selection stats
            if 'selection' in valid_stats[0]:
                peak_ratios = [s['selection'].get('peak_ratio', 0) for s in valid_stats 
                              if 'selection' in s]
                if peak_ratios:
                    print(f"  Avg peak selection ratio: {np.mean(peak_ratios):.2%}")
                
                avg_scores = [s['selection'].get('avg_raw_score', 0) for s in valid_stats 
                             if 'selection' in s]
                if avg_scores:
                    print(f"  Avg raw score: {np.mean(avg_scores):.3f}")
            
            # Revival stats
            if 'pptmas' in valid_stats[0]:
                revival_events = [s['pptmas'].get('revival_events', 0) for s in valid_stats]
                total_revivals = sum(revival_events)
                videos_with_revivals = sum(1 for x in revival_events if x > 0)
                
                if total_revivals > 0:
                    print(f"\n💫 Revival Events:")
                    print(f"  Total revival events: {total_revivals}")
                    print(f"  Videos with revivals: {videos_with_revivals} ({videos_with_revivals/len(valid_stats):.1%})")
    
    # Error statistics
    error_count = sum(1 for s in all_stats if 'error' in s)
    short_video_count = sum(1 for s in all_stats if 'short_video' in s)
    
    if error_count > 0 or short_video_count > 0:
        print(f"\n⚠️  Processing Notes:")
        if error_count > 0:
            print(f"  Videos with errors: {error_count}")
        if short_video_count > 0:
            print(f"  Short videos (≤K frames): {short_video_count}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)