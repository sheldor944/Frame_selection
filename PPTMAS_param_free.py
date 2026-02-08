import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP with TMAS + Curvature-Based Self-Tuning')
    
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
    parser.add_argument('--max_num_frames', type=int, default=64,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    
    # ═══════════════════════════════════════════════════════════════════════
    # CURVATURE-BASED SELF-TUNING
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--use_curvature', action='store_true', default=True,
                        help='Enable curvature-based self-tuning suppression')
    
    parser.add_argument('--curvature_method', type=str, default='laplacian',
                        choices=['second_derivative', 'laplacian', 'gradient_change'],
                        help='Method to compute curvature')
    
    parser.add_argument('--curvature_smoothing', type=int, default=1,
                        help='Gaussian smoothing window size for curvature (0=disabled)')
    
    parser.add_argument('--curvature_normalize', type=str, default='max',
                        choices=['max', 'std', 'iqr', 'none'],
                        help='Normalization method for curvature values')
    
    parser.add_argument('--curvature_clip_percentile', type=float, default=95.0,
                        help='Percentile for clipping outlier curvatures (0-100)')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS MODE SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_mode', type=str, default='auto',
                        choices=['auto', 'additive'],
                        help='''TMAS suppression mode:
                        ★ auto: R0 automatically derived from video length & target frames
                        ★ additive: Base radius (per-group) + TMAS temporal decay on top''')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS AUTO MODE - R0 CALCULATION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_auto_scaling', type=str, default='hybrid',
                        choices=['linear', 'sqrt', 'hybrid'],
                        help='[AUTO MODE] Scaling function for R0 calculation')
    
    parser.add_argument('--tmas_auto_coverage', type=float, default=1.,
                        help='[AUTO MODE] Coverage factor (0.3 - 1.5)')
    
    parser.add_argument('--tmas_auto_min_radius', type=float, default=1.0,
                        help='[AUTO MODE] Minimum R0 value (safety floor)')
    
    parser.add_argument('--tmas_auto_max_radius', type=float, default=None,
                        help='[AUTO MODE] Maximum R0 value (None = no cap)')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS ADDITIVE MODE - BASE + DELTA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_delta_strategy', type=str, default='proportional',
                        choices=['fixed', 'proportional', 'adaptive'],
                        help='[ADDITIVE MODE] How to calculate Delta_R')
    
    parser.add_argument('--tmas_delta_R', type=float, default=3.0,
                        help='[ADDITIVE MODE - fixed strategy] Fixed Delta_R value')
    
    parser.add_argument('--tmas_multiplier', type=float, default=0.7,
                        help='[ADDITIVE MODE - proportional/adaptive] Multiplier for Delta_R')
    
    # Base radii for additive mode
    parser.add_argument('--suppression_radius_15', type=float, default=2.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=15')
    parser.add_argument('--suppression_radius_60', type=float, default=3.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=60')
    parser.add_argument('--suppression_radius_600', type=float, default=5.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=600')
    parser.add_argument('--suppression_radius_3600', type=float, default=8.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=3600')
    parser.add_argument('--suppression_radius_short', type=float, default=2.0,
                        help='[ADDITIVE MODE] Base radius for short videos')
    parser.add_argument('--suppression_radius_medium', type=float, default=3.0,
                        help='[ADDITIVE MODE] Base radius for medium videos')
    parser.add_argument('--suppression_radius_long', type=float, default=5.0,
                        help='[ADDITIVE MODE] Base radius for long videos')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS DECAY CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_decay_mode', type=str, default='half_life',
                        choices=['video_length', 'half_life', 'quartile', 'custom', 'budget_based'],
                        help='Decay rate (λ) calculation method')
    
    parser.add_argument('--tmas_custom_lambda', type=float, default=0.01,
                        help='[DECAY - custom mode] Manual decay rate λ value')
    
    parser.add_argument('--tmas_budget_half_life_ratio', type=float, default=0.25,
                        help='[DECAY - budget_based mode] Half-life as ratio of budget')
    
    parser.add_argument('--tmas_decay_floor', type=float, default=0.0,
                        help='Minimum TMAS contribution (0.0 - 1.0)')
    
    # ═══════════════════════════════════════════════════════════════════════
    # OTHER PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--output_file', type=str, default='./selected_frames_vmme',
                        help='Output directory')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (None = all)')
    parser.add_argument('--min_score_threshold', type=float, default=0,
                        help='Minimum normalized score threshold (0-1)')
    parser.add_argument('--optimize_remaining', action='store_true', default=True,
                        help='Fill remaining slots with best frames (no suppression)')
    
    return parser.parse_args()


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
    # Curvature parameters
    use_curvature: bool = False
    curvature_method: str = 'second_derivative'
    curvature_smoothing: int = 0
    curvature_normalize: str = 'max'
    curvature_clip_percentile: float = 95.0


class CurvatureCalculator:
    """
    Computes curvature (κ) from score signal for self-tuning suppression.
    
    Theory:
        At local maxima: y''(i) < 0 and |y''(i)| is large → κ is large
        In flat regions: y''(i) ≈ 0 → κ is small
        
        Modified radius: R_i = R_TMAS / (1 + κ_i)
        
        This ensures peaks have small suppression radius (protected),
        while flat regions maintain standard TMAS behavior.
    """
    
    def __init__(self, scores: np.ndarray, config: TMASConfig):
        """
        Initialize curvature calculator.
        
        Args:
            scores: Normalized score signal [0, 1]
            config: TMAS configuration
        """
        self.scores = scores
        self.config = config
        self.N = len(scores)
        
        # Compute curvature map
        self.curvature = self._compute_curvature()
        
        # Normalize curvature
        self.curvature = self._normalize_curvature(self.curvature)
        
        # Statistics
        self.stats = self._compute_stats()
    
    def _compute_curvature(self) -> np.ndarray:
        """
        Compute curvature using specified method.
        
        Returns:
            Array of curvature magnitudes |y''(i)|
        """
        if self.N < 3:
            return np.zeros(self.N)
        
        if self.config.curvature_method == 'second_derivative':
            curvature = self._second_derivative()
        elif self.config.curvature_method == 'laplacian':
            curvature = self._laplacian()
        elif self.config.curvature_method == 'gradient_change':
            curvature = self._gradient_change()
        else:
            curvature = self._second_derivative()
        
        # Apply smoothing if requested
        if self.config.curvature_smoothing > 0:
            curvature = self._smooth_curvature(curvature)
        
        return np.abs(curvature)  # Take magnitude
    
    def _second_derivative(self) -> np.ndarray:
        """
        Compute second derivative: y''(i) ≈ y[i+1] - 2*y[i] + y[i-1]
        
        Returns:
            Second derivative array
        """
        curvature = np.zeros(self.N)
        
        for i in range(1, self.N - 1):
            curvature[i] = self.scores[i + 1] - 2 * self.scores[i] + self.scores[i - 1]
        
        # Boundary conditions (forward/backward difference)
        if self.N >= 3:
            curvature[0] = self.scores[1] - self.scores[0]
            curvature[-1] = self.scores[-1] - self.scores[-2]
        
        return curvature
    
    def _laplacian(self) -> np.ndarray:
        """
        Compute discrete Laplacian (same as second derivative in 1D).
        
        Returns:
            Laplacian array
        """
        return self._second_derivative()
    
    def _gradient_change(self) -> np.ndarray:
        """
        Compute change in gradient: ∇²y ≈ (∇y[i] - ∇y[i-1])
        
        Returns:
            Gradient change array
        """
        if self.N < 2:
            return np.zeros(self.N)
        
        # Compute gradient
        gradient = np.gradient(self.scores)
        
        # Compute change in gradient
        curvature = np.gradient(gradient)
        
        return curvature
    
    def _smooth_curvature(self, curvature: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to curvature.
        
        Args:
            curvature: Raw curvature values
            
        Returns:
            Smoothed curvature
        """
        from scipy.ndimage import gaussian_filter1d
        
        sigma = self.config.curvature_smoothing / 3.0  # Window → sigma
        return gaussian_filter1d(curvature, sigma=sigma, mode='nearest')
    
    def _normalize_curvature(self, curvature: np.ndarray) -> np.ndarray:
        """
        Normalize curvature to be scale-invariant.
        
        Args:
            curvature: Raw curvature magnitudes
            
        Returns:
            Normalized curvature
        """
        if self.N == 0:
            return curvature
        
        # Clip outliers
        if self.config.curvature_clip_percentile < 100.0:
            clip_value = np.percentile(curvature, self.config.curvature_clip_percentile)
            curvature = np.clip(curvature, 0, clip_value)
        
        # Normalize
        if self.config.curvature_normalize == 'max':
            max_val = np.max(curvature)
            if max_val > 1e-10:
                curvature = curvature / max_val
        
        elif self.config.curvature_normalize == 'std':
            std_val = np.std(curvature)
            if std_val > 1e-10:
                curvature = (curvature - np.mean(curvature)) / std_val
                curvature = np.abs(curvature)  # Take magnitude after standardization
        
        elif self.config.curvature_normalize == 'iqr':
            q75, q25 = np.percentile(curvature, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-10:
                curvature = curvature / iqr
        
        # else: 'none' - no normalization
        
        return curvature
    
    def get_curvature(self, idx: int) -> float:
        """
        Get curvature at specific frame index.
        
        Args:
            idx: Frame index
            
        Returns:
            Curvature magnitude κ_i
        """
        if 0 <= idx < self.N:
            return float(self.curvature[idx])
        return 0.0
    
    def _compute_stats(self) -> Dict:
        """Compute curvature statistics."""
        if self.N == 0:
            return {}
        
        return {
            'mean': float(np.mean(self.curvature)),
            'std': float(np.std(self.curvature)),
            'min': float(np.min(self.curvature)),
            'max': float(np.max(self.curvature)),
            'median': float(np.median(self.curvature)),
            'q25': float(np.percentile(self.curvature, 25)),
            'q75': float(np.percentile(self.curvature, 75)),
        }
    
    def get_peak_indices(self, threshold_percentile: float = 75.0) -> List[int]:
        """
        Get indices of detected peaks (high curvature).
        
        Args:
            threshold_percentile: Percentile threshold for peak detection
            
        Returns:
            List of peak indices
        """
        if self.N == 0:
            return []
        
        threshold = np.percentile(self.curvature, threshold_percentile)
        peak_indices = np.where(self.curvature >= threshold)[0]
        
        return peak_indices.tolist()


class TemporalMemorySuppressionCalculator:
    """
    Temporal Memory-Aware Suppression (TMAS) Calculator with curvature-based self-tuning.
    
    Modified radius formula:
        R_i = (R_0 * exp(-λ * Δt)) / (1 + κ_i)
        
    where κ_i is the curvature magnitude at frame i.
    """
    
    def __init__(self, config: TMASConfig, total_frames: int, target_frames: int,
                 curvature_calculator: Optional[CurvatureCalculator] = None):
        """
        Initialize TMAS calculator.
        
        Args:
            config: TMAS configuration
            total_frames: Total number of frames in video
            target_frames: Target number of frames to select
            curvature_calculator: Optional curvature calculator for self-tuning
        """
        self.config = config
        self.L = total_frames
        self.target_k = target_frames
        self.frames_selected = 0
        self.curvature_calculator = curvature_calculator
        
        # Global tracking map: frame_idx -> selection_number
        self.selection_map = {}
        self.selection_counter = 0
        
        # Calculate initial radius based on mode
        if config.mode == 'auto':
            self.R0 = self._calculate_auto_R0()
            self.Delta_R = 0.0
        else:  # 'additive'
            self.R0 = config.base_radius if config.base_radius else 3.0
            self.Delta_R = self._calculate_delta_R()
        
        # Calculate decay rate
        self.lambda_decay = self._calculate_decay_rate()
        
        # Statistics
        self.suppression_history = []
        self.curvature_adjustments = []
    
    def _calculate_auto_R0(self) -> float:
        """Calculate R0 for auto mode."""
        if self.L == 0 or self.target_k == 0:
            return self.config.auto_min_radius
        
        base_spacing = self.L / self.target_k
        
        if self.config.auto_scaling == 'linear':
            R0 = base_spacing * self.config.auto_coverage
        elif self.config.auto_scaling == 'sqrt':
            R0 = np.sqrt(base_spacing) * self.config.auto_coverage
        else:  # 'hybrid'
            R0 = (base_spacing ** 0.7) * self.config.auto_coverage
        
        R0 = max(self.config.auto_min_radius, R0)
        if self.config.auto_max_radius is not None:
            R0 = min(self.config.auto_max_radius, R0)
        
        return R0
    
    def _calculate_delta_R(self) -> float:
        """Calculate Delta_R for additive mode."""
        if self.config.delta_strategy == 'fixed':
            return self.config.delta_R
        else:  # 'proportional' or 'adaptive'
            return self.R0 * self.config.multiplier
    
    def _get_delta_R_dynamic(self) -> float:
        """Calculate Delta_R dynamically for adaptive strategy."""
        if self.config.delta_strategy != 'adaptive':
            return self.Delta_R
        
        remaining_budget = max(1, self.target_k - self.frames_selected)
        budget_ratio = remaining_budget / self.target_k
        
        return self.R0 * self.config.multiplier * budget_ratio
    
    def _calculate_decay_rate(self) -> float:
        """Calculate decay rate λ."""
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
        
        else:  # 'custom'
            return self.config.custom_lambda
    
    def get_current_radius_for_position(self, candidate_idx: int) -> float:
        """
        Calculate current suppression radius at candidate position.
        
        With curvature: R_i = R_TMAS / (1 + κ_i)
        Without curvature: R_i = R_TMAS
        
        Args:
            candidate_idx: Position to check
            
        Returns:
            Current effective radius at this position
        """
        if not self.selection_map:
            # No frames selected yet, return maximum radius
            if self.config.mode == 'auto':
                base_radius = self.R0
            else:
                current_delta = self._get_delta_R_dynamic()
                base_radius = self.R0 + current_delta
            
            # Apply curvature adjustment
            if self.config.use_curvature and self.curvature_calculator:
                kappa = self.curvature_calculator.get_curvature(candidate_idx)
                return base_radius / (1.0 + kappa)
            
            return base_radius
        
        # Find maximum suppression from all selected frames
        max_radius = 0.0
        
        for selected_idx, selection_num in self.selection_map.items():
            # Calculate distance based on decay mode
            if self.config.decay_mode == 'budget_based':
                frames_since = self.selection_counter - selection_num
                decay_distance = float(frames_since)
            else:
                decay_distance = abs(candidate_idx - selected_idx)
            
            # Calculate base TMAS radius
            if self.config.mode == 'auto':
                base_radius = self.R0 * np.exp(-self.lambda_decay * decay_distance)
            else:  # 'additive'
                current_delta = self._get_delta_R_dynamic()
                tmas_contribution = current_delta * np.exp(-self.lambda_decay * decay_distance)
                
                # Apply decay floor
                if self.config.decay_floor > 0:
                    tmas_contribution = max(tmas_contribution, current_delta * self.config.decay_floor)
                
                base_radius = self.R0 + tmas_contribution
            
            # Apply curvature adjustment: R_i = R_TMAS / (1 + κ_i)
            if self.config.use_curvature and self.curvature_calculator:
                kappa = self.curvature_calculator.get_curvature(candidate_idx)
                radius = base_radius / (1.0 + kappa)
                
                # Track adjustment
                if len(self.curvature_adjustments) < 100:  # Limit storage
                    self.curvature_adjustments.append({
                        'candidate_idx': candidate_idx,
                        'base_radius': base_radius,
                        'curvature': kappa,
                        'adjusted_radius': radius,
                        'reduction_ratio': radius / base_radius if base_radius > 0 else 1.0
                    })
            else:
                radius = base_radius
            
            max_radius = max(max_radius, radius)
        
        return max_radius
    
    def is_suppressed(self, candidate_idx: int) -> Tuple[bool, Optional[int]]:
        """
        Check if candidate is suppressed by any selected frame.
        
        Args:
            candidate_idx: Index to check
            
        Returns:
            (is_suppressed, suppressing_frame_idx)
        """
        if not self.selection_map:
            return False, None
        
        effective_radius = self.get_current_radius_for_position(candidate_idx)
        
        # Check each selected frame
        for selected_idx in self.selection_map.keys():
            temporal_delta = abs(candidate_idx - selected_idx)
            
            if temporal_delta > 0 and temporal_delta <= effective_radius:
                # This frame is suppressed
                curvature = 0.0
                if self.config.use_curvature and self.curvature_calculator:
                    curvature = self.curvature_calculator.get_curvature(candidate_idx)
                
                self.suppression_history.append({
                    'candidate_idx': candidate_idx,
                    'suppressed_by': selected_idx,
                    'temporal_delta': temporal_delta,
                    'effective_radius': effective_radius,
                    'frames_selected': self.frames_selected,
                    'curvature': curvature
                })
                return True, selected_idx
        
        return False, None
    
    def notify_frame_selected(self, frame_idx: int):
        """
        Notify that a frame has been selected.
        
        Args:
            frame_idx: Index of selected frame
        """
        self.selection_counter += 1
        self.selection_map[frame_idx] = self.selection_counter
        self.frames_selected += 1
    
    def get_stats(self) -> Dict:
        """Get TMAS statistics."""
        stats = {
            'mode': self.config.mode,
            'decay_mode': self.config.decay_mode,
            'use_curvature': self.config.use_curvature,
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
        
        # Curvature statistics
        if self.config.use_curvature and self.curvature_calculator:
            stats['curvature_method'] = self.config.curvature_method
            stats['curvature_stats'] = self.curvature_calculator.stats
            
            if self.curvature_adjustments:
                reductions = [a['reduction_ratio'] for a in self.curvature_adjustments]
                stats['curvature_adjustments'] = {
                    'num_adjustments': len(self.curvature_adjustments),
                    'avg_reduction_ratio': round(np.mean(reductions), 3),
                    'min_reduction_ratio': round(np.min(reductions), 3),
                }
        
        return stats


class KeyframeSelector:
    """
    Selects keyframes using sorted greedy selection with TMAS and curvature-based self-tuning.
    """
    
    def __init__(self, scores: np.ndarray, frame_ids: np.ndarray,
                 tmas_calculator: TemporalMemorySuppressionCalculator = None,
                 min_score_threshold: float = 0.0):
        """
        Initialize keyframe selector.
        
        Args:
            scores: Relevance scores
            frame_ids: Frame IDs
            tmas_calculator: TMAS calculator
            min_score_threshold: Minimum score threshold
        """
        self.scores = np.asarray(scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.N = len(scores)
        self.min_score_threshold = min_score_threshold
        self.tmas_calculator = tmas_calculator
        
        # Normalize scores to [0, 1]
        if self.N > 0:
            score_min, score_max = self.scores.min(), self.scores.max()
            if score_max > score_min:
                self.scores = (self.scores - score_min) / (score_max - score_min)
            else:
                self.scores = np.ones_like(self.scores) * 0.5
        
        # Filter valid frames
        self.valid_mask = self.scores >= min_score_threshold
        self.num_valid = np.sum(self.valid_mask)
    
    def select_keyframes(self, max_frames: int, optimize_remaining: bool = False) -> List[int]:
        """
        Select keyframes with curvature-based self-tuning.
        
        Args:
            max_frames: Maximum frames to select
            optimize_remaining: Fill remaining slots with best available
            
        Returns:
            List of selected frame IDs (sorted)
        """
        if self.N == 0 or self.num_valid == 0:
            return []
        
        if self.num_valid <= max_frames:
            valid_indices = np.where(self.valid_mask)[0]
            return sorted(list(set([int(self.frame_ids[idx]) for idx in valid_indices])))
        
        # Phase 1: TMAS-guided sorted selection with curvature
        selected_indices = self._sorted_tmas_selection(max_frames)
        
        # Phase 2: Fill remaining if enabled
        if optimize_remaining and len(selected_indices) < max_frames:
            selected_indices = self._fill_remaining_slots(selected_indices, max_frames)
        
        # Convert to frame IDs
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        return sorted(list(set(selected_frame_ids)))
    
    def _sorted_tmas_selection(self, max_frames: int) -> List[int]:
        """
        Perform sorted selection with TMAS suppression and curvature adjustment.
        
        Returns:
            List of selected indices
        """
        # Get valid indices
        valid_indices = np.where(self.valid_mask)[0]
        
        # Create sorted list: (score, index) pairs, highest score first
        sorted_candidates = sorted(
            [(self.scores[idx], idx) for idx in valid_indices],
            key=lambda x: (-x[0], x[1])
        )
        
        selected_indices = []
        selected_set = set()
        
        # Process in sorted order
        for score, idx in sorted_candidates:
            # Check if budget exhausted
            if len(selected_indices) >= max_frames:
                break
            
            # Skip if already selected
            if idx in selected_set:
                continue
            
            # Check if suppressed (curvature automatically applied inside)
            if self.tmas_calculator:
                is_suppressed, _ = self.tmas_calculator.is_suppressed(idx)
                if is_suppressed:
                    continue
            
            # Select this frame
            selected_indices.append(idx)
            selected_set.add(idx)
            
            # Notify TMAS calculator
            if self.tmas_calculator:
                self.tmas_calculator.notify_frame_selected(idx)
        
        return selected_indices
    
    def _fill_remaining_slots(self, selected_indices: List[int], max_frames: int) -> List[int]:
        """Fill remaining slots with best available frames."""
        selected_set = set(selected_indices)
        remaining = max_frames - len(selected_indices)
        
        if remaining <= 0:
            return selected_indices
        
        # Get available frames
        valid_indices = np.where(self.valid_mask)[0]
        available = [idx for idx in valid_indices if idx not in selected_set]
        
        if not available:
            return selected_indices
        
        # Sort by score and take top
        available_scores = [(self.scores[idx], idx) for idx in available]
        available_scores.sort(reverse=True, key=lambda x: x[0])
        
        num_to_add = min(remaining, len(available_scores))
        for i in range(num_to_add):
            selected_indices.append(available_scores[i][1])
        
        return selected_indices


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
        use_curvature=args.use_curvature,
        curvature_method=args.curvature_method,
        curvature_smoothing=args.curvature_smoothing,
        curvature_normalize=args.curvature_normalize,
        curvature_clip_percentile=args.curvature_clip_percentile,
    )


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, tmas_config: TMASConfig, args) -> Tuple[List[int], Dict]:
    """
    Process a single video with curvature-based self-tuning.
    
    Args:
        scores: Frame scores
        frame_ids: Frame IDs
        max_frames: Max frames to select
        tmas_config: TMAS config
        args: Arguments
        
    Returns:
        (selected_frame_ids, stats)
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
    
    # Normalize scores for curvature calculation
    if len(scores) > 0:
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            normalized_scores = (scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.ones_like(scores) * 0.5
    else:
        normalized_scores = scores
    
    # Create curvature calculator if enabled
    curvature_calculator = None
    if tmas_config.use_curvature:
        curvature_calculator = CurvatureCalculator(normalized_scores, tmas_config)
    
    # Create TMAS calculator with curvature
    tmas_calculator = TemporalMemorySuppressionCalculator(
        config=tmas_config,
        total_frames=len(scores),
        target_frames=max_frames,
        curvature_calculator=curvature_calculator
    )
    
    # Select keyframes
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
    
    # Add peak detection info
    if curvature_calculator:
        peak_indices = curvature_calculator.get_peak_indices(threshold_percentile=75.0)
        stats['detected_peaks'] = len(peak_indices)
        stats['selected_peaks'] = sum(1 for idx in peak_indices if frame_ids[idx] in selected_frames)
    
    # Validation
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames!"
    assert len(selected_frames) <= max_frames, f"Too many frames: {len(selected_frames)}"
    
    return selected_frames, stats


def build_output_filename(args):
    """Build output filename with curvature info."""
    name = f"selected_tmas_{args.dataset_name}_{args.extract_feature_model}_k{args.max_num_frames}"
    name += f"_{args.tmas_mode}_{args.tmas_decay_mode}"
    
    if args.use_curvature:
        name += f"_curv{args.curvature_method[:4]}"  # e.g., curvseco, curvlapl
        name += f"_norm{args.curvature_normalize[:3]}"  # e.g., normmax, normstd
        
        if args.curvature_smoothing > 0:
            name += f"_sm{args.curvature_smoothing}"
        
        if args.curvature_clip_percentile < 100.0:
            name += f"_clip{int(args.curvature_clip_percentile)}"
    
    if args.tmas_mode == 'auto':
        name += f"_{args.tmas_auto_scaling}_cov{args.tmas_auto_coverage}"
    elif args.tmas_mode == 'additive':
        name += f"_{args.tmas_delta_strategy}"
    
    if args.optimize_remaining:
        name += "_opt"
    
    name += ".json"
    return name


def main(args):
    """Main processing function."""
    print("=" * 80)
    if args.use_curvature:
        print("🚀 TMAS Frame Selection with Curvature-Based Self-Tuning")
        print("   Theory: R_i = R_TMAS / (1 + κ_i)")
        print("   → Peaks (high κ) get small radius → Protected from suppression")
        print("   → Flat regions (low κ) get normal radius → Standard TMAS behavior")
    else:
        print("🚀 TMAS Frame Selection (Standard)")
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
    if args.use_curvature:
        print(f"Curvature method: {args.curvature_method}")
        print(f"Normalization: {args.curvature_normalize}")
    print()
    
    # Process videos
    selected_frames_all = []
    all_stats = []
    
    for idx in range(num_videos):
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_videos}...")
        
        try:
            # Get base radius
            if args.tmas_mode == 'additive':
                base_radius = get_base_suppression_radius(metadata[idx], args, args.dataset_name)
            else:
                base_radius = None
            
            # Create config
            tmas_config = create_tmas_config(args, base_radius)
            
            # Process
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
    
    # Print statistics
    frame_counts = [len(f) for f in selected_frames_all if len(f) > 0]
    if frame_counts:
        print(f"\n📊 Statistics:")
        print(f"  Avg frames: {np.mean(frame_counts):.2f}")
        print(f"  Std frames: {np.std(frame_counts):.2f}")
        print(f"  Min/Max: {np.min(frame_counts)}/{np.max(frame_counts)}")
    
    # Curvature statistics
    if args.use_curvature:
        valid_stats = [s for s in all_stats if 'curvature_stats' in s]
        if valid_stats:
            print(f"\n📈 Curvature Statistics (across {len(valid_stats)} videos):")
            avg_peak_detection = np.mean([s.get('detected_peaks', 0) for s in valid_stats])
            avg_peak_selected = np.mean([s.get('selected_peaks', 0) for s in valid_stats])
            print(f"  Avg detected peaks: {avg_peak_detection:.2f}")
            print(f"  Avg selected peaks: {avg_peak_selected:.2f}")
            
            if any('curvature_adjustments' in s for s in valid_stats):
                adjustments = [s['curvature_adjustments']['avg_reduction_ratio'] 
                              for s in valid_stats if 'curvature_adjustments' in s]
                print(f"  Avg radius reduction: {np.mean(adjustments):.3f}x")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)