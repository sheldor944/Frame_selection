import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP with TMAS: Temporal Memory-Aware Suppression')
    
    parser.add_argument('--dataset_name', type=str, default='videomme', 
                        help='Dataset name: longvideobench or videomme')
    parser.add_argument('--extract_feature_model', type=str, default='clip', 
                        help='Feature extraction model: blip/clip/sevila')
    parser.add_argument('--score_path', type=str, 
                        default='./outscores/videomme/clip/scores.json',
                        help='Path to input scores JSON file')
    parser.add_argument('--frame_path', type=str, 
                        default='./outscores/videomme/clip/frames.json',
                        help='Path to input frame IDs JSON file')
    parser.add_argument('--metadata_path', type=str,
                        default='./datasets/videomme/metadata.json',
                        help='Path to metadata JSON file')
    parser.add_argument('--max_num_frames', type=int, default=32,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    parser.add_argument('--alpha', type=float, default=.85,
                        help='Diffusion decay factor (0-1)')
    parser.add_argument('--diffusion_iterations', type=int, default=0,
                        help='Number of diffusion iterations')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS MODE SELECTION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_mode', type=str, default='auto',
                        choices=['auto', 'additive'],
                        help='''TMAS suppression mode:
                        ★ auto: R0 automatically derived from video length & target frames (NO manual tuning)
                        ★ additive: Base radius (per-group) + TMAS temporal decay on top''')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS AUTO MODE - R0 CALCULATION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_auto_scaling', type=str, default='hybrid',
                        choices=['linear', 'sqrt', 'hybrid'],
                        help='''[AUTO MODE] Scaling function for R0 calculation:
                        
                        ★ linear: R0 = (L/K) × coverage
                          → Direct proportional, grows linearly with video length
                          → Best for: uniform-length datasets
                          → Example: 1000 frames, K=32 → R0 = 31.25 × coverage
                        
                        ★ sqrt: R0 = √(L/K) × coverage
                          → Sublinear growth, prevents huge R0 in long videos
                          → Best for: mixed-length datasets with very long videos
                          → Example: 10000 frames, K=32 → R0 = 17.68 × coverage
                        
                        ★ hybrid: R0 = (L/K)^0.7 × coverage [RECOMMENDED]
                          → Balanced between linear and sqrt
                          → Best for: most use cases
                          → Example: 1000 frames, K=32 → R0 = 15.85 × coverage
                        
                        Formula: R0 = (L / target_K) ^ exponent × coverage_factor
                        Where: L = video length, K = target frames''')
    
    parser.add_argument('--tmas_auto_coverage', type=float, default=1,
                        help='''[AUTO MODE] Coverage factor (0.3 - 1.5):
                        
                        Controls how much of the "ideal spacing" R0 should cover:
                        
                        ★ 0.5: Sparse coverage (gaps between selections, allows revisits)
                        ★ 0.8: Balanced coverage [RECOMMENDED]
                        ★ 1.0: Full coverage (each selection covers its "fair share")
                        ★ 1.5: Dense coverage (overlapping suppression regions)
                        
                        Higher values = stronger initial suppression
                        Lower values = more flexible, allows closer selections initially''')
    
    parser.add_argument('--tmas_auto_min_radius', type=float, default=1.0,
                        help='[AUTO MODE] Minimum R0 value (safety floor)')
    
    parser.add_argument('--tmas_auto_max_radius', type=float, default=None,
                        help='[AUTO MODE] Maximum R0 value (None = no cap)')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS ADDITIVE MODE - BASE + DELTA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_delta_strategy', type=str, default='proportional',
                        choices=['fixed', 'proportional', 'adaptive'],
                        help='''[ADDITIVE MODE] How to calculate Delta_R (additional TMAS radius):
                        
                        ★ fixed: Delta_R = tmas_delta_R (constant value)
                          → Simple, predictable
                          → Use when: you know exact suppression needs
                          → Example: Delta_R = 3.0 always adds 3 frames
                        
                        ★ proportional: Delta_R = Base_R0 × tmas_multiplier [RECOMMENDED]
                          → Scales with video category
                          → Use when: you want consistent relative boost
                          → Example: short(base=2) → Delta=1, long(base=5) → Delta=2.5
                        
                        ★ adaptive: Delta_R = Base_R0 × (remaining_budget / target_K)
                          → Decreases as more frames selected
                          → Use when: you want aggressive early, relaxed late
                          → Example: First selection → full Delta, Last selection → minimal Delta''')
    
    parser.add_argument('--tmas_delta_R', type=float, default=3.0,
                        help='[ADDITIVE MODE - fixed strategy] Fixed Delta_R value')
    
    parser.add_argument('--tmas_multiplier', type=float, default=0.7,
                        help='''[ADDITIVE MODE - proportional/adaptive] Multiplier for Delta_R (0.0 - 2.0):
                        
                        Delta_R = Base_R0 × multiplier
                        
                        ★ 0.3: Gentle TMAS (+30%% extra suppression at t=0)
                        ★ 0.5: Moderate TMAS (+50%%, RECOMMENDED)
                        ★ 0.7: Aggressive TMAS (+70%%)
                        ★ 1.0: Double base radius at t=0
                        
                        Higher values = stronger temporal memory effect''')
    
    # Base radii for additive mode (per-group configuration)
    parser.add_argument('--suppression_radius_15', type=float, default=2.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=15 (LongVideoBench)')
    parser.add_argument('--suppression_radius_60', type=float, default=3.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=60')
    parser.add_argument('--suppression_radius_600', type=float, default=5.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=600')
    parser.add_argument('--suppression_radius_3600', type=float, default=8.0,
                        help='[ADDITIVE MODE] Base radius for duration_group=3600')
    parser.add_argument('--suppression_radius_short', type=float, default=2.0,
                        help='[ADDITIVE MODE] Base radius for short videos (VideoMME)')
    parser.add_argument('--suppression_radius_medium', type=float, default=3.0,
                        help='[ADDITIVE MODE] Base radius for medium videos')
    parser.add_argument('--suppression_radius_long', type=float, default=5.0,
                        help='[ADDITIVE MODE] Base radius for long videos')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS DECAY CONFIGURATION (COMMON TO BOTH MODES)
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_decay_mode', type=str, default='half_life',
                        choices=['video_length', 'half_life', 'quartile', 'custom'],
                        help='''Decay rate (λ) calculation method:
                        
                        Controls how fast suppression weakens over time: R(t) = R0 × exp(-λ × Δt)
                        
                        ★ video_length: λ = log(L) / L
                          → Fast decay, good for short videos
                          → Suppression drops quickly
                        
                        ★ half_life: λ = log(2) / (L / 2K) [RECOMMENDED]
                          → Suppression drops to 50%% at midpoint between ideal selections
                          → Intuitive, balances short & long-range memory
                          → Example: L=1000, K=32 → half_life at ~15.6 frames
                        
                        ★ quartile: λ = log(4) / (L / 4)
                          → Suppression drops to 25%% at quarter video length
                          → Slower decay, longer memory
                          → Good for very long videos
                        
                        ★ custom: λ = tmas_custom_lambda (manual specification)
                          → Full control over decay rate
                          → Use when you know exact temporal memory needs''')
    
    parser.add_argument('--tmas_custom_lambda', type=float, default=0.01,
                        help='[DECAY - custom mode] Manual decay rate λ value')
    
    parser.add_argument('--tmas_decay_floor', type=float, default=0.0,
                        help='''Minimum TMAS contribution (0.0 - 1.0):
                        
                        Prevents TMAS from decaying to zero:
                        TMAS_contrib = max(Delta_R × exp(-λ×Δt), Delta_R × floor)
                        
                        ★ 0.0: Full decay to zero (pure exponential)
                        ★ 0.1: Always retain 10%% of initial TMAS
                        ★ 0.2: Always retain 20%% (longer memory tail)
                        
                        Higher values = suppression never fully disappears''')
    
    # ═══════════════════════════════════════════════════════════════════════
    # TMAS ADVANCED OPTIONS
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--tmas_score_aware_decay', action='store_true', default=False,
                        help='''Enable score-aware decay modification:
                        
                        Makes λ depend on score similarity:
                        λ_effective = λ_base × (1 + β × |score_i - score_j|)
                        
                        Effect: Similar-scored frames decay slower (allow redundancy if both important)
                        Use: --tmas_score_aware_beta to control strength''')
    
    parser.add_argument('--tmas_score_aware_beta', type=float, default=0.3,
                        help='[ADVANCED] Score-aware decay strength (0.0 - 1.0)')
    
    parser.add_argument('--tmas_bidirectional', action='store_true', default=True,
                        help='''Enable bidirectional suppression:
                        
                        Selected frames suppress both forward AND backward in time
                        (default: only suppress forward candidates)
                        
                        Use when: temporal order doesn't matter, want global optimization''')
    
    # ═══════════════════════════════════════════════════════════════════════
    # OTHER PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════
    parser.add_argument('--edge_weight_type', type=str, default='temporal',
                        choices=['uniform', 'score_diff', 'temporal'],
                        help='Edge weight type for diffusion')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (None = all)')
    parser.add_argument('--min_score_threshold', type=float, default=0,
                        help='Minimum normalized score threshold (0-1)')
    parser.add_argument('--optimize_remaining', action='store_true',default=True,
                        help='Fill remaining slots with best frames (no suppression)')
    
    return parser.parse_args()


@dataclass
@dataclass
class TMASConfig:
    """Configuration for Temporal Memory-Aware Suppression"""
    # ═══════════════════════════════════════════════════════════════
    # REQUIRED FIELDS (NO DEFAULTS) - MUST COME FIRST
    # ═══════════════════════════════════════════════════════════════
    mode: str  # 'auto' or 'additive'
    
    # Auto mode parameters
    auto_scaling: str
    auto_coverage: float
    auto_min_radius: float
    
    # Additive mode parameters
    delta_strategy: str
    delta_R: float
    multiplier: float
    
    # Decay parameters
    decay_mode: str
    custom_lambda: float
    decay_floor: float
    
    # Advanced options
    score_aware_decay: bool
    score_aware_beta: float
    bidirectional: bool
    
    # ═══════════════════════════════════════════════════════════════
    # OPTIONAL FIELDS (WITH DEFAULTS) - MUST COME LAST
    # ═══════════════════════════════════════════════════════════════
    auto_max_radius: Optional[float] = None
    base_radius: Optional[float] = None


class TemporalMemorySuppressionCalculator:
    """
    Temporal Memory-Aware Suppression (TMAS) Calculator.
    
    Models suppression as a decaying temporal memory:
    - Recently selected frames strongly suppress neighbors
    - Suppression gradually weakens over time (exponential decay)
    - Allows revisiting previously suppressed regions if they remain informative
    """
    
    def __init__(self, config: TMASConfig, total_frames: int, target_frames: int, 
                 diffused_scores: Optional[np.ndarray] = None):
        """
        Initialize TMAS calculator.
        
        Args:
            config: TMAS configuration
            total_frames: Total number of frames in video
            target_frames: Target number of frames to select
            diffused_scores: Optional diffused scores for score-aware decay
        """
        self.config = config
        self.L = total_frames
        self.target_k = target_frames
        self.diffused_scores = diffused_scores
        self.frames_selected = 0  # Track for adaptive strategy
        
        # Calculate initial radius based on mode
        if config.mode == 'auto':
            self.R0 = self._calculate_auto_R0()
            self.Delta_R = 0.0  # No separate Delta in auto mode
        else:  # 'additive'
            self.R0 = config.base_radius if config.base_radius else 3.0
            self.Delta_R = self._calculate_delta_R()
        
        # Calculate decay rate
        self.lambda_decay = self._calculate_decay_rate()
        
        # Statistics tracking
        self.suppression_history = []
    
    def _calculate_auto_R0(self) -> float:
        """Calculate R0 for auto mode based on video properties."""
        if self.L == 0 or self.target_k == 0:
            return self.config.auto_min_radius
        
        base_spacing = self.L / self.target_k
        
        if self.config.auto_scaling == 'linear':
            R0 = base_spacing * self.config.auto_coverage
        elif self.config.auto_scaling == 'sqrt':
            R0 = np.sqrt(base_spacing) * self.config.auto_coverage
        else:  # 'hybrid'
            R0 = (base_spacing ** 0.7) * self.config.auto_coverage
        
        # Apply bounds
        R0 = max(self.config.auto_min_radius, R0)
        if self.config.auto_max_radius is not None:
            R0 = min(self.config.auto_max_radius, R0)
        
        return R0
    
    def _calculate_delta_R(self) -> float:
        """Calculate Delta_R for additive mode based on strategy."""
        if self.config.delta_strategy == 'fixed':
            return self.config.delta_R
        elif self.config.delta_strategy == 'proportional':
            return self.R0 * self.config.multiplier
        else:  # 'adaptive'
            # Will be recalculated dynamically in get_delta_R_dynamic()
            return self.R0 * self.config.multiplier
    
    def _get_delta_R_dynamic(self) -> float:
        """Calculate Delta_R dynamically for adaptive strategy."""
        if self.config.delta_strategy != 'adaptive':
            return self.Delta_R
        
        # Adaptive: decrease Delta_R as more frames are selected
        remaining_budget = max(1, self.target_k - self.frames_selected)
        budget_ratio = remaining_budget / self.target_k
        
        return self.R0 * self.config.multiplier * budget_ratio
    
    def _calculate_decay_rate(self) -> float:
        """Calculate decay rate λ based on decay mode."""
        if self.L == 0:
            return 0.01
        
        if self.config.decay_mode == 'video_length':
            # λ = log(L) / L
            return np.log(max(2, self.L)) / self.L
        
        elif self.config.decay_mode == 'half_life':
            # Decay to 50% at midpoint between ideal selections
            if self.target_k == 0:
                return 0.01
            half_life = self.L / (2 * self.target_k)
            return np.log(2) / max(1, half_life)
        
        elif self.config.decay_mode == 'quartile':
            # Decay to 25% at quarter of video length
            quartile_distance = self.L / 4
            return np.log(4) / max(1, quartile_distance)
        
        else:  # 'custom'
            return self.config.custom_lambda
    
    def _get_score_aware_lambda(self, selected_idx: int, candidate_idx: int) -> float:
        """Calculate score-aware decay rate."""
        if not self.config.score_aware_decay or self.diffused_scores is None:
            return self.lambda_decay
        
        # Modify λ based on score similarity
        score_diff = abs(self.diffused_scores[selected_idx] - self.diffused_scores[candidate_idx])
        lambda_effective = self.lambda_decay * (1 + self.config.score_aware_beta * score_diff)
        
        return lambda_effective
    
    def get_effective_radius(self, selected_indices: List[int], candidate_idx: int) -> float:
        """
        Calculate effective suppression radius at candidate position.
        
        This is the core TMAS algorithm: R_effective = max over all selected frames
        
        Args:
            selected_indices: List of already selected frame indices
            candidate_idx: Index of candidate frame being considered
        
        Returns:
            Effective suppression radius at this position
        """
        if not selected_indices:
            # No frames selected yet, return maximum radius
            if self.config.mode == 'auto':
                return self.R0
            else:  # additive
                current_delta = self._get_delta_R_dynamic()
                return self.R0 + current_delta
        
        # Calculate maximum suppression from all selected frames
        max_suppression = 0.0
        
        for selected_idx in selected_indices:
            delta_t = abs(candidate_idx - selected_idx)
            
            # Get decay rate (possibly score-aware)
            lambda_effective = self._get_score_aware_lambda(selected_idx, candidate_idx)
            
            if self.config.mode == 'auto':
                # Auto mode: R = R0 * exp(-λ * Δt)
                suppression = self.R0 * np.exp(-lambda_effective * delta_t)
            
            else:  # 'additive'
                # Additive mode: R = R0 + Delta_R * exp(-λ * Δt)
                current_delta = self._get_delta_R_dynamic()
                tmas_contribution = current_delta * np.exp(-lambda_effective * delta_t)
                
                # Apply decay floor
                if self.config.decay_floor > 0:
                    tmas_contribution = max(tmas_contribution, current_delta * self.config.decay_floor)
                
                suppression = self.R0 + tmas_contribution
            
            max_suppression = max(max_suppression, suppression)
        
        return max_suppression
    
    def is_suppressed(self, selected_indices: List[int], candidate_idx: int) -> bool:
        """
        Check if candidate frame is suppressed by any selected frame.
        
        Args:
            selected_indices: List of already selected frame indices
            candidate_idx: Index of candidate frame being considered
        
        Returns:
            True if candidate should be suppressed, False otherwise
        """
        if not selected_indices:
            return False
        
        effective_radius = self.get_effective_radius(selected_indices, candidate_idx)
        
        # Check if any selected frame suppresses this candidate
        for selected_idx in selected_indices:
            delta_t = abs(candidate_idx - selected_idx)
            if delta_t <= effective_radius and delta_t > 0:
                # Track suppression event
                self.suppression_history.append({
                    'candidate_idx': candidate_idx,
                    'suppressed_by': selected_idx,
                    'delta_t': delta_t,
                    'effective_radius': effective_radius
                })
                return True
        
        return False
    
    def notify_frame_selected(self):
        """Notify calculator that a frame has been selected (for adaptive strategy)."""
        self.frames_selected += 1
    
    def get_stats(self) -> Dict:
        """Get comprehensive TMAS statistics."""
        stats = {
            'mode': self.config.mode,
            'R0': round(self.R0, 3),
            'lambda': round(self.lambda_decay, 6),
            'total_frames': self.L,
            'target_frames': self.target_k,
            'frames_selected': self.frames_selected,
        }
        
        if self.config.mode == 'auto':
            stats['auto_scaling'] = self.config.auto_scaling
            stats['auto_coverage'] = self.config.auto_coverage
            stats['max_radius_at_t0'] = round(self.R0, 3)
        
        elif self.config.mode == 'additive':
            current_delta = self._get_delta_R_dynamic()
            stats['base_radius'] = round(self.R0, 3)
            stats['delta_strategy'] = self.config.delta_strategy
            stats['delta_R'] = round(current_delta, 3)
            stats['multiplier'] = self.config.multiplier
            stats['max_radius_at_t0'] = round(self.R0 + current_delta, 3)
        
        # Decay profile at sample distances
        sample_distances = [0, 10, 25, 50, 100, 200]
        stats['decay_profile'] = {}
        
        for dist in sample_distances:
            if dist > self.L:
                continue
            
            if self.config.mode == 'auto':
                r = self.R0 * np.exp(-self.lambda_decay * dist)
            else:  # additive
                current_delta = self._get_delta_R_dynamic()
                tmas_contrib = current_delta * np.exp(-self.lambda_decay * dist)
                if self.config.decay_floor > 0:
                    tmas_contrib = max(tmas_contrib, current_delta * self.config.decay_floor)
                r = self.R0 + tmas_contrib
            
            stats['decay_profile'][f'dt_{dist}'] = round(r, 3)
        
        # Suppression statistics
        if self.suppression_history:
            delta_ts = [s['delta_t'] for s in self.suppression_history]
            stats['suppression_stats'] = {
                'total_suppressions': len(self.suppression_history),
                'avg_delta_t': round(np.mean(delta_ts), 2),
                'max_delta_t': int(np.max(delta_ts)),
                'min_delta_t': int(np.min(delta_ts))
            }
        
        return stats


def get_base_suppression_radius(metadata_entry: dict, args, dataset_name: str) -> float:
    """
    Get base suppression radius based on video duration category.
    Used as R0 for additive mode.
    
    Args:
        metadata_entry: Dictionary containing video metadata
        args: Argument namespace with suppression radius parameters
        dataset_name: Name of the dataset
    
    Returns:
        Base suppression radius for this video
    """
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
            print(f"  ⚠️  Warning: Unknown duration_group '{duration_group}', using default 3.0")
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
            print(f"  ⚠️  Warning: Unknown duration '{duration}', using default 3.0")
            return 3.0
    else:
        print(f"  ⚠️  Warning: Unknown dataset '{dataset_name}', using default 3.0")
        return 3.0


class DiffusionGraph:
    """
    Represents a temporal graph of video frames with diffusion capabilities.
    Optimized version with vectorized operations.
    """
    
    def __init__(self, scores: np.ndarray, frame_ids: np.ndarray, 
                 alpha: float = 0.7, edge_weight_type: str = 'uniform'):
        """
        Initialize the diffusion graph.
        
        Args:
            scores: Array of frame relevance scores
            frame_ids: Array of frame IDs corresponding to scores
            alpha: Diffusion decay factor
            edge_weight_type: Type of edge weighting
        """
        self.scores = np.asarray(scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.alpha = alpha
        self.edge_weight_type = edge_weight_type
        self.N = len(scores)
        
        # Normalize scores to [0, 1]
        if self.N > 0:
            score_min, score_max = self.scores.min(), self.scores.max()
            if score_max > score_min:
                self.scores = (self.scores - score_min) / (score_max - score_min)
            else:
                self.scores = np.ones_like(self.scores) * 0.5
        
        # Initialize diffused scores
        self.diffused_scores = self.scores.copy()
        
        # Build edge weights
        self.edge_weights = self._build_edge_weights()
    
    def _build_edge_weights(self) -> np.ndarray:
        """Build edge weights between adjacent frames."""
        if self.N <= 1:
            return np.array([], dtype=np.float64)
        
        if self.edge_weight_type == 'uniform':
            return np.ones(self.N - 1, dtype=np.float64)
        
        elif self.edge_weight_type == 'score_diff':
            score_diffs = np.abs(np.diff(self.scores))
            weights = 1.0 / (score_diffs + 1e-6)
            weights = weights / weights.max()
            return weights.astype(np.float64)
        
        elif self.edge_weight_type == 'temporal':
            temporal_gaps = np.diff(self.frame_ids)
            weights = 1.0 / (temporal_gaps + 1.0)
            weights = weights / weights.max()
            return weights.astype(np.float64)
        
        return np.ones(self.N - 1, dtype=np.float64)
    
    def diffuse(self, iterations: int = None) -> np.ndarray:
        """Perform diffusion process on the graph using vectorized operations."""
        if self.N <= 1:
            return self.diffused_scores
        
        if iterations is None:
            iterations = max(1, int(np.log2(self.N)))
        
        # Vectorized diffusion
        for _ in range(iterations):
            left_neighbors = np.zeros(self.N, dtype=np.float64)
            right_neighbors = np.zeros(self.N, dtype=np.float64)
            left_weights = np.zeros(self.N, dtype=np.float64)
            right_weights = np.zeros(self.N, dtype=np.float64)
            
            left_neighbors[1:] = self.diffused_scores[:-1]
            left_weights[1:] = self.edge_weights
            
            right_neighbors[:-1] = self.diffused_scores[1:]
            right_weights[:-1] = self.edge_weights
            
            total_weights = left_weights + right_weights
            neighbor_contrib = np.zeros(self.N, dtype=np.float64)
            
            mask = total_weights > 0
            neighbor_contrib[mask] = (
                (left_neighbors[mask] * left_weights[mask] + 
                 right_neighbors[mask] * right_weights[mask]) / 
                total_weights[mask]
            )
            
            self.diffused_scores = (
                self.alpha * self.diffused_scores + 
                (1 - self.alpha) * neighbor_contrib
            )
        
        return self.diffused_scores


class KeyframeSelector:
    """
    Selects keyframes using TMAS-enhanced greedy selection.
    """
    
    def __init__(self, diffused_scores: np.ndarray, frame_ids: np.ndarray,
                 original_scores: np.ndarray = None,
                 tmas_calculator: TemporalMemorySuppressionCalculator = None,
                 min_score_threshold: float = 0.0):
        """
        Initialize keyframe selector with TMAS.
        
        Args:
            diffused_scores: Diffused relevance scores
            frame_ids: Frame IDs corresponding to scores
            original_scores: Original (normalized) scores before diffusion
            tmas_calculator: TMAS calculator instance
            min_score_threshold: Minimum score threshold (0-1)
        """
        self.diffused_scores = np.asarray(diffused_scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.original_scores = original_scores if original_scores is not None else diffused_scores
        self.N = len(diffused_scores)
        self.min_score_threshold = min_score_threshold
        self.tmas_calculator = tmas_calculator
        
        # Filter out frames below threshold
        self.valid_mask = self.original_scores >= min_score_threshold
        self.num_valid = np.sum(self.valid_mask)
    
    def select_keyframes(self, max_frames: int, optimize_remaining: bool = False) -> List[int]:
        """
        Select keyframes using TMAS-enhanced suppression.
        
        Args:
            max_frames: Maximum number of frames to select
            optimize_remaining: If True, fill remaining slots with best frames
        
        Returns:
            List of unique selected frame IDs (sorted)
        """
        if self.N == 0 or self.num_valid == 0:
            return []
        
        # If we have fewer valid frames than requested, return all valid frames
        if self.num_valid <= max_frames:
            valid_indices = np.where(self.valid_mask)[0]
            selected_frame_ids = [int(self.frame_ids[idx]) for idx in valid_indices]
            return sorted(list(set(selected_frame_ids)))
        
        # Phase 1: TMAS-guided greedy selection
        selected_indices = self._tmas_greedy_selection(max_frames)
        
        # Phase 2: Fill remaining slots if enabled
        if optimize_remaining and len(selected_indices) < max_frames:
            selected_indices = self._fill_remaining_slots(selected_indices, max_frames)
        
        # Convert to frame IDs, ensure uniqueness, and sort
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        unique_frame_ids = sorted(list(set(selected_frame_ids)))
        
        # Sanity check
        if len(unique_frame_ids) > max_frames:
            unique_frame_ids = unique_frame_ids[:max_frames]
        
        return unique_frame_ids
    
    def _tmas_greedy_selection(self, max_frames: int) -> List[int]:
        """
        Perform TMAS-guided greedy selection.
        
        Uses temporal memory-aware suppression instead of fixed radius.
        
        Args:
            max_frames: Maximum number of frames to select
        
        Returns:
            List of selected indices
        """
        # Get valid frame indices (above threshold)
        valid_indices = np.where(self.valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Create priority queue (max heap using negative scores)
        candidates = [
            (-self.diffused_scores[idx], idx) 
            for idx in valid_indices
        ]
        heapq.heapify(candidates)
        
        selected_indices = []
        selected_indices_set = set()
        
        while len(selected_indices) < max_frames and candidates:
            # Get highest scoring candidate
            neg_score, idx = heapq.heappop(candidates)
            
            # Skip if already selected
            if idx in selected_indices_set:
                continue
            
            # Check if suppressed by TMAS
            if self.tmas_calculator and self.tmas_calculator.is_suppressed(selected_indices, idx):
                continue
            
            # Select this frame
            selected_indices.append(idx)
            selected_indices_set.add(idx)
            
            # Notify TMAS calculator
            if self.tmas_calculator:
                self.tmas_calculator.notify_frame_selected()
        
        return selected_indices
    
    def _fill_remaining_slots(self, selected_indices: List[int], 
                             max_frames: int) -> List[int]:
        """
        Fill remaining slots with best available frames (no suppression).
        
        Args:
            selected_indices: Already selected frame indices
            max_frames: Maximum total frames to select
        
        Returns:
            Updated list of selected indices
        """
        selected_set = set(selected_indices)
        remaining_slots = max_frames - len(selected_indices)
        
        if remaining_slots <= 0:
            return selected_indices
        
        # Get all valid frames not yet selected
        valid_indices = np.where(self.valid_mask)[0]
        available_indices = [idx for idx in valid_indices if idx not in selected_set]
        
        if not available_indices:
            return selected_indices
        
        # Get scores for available frames
        available_scores = self.diffused_scores[available_indices]
        
        # Sort by score and take top remaining_slots
        num_to_take = min(remaining_slots, len(available_indices))
        top_k_local_indices = np.argsort(available_scores)[-num_to_take:][::-1]
        top_k_indices = [available_indices[i] for i in top_k_local_indices]
        
        # Add to selected (guaranteed unique)
        selected_indices.extend(top_k_indices)
        
        return selected_indices


def create_tmas_config(args, base_radius: Optional[float] = None) -> TMASConfig:
    """
    Create TMAS configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        base_radius: Base suppression radius (for additive mode)
    
    Returns:
        TMASConfig instance
    """
    return TMASConfig(
        mode=args.tmas_mode,
        # Auto mode
        auto_scaling=args.tmas_auto_scaling,
        auto_coverage=args.tmas_auto_coverage,
        auto_min_radius=args.tmas_auto_min_radius,
        auto_max_radius=args.tmas_auto_max_radius,
        # Additive mode
        delta_strategy=args.tmas_delta_strategy,
        delta_R=args.tmas_delta_R,
        multiplier=args.tmas_multiplier,
        base_radius=base_radius,
        # Decay
        decay_mode=args.tmas_decay_mode,
        custom_lambda=args.tmas_custom_lambda,
        decay_floor=args.tmas_decay_floor,
        # Advanced
        score_aware_decay=args.tmas_score_aware_decay,
        score_aware_beta=args.tmas_score_aware_beta,
        bidirectional=args.tmas_bidirectional,
    )


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, tmas_config: TMASConfig, args) -> Tuple[List[int], Dict]:
    """
    Process a single video using DBFP with TMAS.
    
    Args:
        scores: List of frame scores
        frame_ids: List of frame IDs
        max_frames: Maximum frames to select
        tmas_config: TMAS configuration
        args: Argument namespace with DBFP parameters
    
    Returns:
        Tuple of (selected_frame_ids, tmas_stats)
    """
    # Convert to numpy arrays
    scores = np.asarray(scores, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    
    # Apply ratio-based downsampling if needed
    if args.ratio > 1:
        indices = np.arange(0, len(scores), args.ratio)
        scores = scores[indices]
        frame_ids = frame_ids[indices]
    
    # Handle short videos
    if len(scores) <= max_frames:
        unique_frames = sorted(list(set([int(x) for x in frame_ids])))
        return unique_frames[:max_frames], {'short_video': True}
    
    # Create diffusion graph
    graph = DiffusionGraph(
        scores=scores,
        frame_ids=frame_ids,
        alpha=args.alpha,
        edge_weight_type=args.edge_weight_type
    )
    
    # Store original normalized scores
    original_normalized_scores = graph.scores.copy()
    
    # Perform diffusion
    diffusion_iters = args.diffusion_iterations
    if diffusion_iters is None:
        diffusion_iters = max(1, int(np.log2(len(scores))))
    
    diffused_scores = graph.diffuse(iterations=diffusion_iters)
    
    # Create TMAS calculator
    tmas_calculator = TemporalMemorySuppressionCalculator(
        config=tmas_config,
        total_frames=len(scores),
        target_frames=max_frames,
        diffused_scores=diffused_scores if tmas_config.score_aware_decay else None
    )
    
    # Select keyframes with TMAS
    selector = KeyframeSelector(
        diffused_scores=diffused_scores,
        frame_ids=frame_ids,
        original_scores=original_normalized_scores,
        tmas_calculator=tmas_calculator,
        min_score_threshold=args.min_score_threshold
    )
    
    selected_frames = selector.select_keyframes(
        max_frames=max_frames,
        optimize_remaining=args.optimize_remaining
    )
    
    # Get TMAS statistics
    tmas_stats = tmas_calculator.get_stats()
    
    # Final sanity check
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames detected!"
    assert len(selected_frames) <= max_frames, f"Too many frames: {len(selected_frames)} > {max_frames}"
    
    return selected_frames, tmas_stats


def build_output_filename(args):
    """Build output filename based on TMAS parameters."""
    
    # Base name
    name = f"selected_dbfp_tmas_{args.dataset_name}_{args.extract_feature_model}_k{args.max_num_frames}"
    
    # TMAS mode
    name += f"_{args.tmas_mode}"
    
    if args.tmas_mode == 'auto':
        # Auto mode parameters
        name += f"_{args.tmas_auto_scaling}"
        name += f"_cov{args.tmas_auto_coverage}"
    
    elif args.tmas_mode == 'additive':
        # Additive mode parameters
        name += f"_{args.tmas_delta_strategy}"
        if args.tmas_delta_strategy == 'fixed':
            name += f"_d{args.tmas_delta_R}"
        else:
            name += f"_m{args.tmas_multiplier}"
    
    # Decay configuration
    name += f"_{args.tmas_decay_mode}"
    
    if args.tmas_decay_floor > 0:
        name += f"_floor{args.tmas_decay_floor}"
    
    # Advanced options
    if args.tmas_score_aware_decay:
        name += f"_scoreaware{args.tmas_score_aware_beta}"
    
    if args.tmas_bidirectional:
        name += "_bidir"
    
    # Other parameters
    name += f"_alpha{args.alpha}_{args.edge_weight_type}"
    
    if args.diffusion_iterations is not None:
        name += f"_iter{args.diffusion_iterations}"
    
    if args.min_score_threshold > 0:
        name += f"_minscore{args.min_score_threshold}"
    
    if args.optimize_remaining:
        name += "_opt"
    
    name += ".json"
    return name


def print_tmas_configuration(args):
    """Print comprehensive TMAS configuration."""
    print("=" * 80)
    print("🚀 DBFP with TMAS: Temporal Memory-Aware Suppression")
    print("=" * 80)
    print(f"\n📊 Dataset Configuration:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Feature Model: {args.extract_feature_model}")
    print(f"  Max Frames: {args.max_num_frames}")
    print(f"  Min Score Threshold: {args.min_score_threshold}")
    
    print(f"\n🌊 Diffusion Configuration:")
    print(f"  Alpha (decay): {args.alpha}")
    print(f"  Edge Weight Type: {args.edge_weight_type}")
    print(f"  Iterations: {args.diffusion_iterations if args.diffusion_iterations else 'auto (log2(N))'}")
    
    print(f"\n⚡ TMAS Configuration:")
    print(f"  Mode: {args.tmas_mode.upper()}")
    
    if args.tmas_mode == 'auto':
        print(f"\n  📐 Auto Mode - R0 Calculation:")
        print(f"    Scaling: {args.tmas_auto_scaling}")
        print(f"    Coverage Factor: {args.tmas_auto_coverage}")
        print(f"    Min Radius: {args.tmas_auto_min_radius}")
        print(f"    Max Radius: {args.tmas_auto_max_radius if args.tmas_auto_max_radius else 'None (no cap)'}")
        print(f"\n    Formula: R0 = (L/K)^{0.5 if args.tmas_auto_scaling=='sqrt' else 0.7 if args.tmas_auto_scaling=='hybrid' else 1.0} × {args.tmas_auto_coverage}")
    
    elif args.tmas_mode == 'additive':
        print(f"\n  ➕ Additive Mode - Base + TMAS:")
        print(f"    Delta Strategy: {args.tmas_delta_strategy}")
        if args.tmas_delta_strategy == 'fixed':
            print(f"    Fixed Delta_R: {args.tmas_delta_R}")
        else:
            print(f"    Multiplier: {args.tmas_multiplier}")
        
        print(f"\n  📏 Base Radii (per video category):")
        if args.dataset_name == 'longvideobench':
            print(f"    Duration 15s:   {args.suppression_radius_15}")
            print(f"    Duration 60s:   {args.suppression_radius_60}")
            print(f"    Duration 600s:  {args.suppression_radius_600}")
            print(f"    Duration 3600s: {args.suppression_radius_3600}")
        elif args.dataset_name == 'videomme':
            print(f"    Short videos:  {args.suppression_radius_short}")
            print(f"    Medium videos: {args.suppression_radius_medium}")
            print(f"    Long videos:   {args.suppression_radius_long}")
    
    print(f"\n  📉 Decay Configuration:")
    print(f"    Decay Mode: {args.tmas_decay_mode}")
    if args.tmas_decay_mode == 'custom':
        print(f"    Custom Lambda: {args.tmas_custom_lambda}")
    print(f"    Decay Floor: {args.tmas_decay_floor}")
    
    if args.tmas_score_aware_decay or args.tmas_bidirectional:
        print(f"\n  🔬 Advanced Options:")
        if args.tmas_score_aware_decay:
            print(f"    ✓ Score-Aware Decay (β={args.tmas_score_aware_beta})")
        if args.tmas_bidirectional:
            print(f"    ✓ Bidirectional Suppression")
    
    print(f"\n🎯 Selection Strategy:")
    print(f"  Optimize Remaining: {'✅ ENABLED' if args.optimize_remaining else '❌ DISABLED'}")
    
    print("=" * 80 + "\n")


def main(args):
    """Main function to process all videos using DBFP with TMAS."""
    
    print_tmas_configuration(args)
    
    # Load data
    print(f"📂 Loading data...")
    print(f"  Scores: {args.score_path}")
    with open(args.score_path) as f:
        all_scores = json.load(f)
    
    print(f"  Frames: {args.frame_path}")
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    print(f"  Metadata: {args.metadata_path}")
    try:
        with open(args.metadata_path) as f:
            metadata = json.load(f)
        print(f"  ✅ Metadata loaded successfully\n")
    except FileNotFoundError:
        print(f"  ❌ Error: Metadata file not found at {args.metadata_path}")
        if args.tmas_mode == 'additive':
            print(f"  Metadata required for additive mode (duration categories)")
        return
    except json.JSONDecodeError:
        print(f"  ❌ Error: Invalid JSON in metadata file")
        return
    
    # Validate data consistency
    if len(metadata) != len(all_scores):
        print(f"  ⚠️  Warning: Metadata length ({len(metadata)}) != Video count ({len(all_scores)})")
        print(f"  Using minimum length for safety")
    
    # Determine how many videos to process
    num_videos_to_process = min(len(all_scores), len(metadata))
    if args.num_videos is not None:
        num_videos_to_process = min(args.num_videos, num_videos_to_process)
        print(f"  🎯 DEMO MODE: Processing first {num_videos_to_process} videos only\n")
    
    print(f"📊 Processing Summary:")
    print(f"  Total videos available: {len(all_scores)}")
    print(f"  Videos to process: {num_videos_to_process}\n")
    
    # Process each video
    selected_frames_all = []
    all_tmas_stats = []
    
    # Tracking statistics
    processing_stats = {
        'total_processed': 0,
        'total_frames_selected': 0,
        'videos_below_max': 0,
        'errors': 0,
        'avg_R0': [],
        'avg_lambda': [],
        'avg_effective_radius_t0': [],
    }
    
    print(f"🔄 Processing videos...\n")
    
    for idx in range(num_videos_to_process):
        scores = all_scores[idx]
        frame_ids = all_frame_ids[idx]
        metadata_entry = metadata[idx]
        
        # Show progress
        if num_videos_to_process <= 20:
            # Detailed progress for small batches
            if args.dataset_name == 'longvideobench':
                duration_info = metadata_entry.get('duration_group', 'unknown')
            else:
                duration_info = metadata_entry.get('duration', 'unknown')
            print(f"  [{idx + 1}/{num_videos_to_process}] Duration: {duration_info}, Frames: {len(scores)}")
        elif (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{num_videos_to_process} videos...")
        
        try:
            # Get base radius for additive mode
            if args.tmas_mode == 'additive':
                base_radius = get_base_suppression_radius(metadata_entry, args, args.dataset_name)
            else:
                base_radius = None
            
            # Create TMAS config for this video
            tmas_config = create_tmas_config(args, base_radius)
            
            # Process video
            selected_frames, tmas_stats = process_video(
                scores=scores,
                frame_ids=frame_ids,
                max_frames=args.max_num_frames,
                tmas_config=tmas_config,
                args=args
            )
            
            # Store results
            selected_frames_all.append(selected_frames)
            all_tmas_stats.append(tmas_stats)
            
            # Update statistics
            processing_stats['total_processed'] += 1
            processing_stats['total_frames_selected'] += len(selected_frames)
            
            if len(selected_frames) < args.max_num_frames:
                processing_stats['videos_below_max'] += 1
            
            if 'short_video' not in tmas_stats:
                processing_stats['avg_R0'].append(tmas_stats.get('R0', 0))
                processing_stats['avg_lambda'].append(tmas_stats.get('lambda', 0))
                
                if args.tmas_mode == 'auto':
                    processing_stats['avg_effective_radius_t0'].append(tmas_stats.get('max_radius_at_t0', 0))
                elif args.tmas_mode == 'additive':
                    processing_stats['avg_effective_radius_t0'].append(tmas_stats.get('max_radius_at_t0', 0))
            
        except Exception as e:
            print(f"    ❌ Error processing video {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            selected_frames_all.append([])
            all_tmas_stats.append({'error': str(e)})
            processing_stats['errors'] += 1
    
    # Save selected frames
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_file if os.path.isdir(args.output_file) else 
                                os.path.dirname(args.output_file), 
                                build_output_filename(args))
    
    print(f"\n💾 Saving results...")
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    print(f"  ✅ Selected frames saved to: {output_path}")
    
    # Save TMAS statistics
    stats_output_path = output_path.replace('.json', '_tmas_stats.json')
    with open(stats_output_path, 'w') as f:
        json.dump(all_tmas_stats, f, indent=2)
    print(f"  ✅ TMAS statistics saved to: {stats_output_path}")
    
    # Print final statistics
    print(f"\n{'=' * 80}")
    print(f"✅ Processing Complete!")
    print(f"{'=' * 80}")
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Videos processed: {processing_stats['total_processed']}")
    print(f"  Total frames selected: {processing_stats['total_frames_selected']}")
    print(f"  Videos with < max frames: {processing_stats['videos_below_max']}")
    print(f"  Errors: {processing_stats['errors']}")
    
    # Frame count statistics
    frame_counts = [len(frames) for frames in selected_frames_all if len(frames) > 0]
    if frame_counts:
        print(f"\n📊 Frame Selection Statistics:")
        print(f"  Videos with frames: {len(frame_counts)}")
        print(f"  Avg frames per video: {np.mean(frame_counts):.2f}")
        print(f"  Std frames per video: {np.std(frame_counts):.2f}")
        print(f"  Min frames: {np.min(frame_counts)}")
        print(f"  Max frames: {np.max(frame_counts)}")
        print(f"  Median frames: {np.median(frame_counts):.1f}")
    
    # TMAS-specific statistics
    if processing_stats['avg_R0']:
        print(f"\n⚡ TMAS Statistics:")
        print(f"  Avg R0: {np.mean(processing_stats['avg_R0']):.3f}")
        print(f"  Std R0: {np.std(processing_stats['avg_R0']):.3f}")
        print(f"  Avg λ (decay rate): {np.mean(processing_stats['avg_lambda']):.6f}")
        print(f"  Std λ: {np.std(processing_stats['avg_lambda']):.6f}")
        
        if processing_stats['avg_effective_radius_t0']:
            print(f"  Avg Max Radius (t=0): {np.mean(processing_stats['avg_effective_radius_t0']):.3f}")
            print(f"  Std Max Radius (t=0): {np.std(processing_stats['avg_effective_radius_t0']):.3f}")
    
    # Sample decay profiles
    valid_stats = [s for s in all_tmas_stats if 'decay_profile' in s]
    if valid_stats and num_videos_to_process <= 5:
        print(f"\n📉 Sample Decay Profiles:")
        for i, stats in enumerate(valid_stats[:3]):
            print(f"\n  Video {i+1}:")
            print(f"    R0: {stats['R0']}, λ: {stats['lambda']}")
            decay_profile = stats.get('decay_profile', {})
            for dt, radius in decay_profile.items():
                print(f"      {dt}: {radius}")
    
    # Suppression statistics
    suppression_counts = []
    for stats in all_tmas_stats:
        if 'suppression_stats' in stats:
            suppression_counts.append(stats['suppression_stats']['total_suppressions'])
    
    if suppression_counts:
        print(f"\n🚫 Suppression Statistics:")
        print(f"  Videos with suppressions: {len(suppression_counts)}")
        print(f"  Avg suppressions per video: {np.mean(suppression_counts):.1f}")
        print(f"  Max suppressions: {np.max(suppression_counts)}")
        print(f"  Total suppressions: {np.sum(suppression_counts)}")
    
    # Mode-specific insights
    if args.tmas_mode == 'auto':
        print(f"\n🤖 Auto Mode Insights:")
        print(f"  R0 automatically derived from video properties")
        print(f"  No manual tuning required")
        print(f"  Scaling: {args.tmas_auto_scaling}")
        print(f"  Coverage: {args.tmas_auto_coverage}")
    
    elif args.tmas_mode == 'additive':
        print(f"\n➕ Additive Mode Insights:")
        print(f"  Base radius from per-group configuration")
        print(f"  TMAS contribution strategy: {args.tmas_delta_strategy}")
        
        if args.tmas_delta_strategy == 'proportional':
            print(f"  Delta_R = Base × {args.tmas_multiplier}")
        elif args.tmas_delta_strategy == 'fixed':
            print(f"  Fixed Delta_R = {args.tmas_delta_R}")
        elif args.tmas_delta_strategy == 'adaptive':
            print(f"  Adaptive Delta_R (decreases as selection progresses)")
    
    # Warnings and recommendations
    print(f"\n💡 Analysis:")
    
    if processing_stats['videos_below_max'] > processing_stats['total_processed'] * 0.2:
        print(f"  ⚠️  {processing_stats['videos_below_max']} videos ({processing_stats['videos_below_max']/processing_stats['total_processed']*100:.1f}%) have < max frames")
        print(f"     Consider lowering --min_score_threshold or enabling --optimize_remaining")
    
    if processing_stats['errors'] > 0:
        print(f"  ⚠️  {processing_stats['errors']} videos failed to process")
        print(f"     Check error messages above for details")
    
    if frame_counts and np.std(frame_counts) > 5:
        print(f"  ℹ️  High variance in frame counts (std={np.std(frame_counts):.2f})")
        print(f"     This is expected for diverse video lengths")
    
    if not args.optimize_remaining and processing_stats['videos_below_max'] > 0:
        print(f"  💡 Consider using --optimize_remaining to fill remaining slots")
    
    print(f"\n{'=' * 80}")
    print(f"🎉 All done! Output saved to: {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)