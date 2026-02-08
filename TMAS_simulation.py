import numpy as np
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import os


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@dataclass
class TMASConfig:
    """Configuration for TMAS."""
    mode: str = 'auto'
    auto_scaling: str = 'hybrid'
    auto_coverage: float = 1.0
    auto_min_radius: float = 1.0
    auto_max_radius: float = None
    delta_strategy: str = 'proportional'
    delta_R: float = 3.0
    multiplier: float = 0.7
    base_radius: float = None
    decay_mode: str = 'half_life'
    custom_lambda: float = 0.01
    decay_floor: float = 0.0
    budget_half_life_ratio: float = 0.25
    use_curvature: bool = True
    curvature_method: str = 'gradient_change'  # Changed default
    curvature_smoothing: int = 1
    curvature_normalize: str = 'max'
    curvature_clip_percentile: float = 95.0


class SimpleFrameSelectionSimulator:
    """
    Simplified simulator that produces clean output for visualization.
    """
    
    def __init__(self, 
                 scores: np.ndarray,
                 frame_ids: np.ndarray,
                 target_frames: int,
                 tmas_config: TMASConfig,
                 doc_id: Any = None):
        
        self.raw_scores = np.asarray(scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.target_frames = target_frames
        self.config = tmas_config
        self.doc_id = doc_id
        
        self.N = len(scores)
        
        # Normalize scores
        self.normalized_scores = self._normalize_scores()
        
        # Compute curvature
        self.curvature = self._compute_curvature()
        
        # TMAS parameters
        self.R0 = self._calculate_R0()
        self.lambda_decay = self._calculate_lambda()
        
        # Selection state
        self.selection_map: Dict[int, int] = {}  # idx -> selection_number
        self.selection_counter = 0
        self.selected_indices: List[int] = []
        
    def _normalize_scores(self) -> np.ndarray:
        if self.N == 0:
            return np.array([])
        score_min, score_max = self.raw_scores.min(), self.raw_scores.max()
        if score_max > score_min:
            return (self.raw_scores - score_min) / (score_max - score_min)
        return np.ones_like(self.raw_scores) * 0.5
    
    def _compute_curvature(self) -> np.ndarray:
        """Compute curvature using the configured method."""
        if self.N < 3:
            return np.zeros(self.N)
        
        if self.config.curvature_method == 'second_derivative':
            curvature = self._second_derivative_curvature()
        elif self.config.curvature_method == 'laplacian':
            curvature = self._second_derivative_curvature()  # Same as second derivative in 1D
        elif self.config.curvature_method == 'gradient_change':
            curvature = self._gradient_change_curvature()
        else:
            curvature = self._second_derivative_curvature()
        
        curvature = np.abs(curvature)
        
        # Smoothing
        if self.config.curvature_smoothing > 0:
            try:
                from scipy.ndimage import gaussian_filter1d
                sigma = self.config.curvature_smoothing / 3.0
                curvature = gaussian_filter1d(curvature, sigma=sigma, mode='nearest')
            except ImportError:
                pass
        
        # Clip and normalize
        if self.config.curvature_clip_percentile < 100.0:
            clip_value = np.percentile(curvature, self.config.curvature_clip_percentile)
            curvature = np.clip(curvature, 0, clip_value)
        
        max_curv = np.max(curvature)
        if max_curv > 1e-10:
            curvature = curvature / max_curv
        
        return curvature
    
    def _second_derivative_curvature(self) -> np.ndarray:
        """Compute curvature using second derivative."""
        curvature = np.zeros(self.N)
        for i in range(1, self.N - 1):
            curvature[i] = self.normalized_scores[i + 1] - 2 * self.normalized_scores[i] + self.normalized_scores[i - 1]
        
        if self.N >= 2:
            curvature[0] = self.normalized_scores[1] - self.normalized_scores[0]
            curvature[-1] = self.normalized_scores[-1] - self.normalized_scores[-2]
        
        return curvature
    
    def _gradient_change_curvature(self) -> np.ndarray:
        """Compute curvature using gradient change method."""
        # Compute gradient
        gradient = np.gradient(self.normalized_scores)
        # Compute change in gradient (second gradient)
        curvature = np.gradient(gradient)
        return curvature
    
    def _calculate_R0(self) -> float:
        if self.config.mode == 'auto':
            if self.N == 0 or self.target_frames == 0:
                return self.config.auto_min_radius
            
            base_spacing = self.N / self.target_frames
            
            if self.config.auto_scaling == 'linear':
                R0 = base_spacing * self.config.auto_coverage
            elif self.config.auto_scaling == 'sqrt':
                R0 = np.sqrt(base_spacing) * self.config.auto_coverage
            else:  # 'hybrid'
                R0 = (base_spacing ** 0.7) * self.config.auto_coverage
            
            R0 = max(self.config.auto_min_radius, R0)
            if self.config.auto_max_radius is not None:
                R0 = min(self.config.auto_max_radius, R0)
            
            return float(R0)
        else:
            return float(self.config.base_radius) if self.config.base_radius else 3.0
    
    def _calculate_lambda(self) -> float:
        if self.N == 0:
            return 0.01
        
        if self.config.decay_mode == 'video_length':
            return float(np.log(max(2, self.N)) / self.N)
        elif self.config.decay_mode == 'half_life':
            if self.target_frames == 0:
                return 0.01
            half_life = self.N / (2 * self.target_frames)
            return float(np.log(2) / max(1, half_life))
        elif self.config.decay_mode == 'quartile':
            quartile_distance = self.N / 4
            return float(np.log(4) / max(1, quartile_distance))
        elif self.config.decay_mode == 'budget_based':
            if self.target_frames == 0:
                return 0.01
            half_life_selections = max(1, self.target_frames * self.config.budget_half_life_ratio)
            return float(np.log(2) / half_life_selections)
        else:
            return float(self.config.custom_lambda)
    
    def _get_delta_R(self) -> float:
        if self.config.mode != 'additive':
            return 0.0
        
        if self.config.delta_strategy == 'fixed':
            return float(self.config.delta_R)
        elif self.config.delta_strategy == 'proportional':
            return float(self.R0 * self.config.multiplier)
        else:
            remaining_budget = max(1, self.target_frames - len(self.selection_map))
            budget_ratio = remaining_budget / self.target_frames
            return float(self.R0 * self.config.multiplier * budget_ratio)
    
    def _get_suppression_memory(self) -> Dict[str, Dict]:
        """Get current suppression radius for each selected frame."""
        memory = {}
        delta_R = self._get_delta_R()
        
        for idx in self.selected_indices:
            selection_num = self.selection_map[idx]
            frame_id = int(self.frame_ids[idx])
            kappa = float(self.curvature[idx])
            
            # Calculate decay distance
            if self.config.decay_mode == 'budget_based':
                decay_distance = float(self.selection_counter - selection_num)
            else:
                decay_distance = 0.0
            
            decay_factor = float(np.exp(-self.lambda_decay * decay_distance))
            
            # Calculate base radius
            if self.config.mode == 'auto':
                base_radius = float(self.R0 * decay_factor)
            else:
                tmas_contribution = float(delta_R * decay_factor)
                if self.config.decay_floor > 0:
                    tmas_contribution = max(tmas_contribution, delta_R * self.config.decay_floor)
                base_radius = float(self.R0 + tmas_contribution)
            
            # Radius with and without curvature
            radius_with_curv = float(base_radius / (1.0 + kappa))
            radius_without_curv = float(base_radius)
            
            memory[str(frame_id)] = {
                'idx': int(idx),
                'radius_with_curvature': round(radius_with_curv, 4),
                'radius_without_curvature': round(radius_without_curv, 4),
                'curvature': round(kappa, 4),
                'decay_factor': round(decay_factor, 4),
                'selection_order': int(selection_num)
            }
        
        return memory
    
    def _is_suppressed(self, candidate_idx: int, use_curvature: bool) -> Tuple[bool, Optional[int]]:
        """Check if candidate is suppressed."""
        if not self.selection_map:
            return False, None
        
        kappa = float(self.curvature[candidate_idx])
        delta_R = self._get_delta_R()
        
        for selected_idx in self.selected_indices:
            selection_num = self.selection_map[selected_idx]
            
            spatial_distance = abs(candidate_idx - selected_idx)
            if spatial_distance == 0:
                continue
            
            # Calculate decay distance
            if self.config.decay_mode == 'budget_based':
                decay_distance = float(self.selection_counter - selection_num)
            else:
                decay_distance = float(spatial_distance)
            
            # Calculate base radius
            if self.config.mode == 'auto':
                base_radius = float(self.R0 * np.exp(-self.lambda_decay * decay_distance))
            else:
                tmas_contribution = float(delta_R * np.exp(-self.lambda_decay * decay_distance))
                if self.config.decay_floor > 0:
                    tmas_contribution = max(tmas_contribution, delta_R * self.config.decay_floor)
                base_radius = float(self.R0 + tmas_contribution)
            
            # Apply curvature if enabled
            if use_curvature:
                effective_radius = base_radius / (1.0 + kappa)
            else:
                effective_radius = base_radius
            
            if spatial_distance <= effective_radius:
                return True, int(self.frame_ids[selected_idx])
        
        return False, None
    
    def _compute_comparison(self, sorted_candidates: List) -> Dict:
        """Compare what would happen with vs without curvature."""
        suppressed_only_with_curv = []
        suppressed_only_without_curv = []
        
        for score, idx in sorted_candidates:
            if idx in self.selection_map:
                continue
            
            suppressed_with = self._is_suppressed(idx, use_curvature=True)[0]
            suppressed_without = self._is_suppressed(idx, use_curvature=False)[0]
            
            frame_id = int(self.frame_ids[idx])
            
            if suppressed_with and not suppressed_without:
                suppressed_only_with_curv.append({
                    'frame_id': frame_id,
                    'idx': int(idx),
                    'score': round(float(score), 4),
                    'curvature': round(float(self.curvature[idx]), 4)
                })
            elif suppressed_without and not suppressed_with:
                suppressed_only_without_curv.append({
                    'frame_id': frame_id,
                    'idx': int(idx),
                    'score': round(float(score), 4),
                    'curvature': round(float(self.curvature[idx]), 4)
                })
        
        return {
            'would_be_suppressed_only_with_curvature': suppressed_only_with_curv[:10],
            'would_be_suppressed_only_without_curvature': suppressed_only_without_curv[:10],
            'total_different_with_curvature': len(suppressed_only_with_curv),
            'total_different_without_curvature': len(suppressed_only_without_curv)
        }
    
    def _run_selection_pass(self) -> None:
        """Run the selection algorithm to populate selected_indices."""
        valid_mask = self.normalized_scores >= 0.0
        valid_indices = np.where(valid_mask)[0]
        sorted_candidates = sorted(
            [(float(self.normalized_scores[idx]), int(idx)) for idx in valid_indices],
            key=lambda x: (-x[0], x[1])
        )
        
        for score, idx in sorted_candidates:
            if len(self.selected_indices) >= self.target_frames:
                break
            
            is_suppressed, _ = self._is_suppressed(idx, use_curvature=self.config.use_curvature)
            
            if is_suppressed:
                continue
            
            self.selection_counter += 1
            self.selection_map[idx] = self.selection_counter
            self.selected_indices.append(idx)
    
    def run_simple_simulation(self) -> Dict:
        """Run simulation and return simple, clean output."""
        print(f"\n{'='*60}")
        print(f"🔬 SIMPLE FRAME SELECTION SIMULATION")
        print(f"{'='*60}")
        print(f"Total Frames: {self.N}")
        print(f"Target Frames: {self.target_frames}")
        print(f"Curvature Method: {self.config.curvature_method}")
        print(f"Auto Coverage: {self.config.auto_coverage}")
        print(f"R0: {self.R0:.4f}")
        print(f"Lambda: {self.lambda_decay:.6f}")
        print(f"{'='*60}\n")
        
        # First pass: run selection to get final selected frames
        self._run_selection_pass()
        
        # Store final results
        final_selected_by_order = [int(self.frame_ids[i]) for i in self.selected_indices]
        final_selected_sorted = sorted(final_selected_by_order)
        
        # Reset state for detailed step-by-step recording
        self.selection_map = {}
        self.selection_counter = 0
        self.selected_indices = []
        
        # Prepare output structure
        output = {
            'selected_frames': {
                'by_selection_order': final_selected_by_order,
                'sorted_by_frame_id': final_selected_sorted,
                'num_selected': len(final_selected_by_order)
            },
            'input': {
                'frame_ids': [int(x) for x in self.frame_ids],
                'scores': [round(float(x), 4) for x in self.normalized_scores],
                'curvatures': [round(float(x), 4) for x in self.curvature]
            },
            'config': {
                'doc_id': self.doc_id,
                'target_frames': self.target_frames,
                'total_input_frames': self.N,
                'tmas_mode': self.config.mode,
                'decay_mode': self.config.decay_mode,
                'use_curvature': self.config.use_curvature,
                'curvature_method': self.config.curvature_method,
                'auto_coverage': self.config.auto_coverage,
                'auto_scaling': self.config.auto_scaling,
                'R0': round(self.R0, 4),
                'lambda_decay': round(self.lambda_decay, 6)
            },
            'steps': []
        }
        
        # Get sorted candidates
        valid_mask = self.normalized_scores >= 0.0
        valid_indices = np.where(valid_mask)[0]
        sorted_candidates = sorted(
            [(float(self.normalized_scores[idx]), int(idx)) for idx in valid_indices],
            key=lambda x: (-x[0], x[1])
        )
        
        # Second pass: detailed step-by-step
        candidate_pointer = 0
        
        while len(self.selected_indices) < self.target_frames and candidate_pointer < len(sorted_candidates):
            score, idx = sorted_candidates[candidate_pointer]
            candidate_pointer += 1
            
            is_suppressed, _ = self._is_suppressed(idx, use_curvature=self.config.use_curvature)
            
            if is_suppressed:
                continue
            
            self.selection_counter += 1
            self.selection_map[idx] = self.selection_counter
            self.selected_indices.append(idx)
            
            frame_id = int(self.frame_ids[idx])
            
            step_data = {
                'step': self.selection_counter,
                'selected_frame_id': frame_id,
                'selected_idx': int(idx),
                'selected_score': round(float(score), 4),
                'selected_curvature': round(float(self.curvature[idx]), 4),
                'selected_list': [int(self.frame_ids[i]) for i in self.selected_indices],
                'suppression_memory': self._get_suppression_memory(),
                'comparison': self._compute_comparison(sorted_candidates)
            }
            
            output['steps'].append(step_data)
            
            print(f"Step {self.selection_counter}: Selected frame {frame_id} "
                  f"(score={score:.4f}, curv={self.curvature[idx]:.4f})")
        
        print(f"\n{'='*60}")
        print(f"✅ Selected {len(self.selected_indices)} frames")
        print(f"{'='*60}\n")
        
        return convert_to_serializable(output)


def run_simple_simulation(
    dataset_name: str,
    feature_model: str,
    score_path: str,
    frame_path: str,
    metadata_path: str,
    doc_id: int,
    max_frames: int = 32,
    tmas_mode: str = 'auto',
    decay_mode: str = 'budget_based',
    use_curvature: bool = True,
    curvature_method: str = 'gradient_change',  # Match your file
    auto_coverage: float = 1.73,  # Match your file
    auto_scaling: str = 'hybrid',  # Match your file
    curvature_normalize: str = 'max',  # Match your file
    curvature_clip_percentile: float = 95.0,  # Match your file
    output_path: str = None,
    **kwargs
) -> Dict:
    """
    Run simple simulation and return clean output.
    """
    print(f"📂 Loading data...")
    
    with open(score_path) as f:
        all_scores = json.load(f)
    with open(frame_path) as f:
        all_frame_ids = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    if doc_id < 0 or doc_id >= len(all_scores):
        raise ValueError(f"doc_id {doc_id} out of range [0, {len(all_scores)})")
    
    scores = np.array(all_scores[doc_id])
    frame_ids = np.array(all_frame_ids[doc_id])
    meta = metadata[doc_id]
    
    print(f"📹 Video: doc_id={doc_id}, frames={len(scores)}")
    print(f"   Metadata: {meta}")
    
    # Determine base radius for additive mode
    base_radius = None
    if tmas_mode == 'additive':
        if dataset_name == 'longvideobench':
            duration_group = meta.get('duration_group', 60)
            radius_map = {15: 2.0, 60: 3.0, 600: 5.0, 3600: 8.0}
            base_radius = radius_map.get(duration_group, 3.0)
        elif dataset_name == 'videomme':
            duration = meta.get('duration', 'medium')
            radius_map = {'short': 2.0, 'medium': 3.0, 'long': 5.0}
            base_radius = radius_map.get(duration, 3.0)
        else:
            base_radius = 3.0
    
    # Create config - MATCHING YOUR FILE SETTINGS
    config = TMASConfig(
        mode=tmas_mode,
        auto_scaling=auto_scaling,
        auto_coverage=auto_coverage,
        auto_min_radius=kwargs.get('auto_min_radius', 1.0),
        auto_max_radius=kwargs.get('auto_max_radius', None),
        delta_strategy=kwargs.get('delta_strategy', 'proportional'),
        delta_R=kwargs.get('delta_R', 3.0),
        multiplier=kwargs.get('multiplier', 0.7),
        base_radius=base_radius,
        decay_mode=decay_mode,
        custom_lambda=kwargs.get('custom_lambda', 0.01),
        decay_floor=kwargs.get('decay_floor', 0.0),
        budget_half_life_ratio=kwargs.get('budget_half_life_ratio', 0.25),
        use_curvature=use_curvature,
        curvature_method=curvature_method,
        curvature_smoothing=kwargs.get('curvature_smoothing', 1),
        curvature_normalize=curvature_normalize,
        curvature_clip_percentile=curvature_clip_percentile,
    )
    
    print(f"\n📋 Configuration:")
    print(f"   tmas_mode: {tmas_mode}")
    print(f"   decay_mode: {decay_mode}")
    print(f"   curvature_method: {curvature_method}")
    print(f"   auto_coverage: {auto_coverage}")
    print(f"   auto_scaling: {auto_scaling}")
    print(f"   curvature_normalize: {curvature_normalize}")
    print(f"   curvature_clip_percentile: {curvature_clip_percentile}")
    
    # Run simulation
    simulator = SimpleFrameSelectionSimulator(
        scores=scores,
        frame_ids=frame_ids,
        target_frames=max_frames,
        tmas_config=config,
        doc_id=doc_id
    )
    
    result = simulator.run_simple_simulation()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {output_path}")
    
    return result


# ============================================================================
# EXAMPLE USAGE - MATCHING YOUR FILE SETTINGS
# ============================================================================

if __name__ == '__main__':
    # Settings matching: selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73
    result = run_simple_simulation(
        dataset_name='longvideobench',
        feature_model='blip',
        score_path='./outscores/longvideobench/blip/scores.json',
        frame_path='./outscores/longvideobench/blip/frames.json',
        metadata_path='./datasets/longvideobench/metadata.json',
        doc_id=25,
        max_frames=32,
        # Matching your file settings:
        tmas_mode='auto',
        decay_mode='budget_based',
        use_curvature=True,
        curvature_method='gradient_change',  # curvgrad
        auto_coverage=1.73,                   # cov1.73
        auto_scaling='hybrid',                # hybrid
        curvature_normalize='max',            # normmax
        curvature_clip_percentile=95.0,       # clip95
        output_path='./simulation_results/doc_25_simple.json'
    )
    
    print("\n" + "="*60)
    print("📋 SELECTED FRAMES:")
    print("="*60)
    print(f"By selection order: {result['selected_frames']['by_selection_order']}")
    print(f"Sorted: {result['selected_frames']['sorted_by_frame_id']}")
    
    # Now compare with your original file
    print("\n" + "="*60)
    print("🔍 COMPARE WITH YOUR ORIGINAL FILE:")
    print("="*60)
    try:
        with open('./outscores/longvideobench/blip/selected_tmas_longvideobench_blip_k32_auto_budget_based_curvgrad_normmax_clip95_hybrid_cov1.73_c.json') as f:
            original = json.load(f)
        original_frames = original[25] if isinstance(original, list) else original.get('25', [])
        print(f"Original file frames: {sorted(original_frames)}")
        print(f"Simulation frames:    {result['selected_frames']['sorted_by_frame_id']}")
        print(f"Match: {sorted(original_frames) == result['selected_frames']['sorted_by_frame_id']}")
    except Exception as e:
        print(f"Could not load original file: {e}")
        print("Please provide the correct path to compare.")