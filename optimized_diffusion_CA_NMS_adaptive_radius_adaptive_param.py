import heapq
import json
import numpy as np
import argparse
import os
import hashlib
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Set

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP: Diffusion-Based Frame Propagation with Duration-Adaptive Hyperparameters')
    
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
                        help='Path to metadata JSON file containing duration_group/duration fields')
    parser.add_argument('--max_num_frames', type=int, default=32,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    parser.add_argument('--alpha', type=float, default=.85,
                        help='Diffusion decay factor (0-1): controls original vs neighbor influence')
    parser.add_argument('--diffusion_iterations', type=int, default=3,
                        help='Number of diffusion iterations (default: log2(N))')
    
    # ========================================================================
    # ADAPTIVE BASE SUPPRESSION RADIUS (LongVideoBench)
    # ========================================================================
    parser.add_argument('--suppression_radius_15', type=float, default=2.0,
                        help='BASE suppression radius for duration_group=15')
    parser.add_argument('--suppression_radius_60', type=float, default=3.0,
                        help='BASE suppression radius for duration_group=60')
    parser.add_argument('--suppression_radius_600', type=float, default=5.0,
                        help='BASE suppression radius for duration_group=600')
    parser.add_argument('--suppression_radius_3600', type=float, default=8.0,
                        help='BASE suppression radius for duration_group=3600')
    
    # ========================================================================
    # ADAPTIVE BASE SUPPRESSION RADIUS (VideoMME)
    # ========================================================================
    parser.add_argument('--suppression_radius_short', type=float, default=2.0,
                        help='BASE suppression radius for duration=short')
    parser.add_argument('--suppression_radius_medium', type=float, default=3.0,
                        help='BASE suppression radius for duration=medium')
    parser.add_argument('--suppression_radius_long', type=float, default=5.0,
                        help='BASE suppression radius for duration=long')
    
    # ========================================================================
    # ADAPTIVE LAMBDA (Similarity Weight) - LongVideoBench
    # ========================================================================
    parser.add_argument('--lambda_sim_15', type=float, default=1.0,
                        help='Lambda for duration_group=15 (short videos need less extension)')
    parser.add_argument('--lambda_sim_60', type=float, default=1.5,
                        help='Lambda for duration_group=60')
    parser.add_argument('--lambda_sim_600', type=float, default=2.0,
                        help='Lambda for duration_group=600 (longer videos need more extension)')
    parser.add_argument('--lambda_sim_3600', type=float, default=2.5,
                        help='Lambda for duration_group=3600 (very long videos)')
    
    # ========================================================================
    # ADAPTIVE LAMBDA (Similarity Weight) - VideoMME
    # ========================================================================
    parser.add_argument('--lambda_sim_short', type=float, default=1.0,
                        help='Lambda for duration=short')
    parser.add_argument('--lambda_sim_medium', type=float, default=1.5,
                        help='Lambda for duration=medium')
    parser.add_argument('--lambda_sim_long', type=float, default=2.0,
                        help='Lambda for duration=long')
    
    # ========================================================================
    # ADAPTIVE SIGMA_SCORE (Score Strictness for Gaussian) - LongVideoBench
    # ========================================================================
    parser.add_argument('--sigma_score_15', type=float, default=0.15,
                        help='Sigma_s for duration_group=15 (stricter for short videos)')
    parser.add_argument('--sigma_score_60', type=float, default=0.2,
                        help='Sigma_s for duration_group=60')
    parser.add_argument('--sigma_score_600', type=float, default=0.25,
                        help='Sigma_s for duration_group=600 (more lenient for long videos)')
    parser.add_argument('--sigma_score_3600', type=float, default=0.3,
                        help='Sigma_s for duration_group=3600')
    
    # ========================================================================
    # ADAPTIVE SIGMA_SCORE (Score Strictness for Gaussian) - VideoMME
    # ========================================================================
    parser.add_argument('--sigma_score_short', type=float, default=0.05,
                        help='Sigma_s for duration=short')
    parser.add_argument('--sigma_score_medium', type=float, default=0.15,
                        help='Sigma_s for duration=medium')
    parser.add_argument('--sigma_score_long', type=float, default=0.3,
                        help='Sigma_s for duration=long')
    
    # ========================================================================
    # ADAPTIVE TAU_TEMPORAL (Temporal Decay for Gaussian) - LongVideoBench
    # ========================================================================
    parser.add_argument('--tau_temporal_15', type=float, default=5.0,
                        help='Tau for duration_group=15 (faster decay for short videos)')
    parser.add_argument('--tau_temporal_60', type=float, default=8.0,
                        help='Tau for duration_group=60')
    parser.add_argument('--tau_temporal_600', type=float, default=12.0,
                        help='Tau for duration_group=600 (slower decay for long videos)')
    parser.add_argument('--tau_temporal_3600', type=float, default=15.0,
                        help='Tau for duration_group=3600')
    
    # ========================================================================
    # ADAPTIVE TAU_TEMPORAL (Temporal Decay for Gaussian) - VideoMME
    # ========================================================================
    parser.add_argument('--tau_temporal_short', type=float, default=5.0,
                        help='Tau for duration=short')
    parser.add_argument('--tau_temporal_medium', type=float, default=8.0,
                        help='Tau for duration=medium')
    parser.add_argument('--tau_temporal_long', type=float, default=12.0,
                        help='Tau for duration=long')
    
    # ========================================================================
    # OTHER PARAMETERS
    # ========================================================================
    parser.add_argument('--similarity_method', type=str, default='gaussian',
                        choices=['cosine', 'gaussian'],
                        help='Similarity calculation method')
    parser.add_argument('--overlap_radius_multiplier', type=float, default=2.0,
                        help='Multiplier for radius when overlap detected')
    parser.add_argument('--edge_weight_type', type=str, default='temporal',
                        choices=['uniform', 'score_diff', 'temporal'],
                        help='Edge weight type for diffusion')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory for selected frames')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (default: all)')
    parser.add_argument('--min_score_threshold', type=float, default=0,
                        help='Minimum normalized score threshold (0-1)')
    parser.add_argument('--no_optimize_remaining', dest='optimize_remaining', 
                        action='store_false', default=False,
                        help='If set, disables filling remaining slots')
    
    return parser.parse_args()


def generate_config_hash(args) -> str:
    """
    Generate a unique hash from configuration parameters.
    Uses MD5 hash of sorted config string.
    """
    config_dict = {
        'dataset_name': args.dataset_name,
        'extract_feature_model': args.extract_feature_model,
        'max_num_frames': args.max_num_frames,
        'ratio': args.ratio,
        'alpha': args.alpha,
        'diffusion_iterations': args.diffusion_iterations,
        'similarity_method': args.similarity_method,
        'edge_weight_type': args.edge_weight_type,
        'min_score_threshold': args.min_score_threshold,
        'optimize_remaining': args.optimize_remaining,
        'overlap_radius_multiplier': args.overlap_radius_multiplier,
    }
    
    # Add dataset-specific parameters
    if args.dataset_name == 'longvideobench':
        config_dict.update({
            'suppression_radius_15': args.suppression_radius_15,
            'suppression_radius_60': args.suppression_radius_60,
            'suppression_radius_600': args.suppression_radius_600,
            'suppression_radius_3600': args.suppression_radius_3600,
            'lambda_sim_15': args.lambda_sim_15,
            'lambda_sim_60': args.lambda_sim_60,
            'lambda_sim_600': args.lambda_sim_600,
            'lambda_sim_3600': args.lambda_sim_3600,
        })
        if args.similarity_method == 'gaussian':
            config_dict.update({
                'sigma_score_15': args.sigma_score_15,
                'sigma_score_60': args.sigma_score_60,
                'sigma_score_600': args.sigma_score_600,
                'sigma_score_3600': args.sigma_score_3600,
                'tau_temporal_15': args.tau_temporal_15,
                'tau_temporal_60': args.tau_temporal_60,
                'tau_temporal_600': args.tau_temporal_600,
                'tau_temporal_3600': args.tau_temporal_3600,
            })
    elif args.dataset_name == 'videomme':
        config_dict.update({
            'suppression_radius_short': args.suppression_radius_short,
            'suppression_radius_medium': args.suppression_radius_medium,
            'suppression_radius_long': args.suppression_radius_long,
            'lambda_sim_short': args.lambda_sim_short,
            'lambda_sim_medium': args.lambda_sim_medium,
            'lambda_sim_long': args.lambda_sim_long,
        })
        if args.similarity_method == 'gaussian':
            config_dict.update({
                'sigma_score_short': args.sigma_score_short,
                'sigma_score_medium': args.sigma_score_medium,
                'sigma_score_long': args.sigma_score_long,
                'tau_temporal_short': args.tau_temporal_short,
                'tau_temporal_medium': args.tau_temporal_medium,
                'tau_temporal_long': args.tau_temporal_long,
            })
    
    # Create deterministic string from sorted dict
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Generate MD5 hash (first 12 characters for readability)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:12]


def get_descriptive_name(args) -> str:
    """
    Generate a human-readable descriptive name for the configuration.
    """
    name = f"DBFP_{args.dataset_name}_{args.extract_feature_model}_k{args.max_num_frames}"
    name += f"_alpha{args.alpha}_{args.similarity_method}"
    
    if args.dataset_name == 'longvideobench':
        name += f"_r[{args.suppression_radius_15},{args.suppression_radius_60},{args.suppression_radius_600},{args.suppression_radius_3600}]"
        name += f"_l[{args.lambda_sim_15},{args.lambda_sim_60},{args.lambda_sim_600},{args.lambda_sim_3600}]"
    elif args.dataset_name == 'videomme':
        name += f"_r[{args.suppression_radius_short},{args.suppression_radius_medium},{args.suppression_radius_long}]"
        name += f"_l[{args.lambda_sim_short},{args.lambda_sim_medium},{args.lambda_sim_long}]"
    
    if args.similarity_method == 'gaussian':
        name += "_gaussian_params"
    
    name += f"_ovlp{args.overlap_radius_multiplier}"
    
    if args.optimize_remaining:
        name += "_opt"
    
    return name


def save_config_mapping(config_hash: str, descriptive_name: str, args, 
                       output_dir: str, output_filename: str):
    """
    Save configuration mapping to CSV files.
    
    Creates two CSV files:
    1. config_registry.csv - Maps hash to full configuration
    2. filename_mapping.csv - Maps hash to descriptive names
    """
    registry_path = os.path.join(output_dir, "config_registry.csv")
    mapping_path = os.path.join(output_dir, "filename_mapping.csv")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ========================================================================
    # 1. Save to config_registry.csv (detailed configuration)
    # ========================================================================
    
    config_data = {
        'hash': config_hash,
        'timestamp': timestamp,
        'output_file': output_filename,
        'dataset_name': args.dataset_name,
        'extract_feature_model': args.extract_feature_model,
        'max_num_frames': args.max_num_frames,
        'alpha': args.alpha,
        'similarity_method': args.similarity_method,
        'edge_weight_type': args.edge_weight_type,
        'overlap_radius_multiplier': args.overlap_radius_multiplier,
        'min_score_threshold': args.min_score_threshold,
        'optimize_remaining': args.optimize_remaining,
        'diffusion_iterations': args.diffusion_iterations,
    }
    
    # Add dataset-specific parameters
    if args.dataset_name == 'longvideobench':
        config_data.update({
            'suppression_radius_15': args.suppression_radius_15,
            'suppression_radius_60': args.suppression_radius_60,
            'suppression_radius_600': args.suppression_radius_600,
            'suppression_radius_3600': args.suppression_radius_3600,
            'lambda_sim_15': args.lambda_sim_15,
            'lambda_sim_60': args.lambda_sim_60,
            'lambda_sim_600': args.lambda_sim_600,
            'lambda_sim_3600': args.lambda_sim_3600,
        })
        if args.similarity_method == 'gaussian':
            config_data.update({
                'sigma_score_15': args.sigma_score_15,
                'sigma_score_60': args.sigma_score_60,
                'sigma_score_600': args.sigma_score_600,
                'sigma_score_3600': args.sigma_score_3600,
                'tau_temporal_15': args.tau_temporal_15,
                'tau_temporal_60': args.tau_temporal_60,
                'tau_temporal_600': args.tau_temporal_600,
                'tau_temporal_3600': args.tau_temporal_3600,
            })
    elif args.dataset_name == 'videomme':
        config_data.update({
            'suppression_radius_short': args.suppression_radius_short,
            'suppression_radius_medium': args.suppression_radius_medium,
            'suppression_radius_long': args.suppression_radius_long,
            'lambda_sim_short': args.lambda_sim_short,
            'lambda_sim_medium': args.lambda_sim_medium,
            'lambda_sim_long': args.lambda_sim_long,
        })
        if args.similarity_method == 'gaussian':
            config_data.update({
                'sigma_score_short': args.sigma_score_short,
                'sigma_score_medium': args.sigma_score_medium,
                'sigma_score_long': args.sigma_score_long,
                'tau_temporal_short': args.tau_temporal_short,
                'tau_temporal_medium': args.tau_temporal_medium,
                'tau_temporal_long': args.tau_temporal_long,
            })
    
    # Write or append to config_registry.csv
    file_exists = os.path.exists(registry_path)
    with open(registry_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=config_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(config_data)
    
    # ========================================================================
    # 2. Save to filename_mapping.csv (hash to descriptive name)
    # ========================================================================
    
    mapping_data = {
        'hash': config_hash,
        'filename': output_filename,
        'descriptive_name': descriptive_name,
        'timestamp': timestamp,
        'dataset': args.dataset_name,
        'model': args.extract_feature_model,
        'method': args.similarity_method,
    }
    
    file_exists = os.path.exists(mapping_path)
    with open(mapping_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=mapping_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(mapping_data)
    
    print(f"\n📋 Configuration saved:")
    print(f"   Registry: {registry_path}")
    print(f"   Mapping: {mapping_path}")


def get_adaptive_parameters(metadata_entry: dict, args, dataset_name: str) -> Dict:
    """Get ALL adaptive parameters based on video duration category."""
    params = {}
    
    if dataset_name == 'longvideobench':
        duration_group = metadata_entry.get('duration_group', None)
        
        if duration_group == 15:
            params['suppression_radius'] = args.suppression_radius_15
            params['lambda_sim'] = args.lambda_sim_15
            params['sigma_score'] = args.sigma_score_15
            params['tau_temporal'] = args.tau_temporal_15
        elif duration_group == 60:
            params['suppression_radius'] = args.suppression_radius_60
            params['lambda_sim'] = args.lambda_sim_60
            params['sigma_score'] = args.sigma_score_60
            params['tau_temporal'] = args.tau_temporal_60
        elif duration_group == 600:
            params['suppression_radius'] = args.suppression_radius_600
            params['lambda_sim'] = args.lambda_sim_600
            params['sigma_score'] = args.sigma_score_600
            params['tau_temporal'] = args.tau_temporal_600
        elif duration_group == 3600:
            params['suppression_radius'] = args.suppression_radius_3600
            params['lambda_sim'] = args.lambda_sim_3600
            params['sigma_score'] = args.sigma_score_3600
            params['tau_temporal'] = args.tau_temporal_3600
        else:
            params['suppression_radius'] = 3.0
            params['lambda_sim'] = 1.5
            params['sigma_score'] = 0.2
            params['tau_temporal'] = 8.0
            
    elif dataset_name == 'videomme':
        duration = metadata_entry.get('duration', None)
        
        if duration == 'short':
            params['suppression_radius'] = args.suppression_radius_short
            params['lambda_sim'] = args.lambda_sim_short
            params['sigma_score'] = args.sigma_score_short
            params['tau_temporal'] = args.tau_temporal_short
        elif duration == 'medium':
            params['suppression_radius'] = args.suppression_radius_medium
            params['lambda_sim'] = args.lambda_sim_medium
            params['sigma_score'] = args.sigma_score_medium
            params['tau_temporal'] = args.tau_temporal_medium
        elif duration == 'long':
            params['suppression_radius'] = args.suppression_radius_long
            params['lambda_sim'] = args.lambda_sim_long
            params['sigma_score'] = args.sigma_score_long
            params['tau_temporal'] = args.tau_temporal_long
        else:
            params['suppression_radius'] = 3.0
            params['lambda_sim'] = 1.5
            params['sigma_score'] = 0.2
            params['tau_temporal'] = 8.0
    else:
        params['suppression_radius'] = 3.0
        params['lambda_sim'] = 1.5
        params['sigma_score'] = 0.2
        params['tau_temporal'] = 8.0
    
    return params


# ... [Keep all the DiffusionGraph and KeyframeSelector classes exactly as before] ...


class DiffusionGraph:
    """Represents a temporal graph of video frames with diffusion capabilities."""
    
    def __init__(self, scores: np.ndarray, frame_ids: np.ndarray, 
                 alpha: float = 0.7, edge_weight_type: str = 'uniform'):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.alpha = alpha
        self.edge_weight_type = edge_weight_type
        self.N = len(scores)
        
        if self.N > 0:
            score_min, score_max = self.scores.min(), self.scores.max()
            if score_max > score_min:
                self.scores = (self.scores - score_min) / (score_max - score_min)
            else:
                self.scores = np.ones_like(self.scores) * 0.5
        
        self.diffused_scores = self.scores.copy()
        self.edge_weights = self._build_edge_weights()
    
    def _build_edge_weights(self) -> np.ndarray:
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
        if self.N <= 1:
            return self.diffused_scores
        
        if iterations is None:
            iterations = max(1, int(np.log2(self.N)))
        
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
    """Selects keyframes with duration-adaptive similarity-modulated suppression."""
    
    def __init__(self, diffused_scores: np.ndarray, frame_ids: np.ndarray,
                 original_scores: np.ndarray = None,
                 base_suppression_radius: float = None,
                 min_score_threshold: float = 0.0,
                 similarity_method: str = 'cosine',
                 lambda_sim: float = 1.0,
                 sigma_score: float = 0.1,
                 tau_temporal: float = 5.0,
                 overlap_radius_multiplier: float = 2.0):
        
        self.diffused_scores = np.asarray(diffused_scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.original_scores = original_scores if original_scores is not None else diffused_scores
        self.N = len(diffused_scores)
        self.min_score_threshold = min_score_threshold
        
        self.similarity_method = similarity_method
        self.lambda_sim = lambda_sim
        self.sigma_score = sigma_score
        self.tau_temporal = tau_temporal
        self.overlap_radius_multiplier = overlap_radius_multiplier
        
        self.valid_mask = self.original_scores >= min_score_threshold
        self.num_valid = np.sum(self.valid_mask)
        
        if base_suppression_radius is None:
            self.base_suppression_radius = max(1, self.N // 64)
        else:
            self.base_suppression_radius = base_suppression_radius
    
    def _calculate_similarity_cosine(self, score_i: float, score_j: float) -> float:
        score_diff = abs(score_i - score_j)
        similarity = 1.0 - score_diff
        return max(0.0, similarity)
    
    def _calculate_similarity_gaussian(self, score_i: float, score_j: float, 
                                       idx_i: int, idx_j: int) -> float:
        score_diff_sq = (score_i - score_j) ** 2
        score_term = np.exp(-score_diff_sq / (2 * self.sigma_score ** 2))
        
        temporal_dist = abs(idx_i - idx_j)
        temporal_term = np.exp(-temporal_dist / self.tau_temporal)
        
        similarity = score_term * temporal_term
        return similarity
    
    def _calculate_dynamic_radius(self, selected_idx: int, 
                                   neighbor_indices: np.ndarray) -> float:
        if len(neighbor_indices) == 0:
            return self.base_suppression_radius
        
        score_i = self.diffused_scores[selected_idx]
        scores_j = self.diffused_scores[neighbor_indices]
        
        if self.similarity_method == 'cosine':
            similarities = np.array([
                self._calculate_similarity_cosine(score_i, score_j)
                for score_j in scores_j
            ])
        else:
            similarities = np.array([
                self._calculate_similarity_gaussian(score_i, scores_j[k], 
                                                    selected_idx, neighbor_indices[k])
                for k in range(len(neighbor_indices))
            ])
        
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
        dynamic_radius = self.base_suppression_radius * (1 + self.lambda_sim * max_similarity)
        
        return dynamic_radius
    
    def _check_overlap_and_adjust_radius(self, selected_idx: int, 
                                         suppressed_set: Set[int]) -> float:
        start_check = max(0, selected_idx - int(self.base_suppression_radius))
        end_check = min(self.N, selected_idx + int(self.base_suppression_radius) + 1)
        
        neighbor_indices = [i for i in range(start_check, end_check) 
                           if i != selected_idx]
        
        overlap_detected = False
        for idx in neighbor_indices:
            if idx in suppressed_set or self.diffused_scores[idx] <= 0:
                overlap_detected = True
                break
        
        valid_neighbors = [i for i in neighbor_indices if i not in suppressed_set]
        dynamic_radius = self._calculate_dynamic_radius(selected_idx, 
                                                        np.array(valid_neighbors))
        
        if overlap_detected:
            dynamic_radius *= self.overlap_radius_multiplier
        
        return dynamic_radius
    
    def select_keyframes(self, max_frames: int, optimize_remaining: bool = False) -> List[int]:
        if self.N == 0 or self.num_valid == 0:
            return []
        
        if self.num_valid <= max_frames:
            valid_indices = np.where(self.valid_mask)[0]
            selected_frame_ids = [int(self.frame_ids[idx]) for idx in valid_indices]
            return sorted(list(set(selected_frame_ids)))
        
        selected_indices = self._greedy_selection_with_dynamic_suppression(max_frames)
        
        if optimize_remaining and len(selected_indices) < max_frames:
            selected_indices = self._fill_remaining_slots(selected_indices, max_frames)
        
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        unique_frame_ids = sorted(list(set(selected_frame_ids)))
        
        if len(unique_frame_ids) > max_frames:
            unique_frame_ids = unique_frame_ids[:max_frames]
        
        return unique_frame_ids
    
    def _greedy_selection_with_dynamic_suppression(self, max_frames: int) -> List[int]:
        valid_indices = np.where(self.valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        candidates = [
            (-self.diffused_scores[idx], idx) 
            for idx in valid_indices
        ]
        heapq.heapify(candidates)
        
        selected_indices = []
        selected_indices_set = set()
        suppressed = set()
        
        while len(selected_indices) < max_frames and candidates:
            neg_score, idx = heapq.heappop(candidates)
            
            if idx in selected_indices_set or idx in suppressed:
                continue
            
            dynamic_radius = self._check_overlap_and_adjust_radius(idx, suppressed)
            
            selected_indices.append(idx)
            selected_indices_set.add(idx)
            
            start_idx = max(0, idx - int(dynamic_radius))
            end_idx = min(self.N, idx + int(dynamic_radius) + 1)
            
            for i in range(start_idx, end_idx):
                if i != idx and i not in selected_indices_set:
                    suppressed.add(i)
        
        return selected_indices
    
    def _fill_remaining_slots(self, selected_indices: List[int], 
                             max_frames: int) -> List[int]:
        selected_set = set(selected_indices)
        remaining_slots = max_frames - len(selected_indices)
        
        if remaining_slots <= 0:
            return selected_indices
        
        valid_indices = np.where(self.valid_mask)[0]
        available_indices = [idx for idx in valid_indices if idx not in selected_set]
        
        if not available_indices:
            return selected_indices
        
        available_scores = self.diffused_scores[available_indices]
        num_to_take = min(remaining_slots, len(available_indices))
        top_k_local_indices = np.argsort(available_scores)[-num_to_take:][::-1]
        top_k_indices = [available_indices[i] for i in top_k_local_indices]
        
        selected_indices.extend(top_k_indices)
        
        return selected_indices


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, adaptive_params: Dict, args) -> List[int]:
    """Process a single video using DBFP with DURATION-ADAPTIVE parameters."""
    scores = np.asarray(scores, dtype=np.float64)
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    
    if args.ratio > 1:
        indices = np.arange(0, len(scores), args.ratio)
        scores = scores[indices]
        frame_ids = frame_ids[indices]
    
    if len(scores) <= max_frames:
        unique_frames = sorted(list(set([int(x) for x in frame_ids])))
        return unique_frames[:max_frames]
    
    graph = DiffusionGraph(
        scores=scores,
        frame_ids=frame_ids,
        alpha=args.alpha,
        edge_weight_type=args.edge_weight_type
    )
    
    original_normalized_scores = graph.scores.copy()
    
    diffusion_iters = args.diffusion_iterations
    if diffusion_iters is None:
        diffusion_iters = max(1, int(np.log2(len(scores))))
    
    diffused_scores = graph.diffuse(iterations=diffusion_iters)
    
    selector = KeyframeSelector(
        diffused_scores=diffused_scores,
        frame_ids=frame_ids,
        original_scores=original_normalized_scores,
        base_suppression_radius=adaptive_params['suppression_radius'],
        min_score_threshold=args.min_score_threshold,
        similarity_method=args.similarity_method,
        lambda_sim=adaptive_params['lambda_sim'],
        sigma_score=adaptive_params['sigma_score'],
        tau_temporal=adaptive_params['tau_temporal'],
        overlap_radius_multiplier=args.overlap_radius_multiplier
    )
    
    selected_frames = selector.select_keyframes(
        max_frames=max_frames,
        optimize_remaining=args.optimize_remaining
    )
    
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames detected!"
    assert len(selected_frames) <= max_frames, f"Too many frames: {len(selected_frames)} > {max_frames}"
    
    return selected_frames


def main(args):
    """Main function with hash-based filenames and CSV configuration tracking."""
    print("=" * 80)
    print("DBFP: Diffusion-Based Frame Propagation")
    print("WITH DURATION-ADAPTIVE HYPERPARAMETERS & HASH-BASED FILENAMES")
    print("=" * 80)
    
    # Generate configuration hash and descriptive name
    config_hash = generate_config_hash(args)
    descriptive_name = get_descriptive_name(args)
    
    print(f"\n🔑 Configuration Hash: {config_hash}")
    print(f"📝 Descriptive Name: {descriptive_name}")
    print(f"\nDataset: {args.dataset_name}")
    print(f"Feature Model: {args.extract_feature_model}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Alpha (decay): {args.alpha}")
    print(f"Edge Weight Type: {args.edge_weight_type}")
    print(f"Similarity Method: {args.similarity_method.upper()}")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading scores from: {args.score_path}")
    with open(args.score_path) as f:
        all_scores = json.load(f)
    
    print(f"Loading frames from: {args.frame_path}")
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    print(f"Loading metadata from: {args.metadata_path}")
    try:
        with open(args.metadata_path) as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Metadata file not found at {args.metadata_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in metadata file")
        return
    
    if len(metadata) != len(all_scores):
        print(f"⚠️  Warning: Metadata length ({len(metadata)}) != Video count ({len(all_scores)})")
    
    num_videos_to_process = min(len(all_scores), len(metadata))
    if args.num_videos is not None:
        num_videos_to_process = min(args.num_videos, num_videos_to_process)
        print(f"\n🎯 DEMO MODE: Processing first {num_videos_to_process} videos only")
    
    print(f"Total videos loaded: {len(all_scores)}")
    print(f"Videos to process: {num_videos_to_process}\n")
    
    # Process each video
    selected_frames_all = []
    filtered_count = 0
    optimized_count = 0
    duplicate_warnings = 0
    param_usage = {}
    
    for idx in range(num_videos_to_process):
        scores = all_scores[idx]
        frame_ids = all_frame_ids[idx]
        metadata_entry = metadata[idx]
        
        adaptive_params = get_adaptive_parameters(metadata_entry, args, args.dataset_name)
        
        param_key = (
            f"r{adaptive_params['suppression_radius']:.1f}_"
            f"l{adaptive_params['lambda_sim']:.1f}_"
            f"s{adaptive_params['sigma_score']:.2f}_"
            f"t{adaptive_params['tau_temporal']:.1f}"
        )
        param_usage[param_key] = param_usage.get(param_key, 0) + 1
        
        if (idx + 1) % 100 == 0 or num_videos_to_process <= 20:
            print(f"Processing video {idx + 1}/{num_videos_to_process}...")
        
        try:
            selected_frames = process_video(
                scores=scores,
                frame_ids=frame_ids,
                max_frames=args.max_num_frames,
                adaptive_params=adaptive_params,
                args=args
            )
            
            if len(selected_frames) != len(set(selected_frames)):
                print(f"  ⚠️  Warning: Video {idx + 1} has duplicate frames!")
                duplicate_warnings += 1
                selected_frames = sorted(list(set(selected_frames)))[:args.max_num_frames]
            
            selected_frames_all.append(selected_frames)
            
            if len(selected_frames) < args.max_num_frames:
                filtered_count += 1
            if args.optimize_remaining and len(selected_frames) == args.max_num_frames:
                optimized_count += 1
                
        except Exception as e:
            print(f"  ❌ Error processing video {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            selected_frames_all.append([])
    
    # Save results with hash-based filename
    output_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # SHORT FILENAME using hash
    output_filename = f"selected_{config_hash}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    # Save configuration mapping to CSV
    save_config_mapping(config_hash, descriptive_name, args, output_dir, output_filename)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Processing complete!")
    print(f"Selected frames saved to: {output_path}")
    print(f"Configuration hash: {config_hash}")
    
    # Statistics
    frame_counts = [len(frames) for frames in selected_frames_all if len(frames) > 0]
    if frame_counts:
        print(f"\n📈 Statistics:")
        print(f"  Videos processed: {len(selected_frames_all)}")
        print(f"  Videos with frames: {len(frame_counts)}")
        print(f"  Avg frames selected: {np.mean(frame_counts):.2f}")
        print(f"  Min frames: {np.min(frame_counts)}")
        print(f"  Max frames: {np.max(frame_counts)}")
        
        print(f"\n📊 Adaptive Parameter Combinations Used:")
        for params, count in sorted(param_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_videos_to_process) * 100
            print(f"  {params}: {count} videos ({percentage:.1f}%)")
        
        if args.min_score_threshold > 0:
            print(f"\n🔍 Threshold Filtering:")
            print(f"  Videos with < max frames: {filtered_count}")
        
        if args.optimize_remaining:
            print(f"\n⚡ Optimization:")
            print(f"  Videos optimized to max frames: {optimized_count}")
        
        if duplicate_warnings > 0:
            print(f"\n⚠️  Warnings:")
            print(f"  Videos with duplicate frames: {duplicate_warnings}")
    
    print("=" * 80)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)