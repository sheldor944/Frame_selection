import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Set

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP: Diffusion-Based Frame Propagation')
    
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
    parser.add_argument('--diffusion_iterations', type=int, default=1,
                        help='Number of diffusion iterations (default: log2(N))')
    
    # Adaptive suppression radius arguments for LongVideoBench
    parser.add_argument('--suppression_radius_15', type=float, default=2.0,
                        help='Suppression radius for duration_group=15 (LongVideoBench)')
    parser.add_argument('--suppression_radius_60', type=float, default=3.0,
                        help='Suppression radius for duration_group=60 (LongVideoBench)')
    parser.add_argument('--suppression_radius_600', type=float, default=5.0,
                        help='Suppression radius for duration_group=600 (LongVideoBench)')
    parser.add_argument('--suppression_radius_3600', type=float, default=8.0,
                        help='Suppression radius for duration_group=3600 (LongVideoBench)')
    
    # Adaptive suppression radius arguments for VideoMME
    parser.add_argument('--suppression_radius_short', type=float, default=2.0,
                        help='Suppression radius for duration=short (VideoMME)')
    parser.add_argument('--suppression_radius_medium', type=float, default=3.0,
                        help='Suppression radius for duration=medium (VideoMME)')
    parser.add_argument('--suppression_radius_long', type=float, default=5.0,
                        help='Suppression radius for duration=long (VideoMME)')
    
    parser.add_argument('--edge_weight_type', type=str, default='temporal',
                        choices=['uniform', 'score_diff', 'temporal'],
                        help='Edge weight type for diffusion')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory for selected frames')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Number of videos to process (default: all)')
    parser.add_argument('--min_score_threshold', type=float, default=0,
                        help='Minimum normalized score threshold (0-1). Frames below this are excluded.')
    parser.add_argument('--no_optimize_remaining', dest='optimize_remaining', 
                        action='store_false', default=False,
                        help='If set, disables filling remaining slots (optimization is ON by default)')

    
    return parser.parse_args()


def get_suppression_radius(metadata_entry: dict, args, dataset_name: str) -> float:
    """
    Get adaptive suppression radius based on video duration category.
    
    Args:
        metadata_entry: Dictionary containing video metadata
        args: Argument namespace with suppression radius parameters
        dataset_name: Name of the dataset ('longvideobench' or 'videomme')
    
    Returns:
        Suppression radius for this video
    """
    if dataset_name == 'longvideobench':
        # LongVideoBench uses 'duration_group' field with values [15, 60, 600, 3600]
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
            print(f"  ⚠️  Warning: Unknown duration_group '{duration_group}', using default radius 3.0")
            return 3.0
            
    elif dataset_name == 'videomme':
        # VideoMME uses 'duration' field with values ['short', 'medium', 'long']
        duration = metadata_entry.get('duration', None)
        
        if duration == 'short':
            return args.suppression_radius_short
        elif duration == 'medium':
            return args.suppression_radius_medium
        elif duration == 'long':
            return args.suppression_radius_long
        else:
            print(f"  ⚠️  Warning: Unknown duration '{duration}', using default radius 3.0")
            return 3.0
    else:
        print(f"  ⚠️  Warning: Unknown dataset '{dataset_name}', using default radius 3.0")
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
            edge_weight_type: Type of edge weighting ('uniform', 'score_diff', 'temporal')
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
    Selects keyframes using greedy NMS-like strategy on diffused scores.
    Fixed version with proper uniqueness guarantees.
    """
    
    def __init__(self, diffused_scores: np.ndarray, frame_ids: np.ndarray,
                 original_scores: np.ndarray = None,
                 suppression_radius: float = None,
                 min_score_threshold: float = 0.0):
        """
        Initialize keyframe selector.
        
        Args:
            diffused_scores: Diffused relevance scores
            frame_ids: Frame IDs corresponding to scores
            original_scores: Original (normalized) scores before diffusion
            suppression_radius: Temporal suppression radius
            min_score_threshold: Minimum score threshold (0-1)
        """
        self.diffused_scores = np.asarray(diffused_scores, dtype=np.float64)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int32)
        self.original_scores = original_scores if original_scores is not None else diffused_scores
        self.N = len(diffused_scores)
        self.min_score_threshold = min_score_threshold
        
        # STEP 1: Filter out frames below threshold
        self.valid_mask = self.original_scores >= min_score_threshold
        self.num_valid = np.sum(self.valid_mask)
        
        # Set suppression radius
        if suppression_radius is None:
            self.suppression_radius = max(1, self.N // 64)
        else:
            self.suppression_radius = suppression_radius
    
    def select_keyframes(self, max_frames: int, optimize_remaining: bool = False) -> List[int]:
        """
        Select keyframes with proper uniqueness guarantees.
        
        Two-step process:
        1. Select frames with suppression (only from frames above threshold)
        2. If optimize_remaining=True, fill remaining slots with best frames (no suppression, no duplicates)
        
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
            # Ensure uniqueness and sort
            return sorted(list(set(selected_frame_ids)))
        
        # Phase 1: Greedy selection with suppression (only from valid frames)
        selected_indices = self._greedy_selection_with_suppression(max_frames)
        
        # Phase 2: If optimize_remaining and we haven't reached max_frames
        if optimize_remaining and len(selected_indices) < max_frames:
            selected_indices = self._fill_remaining_slots(
                selected_indices, max_frames
            )
        
        # Convert to frame IDs, ensure uniqueness, and sort
        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        unique_frame_ids = sorted(list(set(selected_frame_ids)))
        
        # Sanity check: ensure we don't exceed max_frames
        if len(unique_frame_ids) > max_frames:
            unique_frame_ids = unique_frame_ids[:max_frames]
        
        return unique_frame_ids
    
    def _greedy_selection_with_suppression(self, max_frames: int) -> List[int]:
        """
        STEP 1: Perform greedy selection with suppression.
        Only considers frames above threshold.
        
        Args:
            max_frames: Maximum number of frames to select
        
        Returns:
            List of selected indices (guaranteed unique)
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
        selected_indices_set = set()  # For O(1) duplicate checking
        suppressed = set()
        
        while len(selected_indices) < max_frames and candidates:
            # Get highest scoring candidate
            neg_score, idx = heapq.heappop(candidates)
            
            # Skip if already selected or suppressed
            if idx in selected_indices_set or idx in suppressed:
                continue
            
            # Select this frame
            selected_indices.append(idx)
            selected_indices_set.add(idx)
            
            # Suppress nearby frames
            start_idx = max(0, idx - int(self.suppression_radius))
            end_idx = min(self.N, idx + int(self.suppression_radius) + 1)
            
            for i in range(start_idx, end_idx):
                if i != idx and i not in selected_indices_set:
                    suppressed.add(i)
        
        return selected_indices
    
    def _fill_remaining_slots(self, selected_indices: List[int], 
                             max_frames: int) -> List[int]:
        """
        STEP 2: Fill remaining slots with best available frames.
        No suppression applied, but ensures no duplicates.
        Only considers frames above threshold.
        
        Args:
            selected_indices: Already selected frame indices
            max_frames: Maximum total frames to select
        
        Returns:
            Updated list of selected indices (guaranteed unique)
        """
        selected_set = set(selected_indices)
        remaining_slots = max_frames - len(selected_indices)
        
        if remaining_slots <= 0:
            return selected_indices
        
        # Get all valid frames (above threshold) not yet selected
        valid_indices = np.where(self.valid_mask)[0]
        available_indices = [idx for idx in valid_indices if idx not in selected_set]
        
        if not available_indices:
            return selected_indices
        
        # Get scores for available frames
        available_scores = self.diffused_scores[available_indices]
        
        # Sort by score (descending) and take top remaining_slots
        # Ensure we don't take more than available
        num_to_take = min(remaining_slots, len(available_indices))
        top_k_local_indices = np.argsort(available_scores)[-num_to_take:][::-1]
        top_k_indices = [available_indices[i] for i in top_k_local_indices]
        
        # Add to selected (these are guaranteed unique because we excluded selected_set)
        selected_indices.extend(top_k_indices)
        
        return selected_indices


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, suppression_radius: float, args) -> List[int]:
    """
    Process a single video using DBFP.
    
    Args:
        scores: List of frame scores
        frame_ids: List of frame IDs
        max_frames: Maximum frames to select
        suppression_radius: Suppression radius for this specific video
        args: Argument namespace with DBFP parameters
    
    Returns:
        List of unique selected frame IDs (sorted)
    """
    # Convert to numpy arrays for efficiency
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
        return unique_frames[:max_frames]  # In case of duplicates
    
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
    
    # Select keyframes with adaptive suppression radius
    selector = KeyframeSelector(
        diffused_scores=diffused_scores,
        frame_ids=frame_ids,
        original_scores=original_normalized_scores,
        suppression_radius=suppression_radius,  # Use adaptive radius
        min_score_threshold=args.min_score_threshold
    )
    
    selected_frames = selector.select_keyframes(
        max_frames=max_frames,
        optimize_remaining=args.optimize_remaining
    )
    
    # Final sanity check
    assert len(selected_frames) == len(set(selected_frames)), "Duplicate frames detected!"
    assert len(selected_frames) <= max_frames, f"Too many frames selected: {len(selected_frames)} > {max_frames}"
    
    return selected_frames

def build_output_filename(args):
    """Build output filename based on parameters."""
    
    # Build suppression radius config string
    if args.dataset_name == 'longvideobench':
        sup_config = (
            f"r15_{args.suppression_radius_15}_"
            f"r60_{args.suppression_radius_60}_"
            f"r600_{args.suppression_radius_600}_"
            f"r3600_{args.suppression_radius_3600}"
        )
    elif args.dataset_name == 'videomme':
        sup_config = (
            f"short_{args.suppression_radius_short}_"
            f"med_{args.suppression_radius_medium}_"
            f"long_{args.suppression_radius_long}"
        )
    else:
        sup_config = "default"
    
    name = (
        f"selected_dbfp_"
        f"{args.dataset_name}_"
        f"{args.extract_feature_model}_"
        f"k{args.max_num_frames}_"
        f"alpha{args.alpha}_"
        f"adaptive_{sup_config}_"
        f"{args.edge_weight_type}"
    )
    
    # Add diffusion iterations if specified
    if args.diffusion_iterations is not None:
        name += f"_iter{args.diffusion_iterations}"
    
    if args.min_score_threshold > 0:
        name += f"_minscore{args.min_score_threshold}"
    
    if args.optimize_remaining:
        name += "_optimized"
    
    name += ".json"
    return name


def main(args):
    """Main function to process all videos using DBFP with adaptive suppression."""
    print("=" * 60)
    print("DBFP: Diffusion-Based Frame Propagation (Adaptive Suppression)")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Feature Model: {args.extract_feature_model}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Alpha (decay): {args.alpha}")
    print(f"Edge Weight Type: {args.edge_weight_type}")
    print(f"Diffusion Iterations: {args.diffusion_iterations if args.diffusion_iterations else 'auto (log2(N))'}")
    
    # Print adaptive suppression radius configuration
    print(f"\n📊 Adaptive Suppression Radius Configuration:")
    if args.dataset_name == 'longvideobench':
        print(f"  Duration Group 15s    → Radius: {args.suppression_radius_15}")
        print(f"  Duration Group 60s    → Radius: {args.suppression_radius_60}")
        print(f"  Duration Group 600s   → Radius: {args.suppression_radius_600}")
        print(f"  Duration Group 3600s  → Radius: {args.suppression_radius_3600}")
    elif args.dataset_name == 'videomme':
        print(f"  Short videos   → Radius: {args.suppression_radius_short}")
        print(f"  Medium videos  → Radius: {args.suppression_radius_medium}")
        print(f"  Long videos    → Radius: {args.suppression_radius_long}")
    
    print(f"\nMin Score Threshold: {args.min_score_threshold}")
    print(f"Optimize Remaining: {'✅ ENABLED' if args.optimize_remaining else '❌ DISABLED'}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading scores from: {args.score_path}")
    with open(args.score_path) as f:
        all_scores = json.load(f)
    
    print(f"Loading frames from: {args.frame_path}")
    with open(args.frame_path) as f:
        all_frame_ids = json.load(f)
    
    # Load metadata for duration categories
    print(f"Loading metadata from: {args.metadata_path}")
    try:
        with open(args.metadata_path) as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Metadata file not found at {args.metadata_path}")
        print(f"Please provide a valid metadata file with duration_group (LongVideoBench) or duration (VideoMME) fields")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in metadata file")
        return
    
    # Validate metadata length matches video count
    if len(metadata) != len(all_scores):
        print(f"⚠️  Warning: Metadata length ({len(metadata)}) != Video count ({len(all_scores)})")
        print(f"Using min length for safety")
    
    # Determine how many videos to process
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
    
    # Track suppression radius usage statistics
    radius_usage = {}
    
    for idx in range(num_videos_to_process):
        scores = all_scores[idx]
        frame_ids = all_frame_ids[idx]
        metadata_entry = metadata[idx]
        
        # Get adaptive suppression radius for this video
        suppression_radius = get_suppression_radius(metadata_entry, args, args.dataset_name)
        
        # Track radius usage
        radius_key = f"{suppression_radius:.1f}"
        radius_usage[radius_key] = radius_usage.get(radius_key, 0) + 1
        
        # Show progress
        if num_videos_to_process <= 20:
            # Show duration info for small batches
            if args.dataset_name == 'longvideobench':
                duration_info = metadata_entry.get('duration_group', 'unknown')
            else:
                duration_info = metadata_entry.get('duration', 'unknown')
            print(f"Processing video {idx + 1}/{num_videos_to_process} "
                  f"[Duration: {duration_info}, Radius: {suppression_radius}]...")
        elif (idx + 1) % 100 == 0:
            print(f"Processing video {idx + 1}/{num_videos_to_process}...")
        
        try:
            selected_frames = process_video(
                scores=scores,
                frame_ids=frame_ids,
                max_frames=args.max_num_frames,
                suppression_radius=suppression_radius,  # Pass adaptive radius
                args=args
            )
            
            # Verify uniqueness
            if len(selected_frames) != len(set(selected_frames)):
                print(f"  ⚠️  Warning: Video {idx + 1} has duplicate frames!")
                duplicate_warnings += 1
                selected_frames = sorted(list(set(selected_frames)))[:args.max_num_frames]
            
            selected_frames_all.append(selected_frames)
            
            # Track statistics
            if len(selected_frames) < args.max_num_frames:
                filtered_count += 1
            if args.optimize_remaining and len(selected_frames) == args.max_num_frames:
                optimized_count += 1
                
        except Exception as e:
            print(f"  ❌ Error processing video {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Add empty list to maintain index alignment
            selected_frames_all.append([])
    
    # Save results
    output_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_file, build_output_filename(args))
    
    with open(output_path, 'w') as f:
        json.dump(selected_frames_all, f)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete!")
    print(f"Selected frames saved to: {output_path}")
    
    # Statistics
    frame_counts = [len(frames) for frames in selected_frames_all if len(frames) > 0]
    if frame_counts:
        print(f"\n📈 Statistics:")
        print(f"  Videos processed: {len(selected_frames_all)}")
        print(f"  Videos with frames: {len(frame_counts)}")
        print(f"  Avg frames selected: {np.mean(frame_counts):.2f}")
        print(f"  Min frames: {np.min(frame_counts)}")
        print(f"  Max frames: {np.max(frame_counts)}")
        
        # Show suppression radius usage
        print(f"\n📊 Suppression Radius Usage:")
        for radius, count in sorted(radius_usage.items(), key=lambda x: float(x[0])):
            percentage = (count / num_videos_to_process) * 100
            print(f"  Radius {radius}: {count} videos ({percentage:.1f}%)")
        
        if args.min_score_threshold > 0:
            print(f"\n🔍 Threshold Filtering:")
            print(f"    Videos with < max frames: {filtered_count}")
            print(f"    (due to threshold {args.min_score_threshold})")
        
        if args.optimize_remaining:
            print(f"\n⚡ Optimization:")
            print(f"    Videos optimized to max frames: {optimized_count}")
        
        if duplicate_warnings > 0:
            print(f"\n⚠️  Warnings:")
            print(f"    Videos with duplicate frames: {duplicate_warnings}")
    
    print("=" * 60)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)