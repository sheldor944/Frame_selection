import heapq
import json
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict

def parse_arguments():
    parser = argparse.ArgumentParser(description='DBFP: Diffusion-Based Frame Propagation')
    
    parser.add_argument('--dataset_name', type=str, default='longvideobench', 
                        help='Dataset name: longvideobench or videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip', 
                        help='Feature extraction model: blip/clip/sevila')
    parser.add_argument('--score_path', type=str, 
                        default='./outscores/longvideobench/blip/scores.json',
                        help='Path to input scores JSON file')
    parser.add_argument('--frame_path', type=str, 
                        default='./outscores/longvideobench/blip/frames.json',
                        help='Path to input frame IDs JSON file')
    parser.add_argument('--max_num_frames', type=int, default=16,
                        help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1,
                        help='Sampling ratio for initial frame selection')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Diffusion decay factor (0-1): controls original vs neighbor influence')
    parser.add_argument('--diffusion_iterations', type=int, default=None,
                        help='Number of diffusion iterations (default: log2(N))')
    parser.add_argument('--suppression_radius', type=float, default=3,
                        help='Temporal suppression radius (default: N/M)')
    parser.add_argument('--edge_weight_type', type=str, default='score_diff',
                        choices=['uniform', 'score_diff', 'temporal'],
                        help='Edge weight type for diffusion')
    parser.add_argument('--output_file', type=str, default='./selected_frames',
                        help='Output directory for selected frames')
    parser.add_argument('--num_videos', type=int, default=None,
                    help='Number of videos to process (default: all)')
    
    return parser.parse_args()


class DiffusionGraph:
    """
    Represents a temporal graph of video frames with diffusion capabilities.
    """
    
    def __init__(self, scores: np.ndarray, frame_ids: List[int], 
                 alpha: float = 0.7, edge_weight_type: str = 'uniform'):
        """
        Initialize the diffusion graph.
        
        Args:
            scores: Array of frame relevance scores
            frame_ids: List of frame IDs corresponding to scores
            alpha: Diffusion decay factor
            edge_weight_type: Type of edge weighting ('uniform', 'score_diff', 'temporal')
        """
        self.scores = np.array(scores, dtype=np.float64)
        self.frame_ids = frame_ids
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
        """
        Build edge weights between adjacent frames.
        
        Returns:
            Array of shape (N-1,) containing edge weights
        """
        if self.N <= 1:
            return np.array([])
        
        if self.edge_weight_type == 'uniform':
            # All edges have equal weight
            return np.ones(self.N - 1)
        
        elif self.edge_weight_type == 'score_diff':
            # Weight inversely proportional to score difference (similar frames = stronger connection)
            score_diffs = np.abs(np.diff(self.scores))
            # Avoid division by zero, add small epsilon
            weights = 1.0 / (score_diffs + 1e-6)
            # Normalize
            weights = weights / weights.max()
            return weights
        
        elif self.edge_weight_type == 'temporal':
            # Weight based on temporal gap between frames
            temporal_gaps = np.diff(self.frame_ids)
            # Closer frames = stronger connection
            weights = 1.0 / (temporal_gaps + 1.0)
            # Normalize
            weights = weights / weights.max()
            return weights
        
        else:
            return np.ones(self.N - 1)
    
    def diffuse(self, iterations: int = None) -> np.ndarray:
        """
        Perform diffusion process on the graph.
        
        Args:
            iterations: Number of diffusion iterations (default: log2(N))
        
        Returns:
            Diffused scores array
        """
        if self.N <= 1:
            return self.diffused_scores
        
        if iterations is None:
            iterations = max(1, int(np.log2(self.N)))
        
        # Perform iterative diffusion
        for _ in range(iterations):
            new_scores = self.diffused_scores.copy()
            
            for t in range(self.N):
                # Original score contribution
                original_contrib = self.alpha * self.diffused_scores[t]
                
                # Neighbor contributions
                neighbor_scores = []
                neighbor_weights = []
                
                # Left neighbor
                if t > 0:
                    neighbor_scores.append(self.diffused_scores[t - 1])
                    neighbor_weights.append(self.edge_weights[t - 1])
                
                # Right neighbor
                if t < self.N - 1:
                    neighbor_scores.append(self.diffused_scores[t + 1])
                    neighbor_weights.append(self.edge_weights[t])
                
                # Weighted average of neighbors
                if neighbor_scores:
                    neighbor_weights = np.array(neighbor_weights)
                    neighbor_weights = neighbor_weights / neighbor_weights.sum()
                    neighbor_contrib = (1 - self.alpha) * np.average(
                        neighbor_scores, weights=neighbor_weights
                    )
                else:
                    neighbor_contrib = 0
                
                new_scores[t] = original_contrib + neighbor_contrib
            
            self.diffused_scores = new_scores
        
        return self.diffused_scores


class KeyframeSelector:
    """
    Selects keyframes using greedy NMS-like strategy on diffused scores.
    """
    
    def __init__(self, diffused_scores: np.ndarray, frame_ids: List[int],
                 suppression_radius: float = None):
        """
        Initialize keyframe selector.
        
        Args:
            diffused_scores: Diffused relevance scores
            frame_ids: Frame IDs corresponding to scores
            suppression_radius: Temporal suppression radius
        """
        self.diffused_scores = diffused_scores
        self.frame_ids = np.array(frame_ids)
        self.N = len(diffused_scores)
        
        # Set suppression radius
        if suppression_radius is None:
            # Default: spread selections across timeline
            self.suppression_radius = max(1, self.N // 64)
        else:
            self.suppression_radius = suppression_radius
    
    def select_keyframes(self, max_frames: int) -> List[int]:
        """
        Select keyframes using greedy NMS strategy.
        
        Args:
            max_frames: Maximum number of frames to select
        
        Returns:
            List of selected frame IDs
        """
        if self.N == 0:
            return []
        
        if self.N <= max_frames:
            return self.frame_ids.tolist()
        
        # Create priority queue (max heap using negative scores)
        candidates = [(-score, idx) for idx, score in enumerate(self.diffused_scores)]
        heapq.heapify(candidates)
        
        selected_indices = []
        suppressed = set()
        
        while len(selected_indices) < max_frames and candidates:
            # Get highest scoring candidate
            neg_score, idx = heapq.heappop(candidates)
            
            # Skip if suppressed
            if idx in suppressed:
                continue
            
            # Select this frame
            selected_indices.append(idx)
            
            # Suppress nearby frames
            for i in range(self.N):
                if abs(i - idx) <= self.suppression_radius and i != idx:
                    suppressed.add(i)
        
        # Convert indices to frame IDs and sort temporally
        # selected_frame_ids = [self.frame_ids[idx] for idx in selected_indices]
        # selected_frame_ids.sort()
        

        selected_frame_ids = [int(self.frame_ids[idx]) for idx in selected_indices]
        selected_frame_ids.sort()


        return selected_frame_ids


def process_video(scores: List[float], frame_ids: List[int],
                  max_frames: int, args) -> List[int]:
    """
    Process a single video using DBFP.
    
    Args:
        scores: List of frame scores
        frame_ids: List of frame IDs
        max_frames: Maximum frames to select
        args: Argument namespace with DBFP parameters
    
    Returns:
        List of selected frame IDs
    """
    # Apply ratio-based downsampling if needed
    if args.ratio > 1:
        nums = int(len(scores) / args.ratio)
        scores = [scores[num * args.ratio] for num in range(nums)]
        frame_ids = [frame_ids[num * args.ratio] for num in range(nums)]
    
    # Handle short videos
    if len(scores) <= max_frames:
        return frame_ids
    
    # Create diffusion graph
    graph = DiffusionGraph(
        scores=scores,
        frame_ids=frame_ids,
        alpha=args.alpha,
        edge_weight_type=args.edge_weight_type
    )
    
    # Perform diffusion
    diffusion_iters = args.diffusion_iterations
    if diffusion_iters is None:
        diffusion_iters = max(1, int(np.log2(len(scores))))
    
    diffused_scores = graph.diffuse(iterations=diffusion_iters)
    
    # Select keyframes
    selector = KeyframeSelector(
        diffused_scores=diffused_scores,
        frame_ids=frame_ids,
        suppression_radius=args.suppression_radius
    )
    
    selected_frames = selector.select_keyframes(max_frames)
    
    return selected_frames


def build_output_filename(args):
    name = (
        f"selected_dbfp_"
        f"{args.dataset_name}_"
        f"{args.extract_feature_model}_"
        f"k{args.max_num_frames}_"
        f"alpha{args.alpha}_"
        f"sup{args.suppression_radius}_"
        f"{args.edge_weight_type}.json"
    )
    return name




def main(args):
    """
    Main function to process all videos using DBFP.
    """
    print("=" * 60)
    print("DBFP: Diffusion-Based Frame Propagation")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Feature Model: {args.extract_feature_model}")
    print(f"Max Frames: {args.max_num_frames}")
    print(f"Alpha (decay): {args.alpha}")
    print(f"Edge Weight Type: {args.edge_weight_type}")
    print(f"Diffusion Iterations: {args.diffusion_iterations if args.diffusion_iterations else 'auto (log2(N))'}")
    print(f"Suppression Radius: {args.suppression_radius if args.suppression_radius else 'auto (N/M)'}")
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
            args=args
        )
        
        selected_frames_all.append(selected_frames)
    
    # Save results
    output_dir = os.path.join(args.output_file, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_model_dir = os.path.join(output_dir, args.extract_feature_model)
    os.makedirs(output_model_dir, exist_ok=True)

    output_path = os.path.join(args.output_file, build_output_filename(args))

    # output_path = os.path.join(output_model_dir, 'selected_frames_dbfp1_20_alpha_0.75_frames_temporal.json')

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