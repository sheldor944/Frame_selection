import json
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
import warnings
warnings.filterwarnings('ignore')

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ========================================
# GLOBAL VIDEO PATH CONFIGURATION
# ========================================
PATH_LONGVIDEOBENCH = "/path/to/your/longvideobench/videos"
PATH_VIDEOMME = "/home/train01/.cache/huggingface/custom_video_qa_cache/data/data"
# ========================================


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dense sampling around high-score frames')
    
    # Dataset configuration
    parser.add_argument('--dataset_name', type=str, 
                        default='videomme',
                        choices=['longvideobench', 'videomme'],
                        help='Dataset name (default: videomme)')
    
    parser.add_argument('--dataset_path', type=str,
                        default='./datasets',
                        help='Base path to datasets folder (default: ./datasets)')
    
    parser.add_argument('--video_folder', type=str, 
                        default=None,
                        help='Path to folder containing videos')
    
    parser.add_argument('--json_file', type=str, 
                        default=None,
                        help='Path to JSON file with questions')
    
    # Input files (existing frames and scores)
    parser.add_argument('--input_frames', type=str,
                        default='./outscores/videomme/blip/frames.json',
                        help='Path to existing frames.json file')
    
    parser.add_argument('--input_scores', type=str,
                        default='./outscores/videomme/blip/scores.json',
                        help='Path to existing scores.json file')
    
    # Output files
    parser.add_argument('--output_dir', type=str, 
                        default='./output_dense_sampling',
                        help='Output directory for new frames and scores (default: ./output_dense_sampling)')
    
    # Dense sampling parameters
    parser.add_argument('--score_threshold', type=float,
                        default=70.0,
                        help='Score threshold as percentage of MAX score (e.g., 70 = 70%% of max) (default: 70)')
    
    parser.add_argument('--neighbor_radius', type=int,
                        default=2,
                        help='Number of neighboring seconds to also dense sample (default: 1)')
    
    parser.add_argument('--dense_fps', type=float,
                        default=8.0,
                        help='FPS for dense sampling in high-score regions (default: 5.0)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, 
                        default='blip',
                        choices=['clip', 'blip'],
                        help='Model to use for relevance scoring (default: blip)')
    
    parser.add_argument('--model_name', type=str, 
                        default=None,
                        help='Specific model name')
    
    parser.add_argument('--batch_size', type=int, 
                        default=32,
                        help='Batch size for processing frames (default: 256)')
    
    parser.add_argument('--device', type=str, 
                        default='cuda:1',
                        help='Device to use (default: cuda:0)')
    
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 mixed precision')
    
    # Optional
    parser.add_argument('--save_frames', action='store_true',
                        help='Save extracted frames to disk')
    
    parser.add_argument('--frames_output_dir', type=str, 
                        default='./dense_sampled_frames',
                        help='Directory to save frames')
    
    parser.add_argument('--num_videos', type=int,
                        default=2700,
                        help='Limit number of videos to process')
    
    return parser.parse_args()


class DenseSampler:
    """Extract additional frames around high-score regions"""
    
    def __init__(self, dense_fps: float = 5.0, save_frames: bool = False,
                 output_dir: str = './dense_frames'):
        self.dense_fps = dense_fps
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # def extract_dense_frames(self, video_path: str, video_id: str,
    #                         high_score_frame_indices: List[int], 
    #                         neighbor_radius: int,
    #                         video_fps: float) -> Tuple[List[np.ndarray], List[int]]:
    #     """
    #     Extract frames at dense_fps around specified frame indices
        
    #     Args:
    #         video_path: Path to video file
    #         video_id: Video identifier
    #         high_score_frame_indices: List of HIGH-SCORE frame indices (keep these)
    #         neighbor_radius: Number of seconds around each high-score frame
    #         video_fps: Original video FPS
        
    #     Returns:
    #         frames: List of frame arrays
    #         frame_numbers: List of frame indices
    #     """
    #     if not os.path.exists(video_path):
    #         return [], []
        
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         return [], []
        
    #     # Calculate dense sampling interval
    #     dense_interval = max(1, int(video_fps / self.dense_fps))
        
    #     # Determine all frames to extract
    #     frames_to_extract = set()
        
    #     # ALWAYS keep the original high-score frames
    #     for frame_idx in high_score_frame_indices:
    #         frames_to_extract.add(frame_idx)
        
    #     # Add neighbors at dense_fps
    #     if neighbor_radius > 0:
    #         neighbor_range_frames = int(video_fps * neighbor_radius)
            
    #         for center_frame in high_score_frame_indices:
    #             # Dense sample from (center - radius) to (center + radius)
    #             start = max(0, center_frame - neighbor_range_frames)
    #             end = center_frame + neighbor_range_frames
                
    #             # Sample at dense_fps intervals
    #             for frame_idx in range(start, end + 1, dense_interval):
    #                 frames_to_extract.add(frame_idx)
        
    #     # Sort frame indices
    #     frames_to_extract = sorted(frames_to_extract)
        
    #     frames = []
    #     frame_numbers = []
        
    #     if self.save_frames:
    #         video_output_dir = self.output_dir / video_id
    #         video_output_dir.mkdir(parents=True, exist_ok=True)
        
    #     # Extract frames
    #     for frame_idx in frames_to_extract:
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #         ret, frame = cap.read()
            
    #         if not ret:
    #             continue
            
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frames.append(frame_rgb)
    #         frame_numbers.append(frame_idx)
            
    #         if self.save_frames:
    #             frame_path = video_output_dir / f"dense_frame_{frame_idx:06d}.jpg"
    #             cv2.imwrite(str(frame_path), frame)
        
    #     cap.release()
    #     return frames, frame_numbers
    def extract_dense_frames(self, video_path: str, video_id: str,
                         high_score_frame_indices: List[int], 
                         neighbor_radius: int,
                         video_fps: float) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames at dense_fps around specified frame indices using Decord (FAST)

        Args:
            video_path: Path to video file
            video_id: Video identifier
            high_score_frame_indices: List of HIGH-SCORE frame indices (keep these)
            neighbor_radius: Number of seconds around each high-score frame
            video_fps: Original video FPS

        Returns:
            frames: List of frame arrays (RGB, np.ndarray)
            frame_numbers: List of frame indices
        """

        # Verify path exists
        if not os.path.exists(video_path):
            return [], []

        # Initialize Decord VideoReader (MUCH faster than cv2)
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"⚠ Failed to load video with Decord: {video_path}")
            return [], []

        num_frames = len(vr)

        # Calculate dense sampling interval
        dense_interval = max(1, int(video_fps / self.dense_fps))

        # Determine which frames to extract
        frames_to_extract = set()

        # Always keep high-score frames
        for frame_idx in high_score_frame_indices:
            if frame_idx < num_frames:
                frames_to_extract.add(frame_idx)

        # Add dense neighbors
        if neighbor_radius > 0:
            neighbor_range_frames = int(video_fps * neighbor_radius)

            for center_frame in high_score_frame_indices:
                start = max(0, center_frame - neighbor_range_frames)
                end = min(num_frames - 1, center_frame + neighbor_range_frames)

                for frame_idx in range(start, end + 1, dense_interval):
                    if frame_idx < num_frames:
                        frames_to_extract.add(frame_idx)

        # Sort final list
        frames_to_extract = sorted(frames_to_extract)

        if len(frames_to_extract) == 0:
            return [], []

        # ====== FAST BATCH LOADING ======
        try:
            batch = vr.get_batch(frames_to_extract)   # Decord tensor batch
            batch = batch.asnumpy()                   # convert to numpy RGB
        except Exception as e:
            print(f"⚠ Decord batch load failed for {video_id}: {e}")
            return [], []

        frames = []
        frame_numbers = []

        # Prepare directory for saved frames
        if self.save_frames:
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect frames
        for i, frame_idx in enumerate(frames_to_extract):
            frame_rgb = batch[i]      # RGB image as np.ndarray
            frames.append(frame_rgb)
            frame_numbers.append(frame_idx)

            if self.save_frames:
                frame_path = video_output_dir / f"dense_frame_{frame_idx:06d}.jpg"

                # Convert RGB → BGR for OpenCV save
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), bgr)

        return frames, frame_numbers

class RelevanceScorer:
    """Compute relevance scores between frames and text query"""
    
    def __init__(self, model_type: str = 'clip', model_name: str = None,
                 device: str = 'cuda:0', batch_size: int = 256, use_fp16: bool = False):
        self.model_type = model_type
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.device = device
        
        print(f"[{self.device}] Loading {model_type} model...")
        
        if model_type == 'clip':
            self.model_name = model_name or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
        elif model_type == 'blip':
            self.model_name = model_name or "Salesforce/blip-itm-base-coco"
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
        
        self.model = self.model.to(self.device)
        
        if self.use_fp16 and 'cuda' in self.device:
            self.model = self.model.half()
        
        self.model.eval()
        print(f"[{self.device}] Model loaded\n")
    
    @torch.inference_mode()
    def compute_scores(self, frames: List[np.ndarray], query: str) -> List[float]:
        """Compute relevance scores for frames"""
        if len(frames) == 0:
            return []
        
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        if self.model_type == 'clip':
            scores = self._compute_clip_scores(pil_images, query)
        elif self.model_type == 'blip':
            # scores = self._compute_blip_scores(pil_images, query)
            scores = self._compute_blip_scores_batch(pil_images, query)
        
        return scores
    
    def _compute_clip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Compute CLIP similarity scores"""
        text_inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        if self.use_fp16:
            text_inputs = {k: v.half() if v.dtype == torch.float16 else v 
                          for k, v in text_inputs.items()}
        
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        all_scores = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            image_inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            )
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            if self.use_fp16:
                image_inputs = {k: v.half() if v.dtype == torch.float16 else v 
                               for k, v in image_inputs.items()}
            
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarities = (image_features @ text_features.T).squeeze(-1)
            
            if similarities.dim() == 0:
                all_scores.append(similarities.item())
            else:
                all_scores.extend(similarities.cpu().tolist())
        
        return all_scores
    
    def _compute_blip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Compute BLIP ITM scores"""
        all_scores = []
        
        for image in images:
            inputs = self.processor(images=image, text=query, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float16 else v 
                         for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            score = torch.softmax(outputs.itm_score, dim=1)[0, 1].item()
            all_scores.append(score)
        
        return all_scores
    
    # def _compute_blip_scores_batch(self, images, query):
    #     inputs = self.processor(
    #         images=images,
    #         text=[query] * len(images),
    #         return_tensors="pt",
    #         padding=True
    #     ).to(self.device)

    #     if self.use_fp16:
    #         for k in ["pixel_values", "input_ids", "attention_mask"]:
    #             if k in inputs and inputs[k].dtype == torch.float16:
    #                 inputs[k] = inputs[k].half()

    #     outputs = self.model(**inputs)
    #     scores = torch.softmax(outputs.itm_score, dim=1)[:, 1]
    #     return scores.cpu().tolist()
    def _compute_blip_scores_batch(self, images: List[Image.Image], query: str) -> List[float]:
        """Memory-efficient BLIP ITM scoring with small batches"""
        all_scores = []
        
        # REDUCE batch size for BLIP (it's memory-heavy)
        effective_batch_size = min(8, self.batch_size)  # Max 8 images at once
        
        for i in range(0, len(images), effective_batch_size):
            batch_images = images[i:i + effective_batch_size]
            
            inputs = self.processor(
                images=batch_images,
                text=[query] * len(batch_images),
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            if self.use_fp16:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()
            
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.itm_score, dim=1)[:, 1]
            all_scores.extend(scores.cpu().tolist())
            
            # CRITICAL: Clear memory after each batch
            del inputs, outputs, scores
            torch.cuda.empty_cache()
        
        return all_scores


def find_video_file(video_folder: Path, video_id: str) -> str:
    """Find video file by video_id"""
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    for ext in extensions:
        video_path = video_folder / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    return None


# def get_video_fps(video_path: str) -> float:
#     """Get video FPS"""
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return fps if fps > 0 else 30.0

def get_video_fps(video_path: str) -> float:
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return float(vr.get_avg_fps())
    except:
        return 30.0



def identify_high_score_frames(frames: List[float], scores: List[float],
                               threshold_percent: float) -> List[int]:
    """
    Identify frames with scores >= threshold_percent of MAX score
    
    Args:
        frames: List of frame indices
        scores: List of scores
        threshold_percent: Percentage of max score (e.g., 70 = 70% of max)
    
    Returns:
        List of HIGH-SCORE frame indices
    """
    if len(scores) == 0 or len(frames) == 0:
        return []
    
    # Ensure frames and scores have same length
    if len(frames) != len(scores):
        print(f"  ⚠️  Warning: frames ({len(frames)}) and scores ({len(scores)}) length mismatch")
        min_len = min(len(frames), len(scores))
        frames = frames[:min_len]
        scores = scores[:min_len]
    
    # Calculate threshold as percentage of MAX score
    max_score = max(scores)
    threshold_value = (threshold_percent / 100.0) * max_score
    
    print(f"  Max score: {max_score:.4f}")
    print(f"  Threshold ({threshold_percent}% of max): {threshold_value:.4f}")
    print(f"  Score range: [{min(scores):.4f}, {max_score:.4f}]")
    
    # Find high-score frames
    high_score_frame_indices = []
    for i, score in enumerate(scores):
        if score >= threshold_value:
            high_score_frame_indices.append(int(frames[i]))
    
    print(f"  Found {len(high_score_frame_indices)} high-score frames (>= {threshold_percent}% of max)")
    
    return high_score_frame_indices


def merge_frames_and_scores(original_frames: List[int], original_scores: List[float],
                            new_frames: List[int], new_scores: List[float]) -> Tuple[List[int], List[float]]:
    """
    Merge original and new frames/scores, removing duplicates
    
    Returns:
        merged_frames: Combined and sorted frame indices (NO DUPLICATES)
        merged_scores: Corresponding scores
    """
    # Create dictionary for quick lookup
    frame_score_dict = {}
    
    # Add original frames
    for frame, score in zip(original_frames, original_scores):
        frame_score_dict[int(frame)] = score
    
    # Add/update with new frames (new scores take precedence for duplicates)
    for frame, score in zip(new_frames, new_scores):
        frame_score_dict[int(frame)] = score
    
    # Sort by frame index
    sorted_items = sorted(frame_score_dict.items())
    
    merged_frames = [item[0] for item in sorted_items]
    merged_scores = [item[1] for item in sorted_items]
    
    return merged_frames, merged_scores

def process_dense_sampling(args):
    """Main processing function for dense sampling"""
    
    print("\n" + "="*70)
    print("DENSE SAMPLING AROUND HIGH-SCORE FRAMES")
    print("="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Score threshold: {args.score_threshold}% of MAX score")
    print(f"Neighbor radius: {args.neighbor_radius} seconds")
    print(f"Dense FPS: {args.dense_fps}")
    print(f"Model: {args.model_type}")
    print("="*70 + "\n")
    
    # Load existing frames and scores
    print("Loading existing frames and scores...")
    with open(args.input_frames, 'r') as f:
        all_original_frames = json.load(f)
    
    with open(args.input_scores, 'r') as f:
        all_original_scores = json.load(f)
    
    print(f"Frames file: {len(all_original_frames)} videos")
    print(f"Scores file: {len(all_original_scores)} videos")
    
    # Use the minimum length to avoid errors
    if len(all_original_frames) != len(all_original_scores):
        print(f"\n⚠️  WARNING: Frames and scores have different number of videos!")
        min_len = min(len(all_original_frames), len(all_original_scores))
        all_original_frames = all_original_frames[:min_len]
        all_original_scores = all_original_scores[:min_len]
    
    # Load questions/metadata
    if args.json_file is None:
        args.json_file = os.path.join(args.dataset_path, args.dataset_name, 'include_frame_idx.json')
    
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Align data length
    if len(data) > len(all_original_frames):
        data = data[:len(all_original_frames)]
    
    # Apply num_videos filter
    if args.num_videos:
        num_to_process = min(args.num_videos, len(data))
        data = data[:num_to_process]
        all_original_frames = all_original_frames[:num_to_process]
        all_original_scores = all_original_scores[:num_to_process]
        print(f"Processing {num_to_process} videos (limited by --num_videos)\n")
    
    # Setup video folder
    if args.video_folder is None:
        if args.dataset_name == 'longvideobench':
            args.video_folder = PATH_LONGVIDEOBENCH
        elif args.dataset_name == 'videomme':
            args.video_folder = PATH_VIDEOMME
    
    video_folder = Path(args.video_folder)
    
    # Initialize components
    sampler = DenseSampler(
        dense_fps=args.dense_fps,
        save_frames=args.save_frames,
        output_dir=args.frames_output_dir
    )
    
    scorer = RelevanceScorer(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16
    )
    
    # Process each video
    all_merged_frames = []
    all_merged_scores = []
    
    stats = {
        'total_videos': len(data),
        'videos_with_dense_sampling': 0,
        'total_original_frames': 0,
        'total_new_frames': 0,
        'total_merged_frames': 0,
        'videos_skipped': 0,
        'total_duplicates_removed': 0
    }
    
    print("Processing videos...\n")
    
    for idx in tqdm(range(len(data)), desc="Dense sampling"):
        item = data[idx]
        orig_frames = all_original_frames[idx]
        orig_scores = all_original_scores[idx]
        
        video_id = item.get('videoID', item.get('video_id', f"{idx:03d}"))
        question = item.get('question', '')
        
        stats['total_original_frames'] += len(orig_frames)
        
        # Find video
        video_path = find_video_file(video_folder, video_id)
        if video_path is None or len(orig_frames) == 0:
            all_merged_frames.append(orig_frames)
            all_merged_scores.append(orig_scores)
            stats['videos_skipped'] += 1
            continue
        
        # Get video FPS
        video_fps = get_video_fps(video_path)
        
        # Identify high-score frames (frames with score >= threshold% of max)
        high_score_frame_indices = identify_high_score_frames(
            orig_frames, 
            orig_scores, 
            args.score_threshold
        )
        
        if len(high_score_frame_indices) == 0:
            # No high-score frames, keep original
            all_merged_frames.append(orig_frames)
            all_merged_scores.append(orig_scores)
            continue
        
        # Extract dense frames around high-score regions
        new_frames_data, new_frame_numbers = sampler.extract_dense_frames(
            video_path, 
            video_id, 
            high_score_frame_indices,
            args.neighbor_radius,
            video_fps
        )
        
        if len(new_frames_data) == 0:
            # Failed to extract, keep original
            all_merged_frames.append(orig_frames)
            all_merged_scores.append(orig_scores)
            continue
        
        # Compute scores for new frames
        new_scores = scorer.compute_scores(new_frames_data, question)
        
        # Count frames before merge
        total_before_merge = len(orig_frames) + len(new_frame_numbers)
        
        # Merge original and new frames/scores (removes duplicates)
        merged_frames, merged_scores = merge_frames_and_scores(
            orig_frames, 
            orig_scores,
            new_frame_numbers, 
            new_scores
        )
        
        # Count duplicates removed
        duplicates = total_before_merge - len(merged_frames)
        stats['total_duplicates_removed'] += duplicates
        
        all_merged_frames.append(merged_frames)
        all_merged_scores.append(merged_scores)
        
        # Update stats
        stats['videos_with_dense_sampling'] += 1
        stats['total_new_frames'] += len(new_frame_numbers)
        stats['total_merged_frames'] += len(merged_frames)
    
    # Clear GPU cache
    if 'cuda' in args.device:
        torch.cuda.empty_cache()
    
    # Save results
    output_dir = Path(args.output_dir)
    dataset_output_dir = output_dir / args.dataset_name / args.model_type
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_path = dataset_output_dir / 'frames_dense_r2_f8_.json'
    scores_path = dataset_output_dir / 'scores_dense_r2_f8_.json'
    
    print(f"\nSaving results...")
    with open(frames_path, 'w') as f:
        json.dump(all_merged_frames, f, indent=2)
    
    with open(scores_path, 'w') as f:
        json.dump(all_merged_scores, f, indent=2)
    
    print(f"✅ Dense frames saved to: {frames_path}")
    print(f"✅ Dense scores saved to: {scores_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("DENSE SAMPLING STATISTICS")
    print("="*70)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Videos with dense sampling: {stats['videos_with_dense_sampling']}")
    print(f"Videos skipped (no video/frames): {stats['videos_skipped']}")
    print(f"\nFrame Statistics:")
    print(f"  Original frames: {stats['total_original_frames']}")
    print(f"  New frames extracted: {stats['total_new_frames']}")
    print(f"  Duplicates removed: {stats['total_duplicates_removed']}")
    print(f"  Total frames after merge: {stats['total_merged_frames']}")
    
    if stats['total_videos'] > 0:
        print(f"\nAverages:")
        print(f"  Frames per video (original): {stats['total_original_frames'] / stats['total_videos']:.2f}")
        print(f"  Frames per video (after): {stats['total_merged_frames'] / stats['total_videos']:.2f}")
        print(f"  New frames added per video: {(stats['total_merged_frames'] - stats['total_original_frames']) / stats['total_videos']:.2f}")
        print(f"  Duplicates per video: {stats['total_duplicates_removed'] / stats['total_videos']:.2f}")
    
    print("="*70 + "\n")


def main():
    args = parse_arguments()
    
    # Auto-detect video folder if not provided
    if args.video_folder is None:
        if args.dataset_name == 'longvideobench':
            args.video_folder = PATH_LONGVIDEOBENCH
        elif args.dataset_name == 'videomme':
            args.video_folder = PATH_VIDEOMME
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Process dense sampling
    process_dense_sampling(args)


if __name__ == '__main__':
    main()