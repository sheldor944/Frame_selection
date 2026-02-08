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
import time
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ========================================
# GLOBAL VIDEO PATH CONFIGURATION
# ========================================
PATH_LONGVIDEOBENCH = "/path/to/your/longvideobench/videos"
PATH_VIDEOMME = "/home/train01/.cache/huggingface/custom_video_qa_cache/data/data"
# ========================================


print("hello miraj ")

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
                        help='Number of neighboring seconds to also dense sample (default: 2)')
    
    parser.add_argument('--dense_fps', type=float,
                        default=8.0,
                        help='FPS for dense sampling in high-score regions (default: 8.0)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, 
                        default='blip',
                        choices=['clip', 'blip'],
                        help='Model to use for relevance scoring (default: blip)')
    
    parser.add_argument('--model_name', type=str, 
                        default=None,
                        help='Specific model name')
    
    # OPTIMIZED: Larger batch size for speed
    parser.add_argument('--batch_size', type=int, 
                        default=64,
                        help='Batch size for processing frames (default: 64)')
    
    parser.add_argument('--device', type=str, 
                        default='cuda:1',
                        help='Device to use (default: cuda:1)')
    
    # OPTIMIZED: Enable FP16 by default
    parser.add_argument('--use_fp16', action='store_true',
                        default=True,
                        help='Use FP16 mixed precision (default: True)')
    
    # NEW: DataLoader workers
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='Number of DataLoader workers (default: 4)')
    
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


# ========================================
# OPTIMIZED: Fast Dataset for DataLoader
# ========================================
class FrameDataset(Dataset):
    """Fast dataset for batch processing frames with DataLoader"""
    def __init__(self, frames: List[np.ndarray], processor, model_type: str):
        self.frames = frames
        self.processor = processor
        self.model_type = model_type
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        pil_image = Image.fromarray(frame)
        
        # Preprocess on CPU (parallel across workers)
        if self.model_type == 'clip':
            inputs = self.processor(images=pil_image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        elif self.model_type == 'blip':
            inputs = self.processor(images=pil_image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)


# ========================================
# OPTIMIZED: Ultra-Fast Dense Sampler
# ========================================
class DenseSampler:
    """ULTRA-FAST frame extraction with large chunks and parallel processing"""
    
    def __init__(self, dense_fps: float = 5.0, save_frames: bool = False,
                 output_dir: str = './dense_frames'):
        self.dense_fps = dense_fps
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_dense_frames(self, video_path: str, video_id: str,
                         high_score_frame_indices: List[int], 
                         neighbor_radius: int,
                         video_fps: float) -> Tuple[List[np.ndarray], List[int]]:
        """
        ULTRA-FAST extraction using Decord with optimized settings
        Memory usage: 1.2GB → Safe to use large chunks
        """
        if not os.path.exists(video_path):
            return [], []
        
        try:
            # OPTIMIZED: More threads for faster decoding
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
        except Exception as e:
            print(f"⚠ Failed to load video: {video_path}")
            return [], []
        
        num_frames = len(vr)
        dense_interval = max(1, int(video_fps / self.dense_fps))
        
        # Build frame indices set
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
        
        frames_to_extract = sorted(frames_to_extract)
        
        if len(frames_to_extract) == 0:
            return [], []
        
        # ========== OPTIMIZED: HUGE chunks (1.2GB VRAM → tons of headroom) ==========
        CHUNK_SIZE = 20000  # Load 500 frames at once (much larger!)
        
        frames = []
        frame_numbers = []
        
        if self.save_frames:
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(0, len(frames_to_extract), CHUNK_SIZE):
            chunk_indices = frames_to_extract[i:i + CHUNK_SIZE]
            
            try:
                # Fast batch loading
                batch = vr.get_batch(chunk_indices).asnumpy()
                
                # Collect frames
                for j, frame_idx in enumerate(chunk_indices):
                    frame_rgb = batch[j]
                    frames.append(frame_rgb)
                    frame_numbers.append(frame_idx)
                
                # Parallel frame saving (if enabled)
                if self.save_frames:
                    self._save_frames_parallel(batch, chunk_indices, video_output_dir)
                
                del batch
                
            except Exception as e:
                print(f"⚠ Chunk {i} failed: {e}")
                continue
        
        return frames, frame_numbers
    
    def _save_frames_parallel(self, batch: np.ndarray, indices: List[int], output_dir: Path):
        """Save frames using thread pool for parallel I/O"""
        def save_single(args):
            frame_rgb, frame_idx = args
            frame_path = output_dir / f"dense_frame_{frame_idx:06d}.jpg"
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(save_single, zip(batch, indices))


# ========================================
# OPTIMIZED: Ultra-Fast Relevance Scorer
# ========================================
class RelevanceScorer:
    """ULTRA-FAST relevance scoring with DataLoader and all optimizations"""
    
    def __init__(self, model_type: str = 'clip', model_name: str = None,
                 device: str = 'cuda:0', batch_size: int = 64, 
                 use_fp16: bool = True, num_workers: int = 4):
        self.model_type = model_type
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.device = device
        self.num_workers = num_workers
        
        print(f"\n{'='*70}")
        print(f"[{self.device}] Loading {model_type} model...")
        print(f"{'='*70}")
        
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
        
        # ========== ENABLE ALL CUDA OPTIMIZATIONS ==========
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f"✅ Model loaded successfully")
        print(f"   - Batch size: {batch_size}")
        print(f"   - FP16: {use_fp16}")
        print(f"   - Workers: {num_workers}")
        print(f"   - CUDA optimizations: Enabled")
        print(f"{'='*70}\n")
    
    @torch.inference_mode()
    def compute_scores(self, frames: List[np.ndarray], query: str) -> List[float]:
        """FAST: Compute scores using optimized methods"""
        if len(frames) == 0:
            return []
        
        if self.model_type == 'clip':
            return self._compute_clip_scores_fast(frames, query)
        elif self.model_type == 'blip':
            return self._compute_blip_scores_fast(frames, query)
    
    def _compute_clip_scores_fast(self, frames: List[np.ndarray], query: str) -> List[float]:
        """OPTIMIZED CLIP with DataLoader + Prefetching"""
        
        # Encode text once (reuse for all frames)
        text_inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        del text_inputs
        
        # Create DataLoader for parallel preprocessing
        dataset = FrameDataset(frames, self.processor, 'clip')
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        all_scores = []
        
        for batch_pixels in dataloader:
            batch_pixels = batch_pixels.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                image_features = self.model.get_image_features(pixel_values=batch_pixels)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).squeeze(-1)
            
            all_scores.extend(similarities.cpu().tolist())
            
            del batch_pixels, image_features, similarities
        
        return all_scores
    
    def _compute_blip_scores_fast(self, frames: List[np.ndarray], query: str) -> List[float]:
        """OPTIMIZED BLIP with larger batches"""
        
        all_scores = []
        
        # BLIP can handle larger batches (you have plenty of VRAM)
        effective_batch = min(self.batch_size, 512)  # Cap at 32 for BLIP
        
        # Convert all frames to PIL once
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        for i in range(0, len(pil_images), effective_batch):
            batch_images = pil_images[i:i + effective_batch]
            
            inputs = self.processor(
                images=batch_images,
                text=[query] * len(batch_images),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.itm_score, dim=1)[:, 1]
            
            all_scores.extend(scores.cpu().tolist())
            
            del inputs, outputs, scores
        
        del pil_images
        return all_scores

# ========================================
# Helper Functions
# ========================================
def find_video_file(video_folder: Path, video_id: str) -> str:
    """Find video file by video_id"""
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    for ext in extensions:
        video_path = video_folder / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    return None


def get_video_fps(video_path: str) -> float:
    """Get video FPS using Decord"""
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
        min_len = min(len(frames), len(scores))
        frames = frames[:min_len]
        scores = scores[:min_len]
    
    # Calculate threshold as percentage of MAX score
    max_score = max(scores)
    threshold_value = (threshold_percent / 100.0) * max_score
    
    # Find high-score frames
    high_score_frame_indices = []
    for i, score in enumerate(scores):
        if score >= threshold_value:
            high_score_frame_indices.append(int(frames[i]))
    
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


def get_gpu_memory():
    """Return current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def identify_high_score_frames_verbose(frames: List[float], scores: List[float],
                                       threshold_percent: float, video_id: str,
                                       pbar) -> List[int]:
    """
    Identify frames with scores >= threshold_percent of MAX score
    WITH CLEAN SINGLE-BLOCK PRINTING
    
    Args:
        frames: List of frame indices
        scores: List of scores
        threshold_percent: Percentage of max score (e.g., 70 = 70% of max)
        video_id: Video identifier for logging
        pbar: tqdm progress bar to write to
    
    Returns:
        List of HIGH-SCORE frame indices
    """
    if len(scores) == 0 or len(frames) == 0:
        return []
    
    # Ensure frames and scores have same length
    if len(frames) != len(scores):
        min_len = min(len(frames), len(scores))
        frames = frames[:min_len]
        scores = scores[:min_len]
    
    # Calculate threshold as percentage of MAX score
    max_score = max(scores)
    min_score = min(scores)
    threshold_value = (threshold_percent / 100.0) * max_score
    
    # Find high-score frames
    high_score_frame_indices = []
    for i, score in enumerate(scores):
        if score >= threshold_value:
            high_score_frame_indices.append(int(frames[i]))
    
    # Build the entire message as a single string
    message = (
        f"\n{'─'*70}\n"
        f"📹 Video: {video_id}\n"
        f"{'─'*70}\n"
        f"  📊 Scores: Max={max_score:.4f} | Min={min_score:.4f} | Range=[{min_score:.4f}, {max_score:.4f}]\n"
        f"  🎯 Threshold: {threshold_value:.4f} ({threshold_percent}% of max)\n"
        f"  ✅ High-Score Frames: {len(high_score_frame_indices)}/{len(frames)} ({100*len(high_score_frame_indices)/len(frames):.1f}%)\n"
    )
    
    if len(high_score_frame_indices) > 0:
        message += f"  📍 First 10 indices: {high_score_frame_indices[:10]}\n"
    
    message += f"{'─'*70}\n"
    
    # Print everything at once
    tqdm.write(message)
    
    return high_score_frame_indices


# ========================================
# OPTIMIZED Main Processing Function
# ========================================
def process_dense_sampling(args):
    """OPTIMIZED: Main processing function with detailed progress tracking"""
    
    print("\n" + "="*70)
    print("⚡ ULTRA-FAST DENSE SAMPLING")
    print("="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Score threshold: {args.score_threshold}% of MAX score")
    print(f"Neighbor radius: {args.neighbor_radius} seconds")
    print(f"Dense FPS: {args.dense_fps}")
    print(f"Model: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.use_fp16}")
    print(f"Workers: {args.num_workers}")
    print("="*70 + "\n")
    
    # Load existing frames and scores
    print("Loading existing frames and scores...")
    with open(args.input_frames, 'r') as f:
        all_original_frames = json.load(f)
    
    with open(args.input_scores, 'r') as f:
        all_original_scores = json.load(f)
    
    print(f"✅ Frames file: {len(all_original_frames)} videos")
    print(f"✅ Scores file: {len(all_original_scores)} videos")
    
    # Use the minimum length to avoid errors
    if len(all_original_frames) != len(all_original_scores):
        print(f"\n⚠️  WARNING: Frames and scores have different number of videos!")
        min_len = min(len(all_original_frames), len(all_original_scores))
        all_original_frames = all_original_frames[:min_len]
        all_original_scores = all_original_scores[:min_len]
    
    # Load questions/metadata
    if args.json_file is None:
        args.json_file = os.path.join(args.dataset_path, args.dataset_name, 'include_frame_idx.json')
    
    print(f"Loading questions from: {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} questions")
    
    # Align data length
    min_len = min(len(all_original_frames), len(all_original_scores), len(data))
    all_original_frames = all_original_frames[:min_len]
    all_original_scores = all_original_scores[:min_len]
    data = data[:min_len]
    
    # Apply num_videos filter
    if args.num_videos:
        num_to_process = min(args.num_videos, len(data))
        data = data[:num_to_process]
        all_original_frames = all_original_frames[:num_to_process]
        all_original_scores = all_original_scores[:num_to_process]
        print(f"\n🎯 Processing {num_to_process} videos (limited by --num_videos)\n")
    
    # Setup video folder
    if args.video_folder is None:
        if args.dataset_name == 'longvideobench':
            args.video_folder = PATH_LONGVIDEOBENCH
        elif args.dataset_name == 'videomme':
            args.video_folder = PATH_VIDEOMME
    
    video_folder = Path(args.video_folder)
    print(f"Video folder: {video_folder}\n")
    
    # Initialize components with optimized settings
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
        use_fp16=args.use_fp16,
        num_workers=args.num_workers
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
    
    print("="*70)
    print("Starting video processing...")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Create progress bar with more details
    pbar = tqdm(range(len(data)), desc="Processing videos", ncols=120)
    
    for idx in pbar:
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
            
            # Update progress bar
            pbar.set_postfix({
                'video': video_id[:20],
                'status': 'SKIPPED',
                'frames': 0
            })
            continue
        
        # Get video FPS
        video_fps = get_video_fps(video_path)
        
        # Identify high-score frames (WITH DETAILED PRINTING)
        high_score_frame_indices = identify_high_score_frames_verbose(
            orig_frames, 
            orig_scores, 
            args.score_threshold,
            video_id,
            pbar  # Pass progress bar to update it
        )
        
        if len(high_score_frame_indices) == 0:
            # No high-score frames, keep original
            all_merged_frames.append(orig_frames)
            all_merged_scores.append(orig_scores)
            
            pbar.set_postfix({
                'video': video_id[:20],
                'status': 'NO_HIGH_SCORES',
                'frames': len(orig_frames)
            })
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
            
            pbar.set_postfix({
                'video': video_id[:20],
                'status': 'EXTRACT_FAILED',
                'frames': len(orig_frames)
            })
            continue
        
        # Compute scores for new frames (FAST with DataLoader)
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
        
        # Update progress bar with rich info
        pbar.set_postfix({
            'video': video_id[:20],
            'orig': len(orig_frames),
            'new': len(new_frame_numbers),
            'final': len(merged_frames),
            'dup': duplicates
        })
        
        # Cleanup
        del new_frames_data, new_scores
        
        # Monitor GPU memory every 50 videos
        if (idx + 1) % 50 == 0:
            alloc, res = get_gpu_memory()
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"📊 Progress Update [{idx+1}/{len(data)}]")
            tqdm.write(f"{'='*70}")
            tqdm.write(f"  GPU Memory: {alloc:.2f}GB allocated, {res:.2f}GB reserved")
            tqdm.write(f"  Videos processed: {idx+1}")
            tqdm.write(f"  With dense sampling: {stats['videos_with_dense_sampling']}")
            tqdm.write(f"  Total frames: {stats['total_merged_frames']:,}")
            elapsed = time.time() - start_time
            tqdm.write(f"  Elapsed time: {elapsed/60:.2f} min")
            tqdm.write(f"  Speed: {(idx+1)/elapsed:.2f} videos/sec")
            tqdm.write(f"{'='*70}\n")
    
    pbar.close()
    elapsed_time = time.time() - start_time
    
    # Clear GPU cache
    if 'cuda' in args.device:
        torch.cuda.empty_cache()
    
    # Save results
    output_dir = Path(args.output_dir)
    dataset_output_dir = output_dir / args.dataset_name / args.model_type
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_path = dataset_output_dir / f'frames_dense_r{args.neighbor_radius}_f{int(args.dense_fps)}_ultrafast.json'
    scores_path = dataset_output_dir / f'scores_dense_r{args.neighbor_radius}_f{int(args.dense_fps)}_ultrafast.json'
    
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}")
    
    with open(frames_path, 'w') as f:
        json.dump(all_merged_frames, f, indent=2)
    
    with open(scores_path, 'w') as f:
        json.dump(all_merged_scores, f, indent=2)
    
    print(f"✅ Frames saved to: {frames_path}")
    print(f"✅ Scores saved to: {scores_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 PROCESSING STATISTICS")
    print("="*70)
    print(f"⏱️  Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    print(f"⚡ Speed: {len(data)/elapsed_time:.2f} videos/sec")
    print(f"\nVideos:")
    print(f"  Total processed: {stats['total_videos']}")
    print(f"  With dense sampling: {stats['videos_with_dense_sampling']}")
    print(f"  Skipped: {stats['videos_skipped']}")
    
    print(f"\nFrames:")
    print(f"  Original frames: {stats['total_original_frames']:,}")
    print(f"  New frames extracted: {stats['total_new_frames']:,}")
    print(f"  Duplicates removed: {stats['total_duplicates_removed']:,}")
    print(f"  Total after merge: {stats['total_merged_frames']:,}")
    
    if stats['total_videos'] > 0:
        print(f"\nAverages per video:")
        print(f"  Original frames: {stats['total_original_frames'] / stats['total_videos']:.2f}")
        print(f"  Final frames: {stats['total_merged_frames'] / stats['total_videos']:.2f}")
        print(f"  New frames added: {(stats['total_merged_frames'] - stats['total_original_frames']) / stats['total_videos']:.2f}")
        print(f"  Processing time: {elapsed_time / stats['total_videos']:.3f}s")
    
    # Final GPU memory
    alloc, res = get_gpu_memory()
    print(f"\nFinal GPU Memory:")
    print(f"  Allocated: {alloc:.2f}GB")
    print(f"  Reserved: {res:.2f}GB")
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