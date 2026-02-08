# import json
# import os
# import cv2
# import torch
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Tuple
# import argparse
# from tqdm import tqdm
# from PIL import Image
# import warnings
# import torch.multiprocessing as mp
# warnings.filterwarnings('ignore')

# # Import models
# from transformers import (
#     CLIPProcessor, CLIPModel,
#     BlipProcessor, BlipForImageTextRetrieval
# )

# # ========================================
# # GLOBAL VIDEO PATH CONFIGURATION
# # ========================================
# PATH_LONGVIDEOBENCH = "/path/to/your/longvideobench/videos"
# PATH_VIDEOMME = "/home/train01/.cache/huggingface/custom_video_qa_cache/data/data"
# # ========================================


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Extract frames and compute relevance scores')
    
#     # Dataset configuration
#     parser.add_argument('--dataset_name', type=str, 
#                         default='videomme',
#                         choices=['longvideobench', 'videomme'],
#                         help='Dataset name (default: videomme)')
    
#     parser.add_argument('--dataset_path', type=str,
#                         default='./datasets',
#                         help='Base path to datasets folder (default: ./datasets)')
    
#     parser.add_argument('--video_folder', type=str, 
#                         default=None,
#                         help='Path to folder containing videos (if None, uses global PATH variables)')
    
#     parser.add_argument('--json_file', type=str, 
#                         default=None,
#                         help='Path to JSON file with questions (if None, uses include_frame_idx.json)')
    
#     parser.add_argument('--output_dir', type=str, 
#                         default='./output_scores',
#                         help='Output directory for frames.json and scores.json (default: ./output_scores)')
    
#     # Processing parameters
#     parser.add_argument('--fps', type=float, 
#                         default=12.0,
#                         help='Frames per second to extract (default: 12.0)')
    
#     parser.add_argument('--model_type', type=str, 
#                         default='clip',
#                         choices=['clip', 'blip'],
#                         help='Model to use for relevance scoring (default: clip)')
    
#     parser.add_argument('--model_name', type=str, 
#                         default=None,
#                         help='Specific model name (default: auto-select based on model_type)')
    
#     parser.add_argument('--batch_size', type=int, 
#                         default=512,
#                         help='Batch size for processing frames (default: 16)')
    
#     parser.add_argument('--device', type=str, 
#                         default='cuda:0',
#                         help='Device to use: cuda:0, cuda:1, cpu (default: cuda:0)')
    
#     parser.add_argument('--gpu_id', type=int,
#                         default=0,
#                         help='GPU ID to use (0 or 1, default: 0)')
    
#     # Multi-GPU options
#     parser.add_argument('--use_both_gpus', action='store_true',
#                         help='Use both GPUs in parallel to process videos')
    
#     parser.add_argument('--num_gpus', type=int,
#                         default=2,
#                         help='Number of GPUs to use when --use_both_gpus is set (default: 2)')
    
#     # Optional features
#     parser.add_argument('--save_frames', action='store_true',
#                         help='Save extracted frames to disk')
    
#     parser.add_argument('--frames_output_dir', type=str, 
#                         default='./extracted_frames',
#                         help='Directory to save extracted frames if --save_frames is set')
    
#     parser.add_argument('--num_videos', type=int,
#                         default=None,
#                         help='Limit number of videos to process for testing (default: all)')
    
#     return parser.parse_args()


# class VideoFrameExtractor:
#     """Extract frames from video at specified FPS"""
    
#     def __init__(self, fps: float = 1.0, save_frames: bool = False, 
#                  output_dir: str = './extracted_frames'):
#         self.fps = fps
#         self.save_frames = save_frames
#         self.output_dir = Path(output_dir)
#         if self.save_frames:
#             self.output_dir.mkdir(parents=True, exist_ok=True)
    
#     def extract_frames(self, video_path: str, video_id: str) -> Tuple[List[np.ndarray], List[int]]:
#         """Extract frames from video at specified FPS."""
#         if not os.path.exists(video_path):
#             return [], []
        
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             return [], []
        
#         # Get video properties
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         if video_fps == 0:
#             cap.release()
#             return [], []
        
#         # Calculate frame interval
#         frame_interval = int(video_fps / self.fps)
#         frame_interval = max(1, frame_interval)
        
#         frames = []
#         frame_numbers = []
        
#         # Create output directory for this video if saving frames
#         if self.save_frames:
#             video_output_dir = self.output_dir / video_id
#             video_output_dir.mkdir(parents=True, exist_ok=True)
        
#         frame_count = 0
#         extracted_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Extract frame at specified interval
#             if frame_count % frame_interval == 0:
#                 # Convert BGR to RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(frame_rgb)
#                 frame_numbers.append(frame_count)
                
#                 # Save frame if requested
#                 if self.save_frames:
#                     frame_path = video_output_dir / f"frame_{frame_count:06d}.jpg"
#                     cv2.imwrite(str(frame_path), frame)
                
#                 extracted_count += 1
            
#             frame_count += 1
        
#         cap.release()
        
#         return frames, frame_numbers


# class RelevanceScorer:
#     """Compute relevance scores between frames and text query"""
    
#     def __init__(self, model_type: str = 'clip', model_name: str = None, 
#                  device: str = None, batch_size: int = 32):
#         self.model_type = model_type
#         self.batch_size = batch_size
        
#         if device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device
        
#         if model_type == 'clip':
#             self.model_name = model_name or "openai/clip-vit-base-patch32"
#             self.processor = CLIPProcessor.from_pretrained(self.model_name)
#             self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        
#         elif model_type == 'blip':
#             self.model_name = model_name or "Salesforce/blip-itm-base-coco"
#             self.processor = BlipProcessor.from_pretrained(self.model_name)
#             self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).to(self.device)
        
#         self.model.eval()
    
#     def compute_scores(self, frames: List[np.ndarray], query: str) -> List[float]:
#         """Compute relevance scores for frames given a text query."""
#         if len(frames) == 0:
#             return []
        
#         # Convert all frames to PIL Images first
#         pil_images = [Image.fromarray(frame) for frame in frames]
        
#         if self.model_type == 'clip':
#             scores = self._compute_clip_scores(pil_images, query)
#         elif self.model_type == 'blip':
#             scores = self._compute_blip_scores(pil_images, query)
#         else:
#             raise ValueError(f"Unknown model type: {self.model_type}")
        
#         return scores
    
#     def _compute_clip_scores(self, images: List[Image.Image], query: str) -> List[float]:
#         """Compute CLIP similarity scores"""
#         # Get text features once
#         text_inputs = self.processor(
#             text=query,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(self.device)
        
#         with torch.no_grad():
#             text_features = self.model.get_text_features(**text_inputs)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
#         scores = []
        
#         # Process images in batches
#         for i in range(0, len(images), self.batch_size):
#             batch_images = images[i:i + self.batch_size]
            
#             image_inputs = self.processor(
#                 images=batch_images,
#                 return_tensors="pt"
#             ).to(self.device)
            
#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**image_inputs)
#                 image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
#                 # Compute similarity for each image in batch
#                 batch_similarities = (image_features @ text_features.T).squeeze(-1)
                
#                 # Convert to list
#                 if batch_similarities.dim() == 0:
#                     scores.append(batch_similarities.item())
#                 else:
#                     scores.extend(batch_similarities.cpu().tolist())
        
#         return scores
    
#     def _compute_blip_scores(self, images: List[Image.Image], query: str) -> List[float]:
#         """Compute BLIP ITM scores"""
#         scores = []
        
#         for image in images:
#             inputs = self.processor(
#                 images=image,
#                 text=query,
#                 return_tensors="pt"
#             ).to(self.device)
            
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 itm_score = outputs.itm_score
#                 score = torch.softmax(itm_score, dim=1)[0, 1].item()
#                 scores.append(score)
        
#         return scores


# def find_video_file(video_folder: Path, video_id: str) -> str:
#     """Find video file by video_id, checking common extensions."""
#     extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
#     for ext in extensions:
#         video_path = video_folder / f"{video_id}{ext}"
#         if video_path.exists():
#             return str(video_path)
    
#     return None


# def process_video_subset(gpu_id: int, data_subset: List[Dict], args_dict: Dict):
#     """Process a subset of videos on a specific GPU"""
    
#     # Reconstruct args namespace
#     class Args:
#         pass
#     args = Args()
#     for key, value in args_dict.items():
#         setattr(args, key, value)
    
#     # Set device for this GPU
#     device = f'cuda:{gpu_id}'
    
#     print(f"\n{'='*60}")
#     print(f"GPU {gpu_id}: Starting to process {len(data_subset)} videos")
#     print(f"{'='*60}\n")
    
#     # Initialize components for this GPU
#     video_folder = Path(args.video_folder)
#     extractor = VideoFrameExtractor(
#         fps=args.fps,
#         save_frames=args.save_frames,
#         output_dir=args.frames_output_dir
#     )
    
#     scorer = RelevanceScorer(
#         model_type=args.model_type,
#         model_name=args.model_name,
#         device=device,
#         batch_size=args.batch_size
#     )
    
#     results = []
    
#     # Process each video with progress bar
#     pbar = tqdm(data_subset, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)
    
#     for idx, item in enumerate(pbar):
#         video_id = item.get('videoID', item.get('video_id', f"{idx:03d}"))
#         question = item.get('question', '')
        
#         # Update progress bar
#         pbar.set_postfix({'video': video_id[:20]})
        
#         # Find video file
#         video_path = find_video_file(video_folder, video_id)
        
#         if video_path is None:
#             results.append({
#                 'video_id': video_id,
#                 'frames': [],
#                 'scores': []
#             })
#             continue
        
#         # Extract frames
#         frames, frame_numbers = extractor.extract_frames(video_path, video_id)
        
#         if len(frames) == 0:
#             results.append({
#                 'video_id': video_id,
#                 'frames': [],
#                 'scores': []
#             })
#             continue
        
#         # Compute relevance scores
#         scores = scorer.compute_scores(frames, question)
        
#         # Store results
#         results.append({
#             'video_id': video_id,
#             'frames': frame_numbers,
#             'scores': scores
#         })
    
#     print(f"\nGPU {gpu_id}: Completed {len(data_subset)} videos")
    
#     return results


# def process_videos_parallel(args):
#     """Process videos using multiple GPUs in parallel"""
    
#     print("="*60)
#     print("Parallel Video Processing with Multiple GPUs")
#     print("="*60)
#     print(f"Dataset: {args.dataset_name}")
#     print(f"Video folder: {args.video_folder}")
#     print(f"JSON file: {args.json_file}")
#     print(f"Output dir: {args.output_dir}")
#     print(f"Number of GPUs: {args.num_gpus}")
#     print(f"FPS: {args.fps}")
#     print(f"Model: {args.model_type}")
#     print(f"Batch size: {args.batch_size}")
#     print("="*60)
    
#     # Validate paths
#     if not os.path.exists(args.video_folder):
#         raise FileNotFoundError(f"❌ Video folder not found: {args.video_folder}")
    
#     if not os.path.exists(args.json_file):
#         raise FileNotFoundError(f"❌ JSON file not found: {args.json_file}")
    
#     # Load data
#     print("\nLoading JSON data...")
#     with open(args.json_file, 'r') as f:
#         data = json.load(f)
    
#     if args.num_videos is not None:
#         data = data[:args.num_videos]
#         print(f"⚠️  Processing only first {args.num_videos} videos (test mode)")
    
#     print(f"Loaded {len(data)} questions")
    
#     # Split data for multiple GPUs
#     chunk_size = len(data) // args.num_gpus
#     data_splits = []
    
#     print(f"\nSplitting data across {args.num_gpus} GPUs:")
#     for i in range(args.num_gpus):
#         start_idx = i * chunk_size
#         if i == args.num_gpus - 1:
#             # Last GPU gets remaining videos
#             end_idx = len(data)
#         else:
#             end_idx = (i + 1) * chunk_size
        
#         data_splits.append(data[start_idx:end_idx])
#         print(f"  GPU {i}: {len(data_splits[i])} videos (indices {start_idx}-{end_idx-1})")
    
#     # Convert args to dict for multiprocessing
#     args_dict = vars(args)
    
#     # Set multiprocessing start method
#     try:
#         mp.set_start_method('spawn', force=True)
#     except RuntimeError:
#         pass  # Already set
    
#     print(f"\n{'='*60}")
#     print(f"Starting parallel processing...")
#     print(f"{'='*60}\n")
    
#     # Create processes for each GPU
#     with mp.Pool(processes=args.num_gpus) as pool:
#         results_list = pool.starmap(
#             process_video_subset,
#             [(gpu_id, data_splits[gpu_id], args_dict) for gpu_id in range(args.num_gpus)]
#         )
    
#     print(f"\n{'='*60}")
#     print("All GPUs finished! Combining results...")
#     print(f"{'='*60}\n")
    
#     # Combine results from all GPUs in original order
#     all_frames = []
#     all_scores = []
    
#     for results in results_list:
#         for result in results:
#             all_frames.append(result['frames'])
#             all_scores.append(result['scores'])
    
#     # Save results
#     output_dir = Path(args.output_dir)
#     dataset_output_dir = output_dir / args.dataset_name / args.model_type
#     dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
#     frames_path = dataset_output_dir / 'frames.json'
#     scores_path = dataset_output_dir / 'scores.json'
    
#     print("Saving results...")
#     with open(frames_path, 'w') as f:
#         json.dump(all_frames, f)
    
#     with open(scores_path, 'w') as f:
#         json.dump(all_scores, f)
    
#     print(f"✅ Frames saved to: {frames_path}")
#     print(f"✅ Scores saved to: {scores_path}")
    
#     # Print statistics
#     print("\n" + "="*60)
#     print("Final Statistics:")
#     print("="*60)
#     total_frames = sum(len(f) for f in all_frames)
#     videos_processed = sum(1 for f in all_frames if len(f) > 0)
#     print(f"Videos processed: {videos_processed}/{len(data)}")
#     print(f"Total frames extracted: {total_frames}")
#     if videos_processed > 0:
#         print(f"Average frames per video: {total_frames / videos_processed:.2f}")
#     print("="*60)


# def process_videos(args):
#     """Main processing function for single GPU (original code)"""
    
#     # Set device based on gpu_id
#     if args.device == 'cuda:0' and args.gpu_id != 0:
#         args.device = f'cuda:{args.gpu_id}'
    
#     print("=" * 60)
#     print("Video Frame Extraction and Relevance Scoring")
#     print("=" * 60)
#     print(f"Dataset: {args.dataset_name}")
#     print(f"Video folder: {args.video_folder}")
#     print(f"JSON file: {args.json_file}")
#     print(f"Output dir: {args.output_dir}")
#     print(f"FPS: {args.fps}")
#     print(f"Model: {args.model_type}")
#     print(f"Device: {args.device}")
#     print(f"Batch size: {args.batch_size}")
#     print("=" * 60)
    
#     # Validate paths
#     if not os.path.exists(args.video_folder):
#         raise FileNotFoundError(f"❌ Video folder not found: {args.video_folder}")
    
#     if not os.path.exists(args.json_file):
#         raise FileNotFoundError(f"❌ JSON file not found: {args.json_file}")
    
#     # Load JSON data
#     print("\nLoading JSON data...")
#     with open(args.json_file, 'r') as f:
#         data = json.load(f)
    
#     # Limit number of videos if specified
#     if args.num_videos is not None:
#         data = data[:args.num_videos]
#         print(f"⚠️  Processing only first {args.num_videos} videos (test mode)")
    
#     print(f"Loaded {len(data)} questions")

#     # Initialize components
#     video_folder = Path(args.video_folder)
#     extractor = VideoFrameExtractor(
#         fps=args.fps,
#         save_frames=args.save_frames,
#         output_dir=args.frames_output_dir
#     )
    
#     scorer = RelevanceScorer(
#         model_type=args.model_type,
#         model_name=args.model_name,
#         device=args.device,
#         batch_size=args.batch_size
#     )
    
#     # Storage for results
#     all_frames = []
#     all_scores = []
    
#     # Process each video/question
#     print("\nProcessing videos...")
#     for idx, item in enumerate(tqdm(data, desc="Processing")):
#         video_id = item.get('videoID', item.get('video_id', f"{idx:03d}"))
#         question = item.get('question', '')
        
#         # Find video file
#         video_path = find_video_file(video_folder, video_id)
        
#         if video_path is None:
#             all_frames.append([])
#             all_scores.append([])
#             continue
        
#         # Extract frames
#         frames, frame_numbers = extractor.extract_frames(video_path, video_id)
        
#         if len(frames) == 0:
#             all_frames.append([])
#             all_scores.append([])
#             continue
        
#         # Compute relevance scores
#         scores = scorer.compute_scores(frames, question)
        
#         # Store results
#         all_frames.append(frame_numbers)
#         all_scores.append(scores)
    
#     # Save results
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Create subdirectory for dataset
#     dataset_output_dir = output_dir / args.dataset_name / args.model_type
#     dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
#     frames_path = dataset_output_dir / 'frames.json'
#     scores_path = dataset_output_dir / 'scores.json'
    
#     print(f"\nSaving results...")
#     with open(frames_path, 'w') as f:
#         json.dump(all_frames, f)
    
#     with open(scores_path, 'w') as f:
#         json.dump(all_scores, f)
    
#     print(f"✅ Frames saved to: {frames_path}")
#     print(f"✅ Scores saved to: {scores_path}")
    
#     # Print statistics
#     print("\n" + "=" * 60)
#     print("Statistics:")
#     print("=" * 60)
#     total_frames = sum(len(f) for f in all_frames)
#     videos_processed = sum(1 for f in all_frames if len(f) > 0)
#     print(f"Videos processed: {videos_processed}/{len(data)}")
#     print(f"Total frames extracted: {total_frames}")
#     if videos_processed > 0:
#         print(f"Average frames per video: {total_frames / videos_processed:.2f}")
#     print("=" * 60)


# def main():
#     args = parse_arguments()
    
#     # Auto-detect JSON file path based on dataset_name
#     if args.json_file is None:
#         args.json_file = os.path.join(args.dataset_path, args.dataset_name, 'include_frame_idx.json')
    
#     # Use global video path variables
#     if args.video_folder is None:
#         if args.dataset_name == 'longvideobench':
#             args.video_folder = PATH_LONGVIDEOBENCH
#         elif args.dataset_name == 'videomme':
#             args.video_folder = PATH_VIDEOMME
#         else:
#             raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
#     # Choose processing method based on use_both_gpus flag
#     if args.use_both_gpus:
#         process_videos_parallel(args)
#     else:
#         process_videos(args)


# if __name__ == '__main__':
#     main()

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
import warnings
import torch.multiprocessing as mp
from queue import Queue
from threading import Thread
warnings.filterwarnings('ignore')

# Import models
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForImageTextRetrieval
)

# ========================================
# GLOBAL VIDEO PATH CONFIGURATION
# ========================================
PATH_LONGVIDEOBENCH = "/path/to/your/longvideobench/videos"
PATH_VIDEOMME = "/home/train01/.cache/huggingface/custom_video_qa_cache/data/data"
# ========================================


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract frames and compute relevance scores')
    
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
                        help='Path to folder containing videos (if None, uses global PATH variables)')
    
    parser.add_argument('--json_file', type=str, 
                        default=None,
                        help='Path to JSON file with questions (if None, uses include_frame_idx.json)')
    
    parser.add_argument('--output_dir', type=str, 
                        default='./output_scores',
                        help='Output directory for frames.json and scores.json (default: ./output_scores)')
    
    # Processing parameters
    parser.add_argument('--fps', type=float, 
                        default=1.0,  # Reduced for faster processing
                        help='Frames per second to extract (default: 1.0)')
    
    parser.add_argument('--model_type', type=str, 
                        default='blip',
                        choices=['clip', 'blip'],
                        help='Model to use for relevance scoring (default: clip)')
    
    parser.add_argument('--model_name', type=str, 
                        default=None,
                        help='Specific model name (default: auto-select based on model_type)')
    
    parser.add_argument('--batch_size', type=int, 
                        default=256,  # Start conservative
                        help='Batch size for processing frames (default: 256)')
    
    parser.add_argument('--device', type=str, 
                        default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, cpu (default: cuda:0)')
    
    parser.add_argument('--gpu_id', type=int,
                        default=0,
                        help='GPU ID to use (0 or 1, default: 0)')
    
    # Multi-GPU options
    parser.add_argument('--use_both_gpus', action='store_true',
                        help='Use both GPUs in parallel to process videos')
    
    parser.add_argument('--num_gpus', type=int,
                        default=2,
                        help='Number of GPUs to use when --use_both_gpus is set (default: 2)')
    
    # Performance optimization
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 mixed precision for faster inference')
    
    parser.add_argument('--compile_model', action='store_true',
                        help='Use torch.compile for faster inference (PyTorch 2.0+)')
    
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for faster CPU-GPU transfer')
    
    # Optional features
    parser.add_argument('--save_frames', action='store_true',
                        help='Save extracted frames to disk')
    
    parser.add_argument('--frames_output_dir', type=str, 
                        default='./extracted_frames',
                        help='Directory to save extracted frames if --save_frames is set')
    
    parser.add_argument('--num_videos', type=int,
                        default=None,
                        help='Limit number of videos to process for testing (default: all)')
    
    return parser.parse_args()


class FastVideoFrameExtractor:
    """Ultra-fast frame extractor using OpenCV acceleration"""
    
    def __init__(self, fps: float = 1.0, save_frames: bool = False, 
                 output_dir: str = './extracted_frames'):
        self.fps = fps
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str, video_id: str) -> Tuple[List[np.ndarray], List[int]]:
        """Ultra-fast frame extraction with direct frame seeking"""
        if not os.path.exists(video_path):
            return [], []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            cap.release()
            return [], []
        
        # Calculate which frames to extract
        frame_interval = max(1, int(video_fps / self.fps))
        frames_to_extract = list(range(0, total_frames, frame_interval))
        
        frames = []
        frame_numbers = []
        
        if self.save_frames:
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fast extraction using seeking
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_numbers.append(frame_idx)
            
            if self.save_frames:
                frame_path = video_output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
        
        cap.release()
        
        return frames, frame_numbers


class GPUOptimizedRelevanceScorer:
    """Maximally optimized scorer to ensure GPU is fully utilized"""
    
    def __init__(self, model_type: str = 'clip', model_name: str = None, 
                 device: str = 'cuda:0', batch_size: int = 256, 
                 use_fp16: bool = False, compile_model: bool = False,
                 pin_memory: bool = False):
        self.model_type = model_type
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.device = device
        self.pin_memory = pin_memory
        
        print(f"\n[{self.device}] Initializing GPU-optimized scorer...")
        print(f"  - Batch size: {batch_size}")
        print(f"  - FP16: {use_fp16}")
        print(f"  - Compile: {compile_model}")
        print(f"  - Pin memory: {pin_memory}")
        
        # Load model
        if model_type == 'clip':
            self.model_name = model_name or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
        elif model_type == 'blip':
            self.model_name = model_name or "Salesforce/blip-itm-base-coco"
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
        
        # Move to GPU
        self.model = self.model.to(self.device)
        
        # Enable FP16
        if self.use_fp16 and 'cuda' in self.device:
            self.model = self.model.half()
            print(f"  ✓ FP16 enabled")
        
        # Set to eval mode
        self.model.eval()
        
        # Compile model (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print(f"  ✓ Model compiled")
            except:
                print(f"  ⚠ torch.compile not available")
        
        # Enable cudnn optimizations
        if 'cuda' in self.device:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"  ✓ cuDNN optimizations enabled")
        
        print(f"[{self.device}] ✓ Scorer ready!\n")
    
    @torch.inference_mode()
    def compute_scores_batch(self, frames: List[np.ndarray], queries: List[str]) -> List[List[float]]:
        """Process multiple videos at once for better GPU utilization"""
        if len(frames) == 0:
            return []
        
        # This processes frames from MULTIPLE videos together
        # to keep GPU busy
        all_results = []
        
        for video_frames, query in zip(frames, queries):
            if len(video_frames) == 0:
                all_results.append([])
                continue
            
            pil_images = [Image.fromarray(frame) for frame in video_frames]
            
            if self.model_type == 'clip':
                scores = self._compute_clip_scores(pil_images, query)
            else:
                scores = self._compute_blip_scores(pil_images, query)
            
            all_results.append(scores)
        
        return all_results
    
    @torch.inference_mode()
    def compute_scores(self, frames: List[np.ndarray], query: str) -> List[float]:
        """Compute relevance scores for a single video"""
        if len(frames) == 0:
            return []
        
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        if self.model_type == 'clip':
            scores = self._compute_clip_scores(pil_images, query)
        elif self.model_type == 'blip':
            scores = self._compute_blip_scores(pil_images, query)
        
        return scores
    
    def _compute_clip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Optimized CLIP scoring"""
        
        # Process text once
        text_inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        text_inputs = {k: v.to(self.device, non_blocking=True) 
                      for k, v in text_inputs.items()}
        
        if self.use_fp16:
            text_inputs = {k: v.half() if v.dtype == torch.float32 else v 
                          for k, v in text_inputs.items()}
        
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        all_scores = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            image_inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            )
            
            # Move to device
            image_inputs = {k: v.to(self.device, non_blocking=True) 
                           for k, v in image_inputs.items()}
            
            if self.use_fp16:
                image_inputs = {k: v.half() if v.dtype == torch.float32 else v 
                               for k, v in image_inputs.items()}
            
            # Compute features
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarities = (image_features @ text_features.T).squeeze(-1)
            
            if similarities.dim() == 0:
                all_scores.append(similarities.item())
            else:
                all_scores.extend(similarities.cpu().tolist())
        
        return all_scores
    
    def _compute_blip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Optimized BLIP scoring"""
        all_scores = []
        
        batch_size = min(self.batch_size, 32)
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            for image in batch_images:
                inputs = self.processor(images=image, text=query, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) 
                         for k, v in inputs.items()}
                
                if self.use_fp16:
                    inputs = {k: v.half() if v.dtype == torch.float32 else v 
                             for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                score = torch.softmax(outputs.itm_score, dim=1)[0, 1].item()
                all_scores.append(score)
        
        return all_scores


def find_video_file(video_folder: Path, video_id: str) -> str:
    """Find video file by video_id"""
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    for ext in extensions:
        video_path = video_folder / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    return None


def process_video_subset(gpu_id: int, data_subset: List[Dict], args_dict: Dict):
    """Process videos on specific GPU with maximum utilization"""
    
    # Reconstruct args
    class Args:
        pass
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    device = f'cuda:{gpu_id}'
    
    print(f"\n{'='*70}")
    print(f"GPU {gpu_id}: Initializing")
    print(f"{'='*70}")
    print(f"  Videos to process: {len(data_subset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  FP16: {args.use_fp16}")
    print(f"  FPS: {args.fps}")
    print(f"{'='*70}\n")
    
    # Initialize components
    video_folder = Path(args.video_folder)
    extractor = FastVideoFrameExtractor(
        fps=args.fps,
        save_frames=args.save_frames,
        output_dir=args.frames_output_dir
    )
    
    scorer = GPUOptimizedRelevanceScorer(
        model_type=args.model_type,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        compile_model=args.compile_model,
        pin_memory=args.pin_memory
    )
    
    results = []
    
    # Process with progress bar
    pbar = tqdm(data_subset, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)
    
    for item in pbar:
        video_id = item.get('videoID', item.get('video_id', ''))
        question = item.get('question', '')
        
        pbar.set_postfix({'vid': video_id[:15]})
        
        # Find and extract
        video_path = find_video_file(video_folder, video_id)
        
        if video_path is None:
            results.append({'video_id': video_id, 'frames': [], 'scores': []})
            continue
        
        frames, frame_numbers = extractor.extract_frames(video_path, video_id)
        
        if len(frames) == 0:
            results.append({'video_id': video_id, 'frames': [], 'scores': []})
            continue
        
        # GPU computation
        scores = scorer.compute_scores(frames, question)
        
        results.append({
            'video_id': video_id,
            'frames': frame_numbers,
            'scores': scores
        })
    
    # Cleanup
    if 'cuda' in device:
        torch.cuda.empty_cache()
    
    print(f"\nGPU {gpu_id}: ✓ Completed {len(data_subset)} videos\n")
    
    return results


def process_videos_parallel(args):
    """Process with both GPUs"""
    
    print("\n" + "="*70)
    print("PARALLEL GPU PROCESSING")
    print("="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.use_fp16}")
    print("="*70 + "\n")
    
    # Validate
    if not os.path.exists(args.video_folder):
        raise FileNotFoundError(f"Video folder not found: {args.video_folder}")
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"JSON file not found: {args.json_file}")
    
    # Load data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    if args.num_videos:
        data = data[:args.num_videos]
    
    print(f"Loaded {len(data)} videos\n")
    
    # Split data
    chunk_size = len(data) // args.num_gpus
    data_splits = []
    
    for i in range(args.num_gpus):
        start = i * chunk_size
        end = len(data) if i == args.num_gpus - 1 else (i + 1) * chunk_size
        data_splits.append(data[start:end])
        print(f"GPU {i}: {len(data_splits[i])} videos")
    
    # Multiprocessing
    args_dict = vars(args)
    
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    print(f"\nStarting parallel processing...\n")
    
    # Create pool and process
    with mp.Pool(processes=args.num_gpus) as pool:
        results_list = pool.starmap(
            process_video_subset,
            [(gpu_id, data_splits[gpu_id], args_dict) for gpu_id in range(args.num_gpus)]
        )
    
    print(f"\n{'='*70}")
    print("All GPUs finished! Combining results...")
    print(f"{'='*70}\n")
    
    # Combine results
    all_frames = []
    all_scores = []
    
    for results in results_list:
        for result in results:
            all_frames.append(result['frames'])
            all_scores.append(result['scores'])
    
    # Save results
    output_dir = Path(args.output_dir)
    dataset_output_dir = output_dir / args.dataset_name / args.model_type
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_path = dataset_output_dir / 'frames.json'
    scores_path = dataset_output_dir / 'scores.json'
    
    print("Saving results...")
    with open(frames_path, 'w') as f:
        json.dump(all_frames, f)
    
    with open(scores_path, 'w') as f:
        json.dump(all_scores, f)
    
    print(f"✅ Frames saved to: {frames_path}")
    print(f"✅ Scores saved to: {scores_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    total_frames = sum(len(f) for f in all_frames)
    videos_processed = sum(1 for f in all_frames if len(f) > 0)
    print(f"Videos processed: {videos_processed}/{len(data)}")
    print(f"Total frames extracted: {total_frames}")
    if videos_processed > 0:
        print(f"Average frames per video: {total_frames / videos_processed:.2f}")
    print("="*70 + "\n")


def process_videos(args):
    """Single GPU processing"""
    
    if args.device == 'cuda:0' and args.gpu_id != 0:
        args.device = f'cuda:{args.gpu_id}'
    
    print("\n" + "="*70)
    print("SINGLE GPU PROCESSING")
    print("="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.use_fp16}")
    print("="*70 + "\n")
    
    # Validate
    if not os.path.exists(args.video_folder):
        raise FileNotFoundError(f"Video folder not found: {args.video_folder}")
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"JSON file not found: {args.json_file}")
    
    # Load data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    if args.num_videos:
        data = data[:args.num_videos]
    
    print(f"Loaded {len(data)} videos\n")
    
    # Initialize
    video_folder = Path(args.video_folder)
    extractor = FastVideoFrameExtractor(
        fps=args.fps,
        save_frames=args.save_frames,
        output_dir=args.frames_output_dir
    )
    
    scorer = GPUOptimizedRelevanceScorer(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        compile_model=args.compile_model,
        pin_memory=args.pin_memory
    )
    
    # Process
    all_frames = []
    all_scores = []
    
    print("Processing videos...\n")
    for item in tqdm(data, desc="Processing"):
        video_id = item.get('videoID', item.get('video_id', ''))
        question = item.get('question', '')
        
        video_path = find_video_file(video_folder, video_id)
        
        if video_path is None:
            all_frames.append([])
            all_scores.append([])
            continue
        
        frames, frame_numbers = extractor.extract_frames(video_path, video_id)
        
        if len(frames) == 0:
            all_frames.append([])
            all_scores.append([])
            continue
        
        scores = scorer.compute_scores(frames, question)
        
        all_frames.append(frame_numbers)
        all_scores.append(scores)
    
    # Cleanup
    if 'cuda' in args.device:
        torch.cuda.empty_cache()
    
    # Save
    output_dir = Path(args.output_dir)
    dataset_output_dir = output_dir / args.dataset_name / args.model_type
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_path = dataset_output_dir / 'frames.json'
    scores_path = dataset_output_dir / 'scores.json'
    
    print(f"\nSaving results...")
    with open(frames_path, 'w') as f:
        json.dump(all_frames, f)
    
    with open(scores_path, 'w') as f:
        json.dump(all_scores, f)
    
    print(f"✅ Frames saved to: {frames_path}")
    print(f"✅ Scores saved to: {scores_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    total_frames = sum(len(f) for f in all_frames)
    videos_processed = sum(1 for f in all_frames if len(f) > 0)
    print(f"Videos processed: {videos_processed}/{len(data)}")
    print(f"Total frames extracted: {total_frames}")
    if videos_processed > 0:
        print(f"Average frames per video: {total_frames / videos_processed:.2f}")
    print("="*70 + "\n")


def main():
    args = parse_arguments()
    
    # Auto-detect paths
    if args.json_file is None:
        args.json_file = os.path.join(args.dataset_path, args.dataset_name, 'include_frame_idx.json')
    
    if args.video_folder is None:
        if args.dataset_name == 'longvideobench':
            args.video_folder = PATH_LONGVIDEOBENCH
        elif args.dataset_name == 'videomme':
            args.video_folder = PATH_VIDEOMME
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Route to correct function
    if args.use_both_gpus:
        process_videos_parallel(args)
    else:
        process_videos(args)


if __name__ == '__main__':
    main()