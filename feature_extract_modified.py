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
                        default=12.0,
                        help='Frames per second to extract (default: 1.0)')
    
    parser.add_argument('--model_type', type=str, 
                        default='clip',
                        choices=['clip', 'blip'],
                        help='Model to use for relevance scoring (default: clip)')
    
    parser.add_argument('--model_name', type=str, 
                        default=None,
                        help='Specific model name (default: auto-select based on model_type)')
    
    parser.add_argument('--batch_size', type=int, 
                        default=16,
                        help='Batch size for processing frames (default: 16)')
    
    parser.add_argument('--device', type=str, 
                        default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, cpu (default: cuda:0)')
    
    parser.add_argument('--gpu_id', type=int,
                        default=0,
                        help='GPU ID to use (0 or 1, default: 0)')
    
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


class VideoFrameExtractor:
    """Extract frames from video at specified FPS"""
    
    def __init__(self, fps: float = 1.0, save_frames: bool = False, 
                 output_dir: str = './extracted_frames'):
        self.fps = fps
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str, video_id: str) -> Tuple[List[np.ndarray], List[int]]:
        """Extract frames from video at specified FPS."""
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            return [], []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return [], []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            print(f"Error: Invalid FPS for video: {video_path}")
            cap.release()
            return [], []
        
        # Calculate frame interval
        frame_interval = int(video_fps / self.fps)
        frame_interval = max(1, frame_interval)
        
        frames = []
        frame_numbers = []
        
        # Create output directory for this video if saving frames
        if self.save_frames:
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_numbers.append(frame_count)
                
                # Save frame if requested
                if self.save_frames:
                    frame_path = video_output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"  Extracted {extracted_count} frames from {total_frames} total frames (FPS: {video_fps:.2f})")
        
        return frames, frame_numbers


class RelevanceScorer:
    """Compute relevance scores between frames and text query"""
    
    def __init__(self, model_type: str = 'clip', model_name: str = None, 
                 device: str = None, batch_size: int = 32):
        self.model_type = model_type
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        if model_type == 'clip':
            self.model_name = model_name or "openai/clip-vit-base-patch32"
            print(f"Loading CLIP model: {self.model_name}")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        
        elif model_type == 'blip':
            self.model_name = model_name or "Salesforce/blip-itm-base-coco"
            print(f"Loading BLIP model: {self.model_name}")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def compute_scores(self, frames: List[np.ndarray], query: str) -> List[float]:
        """Compute relevance scores for frames given a text query."""
        if len(frames) == 0:
            return []
        
        # Convert all frames to PIL Images first
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        if self.model_type == 'clip':
            scores = self._compute_clip_scores(pil_images, query)
        elif self.model_type == 'blip':
            scores = self._compute_blip_scores(pil_images, query)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return scores
    
    def _compute_clip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Compute CLIP similarity scores - FIXED VERSION"""
        # Get text features once
        text_inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        scores = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            image_inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity for each image in batch
                batch_similarities = (image_features @ text_features.T).squeeze(-1)
                
                # Convert to list
                if batch_similarities.dim() == 0:
                    scores.append(batch_similarities.item())
                else:
                    scores.extend(batch_similarities.cpu().tolist())
        
        return scores
    
    def _compute_blip_scores(self, images: List[Image.Image], query: str) -> List[float]:
        """Compute BLIP ITM scores"""
        scores = []
        
        for image in images:
            inputs = self.processor(
                images=image,
                text=query,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                itm_score = outputs.itm_score
                score = torch.softmax(itm_score, dim=1)[0, 1].item()
                scores.append(score)
        
        return scores


def find_video_file(video_folder: Path, video_id: str) -> str:
    """Find video file by video_id, checking common extensions."""
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    for ext in extensions:
        video_path = video_folder / f"{video_id}{ext}"
        if video_path.exists():
            return str(video_path)
    
    return None


def process_videos(args):
    """Main processing function"""
    
    # Auto-detect JSON file path based on dataset_name
    if args.json_file is None:
        args.json_file = os.path.join(args.dataset_path, args.dataset_name, 'include_frame_idx.json')
    
    # Use global video path variables
    if args.video_folder is None:
        if args.dataset_name == 'longvideobench':
            args.video_folder = PATH_LONGVIDEOBENCH
        elif args.dataset_name == 'videomme':
            args.video_folder = PATH_VIDEOMME
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Set device based on gpu_id
    if args.device == 'cuda:0' and args.gpu_id != 0:
        args.device = f'cuda:{args.gpu_id}'
    
    print("=" * 60)
    print("Video Frame Extraction and Relevance Scoring")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Video folder: {args.video_folder}")
    print(f"JSON file: {args.json_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Validate paths
    if not os.path.exists(args.video_folder):
        raise FileNotFoundError(f"❌ Video folder not found: {args.video_folder}\n"
                              f"Please update the global path variable for {args.dataset_name.upper()}")
    
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"❌ JSON file not found: {args.json_file}")
    
    # Load JSON data
    print("\nLoading JSON data...")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Limit number of videos if specified
    if args.num_videos is not None:
        data = data[:args.num_videos]
        print(f"⚠️  Processing only first {args.num_videos} videos (test mode)")
    
    print(f"Loaded {len(data)} questions")

    # Initialize components
    video_folder = Path(args.video_folder)
    extractor = VideoFrameExtractor(
        fps=args.fps,
        save_frames=args.save_frames,
        output_dir=args.frames_output_dir
    )
    
    scorer = RelevanceScorer(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Storage for results
    all_frames = []
    all_scores = []
    
    # Process each video/question
    print("\nProcessing videos...")
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        video_id = item.get('videoID', item.get('video_id', f"{idx:03d}"))
        question = item.get('question', '')
        
        # Find video file
        video_path = find_video_file(video_folder, video_id)
        
        if video_path is None:
            print(f"\nWarning: Video not found for ID {video_id}, skipping...")
            all_frames.append([])
            all_scores.append([])
            continue
        
        print(f"\n[{idx + 1}/{len(data)}] Processing: {video_id}")
        print(f"  Question: {question[:100]}...")
        
        # Extract frames
        frames, frame_numbers = extractor.extract_frames(video_path, video_id)
        
        if len(frames) == 0:
            print(f"  No frames extracted, skipping...")
            all_frames.append([])
            all_scores.append([])
            continue
        
        # Compute relevance scores
        print(f"  Computing relevance scores...")
        scores = scorer.compute_scores(frames, question)
        
        # Store results
        all_frames.append(frame_numbers)
        all_scores.append(scores)
        
        # Print score statistics
        if scores and len(scores) > 0:
            scores_array = np.array(scores)
            print(f"  Scores: min={scores_array.min():.4f}, max={scores_array.max():.4f}, mean={scores_array.mean():.4f}")
        else:
            print(f"  Scores: (no scores computed)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for dataset
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
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    total_frames = sum(len(f) for f in all_frames)
    videos_processed = sum(1 for f in all_frames if len(f) > 0)
    print(f"Videos processed: {videos_processed}/{len(data)}")
    print(f"Total frames extracted: {total_frames}")
    if videos_processed > 0:
        print(f"Average frames per video: {total_frames / videos_processed:.2f}")
    print("=" * 60)


def main():
    args = parse_arguments()
    process_videos(args)


if __name__ == '__main__':
    main()