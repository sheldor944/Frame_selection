#!/usr/bin/env python3
"""
Run frame extraction on multiple GPUs in parallel
"""
import subprocess
import json
import argparse
from pathlib import Path
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run extraction on multiple GPUs')
    
    parser.add_argument('--dataset_name', type=str, 
                        default='videomme',
                        choices=['longvideobench', 'videomme'],
                        help='Dataset name')
    
    parser.add_argument('--model_type', type=str, 
                        default='clip',
                        choices=['clip', 'blip'],
                        help='Model type')
    
    parser.add_argument('--fps', type=float, 
                        default=1.0,
                        help='Frames per second')
    
    parser.add_argument('--batch_size', type=int, 
                        default=16,
                        help='Batch size per GPU')
    
    parser.add_argument('--gpus', type=str,
                        default='0,1',
                        help='Comma-separated GPU IDs to use (default: 0,1)')
    
    return parser.parse_args()


def split_dataset(json_file, num_splits):
    """Split dataset into N parts"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    chunk_size = (total + num_splits - 1) // num_splits
    
    splits = []
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total)
        splits.append((start_idx, end_idx))
    
    return splits, total


def run_gpu_process(gpu_id, start_idx, end_idx, args):
    """Run extraction on a single GPU"""
    output_suffix = f"_gpu{gpu_id}_part{start_idx}-{end_idx}"
    
    cmd = [
        'python3', 'feature_extract_modified.py',
        '--dataset_name', args.dataset_name,
        '--model_type', args.model_type,
        '--fps', str(args.fps),
        '--batch_size', str(args.batch_size),
        '--gpu_id', str(gpu_id),
        '--start_idx', str(start_idx),
        '--end_idx', str(end_idx),
        '--output_suffix', output_suffix
    ]
    
    return subprocess.Popen(cmd)


def main():
    args = parse_arguments()
    
    # Get GPU list
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    
    # Determine JSON file
    json_file = f'./datasets/{args.dataset_name}/include_frame_idx.json'
    
    if not Path(json_file).exists():
        print(f"❌ JSON file not found: {json_file}")
        return 1
    
    # Split dataset
    print(f"Splitting dataset across {num_gpus} GPUs...")
    splits, total = split_dataset(json_file, num_gpus)
    
    print(f"Total videos: {total}")
    for i, (start, end) in enumerate(splits):
        print(f"  GPU {gpu_ids[i]}: videos {start}-{end-1} ({end-start} videos)")
    
    # Launch processes
    print("\nLaunching GPU processes...")
    processes = []
    for i, (start, end) in enumerate(splits):
        gpu_id = gpu_ids[i]
        print(f"  Starting GPU {gpu_id}...")
        proc = run_gpu_process(gpu_id, start, end, args)
        processes.append((gpu_id, proc))
    
    print("\n✅ All processes launched!")
    print("Monitor progress in separate terminals with:")
    for gpu_id, _ in processes:
        print(f"  watch -n 1 nvidia-smi")
    
    # Wait for completion
    print("\nWaiting for processes to complete...")
    for gpu_id, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"✅ GPU {gpu_id} completed successfully")
        else:
            print(f"❌ GPU {gpu_id} failed with code {proc.returncode}")
    
    print("\n" + "="*60)
    print("All GPU processes completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Merge the output files with: python3 merge_results.py")
    print("2. Check output in: ./output_scores/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())