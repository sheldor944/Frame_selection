import json
import sys

def test_frames_scores_alignment(frames_path, scores_path):
    """
    Test if frames.json and scores.json have matching lengths for all videos
    """
    
    print("="*70)
    print("FRAMES AND SCORES ALIGNMENT TEST")
    print("="*70)
    
    # Load files
    print(f"\nLoading files...")
    print(f"  Frames: {frames_path}")
    print(f"  Scores: {scores_path}")
    
    try:
        with open(frames_path, 'r') as f:
            frames_data = json.load(f)
        print(f"  ✓ Frames loaded: {len(frames_data)} videos")
    except Exception as e:
        print(f"  ✗ Error loading frames: {e}")
        return
    
    try:
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)
        print(f"  ✓ Scores loaded: {len(scores_data)} videos")
    except Exception as e:
        print(f"  ✗ Error loading scores: {e}")
        return
    
    # Check if same number of videos
    if len(frames_data) != len(scores_data):
        print(f"\n✗ ERROR: Different number of videos!")
        print(f"  Frames: {len(frames_data)} videos")
        print(f"  Scores: {len(scores_data)} videos")
        return
    
    print(f"  ✓ Both files have {len(frames_data)} videos")
    
    # Test each video
    print(f"\n{'='*70}")
    print("TESTING EACH VIDEO")
    print(f"{'='*70}\n")
    
    total_videos = len(frames_data)
    matching_videos = 0
    mismatched_videos = 0
    empty_videos = 0
    
    mismatches = []
    
    for i in range(total_videos):
        frames = frames_data[i]
        scores = scores_data[i]
        
        frames_len = len(frames)
        scores_len = len(scores)
        
        # Check if both empty
        if frames_len == 0 and scores_len == 0:
            empty_videos += 1
            continue
        
        # Check if lengths match
        if frames_len == scores_len:
            matching_videos += 1
        else:
            mismatched_videos += 1
            mismatches.append({
                'video_idx': i,
                'frames_len': frames_len,
                'scores_len': scores_len,
                'difference': abs(frames_len - scores_len)
            })
    
    # Print summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos:           {total_videos}")
    print(f"Matching videos:        {matching_videos} ({matching_videos/total_videos*100:.1f}%)")
    print(f"Mismatched videos:      {mismatched_videos} ({mismatched_videos/total_videos*100:.1f}%)")
    print(f"Empty videos:           {empty_videos} ({empty_videos/total_videos*100:.1f}%)")
    print(f"{'='*70}\n")
    
    # Show detailed mismatches
    if mismatched_videos > 0:
        print(f"{'='*70}")
        print(f"MISMATCHED VIDEOS (showing first 20)")
        print(f"{'='*70}")
        print(f"{'Video':<10} {'Frames':<15} {'Scores':<15} {'Diff':<10}")
        print(f"{'-'*70}")
        
        for mismatch in mismatches[:20]:
            video_idx = mismatch['video_idx']
            frames_len = mismatch['frames_len']
            scores_len = mismatch['scores_len']
            diff = mismatch['difference']
            
            print(f"{video_idx:<10} {frames_len:<15} {scores_len:<15} {diff:<10}")
        
        if len(mismatches) > 20:
            print(f"\n... and {len(mismatches) - 20} more mismatched videos")
        
        print(f"\n{'='*70}")
        print("DETAILED ANALYSIS OF FIRST MISMATCHED VIDEO")
        print(f"{'='*70}")
        
        first_mismatch = mismatches[0]
        video_idx = first_mismatch['video_idx']
        frames = frames_data[video_idx]
        scores = scores_data[video_idx]
        
        print(f"\nVideo index: {video_idx}")
        print(f"Frames length: {len(frames)}")
        print(f"Scores length: {len(scores)}")
        print(f"\nFirst 20 frame indices:")
        print(frames[:20])
        print(f"\nFirst 20 scores:")
        print(scores[:20])
        
        if len(frames) > 20:
            print(f"\nLast 10 frame indices:")
            print(frames[-10:])
        if len(scores) > 20:
            print(f"\nLast 10 scores:")
            print(scores[-10:])
    
    # Statistics
    if mismatched_videos > 0:
        print(f"\n{'='*70}")
        print("MISMATCH STATISTICS")
        print(f"{'='*70}")
        
        differences = [m['difference'] for m in mismatches]
        frames_lens = [m['frames_len'] for m in mismatches]
        scores_lens = [m['scores_len'] for m in mismatches]
        
        print(f"Average frames per mismatched video: {sum(frames_lens)/len(frames_lens):.1f}")
        print(f"Average scores per mismatched video: {sum(scores_lens)/len(scores_lens):.1f}")
        print(f"Average difference: {sum(differences)/len(differences):.1f}")
        print(f"Max difference: {max(differences)}")
        print(f"Min difference: {min(differences)}")
        
        # Check if there's a pattern
        ratio = sum(scores_lens) / sum(frames_lens) if sum(frames_lens) > 0 else 0
        print(f"\nScores/Frames ratio: {ratio:.2f}")
        
        if ratio > 10:
            print(f"\n⚠️  LIKELY ISSUE: Scores contain many more entries than frames!")
            print(f"   This suggests frames.json contains SAMPLE POINTS (e.g., every 1 second)")
            print(f"   while scores.json contains SCORES FOR ALL EXTRACTED FRAMES")
            print(f"   (e.g., if FPS=12, then 12 scores per frame index)")
    
    print(f"\n{'='*70}")
    
    # Return status
    if mismatched_videos == 0:
        print("✓ TEST PASSED: All videos have matching frames and scores lengths")
        return True
    else:
        print("✗ TEST FAILED: Some videos have mismatched lengths")
        return False


if __name__ == '__main__':
    # Default paths
    frames_path = './output_dense_sampling/videomme/blip/frames_dense.json'
    scores_path = './output_dense_sampling/videomme/blip/scores_dense.json'
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        frames_path = sys.argv[1]
        scores_path = sys.argv[2]
    
    test_frames_scores_alignment(frames_path, scores_path)