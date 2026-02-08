import json
import argparse

def create_longvideobench_metadata(input_json_path, output_metadata_path):
    """Create metadata file for LongVideoBench dataset."""
    print("=" * 60)
    print("Creating LongVideoBench Metadata")
    print("=" * 60)
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries found: {len(data)}")
    
    metadata = []
    duration_group_counts = {15: 0, 60: 0, 600: 0, 3600: 0}
    
    for idx, entry in enumerate(data):
        video_id = entry.get('video_id', f'video_{idx}')
        duration_group = entry.get('duration_group', None)
        duration = entry.get('duration', None)
        
        if duration_group is None:
            print(f"⚠️  Warning: Entry {idx} missing 'duration_group'")
            if duration is not None:
                if duration <= 15:
                    duration_group = 15
                elif duration <= 60:
                    duration_group = 60
                elif duration <= 600:
                    duration_group = 600
                else:
                    duration_group = 3600
            else:
                duration_group = 3600
        
        metadata_entry = {
            "video_id": video_id,
            "duration_group": duration_group,
            "duration": duration,
            "video_path": entry.get('video_path', ''),
            "question_id": entry.get('id', '')
        }
        
        metadata.append(metadata_entry)
        duration_group_counts[duration_group] += 1
    
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to: {output_metadata_path}")
    print(f"Total videos: {len(metadata)}")
    print(f"\nDuration Group Distribution:")
    for group in [15, 60, 600, 3600]:
        count = duration_group_counts[group]
        print(f"  {group:4d}s: {count:5d} videos ({count/len(metadata)*100:.1f}%)")
    
    return metadata


def create_videomme_metadata(input_json_path, output_metadata_path):
    """Create metadata file for VideoMME dataset."""
    print("=" * 60)
    print("Creating VideoMME Metadata")
    print("=" * 60)
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries found: {len(data)}")
    
    metadata = []
    duration_counts = {"short": 0, "medium": 0, "long": 0}
    
    for idx, entry in enumerate(data):
        video_id = entry.get('video_id', f'video_{idx}')
        duration = entry.get('duration', 'medium')
        
        if duration not in ["short", "medium", "long"]:
            print(f"⚠️  Warning: Entry {idx} has invalid duration: {duration}")
            duration = "medium"
        
        metadata_entry = {
            "video_id": video_id,
            "duration": duration,
            "domain": entry.get('domain', ''),
            "sub_category": entry.get('sub_category', ''),
            "videoID": entry.get('videoID', ''),
            "question_id": entry.get('question_id', '')
        }
        
        metadata.append(metadata_entry)
        duration_counts[duration] += 1
    
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to: {output_metadata_path}")
    print(f"Total videos: {len(metadata)}")
    print(f"\nDuration Distribution:")
    for dur in ["short", "medium", "long"]:
        count = duration_counts[dur]
        print(f"  {dur:6s}: {count:5d} videos ({count/len(metadata)*100:.1f}%)")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Create metadata files for video datasets')
    parser.add_argument('--dataset', type=str, default="videomme",
                        choices=['longvideobench', 'videomme'],
                        help='Dataset name')
    
    args = parser.parse_args()
    
    if args.dataset == 'longvideobench':
        create_longvideobench_metadata("./datasets/longvideobench/include_frame_idx.json", "datasets/longvideobench/metadata.json")
    elif args.dataset == 'videomme':
        create_videomme_metadata("./datasets/videomme/include_frame_idx.json", "datasets/videomme/metadatta.json")


if __name__ == "__main__":
    main()