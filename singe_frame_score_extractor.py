import json

def extract_example(score_file, frames_file, example_id, output_dir="."):
    """
    Extract a single example from score and frames files and create new files.
    
    Args:
        score_file: Path to the original score file
        frames_file: Path to the original frames file
        example_id: Index of the example to extract (0-based)
        output_dir: Directory to save the output files
    """
    
    # Read the original score file
    with open(score_file, 'r') as f:
        scores = json.load(f)
    
    # Read the original frames file
    with open(frames_file, 'r') as f:
        frames = json.load(f)
    
    # Validate the example_id
    if example_id < 0 or example_id >= len(scores):
        raise ValueError(f"example_id {example_id} is out of range. Valid range: 0 to {len(scores)-1}")
    
    if example_id >= len(frames):
        raise ValueError(f"example_id {example_id} is out of range for frames. Valid range: 0 to {len(frames)-1}")
    
    # Extract the specific example
    extracted_score = scores[example_id]
    extracted_frame = frames[example_id]
    
    # Create output file names
    score_output = f"{output_dir}/score_{example_id}.json"
    frame_output = f"{output_dir}/frame_{example_id}.json"
    
    # Write the extracted score to a new file
    with open(score_output, 'w') as f:
        json.dump(extracted_score, f, indent=2)
    
    # Write the extracted frame to a new file
    with open(frame_output, 'w') as f:
        json.dump(extracted_frame, f, indent=2)
    
    print(f"Successfully extracted example {example_id}")
    print(f"Score saved to: {score_output}")
    print(f"Frame saved to: {frame_output}")
    
    return extracted_score, extracted_frame


def main():
    # Configuration - modify these paths as needed
    score_file = "./outscores/longvideobench/blip/scores.json"      # Path to your original score file
    frames_file = "./outscores/longvideobench/blip/frames.json"     # Path to your original frames file
    
    # Get the example ID from user input
    example_id = 25
    
    # Extract and save
    extract_example(score_file, frames_file, example_id)


if __name__ == "__main__":
    main()