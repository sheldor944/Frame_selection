import json

input_path = "/home/hpc4090/miraj/AKS/AKS/selected_frames/videomme/blip/selected_frames_random.json"       # your source file
output_path = "/home/hpc4090/miraj/AKS/AKS/selected_frames/videomme/blip/all_zero.json"  # your new file

with open(input_path, 'r') as f:
    data = json.load(f)

# Replace all numbers with 0 (preserve structure)
zero_data = [[0 for _ in inner] for inner in data]

with open(output_path, 'w') as f:
    json.dump(zero_data, f, indent=2)

print(f"✅ Created {output_path} with all values set to 0.")
