import json
from pathlib import Path
from collections import defaultdict

def group_identical_json_files(folder_path):
    groups = defaultdict(list)
    
    for file in Path(folder_path).glob('*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
        data_str = json.dumps(data, sort_keys=True)  # Normalize to string
        groups[data_str].append(file.name)
    
    return dict(groups)

# Usage
groups = group_identical_json_files('selected_frames_LV')
for data_hash, files in groups.items():
    print(f"{len(files)} identical files: {files}")