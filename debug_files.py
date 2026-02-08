import json

# Check frames.json
with open('./output_dense_sampling/videomme/blip/frames_dense.json', 'r') as f:
    frames = json.load(f)

print(f"frames.json:")
print(f"  Type: {type(frames)}")
print(f"  Length: {len(frames)}")
print(f"  First entry type: {type(frames[0])}")
print(f"  First entry length: {len(frames[0])}")
print(f"  First entry: {frames[0][:10] if len(frames[0]) > 10 else frames[0]}")

# Check scores.json
with open('./output_dense_sampling/videomme/blip/scores_dense.json', 'r') as f:
    scores = json.load(f)

print(f"\nscores.json:")
print(f"  Type: {type(scores)}")
print(f"  Length: {len(scores)}")
print(f"  First entry type: {type(scores[0])}")
print(f"  First entry length: {len(scores[0])}")
print(f"  First entry: {scores[0][:10] if len(scores[0]) > 10 else scores[0]}")