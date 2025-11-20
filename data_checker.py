import json
import numpy as np
import os

def check_data_dimensions(score_path, frame_path):
    print("=== Data Structure Inspection ===")
    print(f"Score file: {score_path}")
    print(f"Frame file: {frame_path}")
    
    with open(score_path) as f:
        itm_outs = json.load(f)
    with open(frame_path) as f:
        fn_outs = json.load(f)
    
    print(f"\nNumber of videos (score sets): {len(itm_outs)}")
    print(f"Number of videos (frame sets): {len(fn_outs)}")

    # inspect one or few examples
    for i, (itm_out, fn_out) in enumerate(zip(itm_outs, fn_outs)):
        print(f"\n--- Video {i} ---")
        print(f"Type of itm_out: {type(itm_out)} | len: {len(itm_out)}")
        print(f"Type of fn_out: {type(fn_out)} | len: {len(fn_out)}")

        # Check first few items
        if len(itm_out) > 0:
            print(f"First itm_out[0]: {itm_out[0]} (type: {type(itm_out[0])})")
        if len(fn_out) > 0:
            print(f"First fn_out[0]: {fn_out[0]} (type: {type(fn_out[0])})")

        # Check numeric stats if applicable
        try:
            score_array = np.array(itm_out, dtype=float)
            print(f"Converted to np.array: shape={score_array.shape}, dtype={score_array.dtype}")
            print(f"Min={np.min(score_array):.4f}, Max={np.max(score_array):.4f}, Mean={np.mean(score_array):.4f}")
        except Exception as e:
            print(f"⚠️ Could not convert itm_out to numeric array: {e}")

        # Check frame identifiers
        try:
            print(f"Example frame IDs: {fn_out[:5]}")
        except:
            pass

        if i >= 2:  # just inspect first 3 videos
            break

    print("\n=== meanstd Input Simulation ===")
    # simulate what goes into meanstd
    dummy_score = np.random.rand(10)
    dummy_fn = [f"frame_{i}" for i in range(10)]
    print(f"dummy_score: shape={dummy_score.shape}, dtype={dummy_score.dtype}")
    print(f"dummy_fn: len={len(dummy_fn)}, first 3={dummy_fn[:3]}")

    print("\n✅ Inspection complete. Use these outputs to confirm structure before implementing DBFP.")

if __name__ == "__main__":
    score_path = "./outscores/longvideobench/blip/scores.json"
    frame_path = "./outscores/longvideobench/blip/frames.json"
    check_data_dimensions(score_path, frame_path)
