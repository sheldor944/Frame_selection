import math
import subprocess
import os

PYTHON = "python"
SCRIPT = "frame_select_aks.py"

DATASETS = ["longvideobench", "videomme"]
FEATURES = ["clip", "blip", "sevila"]
AKS_VALUES = [8, 16, 32]
RATIOS = [1]

OUTPUT_ROOT = "./Thesis_TOP_K"
OUTSCORE_ROOT = "./outscores"


def build_paths(dataset, feature):
    score_path = f"{OUTSCORE_ROOT}/{dataset}/{feature}/scores.json"
    frame_path = f"{OUTSCORE_ROOT}/{dataset}/{feature}/frames.json"
    return score_path, frame_path


def build_output_filename(dataset, feature, aks, ratio):
    return f"selected_{dataset}_{feature}_aks{aks}_ratio{ratio}.json"


def compute_depth(aks):
    return 0
    # return int(math.log2(aks))


def run_one(dataset, feature, aks, ratio):
    score_path, frame_path = build_paths(dataset, feature)
    depth = compute_depth(aks)

    output_filename = build_output_filename(dataset, feature, aks, ratio)

    cmd = [
        PYTHON, SCRIPT,
        "--dataset_name", dataset,
        "--extract_feature_model", feature,
        "--score_path", score_path,
        "--frame_path", frame_path,
        "--max_num_frames", str(aks),
        "--ratio", str(ratio),
        "--all_depth", str(depth),
        "--output_file", OUTPUT_ROOT,
        "--output_name", output_filename
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for dataset in DATASETS:
        for feature in FEATURES:
            for aks in AKS_VALUES:
                for ratio in RATIOS:
                    run_one(dataset, feature, aks, ratio)


if __name__ == "__main__":
    main()
