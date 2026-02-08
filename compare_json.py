import json
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

differences = []

def normalize_value(v):
    """
    Convert ints and floats to a common float type for comparison.
    Everything else returned as-is.
    """
    if isinstance(v, (int, float)):
        return float(v)
    return v

def deep_compare(a, b, path=""):
    # Normalize primitive numeric types (int vs float)
    if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
        if normalize_value(a) != normalize_value(b):
            differences.append(f"Value mismatch at {path}: {a} != {b}")
        return

    # Dict comparison
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        # Missing keys
        for key in keys_a - keys_b:
            differences.append(f"Key '{key}' missing in second file at {path}")

        for key in keys_b - keys_a:
            differences.append(f"Extra key '{key}' in second file at {path}")

        # Compare shared keys
        for key in keys_a & keys_b:
            deep_compare(a[key], b[key], f"{path}/{key}")
        return

    # List comparison
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append(f"List length mismatch at {path}: {len(a)} != {len(b)}")
            return

        for i in range(len(a)):
            deep_compare(a[i], b[i], f"{path}[{i}]")
        return

    # If types differ but not primitive, mark mismatch
    differences.append(f"Type mismatch at {path}: {type(a)} != {type(b)}")


# Load files
json1 = load_json(file1)
json2 = load_json(file2)

# Compare deeply
deep_compare(json1, json2)

# Output result
if not differences:
    print("✅ JSON files are the SAME (value-wise, ignoring int/float type).")
else:
    print("❌ JSON files differ in values:")
    for diff in differences:
        print("-", diff)
