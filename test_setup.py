#!/usr/bin/env python3
"""
Quick test script to verify setup before running the main extraction
"""
import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'tqdm': 'tqdm'
    }
    
    failed = []
    for package, pip_name in packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} - NOT INSTALLED")
            failed.append(pip_name)
    
    return len(failed) == 0

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print(f"  ⚠️  CUDA not available - will use CPU (slower)")
        return False

def test_paths():
    """Test if dataset paths exist"""
    print("\nTesting paths...")
    import os
    
    paths = {
        'datasets/longvideobench/include_frame_idx.json': 'LongVideoBench JSON',
        'datasets/videomme/include_frame_idx.json': 'VideoMME JSON',
    }
    
    all_ok = True
    for path, name in paths.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: {path} - NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("Setup Verification Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test GPU
    gpu_ok = test_gpu()
    
    # Test paths
    paths_ok = test_paths()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if imports_ok and paths_ok:
        print("✅ All checks passed! Ready to run extraction.")
        if gpu_ok:
            print("✅ GPU available - will be fast!")
        else:
            print("⚠️  No GPU - will be slower but still works")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())