#!/usr/bin/env python3
"""
Root-level launcher for SegFormer + YOLOv8 Navigation v5.

Delegates to the actual implementation in src/segformer_yolo_navigation_v5.py.
Run from project root:
    python segformer_yolo_navigation_v5.py --source 00a0f008-a315437f.mov --output out.mp4
"""
import os
import sys

# Add src/ to Python path so all imports resolve correctly
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _src_dir)

# Change working directory to project root (where models/ and .mov files live)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from segformer_yolo_navigation_v5 import main

if __name__ == "__main__":
    main()
