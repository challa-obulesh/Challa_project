#!/usr/bin/env python3
"""
Live webcam / camera runner for Jetson AGX Orin navigation pipeline.

Supports USB webcams, CSI cameras, and RealSense (via V4L2).

Usage:
    python3 run_webcam.py                       # Default camera (0)
    python3 run_webcam.py --source 1             # Camera index 1
    python3 run_webcam.py --source /dev/video0   # Specific device
"""

import os
import sys
import argparse

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetson_pipeline import NavigationPipeline


def run_webcam(args):
    """Run live navigation pipeline on webcam feed."""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Model paths
    models_dir = os.path.join(project_root, args.models_dir)
    yolo_engine = os.path.join(models_dir, "yolov8n.engine")
    seg_engine = os.path.join(models_dir, "segformer_b0.engine")
    yolo_weights = os.path.join(project_root, "yolov8n.pt")

    # ── Initialise Pipeline ──────────────────────────────────────────
    pipeline = NavigationPipeline(
        yolo_path=yolo_engine,
        segformer_path=seg_engine,
        yolo_weights_fallback=yolo_weights,
        seg_input_size=(args.seg_h, args.seg_w),
        conf_thresh=args.conf_thresh,
        panel_scale=args.panel_scale,
    )

    # ── Open Camera ──────────────────────────────────────────────────
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # String path (e.g., /dev/video0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open camera: {source}")
        sys.exit(1)

    # Set resolution if specified
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[Webcam] Source: {source} ({actual_w}×{actual_h})")
    print("[Webcam] Press 'q' to quit")
    print("-" * 60)

    # ── Main Loop ────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Webcam] Frame read failed, retrying...")
                continue

            # Process
            result = pipeline.process_frame(frame)
            grid = result["grid"]

            # Show
            display_scale = args.display_scale
            if display_scale != 1.0:
                display = cv2.resize(grid, (0, 0),
                                     fx=display_scale, fy=display_scale)
            else:
                display = grid

            cv2.imshow("Jetson Navigation — Live", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save screenshot
                cv2.imwrite("screenshot.jpg", grid)
                print("[Webcam] Screenshot saved: screenshot.jpg")

    except KeyboardInterrupt:
        print("\n[Webcam] Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Jetson Navigation — Webcam Runner")
    parser.add_argument("--source", "-s", default="0", help="Camera source")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    parser.add_argument("--seg-h", type=int, default=512, help="SegFormer input height")
    parser.add_argument("--seg-w", type=int, default=512, help="SegFormer input width")
    parser.add_argument("--conf-thresh", type=float, default=0.35, help="YOLO confidence")
    parser.add_argument("--panel-scale", type=float, default=0.5, help="Panel scale")
    parser.add_argument("--display-scale", type=float, default=0.7, help="Display scale")
    args = parser.parse_args()

    run_webcam(args)


if __name__ == "__main__":
    main()
