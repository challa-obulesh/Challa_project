#!/usr/bin/env python3
"""
One-time TensorRT engine export for Jetson AGX Orin.

Exports:
  1. YOLOv8n  → TensorRT FP16 engine (via Ultralytics)
  2. SegFormer-B0 → ONNX → TensorRT FP16 engine

Run once; engines are cached on disk for subsequent inference.

Usage:
    python3 export_tensorrt.py [--models-dir ./models]
"""

import os
import sys
import argparse
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def export_yolo_trt(
    weights_path: str,
    output_dir: str,
    imgsz: int = 640,
    half: bool = True,
):
    """Export YOLOv8n to TensorRT FP16 engine via Ultralytics."""
    from ultralytics import YOLO

    engine_path = os.path.join(output_dir, "yolov8n.engine")
    if os.path.exists(engine_path):
        print(f"[YOLO] Engine already exists: {engine_path}")
        return engine_path

    print(f"[YOLO] Loading weights: {weights_path}")
    model = YOLO(weights_path)

    print(f"[YOLO] Exporting to TensorRT FP16 (imgsz={imgsz})...")
    t0 = time.time()
    result = model.export(
        format="engine",
        half=half,
        imgsz=imgsz,
        device=0,
        simplify=True,
        workspace=2.0,
    )
    elapsed = time.time() - t0
    print(f"[YOLO] Export complete in {elapsed:.1f}s → {result}")

    # Ultralytics saves the engine next to the weights; move to output dir
    default_engine = weights_path.replace(".pt", ".engine")
    if os.path.exists(default_engine) and default_engine != engine_path:
        os.rename(default_engine, engine_path)
        print(f"[YOLO] Moved to: {engine_path}")

    return engine_path


def export_segformer_trt(
    output_dir: str,
    model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    input_size: tuple = (512, 512),
    half: bool = True,
):
    """Export SegFormer-B0 to ONNX then TensorRT FP16."""
    from trt_segformer import export_segformer_onnx, build_trt_engine

    onnx_path = os.path.join(output_dir, "segformer_b0.onnx")
    engine_path = os.path.join(output_dir, "segformer_b0.engine")

    if os.path.exists(engine_path):
        print(f"[SegFormer] Engine already exists: {engine_path}")
        return engine_path

    # Step 1: Export ONNX
    if not os.path.exists(onnx_path):
        export_segformer_onnx(onnx_path, model_name, input_size)
    else:
        print(f"[SegFormer] ONNX already exists: {onnx_path}")

    # Step 2: Build TRT engine
    build_trt_engine(onnx_path, engine_path, fp16=half, max_workspace_gb=2.0)

    return engine_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO + SegFormer to TensorRT engines"
    )
    parser.add_argument(
        "--models-dir", type=str, default="./models",
        help="Directory to save exported engines"
    )
    parser.add_argument(
        "--yolo-weights", type=str, default="./yolov8n.pt",
        help="Path to YOLOv8n weights"
    )
    parser.add_argument(
        "--seg-size", type=int, nargs=2, default=[512, 512],
        help="SegFormer input size (H W)"
    )
    parser.add_argument(
        "--skip-yolo", action="store_true",
        help="Skip YOLO export"
    )
    parser.add_argument(
        "--skip-segformer", action="store_true",
        help="Skip SegFormer export"
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("=" * 60)
    print("  TensorRT Engine Export for Jetson AGX Orin")
    print("=" * 60)

    # Export YOLO
    if not args.skip_yolo:
        print("\n── YOLO v8n ────────────────────────────────────────")
        yolo_engine = export_yolo_trt(
            args.yolo_weights, args.models_dir
        )
        print(f"  ✓ YOLO engine: {yolo_engine}")

    # Export SegFormer
    if not args.skip_segformer:
        print("\n── SegFormer-B0 ────────────────────────────────────")
        seg_engine = export_segformer_trt(
            args.models_dir,
            input_size=tuple(args.seg_size),
        )
        print(f"  ✓ SegFormer engine: {seg_engine}")

    print("\n" + "=" * 60)
    print("  All exports complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
