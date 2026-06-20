#!/usr/bin/env python3
"""
Video file runner with full benchmark logging.

Processes video through the Jetson navigation pipeline and outputs:
  - 4-panel grid video (MP4)
  - Benchmark JSON with per-frame timings

Usage:
    python3 run_video.py --input ../00a0f008-a315437f.mov
    python3 run_video.py --input ../00a0f008-a315437f.mov --output out.mp4 --no-display
"""

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jetson_pipeline import NavigationPipeline


def run_video(args):
    """Process a video file through the navigation pipeline."""

    # ── Resolve paths ────────────────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)

    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)

    # Output paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video = args.output or os.path.join(
        project_root, f"out_jetson_{base_name}.mp4"
    )
    benchmark_path = args.benchmark or os.path.join(
        project_root, "src", f"benchmark_{base_name}.json"
    )

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

    # ── Open Input Video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Video] Input: {input_path}")
    print(f"  Resolution: {in_w}×{in_h} @ {in_fps:.1f} FPS, {total_frames} frames")

    # ── Setup Output Writer ──────────────────────────────────────────
    writer = None
    frame_timings = []
    frame_idx = 0

    print(f"[Video] Processing... (max {args.max_frames or total_frames} frames)")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.max_frames and frame_idx >= args.max_frames:
                break

            # Process frame
            result = pipeline.process_frame(frame)
            grid = result["grid"]

            # Initialise writer on first output frame
            if writer is None:
                gh, gw = grid.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = min(in_fps, 30.0)
                writer = cv2.VideoWriter(output_video, fourcc, out_fps, (gw, gh))
                print(f"[Video] Output: {output_video} ({gw}×{gh} @ {out_fps:.0f} FPS)")

            writer.write(grid)

            # Store timing
            timings = result["timings"]
            timings["frame_idx"] = frame_idx
            timings["fps"] = result["fps"]
            timings["num_detections"] = len(result["detections"])
            timings["path_length"] = len(result["path"])
            frame_timings.append(timings)

            # Progress
            if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                fps = result["fps"]
                total_ms = timings["total_ms"]
                print(
                    f"  Frame {frame_idx:4d}/{total_frames} | "
                    f"FPS: {fps:5.1f} | "
                    f"Total: {total_ms:5.1f}ms | "
                    f"YOLO: {timings['yolo_ms']:5.1f}ms | "
                    f"Seg: {timings['segformer_ms']:5.1f}ms | "
                    f"Dets: {timings['num_detections']}"
                )

            # Display (optional)
            if not args.no_display:
                display = cv2.resize(grid, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Jetson Navigation", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[Video] Stopped by user")
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Generate Benchmark Report ────────────────────────────────────
    if frame_timings:
        total_ms_list = [t["total_ms"] for t in frame_timings]
        yolo_ms_list = [t["yolo_ms"] for t in frame_timings]
        seg_ms_list = [t["segformer_ms"] for t in frame_timings]

        benchmark = {
            "input_file": os.path.basename(input_path),
            "input_resolution": f"{in_w}x{in_h}",
            "total_frames_processed": len(frame_timings),
            "summary": {
                "avg_fps": float(np.mean([t["fps"] for t in frame_timings[-30:]])),
                "avg_total_ms": float(np.mean(total_ms_list)),
                "avg_yolo_ms": float(np.mean(yolo_ms_list)),
                "avg_segformer_ms": float(np.mean(seg_ms_list)),
                "min_fps": float(1000.0 / max(total_ms_list)),
                "max_fps": float(1000.0 / min(total_ms_list)),
                "p50_total_ms": float(np.percentile(total_ms_list, 50)),
                "p95_total_ms": float(np.percentile(total_ms_list, 95)),
                "p99_total_ms": float(np.percentile(total_ms_list, 99)),
            },
            "target_fps": 15,
            "target_met": bool(np.mean([t["fps"] for t in frame_timings[-30:]]) >= 15),
            "per_frame": frame_timings,
        }

        with open(benchmark_path, "w") as f:
            json.dump(benchmark, f, indent=2)

        print("\n" + "=" * 60)
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        s = benchmark["summary"]
        print(f"  Frames processed:  {benchmark['total_frames_processed']}")
        print(f"  Average FPS:       {s['avg_fps']:.1f}")
        print(f"  Avg total time:    {s['avg_total_ms']:.1f}ms")
        print(f"  Avg YOLO time:     {s['avg_yolo_ms']:.1f}ms")
        print(f"  Avg SegFormer:     {s['avg_segformer_ms']:.1f}ms")
        print(f"  Min/Max FPS:       {s['min_fps']:.1f} / {s['max_fps']:.1f}")
        print(f"  P50/P95/P99:       {s['p50_total_ms']:.1f} / {s['p95_total_ms']:.1f} / {s['p99_total_ms']:.1f}ms")
        print(f"  Target (>15 FPS):  {'✅ MET' if benchmark['target_met'] else '❌ NOT MET'}")
        print(f"  Benchmark saved:   {benchmark_path}")
        print(f"  Output video:      {output_video}")
        print("=" * 60)

    return benchmark if frame_timings else None


def main():
    parser = argparse.ArgumentParser(description="Jetson Navigation — Video Runner")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", default=None, help="Output video file")
    parser.add_argument("--benchmark", "-b", default=None, help="Benchmark JSON output")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--seg-h", type=int, default=512, help="SegFormer input height")
    parser.add_argument("--seg-w", type=int, default=512, help="SegFormer input width")
    parser.add_argument("--conf-thresh", type=float, default=0.35, help="YOLO confidence")
    parser.add_argument("--panel-scale", type=float, default=0.5, help="Panel scale factor")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display")
    args = parser.parse_args()

    run_video(args)


if __name__ == "__main__":
    main()
