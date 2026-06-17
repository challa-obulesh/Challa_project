"""
run_video.py
============
Week 1 + Week 4  –  Video File Pipeline with Benchmarking
-----------------------------------------------------------
Runs the full pipeline on any MP4 / AVI video file and saves:
  • Annotated output video  (out_traversability.mp4)
  • Per-frame heatmap video (out_heatmap.mp4)
  • Benchmark JSON          (video_benchmark.json)

Usage:
    python run_video.py
    python run_video.py --video /path/to/file.mp4
    python run_video.py --video /path/to/file.mp4 --skip 2
"""

import argparse
import cv2
import torch
import numpy as np
import os

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from traversability_scorer import (
    build_score_map,
    build_traversable_mask,
    build_obstacle_mask,
    build_heatmap,
    overlay_heatmap,
    score_map_stats,
)
from benchmarker import Benchmarker

# ─────────────────────────────────────────────────────
#  CLI Arguments
# ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="SegFormer traversability video pipeline"
)
parser.add_argument(
    "--video", default="/home/sdv/zed_navigation_3min.mp4",
    help="Path to input video file"
)
parser.add_argument(
    "--skip", type=int, default=3,
    help="Process every Nth frame (default 3)"
)
parser.add_argument(
    "--alpha", type=float, default=0.45,
    help="Heatmap overlay alpha (0-1)"
)
parser.add_argument(
    "--dataset", type=str, default="ade20k", choices=["ade20k", "cityscapes"],
    help="Which model and score map to use"
)
args = parser.parse_args()

VIDEO_PATH    = args.video
FRAME_SKIP    = args.skip
HEATMAP_ALPHA = args.alpha
DATASET       = args.dataset

if DATASET == "cityscapes":
    MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
else:
    MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_TRAV   = f"{base_name}_{DATASET}_traversability.mp4"
OUTPUT_HEAT   = f"{base_name}_{DATASET}_heatmap.mp4"
OUTPUT_JSON   = f"{base_name}_{DATASET}_benchmark.json"

# ─────────────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────────────
print("Loading SegFormer-B0 …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME, local_files_only=False
)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME, local_files_only=False
)
model.to(device).eval()
print("Model loaded.")

# ─────────────────────────────────────────────────────
#  Open Video + Writers
# ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
writer_main    = cv2.VideoWriter(OUTPUT_TRAV, fourcc,
                                 fps_src / FRAME_SKIP, (w_src, h_src))
writer_heatmap = cv2.VideoWriter(OUTPUT_HEAT, fourcc,
                                 fps_src / FRAME_SKIP, (w_src, h_src))

bench     = Benchmarker(device_name=str(device))
frame_id  = 0
proc_id   = 0

print(f"\nProcessing: {VIDEO_PATH}")
print(f"Frame skip : {FRAME_SKIP}  (processing every {FRAME_SKIP}th frame)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    proc_id += 1
    h, w = frame.shape[:2]

    bench.start_frame()

    # ── Inference ────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Traversability Pipeline ───────────────────────
    score_map        = build_score_map(seg_map, dataset=DATASET)
    traversable_mask = build_traversable_mask(score_map)
    obstacle_mask    = build_obstacle_mask(score_map)
    heatmap          = build_heatmap(score_map)

    stats  = score_map_stats(score_map)
    timing = bench.end_frame(stats)
    fps_live = round(1000.0 / timing["latency_ms"], 1) if timing["latency_ms"] > 0 else 0.0

    print(
        f"Frame {frame_id:5d} | "
        f"{timing['latency_ms']:6.1f} ms | "
        f"{fps_live:5.1f} FPS | "
        f"Safe {stats['safe_pixel_pct']:5.1f}% | "
        f"Obs {stats['obstacle_pixel_pct']:5.1f}%"
    )

    # ── Visualisation ────────────────────────────────
    display = overlay_heatmap(frame, heatmap, alpha=HEATMAP_ALPHA)

    green_l = np.zeros_like(frame); green_l[:] = (0, 200, 0)
    red_l   = np.zeros_like(frame); red_l[:]   = (0, 0, 200)

    display = np.where(
        traversable_mask[:, :, None] == 255,
        cv2.addWeighted(display, 0.85, green_l, 0.15, 0), display
    )
    display = np.where(
        obstacle_mask[:, :, None] == 255,
        cv2.addWeighted(display, 0.85, red_l, 0.15, 0), display
    )

    hud = [
        f"FPS: {fps_live}  |  Latency: {timing['latency_ms']:.1f} ms",
        f"Safe: {stats['safe_pixel_pct']:.1f}%  |  Obs: {stats['obstacle_pixel_pct']:.1f}%  |  Score: {stats['mean_score']:.3f}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(display, line, (12, 32 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Add legend
    cv2.rectangle(display, (10, h - 90), (200, h - 25), (0, 0, 0), -1)
    cv2.circle(display, (25, h - 70), 8, (0, 200, 0), -1)
    cv2.putText(display, "Traversable", (40, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(display, (25, h - 45), 8, (0, 0, 200), -1)
    cv2.putText(display, "Obstacle", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(display,
                "SegFormer-B0 | Semantic Traversability | IEEE",
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    writer_main.write(display)
    writer_heatmap.write(heatmap)

cap.release()
writer_main.release()
writer_heatmap.release()

bench.print_summary()
bench.save(OUTPUT_JSON)

print("\nOutput videos:")
print(f"  {OUTPUT_TRAV}")
print(f"  {OUTPUT_HEAT}")
print(f"  {OUTPUT_JSON}")
print("Done.")
