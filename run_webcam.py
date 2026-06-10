"""
run_webcam.py
=============
Week 1  –  Live Webcam Pipeline
---------------------------------
Full real-time pipeline on a USB / built-in webcam:
  Webcam → SegFormer-B0 → Score Map → Masks → Heatmap → HUD

Press  Q  to quit.
Press  S  to save the current frame.
Press  B  to print a live benchmark snapshot.
"""

import cv2
import torch
import numpy as np
import time

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
#  Configuration
# ─────────────────────────────────────────────────────
CAMERA_INDEX   = 0
MODEL_NAME     = "nvidia/segformer-b0-finetuned-ade-512-512"
HEATMAP_ALPHA  = 0.50
INFER_SIZE     = (512, 512)   # resize input for faster inference
OUTPUT_JSON    = "webcam_benchmark.json"

# ─────────────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────────────
print("Loading SegFormer-B0 …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME, local_files_only=True
)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME, local_files_only=True
)
model.to(device).eval()
print("Model loaded. Opening webcam …")

# ─────────────────────────────────────────────────────
#  Open Webcam
# ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam index {CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

bench      = Benchmarker(device_name=str(device))
save_count = 0

print("\nControls:  Q = quit | S = save frame | B = benchmark snapshot\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam frame read failed.")
        break

    h, w = frame.shape[:2]

    bench.start_frame()

    # ── SegFormer Inference ──────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs  = processor(images=rgb, return_tensors="pt")
    inputs  = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # ── Traversability Score Map (IEEE) ─────────────
    score_map        = build_score_map(seg_map)
    traversable_mask = build_traversable_mask(score_map)
    obstacle_mask    = build_obstacle_mask(score_map)
    heatmap          = build_heatmap(score_map)

    # ── Benchmark ────────────────────────────────────
    stats  = score_map_stats(score_map)
    timing = bench.end_frame(stats)
    fps_live = round(1000.0 / timing["latency_ms"], 1) if timing["latency_ms"] > 0 else 0.0

    # ── Visualisation ────────────────────────────────
    display = overlay_heatmap(frame, heatmap, alpha=HEATMAP_ALPHA)

    # Green = traversable, Red = obstacle tint
    green_l = np.zeros_like(frame); green_l[:] = (0, 200, 0)
    red_l   = np.zeros_like(frame); red_l[:]   = (0, 0, 200)

    display = np.where(
        traversable_mask[:, :, None] == 255,
        cv2.addWeighted(display, 0.85, green_l, 0.15, 0),
        display,
    )
    display = np.where(
        obstacle_mask[:, :, None] == 255,
        cv2.addWeighted(display, 0.85, red_l, 0.15, 0),
        display,
    )

    # HUD
    hud = [
        f"FPS      : {fps_live}",
        f"Latency  : {timing['latency_ms']:.1f} ms",
        f"Safe     : {stats['safe_pixel_pct']:.1f}%",
        f"Obstacle : {stats['obstacle_pixel_pct']:.1f}%",
        f"Uncertain: {stats['uncertain_pixel_pct']:.1f}%",
        f"Mean Scr : {stats['mean_score']:.3f}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(display, line, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    cv2.putText(
        display,
        "SegFormer-B0 | Traversability Heatmap | IEEE",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    cv2.imshow("Live Webcam – Semantic Traversability", display)
    cv2.imshow("Traversability Heatmap",                heatmap)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        save_count += 1
        fname = f"webcam_save_{save_count:03d}.jpg"
        cv2.imwrite(fname, display)
        print(f"[SAVED] {fname}")
    elif key == ord("b"):
        bench.print_summary()

cap.release()
cv2.destroyAllWindows()

bench.print_summary()
bench.save(OUTPUT_JSON)
print("Webcam session ended.")
