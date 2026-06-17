"""
semantic_astar_navigation.py============================
Main Pipeline  –  Video File Mode
-----------------------------------
Full Architecture:
  RGB Camera (video file)
        │
        ▼
  SegFormer-B0  (ADE20K-512)
        │
        ▼
  Semantic Segmentation
        │
        ├──► Traversable Mask   (score ≥ 0.50)
        │
        ├──► Obstacle Mask      (score < 0.25)
        │
        ▼
  Traversability Score Map      ← Novel IEEE contribution
        │
        ▼
  Semantic Heatmap              ← Adaptive colour overlay
        │
        ▼
  Cost-Aware Occupancy Grid
        │
        ▼
  A* Path Planning
        │
        ▼
  Navigation Decision (FORWARD / LEFT / RIGHT / STOP)
        │
        ▼
  Performance Benchmarking      (FPS · Latency · GPU · RAM)

 
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
from occupancy_grid import build_grid, build_cost_grid
from astar_planner  import astar_cost
from benchmarker    import Benchmarker

# ─────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────
VIDEO_PATH   = "/home/sdv/zed_navigation_3min.mp4"
MODEL_NAME   = "nvidia/segformer-b0-finetuned-ade-512-512"
GRID_SIZE    = 10          # pixels per grid cell
FRAME_SKIP   = 3           # process every Nth frame for speed
HEATMAP_ALPHA = 0.45       # heatmap transparency over raw frame
SAVE_EVERY   = 30          # save a debug frame every N processed frames
OUTPUT_JSON  = "seg_benchmark.json"

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
print("Model loaded.")

# ─────────────────────────────────────────────────────
#  Open Video
# ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

print(f"Opened video: {VIDEO_PATH}")

bench = Benchmarker(device_name=str(device))

frame_id       = 0
processed_id   = 0

print("Starting pipeline …\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ── Frame skip for real-time FPS ────────────────
    if frame_id % FRAME_SKIP != 0:
        continue

    processed_id += 1
    h, w = frame.shape[:2]

    bench.start_frame()

    # ─────────────────────────────────────────────
    #  STEP 1 │ SegFormer Inference
    # ─────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Upscale logits → full resolution label map
    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # ─────────────────────────────────────────────
    #  STEP 2 │ Traversability Score Map  (IEEE)
    # ─────────────────────────────────────────────
    score_map = build_score_map(seg_map)

    # ─────────────────────────────────────────────
    #  STEP 3 │ Binary Masks
    # ─────────────────────────────────────────────
    traversable_mask = build_traversable_mask(score_map)
    obstacle_mask    = build_obstacle_mask(score_map)

    # ─────────────────────────────────────────────
    #  STEP 4 │ Semantic Heatmap  (IEEE)
    # ─────────────────────────────────────────────
    heatmap = build_heatmap(score_map)

    # ─────────────────────────────────────────────
    #  STEP 5 │ Cost-Aware Occupancy Grid
    # ─────────────────────────────────────────────
    cost_grid = build_cost_grid(score_map, grid_size=GRID_SIZE)
    bin_grid  = build_grid(score_map,      grid_size=GRID_SIZE)

    rows, cols = bin_grid.shape

    # ─────────────────────────────────────────────
    #  STEP 6 │ A* Path Planning
    # ─────────────────────────────────────────────
    # Start: bottom rows of grid  (robot position)
    start = None
    for r in range(rows - 1, max(rows - 8, 0), -1):
        for c in range(cols):
            if bin_grid[r, c] == 0:
                start = (r, c)
                break
        if start:
            break

    # Goal: top rows of grid  (navigation target)
    goal = None
    for r in range(0, min(8, rows)):
        for c in range(cols):
            if bin_grid[r, c] == 0:
                goal = (r, c)
                break
        if goal:
            break

    path = None
    if start and goal:
        path = astar_cost(cost_grid, start, goal)

    # ─────────────────────────────────────────────
    #  STEP 7 │ Navigation Decision
    # ─────────────────────────────────────────────
    decision = "STOP"
    if path and len(path) > 5:
        dx = path[5][1] - path[0][1]
        if   dx < -2: decision = "LEFT"
        elif dx >  2: decision = "RIGHT"
        else:         decision = "FORWARD"

    # ─────────────────────────────────────────────
    #  STEP 8 │ End-frame benchmark
    # ─────────────────────────────────────────────
    stats  = score_map_stats(score_map)
    timing = bench.end_frame(stats)

    fps_live = round(1000.0 / timing["latency_ms"], 1) if timing["latency_ms"] > 0 else 0

    print(
        f"Frame {frame_id:5d} | Proc #{processed_id:4d} | "
        f"{timing['latency_ms']:6.1f} ms | {fps_live:5.1f} FPS | "
        f"Decision: {decision:<8s} | "
        f"Safe: {stats['safe_pixel_pct']:5.1f}%"
    )

    # ─────────────────────────────────────────────
    #  STEP 9 │ Visualisation
    # ─────────────────────────────────────────────
    # a) Heatmap overlay on raw frame
    vis_heatmap = overlay_heatmap(frame, heatmap, alpha=HEATMAP_ALPHA)

    # b) Green tint for traversable, red tint for obstacle
    green_layer = np.zeros_like(frame); green_layer[:] = (0, 200, 0)
    red_layer   = np.zeros_like(frame); red_layer[:]   = (0, 0, 200)

    vis_heatmap = np.where(
        traversable_mask[:, :, None] == 255,
        cv2.addWeighted(vis_heatmap, 0.8, green_layer, 0.2, 0),
        vis_heatmap,
    )
    vis_heatmap = np.where(
        obstacle_mask[:, :, None] == 255,
        cv2.addWeighted(vis_heatmap, 0.8, red_layer, 0.2, 0),
        vis_heatmap,
    )

    # c) Draw A* path
    if path:
        for pr, pc in path:
            px = pc * GRID_SIZE + GRID_SIZE // 2
            py = pr * GRID_SIZE + GRID_SIZE // 2
            cv2.circle(vis_heatmap, (px, py), 3, (255, 0, 255), -1)

    # d) Traversable / Obstacle binary masks (side panels)
    trav_bgr = cv2.cvtColor(traversable_mask, cv2.COLOR_GRAY2BGR)
    obs_bgr  = cv2.cvtColor(obstacle_mask,    cv2.COLOR_GRAY2BGR)

    # Resize panels to same height for concatenation
    panel_w = w // 3
    trav_panel = cv2.resize(trav_bgr, (panel_w, h // 2))
    obs_panel  = cv2.resize(obs_bgr,  (panel_w, h // 2))

    cv2.putText(trav_panel, "Traversable Mask", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(obs_panel,  "Obstacle Mask",    (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # e) HUD text on main display
    hud_lines = [
        f"Decision : {decision}",
        f"FPS      : {fps_live}",
        f"Latency  : {timing['latency_ms']:.1f} ms",
        f"Safe     : {stats['safe_pixel_pct']:.1f}%",
        f"Obstacle : {stats['obstacle_pixel_pct']:.1f}%",
        f"Mean Scr : {stats['mean_score']:.3f}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(
            vis_heatmap, line,
            (15, 35 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2,
        )

    cv2.putText(
        vis_heatmap,
        "SegFormer-B0 | Traversability Heatmap (IEEE)",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1,
    )

    # ─────────────────────────────────────────────
    #  STEP 10 │ Save debug frames
    # ─────────────────────────────────────────────
    if processed_id % SAVE_EVERY == 0:
        cv2.imwrite(f"out_frame_{frame_id:05d}.jpg", vis_heatmap)
        cv2.imwrite(f"heatmap_{frame_id:05d}.jpg",   heatmap)
        print(f"  Saved debug frames for frame {frame_id}")

cap.release()

# ─────────────────────────────────────────────────────
#  Final Benchmark Report
# ─────────────────────────────────────────────────────
bench.print_summary()
bench.save(OUTPUT_JSON)

print("Pipeline complete.")
