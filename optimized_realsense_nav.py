"""
optimized_realsense_nav.py
==========================
Week 3 + Week 4 + IEEE Extension  –  RealSense Live Navigation
---------------------------------------------------------------
Full pipeline on Intel RealSense D435 / D455:

  RealSense RGB + Depth
        │
        ▼
  LRASPP-MobileNetV3 (TorchScript JIT, FP16, 320×240)
        │
        ▼
  Semantic Segmentation
        │
        ├──► Traversable Mask
        ├──► Obstacle Mask
        │
        ▼
  Traversability Score Map        ← IEEE Novel Module
        │
        ▼
  Semantic Heatmap Overlay
        │
        ▼
  Depth-Fused Safe Zone           ← Depth inflates obstacles
        │
        ▼
  Histogram Navigation Planner    ← Finds widest safe corridor
        │
        ▼
  Navigation State Machine        ← FORWARD / STEER L/R / BRAKE
        │
        ▼
  Performance Benchmarking        ← FPS · Latency · GPU · RAM

Press  Q  to quit.
"""

import os
import time
import cv2
import numpy as np
import torch

os.environ["LC_ALL"] = "C"
os.environ["LANG"]   = "C"

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False
    print("[WARNING] pyrealsense2 not found – running in DEMO mode (webcam fallback).")

from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
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
#  Model – JIT-compiled wrapper for max speed
# ─────────────────────────────────────────────────────

class SegWrapper(torch.nn.Module):
    """Strips the dict wrapper so TorchScript JIT can trace it."""
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, x):
        return self.model(x)["out"]


print("Loading LRASPP-MobileNetV3 (optimised) …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

weights   = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
base_model = lraspp_mobilenet_v3_large(weights=weights).to(device).eval()
wrapped    = SegWrapper(base_model)

if device.type == "cuda":
    wrapped  = wrapped.half()
    dummy    = torch.rand(1, 3, 240, 320).to(device).half()
else:
    dummy    = torch.rand(1, 3, 240, 320).to(device)

print("Compiling TorchScript JIT …")
with torch.no_grad():
    jit_model = torch.jit.trace(wrapped, dummy, strict=False)
print("JIT compiled.")

# ─────────────────────────────────────────────────────
#  Traversability Score Table for COCO (21 classes)
#  Used by LRASPP model (COCO labels differ from ADE20K)
# ─────────────────────────────────────────────────────
COCO_TRAVERSABILITY = {
    0:  1.0,   # background / floor
    15: 0.10,  # person
    2:  0.10,  # bicycle
    7:  0.10,  # car
    14: 0.15,  # motorbike
}


def build_coco_score_map(pred: np.ndarray) -> np.ndarray:
    """Build a traversability score map for COCO-label predictions."""
    score_map = np.full(pred.shape, 0.5, dtype=np.float32)
    for cls_id, score in COCO_TRAVERSABILITY.items():
        score_map[pred == cls_id] = score
    return score_map


# ─────────────────────────────────────────────────────
#  Sensor Setup
# ─────────────────────────────────────────────────────
HEATMAP_ALPHA  = 0.45
OUTPUT_JSON    = "realsense_benchmark.json"

bench = Benchmarker(device_name=str(device))

if _RS_AVAILABLE:
    print("Initialising RealSense …")
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(cfg)
    align       = rs.align(rs.stream.color)
    decimation  = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    print("RealSense ready.")
else:
    cap = cv2.VideoCapture(0)
    print("Using webcam as fallback …")

previous_safe_x = 320

try:
    while True:
        # ── Frame Capture ────────────────────────────────
        if _RS_AVAILABLE:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame    = aligned_frames.get_color_frame()
            depth_frame    = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            depth_frame  = decimation.process(depth_frame)
            frame        = np.asanyarray(color_frame.get_data())
            depth_image  = np.asanyarray(depth_frame.get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                break
            depth_image = None

        h, w = frame.shape[:2]

        bench.start_frame()

        # ── AI Inference (320×240 → upscale) ────────────
        small = cv2.resize(frame, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        if device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            output = jit_model(tensor)[0]

        pred = output.argmax(0).byte().cpu().numpy()
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── Traversability Score Map (IEEE) ─────────────
        score_map = build_coco_score_map(pred)

        traversable_mask = build_traversable_mask(score_map)
        obstacle_mask    = build_obstacle_mask(score_map)
        heatmap          = build_heatmap(score_map)

        # ── Depth Fusion (RealSense only) ────────────────
        if depth_image is not None:
            depth_m    = cv2.resize(
                depth_image * 0.001, (w, h), interpolation=cv2.INTER_NEAREST
            )
            depth_obs  = np.where(
                (depth_m > 0.1) & (depth_m < 1.5), 255, 0
            ).astype(np.uint8)
            depth_obs[:int(h * 0.4), :] = 0
            inflated_obs = cv2.dilate(
                depth_obs, np.ones((11, 11), np.uint8), iterations=1
            )
            traversable_mask[:int(h * 0.4), :] = 0
            traversable_mask[inflated_obs == 255] = 0
        else:
            inflated_obs  = obstacle_mask
            depth_m       = np.zeros((h, w), dtype=np.float32)

        safe_floor = traversable_mask

        # ── Histogram Path Planner ────────────────────────
        nav_zone  = safe_floor[h - 120:h, :]
        histogram = np.sum(nav_zone == 255, axis=0).astype(np.float32)
        histogram = cv2.GaussianBlur(
            histogram.reshape(1, -1), (35, 1), 0
        ).flatten()

        peak_val    = float(np.max(histogram))
        safe_idx    = np.where(histogram > peak_val * 0.55)[0] if peak_val > 0 else []

        if len(safe_idx) > 0:
            segs   = np.split(safe_idx, np.where(np.diff(safe_idx) != 1)[0] + 1)
            best   = max(segs, key=len)
            target_x = int((best[0] + best[-1]) / 2)
        else:
            target_x = w // 2

        smoothed_x      = int(0.75 * previous_safe_x + 0.25 * target_x)
        previous_safe_x = smoothed_x
        centre_dist     = float(depth_m[h - 120, smoothed_x]) if depth_image is not None else 0.0

        # ── Navigation State Machine ──────────────────────
        err = smoothed_x - w // 2
        if   0 < centre_dist < 0.80: decision, col = "EMERGENCY BRAKE",    (0, 0, 255)
        elif err >  55:              decision, col = "STEER RIGHT",          (0, 165, 255)
        elif err < -55:              decision, col = "STEER LEFT",           (0, 165, 255)
        else:                        decision, col = "PATH CLEAR: FORWARD",  (0, 255, 0)

        # ── Benchmark ─────────────────────────────────────
        stats  = score_map_stats(score_map)
        timing = bench.end_frame(stats)
        fps    = round(1000.0 / timing["latency_ms"], 1) if timing["latency_ms"] > 0 else 0

        # ── Visualisation ─────────────────────────────────
        display = overlay_heatmap(frame, heatmap, alpha=HEATMAP_ALPHA)

        green_l = np.zeros_like(frame); green_l[:] = (0, 200, 0)
        red_l   = np.zeros_like(frame); red_l[:]   = (0, 0, 200)

        display = np.where(
            safe_floor[:, :, None] == 255,
            cv2.addWeighted(display, 0.7, green_l, 0.3, 0), display
        )
        display = np.where(
            inflated_obs[:, :, None] == 255,
            cv2.addWeighted(display, 0.8, red_l, 0.2, 0), display
        )

        # Steering guide line
        cv2.line(display, (w // 2, h), (smoothed_x, h - 120), (0, 255, 255), 3)
        cv2.circle(display, (smoothed_x, h - 120), 8, (0, 255, 255), -1)

        # HUD
        hud = [
            (f"Decision : {decision}",                          col),
            (f"FPS      : {fps}",                              (255, 255, 0)),
            (f"Latency  : {timing['latency_ms']:.1f} ms",       (255, 255, 0)),
            (f"Range    : {centre_dist:.2f} m",                 (255, 255, 255)),
            (f"Safe     : {stats['safe_pixel_pct']:.1f}%",      (0, 255, 0)),
            (f"Mean Scr : {stats['mean_score']:.3f}",           (0, 200, 255)),
        ]
        for i, (text, c) in enumerate(hud):
            cv2.putText(display, text, (15, 35 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

        cv2.putText(
            display,
            "RealSense | SegFormer | Traversability Heatmap | IEEE",
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
        )

        cv2.imshow("RealSense Semantic Navigation", display)
        cv2.imshow("Traversability Heatmap",        heatmap)

        if cv2.waitKey(1) == ord("q"):
            break

finally:
    if _RS_AVAILABLE:
        pipeline.stop()
    else:
        cap.release()

    cv2.destroyAllWindows()

    bench.print_summary()
    bench.save(OUTPUT_JSON)
    print("RealSense session ended.")
