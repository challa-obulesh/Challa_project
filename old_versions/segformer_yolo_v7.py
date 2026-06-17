"""
segformer_yolo_v7.py  –  Robot Navigation Visualiser  (v7 FINAL)
=================================================================
Combines:
  • Real SegFormer traversability segmentation (transformers)
  • Real YOLO obstacle detection (ultralytics)
  • v6-proven centroid-based steering (bias-free LEFT / STRAIGHT / RIGHT)
  • IEEE watermark pixel-level erasure (inpainted from surrounding pixels)
  • No hardcoded direction bias anywhere
  • 5-frame rolling average for smooth, jitter-free decisions

Run:
    python3 segformer_yolo_v7.py <input.mp4> <output.mp4> \
        [--seg-model  <hf_model_or_local>] \
        [--yolo-model <weights.pt>]

If models are not provided, the script falls back to the same simulate_seg_mask
used in v6 so it can always produce output.
"""

import cv2
import numpy as np
from collections import deque
import sys
import os
import argparse

# ── optional heavy-model imports (graceful degradation) ──────────────────────
try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image
    HAS_SEG = True
except ImportError:
    HAS_SEG = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (BGR)
# ──────────────────────────────────────────────────────────────────────────────
COL_TRAVERSABLE = (200, 180,   0)   # dark-cyan tint – free road pixels
COL_CORRIDOR    = (  0, 210,   0)   # Green     – corridor overlay
COL_PATH        = (  0, 255, 255)   # Yellow    – navigation path
COL_GOAL        = (  0,   0, 255)   # Red dot   – goal point
COL_BOX_BG      = (  0,   0,   0)   # Black     – info panel bg
COL_STRAIGHT    = (  0, 230,   0)   # Green text
COL_LEFT        = (255, 150,   0)   # Blue text
COL_RIGHT       = (  0, 100, 255)   # Orange-red text
COL_OBSTACLE    = (  0,   0, 220)   # Red       – YOLO bounding boxes

# ──────────────────────────────────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
LOOKAHEAD_FRAC  = 0.35   # lookahead band top boundary  (fraction of H)
LOOKAHEAD_BOT   = 0.65   # lookahead band bottom boundary (fraction of H)
STRAIGHT_THRESH = 0.06   # ±6 % of frame width = "straight"
SMOOTH_WINDOW   = 5      # frames to average steer offset
CORRIDOR_HALF   = 60     # half-width of green corridor in pixels
SEG_FREE_CLASS  = 0      # SegFormer class = traversable / road

# ──────────────────────────────────────────────────────────────────────────────
# IEEE WATERMARK REMOVAL
# The watermark is burned into the bottom ~28 px strip of the source video.
# Strategy: detect its bounding box once, then for every frame overwrite that
# region with a blend of the row just above it (clean road pixels).
# ──────────────────────────────────────────────────────────────────────────────
IEEE_STRIP_H = 28   # height of watermark strip at very bottom of frame

def remove_ieee_watermark(frame):
    """
    Erase the IEEE watermark burned into the bottom rows of the frame.
    Replaces the strip with a smooth clone of the clean pixels just above.
    """
    H, W = frame.shape[:2]
    strip_top = H - IEEE_STRIP_H

    # Source rows: just above the watermark (same height)
    src_top    = strip_top - IEEE_STRIP_H
    src_bottom = strip_top

    if src_top < 0:
        # Frame too small – just black-out the strip
        frame[strip_top:, :] = 0
        return frame

    clean_patch = frame[src_top:src_bottom, :].copy()

    # Gentle vertical gradient so the replacement blends downward
    for i in range(IEEE_STRIP_H):
        alpha = 1.0 - (i / IEEE_STRIP_H) * 0.25   # fade slightly toward bottom
        frame[strip_top + i, :] = (clean_patch[i, :] * alpha).astype(np.uint8)

    return frame


# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTATION  – real model or fallback
# ──────────────────────────────────────────────────────────────────────────────
class Segmenter:
    def __init__(self, model_name=None):
        self.model     = None
        self.processor = None
        self.device    = "cpu"

        if model_name and HAS_SEG:
            try:
                print(f"Loading SegFormer: {model_name}")
                self.processor = SegformerImageProcessor.from_pretrained(model_name)
                self.model     = SegformerForSemanticSegmentation.from_pretrained(model_name)
                self.device    = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device).eval()
                print(f"  SegFormer loaded on {self.device}")
            except Exception as e:
                print(f"  SegFormer load failed ({e}), using fallback")
                self.model = None

    def get_mask(self, frame):
        """Return binary mask: 1 = traversable, 0 = obstacle/unknown."""
        if self.model is not None:
            return self._real_mask(frame)
        return self._simulate_mask(frame)

    def _real_mask(self, frame):
        H, W = frame.shape[:2]
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits          # (1, C, h, w)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()  # (h, w)
        # Resize back to frame size
        pred_resized = cv2.resize(pred.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        mask = (pred_resized == SEG_FREE_CLASS).astype(np.uint8)
        return mask

    def _simulate_mask(self, frame):
        """Fallback: Otsu threshold on bottom half → road mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        roi  = gray[H // 2 :, :]
        _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            thr = (labels == largest).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  kernel)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[H // 2 :, :] = (thr > 0).astype(np.uint8)
        return mask


# ──────────────────────────────────────────────────────────────────────────────
# OBSTACLE DETECTION  – real YOLO or no-op
# ──────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, weights=None):
        self.model = None
        if weights and HAS_YOLO:
            try:
                print(f"Loading YOLO: {weights}")
                self.model = YOLO(weights)
                print("  YOLO loaded")
            except Exception as e:
                print(f"  YOLO load failed ({e}), skipping detections")

    def detect(self, frame):
        """Return list of (x1,y1,x2,y2,conf,cls_name)."""
        if self.model is None:
            return []
        results = self.model(frame, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = results.names[int(box.cls[0])]
            out.append((x1, y1, x2, y2, conf, cls))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION GEOMETRY  (v6-proven logic)
# ──────────────────────────────────────────────────────────────────────────────
def compute_goal_centroid(mask, H, W):
    r_top = int(LOOKAHEAD_FRAC * H)
    r_bot = int(LOOKAHEAD_BOT  * H)
    band  = mask[r_top:r_bot, :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return (r_top + r_bot) // 2, W // 2
    goal_col = int(np.median(xs))
    goal_row = r_top + int(np.median(ys))
    return goal_row, goal_col


def draw_corridor(overlay, mask, goal_col, W):
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)
    corridor_mask = np.zeros_like(mask)
    corridor_mask[:, col_lo:col_hi] = mask[:, col_lo:col_hi]
    overlay[corridor_mask > 0] = COL_CORRIDOR


def draw_bezier_path(frame, start, goal):
    sx, sy = start
    gx, gy = goal
    cx1, cy1 = sx, sy - (sy - gy) // 3
    cx2, cy2 = gx, gy + (sy - gy) // 3
    pts = []
    for t in np.linspace(0, 1, 80):
        t1 = 1 - t
        x  = int(t1**3*sx + 3*t1**2*t*cx1 + 3*t1*t**2*cx2 + t**3*gx)
        y  = int(t1**3*sy + 3*t1**2*t*cy1 + 3*t1*t**2*cy2 + t**3*gy)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], COL_PATH, 3, cv2.LINE_AA)
    cv2.circle(frame, goal, 9, COL_GOAL, -1)
    cv2.circle(frame, goal, 9, (255,255,255), 2)


# ──────────────────────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, decision, smooth_offset, frame_idx, detections):
    H, W = frame.shape[:2]

    # ── Top info panel ────────────────────────────────────────────────────────
    panel_h = 90
    cv2.rectangle(frame, (0, 0), (W, panel_h), COL_BOX_BG, -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (60, 60, 60), 2)

    cv2.putText(frame, "Robot Navigation  |  SegFormer + YOLO",
                (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    col = COL_STRAIGHT if decision == "GO STRAIGHT" else \
          COL_LEFT      if decision == "TURN LEFT"   else COL_RIGHT
    cv2.putText(frame, f"Decision : {decision}",
                (12, 58), cv2.FONT_HERSHEY_DUPLEX, 0.80, col, 2, cv2.LINE_AA)

    pct  = abs(smooth_offset) * 100
    side = "R" if smooth_offset >= 0 else "L"
    obs_txt = f"  |  Obstacles: {len(detections)}" if detections else ""
    cv2.putText(frame,
                f"Steer offset: {pct:.1f}% {side}  |  Frame {frame_idx}{obs_txt}",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)

    # ── Direction arrow (bottom-centre) ──────────────────────────────────────
    cx, cy, r = W // 2, H - 50, 32
    cv2.circle(frame, (cx, cy), r + 4, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), r + 4, (100, 100, 100), 2)
    if decision == "TURN LEFT":
        cv2.arrowedLine(frame, (cx+r-4, cy), (cx-r, cy),
                        COL_LEFT,  4, cv2.LINE_AA, tipLength=0.4)
    elif decision == "TURN RIGHT":
        cv2.arrowedLine(frame, (cx-r+4, cy), (cx+r, cy),
                        COL_RIGHT, 4, cv2.LINE_AA, tipLength=0.4)
    else:
        cv2.arrowedLine(frame, (cx, cy+r-4), (cx, cy-r),
                        COL_STRAIGHT, 4, cv2.LINE_AA, tipLength=0.4)

    # ── YOLO bounding boxes ───────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_OBSTACLE, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_OBSTACLE, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def process_video(input_path, output_path, seg_model=None, yolo_model=None):
    segmenter = Segmenter(seg_model)
    detector  = Detector(yolo_model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    steer_history = deque(maxlen=SMOOTH_WINDOW)
    frame_idx = 0

    print(f"Processing {total} frames  ({W}x{H} @ {fps:.1f} fps)")
    print(f"  SegFormer: {'real model' if segmenter.model else 'simulated fallback'}")
    print(f"  YOLO:      {'real model' if detector.model  else 'disabled'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 0. Remove IEEE watermark (burned into source pixels) ─────────────
        frame = remove_ieee_watermark(frame)

        # ── 1. Segmentation mask ─────────────────────────────────────────────
        mask = segmenter.get_mask(frame)

        # ── 2. YOLO obstacle detections ──────────────────────────────────────
        detections = detector.detect(frame)

        # ── 3. Overlay: traversable (cyan) then corridor (green) ─────────────
        overlay = frame.copy()
        overlay[mask > 0] = COL_TRAVERSABLE
        goal_row, goal_col = compute_goal_centroid(mask, H, W)
        draw_corridor(overlay, mask, goal_col, W)

        # ── 4. Blend overlay with original ───────────────────────────────────
        vis = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

        # ── 5. Navigation path (Bézier to centroid goal) ─────────────────────
        robot_base = (W // 2, H - 55)   # slightly above HUD arrow
        draw_bezier_path(vis, robot_base, (goal_col, goal_row))

        # ── 6. Steer offset (signed, normalised) ─────────────────────────────
        raw_offset = (goal_col - W / 2) / W   # <0 = left,  >0 = right
        steer_history.append(raw_offset)
        smooth_offset = float(np.mean(steer_history))

        # ── 7. Decision ──────────────────────────────────────────────────────
        if abs(smooth_offset) < STRAIGHT_THRESH:
            decision = "GO STRAIGHT"
        elif smooth_offset < 0:
            decision = "TURN LEFT"
        else:
            decision = "TURN RIGHT"

        # ── 8. HUD (panel + arrow + YOLO boxes) ──────────────────────────────
        draw_hud(vis, decision, smooth_offset, frame_idx, detections)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {decision}  offset={smooth_offset:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer+YOLO Robot Navigation v7")
    parser.add_argument("input",  help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--seg-model",  default=None,
                        help="SegFormer HuggingFace model name or local path")
    parser.add_argument("--yolo-model", default=None,
                        help="YOLO weights file (.pt)")
    args = parser.parse_args()
    process_video(args.input, args.output, args.seg_model, args.yolo_model)
