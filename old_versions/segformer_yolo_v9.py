"""
segformer_yolo_v9.py  –  Robot Navigation Visualiser (v9 FINAL)
=================================================================
Combines:
  • Real SegFormer / Fallback Segmentation
  • Real YOLO / Fallback Detection
  • 65% traversable centroid + 35% optical flow lateral offset
  • EMA smoothed decision offset
  • 5-class steering: HARD LEFT | LEFT | STRAIGHT | RIGHT | HARD RIGHT
  • Compact 52-px HUD bar with mini offset gauge
  • Dynamic Bézier curve control points tracking real centroid
  • Pixel-level IEEE watermark erasure
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time

# ── optional heavy-model imports ──────────────────────
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
COL_TRAVERSABLE = (200, 180,   0)
COL_CORRIDOR    = (  0, 210,   0)
COL_PATH        = (  0, 255, 255)
COL_GOAL        = (  0,   0, 255)
COL_BOX_BG      = (  15,  15,  15)
COL_STRAIGHT    = (  0, 230,   0)
COL_LEFT        = (255, 150,   0)
COL_RIGHT       = (  0, 100, 255)
COL_OBSTACLE    = (  0,   0, 220)

# ──────────────────────────────────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
LOOKAHEAD_TOP   = 0.15   # 15% to 50% band
LOOKAHEAD_BOT   = 0.50
SOFT_THRESH     = 0.05
HARD_THRESH     = 0.15
EMA_ALPHA       = 0.20
CORRIDOR_HALF   = 60
SEG_FREE_CLASS  = 0
IEEE_STRIP_H    = 28


def remove_ieee_watermark(frame):
    H, W = frame.shape[:2]
    strip_top = H - IEEE_STRIP_H
    src_top = strip_top - IEEE_STRIP_H
    src_bottom = strip_top
    if src_top < 0:
        frame[strip_top:, :] = 0
        return frame
    clean_patch = frame[src_top:src_bottom, :].copy()
    for i in range(IEEE_STRIP_H):
        alpha = 1.0 - (i / IEEE_STRIP_H) * 0.25
        frame[strip_top + i, :] = (clean_patch[i, :] * alpha).astype(np.uint8)
    return frame

# ──────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW
# ──────────────────────────────────────────────────────────────────────────────
class FlowEstimator:
    def __init__(self):
        self.prev_gray = None

    def get_offset(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        W = gray.shape[1]
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
            
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=10)
        flow_offset = 0.0
        
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                if len(good_new) > 0:
                    dxs = good_new[:, 0] - good_old[:, 0]
                    mean_dx = np.median(dxs)
                    # Robot turns left -> pixels move right (dx > 0) -> negative offset
                    flow_offset = - (mean_dx / (W * 0.1))
                    flow_offset = max(-0.5, min(0.5, flow_offset))
                    
        self.prev_gray = gray
        return flow_offset

# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTATION & YOLO
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
        if self.model is not None: return self._real_mask(frame)
        return self._simulate_mask(frame)

    def _real_mask(self, frame):
        H, W = frame.shape[:2]
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
        pred_resized = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return (pred_resized == SEG_FREE_CLASS).astype(np.uint8)

    def _simulate_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        roi  = gray[H // 2 :, :]
        _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        if self.model is None: return []
        results = self.model(frame, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = results.names[int(box.cls[0])]
            out.append((x1, y1, x2, y2, conf, cls))
        return out

# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def compute_goal_centroid(mask, H, W):
    r_top = int(LOOKAHEAD_TOP * H)
    r_bot = int(LOOKAHEAD_BOT * H)
    band  = mask[r_top:r_bot, :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return (r_top + r_bot) // 2, W // 2
    return r_top + int(np.median(ys)), int(np.median(xs))

def draw_corridor(overlay, mask, goal_col, W):
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)
    corridor_mask = np.zeros_like(mask)
    corridor_mask[:, col_lo:col_hi] = mask[:, col_lo:col_hi]
    overlay[corridor_mask > 0] = COL_CORRIDOR

def draw_bezier_path(frame, start, goal, smooth_offset):
    sx, sy = start
    gx, gy = goal
    W = frame.shape[1]
    
    # Control point shifts proportionally to offset for physical bending
    cx1 = sx + int(smooth_offset * W * 1.5)
    cy1 = sy - (sy - gy) // 3
    cx2 = gx - int(smooth_offset * W * 0.5)
    cy2 = gy + (sy - gy) // 3
    
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

def draw_hud(frame, decision, smooth_offset, fps, detections):
    H, W = frame.shape[:2]
    bar_h = 52
    
    cv2.rectangle(frame, (0, 0), (W, bar_h), COL_BOX_BG, -1)
    cv2.line(frame, (0, bar_h), (W, bar_h), (60, 60, 60), 2)
    
    # Decision badge
    if "HARD RIGHT" in decision: col = (0, 0, 255)
    elif "HARD LEFT" in decision: col = (255, 0, 0)
    elif "STRAIGHT" in decision: col = COL_STRAIGHT
    elif "LEFT" in decision: col = COL_LEFT
    else: col = COL_RIGHT
        
    cv2.putText(frame, f"[{decision}]", (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2, cv2.LINE_AA)
    
    # Mini offset gauge
    gauge_w = 160
    gauge_cx = W // 2
    gauge_cy = bar_h // 2
    cv2.line(frame, (gauge_cx - gauge_w//2, gauge_cy), (gauge_cx + gauge_w//2, gauge_cy), (100, 100, 100), 2)
    cv2.line(frame, (gauge_cx, gauge_cy - 8), (gauge_cx, gauge_cy + 8), (150, 150, 150), 2)
    
    # Needle
    needle_x = gauge_cx + int(smooth_offset * (gauge_w // 2) / 0.3)
    needle_x = max(gauge_cx - gauge_w//2, min(gauge_cx + gauge_w//2, needle_x))
    cv2.circle(frame, (needle_x, gauge_cy), 7, col, -1)
    cv2.circle(frame, (needle_x, gauge_cy), 4, (255, 255, 255), -1)
    
    # FPS and Obstacles
    obs_txt = f"OBS: {len(detections)}" if detections else ""
    info_txt = f"FPS: {fps:.1f} | {obs_txt}"
    cv2.putText(frame, info_txt, (W - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # YOLO bounding boxes in the mid-frame ROI
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_OBSTACLE, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1-6, bar_h+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_OBSTACLE, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def process_video(input_path, output_path, seg_model=None, yolo_model=None):
    segmenter = Segmenter(seg_model)
    detector  = Detector(yolo_model)
    flow_est  = FlowEstimator()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, video_fps, (W, H))

    smooth_offset = 0.0
    frame_idx = 0
    t_start = time.time()

    print(f"Processing {total} frames  ({W}x{H} @ {video_fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = remove_ieee_watermark(frame)
        mask = segmenter.get_mask(frame)
        detections = detector.detect(frame)
        flow_off = flow_est.get_offset(frame)

        overlay = frame.copy()
        overlay[mask > 0] = COL_TRAVERSABLE
        goal_row, goal_col = compute_goal_centroid(mask, H, W)
        draw_corridor(overlay, mask, goal_col, W)

        vis = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

        # Signal blending
        centroid_off = (goal_col - W / 2) / W
        raw_offset = 0.65 * centroid_off + 0.35 * flow_off

        # EMA
        if frame_idx == 0: smooth_offset = raw_offset
        else: smooth_offset = EMA_ALPHA * raw_offset + (1 - EMA_ALPHA) * smooth_offset

        # 5-class Decision
        abs_off = abs(smooth_offset)
        if abs_off < SOFT_THRESH:
            decision = "STRAIGHT"
        elif abs_off < HARD_THRESH:
            decision = "LEFT" if smooth_offset < 0 else "RIGHT"
        else:
            decision = "HARD LEFT" if smooth_offset < 0 else "HARD RIGHT"

        # Draws
        robot_base = (W // 2, H - 20)
        draw_bezier_path(vis, robot_base, (goal_col, goal_row), smooth_offset)
        
        # Calculate processing FPS
        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / elapsed
        
        draw_hud(vis, decision, smooth_offset, current_fps, detections)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {decision}  offset={smooth_offset:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--seg-model", default=None)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args()
    process_video(args.input, args.output, args.seg_model, args.yolo_model)
