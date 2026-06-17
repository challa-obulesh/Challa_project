"""
segformer_yolo_v10.py  –  Robot Navigation (v10 CORRECTED)
==========================================================
FIXES from v9:
  FIX-1: Path is SHORT — only extends ~15% of frame height ahead from robot
         (not the full lookahead band to the horizon)
  FIX-2: Lookahead band narrowed to 55%–75% of frame (just ahead of robot)
         so centroid tracks the IMMEDIATE road, not distant vanishing point
  FIX-3: Real SegFormer Cityscapes model: class 0 = road (correct for driving)
  FIX-4: Decision based on where the road IS relative to frame centre
         (not optical flow which adds noise on straight roads)
  FIX-5: HUD matches v5 style: SAFE - FORWARD / SAFE - TURN LEFT / SAFE - TURN RIGHT
  FIX-6: Short Bézier path with goal dot close to robot
  FIX-7: Green corridor only in bottom portion of frame (near robot)
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time

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

# ── COLOURS (BGR) ────────────────────────────────────────────────────────────
COL_TRAVERSABLE = (200, 180,   0)   # Cyan-ish
COL_CORRIDOR    = (  0, 200,   0)   # Green
COL_PATH        = (  0, 255, 255)   # Yellow
COL_GOAL        = (  0,   0, 255)   # Red dot
COL_STRAIGHT    = (  0, 230,   0)
COL_LEFT        = (255, 150,   0)
COL_RIGHT       = (  0, 100, 255)
COL_OBSTACLE    = (  0,   0, 220)
COL_HUD_BG      = ( 15,  15,  15)

# ── PARAMETERS ───────────────────────────────────────────────────────────────
# Lookahead band: only the road JUST AHEAD of the robot (55%-75% of frame height)
LOOK_TOP_FRAC   = 0.55     # top of lookahead band (55% down from top)
LOOK_BOT_FRAC   = 0.75     # bottom of lookahead band (75% down from top)
PATH_LENGTH_PX  = 120      # max path length in pixels ahead of robot base
STEER_THRESH    = 0.04     # ±4% = straight
CORRIDOR_HALF   = 55       # half-width of green corridor
EMA_ALPHA       = 0.25     # smoothing
IEEE_STRIP_H    = 28       # watermark strip height

# Cityscapes SegFormer classes that are traversable (road=0, sidewalk=1)
TRAVERSABLE_CLASSES = {0, 1}


def remove_ieee_watermark(frame):
    H, W = frame.shape[:2]
    st = H - IEEE_STRIP_H
    src = st - IEEE_STRIP_H
    if src < 0:
        frame[st:, :] = 0
        return frame
    patch = frame[src:st, :].copy()
    for i in range(IEEE_STRIP_H):
        a = 1.0 - (i / IEEE_STRIP_H) * 0.25
        frame[st + i, :] = (patch[i, :] * a).astype(np.uint8)
    return frame


# ── SEGMENTATION ─────────────────────────────────────────────────────────────
class Segmenter:
    def __init__(self, model_name=None):
        self.model = None
        self.processor = None
        self.device = "cpu"
        if model_name and HAS_SEG:
            try:
                print(f"Loading SegFormer: {model_name}")
                self.processor = SegformerImageProcessor.from_pretrained(model_name)
                self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device).eval()
                print(f"  SegFormer loaded on {self.device}")
            except Exception as e:
                print(f"  SegFormer load failed ({e}), using fallback")
                self.model = None

    def get_mask(self, frame):
        if self.model is not None:
            return self._real_mask(frame)
        return self._simulate_mask(frame)

    def _real_mask(self, frame):
        H, W = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
        pred_resized = cv2.resize(pred.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
        # Multiple traversable classes
        mask = np.zeros((H, W), dtype=np.uint8)
        for cls_id in TRAVERSABLE_CLASSES:
            mask[pred_resized == cls_id] = 1
        return mask

    def _simulate_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        # Use bottom 40% of frame for road detection
        roi_top = int(H * 0.6)
        roi = gray[roi_top:, :]
        _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            thr = (labels == largest).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[roi_top:, :] = (thr > 0).astype(np.uint8)
        return mask


# ── DETECTION ────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, weights=None):
        self.model = None
        if weights and HAS_YOLO:
            try:
                print(f"Loading YOLO: {weights}")
                self.model = YOLO(weights)
                print("  YOLO loaded")
            except Exception as e:
                print(f"  YOLO load failed ({e})")

    def detect(self, frame):
        if self.model is None:
            return []
        results = self.model(frame, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = results.names[int(box.cls[0])]
            out.append((x1, y1, x2, y2, conf, cls))
        return out


# ── NAVIGATION ───────────────────────────────────────────────────────────────
def compute_road_centroid(mask, H, W):
    """
    Find the horizontal centroid of traversable pixels in the NEAR lookahead band.
    This band is 55%-75% of frame height — just ahead of the robot, not the horizon.
    Returns (goal_row, goal_col).
    """
    r_top = int(LOOK_TOP_FRAC * H)
    r_bot = int(LOOK_BOT_FRAC * H)
    band = mask[r_top:r_bot, :]
    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        return (r_top + r_bot) // 2, W // 2
    return r_top + int(np.median(ys)), int(np.median(xs))


def draw_short_path(frame, robot_base, goal_col, goal_row, H, W):
    """
    Draw a SHORT Bézier path from robot base to a point only PATH_LENGTH_PX ahead.
    The goal dot is placed at the actual traversable centroid but clamped close.
    """
    sx, sy = robot_base

    # Clamp goal row so path is SHORT
    max_goal_row = sy - PATH_LENGTH_PX
    actual_goal_row = max(max_goal_row, goal_row)

    gx, gy = goal_col, actual_goal_row

    # Control points for gentle curve
    cx1 = sx + (gx - sx) // 3
    cy1 = sy - (sy - gy) // 3
    cx2 = gx - (gx - sx) // 4
    cy2 = gy + (sy - gy) // 4

    pts = []
    for t in np.linspace(0, 1, 50):
        t1 = 1 - t
        x = int(t1**3*sx + 3*t1**2*t*cx1 + 3*t1*t**2*cx2 + t**3*gx)
        y = int(t1**3*sy + 3*t1**2*t*cy1 + 3*t1*t**2*cy2 + t**3*gy)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], COL_PATH, 3, cv2.LINE_AA)

    # Goal dot at end of short path
    cv2.circle(frame, (gx, gy), 8, COL_GOAL, -1)
    cv2.circle(frame, (gx, gy), 8, (255, 255, 255), 2)


def draw_corridor(overlay, mask, goal_col, H, W):
    """Draw green corridor only in the bottom portion of frame (near robot)."""
    corridor_top = int(LOOK_TOP_FRAC * H)
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)
    corridor_mask = np.zeros_like(mask)
    corridor_mask[corridor_top:, col_lo:col_hi] = mask[corridor_top:, col_lo:col_hi]
    overlay[corridor_mask > 0] = COL_CORRIDOR


def draw_hud(frame, decision, smooth_offset, fps, frame_idx, total, detections):
    """V5-style HUD: status text at top, info bar, direction arrow at bottom."""
    H, W = frame.shape[:2]

    # ── Top panel ────────────────────────────────────────────────────────────
    panel_h = 75
    cv2.rectangle(frame, (0, 0), (W, panel_h), COL_HUD_BG, -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (60, 60, 60), 2)

    # Decision colour
    if "FORWARD" in decision:
        col = COL_STRAIGHT
    elif "LEFT" in decision:
        col = COL_LEFT
    else:
        col = COL_RIGHT

    # Decision text (v5 style: "SAFE - FORWARD")
    cv2.putText(frame, decision, (15, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, col, 2, cv2.LINE_AA)

    # Info line
    pct = abs(smooth_offset) * 100
    side = "R" if smooth_offset >= 0 else "L"
    obs_count = len(detections)
    info = f"Steer: {pct:.1f}% {side}  |  Frame {frame_idx}/{total}  |  FPS: {fps:.1f}  |  Obs: {obs_count}"
    cv2.putText(frame, info, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Direction arrow at bottom ────────────────────────────────────────────
    cx, cy, r = W // 2, H - 40, 28
    cv2.circle(frame, (cx, cy), r + 4, (30, 30, 30), -1)
    cv2.circle(frame, (cx, cy), r + 4, (90, 90, 90), 2)

    if "LEFT" in decision:
        cv2.arrowedLine(frame, (cx+r-4, cy), (cx-r, cy),
                        COL_LEFT, 4, cv2.LINE_AA, tipLength=0.4)
    elif "RIGHT" in decision:
        cv2.arrowedLine(frame, (cx-r+4, cy), (cx+r, cy),
                        COL_RIGHT, 4, cv2.LINE_AA, tipLength=0.4)
    else:
        cv2.arrowedLine(frame, (cx, cy+r-4), (cx, cy-r),
                        COL_STRAIGHT, 4, cv2.LINE_AA, tipLength=0.4)

    # ── YOLO bounding boxes ──────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf, cls) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_OBSTACLE, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1-6, panel_h + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_OBSTACLE, 1, cv2.LINE_AA)


# ── LEGEND ───────────────────────────────────────────────────────────────────
def draw_legend(frame):
    H, W = frame.shape[:2]
    lx, ly = 10, H - 100
    cv2.rectangle(frame, (lx-5, ly-5), (lx+175, ly+65), (0,0,0), -1)
    cv2.rectangle(frame, (lx-5, ly-5), (lx+175, ly+65), (60,60,60), 1)
    cv2.circle(frame, (lx+10, ly+12), 8, COL_TRAVERSABLE, -1)
    cv2.putText(frame, "Traversable", (lx+25, ly+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.circle(frame, (lx+10, ly+32), 8, COL_CORRIDOR, -1)
    cv2.putText(frame, "Corridor", (lx+25, ly+37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.circle(frame, (lx+10, ly+52), 8, COL_OBSTACLE, -1)
    cv2.putText(frame, "Obstacle", (lx+25, ly+57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)


# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def process_video(input_path, output_path, seg_model=None, yolo_model=None):
    segmenter = Segmenter(seg_model)
    detector = Detector(yolo_model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (W, H))

    smooth_offset = 0.0
    frame_idx = 0
    t_start = time.time()

    print(f"Processing {total} frames  ({W}x{H} @ {video_fps:.1f} fps)")
    print(f"  SegFormer: {'real model' if segmenter.model else 'simulated fallback'}")
    print(f"  YOLO:      {'real model' if detector.model else 'disabled'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 0. Remove watermark if present
        frame = remove_ieee_watermark(frame)

        # 1. Segmentation
        mask = segmenter.get_mask(frame)

        # 2. YOLO detections
        detections = detector.detect(frame)

        # 3. Overlay: traversable (cyan) → bottom half only for visibility
        overlay = frame.copy()
        overlay[mask > 0] = COL_TRAVERSABLE

        # 4. Goal centroid in the NEAR lookahead band
        goal_row, goal_col = compute_road_centroid(mask, H, W)

        # 5. Corridor (green) near robot
        draw_corridor(overlay, mask, goal_col, H, W)

        # 6. Blend
        vis = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

        # 7. SHORT navigation path
        robot_base = (W // 2, H - 20)
        draw_short_path(vis, robot_base, goal_col, goal_row, H, W)

        # 8. Steer offset
        raw_offset = (goal_col - W / 2) / W
        if frame_idx == 0:
            smooth_offset = raw_offset
        else:
            smooth_offset = EMA_ALPHA * raw_offset + (1 - EMA_ALPHA) * smooth_offset

        # 9. Decision (v5 style labels)
        if abs(smooth_offset) < STEER_THRESH:
            decision = "SAFE - FORWARD"
        elif smooth_offset < 0:
            decision = "SAFE - TURN LEFT"
        else:
            decision = "SAFE - TURN RIGHT"

        # 10. HUD + Legend
        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / max(elapsed, 0.001)
        draw_hud(vis, decision, smooth_offset, current_fps, frame_idx, total, detections)
        draw_legend(vis)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {decision}  offset={smooth_offset:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone -> {output_path}  ({frame_idx} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer+YOLO Navigation v10")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--seg-model", default=None)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args()
    process_video(args.input, args.output, args.seg_model, args.yolo_model)
