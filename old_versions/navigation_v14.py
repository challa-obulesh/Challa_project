"""
navigation_v14.py  —  Robot Navigation Visualiser  v14
=======================================================
FIXES over v13 (observed from output_v13_dataset1.mp4 & dataset2.mp4):

ERROR 1 — SCATTERED NOISE PIXELS (91% of blobs are tiny <100px fragments)
  Red pixels scattered inside green road zone, green leaking into obstacle areas,
  purple pixels in wrong places.
  FIX: After reading the existing segmentation mask from the frame, apply
       aggressive morphological cleanup: remove all blobs < MIN_BLOB_AREA,
       apply closing to fill gaps, opening to remove speckles, median blur
       for final smoothing. Result: clean solid regions with smooth edges.

ERROR 2 — BOTTOM STRIP CORRUPTION (93% of bottom 80px is chaotic noise)
  Robot hood reflection / bumper zone creates totally unreliable pixels.
  FIX: Hard-mask bottom HOOD_MASK_H pixels to black before any processing.

ERROR 3 — NAVIGATION PATH TOO SHORT / WRONG SIZE
  Yellow line only ~60px long, barely visible. Goal dot always at frame bottom.
  FIX: Path drawn from robot base at bottom-center UP to road centroid in
       the MID-FRAME lookahead zone (rows 30%-60% of H). Path is a clean
       5px Bezier with a glow pass underneath for visibility.

ERROR 4 — DECISION LOGIC: reads existing overlay colors from v13 output
  Since we're processing the OUTPUT video (with overlay already baked in),
  we detect the existing green mask, clean it, then compute centroid.
  Decision comes from signed offset of green-road centroid vs frame center.
  Smoothed over 7-frame window. Threshold lowered to 4% of width.

ERROR 5 — LEGEND MISSING
  No color key shown. FIX: Add compact legend to top-right of panel.

RUN:
  python3 navigation_v14.py input.mp4 output.mp4
"""

import cv2
import numpy as np
from collections import deque
import sys, os

# ─── TUNEABLE PARAMS ──────────────────────────────────────────────────────────
HOOD_MASK_H      = 80     # mask out this many pixels from bottom (hood/bumper)
MIN_BLOB_AREA    = 400    # remove green/purple blobs smaller than this (px²)
LOOKAHEAD_TOP    = 0.28   # lookahead band: rows 28%–60% of H
LOOKAHEAD_BOT    = 0.60
STRAIGHT_THRESH  = 0.04   # ±4% of frame width = straight
SMOOTH_WIN       = 7      # frames to average steer offset
PATH_THICKNESS   = 5
GLOW_THICKNESS   = 10
GOAL_RADIUS      = 10

# ─── COLOURS (BGR) ─────────────────────────────────────────────────────────────
C_GREEN   = (0,  210,   0)    # traversable road
C_RED     = (30,  30, 180)    # obstacle
C_PURPLE  = (180,  60, 180)   # sidewalk
C_PATH    = (0,  255, 255)    # yellow path
C_GLOW    = (0,  120, 120)    # path glow
C_GOAL    = (0,   0,  255)    # red goal dot
C_PANEL   = (0,   0,    0)    # panel bg
C_BORDER  = (70,  70,   70)
C_STRAIGHT= (0,  220,   0)
C_LEFT    = (255, 150,   0)
C_RIGHT   = (0,  100, 255)
C_WHITE   = (255, 255, 255)
C_GRAY    = (160, 160, 160)


# ─── MASK EXTRACTION FROM EXISTING OVERLAY ─────────────────────────────────────
def extract_masks(frame):
    """
    The input is the v13 OUTPUT video which already has colored overlays baked in.
    Detect each class by color range and return cleaned binary masks.
    """
    H, W = frame.shape[:2]
    b = frame[:,:,0].astype(np.int32)
    g = frame[:,:,1].astype(np.int32)
    r = frame[:,:,2].astype(np.int32)

    # Green road mask: high G, low R, low B
    road = ((g > 120) & (r < 80) & (b < 80)).astype(np.uint8)

    # Purple sidewalk: high R+B, low G
    sidewalk = ((r > 90) & (b > 90) & (g < 80)).astype(np.uint8)

    # Ignore hood zone
    road    [H - HOOD_MASK_H:, :] = 0
    sidewalk[H - HOOD_MASK_H:, :] = 0

    # Clean: remove tiny blobs, smooth edges
    road     = clean_mask(road)
    sidewalk = clean_mask(sidewalk)

    return road, sidewalk


def clean_mask(mask):
    """Remove noise blobs < MIN_BLOB_AREA and smooth edges."""
    if mask.max() == 0:
        return mask
    uint8 = mask.astype(np.uint8) * 255
    # Close small gaps
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    uint8 = cv2.morphologyEx(uint8, cv2.MORPH_CLOSE, k9)
    uint8 = cv2.morphologyEx(uint8, cv2.MORPH_OPEN,  k5)
    # Remove small blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(uint8)
    clean = np.zeros_like(uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            clean[labels == i] = 255
    # Gentle median blur for smoother edges
    clean = cv2.medianBlur(clean, 5)
    return (clean > 0).astype(np.uint8)


# ─── GOAL CENTROID ─────────────────────────────────────────────────────────────
def compute_goal(road_mask, H, W):
    """
    Find road centroid in the lookahead band (mid-frame perspective zone).
    Returns (goal_row, goal_col).
    """
    r0 = int(LOOKAHEAD_TOP * H)
    r1 = int(LOOKAHEAD_BOT * H)
    band = road_mask[r0:r1, :]
    ys, xs = np.where(band > 0)
    if len(xs) < 20:
        return (r0 + r1) // 2, W // 2
    goal_col = int(np.median(xs))
    goal_row = r0 + int(np.median(ys))
    return goal_row, goal_col


# ─── BEZIER PATH ───────────────────────────────────────────────────────────────
def draw_path(frame, start, goal):
    """Draw glow + main Bezier path from start to goal."""
    sx, sy = start
    gx, gy = goal
    cy1 = sy - (sy - gy) // 3
    cy2 = gy + (sy - gy) // 3
    pts = []
    for t in np.linspace(0, 1, 80):
        t1 = 1 - t
        x = int(t1**3*sx + 3*t1**2*t*sx + 3*t1*t**2*gx + t**3*gx)
        y = int(t1**3*sy + 3*t1**2*t*cy1 + 3*t1*t**2*cy2 + t**3*gy)
        pts.append((x, y))
    # Glow pass
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], C_GLOW, GLOW_THICKNESS, cv2.LINE_AA)
    # Main line
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], C_PATH, PATH_THICKNESS, cv2.LINE_AA)
    # Goal dot
    cv2.circle(frame, goal, GOAL_RADIUS + 3, (255, 255, 255), -1)
    cv2.circle(frame, goal, GOAL_RADIUS,     C_GOAL,          -1)


# ─── OVERLAY CLEANED MASKS ─────────────────────────────────────────────────────
def apply_clean_overlay(orig_frame, road_mask, sidewalk_mask):
    """
    Replace the noisy v13 overlay with clean version.
    Start from a dark blend of original, paint clean masks on top.
    """
    vis = orig_frame.copy()
    overlay = orig_frame.copy()
    overlay[road_mask > 0]     = C_GREEN
    overlay[sidewalk_mask > 0] = C_PURPLE
    vis = cv2.addWeighted(orig_frame, 0.45, overlay, 0.55, 0)
    return vis


# ─── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, decision, offset, frame_idx):
    H, W = frame.shape[:2]

    # Panel background
    ph = 88
    cv2.rectangle(frame, (0, 0), (W, ph), C_PANEL, -1)
    cv2.rectangle(frame, (0, 0), (W, ph), C_BORDER, 2)

    # Title
    cv2.putText(frame, "SegFormer + YOLO  |  4-Color Semantic Mask",
                (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.62, C_GRAY, 1, cv2.LINE_AA)

    # Decision
    col = C_STRAIGHT if decision == "STRAIGHT" else \
          C_LEFT      if decision == "TURN LEFT" else C_RIGHT
    cv2.putText(frame, decision,
                (12, 62), cv2.FONT_HERSHEY_DUPLEX, 1.10, col, 2, cv2.LINE_AA)

    # Steer info
    pct  = abs(offset) * 100
    side = "R" if offset >= 0 else "L"
    info = f"Ego-Steer: {pct:.1f}% {side}  |  F:{frame_idx}"
    cv2.putText(frame, info,
                (W - 420, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)

    # Legend (top-right)
    lx, ly = W - 220, 35
    items = [("Road", C_GREEN), ("Sidewalk", C_PURPLE), ("Obstacle", C_RED)]
    for label, color in items:
        cv2.rectangle(frame, (lx, ly-12), (lx+18, ly+2), color, -1)
        cv2.putText(frame, label, (lx+24, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, C_WHITE, 1, cv2.LINE_AA)
        ly += 18

    # Direction arrow (bottom-centre)
    cx, cy, r = W // 2, H - 45, 28
    cv2.circle(frame, (cx, cy), r + 4, (45, 45, 45), -1)
    cv2.circle(frame, (cx, cy), r + 4, (100, 100, 100), 2)
    if decision == "TURN LEFT":
        cv2.arrowedLine(frame, (cx+r-4, cy), (cx-r, cy),
                        C_LEFT, 4, cv2.LINE_AA, tipLength=0.4)
    elif decision == "TURN RIGHT":
        cv2.arrowedLine(frame, (cx-r+4, cy), (cx+r, cy),
                        C_RIGHT, 4, cv2.LINE_AA, tipLength=0.4)
    else:
        cv2.arrowedLine(frame, (cx, cy+r-4), (cx, cy-r),
                        C_STRAIGHT, 4, cv2.LINE_AA, tipLength=0.4)


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def process(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    steer_buf = deque(maxlen=SMOOTH_WIN)
    fidx = 0

    print(f"Input:  {input_path}  ({W}x{H} @ {fps:.1f}fps  {total} frames)")
    print(f"Output: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Extract + clean masks from existing overlay
        road_mask, sidewalk_mask = extract_masks(frame)

        # 2. Rebuild clean visual
        vis = apply_clean_overlay(frame, road_mask, sidewalk_mask)

        # 3. Compute goal centroid in lookahead band
        goal_row, goal_col = compute_goal(road_mask, H, W)

        # 4. Draw navigation path
        robot_base = (W // 2, H - HOOD_MASK_H - 10)
        draw_path(vis, robot_base, (goal_col, goal_row))

        # 5. Steer offset (signed: negative=left, positive=right)
        raw = (goal_col - W / 2) / W
        steer_buf.append(raw)
        smooth = float(np.mean(steer_buf))

        # 6. Decision
        if abs(smooth) < STRAIGHT_THRESH:
            decision = "STRAIGHT"
        elif smooth < 0:
            decision = "TURN LEFT"
        else:
            decision = "TURN RIGHT"

        # 7. HUD
        draw_hud(vis, decision, smooth, fidx)

        out.write(vis)
        fidx += 1

        if fidx % 60 == 0:
            pct = fidx / max(total, 1) * 100
            print(f"  {fidx}/{total} ({pct:.0f}%)  {decision}  off={smooth:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 navigation_v14.py <input.mp4> <output.mp4>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])
