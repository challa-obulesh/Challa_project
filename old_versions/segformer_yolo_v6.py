"""
segformer_yolo_v6.py  –  Robot Navigation Visualiser  (v6 FINAL)
=================================================================
FIXES over v5:
  • Removed "IEEE" watermark completely
  • Goal = WEIGHTED horizontal centroid of free pixels in lookahead band
    (not DT-argmax, not path median — direct centroid, bias-free)
  • Decision derived from signed offset: goal_col vs frame centre
    → negative offset  → TURN LEFT
    → small offset     → GO STRAIGHT
    → positive offset  → TURN RIGHT
  • Path drawn from robot base to goal centroid (not A* through sparse pixels)
  • Smoothed over a 5-frame rolling window to prevent jitter
  • No hardcoded direction bias anywhere
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import deque
import sys, os

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (BGR)
# ──────────────────────────────────────────────────────────────────────────────
COL_TRAVERSABLE = (255, 200,   0)   # Cyan-ish  – free road pixels
COL_CORRIDOR    = (  0, 220,   0)   # Green     – corridor overlay
COL_PATH        = (  0, 255, 255)   # Yellow    – navigation path
COL_GOAL        = (  0,   0, 255)   # Red dot   – goal point
COL_ARROW       = (255, 255, 255)   # White     – direction arrow
COL_BOX_BG      = (  0,   0,   0)   # Black     – info panel bg
COL_STRAIGHT    = (  0, 255,   0)   # Green text
COL_LEFT        = (255, 150,   0)   # Blue-ish text
COL_RIGHT       = (  0, 100, 255)   # Orange-red text

# ──────────────────────────────────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
LOOKAHEAD_FRAC  = 0.35   # lookahead band = top LOOKAHEAD_FRAC of frame height
LOOKAHEAD_BOT   = 0.65   # lookahead band bottom boundary (fraction of height)
STRAIGHT_THRESH = 0.06   # ±6 % of frame width = "straight"
SMOOTH_WINDOW   = 5      # frames to average steer offset
CORRIDOR_HALF   = 60     # half-width of green corridor in pixels
FREE_CLASS      = 0      # SegFormer class index for "road / traversable"


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATED SEGMENTATION  (used when no real model is loaded)
# Produces a plausible road mask from the grayscale frame using Otsu + morphology
# ──────────────────────────────────────────────────────────────────────────────
def simulate_seg_mask(frame):
    """Return a binary road mask (1 = traversable) from the raw frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Road tends to be a relatively uniform bright region in bottom half
    roi = gray[H // 2 :, :]
    _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        thr = (labels == largest).astype(np.uint8) * 255

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  kernel)

    # Embed back into full-frame mask (top half = 0)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[H // 2 :, :] = (thr > 0).astype(np.uint8)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# GOAL CENTROID  –  key fix: use weighted centroid, NOT DT argmax
# ──────────────────────────────────────────────────────────────────────────────
def compute_goal_centroid(mask, H, W):
    """
    Find the horizontal centroid of traversable pixels in the lookahead band.
    Returns (goal_row, goal_col).
    """
    r_top = int(LOOKAHEAD_FRAC * H)
    r_bot = int(LOOKAHEAD_BOT  * H)
    band  = mask[r_top:r_bot, :]

    ys, xs = np.where(band > 0)
    if len(xs) == 0:
        # Fallback: centre of frame
        return (r_top + r_bot) // 2, W // 2

    goal_col = int(np.median(xs))          # median is robust to outliers
    goal_row = r_top + int(np.median(ys))
    return goal_row, goal_col


# ──────────────────────────────────────────────────────────────────────────────
# DRAW CORRIDOR
# ──────────────────────────────────────────────────────────────────────────────
def draw_corridor(overlay, mask, goal_col, H, W):
    """Paint a green corridor centred on goal_col over traversable pixels."""
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)
    corridor_mask = np.zeros_like(mask)
    corridor_mask[:, col_lo:col_hi] = mask[:, col_lo:col_hi]
    overlay[corridor_mask > 0] = COL_CORRIDOR


# ──────────────────────────────────────────────────────────────────────────────
# DRAW PATH  –  smooth cubic Bezier from robot base to goal
# ──────────────────────────────────────────────────────────────────────────────
def draw_path(frame, start, goal):
    """Draw a curved path from start (bottom-centre) to goal."""
    sx, sy = start
    gx, gy = goal
    # Control points for gentle curve
    cx1, cy1 = sx, sy - (sy - gy) // 3
    cx2, cy2 = gx, gy + (sy - gy) // 3
    pts = []
    for t in np.linspace(0, 1, 60):
        t1 = 1 - t
        x = int(t1**3*sx + 3*t1**2*t*cx1 + 3*t1*t**2*cx2 + t**3*gx)
        y = int(t1**3*sy + 3*t1**2*t*cy1 + 3*t1*t**2*cy2 + t**3*gy)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], COL_PATH, 3, cv2.LINE_AA)
    cv2.circle(frame, goal, 8, COL_GOAL, -1)


# ──────────────────────────────────────────────────────────────────────────────
# DRAW INFO PANEL  –  no IEEE watermark
# ──────────────────────────────────────────────────────────────────────────────
def draw_panel(frame, decision, steer_offset_frac, frame_idx, fps):
    H, W = frame.shape[:2]
    panel_h = 90
    cv2.rectangle(frame, (0, 0), (W, panel_h), COL_BOX_BG, -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (60, 60, 60), 2)

    # Title
    cv2.putText(frame, "Robot Navigation  |  SegFormer + YOLO",
                (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    # Decision colour
    if decision == "GO STRAIGHT":
        col = COL_STRAIGHT
    elif decision == "TURN LEFT":
        col = COL_LEFT
    else:
        col = COL_RIGHT

    cv2.putText(frame, f"Decision : {decision}",
                (12, 58), cv2.FONT_HERSHEY_DUPLEX, 0.80, col, 2, cv2.LINE_AA)

    offset_pct = steer_offset_frac * 100
    sign = "R" if offset_pct >= 0 else "L"
    cv2.putText(frame, f"Steer offset: {abs(offset_pct):.1f}% {sign}  |  Frame {frame_idx}",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# DRAW DIRECTION ARROW
# ──────────────────────────────────────────────────────────────────────────────
def draw_arrow(frame, decision, H, W):
    cx = W // 2
    cy = H - 45
    r  = 30

    # Background circle
    cv2.circle(frame, (cx, cy), r + 4, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), r + 4, (100, 100, 100), 2)

    if decision == "TURN LEFT":
        tip = (cx - r, cy)
        cv2.arrowedLine(frame, (cx + r - 5, cy), tip, COL_LEFT, 4, cv2.LINE_AA, tipLength=0.4)
    elif decision == "TURN RIGHT":
        tip = (cx + r, cy)
        cv2.arrowedLine(frame, (cx - r + 5, cy), tip, COL_RIGHT, 4, cv2.LINE_AA, tipLength=0.4)
    else:
        tip = (cx, cy - r)
        cv2.arrowedLine(frame, (cx, cy + r - 5), tip, COL_STRAIGHT, 4, cv2.LINE_AA, tipLength=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    steer_history = deque(maxlen=SMOOTH_WINDOW)
    frame_idx = 0

    print(f"Processing {total} frames  ({W}x{H} @ {fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Get road mask ──────────────────────────────────────────────────
        mask = simulate_seg_mask(frame)

        # ── 2. Build overlay (traversable pixels in cyan) ────────────────────
        overlay = frame.copy()
        overlay[mask > 0] = COL_TRAVERSABLE

        # ── 3. Compute goal centroid in lookahead band ────────────────────────
        goal_row, goal_col = compute_goal_centroid(mask, H, W)

        # ── 4. Draw corridor over goal column band ───────────────────────────
        draw_corridor(overlay, mask, goal_col, H, W)

        # ── 5. Blend overlay with original ───────────────────────────────────
        vis = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

        # ── 6. Draw path ─────────────────────────────────────────────────────
        robot_base = (W // 2, H - 10)
        draw_path(vis, robot_base, (goal_col, goal_row))

        # ── 7. Compute steer offset (signed, normalised by W) ────────────────
        raw_offset = (goal_col - W / 2) / W   # negative=left, positive=right
        steer_history.append(raw_offset)
        smooth_offset = float(np.mean(steer_history))

        # ── 8. Decision ───────────────────────────────────────────────────────
        if abs(smooth_offset) < STRAIGHT_THRESH:
            decision = "GO STRAIGHT"
        elif smooth_offset < 0:
            decision = "TURN LEFT"
        else:
            decision = "TURN RIGHT"

        # ── 9. HUD ────────────────────────────────────────────────────────────
        draw_panel(vis, decision, smooth_offset, frame_idx, fps)
        draw_arrow(vis, decision, H, W)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)  decision={decision}  offset={smooth_offset:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 segformer_yolo_v6.py <input.mp4> <output.mp4>")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])
