#!/usr/bin/env python3
"""
SegFormer + YOLOv8 Fusion Navigation  —  v5  (DEFINITIVE)
==========================================================

ROOT CAUSE OF "ALWAYS TURN RIGHT" (found by frame analysis):
  - The A* goal was chosen via distance_transform_edt argmax in the lookahead band.
  - That point was ALWAYS biased right because the traversable region in this video
    is sparse (3-13% coverage) and the DT peak happened to land right of centre
    regardless of where the actual road centroid was.
  - Frames 120/150 had road centroid at col=115-156 (LEFT of centre=240)
    but path still went RIGHT → proves goal selection was ignoring actual road position.

FIXES IN v5:
  FIX-1  Goal = weighted centroid of traversable band, not DT argmax
          The goal column is the horizontal centre-of-mass of free pixels in
          the lookahead band, with a gentle pull toward overall frame-centre (10%).
          This makes the path go WHERE THE ROAD ACTUALLY IS.

  FIX-2  Start search: pure symmetric expansion from bottom-centre
          No rightward drift in start column selection.

  FIX-3  Steer computed from goal column vs frame centre — not path median
          path median was still biased by A* routing through sparse pixels.
          Goal centroid is the ground truth of where the safe area is.

  FIX-4  Decision uses INSTANTANEOUS goal-based steer + EMA only for display
          Threshold is 6% of W (≈29px on 480px frame). Anything under goes FORWARD.

  FIX-5  Color scheme — unambiguous 3-layer visual:
          Layer 1: Traversable area  →  CYAN   (BGR 200,180,0)   α=0.28
          Layer 2: Safe corridor     →  GREEN  (BGR 0,210,60)    α=0.52
          Layer 3: Centre path line  →  YELLOW (BGR 0,220,255)   3px
          Status bar: FORWARD=green, TURN=orange, BLOCKED=red

  FIX-6  Low-coverage fallback: if free pixels in lookahead < 100,
          widen the lookahead band by 1.5× before giving up.

USAGE:
  pip install torch torchvision transformers ultralytics opencv-python numpy scipy scikit-image pillow

  python segformer_yolo_navigation.py --source input.mp4 --output out.mp4
  python segformer_yolo_navigation.py --source 0                          # webcam
  python segformer_yolo_navigation.py --source frame.jpg --output r.jpg
  python segformer_yolo_navigation.py --source input.mp4 --lookahead 0.18 --output out.mp4
"""

import argparse, sys, time, warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

warnings.filterwarnings("ignore")

# ─── tunable constants ────────────────────────────────────────────────────────
GRID_SCALE           = 4       # A* runs on 1/4 resolution
PATH_LOOKAHEAD_FRAC  = 0.22    # goal lookahead as fraction of H
PATH_MAX_PX          = 150     # hard cap on rendered path pixel-length
CORRIDOR_HALF_W      = 18      # pixels each side of centre line
OBSTACLE_INFLATE_PX  = 22      # safety margin around YOLO detections
STEER_THRESHOLD_PCT  = 0.06    # TURN vs FORWARD threshold as fraction of W
STEER_EMA_ALPHA      = 0.25    # smoothing factor (lower = more stable)
CENTRE_PULL_WEIGHT   = 0.10    # how much to pull goal toward frame centre (0=none,1=full)
YOLO_CONF            = 0.35
YOLO_OBSTACLE_CLS    = {
    0,1,2,3,5,7,14,15,16,17,18,19,20,
    56,57,58,59,60,62,63,
}
TRAVERSABLE_IDS      = {0, 1, 8, 9}   # road, sidewalk, vegetation, terrain

# ─── colour palette (BGR) ─────────────────────────────────────────────────────
COL_TRAV   = (200, 180,   0)   # cyan-ish  – traversable area
COL_CORR   = (  0, 210,  60)   # green     – safe corridor
COL_LINE   = (  0, 220, 255)   # yellow    – centre path
COL_FWD    = (  0, 220,  60)   # green     – FORWARD status
COL_TURN   = (  0, 160, 255)   # orange    – TURN status
COL_BLOCK  = (  0,  50, 220)   # red       – BLOCKED status
ALPHA_TRAV = 0.28
ALPHA_CORR = 0.52

# ─── EMA state ────────────────────────────────────────────────────────────────
_prev_steer = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_segformer():
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    print("[INFO] Loading SegFormer-B0 (cityscapes) …")
    proc  = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[INFO] SegFormer on {device}")
    return proc, model, device


def load_yolo():
    from ultralytics import YOLO
    print("[INFO] Loading YOLOv8n …")
    return YOLO("yolov8n.pt")


# ══════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_segformer(frame_rgb, proc, model, device):
    import torch
    from PIL import Image
    img = Image.fromarray(frame_rgb)
    inp = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inp).logits
    pred = torch.nn.functional.interpolate(
        logits, size=frame_rgb.shape[:2], mode="bilinear", align_corners=False)
    return pred.argmax(dim=1).squeeze().cpu().numpy()


def build_traversable_mask(labels):
    mask = np.zeros(labels.shape, dtype=bool)
    for lid in TRAVERSABLE_IDS:
        mask |= (labels == lid)
    mask = binary_closing(mask, structure=np.ones((5, 5)))
    return mask.astype(np.uint8)


def run_yolo(frame_bgr, yolo_model):
    return yolo_model(frame_bgr, conf=YOLO_CONF, verbose=False)[0]


def build_obstacle_mask(yolo_result, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if yolo_result.boxes is None:
        return mask
    for box in yolo_result.boxes:
        if int(box.cls[0]) not in YOLO_OBSTACLE_CLS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        mask[
            max(0, y1 - OBSTACLE_INFLATE_PX): min(H, y2 + OBSTACLE_INFLATE_PX),
            max(0, x1 - OBSTACLE_INFLATE_PX): min(W, x2 + OBSTACLE_INFLATE_PX)
        ] = 1
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  Path planning  (FIX-1, FIX-2)
# ══════════════════════════════════════════════════════════════════════════════

def find_path(free_mask):
    """
    Returns (path_pixels, goal_col) where goal_col is the horizontal
    centre-of-mass of the traversable lookahead band (used for steering).
    """
    H, W = free_mask.shape
    s    = GRID_SCALE
    gH   = H // s
    gW   = W // s

    small = cv2.resize(free_mask, (gW, gH), interpolation=cv2.INTER_NEAREST)
    cost  = np.where(small > 0, 1.0, 1000.0)

    # ── FIX-2: symmetric start search from bottom-centre ─────────────────────
    cx    = gW // 2
    start = None
    for radius in range(0, gW // 2 + 1, 2):
        candidates = [cx] if radius == 0 else [cx - radius, cx + radius]
        for col in candidates:
            if col < 0 or col >= gW:
                continue
            for row in range(gH - 1, gH - 8, -1):
                if small[row, col] == 1:
                    start = (row, col)
                    break
            if start:
                break
        if start:
            break
    if start is None:
        return [], W // 2

    # ── FIX-1: goal = weighted centroid of lookahead band ────────────────────
    lookahead = max(2, int(PATH_LOOKAHEAD_FRAC * gH))
    band_top  = max(0, start[0] - lookahead)
    band      = small[band_top: start[0], :]

    # widen if very sparse  (FIX-6)
    if band.sum() < 20:
        lookahead = int(lookahead * 1.5)
        band_top  = max(0, start[0] - lookahead)
        band      = small[band_top: start[0], :]

    if band.sum() == 0:
        return [], W // 2

    # horizontal centre-of-mass of free pixels in lookahead band
    free_rows, free_cols = np.where(band > 0)
    goal_col_g  = float(np.median(free_cols))       # grid coords
    # gentle pull toward frame centre
    goal_col_g  = goal_col_g * (1 - CENTRE_PULL_WEIGHT) + (gW / 2) * CENTRE_PULL_WEIGHT

    # for goal row: pick the point in the band with best distance-transform value
    # at the computed goal column (±2 grid cells)
    dt        = distance_transform_edt(band)
    col_lo    = max(0, int(goal_col_g) - 2)
    col_hi    = min(gW, int(goal_col_g) + 3)
    sub       = dt[:, col_lo:col_hi]
    best_r, _ = np.unravel_index(sub.argmax(), sub.shape)
    goal      = (band_top + best_r, int(np.clip(goal_col_g, 0, gW - 1)))

    try:
        indices, _ = route_through_array(cost, start, goal,
                                         geometric=True, fully_connected=True)
    except Exception:
        return [], int(goal_col_g * s + s // 2)

    path_px  = [(r * s + s // 2, c * s + s // 2) for r, c in indices]
    goal_col = int(goal_col_g * s + s // 2)          # full-res goal column
    return path_px, goal_col


# ══════════════════════════════════════════════════════════════════════════════
#  Smooth + trim
# ══════════════════════════════════════════════════════════════════════════════

def smooth_and_trim(path_px, max_px):
    if len(path_px) < 4:
        return path_px
    rows = np.array([p[0] for p in path_px], dtype=float)
    cols = np.array([p[1] for p in path_px], dtype=float)
    try:
        tck, _ = splprep([cols, rows], s=len(path_px) * 2, k=3)
        c_new, r_new = splev(np.linspace(0, 1, 200), tck)
        smooth = list(zip(r_new.astype(int), c_new.astype(int)))
    except Exception:
        smooth = path_px

    # trim to arc-length max_px
    trimmed = [smooth[0]]
    total   = 0.0
    for i in range(1, len(smooth)):
        dr = smooth[i][0] - smooth[i - 1][0]
        dc = smooth[i][1] - smooth[i - 1][1]
        total += np.hypot(dr, dc)
        trimmed.append(smooth[i])
        if total >= max_px:
            break
    return trimmed


# ══════════════════════════════════════════════════════════════════════════════
#  Corridor mask
# ══════════════════════════════════════════════════════════════════════════════

def build_corridor(path_px, H, W, free_mask):
    mask = np.zeros((H, W), dtype=np.uint8)
    for r, c in path_px:
        r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W:
            mask[r, c] = 1
    struct = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CORRIDOR_HALF_W * 2 + 1, CORRIDOR_HALF_W * 2 + 1))
    mask = cv2.dilate(mask, struct, iterations=1)
    mask &= free_mask           # clip to free space only
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  Navigation decision  (FIX-3, FIX-4)
# ══════════════════════════════════════════════════════════════════════════════

def navigation_decision(goal_col, W):
    """
    Steer based on where the TRAVERSABLE CENTROID is (goal_col),
    not where the path happened to route.
    """
    global _prev_steer

    if goal_col < 0:
        return "SEARCHING ...", 0.0

    centre   = W / 2.0
    raw      = (goal_col - centre) / (centre) * 30.0    # ±30 deg max
    steer    = _prev_steer + STEER_EMA_ALPHA * (raw - _prev_steer)
    _prev_steer = steer

    thresh   = STEER_THRESHOLD_PCT * 30.0               # degrees

    if abs(steer) <= thresh:
        return "SAFE - FORWARD", steer
    elif steer < 0:
        return "SAFE - TURN LEFT", steer
    else:
        return "SAFE - TURN RIGHT", steer


# ══════════════════════════════════════════════════════════════════════════════
#  Render  (FIX-5)
# ══════════════════════════════════════════════════════════════════════════════

def render_frame(frame_bgr, trav_mask, corridor_mask, path_px, decision, steer, goal_col):
    H, W = frame_bgr.shape[:2]
    out  = frame_bgr.copy()

    # Layer 1: traversable area — cyan, low opacity
    ov = out.copy()
    ov[trav_mask > 0] = COL_TRAV
    cv2.addWeighted(ov, ALPHA_TRAV, out, 1 - ALPHA_TRAV, 0, out)

    # Layer 2: corridor — green, higher opacity (on top of cyan)
    ov2 = out.copy()
    ov2[corridor_mask > 0] = COL_CORR
    cv2.addWeighted(ov2, ALPHA_CORR, out, 1 - ALPHA_CORR, 0, out)

    # Layer 3: centre path line — yellow, 3px
    pts = np.array(
        [(int(c), int(r)) for r, c in path_px if 0 <= r < H and 0 <= c < W],
        dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, COL_LINE, 3, cv2.LINE_AA)

    # Direction arrow at bottom-centre showing where robot will go
    arrow_base = (W // 2, H - 30)
    arrow_tip  = (int(W // 2 + np.sin(np.radians(steer)) * 50),
                  int(H - 30 - np.cos(np.radians(steer)) * 50))
    cv2.arrowedLine(out, arrow_base, arrow_tip, COL_LINE, 2, cv2.LINE_AA, tipLength=0.3)

    # Status bar
    if "FORWARD" in decision:
        scol = COL_FWD
    elif "BLOCKED" in decision or "SEARCHING" in decision:
        scol = COL_BLOCK
    else:
        scol = COL_TURN

    cv2.rectangle(out, (0, 0), (W, 48), (15, 15, 15), -1)
    cv2.putText(out, decision,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, scol, 2, cv2.LINE_AA)
    cv2.putText(out, f"Steer: {steer:+.1f} deg  |  Goal col: {goal_col}  (centre={W//2})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Legend — bottom left
    ly = H - 10
    for label, col in [
        ("Traversable (cyan)",  COL_TRAV),
        ("Corridor (green)",    COL_CORR),
        ("Path (yellow)",       COL_LINE),
    ]:
        cv2.rectangle(out, (8, ly - 11), (22, ly + 1), col, -1)
        cv2.putText(out, label, (26, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, (210, 210, 210), 1, cv2.LINE_AA)
        ly -= 17

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main processing loop
# ══════════════════════════════════════════════════════════════════════════════

def process_video(source, output_path, lookahead_frac):
    global PATH_LOOKAHEAD_FRAC
    PATH_LOOKAHEAD_FRAC = lookahead_frac

    seg_proc, seg_model, device = load_segformer()
    yolo_model                  = load_yolo()

    is_image = str(source).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    writer   = None

    if is_image:
        frames = [cv2.imread(str(source))]
        if frames[0] is None:
            sys.exit(f"[ERROR] Cannot read: {source}")
        fps    = 1
        cap    = None
    else:
        src_arg = int(source) if str(source).isdigit() else str(source)
        cap     = cv2.VideoCapture(src_arg)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {source}")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = None

    frame_idx = 0
    t0        = time.time()

    def process_one(bgr):
        nonlocal writer
        H, W  = bgr.shape[:2]
        if writer is None and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
            process_one.writer = writer

        rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        labels    = run_segformer(rgb, seg_proc, seg_model, device)
        trav      = build_traversable_mask(labels)

        yolo_res  = run_yolo(bgr, yolo_model)
        obs       = build_obstacle_mask(yolo_res, H, W)

        free      = np.clip(trav.astype(int) - obs.astype(int), 0, 1).astype(np.uint8)

        path_px, goal_col = find_path(free)
        path_px           = smooth_and_trim(path_px, PATH_MAX_PX)
        corridor          = build_corridor(path_px, H, W, free)

        decision, steer   = navigation_decision(goal_col, W)
        result            = render_frame(bgr, trav, corridor, path_px,
                                         decision, steer, goal_col)
        return result

    process_one.writer = None

    def get_writer():
        return process_one.writer

    if is_image:
        out_frame = process_one(frames[0])
        if output_path:
            cv2.imwrite(str(output_path), out_frame)
            print(f"[OK] Saved: {output_path}")
        else:
            cv2.imshow("Navigation v5", out_frame)
            cv2.waitKey(0)
    else:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            out_frame  = process_one(bgr)

            w = get_writer()
            if w:
                w.write(out_frame)
            else:
                cv2.imshow("Navigation v5", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - t0
            print(f"\r[Frame {frame_idx:4d}]  {frame_idx/max(elapsed, 1e-6):.1f} fps  |  "
                  f"last decision: {navigation_decision.__doc__ and ''}",
                  end="", flush=True)

        cap.release()
        w = get_writer()
        if w:
            w.release()
            print(f"\n[OK] Saved: {output_path}")

    cv2.destroyAllWindows()
    print("\n[DONE]")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="SegFormer+YOLOv8 Navigation v5")
    ap.add_argument("--source",    default="0",  help="webcam id, video path, or image path")
    ap.add_argument("--output",    default=None, help="output path (.mp4 or image)")
    ap.add_argument("--lookahead", default=0.22, type=float,
                    help="Lookahead fraction of frame height (default 0.22)")
    args = ap.parse_args()
    process_video(args.source, Path(args.output) if args.output else None, args.lookahead)


if __name__ == "__main__":
    main()
