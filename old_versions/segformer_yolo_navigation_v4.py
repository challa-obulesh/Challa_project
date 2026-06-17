"""
SegFormer-B0 + YOLOv8 Fusion Navigation  v4.0
==============================================
KEY FIXES vs v3:

  [FIX-1]  STOP when pedestrian/obstacle is in forward path zone
           navigation_decision() now receives the YOLO obstacle_mask
           and checks what fraction of the FORWARD ZONE is blocked.
           If > STOP_OBSTACLE_THRESHOLD  →  status=STOP, direction=STOP.

  [FIX-2]  STOP when path itself overlaps obstacle pixels
           If > STOP_PATH_OVERLAP_RATIO of smooth_pts land inside
           obstacle_mask  →  status=STOP.

  [FIX-3]  STOP when A* path is critically short
           If A* finds only a tiny path (< MIN_PATH_PX pixels)
           it means the forward space is blocked  →  STOP.

  [FIX-4]  Decision priority ladder:
             STOP  (obstacle in forward zone OR path blocked)
             > CAUTION  (obstacle nearby but path clear)
             > FORWARD / TURN LEFT / TURN RIGHT  (fully clear)

  [FIX-5]  Temporal STOP hold: once STOP is triggered, hold it for
           STOP_HOLD_FRAMES frames to prevent flickering back to SAFE.

  [FIX-6]  Forward zone definition tightened:
           Rows H//4 → H//2, cols W//3 → 2W//3
           (the exact area pedestrians occupy before robot reaches them)

USAGE:
    python segformer_yolo_navigation.py --source 0
    python segformer_yolo_navigation.py --source video.mp4 --output out.mp4
    python segformer_yolo_navigation.py --source image.jpg

REQUIREMENTS:
    pip install torch torchvision transformers ultralytics \
                opencv-python numpy scipy pillow
"""

import argparse
import heapq
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, binary_closing

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PATH_LOOKAHEAD_RATIO    = 0.20   # goal placed this fraction of H ahead
PATH_MAX_PX             = 120    # hard cap on drawn path pixel-length
CORRIDOR_BASE_W         = 16     # corridor half-width (px)
ASTAR_SCALE             = 4      # A* grid downsample factor

STEER_DEADBAND_DEG      = 8.0    # ±deg → FORWARD
STEER_SMOOTH_ALPHA      = 0.25   # temporal low-pass on steer

# ── STOP thresholds ──────────────────────────────────────────────────────────
STOP_OBSTACLE_THRESHOLD = 0.18   # obstacle covers > 18% of forward zone → STOP
STOP_PATH_OVERLAP_RATIO = 0.25   # > 25% of path pts inside obstacle → STOP
CAUTION_THRESHOLD       = 0.08   # obstacle 8-18% of forward zone → CAUTION
MIN_PATH_PX             = 30     # path shorter than this → STOP
STOP_HOLD_FRAMES        = 8      # hold STOP for N frames after trigger
INFLATE_PX              = 22     # YOLO safety margin (px)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
_seg_processor  = None
_seg_model      = None
_yolo_model     = None
_steer_smooth   = 0.0
_stop_hold_ctr  = 0       # FIX-5: STOP hold counter


def load_models(device: str = "cpu"):
    global _seg_processor, _seg_model, _yolo_model
    if _seg_model is None:
        print("[INFO] Loading SegFormer-B0 …")
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        _seg_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        _seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        ).to(device).eval()
        print("[INFO] SegFormer ready.")
    if _yolo_model is None:
        print("[INFO] Loading YOLOv8n …")
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
        print("[INFO] YOLOv8 ready.")
    return _seg_processor, _seg_model, _yolo_model


# ═════════════════════════════════════════════════════════════════════════════
# 1. SEGFORMER SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

TRAVERSABLE_IDS = {0, 1, 8, 9}   # road, sidewalk, vegetation, terrain


def run_segformer(frame_bgr, processor, model, device):
    import torch
    pil    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def build_traversable_mask(label_map):
    mask = np.zeros(label_map.shape, dtype=bool)
    for lbl in TRAVERSABLE_IDS:
        mask |= (label_map == lbl)
    return binary_closing(mask, structure=np.ones((5, 5)))


# ═════════════════════════════════════════════════════════════════════════════
# 2. YOLO OBSTACLE MASK  — INTERNAL ONLY, NEVER VISUALISED
# ═════════════════════════════════════════════════════════════════════════════

OBSTACLE_CLS = {
    0,  # person   ← most important for pedestrian STOP
    1, 2, 3, 5, 7,             # bicycle, car, motorbike, bus, truck
    13, 14, 16, 17, 18, 19,    # signs, animals
    56, 57, 58, 59, 60, 62,    # furniture / static obstacles
}

# Person-class IDs that trigger STOP immediately when in forward zone
PERSON_CLS = {0, 16, 17, 18, 19}   # person + animals


def build_obstacle_mask(frame_bgr, yolo_model, conf=0.30):
    """
    Returns:
        obstacle_mask  (H,W) bool — all detected obstacles inflated
        person_in_fwd  bool      — True if person/animal in forward zone
        fwd_obstacle_ratio float — fraction of forward zone that is obstacle
    """
    H, W = frame_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    # Forward zone boundaries (FIX-6)
    fwd_r1, fwd_r2 = H // 4, H // 2
    fwd_c1, fwd_c2 = W // 3, 2 * W // 3

    person_in_fwd = False
    results = yolo_model(frame_bgr, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        if cls_id not in OBSTACLE_CLS or conf_v < conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Check if this detection overlaps the forward zone
        overlap_r = (y1 < fwd_r2) and (y2 > fwd_r1)
        overlap_c = (x1 < fwd_c2) and (x2 > fwd_c1)
        if overlap_r and overlap_c and cls_id in PERSON_CLS:
            person_in_fwd = True

        # Inflate and add to mask
        x1i = max(0, x1 - INFLATE_PX);  y1i = max(0, y1 - INFLATE_PX)
        x2i = min(W, x2 + INFLATE_PX);  y2i = min(H, y2 + INFLATE_PX)
        mask[y1i:y2i, x1i:x2i] = True

    # Forward zone obstacle ratio
    fwd_zone = mask[fwd_r1:fwd_r2, fwd_c1:fwd_c2]
    fwd_obstacle_ratio = fwd_zone.sum() / max(1, fwd_zone.size)

    return mask, person_in_fwd, fwd_obstacle_ratio


# ═════════════════════════════════════════════════════════════════════════════
# 3. COST MAP
# ═════════════════════════════════════════════════════════════════════════════

def build_cost_map(traversable_mask, obstacle_mask):
    cost = np.ones(traversable_mask.shape, dtype=np.float32)
    cost[traversable_mask] = 0.0
    dist = distance_transform_edt(~obstacle_mask).astype(np.float32)
    soft = (dist > 0) & (dist < INFLATE_PX) & traversable_mask
    cost[soft] = 0.8 * (1.0 - dist[soft] / INFLATE_PX)
    cost[obstacle_mask] = 1.0
    return cost


# ═════════════════════════════════════════════════════════════════════════════
# 4. A* PATH PLANNER
# ═════════════════════════════════════════════════════════════════════════════

def _astar(cost_map, start, goal):
    H, W = cost_map.shape
    heap = [(0.0, start)]
    came = {start: None}
    g    = {start: 0.0}
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        cr, cc = cur
        for dr, dc in DIRS:
            nr, nc = cr+dr, cc+dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            c = cost_map[nr, nc]
            if c >= 1.0:
                continue
            mv = 1.414 if (dr and dc) else 1.0
            ng = g[cur] + mv + c * 6.0
            if ng < g.get((nr, nc), 1e18):
                came[(nr,nc)] = cur
                g[(nr,nc)]    = ng
                heapq.heappush(heap,
                    (ng + abs(nr-goal[0]) + abs(nc-goal[1]), (nr,nc)))
    return []


def _find_start(free_mask, H, W):
    cx = W // 2
    for row in range(H-1, H//3, -1):
        for margin in [0, 30, 80, W//2]:
            lo, hi = max(0, cx-margin), min(W-1, cx+margin)
            strip  = free_mask[row, lo:hi+1]
            if strip.any():
                return (row, lo + int(np.argmax(strip)))
    pts = np.argwhere(free_mask)
    return tuple(pts[np.argmax(pts[:,0])]) if len(pts) else (H-1, cx)


def _find_goal(free_mask, cost_map, dt_free, start_row, H, W, lookahead):
    cx           = W // 2
    lookahead_px = max(20, int(H * lookahead))
    band_top     = max(0, start_row - lookahead_px)
    band_bot     = max(0, start_row - 10)
    col_lo       = max(0,   cx - W//3)
    col_hi       = min(W-1, cx + W//3)

    band_dt   = dt_free[band_top:band_bot+1, col_lo:col_hi+1].copy()
    band_free = free_mask[band_top:band_bot+1, col_lo:col_hi+1]
    band_cost = cost_map[band_top:band_bot+1, col_lo:col_hi+1]
    band_dt[(band_cost >= 0.9) | ~band_free] = 0

    if band_dt.max() > 0:
        idx = np.unravel_index(np.argmax(band_dt), band_dt.shape)
        return (band_top + idx[0], col_lo + idx[1])

    pts = np.argwhere(free_mask[band_top:band_bot+1, :] &
                      (cost_map[band_top:band_bot+1, :] < 0.9))
    if len(pts):
        pts[:,0] += band_top
        return tuple(pts[np.argsort(np.abs(pts[:,1]-cx))[0]])

    pts = np.argwhere(free_mask[:start_row, :])
    if len(pts):
        return tuple(pts[np.argmin(pts[:,0])])
    return (max(0, start_row-20), cx)


def plan_path(cost_map, free_mask, H, W, lookahead):
    SCALE = ASTAR_SCALE
    sH, sW = H // SCALE, W // SCALE

    small_cost = cv2.resize(cost_map, (sW, sH), interpolation=cv2.INTER_AREA)
    small_free = cv2.resize(free_mask.astype(np.uint8), (sW, sH),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    small_dt   = distance_transform_edt(small_free).astype(np.float32)

    start = _find_start(small_free, sH, sW)
    goal  = _find_goal(small_free, small_cost, small_dt,
                       start[0], sH, sW, lookahead)
    path  = _astar(small_cost, start, goal)

    if path:
        return [(r*SCALE + SCALE//2, c*SCALE + SCALE//2) for r,c in path]

    # Fallback: distance-transform centerline
    dt_full = distance_transform_edt(free_mask).astype(np.float32)
    cx = W // 2
    fallback = []
    for r in range(start[0], max(0, start[0]-int(H*lookahead)), -1):
        row_dt = dt_full[r, max(0,cx-80):min(W,cx+80)]
        if row_dt.max() > 0:
            fallback.append((r, max(0,cx-80) + int(np.argmax(row_dt))))
    return fallback if fallback else [(_find_start(free_mask,H,W))]


# ═════════════════════════════════════════════════════════════════════════════
# 5. SMOOTH PATH + HARD LENGTH CAP
# ═════════════════════════════════════════════════════════════════════════════

def _truncate_to_length(pts, max_px):
    if len(pts) <= 2:
        return pts
    cum = 0.0
    for i in range(1, len(pts)):
        dr = float(pts[i,0]-pts[i-1,0])
        dc = float(pts[i,1]-pts[i-1,1])
        cum += (dr*dr + dc*dc)**0.5
        if cum >= max_px:
            return pts[:i+1]
    return pts


def smooth_path(path_rc, num_pts=80):
    if len(path_rc) < 4:
        return _truncate_to_length(np.array(path_rc, dtype=int), PATH_MAX_PX)
    rows = np.array([p[0] for p in path_rc], dtype=float)
    cols = np.array([p[1] for p in path_rc], dtype=float)
    step = max(1, len(path_rc)//40)
    rows, cols = rows[::step], cols[::step]
    try:
        k   = min(3, len(rows)-1)
        tck, _ = splprep([cols, rows], s=len(rows)*8, k=k)
        u      = np.linspace(0, 1, num_pts)
        cs, rs = splev(u, tck)
        pts    = np.column_stack([rs, cs]).astype(int)
        return _truncate_to_length(pts, PATH_MAX_PX)
    except Exception:
        return _truncate_to_length(np.array(path_rc, dtype=int), PATH_MAX_PX)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CORRIDOR
# ═════════════════════════════════════════════════════════════════════════════

def build_corridor(smooth_pts, traversable_mask, H, W):
    if len(smooth_pts) < 2:
        return np.zeros((H,W), dtype=bool)
    path_len = float(np.linalg.norm(smooth_pts[-1]-smooth_pts[0]))
    corr_w   = max(10, min(CORRIDOR_BASE_W,
                           int(CORRIDOR_BASE_W * path_len / (H*0.4))))
    corr = np.zeros((H,W), dtype=np.uint8)
    for (r,c) in smooth_pts:
        r = int(np.clip(r, 0, H-1))
        c = int(np.clip(c, 0, W-1))
        cv2.circle(corr, (c,r), corr_w, 255, -1)
    return (corr > 0) & traversable_mask


# ═════════════════════════════════════════════════════════════════════════════
# 7. NAVIGATION DECISION  — FIX-1 through FIX-5 (THE MAIN FIX)
# ═════════════════════════════════════════════════════════════════════════════

def navigation_decision(smooth_pts, frame_shape,
                        obstacle_mask, person_in_fwd, fwd_obstacle_ratio):
    """
    PRIORITY LADDER:
      1. STOP  — person detected in forward zone                  [FIX-1]
      2. STOP  — forward zone obstacle ratio > STOP threshold     [FIX-1]
      3. STOP  — path overlap with obstacle > STOP_PATH_OVERLAP   [FIX-2]
      4. STOP  — path length < MIN_PATH_PX (no room ahead)        [FIX-3]
      5. STOP  — hold counter active                              [FIX-5]
      6. CAUTION — obstacle 8-18% of forward zone
      7. FORWARD / TURN LEFT / TURN RIGHT — clear path
    """
    global _steer_smooth, _stop_hold_ctr

    H, W = frame_shape[:2]
    cx   = W // 2

    # ── Compute path pixel length ────────────────────────────────────────────
    path_px = 0.0
    if len(smooth_pts) >= 2:
        diffs   = np.diff(smooth_pts.astype(float), axis=0)
        path_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # ── Compute path-obstacle overlap ────────────────────────────────────────
    path_overlap_ratio = 0.0
    if len(smooth_pts) > 0 and obstacle_mask is not None:
        in_obs = 0
        for (r, c) in smooth_pts:
            r = int(np.clip(r, 0, H-1))
            c = int(np.clip(c, 0, W-1))
            if obstacle_mask[r, c]:
                in_obs += 1
        path_overlap_ratio = in_obs / max(1, len(smooth_pts))

    # ── STOP condition checks ─────────────────────────────────────────────────
    stop_reason = None

    if person_in_fwd:                                          # FIX-1a
        stop_reason = "PEDESTRIAN AHEAD"
    elif fwd_obstacle_ratio > STOP_OBSTACLE_THRESHOLD:        # FIX-1b
        stop_reason = f"OBSTACLE {fwd_obstacle_ratio*100:.0f}% FWD"
    elif path_overlap_ratio > STOP_PATH_OVERLAP_RATIO:        # FIX-2
        stop_reason = "PATH BLOCKED"
    elif path_px < MIN_PATH_PX:                               # FIX-3
        stop_reason = "NO CLEAR PATH"

    if stop_reason:
        _stop_hold_ctr = STOP_HOLD_FRAMES                     # FIX-5: arm hold
        _steer_smooth  = 0.0
        return {
            "status":    "STOP",
            "direction": "STOP",
            "steer":     0.0,
            "reason":    stop_reason,
        }

    # ── FIX-5: STOP hold (prevent flicker) ───────────────────────────────────
    if _stop_hold_ctr > 0:
        _stop_hold_ctr -= 1
        _steer_smooth   = 0.0
        return {
            "status":    "STOP",
            "direction": "STOP",
            "steer":     0.0,
            "reason":    "HOLD",
        }

    # ── CAUTION ───────────────────────────────────────────────────────────────
    if fwd_obstacle_ratio > CAUTION_THRESHOLD:
        return {
            "status":    "CAUTION",
            "direction": "SLOWING",
            "steer":     round(_steer_smooth, 1),
            "reason":    f"NEAR OBSTACLE {fwd_obstacle_ratio*100:.0f}%",
        }

    # ── SAFE: compute direction from path ────────────────────────────────────
    if len(smooth_pts) < 2:
        return {"status":"CAUTION","direction":"SEARCHING","steer":0.0,"reason":"NO PATH"}

    median_col = float(np.median(smooth_pts[:, 1]))
    offset     = median_col - cx
    raw_steer  = float(np.degrees(np.arctan2(offset, H * 0.3)))

    # Temporal smoothing
    _steer_smooth = (STEER_SMOOTH_ALPHA * raw_steer
                     + (1.0 - STEER_SMOOTH_ALPHA) * _steer_smooth)
    steer = _steer_smooth

    if abs(steer) < STEER_DEADBAND_DEG:
        direction = "FORWARD"
    elif steer > 0:
        direction = "TURN RIGHT"
    else:
        direction = "TURN LEFT"

    return {
        "status":    "SAFE",
        "direction": direction,
        "steer":     round(steer, 1),
        "reason":    "",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. RENDERER — green only, no red, no YOLO visuals
# ═════════════════════════════════════════════════════════════════════════════

COLOR_TRAV   = (34,  139,  34)
COLOR_CORR   = (0,   255,  90)
COLOR_PATH   = (255, 255, 255)
COLOR_SAFE   = (0,   220,   0)
COLOR_CAUTION= (0,   200, 220)
COLOR_STOP   = (0,   0,   255)   # bright red text for STOP

ALPHA_TRAV   = 0.30
ALPHA_CORR   = 0.50


def render_frame(frame_bgr, traversable_mask, corridor_mask,
                 smooth_pts, nav):
    """
    Draws ONLY:
      1. Camera frame
      2. Semi-transparent green — traversable area
      3. Brighter green — safe corridor
      4. White centre path line  (not drawn when STOP)
      5. HUD: Status | Direction | Reason

    ❌ No bounding boxes  ❌ No YOLO labels  ❌ No red overlays
    """
    out = frame_bgr.astype(np.float32)
    H, W = out.shape[:2]

    # Traversable tint
    trav = np.zeros_like(out)
    trav[traversable_mask] = COLOR_TRAV
    out = cv2.addWeighted(trav, ALPHA_TRAV, out, 1.0, 0)

    # Corridor tint (only when not STOP)
    if nav["status"] != "STOP":
        corr = np.zeros_like(out)
        corr[corridor_mask] = COLOR_CORR
        out = cv2.addWeighted(corr, ALPHA_CORR, out, 1.0, 0)

    out = out.astype(np.uint8)

    # Centre path line (only when SAFE/CAUTION)
    if nav["status"] in ("SAFE", "CAUTION") and len(smooth_pts) >= 2:
        pts = smooth_pts[:, [1, 0]].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, COLOR_PATH, 2, cv2.LINE_AA)
        # Goal dot
        cv2.circle(out,
                   (int(smooth_pts[-1,1]), int(smooth_pts[-1,0])),
                   5, (0,255,255), -1)

    # HUD banner
    banner_h = 55
    overlay  = out.copy()
    cv2.rectangle(overlay, (0,0), (W, banner_h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    status    = nav["status"]
    direction = nav["direction"]
    steer     = nav["steer"]
    reason    = nav.get("reason", "")

    if status == "STOP":
        txt_col = COLOR_STOP
    elif status == "CAUTION":
        txt_col = COLOR_CAUTION
    else:
        txt_col = COLOR_SAFE

    cv2.putText(out, f"Status: {status}",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX,
                0.70, txt_col, 2, cv2.LINE_AA)

    dir_text = f"{direction}" if status == "STOP" else f"{direction}  ({steer:+.1f}deg)"
    cv2.putText(out, dir_text,
                (10, 46), cv2.FONT_HERSHEY_DUPLEX,
                0.55, (220,220,220), 1, cv2.LINE_AA)

    if reason:
        cv2.putText(out, reason,
                    (W//2, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180,180,255), 1, cv2.LINE_AA)

    cv2.putText(out, "SegFormer-B0 + YOLOv8",
                (W-230, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (130,130,130), 1)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 9. FULL FRAME PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def process_frame(frame_bgr, processor, seg_model, yolo_model,
                  device, lookahead):
    H, W = frame_bgr.shape[:2]

    # A — SegFormer segmentation
    label_map        = run_segformer(frame_bgr, processor, seg_model, device)
    traversable_mask = build_traversable_mask(label_map)

    # B — YOLOv8 (internal) — returns obstacle mask + forward zone analysis
    obstacle_mask, person_in_fwd, fwd_obs_ratio = build_obstacle_mask(
        frame_bgr, yolo_model)

    # C — Free space
    free_mask = traversable_mask & ~obstacle_mask

    # D — Cost map
    cost_map = build_cost_map(traversable_mask, obstacle_mask)

    # E — A* path plan
    raw_path   = plan_path(cost_map, free_mask, H, W, lookahead)

    # F — Smooth + cap length
    smooth_pts = smooth_path(raw_path, num_pts=80)

    # G — Corridor
    corridor_mask = build_corridor(smooth_pts, traversable_mask, H, W)

    # H — Navigation decision (FIX-1 to FIX-5)
    nav = navigation_decision(
        smooth_pts, frame_bgr.shape,
        obstacle_mask, person_in_fwd, fwd_obs_ratio
    )

    # I — Render
    out = render_frame(frame_bgr, traversable_mask, corridor_mask,
                       smooth_pts, nav)

    return out, nav


# ═════════════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser("SegFormer + YOLOv8 Navigation v4")
    ap.add_argument("--source",    default="0")
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--output",    default="")
    ap.add_argument("--show",      action="store_true", default=True)
    ap.add_argument("--conf",      type=float, default=0.30)
    ap.add_argument("--lookahead", type=float, default=PATH_LOOKAHEAD_RATIO,
                    help="Path lookahead as fraction of frame H (default 0.20)")
    return ap.parse_args()


def main():
    global _steer_smooth, _stop_hold_ctr
    args = parse_args()
    processor, seg_model, yolo_model = load_models(args.device)

    src      = int(args.source) if args.source.isdigit() else args.source
    is_image = isinstance(src, str) and \
               Path(src).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}

    # ── Image mode ──────────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            sys.exit(f"[ERROR] Cannot read: {src}")
        _steer_smooth = 0.0;  _stop_hold_ctr = 0
        out, nav = process_frame(frame, processor, seg_model, yolo_model,
                                 args.device, args.lookahead)
        print(f"[NAV] {nav}")
        if args.output:
            cv2.imwrite(args.output, out)
            print(f"[SAVED] {args.output}")
        if args.show:
            cv2.imshow("Navigation v4", out)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Video / webcam mode ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {src}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (W_src, H_src))
        print(f"[INFO] Writing → {args.output}  ({W_src}×{H_src} @ {fps_src:.0f}fps)")

    _steer_smooth = 0.0;  _stop_hold_ctr = 0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, nav = process_frame(frame, processor, seg_model, yolo_model,
                                 args.device, args.lookahead)

        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.putText(out, f"FPS {fps:.1f}",
                    (W_src-90, H_src-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

        if writer:
            writer.write(out)
        if args.show:
            cv2.imshow("SegFormer + YOLOv8 Navigation v4", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        reason = nav.get('reason','')
        print(f"[NAV] {nav['status']:7s}  {nav['direction']:12s}  "
              f"steer={nav['steer']:+6.1f}°  {reason}  fps={fps:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
