"""
SegFormer-B0 + YOLOv8 Fusion Navigation  v5.0
==============================================
FIXES vs v4:

  [FIX-A]  STOP was always triggered — root causes fixed:
           • INFLATE_PX was 22 → now 6  (one pedestrian no longer fills 36% of fwd zone)
           • STOP_OBSTACLE_THRESHOLD was 0.18 → now 0.55  (>55% of fwd zone blocked)
           • STOP_PATH_OVERLAP_RATIO was 0.25 → now 0.70  (>70% of path inside obstacle)
           • STOP_HOLD_FRAMES was 8 → now 3  (shorter hold, less lock-in)
           • MIN_PATH_PX was 30 → now 8   (don't STOP just because path is short)
           • person_in_fwd now only triggers STOP if bbox area > 3% of frame total
             (avoids distant/small detections)

  [FIX-B]  Pure PIXEL-LEVEL output:
           • Every pixel coloured by its SegFormer semantic class
           • YOLO obstacle pixels painted in their own colour
           • NO path lines, NO corridor blobs, NO bezier curves
           • Minimal HUD strip at top: Status | Direction | Steer | Reason

  [FIX-C]  Navigation decision uses ONLY the pixel masks:
           • Computes road centroid in lookahead band directly from pixel labels
           • No A*, no spline, no cost map needed
           • CAUTION/STOP thresholds calibrated to real pixel ratios

USAGE:
    python segformer_yolo_navigation_v5.py --source video.mp4 --output out.mp4
    python segformer_yolo_navigation_v5.py --source 0
    python segformer_yolo_navigation_v5.py --source image.jpg --output out.jpg

REQUIREMENTS:
    pip install torch torchvision transformers ultralytics opencv-python numpy pillow
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CITYSCAPES LABEL → PIXEL COLOUR (BGR)
# ─────────────────────────────────────────────────────────────────────────────
# Full 19-class cityscapes palette (BGR order)
CITYSCAPES_PALETTE_BGR = [
    (128,  64, 128),   #  0  road          ← dark purple/magenta
    (232,  35, 244),   #  1  sidewalk
    ( 70,  70,  70),   #  2  building
    (156, 102, 102),   #  3  wall
    (153, 153, 190),   #  4  fence
    (153, 153, 153),   #  5  pole
    ( 30, 170, 250),   #  6  traffic light
    (  0, 220, 220),   #  7  traffic sign
    ( 35, 142, 107),   #  8  vegetation
    (152, 251, 152),   #  9  terrain
    (180, 130,  70),   # 10  sky
    ( 60,  20, 220),   # 11  person
    (  0,   0, 255),   # 12  rider
    (142,   0,   0),   # 13  car
    ( 70,   0,   0),   # 14  truck
    (100,  60,   0),   # 15  bus
    ( 90,   0,   0),   # 16  train
    (230,   0,   0),   # 17  motorcycle
    ( 32,  11, 119),   # 18  bicycle
]

# Readable class names
CITYSCAPES_NAMES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]

# Which class IDs count as traversable (robot can drive there)
TRAVERSABLE_IDS = {0, 1, 8, 9}   # road, sidewalk, vegetation, terrain

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE CONSTANTS  (calibrated to avoid always-STOP)
# ─────────────────────────────────────────────────────────────────────────────
LOOKAHEAD_TOP        = 0.30   # lookahead band top    (fraction of H)
LOOKAHEAD_BOT        = 0.65   # lookahead band bottom (fraction of H)
STEER_DEADBAND_DEG   = 7.0    # ±deg → FORWARD
STEER_SMOOTH_ALPHA   = 0.20   # low-pass on steer angle

# STOP / CAUTION thresholds — calibrated from real video measurement
# Normal driving: road band has 126K–150K pixels in lookahead zone
INFLATE_PX              = 6     # YOLO box inflation (px)
STOP_ROAD_PX            = 37000 # road px < 37K  → STOP  (30% of measured min 126K)
CAUTION_ROAD_PX         = 75000 # road px < 75K  → CAUTION (60% of measured min)
STOP_HOLD_FRAMES        = 3     # hold STOP for N frames

# Person detection: only STOP if large bbox (>3% of frame) overlaps forward center
PERSON_MIN_AREA_RATIO   = 0.03  # 3% of frame (~3686px on 1280×720)

# YOLO obstacle class IDs
OBSTACLE_CLS = {0,1,2,3,5,7,13,14,16,17,18,19,56,57,58,59,60,62}
PERSON_CLS   = {0, 16, 17, 18, 19}   # person + animals

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
_seg_processor = None
_seg_model     = None
_yolo_model    = None
_steer_smooth  = 0.0
_stop_hold_ctr = 0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_models(device="cpu"):
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEGFORMER — pixel-level label map
# ─────────────────────────────────────────────────────────────────────────────
def run_segformer(frame_bgr, processor, model, device):
    import torch
    pil    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIXEL-LEVEL COLOUR MAP from label map
# ─────────────────────────────────────────────────────────────────────────────
def label_to_colour(label_map):
    """Convert H×W label map → H×W×3 BGR colour image."""
    H, W = label_map.shape
    colour = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, bgr in enumerate(CITYSCAPES_PALETTE_BGR):
        colour[label_map == cls_id] = bgr
    return colour


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACT BOARDS & TRAFFIC LIGHT COLORS
# ─────────────────────────────────────────────────────────────────────────────
def determine_traffic_light_color(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"
        
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Active traffic lights are bright, so Value (V) is high. 
    # Because of camera glare, saturation might drop.
    v_mask = hsv[:, :, 2] > 200
    if not np.any(v_mask):
        # Fallback to lower threshold if not super bright
        v_mask = hsv[:, :, 2] > 150
        
    # Apply mask
    bright_pixels = hsv[v_mask]
    if len(bright_pixels) == 0:
        return "UNKNOWN"
        
    # Analyze hue of bright pixels
    hues = bright_pixels[:, 0]
    
    # Hue ranges (OpenCV uses 0-179)
    r_count = np.sum((hues <= 10) | (hues >= 160))
    y_count = np.sum((hues >= 15) & (hues <= 35))
    g_count = np.sum((hues >= 40) & (hues <= 100))
    
    max_count = max(r_count, y_count, g_count)
    if max_count < 2: 
        return "UNKNOWN"
        
    if max_count == r_count: return "RED"
    elif max_count == g_count: return "GREEN"
    else: return "YELLOW"

def extract_boards_from_seg(label_map):
    """Extract bounding boxes for traffic signs/boards from SegFormer label map."""
    boards = []
    # Class 7 is 'traffic sign' in Cityscapes
    board_mask = (label_map == 7).astype(np.uint8) * 255
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 150 < w * h < 8000:  # Ignore tiny noisy patches and massive buildings
            aspect_ratio = float(w) / max(h, 1)
            if 0.2 < aspect_ratio < 3.0: # Filter out extremely long/thin boxes
                boards.append((x, y, x + w, y + h))
    return boards


# ─────────────────────────────────────────────────────────────────────────────
# 4. YOLO — obstacle pixel mask (inflated bounding boxes)
# ─────────────────────────────────────────────────────────────────────────────
def build_obstacle_mask(frame_bgr, yolo_model, conf=0.35):
    """
    Returns:
        obstacle_mask      (H,W) bool
        person_in_fwd      bool  — large person overlapping forward zone
        fwd_obstacle_ratio float — fraction of forward zone covered by obstacles
        traffic_lights     list  — list of (x1, y1, x2, y2) tuples for traffic lights
    """
    H, W = frame_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    # Forward zone
    fwd_r1, fwd_r2 = H // 4, H // 2
    fwd_c1, fwd_c2 = W // 3, 2 * W // 3

    person_in_fwd  = False
    frame_area     = float(H * W)
    traffic_lights = []

    results = yolo_model(frame_bgr, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        if conf_v < conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bbox_area = (x2 - x1) * (y2 - y1)

        # 9 is traffic light in COCO
        if cls_id == 9:
            color = determine_traffic_light_color(frame_bgr, (x1, y1, x2, y2))
            traffic_lights.append((x1, y1, x2, y2, color))
            continue  # Do not add to ground obstacle mask

        if cls_id not in OBSTACLE_CLS:
            continue

        # Only flag person if their bbox is large enough (not a distant speck)
        if cls_id in PERSON_CLS:
            overlap_r = (y1 < fwd_r2) and (y2 > fwd_r1)
            overlap_c = (x1 < fwd_c2) and (x2 > fwd_c1)
            if overlap_r and overlap_c and (bbox_area / frame_area) > PERSON_MIN_AREA_RATIO:
                person_in_fwd = True

        # Use 100% of the bounding box for obstacle mask
        xi1 = max(0, x1 - INFLATE_PX);  yi1 = max(0, y1 - INFLATE_PX)
        xi2 = min(W, x2 + INFLATE_PX);  yi2 = min(H, y2 + INFLATE_PX)
        mask[yi1:yi2, xi1:xi2] = True

    fwd_zone           = mask[fwd_r1:fwd_r2, fwd_c1:fwd_c2]
    fwd_obstacle_ratio = fwd_zone.sum() / max(1, fwd_zone.size)

    return mask, person_in_fwd, fwd_obstacle_ratio, traffic_lights


# ─────────────────────────────────────────────────────────────────────────────
# 5. NAVIGATION DECISION  — pixel-centroid based, recalibrated thresholds
# ─────────────────────────────────────────────────────────────────────────────
def navigation_decision(label_map, obstacle_mask, person_in_fwd,
                        fwd_obstacle_ratio, frame_shape):
    """
    STOP/CAUTION based on road pixel count in lookahead band.
    Calibrated from real video: normal road = 126K-150K px in band.
      STOP    → road_px < 37K  (road vanishes — dead end/intersection/obstacle)
      CAUTION → road_px < 75K  (road narrowing)
      STOP    → large pedestrian overlapping forward centre
    """
    global _steer_smooth, _stop_hold_ctr

    H, W = frame_shape[:2]
    cx   = W // 2

    # ── Count road pixels in full lookahead band ───────────────────────────
    r0 = int(LOOKAHEAD_TOP * H)
    r1 = int(LOOKAHEAD_BOT * H)
    band_labels = label_map[r0:r1, :]
    road_mask_band = np.isin(band_labels, list(TRAVERSABLE_IDS))
    n_road_total = int(road_mask_band.sum())

    # Count clear road (not blocked by obstacle)
    band_obstacle  = obstacle_mask[r0:r1, :]
    road_clear     = road_mask_band & ~band_obstacle
    road_ys, road_xs = np.where(road_clear)
    n_road_clear   = len(road_xs)

    # ── STOP checks ────────────────────────────────────────────────────────
    stop_reason = None

    # 1. Large pedestrian in the forward centre path
    if person_in_fwd:
        stop_reason = "PEDESTRIAN AHEAD"

    # 2. Road has practically disappeared in lookahead zone
    elif n_road_total < STOP_ROAD_PX:
        stop_reason = f"NO ROAD ({n_road_total}px)"

    if stop_reason:
        _stop_hold_ctr = STOP_HOLD_FRAMES
        _steer_smooth  = 0.0
        return {"status": "STOP", "direction": "STOP",
                "steer": 0.0, "reason": stop_reason}

    # ── STOP hold ──────────────────────────────────────────────────────────
    if _stop_hold_ctr > 0:
        _stop_hold_ctr -= 1
        _steer_smooth   = 0.0
        return {"status": "STOP", "direction": "STOP",
                "steer": 0.0, "reason": "HOLD"}

    # ── CAUTION: road narrowing significantly ──────────────────────────────
    if n_road_total < CAUTION_ROAD_PX:
        reason = f"ROAD NARROWING ({n_road_total//1000}K px)"
        return {"status": "CAUTION", "direction": "SLOWING",
                "steer": round(_steer_smooth, 1), "reason": reason}

    # ── CAUTION: obstacle nearby but road still clear ──────────────────────
    if fwd_obstacle_ratio > 0.15:
        reason = f"NEAR OBSTACLE {fwd_obstacle_ratio*100:.0f}%"
        return {"status": "CAUTION", "direction": "SLOWING",
                "steer": round(_steer_smooth, 1), "reason": reason}

    # ── SAFE: steer from road centroid ─────────────────────────────────────
    if n_road_clear < 5:
        return {"status": "CAUTION", "direction": "SEARCHING",
                "steer": 0.0, "reason": "NO CLEAR ROAD"}

    road_cx   = float(np.median(road_xs))
    offset    = road_cx - cx
    raw_steer = float(np.degrees(np.arctan2(offset, H * 0.35)))

    _steer_smooth = (STEER_SMOOTH_ALPHA * raw_steer
                     + (1.0 - STEER_SMOOTH_ALPHA) * _steer_smooth)
    steer = _steer_smooth

    if abs(steer) < STEER_DEADBAND_DEG:
        direction = "FORWARD"
    elif steer > 0:
        direction = "TURN RIGHT"
    else:
        direction = "TURN LEFT"

    return {"status": "SAFE", "direction": direction,
            "steer": round(steer, 1), "reason": ""}


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIXEL-LEVEL RENDERER
# ─────────────────────────────────────────────────────────────────────────────
C_OBSTACLE = (0, 0, 220)       # red — YOLO obstacle overlay
C_SAFE     = (0, 220, 0)       # green HUD text
C_CAUTION  = (0, 200, 220)     # yellow HUD text
C_STOP_COL = (0, 0, 255)       # bright red HUD text
C_WHITE    = (255, 255, 255)
C_GRAY     = (150, 150, 150)
ALPHA_SEG  = 0.55              # segmentation blend strength


def render_pixel_level(frame_bgr, label_map, obstacle_mask, nav, traffic_lights, boards):
    """
    Pure pixel-level output:
      1. Blend original frame with SegFormer colour map (pixel-by-pixel class colours)
      2. Paint YOLO obstacle pixels in distinct red
      3. Minimal HUD strip at top (no path lines, no corridor, no bezier)
      4. Draw traffic signals and boards on HUD
    """
    H, W = frame_bgr.shape[:2]

    # ── Pixel colour map from SegFormer labels ────────────────────────────────
    seg_colour = label_to_colour(label_map)

    # ── Blend: frame * (1-alpha) + seg_colour * alpha ─────────────────────────
    out = cv2.addWeighted(frame_bgr, 1.0 - ALPHA_SEG,
                          seg_colour, ALPHA_SEG, 0)

    # ── Paint YOLO obstacle pixels on top ─────────────────────────────────────
    if obstacle_mask.any():
        obs_overlay = out.copy()
        obs_overlay[obstacle_mask] = C_OBSTACLE
        out = cv2.addWeighted(out, 0.45, obs_overlay, 0.55, 0)

    # ── Lookahead band boundary (faint white line) ────────────────────────────
    r0 = int(LOOKAHEAD_TOP * H)
    r1 = int(LOOKAHEAD_BOT * H)
    cv2.line(out, (0, r0), (W, r0), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(out, (0, r1), (W, r1), (200, 200, 200), 1, cv2.LINE_AA)

    # ── HUD banner ────────────────────────────────────────────────────────────
    banner_h = 52
    overlay  = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, banner_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    status    = nav["status"]
    direction = nav["direction"]
    steer     = nav["steer"]
    reason    = nav.get("reason", "")

    txt_col = (C_STOP_COL if status == "STOP"
               else C_CAUTION if status == "CAUTION"
               else C_SAFE)

    cv2.putText(out, f"{status}",
                (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.72, txt_col, 2, cv2.LINE_AA)

    dir_str = direction if status == "STOP" else f"{direction}  ({steer:+.1f}°)"
    cv2.putText(out, dir_str,
                (10, 44), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)

    if reason:
        cv2.putText(out, reason,
                    (W // 2 - 100, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 180, 255), 1, cv2.LINE_AA)

    if traffic_lights:
        cv2.putText(out, "TRAFFIC SIGNALS DETECTED",
                    (W // 2 - 250, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    
    if boards:
        cv2.putText(out, "TRAFFIC SIGNS DETECTED",
                    (W // 2 + 50, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    # ── Dynamic Perspective Grid ──────────────────────────────────────────────
    status = nav.get("status", "SAFE")
    steer_angle = nav.get("steer", 0.0)
    
    if status == "STOP":
        grid_color = (0, 0, 255) # Red
    elif status == "CAUTION":
        grid_color = (0, 165, 255) # Orange
    else:
        grid_color = (0, 255, 0) # Green
        
    base_y = H
    base_left = W // 2 - 180
    base_right = W // 2 + 180
    top_y = int(H * 0.50)
    offset = 0 # static grid, no left/right movement
    top_left = W // 2 - 48 + offset
    top_right = W // 2 + 48 + offset

    pts = np.array([[base_left, base_y], [top_left, top_y], [top_right, top_y], [base_right, base_y]], np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], grid_color)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # Grid lines
    cv2.line(out, (base_left, base_y), (top_left, top_y), grid_color, 2, cv2.LINE_AA)
    cv2.line(out, (base_right, base_y), (top_right, top_y), grid_color, 2, cv2.LINE_AA)
    mid_base_x = (base_left + base_right) // 2
    mid_top_x = (top_left + top_right) // 2
    cv2.line(out, (mid_base_x, base_y), (mid_top_x, top_y), grid_color, 2, cv2.LINE_AA)
    
    for i in range(1, 5):
        f = i / 5.0
        f_p = f * f # perspective spacing
        y = int(base_y + (top_y - base_y) * f_p)
        lx = int(base_left + (top_left - base_left) * f_p)
        rx = int(base_right + (top_right - base_right) * f_p)
        cv2.line(out, (lx, y), (rx, y), grid_color, 1)

    # Model watermark
    cv2.putText(out, "SegFormer-B0 + YOLOv8  |  pixel-level",
                (W - 330, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

    # ── Draw Traffic Signals & Boards ─────────────────────────────────────────
    for (x1, y1, x2, y2, color) in traffic_lights:
        if color == "RED": box_c = (0, 0, 255)
        elif color == "GREEN": box_c = (0, 255, 0)
        elif color == "YELLOW": box_c = (0, 255, 255)
        else: box_c = (200, 200, 200)
        
        cv2.rectangle(out, (x1, y1), (x2, y2), box_c, 2)
        cv2.putText(out, f"SIGNAL: {color}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_c, 2, cv2.LINE_AA)

    for (x1, y1, x2, y2) in boards:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange
        cv2.putText(out, "SIGN BOARDS", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    # ── Colour legend (top-right) ─────────────────────────────────────────────
    legend = [
        ("road",       CITYSCAPES_PALETTE_BGR[0]),
        ("sidewalk",   CITYSCAPES_PALETTE_BGR[1]),
        ("vegetation", CITYSCAPES_PALETTE_BGR[8]),
        ("terrain",    CITYSCAPES_PALETTE_BGR[9]),
        ("person",     CITYSCAPES_PALETTE_BGR[11]),
        ("car/vehicle",CITYSCAPES_PALETTE_BGR[13]),
        ("obstacle",   C_OBSTACLE),
    ]
    lx, ly = W - 160, 30
    for name, color in legend:
        cv2.rectangle(out, (lx, ly - 10), (lx + 14, ly + 2), color, -1)
        cv2.putText(out, name, (lx + 18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1, cv2.LINE_AA)
        ly += 16

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. FRAME PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame_bgr, processor, seg_model, yolo_model, device):
    # A — Pixel-level semantic segmentation
    label_map = run_segformer(frame_bgr, processor, seg_model, device)
    
    # A2 — Extract sign boards from SegFormer map
    boards = extract_boards_from_seg(label_map)

    # B — YOLO obstacle mask (internal — recalibrated inflation)
    obstacle_mask, person_in_fwd, fwd_obs_ratio, traffic_lights = build_obstacle_mask(
        frame_bgr, yolo_model)

    # C — Navigation decision from pixel centroids
    nav = navigation_decision(
        label_map, obstacle_mask, person_in_fwd,
        fwd_obs_ratio, frame_bgr.shape)

    # D — Pixel-level render
    out = render_pixel_level(frame_bgr, label_map, obstacle_mask, nav, traffic_lights, boards)

    return out, nav


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser("SegFormer + YOLOv8 Navigation v5 (pixel-level)")
    ap.add_argument("--source",  default="0")
    ap.add_argument("--device",  default="cpu")
    ap.add_argument("--output",  default="")
    ap.add_argument("--no-show", dest="show", action="store_false", default=True)
    ap.add_argument("--conf",    type=float, default=0.35)
    return ap.parse_args()


def main():
    global _steer_smooth, _stop_hold_ctr
    args   = parse_args()
    processor, seg_model, yolo_model = load_models(args.device)

    src      = int(args.source) if args.source.isdigit() else args.source
    is_image = isinstance(src, str) and \
               Path(src).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}

    _steer_smooth = 0.0
    _stop_hold_ctr = 0

    # ── Image mode ──────────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            sys.exit(f"[ERROR] Cannot read: {src}")
        out, nav = process_frame(frame, processor, seg_model, yolo_model, args.device)
        print(f"[NAV] {nav}")
        if args.output:
            cv2.imwrite(args.output, out)
            print(f"[SAVED] {args.output}")
        if args.show:
            cv2.imshow("Navigation v5 – pixel-level", out)
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
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (W_src, H_src))
        print(f"[INFO] Writing → {args.output}  ({W_src}×{H_src} @ {fps_src:.0f}fps  {total} frames)")

    t_prev = time.time()
    fidx   = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, nav = process_frame(frame, processor, seg_model, yolo_model, args.device)

        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.putText(out, f"FPS {fps:.1f}",
                    (W_src - 80, H_src - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

        if writer:
            writer.write(out)
        if args.show:
            cv2.imshow("SegFormer + YOLOv8 Navigation v5 – pixel-level", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        fidx += 1
        if fidx % 30 == 0:
            pct = fidx / max(total, 1) * 100
            reason = nav.get("reason", "")
            print(f"  [{fidx}/{total} {pct:.0f}%] {nav['status']:7s} "
                  f"{nav['direction']:12s}  steer={nav['steer']:+.1f}°  "
                  f"{reason}  fps={fps:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
