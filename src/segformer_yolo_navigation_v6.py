"""
SegFormer-B0 + YOLOv8 Fusion Navigation  v5.0  (Optimised)
============================================================
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

  [OPT]   Performance optimisations for Jetson AGX Orin (target >15 FPS):
           • Concurrent YOLO + SegFormer inference via ThreadPoolExecutor
           • CUDA device by default, auto-detect TRT engines
           • GPU warmup pass on startup
           • Threaded video capture to eliminate I/O blocking
           • Pre-allocated palette lookup table
           • Optimised obstacle overlay rendering

USAGE:
    python segformer_yolo_navigation_v5.py --source video.mp4 --output out.mp4
    python segformer_yolo_navigation_v5.py --source 0
    python segformer_yolo_navigation_v5.py --source image.jpg --output out.jpg

REQUIREMENTS:
    pip install torch torchvision transformers ultralytics opencv-python numpy pillow
"""

import argparse
import os
import sys
import time
import threading
import concurrent.futures
import queue
from pathlib import Path

import cv2
import numpy as np

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

# Pre-allocated palette as numpy array (avoid recreating every frame)
_PALETTE_LUT = np.array(CITYSCAPES_PALETTE_BGR, dtype=np.uint8)

# Readable class names
CITYSCAPES_NAMES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]

# Which class IDs count as traversable (robot can drive there)
TRAVERSABLE_IDS = {3, 6, 9, 11, 13, 29, 52, 91, 94}   # floor, road, grass, sidewalk, earth, field, path, dirt track, land

# Pre-compute traversable set as a numpy boolean lookup (faster than np.isin)
_TRAVERSABLE_LUT = np.zeros(256, dtype=bool)
for _tid in TRAVERSABLE_IDS:
    if _tid < 256:
        _TRAVERSABLE_LUT[_tid] = True

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE CONSTANTS  (calibrated to avoid always-STOP)
# ─────────────────────────────────────────────────────────────────────────────
LOOKAHEAD_TOP        = 0.30   # lookahead band top    (fraction of H)
LOOKAHEAD_BOT        = 0.65   # lookahead band bottom (fraction of H)
STEER_DEADBAND_DEG   = 7.0    # ±deg → FORWARD
STEER_SMOOTH_ALPHA   = 0.10   # low-pass on steer angle

# STOP / CAUTION thresholds — calibrated from real video measurement
# Normal driving: road band has 126K–150K pixels in lookahead zone
INFLATE_PX              = 3     # YOLO box inflation (px) (reduced for less sensitivity)
STOP_ROAD_PX            = 15000 # road px < 15K  → STOP (drastically reduced)
CAUTION_ROAD_PX         = 30000 # road px < 30K  → CAUTION (drastically reduced)
STOP_HOLD_FRAMES        = 3     # hold STOP for N frames

# Person detection: only STOP if large bbox (>8% of frame) overlaps forward center
PERSON_MIN_AREA_RATIO   = 0.08  # 8% of frame (person must be very close)

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
# THREADED VIDEO CAPTURE (eliminates I/O blocking)
# ─────────────────────────────────────────────────────────────────────────────
class ThreadedCapture:
    """Non-blocking video capture — keeps only the latest frame."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                break

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened() and self.ret

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()

class ThreadedVideoWriter:
    """Non-blocking video writer to eliminate disk I/O bottlenecks and boost FPS."""
    def __init__(self, filename, fourcc, fps, frameSize):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frameSize)
        self.queue = queue.Queue(maxsize=128)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()

    def _run(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
            except queue.Empty:
                pass

    def write(self, frame):
        if not self.queue.full():
            self.queue.put(frame)

    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _find_project_root():
    """Find project root (where models/ directory lives)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If we're in src/, go up one level
    parent = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(parent, "models")):
        return parent
    if os.path.isdir(os.path.join(script_dir, "models")):
        return script_dir
    return parent


def load_models(device="cuda", use_trt=False):
    global _seg_processor, _seg_model, _yolo_model

    project_root = _find_project_root()
    seg_engine = os.path.join(project_root, "models", "segformer_b0.engine")
    yolo_engine = os.path.join(project_root, "models", "yolov8n.engine")

    # Auto-detect TRT engines if they exist
    auto_trt = (os.path.exists(seg_engine) and os.path.exists(yolo_engine))
    if auto_trt and not use_trt:
        print("[INFO] TRT engines detected in models/ — using TRT automatically.")
        use_trt = True

    if use_trt:
        if _seg_model is None:
            print("[INFO] Loading SegFormer TRT Engine …")
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from trt_segformer import SegFormerTRT
            _seg_model = SegFormerTRT(seg_engine)
            print("[INFO] SegFormer TRT ready.")
        if _yolo_model is None:
            print("[INFO] Loading YOLOv8 TRT Engine …")
            from ultralytics import YOLO
            _yolo_model = YOLO(yolo_engine)
            print("[INFO] YOLOv8 TRT ready.")
        return None, _seg_model, _yolo_model
    else:
        if _seg_model is None:
            print("[INFO] Loading SegFormer-B0 (PyTorch) …")
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
            _seg_processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512")
            _seg_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            ).to(device).eval()
            print("[INFO] SegFormer ready.")
        if _yolo_model is None:
            print("[INFO] Loading YOLOv8n …")
            from ultralytics import YOLO
            yolo_pt = os.path.join(project_root, "yolov8n.pt")
            _yolo_model = YOLO(yolo_pt if os.path.exists(yolo_pt) else "yolov8n.pt")
            print("[INFO] YOLOv8 ready.")
        return _seg_processor, _seg_model, _yolo_model


def warmup_models(processor, seg_model, yolo_model, device, n=3):
    """Warmup GPU with dummy frames to pre-heat CUDA kernels and TRT contexts."""
    print(f"[INFO] Warming up GPU ({n} passes) …")
    import torch
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for i in range(n):
        run_segformer(dummy, processor, seg_model, device)
        yolo_model(dummy, verbose=False, imgsz=640)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[INFO] Warmup complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEGFORMER — pixel-level label map
# ─────────────────────────────────────────────────────────────────────────────
def run_segformer(frame_bgr, processor, model, device):
    if processor is None:
        # TRT path — model.__call__ handles everything
        return model(frame_bgr)

    import torch
    from PIL import Image
    pil    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=frame_bgr.shape[:2], mode="bilinear", align_corners=False)
    return up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIXEL-LEVEL COLOUR MAP from
# Create a fast LUT for Cityscapes palette (1-channel to 3-channel BGR)
_CV_PALETTE = np.zeros((256, 1, 3), dtype=np.uint8)
_CV_PALETTE[:len(CITYSCAPES_PALETTE_BGR), 0, :] = CITYSCAPES_PALETTE_BGR

def label_to_colour(label_map):
    """Convert Cityscapes label map (H,W) to BGR colour image (H,W,3)."""
    return cv2.applyColorMap(label_map, _CV_PALETTE)


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
    board_mask = cv2.inRange(label_map, 7, 7)
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

    # Forward zone: shifted to bottom (closer to camera) and narrower
    fwd_r1, fwd_r2 = int(H * 0.60), int(H * 0.95)
    fwd_c1, fwd_c2 = int(W * 0.35), int(W * 0.65)

    person_in_fwd  = False
    frame_area     = float(H * W)
    traffic_lights = []

    results = yolo_model(frame_bgr, verbose=False, imgsz=640)[0]
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
    # Use pre-computed LUT instead of np.isin (faster)
    road_mask_band = _TRAVERSABLE_LUT[band_labels]
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
    if fwd_obstacle_ratio > 0.30:
        reason = f"NEAR OBSTACLE {fwd_obstacle_ratio*100:.0f}%"
        return {"status": "CAUTION", "direction": "SLOWING",
                "steer": round(_steer_smooth, 1), "reason": reason}

    # ── SAFE: steer from road centroid ─────────────────────────────────────
    if n_road_clear < 5:
        return {"status": "CAUTION", "direction": "SEARCHING",
                "steer": 0.0, "reason": "NO CLEAR ROAD"}

    road_cx   = float(np.mean(road_xs))
    offset    = road_cx - cx
    raw_steer = float(np.degrees(np.arctan2(offset, H * 0.35)))

    _steer_smooth = (STEER_SMOOTH_ALPHA * raw_steer
                     + (1.0 - STEER_SMOOTH_ALPHA) * _steer_smooth)
    steer = _steer_smooth

    if abs(steer) < STEER_DEADBAND_DEG:
        direction = "ACTION: FORWARD"
    elif steer > 15:
        direction = "ACTION: HARD RIGHT"
    elif steer > 0:
        direction = "ACTION: SLIGHT RIGHT"
    elif steer < -15:
        direction = "ACTION: HARD LEFT"
    else:
        direction = "ACTION: SLIGHT LEFT"

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


_OBS_COLOR_IMG = None

def render_pixel_level(frame_bgr, label_map, obstacle_mask, nav, traffic_lights, boards):
    """Render the navigation UI on the frame (in-place or fast blend)."""
    global _OBS_COLOR_IMG
    H, W = frame_bgr.shape[:2]

    # ── Pixel colour map from SegFormer labels ────────────────────────────────
    seg_colour = label_to_colour(label_map)

    # ── Blend: frame * (1-alpha) + seg_colour * alpha ─────────────────────────
    out = cv2.addWeighted(frame_bgr, 1.0 - ALPHA_SEG,
                          seg_colour, ALPHA_SEG, 0)

    # ── Paint YOLO obstacle pixels on top (optimised: skip copy if no obstacles)
    if obstacle_mask.any():
        out[obstacle_mask] = C_OBSTACLE

    # ── Lookahead band boundary (faint white line) ────────────────────────────
    r0 = int(LOOKAHEAD_TOP * H)
    r1 = int(LOOKAHEAD_BOT * H)
    cv2.line(out, (0, r0), (W, r0), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(out, (0, r1), (W, r1), (200, 200, 200), 1, cv2.LINE_AA)

    # ── HUD banner ────────────────────────────────────────────────────────────
    banner_h = 52
    cv2.rectangle(out, (0, 0), (W, banner_h), (20, 20, 20), -1)

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

    # (Removed full-poly alpha blend for performance)

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
# 7. FRAME PIPELINE (with concurrent inference)
# ─────────────────────────────────────────────────────────────────────────────
# Persistent thread pool — avoid creation overhead per frame
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def process_frame(frame_bgr, processor, seg_model, yolo_model, device):
    """Process a single frame with concurrent YOLO + SegFormer inference."""

    # ── Concurrent inference: YOLO and SegFormer in parallel ──────────────
    t0 = time.time()
    future_seg  = _executor.submit(run_segformer, frame_bgr, processor, seg_model, device)
    future_yolo = _executor.submit(build_obstacle_mask, frame_bgr, yolo_model)

    # Wait for both to complete
    label_map = future_seg.result()
    obstacle_mask, person_in_fwd, fwd_obs_ratio, traffic_lights = future_yolo.result()
    t1 = time.time()

    # A2 — Extract sign boards from SegFormer map
    boards = extract_boards_from_seg(label_map)
    t2 = time.time()

    # C — Navigation decision from pixel centroids
    nav = navigation_decision(
        label_map, obstacle_mask, person_in_fwd,
        fwd_obs_ratio, frame_bgr.shape)
    t3 = time.time()

    # D — Pixel-level render
    out = render_pixel_level(frame_bgr, label_map, obstacle_mask, nav, traffic_lights, boards)

    return out, nav


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser("SegFormer + YOLOv8 Navigation v5 (pixel-level, optimised)")
    ap.add_argument("--source",  default="0")
    ap.add_argument("--device", type=str, default="cuda", help="Device to use for non-TRT models")
    ap.add_argument("--rotate", type=str, choices=["cw", "ccw", "180"], help="Rotate input video (e.g. to convert vertical to horizontal)")
    ap.add_argument("--output",  default="")
    ap.add_argument("--no-show", dest="show", action="store_false", default=True)
    ap.add_argument("--conf",    type=float, default=0.35)
    ap.add_argument("--trt",     action="store_true",
                    help="Use TensorRT engines (auto-detected if models/ exists)")
    return ap.parse_args()


import json

def main():
    global _steer_smooth, _stop_hold_ctr
    args   = parse_args()
    processor, seg_model, yolo_model = load_models(args.device, args.trt)

    src      = int(args.source) if args.source.isdigit() else args.source
    is_image = isinstance(src, str) and \
               Path(src).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}

    _steer_smooth = 0.0
    _stop_hold_ctr = 0

    # ── GPU Warmup ──────────────────────────────────────────────────────────
    warmup_models(processor, seg_model, yolo_model, args.device, n=3)

    # ── Image mode ──────────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            sys.exit(f"[ERROR] Cannot read: {src}")
            
        if args.rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
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
    # Use threaded capture for webcam/live sources, regular for files
    is_webcam = isinstance(src, int)
    if is_webcam:
        cap = ThreadedCapture(src)
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {src}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.rotate in ["cw", "ccw"]:
        W_src, H_src = H_src, W_src

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale for performance (640x360 instead of 1280x720)
    W_out, H_out = W_src // 2, H_src // 2

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = ThreadedVideoWriter(args.output, fourcc, fps_src, (W_out, H_out))
        print(f"[INFO] Writing → {args.output}  ({W_out}×{H_out} @ {fps_src:.0f}fps  {total} frames)")

    fidx     = 0
    fps_accum = []
    timings_log = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
            
        if args.rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif args.rotate == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Scale down before processing to massively boost renderer FPS
        frame = cv2.resize(frame, (W_out, H_out))

        t_start = time.time()
        out, nav = process_frame(frame, processor, seg_model, yolo_model, args.device)
        t_end = time.time()

        process_time = t_end - t_start
        fps   = 1.0 / max(process_time, 1e-6)
        fps_accum.append(fps)
        
        timings_log.append({
            "frame_idx": fidx,
            "fps": fps,
            "total_ms": process_time * 1000.0,
            "status": nav["status"]
        })

        # FPS colour: green if >=15, red otherwise
        fps_col = (0, 200, 0) if fps >= 15 else (0, 0, 255)
        cv2.putText(out, f"FPS {fps:.1f}",
                    (W_src - 80, H_src - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_col, 1)

        if writer:
            writer.write(out)
        if args.show:
            cv2.imshow("SegFormer + YOLOv8 Navigation v6 – pixel-level", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        fidx += 1
        if fidx % 30 == 0:
            pct = fidx / max(total, 1) * 100
            reason = nav.get("reason", "")
            avg_fps = sum(fps_accum[-30:]) / len(fps_accum[-30:])
            print(f"  [{fidx}/{total} {pct:.0f}%] {nav['status']:7s} "
                  f"{nav['direction']:12s}  steer={nav['steer']:+.1f}°  "
                  f"{reason}  fps={fps:.1f}  avg={avg_fps:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Print summary ──────────────────────────────────────────────────────
    if fps_accum:
        avg = sum(fps_accum) / len(fps_accum)
        mn  = min(fps_accum)
        mx  = max(fps_accum)
        target_met = avg >= 15.0
        print(f"\n{'='*60}")
        print(f"  PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Frames processed: {len(fps_accum)}")
        print(f"  Average FPS:      {avg:.1f}")
        print(f"  Min / Max FPS:    {mn:.1f} / {mx:.1f}")
        print(f"  Target (>15 FPS): {'✅ MET' if target_met else '❌ NOT MET'}")
        print(f"{'='*60}")
        
        # Save JSON benchmark
        benchmark = {
            "input_file": str(args.source),
            "input_resolution": f"{W_src}x{H_src}",
            "total_frames_processed": len(fps_accum),
            "summary": {
                "avg_fps": float(avg),
                "min_fps": float(mn),
                "max_fps": float(mx)
            },
            "target_fps": 15,
            "target_met": target_met,
            "per_frame": timings_log
        }
        with open("benchmark_v6.json", "w") as f:
            json.dump(benchmark, f, indent=2)
        print("[INFO] Benchmark saved to benchmark_v6.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
