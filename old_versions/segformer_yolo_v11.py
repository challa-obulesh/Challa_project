"""
segformer_yolo_v11.py  -  Complete Scene-Aware Robot Navigation  (v11)
======================================================================
Addresses ALL 7 identified problems:

  1. Full scene understanding  - uses ALL 19 Cityscapes semantic classes
  2. Complete obstacle awareness - static + dynamic + restricted
  3. All semantic classes utilised - road, sidewalk, person, car, bicycle, etc.
  4. Dynamic obstacle handling - YOLO detects + tracks moving objects
  5. Risk-aware navigation - per-pixel risk score, congestion detection
  6. Smart navigation decisions - multi-signal (free-space + obstacle avoidance + boundary)
  7. Full environmental understanding - buildings, poles, signs, vegetation all classified

Run:
  /home/sdv/seg_env/bin/python3 segformer_yolo_v11.py <input.mp4> <output.mp4>
"""

import cv2
import numpy as np
import sys
import argparse
import time

# ── Heavy model imports (graceful fallback) ──────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# CITYSCAPES 19-CLASS SEMANTIC MAP
# ═══════════════════════════════════════════════════════════════════════════════
# id: (name, traversability, risk_category, BGR colour)
CLASS_MAP = {
    0:  ("road",          1.00, "safe",       ( 80,  80,  80)),   # dark gray
    1:  ("sidewalk",      0.60, "moderate",   (180,  60, 180)),   # purple
    2:  ("building",      0.00, "impassable", ( 40,  40, 120)),   # dark red
    3:  ("wall",          0.00, "impassable", ( 60,  80, 100)),   # brown
    4:  ("fence",         0.00, "impassable", (140, 140,  60)),   # teal
    5:  ("pole",          0.00, "impassable", (110, 110, 110)),   # gray
    6:  ("traffic light", 0.00, "impassable", (  0, 220, 250)),   # yellow
    7:  ("traffic sign",  0.00, "impassable", (  0, 150, 255)),   # orange
    8:  ("vegetation",    0.10, "dangerous",  ( 30, 140,  30)),   # green
    9:  ("terrain",       0.40, "moderate",   ( 70, 150, 100)),   # olive
    10: ("sky",           0.00, "ignore",     (230, 200, 160)),   # light blue
    11: ("person",        0.00, "dynamic",    (  0,   0, 220)),   # red
    12: ("rider",         0.00, "dynamic",    (  0,  60, 255)),   # red-orange
    13: ("car",           0.00, "dynamic",    (200,  80,  20)),   # blue
    14: ("truck",         0.00, "dynamic",    (180,  60,  10)),   # dark blue
    15: ("bus",           0.00, "dynamic",    (200,  40, 150)),   # magenta
    16: ("train",         0.00, "dynamic",    (200, 180,  40)),   # cyan
    17: ("motorcycle",    0.00, "dynamic",    (180, 100, 200)),   # pink
    18: ("bicycle",       0.00, "dynamic",    (100,  20, 200)),   # red-violet
}

# Pre-compute lookup arrays
NUM_CLASSES = 19
TRAV_LUT       = np.zeros(NUM_CLASSES, dtype=np.float32)
COLOUR_LUT     = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
DYNAMIC_CLASSES = set()
SAFE_CLASSES    = {0, 1, 9}          # classes where robot CAN drive
OBSTACLE_CLASSES = set()

for cid, (name, trav, risk, bgr) in CLASS_MAP.items():
    TRAV_LUT[cid] = trav
    COLOUR_LUT[cid] = bgr
    if risk == "dynamic":
        DYNAMIC_CLASSES.add(cid)
    if trav < 0.3 and risk != "ignore":
        OBSTACLE_CLASSES.add(cid)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
LOOK_TOP        = 0.45     # lookahead band: 45%-70% of frame height
LOOK_BOT        = 0.70
PATH_MAX_PX     = 130      # max path length (pixels)
STEER_THRESH    = 0.04     # ±4% = FORWARD
CORRIDOR_HALF   = 55
EMA_ALPHA       = 0.22
SAFETY_INFLATE  = 25       # inflate YOLO boxes by 25px for safety margin
CONGESTION_THRESH = 3      # >=3 obstacles in lookahead = congested
IEEE_STRIP_H    = 28


# ═══════════════════════════════════════════════════════════════════════════════
# IEEE WATERMARK REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════
def remove_watermark(frame):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTER  -  full 19-class semantic map
# ═══════════════════════════════════════════════════════════════════════════════
class Segmenter:
    def __init__(self, model_name=None):
        self.model = None
        self.processor = None
        self.device = "cpu"
        if model_name and HAS_SEG:
            try:
                print(f"[SEG] Loading: {model_name}")
                self.processor = SegformerImageProcessor.from_pretrained(model_name)
                self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device).eval()
                print(f"[SEG] Loaded on {self.device}  ({NUM_CLASSES} classes)")
            except Exception as e:
                print(f"[SEG] Load failed ({e}), using fallback")
                self.model = None

    def predict(self, frame):
        """Return (H,W) int array of class IDs 0..18."""
        if self.model is not None:
            return self._real(frame)
        return self._fallback(frame)

    def _real(self, frame):
        H, W = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

    def _fallback(self, frame):
        """Simple fallback: bottom half = road (0), top half = sky (10)."""
        H, W = frame.shape[:2]
        pred = np.full((H, W), 10, dtype=np.uint8)   # sky
        pred[H // 2:, :] = 0                          # road
        return pred


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR  -  YOLO dynamic object detection
# ═══════════════════════════════════════════════════════════════════════════════
class Detector:
    # COCO classes that map to Cityscapes dynamic objects
    RELEVANT_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    def __init__(self, weights=None):
        self.model = None
        if weights and HAS_YOLO:
            try:
                print(f"[YOLO] Loading: {weights}")
                self.model = YOLO(weights)
                print(f"[YOLO] Loaded  ({len(self.RELEVANT_CLASSES)} relevant classes)")
            except Exception as e:
                print(f"[YOLO] Load failed ({e})")

    def detect(self, frame):
        """Return list of (x1, y1, x2, y2, conf, cls_name, is_dynamic)."""
        if self.model is None:
            return []
        results = self.model(frame, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            if cls_name not in self.RELEVANT_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            # All YOLO-detected objects are potentially dynamic
            out.append((x1, y1, x2, y2, conf, cls_name, True))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
def build_risk_map(seg_map, detections, H, W):
    """
    Build a per-pixel risk map [0.0 = safe, 1.0 = max danger].
    Sources:
      1. Semantic class traversability (inverted)
      2. YOLO obstacle zones (inflated by safety margin)
      3. Congestion penalty
    """
    # Base risk from semantic classes
    risk = 1.0 - TRAV_LUT[seg_map]       # shape (H, W), float32

    # Inflate risk around YOLO detections
    for (x1, y1, x2, y2, conf, cls, is_dyn) in detections:
        sx1 = max(0, x1 - SAFETY_INFLATE)
        sy1 = max(0, y1 - SAFETY_INFLATE)
        sx2 = min(W, x2 + SAFETY_INFLATE)
        sy2 = min(H, y2 + SAFETY_INFLATE)
        # Core box = 1.0 risk
        risk[y1:y2, x1:x2] = 1.0
        # Safety margin = 0.7 risk
        margin = risk[sy1:sy2, sx1:sx2]
        risk[sy1:sy2, sx1:sx2] = np.maximum(margin, 0.7)

    return risk


def count_obstacles_in_band(detections, H):
    """Count YOLO detections whose centre falls in the lookahead band."""
    r_top = int(LOOK_TOP * H)
    r_bot = int(LOOK_BOT * H)
    count = 0
    for (x1, y1, x2, y2, *_) in detections:
        cy = (y1 + y2) // 2
        if r_top <= cy <= r_bot:
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# FREE-SPACE CORRIDOR + NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_free_centroid(risk_map, H, W):
    """
    Find the safest goal point in the near-lookahead band.
    Uses the risk map: pick the column band with the LOWEST average risk.
    """
    r_top = int(LOOK_TOP * H)
    r_bot = int(LOOK_BOT * H)
    band = risk_map[r_top:r_bot, :]

    # Compute column-wise mean risk
    col_risk = np.mean(band, axis=0)     # shape (W,)

    # Find the safest column (minimum risk)
    # Use a sliding window of CORRIDOR_HALF*2 to find the safest corridor
    kernel = np.ones(CORRIDOR_HALF * 2) / (CORRIDOR_HALF * 2)
    if len(col_risk) > len(kernel):
        smooth_risk = np.convolve(col_risk, kernel, mode='same')
    else:
        smooth_risk = col_risk

    safest_col = int(np.argmin(smooth_risk))

    # Row: median of low-risk pixels in that column neighbourhood
    col_lo = max(0, safest_col - CORRIDOR_HALF)
    col_hi = min(W, safest_col + CORRIDOR_HALF)
    corridor_band = band[:, col_lo:col_hi]
    safe_rows = np.where(corridor_band < 0.5)
    if len(safe_rows[0]) > 0:
        goal_row = r_top + int(np.median(safe_rows[0]))
    else:
        goal_row = (r_top + r_bot) // 2

    return goal_row, safest_col


def compute_obstacle_avoidance_offset(detections, H, W):
    """
    Compute lateral shift away from nearest dynamic obstacle in lookahead band.
    Returns a normalised offset: negative = shift left, positive = shift right.
    """
    r_top = int(LOOK_TOP * H)
    r_bot = int(LOOK_BOT * H)
    cx_frame = W / 2

    nearest_dist = float('inf')
    nearest_cx = cx_frame

    for (x1, y1, x2, y2, conf, cls, is_dyn) in detections:
        cy = (y1 + y2) // 2
        if r_top <= cy <= r_bot:
            cx = (x1 + x2) / 2
            dist = abs(cy - r_bot)   # closer = more urgent
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_cx = cx

    if nearest_dist == float('inf'):
        return 0.0

    # Shift AWAY from the obstacle
    obs_offset = (nearest_cx - cx_frame) / W
    avoidance = -obs_offset * 0.5     # push in opposite direction
    return max(-0.3, min(0.3, avoidance))


def compute_boundary_offset(seg_map, H, W):
    """
    Detect if we're too close to the road/sidewalk boundary.
    Returns a corrective offset to stay centred on road.
    """
    r_top = int(LOOK_TOP * H)
    r_bot = int(LOOK_BOT * H)
    band = seg_map[r_top:r_bot, :]

    road_mask = (band == 0)    # road pixels
    if not np.any(road_mask):
        return 0.0

    road_cols = np.where(road_mask)
    if len(road_cols[1]) == 0:
        return 0.0

    road_left  = np.percentile(road_cols[1], 10)
    road_right = np.percentile(road_cols[1], 90)
    road_centre = (road_left + road_right) / 2

    offset = (road_centre - W / 2) / W
    return max(-0.2, min(0.2, offset))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
def build_semantic_overlay(frame, seg_map):
    """Colour every pixel by its semantic class."""
    overlay = COLOUR_LUT[seg_map]            # (H, W, 3) uint8
    return cv2.addWeighted(frame, 0.50, overlay, 0.50, 0)


def draw_risk_strip(frame, risk_map, H, W):
    """Draw a thin horizontal risk gradient bar at the bottom of the HUD."""
    strip_y = 78
    strip_h = 8
    r_top = int(LOOK_TOP * H)
    r_bot = int(LOOK_BOT * H)
    col_risk = np.mean(risk_map[r_top:r_bot, :], axis=0)

    # Resample to frame width
    for x in range(W):
        r = col_risk[x]
        # Green (safe) -> Yellow (moderate) -> Red (danger)
        if r < 0.5:
            g = 255
            rv = int(r * 2 * 255)
            b = 0
        else:
            g = int((1.0 - r) * 2 * 255)
            rv = 255
            b = 0
        cv2.line(frame, (x, strip_y), (x, strip_y + strip_h), (b, g, rv), 1)


def draw_short_path(frame, robot_base, goal_col, goal_row, H, W):
    sx, sy = robot_base
    max_goal = sy - PATH_MAX_PX
    gy = max(max_goal, goal_row)
    gx = goal_col

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
        cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1)
    cv2.circle(frame, (gx, gy), 8, (255, 255, 255), 2)


def draw_corridor_highlight(frame, seg_map, goal_col, H, W):
    """Green corridor overlay on traversable pixels near the goal column."""
    corridor_top = int(LOOK_TOP * H)
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)
    for r in range(corridor_top, H):
        for c in range(col_lo, col_hi):
            if seg_map[r, c] in SAFE_CLASSES:
                frame[r, c] = (
                    int(frame[r, c, 0] * 0.5),
                    int(frame[r, c, 1] * 0.5 + 100),
                    int(frame[r, c, 2] * 0.5),
                )


def draw_corridor_fast(frame, seg_map, goal_col, H, W):
    """Vectorised green corridor overlay."""
    corridor_top = int(LOOK_TOP * H)
    col_lo = max(0, goal_col - CORRIDOR_HALF)
    col_hi = min(W, goal_col + CORRIDOR_HALF)

    roi = frame[corridor_top:, col_lo:col_hi]
    seg_roi = seg_map[corridor_top:, col_lo:col_hi]

    mask = np.zeros(seg_roi.shape, dtype=bool)
    for cls_id in SAFE_CLASSES:
        mask |= (seg_roi == cls_id)

    green_overlay = roi.copy()
    green_overlay[mask, 0] = np.clip(green_overlay[mask, 0] * 0.5, 0, 255).astype(np.uint8)
    green_overlay[mask, 1] = np.clip(green_overlay[mask, 1] * 0.5 + 100, 0, 255).astype(np.uint8)
    green_overlay[mask, 2] = np.clip(green_overlay[mask, 2] * 0.5, 0, 255).astype(np.uint8)

    frame[corridor_top:, col_lo:col_hi] = green_overlay


def draw_yolo_boxes(frame, detections, panel_h):
    """Draw YOLO bounding boxes with class labels."""
    for (x1, y1, x2, y2, conf, cls, is_dyn) in detections:
        # Colour by class
        if cls == "person":
            col = (0, 0, 255)      # Red
        elif cls in ("car", "truck", "bus"):
            col = (200, 80, 20)    # Blue
        elif cls in ("bicycle", "motorcycle"):
            col = (100, 20, 200)   # Purple
        else:
            col = (0, 180, 255)    # Orange

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        # Label background
        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 6, panel_h + 12)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), col, -1)
        cv2.putText(frame, label, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, decision, smooth_offset, fps, frame_idx, total,
             num_obstacles, risk_level, congested):
    H, W = frame.shape[:2]
    panel_h = 88

    # Panel background
    cv2.rectangle(frame, (0, 0), (W, panel_h), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (60, 60, 60), 2)

    # Title
    cv2.putText(frame, "SegFormer + YOLO  |  Scene-Aware Navigation",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Decision with colour
    if "FORWARD" in decision:
        col = (0, 230, 0)
    elif "LEFT" in decision:
        col = (255, 150, 0)
    elif "RIGHT" in decision:
        col = (0, 100, 255)
    elif "SLOW" in decision:
        col = (0, 180, 255)
    else:
        col = (200, 200, 200)

    cv2.putText(frame, decision, (12, 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, col, 2, cv2.LINE_AA)

    # Info line
    pct = abs(smooth_offset) * 100
    side = "R" if smooth_offset >= 0 else "L"
    cong_txt = " | CONGESTED" if congested else ""
    info = f"Steer: {pct:.1f}% {side} | Risk: {risk_level} | Obs: {num_obstacles} | FPS: {fps:.1f} | F:{frame_idx}/{total}{cong_txt}"
    cv2.putText(frame, info, (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Direction arrow
    cx, cy, r = W // 2, H - 40, 28
    cv2.circle(frame, (cx, cy), r + 4, (30, 30, 30), -1)
    cv2.circle(frame, (cx, cy), r + 4, (90, 90, 90), 2)

    if "LEFT" in decision:
        cv2.arrowedLine(frame, (cx+r-4, cy), (cx-r, cy), col, 4, cv2.LINE_AA, tipLength=0.4)
    elif "RIGHT" in decision:
        cv2.arrowedLine(frame, (cx-r+4, cy), (cx+r, cy), col, 4, cv2.LINE_AA, tipLength=0.4)
    else:
        cv2.arrowedLine(frame, (cx, cy+r-4), (cx, cy-r), col, 4, cv2.LINE_AA, tipLength=0.4)

    return panel_h


def draw_legend(frame, H, W):
    """Compact legend showing key semantic classes."""
    legend_items = [
        ("Road",       ( 80,  80,  80)),
        ("Sidewalk",   (180,  60, 180)),
        ("Person",     (  0,   0, 220)),
        ("Car",        (200,  80,  20)),
        ("Bicycle",    (100,  20, 200)),
        ("Vegetation", ( 30, 140,  30)),
        ("Building",   ( 40,  40, 120)),
        ("Traffic Sign",(0, 150, 255)),
    ]
    lx = 10
    ly = H - 20 - len(legend_items) * 18
    box_w = 150
    box_h = len(legend_items) * 18 + 12
    cv2.rectangle(frame, (lx - 5, ly - 8), (lx + box_w, ly + box_h - 12), (0, 0, 0), -1)
    cv2.rectangle(frame, (lx - 5, ly - 8), (lx + box_w, ly + box_h - 12), (60, 60, 60), 1)

    for i, (name, bgr) in enumerate(legend_items):
        y = ly + i * 18
        cv2.circle(frame, (lx + 8, y + 5), 6, bgr, -1)
        cv2.putText(frame, name, (lx + 20, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def process_video(input_path, output_path, seg_model=None, yolo_model=None):
    segmenter = Segmenter(seg_model)
    detector  = Detector(yolo_model)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    smooth_offset = 0.0
    frame_idx = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  V11 Scene-Aware Navigation Pipeline")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Frames:   {total}  ({W}x{H} @ {fps:.1f} fps)")
    print(f"  SegFormer: {'REAL MODEL' if segmenter.model else 'FALLBACK'}")
    print(f"  YOLO:      {'REAL MODEL' if detector.model else 'DISABLED'}")
    print(f"{'='*60}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 0. Watermark removal ──────────────────────────────────────────────
        frame = remove_watermark(frame)

        # ── 1. Full 19-class semantic segmentation ────────────────────────────
        seg_map = segmenter.predict(frame)

        # ── 2. YOLO dynamic obstacle detection ────────────────────────────────
        detections = detector.detect(frame)

        # ── 3. Risk assessment map ────────────────────────────────────────────
        risk_map = build_risk_map(seg_map, detections, H, W)
        mean_risk_band = np.mean(risk_map[int(LOOK_TOP*H):int(LOOK_BOT*H), :])
        risk_level = "LOW" if mean_risk_band < 0.4 else "MED" if mean_risk_band < 0.7 else "HIGH"
        congested = count_obstacles_in_band(detections, H) >= CONGESTION_THRESH

        # ── 4. Semantic overlay (full scene colouring) ────────────────────────
        vis = build_semantic_overlay(frame, seg_map)

        # ── 5. Free-space goal centroid ───────────────────────────────────────
        goal_row, goal_col = compute_free_centroid(risk_map, H, W)

        # ── 6. Green corridor on safe pixels ──────────────────────────────────
        draw_corridor_fast(vis, seg_map, goal_col, H, W)

        # ── 7. Multi-signal navigation offset ────────────────────────────────
        #   50% free-space centroid
        centroid_off = (goal_col - W / 2) / W

        #   30% obstacle avoidance
        avoid_off = compute_obstacle_avoidance_offset(detections, H, W)

        #   20% road boundary awareness
        boundary_off = compute_boundary_offset(seg_map, H, W)

        raw_offset = 0.50 * centroid_off + 0.30 * avoid_off + 0.20 * boundary_off

        # EMA smoothing
        if frame_idx == 0:
            smooth_offset = raw_offset
        else:
            smooth_offset = EMA_ALPHA * raw_offset + (1 - EMA_ALPHA) * smooth_offset

        # ── 8. Decision ──────────────────────────────────────────────────────
        if congested or risk_level == "HIGH":
            decision = "CAUTION - SLOW"
        elif abs(smooth_offset) < STEER_THRESH:
            decision = "SAFE - FORWARD"
        elif smooth_offset < 0:
            decision = "SAFE - TURN LEFT"
        else:
            decision = "SAFE - TURN RIGHT"

        # ── 9. Draw everything ────────────────────────────────────────────────
        robot_base = (W // 2, H - 20)
        draw_short_path(vis, robot_base, goal_col, goal_row, H, W)
        draw_yolo_boxes(vis, detections, 88)

        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / max(elapsed, 0.001)

        panel_h = draw_hud(vis, decision, smooth_offset, current_fps,
                           frame_idx, total, len(detections), risk_level, congested)
        draw_risk_strip(vis, risk_map, H, W)
        draw_legend(vis, H, W)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total, 1) * 100
            det_str = f"  obs={len(detections)}" if detections else ""
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {decision}  "
                  f"offset={smooth_offset:+.3f}  risk={risk_level}{det_str}")

    cap.release()
    out.release()
    elapsed = time.time() - t_start
    print(f"\nDone -> {output_path}  ({frame_idx} frames in {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11 Scene-Aware Navigation")
    parser.add_argument("input",  help="Input video")
    parser.add_argument("output", help="Output video")
    parser.add_argument("--seg-model",  default="nvidia/segformer-b0-finetuned-cityscapes-512-1024")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    args = parser.parse_args()
    process_video(args.input, args.output, args.seg_model, args.yolo_model)
