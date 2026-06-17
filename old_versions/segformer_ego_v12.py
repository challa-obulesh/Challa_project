"""
segformer_ego_v12.py - Ego-Motion Navigation (Pixel-Level SegFormer Only)
=========================================================================
1. NO BOUNDING BOXES: Pure pixel-level semantic detection via SegFormer.
2. ROBOT MOVEMENT DECISIONS: Uses Lucas-Kanade Optical Flow to track actual 
   physical camera ego-motion, guaranteeing decisions match physical turning.
"""

import cv2
import numpy as np
import sys
import argparse
import time

try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image
    HAS_SEG = True
except ImportError:
    HAS_SEG = False

# ═══════════════════════════════════════════════════════════════════════════════
# CITYSCAPES 19-CLASS SEMANTIC MAP
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_MAP = {
    0:  ("road",          ( 80,  80,  80)),   # dark gray
    1:  ("sidewalk",      (180,  60, 180)),   # purple
    2:  ("building",      ( 40,  40, 120)),   # dark red
    3:  ("wall",          ( 60,  80, 100)),   # brown
    4:  ("fence",         (140, 140,  60)),   # teal
    5:  ("pole",          (110, 110, 110)),   # gray
    6:  ("traffic light", (  0, 220, 250)),   # yellow
    7:  ("traffic sign",  (  0, 150, 255)),   # orange
    8:  ("vegetation",    ( 30, 140,  30)),   # green
    9:  ("terrain",       ( 70, 150, 100)),   # olive
    10: ("sky",           (230, 200, 160)),   # light blue
    11: ("person",        (  0,   0, 255)),   # pure red
    12: ("rider",         (  0,  60, 255)),   # red-orange
    13: ("car",           (255,  50,  50)),   # pure blue
    14: ("truck",         (200,  20,  20)),   # dark blue
    15: ("bus",           (200,  40, 150)),   # magenta
    16: ("train",         (200, 180,  40)),   # cyan
    17: ("motorcycle",    (180, 100, 200)),   # pink
    18: ("bicycle",       (100,  20, 200)),   # red-violet
}

NUM_CLASSES = 19
COLOUR_LUT = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
for cid, (name, bgr) in CLASS_MAP.items():
    COLOUR_LUT[cid] = bgr

# Classes that represent obstacles we should actively avoid at pixel level
OBSTACLE_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18} # people, vehicles

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
LOOK_TOP = 0.50
LOOK_BOT = 0.85
PATH_MAX_PX = 140
EMA_ALPHA = 0.15
STEER_THRESH = 0.05
IEEE_STRIP_H = 28

# ═══════════════════════════════════════════════════════════════════════════════
# IEEE WATERMARK REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════
def remove_watermark(frame):
    H, W = frame.shape[:2]
    st = H - IEEE_STRIP_H
    src = st - IEEE_STRIP_H
    if src < 0:
        return frame
    patch = frame[src:st, :].copy()
    for i in range(IEEE_STRIP_H):
        a = 1.0 - (i / IEEE_STRIP_H) * 0.25
        frame[st + i, :] = (patch[i, :] * a).astype(np.uint8)
    return frame

# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTER
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
                print(f"[SEG] Loaded on {self.device}")
            except Exception as e:
                print(f"[SEG] Load failed: {e}")

    def predict(self, frame):
        H, W = frame.shape[:2]
        if self.model is None:
            pred = np.full((H, W), 10, dtype=np.uint8)
            pred[H // 2:, :] = 0
            return pred
            
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW (EGO-MOTION)
# ═══════════════════════════════════════════════════════════════════════════════
class EgoMotionTracker:
    def __init__(self):
        self.prev_gray = None
        self.p0 = None
        
    def update(self, frame, H, W):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only track features in the lower middle (the road) to get true ego-motion
        # ignoring sky and far edges
        mask = np.zeros_like(gray)
        mask[int(H*0.4):int(H*0.9), int(W*0.2):int(W*0.8)] = 255
        
        ego_dx = 0.0
        
        if self.prev_gray is not None and self.p0 is not None and len(self.p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, 
                                                  winSize=(15, 15), maxLevel=2)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]
                
                if len(good_new) > 5:
                    # Calculate horizontal shift of all tracked points
                    dxs = good_new[:, 0] - good_old[:, 0]
                    # Median shift (pixels). If pixels moved right (dx>0), camera turned left.
                    median_dx = np.median(dxs)
                    
                    # Convert to a normalized steering offset (-1.0 to 1.0)
                    # dx is usually small (e.g., -5 to 5 pixels per frame)
                    # Amplify and normalize
                    ego_dx = -(median_dx * 3.0) / (W * 0.1)
                    ego_dx = max(-1.0, min(1.0, ego_dx))
        
        # Find new features to track for next frame
        self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, 
                                         qualityLevel=0.1, minDistance=10, blockSize=7)
        self.prev_gray = gray
        return ego_dx

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION & PATH
# ═══════════════════════════════════════════════════════════════════════════════
def build_pixel_overlay(frame, seg_map):
    """
    Apply semantic colors.
    For road/buildings/etc, use 40% opacity.
    For obstacles (cars/people), use 70% opacity so they pop as pixel-level detections.
    """
    overlay = frame.copy()
    
    # Backgrounds
    bg_mask = ~np.isin(seg_map, list(OBSTACLE_CLASSES))
    bg_overlay = COLOUR_LUT[seg_map]
    
    # Blend background
    overlay[bg_mask] = cv2.addWeighted(frame[bg_mask], 0.6, bg_overlay[bg_mask], 0.4, 0)
    
    # Obstacles (Cars, People) - pop brightly
    obs_mask = np.isin(seg_map, list(OBSTACLE_CLASSES))
    obs_overlay = COLOUR_LUT[seg_map]
    overlay[obs_mask] = cv2.addWeighted(frame[obs_mask], 0.2, obs_overlay[obs_mask], 0.8, 0)
    
    return overlay

def draw_hud(frame, decision, smooth_offset, fps, f_idx, total):
    H, W = frame.shape[:2]
    panel_h = 70

    cv2.rectangle(frame, (0, 0), (W, panel_h), (10, 10, 10), -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (80, 80, 80), 2)

    cv2.putText(frame, "Pixel-Level SegFormer | Ego-Motion Navigation",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if "FORWARD" in decision:
        col = (0, 255, 0)
    elif "LEFT" in decision:
        col = (255, 180, 0)
    else:
        col = (0, 150, 255)

    cv2.putText(frame, decision, (12, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)

    pct = abs(smooth_offset) * 100
    side = "R" if smooth_offset >= 0 else "L"
    info = f"Ego-Steer: {pct:.1f}% {side} | FPS: {fps:.1f} | F:{f_idx}/{total}"
    cv2.putText(frame, info, (W - 320, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
                
    return panel_h

def draw_dynamic_path(frame, smooth_offset, H, W):
    """Path exactly follows the ego-motion offset."""
    sx, sy = W // 2, H - 10
    
    # Map the -1.0 to 1.0 offset to a horizontal pixel shift at the goal line
    max_shift = W // 3
    gx = int(sx + smooth_offset * max_shift)
    gy = sy - PATH_MAX_PX
    
    # Bezier curve
    cx1 = sx
    cy1 = sy - (sy - gy) // 2
    cx2 = gx
    cy2 = gy + (sy - gy) // 2

    pts = []
    for t in np.linspace(0, 1, 50):
        t1 = 1 - t
        x = int(t1**3*sx + 3*t1**2*t*cx1 + 3*t1*t**2*cx2 + t**3*gx)
        y = int(t1**3*sy + 3*t1**2*t*cy1 + 3*t1*t**2*cy2 + t**3*gy)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 4, cv2.LINE_AA)
    
    cv2.circle(frame, (gx, gy), 10, (0, 0, 255), -1)
    cv2.circle(frame, (gx, gy), 10, (255, 255, 255), 3)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def process_video(input_path, output_path, seg_model):
    segmenter = Segmenter(seg_model)
    tracker = EgoMotionTracker()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {input_path}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    smooth_offset = 0.0
    frame_idx = 0
    t_start = time.time()

    print(f"\n============================================================")
    print(f"  V12 Ego-Motion (Pixel-Level SegFormer) Pipeline")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"============================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = remove_watermark(frame)

        # 1. Pure Semantic Segmentation
        seg_map = segmenter.predict(frame)

        # 2. True Ego-Motion Tracking (Optical Flow)
        raw_ego_offset = tracker.update(frame, H, W)

        # Blend: We also look slightly at the semantic road center so we don't drift,
        # but ego-motion dictates the turn!
        road_mask = (seg_map[int(H*0.6):int(H*0.8), :] == 0)
        semantic_offset = 0.0
        if np.any(road_mask):
            road_cols = np.where(road_mask)[1]
            if len(road_cols) > 0:
                road_cx = np.median(road_cols)
                semantic_offset = (road_cx - W/2) / (W/2) * 0.5
        
        # 80% True Camera Movement, 20% Semantic road position
        raw_offset = 0.8 * raw_ego_offset + 0.2 * semantic_offset

        # Smooth heavily
        if frame_idx == 0:
            smooth_offset = raw_offset
        else:
            smooth_offset = EMA_ALPHA * raw_offset + (1 - EMA_ALPHA) * smooth_offset

        # 3. Decision
        if abs(smooth_offset) < STEER_THRESH:
            decision = "STRAIGHT"
        elif smooth_offset < 0:
            decision = "TURN LEFT"
        else:
            decision = "TURN RIGHT"

        # 4. Rendering
        vis = build_pixel_overlay(frame, seg_map)
        draw_dynamic_path(vis, smooth_offset, H, W)
        
        elapsed = time.time() - t_start
        current_fps = (frame_idx + 1) / max(elapsed, 0.001)
        draw_hud(vis, decision, smooth_offset, current_fps, frame_idx, total)

        out.write(vis)
        frame_idx += 1

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {decision}  off={smooth_offset:+.3f}")

    cap.release()
    out.release()
    print(f"\nDone -> {output_path}  ({frame_idx} frames in {time.time()-t_start:.1f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--seg-model", default="nvidia/segformer-b0-finetuned-cityscapes-512-1024")
    args = parser.parse_args()
    process_video(args.input, args.output, args.seg_model)
