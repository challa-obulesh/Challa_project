"""
segformer_ego_v13.py - Ego-Motion Navigation with 4-Color Semantic Grouping
===========================================================================
Groups all 19 Cityscapes classes into 4 distinct colors as requested:
1. SAFE ROAD (Green)
2. OBSTACLES (Red)
3. SIDEWALK (Purple)
4. SKY (Blue)
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
# 4-COLOR SEMANTIC GROUPING MAP (BGR Format)
# ═══════════════════════════════════════════════════════════════════════════════
# The user specifically requested 4 colors: Safe, Obstacle, Sidewalk, Sky.
# BGR Colors:
COLOR_SAFE     = (  0, 255,   0)  # Green
COLOR_OBSTACLE = (  0,   0, 255)  # Red
COLOR_SIDEWALK = (255,   0, 255)  # Purple
COLOR_SKY      = (250, 200, 100)  # Light Blue

CLASS_MAP = {
    0:  ("road",          COLOR_SAFE),
    1:  ("sidewalk",      COLOR_SIDEWALK),
    2:  ("building",      COLOR_OBSTACLE),
    3:  ("wall",          COLOR_OBSTACLE),
    4:  ("fence",         COLOR_OBSTACLE),
    5:  ("pole",          COLOR_OBSTACLE),
    6:  ("traffic light", COLOR_OBSTACLE),
    7:  ("traffic sign",  COLOR_OBSTACLE),
    8:  ("vegetation",    COLOR_OBSTACLE),
    9:  ("terrain",       COLOR_SAFE),
    10: ("sky",           COLOR_SKY),
    11: ("person",        COLOR_OBSTACLE),
    12: ("rider",         COLOR_OBSTACLE),
    13: ("car",           COLOR_OBSTACLE),
    14: ("truck",         COLOR_OBSTACLE),
    15: ("bus",           COLOR_OBSTACLE),
    16: ("train",         COLOR_OBSTACLE),
    17: ("motorcycle",    COLOR_OBSTACLE),
    18: ("bicycle",       COLOR_OBSTACLE),
}

NUM_CLASSES = 19
COLOUR_LUT = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
for cid, (name, bgr) in CLASS_MAP.items():
    COLOUR_LUT[cid] = bgr

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
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
                    dxs = good_new[:, 0] - good_old[:, 0]
                    median_dx = np.median(dxs)
                    ego_dx = -(median_dx * 3.0) / (W * 0.1)
                    ego_dx = max(-1.0, min(1.0, ego_dx))
        
        self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, 
                                         qualityLevel=0.1, minDistance=10, blockSize=7)
        self.prev_gray = gray
        return ego_dx

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION & PATH
# ═══════════════════════════════════════════════════════════════════════════════
def build_pixel_overlay(frame, seg_map):
    overlay = COLOUR_LUT[seg_map]
    # Use 50% opacity blend so it's clearly visible but you can still see the road
    return cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

def draw_hud(frame, decision, smooth_offset, fps, f_idx, total):
    H, W = frame.shape[:2]
    panel_h = 70

    cv2.rectangle(frame, (0, 0), (W, panel_h), (10, 10, 10), -1)
    cv2.rectangle(frame, (0, 0), (W, panel_h), (80, 80, 80), 2)

    cv2.putText(frame, "4-Color Semantic Mask | Ego-Motion Tracker",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if "FORWARD" in decision or "STRAIGHT" in decision:
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
    sx, sy = W // 2, H - 10
    max_shift = W // 3
    gx = int(sx + smooth_offset * max_shift)
    gy = sy - PATH_MAX_PX
    
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
    print(f"  V13 4-Color Grouping Pipeline")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"============================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = remove_watermark(frame)

        # 1. Segmentation
        seg_map = segmenter.predict(frame)

        # 2. Ego-Motion Tracking
        raw_ego_offset = tracker.update(frame, H, W)

        road_mask = (seg_map[int(H*0.6):int(H*0.8), :] == 0)
        semantic_offset = 0.0
        if np.any(road_mask):
            road_cols = np.where(road_mask)[1]
            if len(road_cols) > 0:
                road_cx = np.median(road_cols)
                semantic_offset = (road_cx - W/2) / (W/2) * 0.5
        
        raw_offset = 0.8 * raw_ego_offset + 0.2 * semantic_offset

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
