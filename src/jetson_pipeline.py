"""
Jetson AGX Orin — Optimised Navigation Pipeline.

Combines TensorRT-accelerated SegFormer + YOLO for real-time semantic
segmentation, obstacle detection, traversability scoring, and A* path planning.

Target: >15 FPS on Jetson AGX Orin with 720×1280 input.

Architecture:
    Frame → [YOLO TRT] → detections ─┐
         → [SegFormer TRT] → seg_map ─┤
                                       ├→ Traversability Map
                                       ├→ A* Path Planning
                                       └→ 4-Panel Visualisation
"""

import os
import sys
import time
import threading
from collections import deque
import concurrent.futures

import cv2
import numpy as np
import torch

from ultralytics import YOLO

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_segformer import SegFormerTRT
from traversability import (
    seg_to_traversability,
    apply_distance_weighting,
    seg_to_colourmap,
    traversability_to_heatmap,
    inject_yolo_obstacles,
    draw_path_on_frame,
    AStarPlanner,
)


# ── Threaded Video Capture ───────────────────────────────────────────────────

class ThreadedCapture:
    """
    Non-blocking video capture using a background thread.
    Keeps only the latest frame to avoid buffering lag.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        # Read first frame
        self.ret, self.frame = self.cap.read()

        # Start background thread
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


# ── YOLO Detection Wrapper ──────────────────────────────────────────────────

# Classes that are considered obstacles for navigation
OBSTACLE_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign",
    13: "bench", 16: "dog", 17: "horse",
}


def parse_yolo_detections(results, conf_thresh: float = 0.35) -> list:
    """
    Parse YOLO results into a list of detection dicts.

    Returns:
        List of {'bbox': (x1,y1,x2,y2), 'conf': float, 'cls': int, 'name': str}
    """
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                if conf >= conf_thresh and cls in OBSTACLE_CLASSES:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf,
                        "cls": cls,
                        "name": OBSTACLE_CLASSES.get(cls, f"class_{cls}"),
                    })
    return detections


def draw_yolo_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw YOLO detection boxes with labels on a frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['name']} {det['conf']:.2f}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2, cv2.LINE_AA)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 140, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ── Main Pipeline ────────────────────────────────────────────────────────────

class NavigationPipeline:
    """
    Full navigation pipeline optimised for Jetson AGX Orin.

    Processes each frame through:
      1. YOLO object detection
      2. SegFormer semantic segmentation
      3. Traversability scoring
      4. A* path planning
      5. 4-panel grid visualisation
    """

    def __init__(
        self,
        yolo_path: str = "models/yolov8n.engine",
        segformer_path: str = "models/segformer_b0.engine",
        yolo_weights_fallback: str = "yolov8n.pt",
        seg_input_size: tuple = (512, 512),
        grid_size: tuple = (40, 24),
        conf_thresh: float = 0.35,
        panel_scale: float = 0.5,
    ):
        """
        Args:
            yolo_path:            Path to YOLO TRT engine.
            segformer_path:       Path to SegFormer TRT engine.
            yolo_weights_fallback: Fallback .pt weights if no engine.
            seg_input_size:       SegFormer input resolution.
            grid_size:            A* planning grid (rows, cols).
            conf_thresh:          YOLO confidence threshold.
            panel_scale:          Scale factor for each panel in the grid.
        """
        print("=" * 60)
        print("  Navigation Pipeline — Jetson AGX Orin")
        print("=" * 60)

        # ── Load YOLO ───────────────────────────────────────────────
        if os.path.exists(yolo_path):
            print(f"[Pipeline] Loading YOLO TRT: {yolo_path}")
            self.yolo = YOLO(yolo_path, task="detect")
        elif os.path.exists(yolo_weights_fallback):
            print(f"[Pipeline] Loading YOLO PyTorch: {yolo_weights_fallback}")
            self.yolo = YOLO(yolo_weights_fallback)
        else:
            raise FileNotFoundError(
                f"No YOLO model found at {yolo_path} or {yolo_weights_fallback}"
            )

        # ── Load SegFormer ──────────────────────────────────────────
        self.segformer = SegFormerTRT(
            engine_path=segformer_path,
            input_size=seg_input_size,
        )

        # ── A* Planner ──────────────────────────────────────────────
        self.planner = AStarPlanner(grid_size=grid_size)

        # ── Config ──────────────────────────────────────────────────
        self.conf_thresh = conf_thresh
        self.panel_scale = panel_scale

        # ── FPS tracking ────────────────────────────────────────────
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0

        # Warmup
        self._warmup()

        print("[Pipeline] Ready!")
        print("=" * 60)

    def _warmup(self, n: int = 3):
        """Warmup GPU with dummy frames."""
        print("[Pipeline] Warming up GPU...")
        dummy = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
        for _ in range(n):
            self.yolo(dummy, device=0, verbose=False, imgsz=640)
            self.segformer(dummy)
        torch.cuda.synchronize()
        print("[Pipeline] Warmup complete")

    # ── Core Processing ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            dict with keys:
                'grid':         4-panel grid visualisation (BGR)
                'seg_map':      (H, W) uint8 segmentation class IDs
                'trav_map':     (H, W) float32 traversability map
                'detections':   list of YOLO detection dicts
                'path':         list of (x, y) pixel coords
                'fps':          current FPS estimate
                'timings':      dict of per-component times (ms)
        """
        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        # ── 1. & 2. Concurrent YOLO and SegFormer ──────────────────
        def run_yolo():
            t0 = time.perf_counter()
            yolo_results = self.yolo(frame, device=0, verbose=False, imgsz=640)
            torch.cuda.synchronize()
            t_yolo = (time.perf_counter() - t0) * 1000
            detections_res = parse_yolo_detections(yolo_results, self.conf_thresh)
            return t_yolo, detections_res

        def run_seg():
            t0 = time.perf_counter()
            seg_map_res = self.segformer(frame, original_size=(h, w))
            torch.cuda.synchronize()
            t_seg = (time.perf_counter() - t0) * 1000
            return t_seg, seg_map_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_yolo = executor.submit(run_yolo)
            future_seg = executor.submit(run_seg)
            t_yolo, detections = future_yolo.result()
            t_seg, seg_map = future_seg.result()

        # ── 3. Traversability ──────────────────────────────────────
        t0 = time.perf_counter()
        trav_map = seg_to_traversability(seg_map)
        trav_map = inject_yolo_obstacles(trav_map, detections, (h, w))
        trav_weighted = apply_distance_weighting(trav_map)
        t_trav = (time.perf_counter() - t0) * 1000

        # ── 4. Path Planning ──────────────────────────────────────
        t0 = time.perf_counter()
        path_grid = self.planner.plan(trav_weighted)
        path_pixels = self.planner.path_to_frame_coords(path_grid, (h, w))
        t_plan = (time.perf_counter() - t0) * 1000

        # ── 5. Visualisation ──────────────────────────────────────
        t0 = time.perf_counter()
        grid = self._build_grid(
            frame, seg_map, trav_weighted, detections, path_pixels
        )
        t_viz = (time.perf_counter() - t0) * 1000

        # ── FPS ───────────────────────────────────────────────────
        t_total = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / t_total if t_total > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        self.frame_count += 1

        timings = {
            "yolo_ms": t_yolo,
            "segformer_ms": t_seg,
            "traversability_ms": t_trav,
            "planning_ms": t_plan,
            "visualisation_ms": t_viz,
            "total_ms": t_total,
        }

        return {
            "grid": grid,
            "seg_map": seg_map,
            "trav_map": trav_weighted,
            "detections": detections,
            "path": path_pixels,
            "fps": avg_fps,
            "timings": timings,
        }

    # ── Grid Visualisation ───────────────────────────────────────────────

    def _build_grid(
        self,
        frame: np.ndarray,
        seg_map: np.ndarray,
        trav_map: np.ndarray,
        detections: list,
        path_pixels: list,
    ) -> np.ndarray:
        """
        Build a 2×2 grid visualisation:
            ┌──────────────────┬──────────────────┐
            │  Original + YOLO │  Segmentation    │
            ├──────────────────┼──────────────────┤
            │  Traversability  │  Navigation Path │
            └──────────────────┴──────────────────┘
        """
        h, w = frame.shape[:2]
        ph = int(h * self.panel_scale)
        pw = int(w * self.panel_scale)

        # Scale down base frame
        small_frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # Scale bounding boxes
        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            scaled_detections.append({
                "bbox": (int(x1 * self.panel_scale), int(y1 * self.panel_scale), 
                         int(x2 * self.panel_scale), int(y2 * self.panel_scale)),
                "name": det["name"],
                "conf": det["conf"]
            })

        # Scale path pixels
        scaled_path = [(int(x * self.panel_scale), int(y * self.panel_scale)) for x, y in path_pixels]

        # Panel 1: Original + YOLO detections
        p1 = small_frame.copy()
        draw_yolo_boxes(p1, scaled_detections)
        self._add_label(p1, "Original + Detections")

        # Panel 2: Segmentation overlay
        small_seg_map = cv2.resize(seg_map, (pw, ph), interpolation=cv2.INTER_NEAREST)
        seg_colour = seg_to_colourmap(small_seg_map)
        p2 = cv2.addWeighted(small_frame, 0.4, seg_colour, 0.6, 0)
        self._add_label(p2, "Segmentation")

        # Panel 3: Traversability heatmap
        small_trav_map = cv2.resize(trav_map, (pw, ph), interpolation=cv2.INTER_LINEAR)
        heatmap = traversability_to_heatmap(small_trav_map)
        p3 = cv2.addWeighted(small_frame, 0.3, heatmap, 0.7, 0)
        self._add_label(p3, "Traversability")

        # Panel 4: Navigation path
        p4 = small_frame.copy()
        draw_path_on_frame(p4, scaled_path, colour=(0, 255, 0), thickness=2, dot_radius=2)
        draw_yolo_boxes(p4, scaled_detections)
        p4 = cv2.addWeighted(p4, 0.8, heatmap, 0.2, 0)
        self._add_label(p4, "Navigation Path")

        # Assemble 2×2 grid
        top = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        grid = np.vstack([top, bottom])

        # FPS overlay
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_text = f"FPS: {avg_fps:.1f}"
            fps_colour = (0, 255, 0) if avg_fps >= 15 else (0, 0, 255)
            cv2.putText(grid, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(grid, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_colour, 2, cv2.LINE_AA)

        return grid

    @staticmethod
    def _add_label(frame: np.ndarray, text: str):
        """Add a label banner to the top of a frame."""
        h, w = frame.shape[:2]
        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
