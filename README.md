# Real-Time Semantic Segmentation for Autonomous Navigation on NVIDIA Jetson
## End-to-End Semantic Steering, Obstacle Avoidance, and Traffic Signal Detection

**IEEE Paper Title:** *Real-Time Semantic Segmentation for Autonomous Navigation on NVIDIA Jetson*

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Model-SegFormer--B0-yellow)](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024)
[![Ultralytics](https://img.shields.io/badge/Model-YOLOv8n-green)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## System Architecture

![Architecture Diagram](figures/architecture_diagram.png)

---

## Result Figures

# V5 Scene-Aware Navigation System (SegFormer + YOLOv8)

This project has been completely rebuilt to provide **full scene understanding** and **risk-aware autonomous navigation**.

## Architecture (V5)
The v5 pipeline fuses two state-of-the-art models:
1. **NVIDIA SegFormer (b0)**: Provides dense semantic segmentation for 19 Cityscapes classes (roads, buildings, vegetation, sidewalks, people, etc.).
2. **Ultralytics YOLOv8**: Detects and tracks dynamic obstacles (pedestrians, cars, trucks, bicycles) and traffic lights.

## Core Navigation Logic
Instead of complex A* paths or abstract risk maps, V5 calculates navigation decisions using a pure, highly optimized **pixel-level** approach:
- **Direct Pixel Counting**: The system evaluates a "lookahead band" by counting traversable pixels (road, sidewalk). Significant drops in traversable pixels automatically trigger `CAUTION` or `STOP` states.
- **Traffic Signal & Sign Detection**: YOLOv8 locates traffic lights (identifying their state as RED, GREEN, or YELLOW), while SegFormer isolates traffic sign regions.
- **Calibrated Obstacle Response**: Obstacles are inflated into a boolean mask. Only significant obstacles or proximate pedestrians (bbox area > 3% of frame) trigger a full stop, preventing false-positive lock-ups.
- **Smart Steering**: Finds the median X-coordinate (centroid) of the clear road pixels to determine a smooth steering angle, combined with a low-pass filter for stability.
- **Dynamic Perspective Grid**: An augmented-reality HUD overlays a grid, providing clear visual feedback of the robot's intended path.

## Running the Pipeline
You **must** use the environment that has both `transformers` and `ultralytics` installed:
```bash
python3 segformer_yolo_navigation_v5.py --source <input.mp4> --output <output.mp4>
```
|-------------------|-------------|
| Mean FPS          | **~6.7**    |
| Mean Latency      | **136.54 ms** |
| P95 Latency       | **156.16 ms** |
| RAM Usage (peak)  | 982 MB      |
| Safe Pixels       | 61.67%      |
| Obstacle Area     | 32.12%      |

> **Projected on Jetson Orin (JIT + FP16):** >15 FPS — exceeds the internship target.

---

## Project Structure

```text
Challa_project/
│
├── segformer_yolo_navigation_v5.py         ← ★ Final Navigation Pipeline (SegFormer+YOLO+Steering)
├── traversability_scorer.py     ← IEEE Novel Module
├── semantic_astar_navigation.py ← A* Path Planning
├── run_webcam.py                ← Live webcam (Week 1)
├── run_video.py                 ← Video + benchmark (Week 1, 4)
├── optimized_realsense_nav.py   ← RealSense live nav (Week 3)
│
├── figures/                     ← Visualization examples
├── INTERNSHIP_REPORT.md         ← 17-section technical report
└── README.md                    ← This file
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch transformers ultralytics opencv-python numpy scipy
```

### 2. Download YOLOv8 Weights
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. Run Navigation Pipeline
```bash
python3 segformer_yolo_navigation_v5.py --source input_video.mp4 --output output_video.mp4
```
*(If models are not provided, it gracefully falls back to a simulated OpenCV mask for fast testing).*

---

## Internship Deliverables Checklist

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Live Semantic Segmentation | `run_webcam.py` | ✅ |
| 2 | Traversable Area Detection | `segformer_yolo_navigation_v5.py` | ✅ |
| 3 | Obstacle Detection (YOLO)  | `segformer_yolo_navigation_v5.py` | ✅ |
| 4 | Centroid Steering Logic    | `segformer_yolo_navigation_v5.py` | ✅ |
| 5 | FPS Benchmark Report       | `video_benchmark.json` | ✅ |
| 6 | Indoor/Outdoor Testing     | `segformer_yolo_navigation_v5.py` | ✅ |
| 7 | Final Report (15–20 pages) | `INTERNSHIP_REPORT.md` | ✅ |
| 8 | GitHub Repository          | This repo | ✅ |

---

## IEEE Paper Contributions

1. Real-time **SegFormer-B0 + YOLOv8** fusion pipeline.
2. Centroid-based goal selection for jitter-free steering.
3. **Semantic Traversability Scoring Framework** ← novel contribution.
4. Pixel-level artifact/watermark erasure prior to inference.
5. Real-time benchmarking: FPS / latency / GPU / RAM evaluation.

---

## Citation

```bibtex
@inproceedings{challa2026jetsonnav,
  title     = {Real-Time Semantic Segmentation for Autonomous Navigation on NVIDIA Jetson},
  author    = {Challa, [Your Name]},
  booktitle = {IEEE Conference on Robotics and Automation},
  year      = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
