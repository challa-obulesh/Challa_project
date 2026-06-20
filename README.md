# Autonomous Navigation on Jetson AGX Orin
**Models:** SegFormer-B0 & YOLOv8n
**Hardware:** NVIDIA Jetson AGX Orin

## Project Overview
This project implements a real-time autonomous navigation system leveraging deep learning for semantic segmentation and object detection. It processes video feeds to make immediate driving decisions (steering, throttle) while identifying and avoiding obstacles, parsing traffic signals, and reading sign boards.

## Features & Implementation
- **Sensor Fusion:** Fuses SegFormer's drivable surface mask with YOLOv8's inflated bounding boxes to isolate a safe path.
- **Pixel-Level Kinematic Navigation:** Computes the centroid of the clear road area to dictate realistic driving actions (`ACTION: HARD LEFT`, `ACTION: SLIGHT RIGHT`, `ACTION: FORWARD`, etc.) without resorting to heavy pathfinding algorithms.
- **Traffic Signal Detection:** Extracts colors (RED, GREEN, YELLOW) from traffic lights utilizing HSV thresholding to make regulatory stops.
- **Sign Board Parsing:** Identifies 'traffic sign' pixel blobs to draw accurate bounding boxes.
- **Smoothed Steering:** Employs a low-pass filter (alpha = 0.10) for robust frame-to-frame steering stability.

## Performance Metrics & Optimizations
We achieved significant frame-rate improvements by targeting hardware-level bottlenecks:
- **FPS Achieved:** Live inference reliably achieves **>15 FPS** (averaging ~13-16 FPS during heavily I/O-bound local MP4 disk writes, but scaling up smoothly in memory).
- **Latency:** Core inference takes roughly **~60-80ms** per frame.
- **TensorRT (TRT):** Accelerated both SegFormer and YOLO using natively exported `.engine` models for maximum throughput on the Jetson's Ampere GPU.
- **Threaded Inference:** Uses `ThreadPoolExecutor` to process YOLO and SegFormer completely in parallel.
- **Threaded Video I/O:** Both camera capture and video writer (`cv2.VideoWriter`) operate on isolated background threads with queues, preventing disk and camera bottlenecking from stalling the neural networks.
- **Lookup Tables (LUTs):** Color mapping is optimized using pre-calculated numpy index matrices.

## Quick Start
Run the optimized pipeline (v6) on local video files:
```bash
./run_v6.sh
```
Or execute directly on a webcam feed / video file:
```bash
python3 src/segformer_yolo_navigation_v6.py --source <video/0> --trt
```
