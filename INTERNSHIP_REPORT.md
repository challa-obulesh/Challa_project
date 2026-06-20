# Internship Report: Autonomous AI Navigation on Jetson AGX Orin

## 1. Introduction and Objectives
The core objective of this internship project was to build and optimize a comprehensive autonomous navigation pipeline capable of understanding complex street scenes and making real-time driving decisions on embedded hardware (NVIDIA Jetson AGX Orin). The goals included achieving over 15 Frames Per Second (FPS) while successfully running heavy semantic segmentation alongside object detection.

## 2. Project Timeline & Accomplishments

### Phase 1: Foundation and Model Integration
- Integrated **SegFormer-B0** to provide pixel-perfect semantic understanding of 19 different environmental classes (Road, Sidewalk, Sky, Vegetation, etc.).
- Integrated **YOLOv8n** to handle dynamic object detection for cars, pedestrians, traffic lights, and stop signs.
- Devised a "fusion" algorithm: inflating YOLO bounding boxes to carve out "obstacle masks" from the SegFormer "drivable surface" mask.

### Phase 2: Navigation Logic & HUD
- Shifted from complex A* pathfinding to an optimized pixel-centroid tracking system within a dedicated "forward-lookahead" band.
- Built logic to translate road centroids into raw steering angles, which are stabilized using a low-pass filter.
- Developed dynamic HUD overlays including a perspective grid, steering angle, status readouts (SAFE, CAUTION, STOP), and traffic signal states (Red/Yellow/Green extraction using HSV masks).

### Phase 3: Hardware Optimization & TensorRT
- Converted standard PyTorch models to highly efficient **TensorRT engines**.
- Implemented concurrent model inference using Python's `ThreadPoolExecutor` so both AI networks evaluate the frame simultaneously.
- Rewrote the system color mapping to utilize pre-allocated Numpy Lookup Tables (LUTs), minimizing array conversion times.

### Phase 4: Refinement and Tuning (v6)
- Fine-tuned the obstacle detection threshold to focus only on the immediate bottom 40% of the camera view, eliminating false positives from distant objects.
- Drastically modified the navigation logic to output explicit Kinematic Actions (`ACTION: HARD LEFT`, `ACTION: SLIGHT RIGHT`, `ACTION: FORWARD`) instead of vague suggestions.
- Addressed I/O bottlenecks by isolating `cv2.VideoCapture` and `cv2.VideoWriter` into dedicated background threads, maximizing core processing speed.

## 3. Key Results & Final Metrics
The culmination of the project yielded a highly robust real-time pipeline that met and exceeded initial expectations:
- **Throughput (FPS):** Successfully breached the **15 FPS** goal during live operation. Core inference evaluates in parallel, completing cycles in roughly **~65ms**.
- **Latency:** Dropped end-to-end latency significantly through asynchronous threading and GPU warmups.
- **Accuracy:** Correctly navigates complex paths, reliably detects pedestrians overlapping the central forward zone (triggering hard stops if they occupy >8% of the frame), and smoothly dictates steering angles.

## 4. Conclusion
This project successfully demonstrated the feasibility of deploying heavy, dual-model architectures on the edge. By shifting from standard synchronous inference to a fully multithreaded, TensorRT-accelerated pipeline with direct pixel-centroid control, the system achieves highly stable and fast autonomous navigation.
