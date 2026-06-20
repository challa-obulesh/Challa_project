import cv2
import time
import numpy as np
import concurrent.futures
import torch
import sys

sys.path.insert(0, "./src")
from trt_segformer import SegFormerTRT
from ultralytics import YOLO

yolo = YOLO("models/yolov8n.engine", task='detect')
seg = SegFormerTRT("models/segformer_b0.engine")
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup
for _ in range(3):
    yolo(frame, imgsz=640, verbose=False)
    seg(frame)

def run_yolo():
    t0 = time.time()
    res = yolo(frame, imgsz=640, verbose=False)[0]
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

def run_seg():
    t0 = time.time()
    res = seg(frame)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    t0 = time.time()
    fy = ex.submit(run_yolo)
    fs = ex.submit(run_seg)
    ry = fy.result()
    rs = fs.result()
    t1 = time.time()
    print(f"Concurrent: Total {(t1-t0)*1000:.1f}ms (YOLO {ry:.1f}ms, Seg {rs:.1f}ms)")

t0 = time.time()
ry = run_yolo()
rs = run_seg()
t1 = time.time()
print(f"Sequential: Total {(t1-t0)*1000:.1f}ms (YOLO {ry:.1f}ms, Seg {rs:.1f}ms)")
