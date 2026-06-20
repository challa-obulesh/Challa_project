import time
import cv2
import numpy as np

# Simulate full resolution operations
frame_bgr = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
label_map = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)
LUT = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

t0 = time.time()
for _ in range(100):
    seg_colour = LUT[label_map]
    out = cv2.addWeighted(frame_bgr, 0.45, seg_colour, 0.55, 0)
print(f"Full res blend: {(time.time() - t0)/100 * 1000:.2f} ms")

t0 = time.time()
for _ in range(100):
    small_frame = cv2.resize(frame_bgr, (640, 360))
    small_label = cv2.resize(label_map, (640, 360), interpolation=cv2.INTER_NEAREST)
    seg_colour = LUT[small_label]
    out = cv2.addWeighted(small_frame, 0.45, seg_colour, 0.55, 0)
print(f"Half res blend: {(time.time() - t0)/100 * 1000:.2f} ms")
