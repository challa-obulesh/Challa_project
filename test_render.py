import time
import cv2
import numpy as np

frame_bgr = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
label_map = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)
LUT = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

obstacle_mask = np.random.randint(0, 2, (720, 1280), dtype=np.uint8)
OBS_COLOR = (0, 0, 255)
_OBS_COLOR_IMG = np.full(frame_bgr.shape, OBS_COLOR, dtype=np.uint8)

t0 = time.time()
for _ in range(100):
    seg_colour = LUT[label_map]
    out = cv2.addWeighted(frame_bgr, 0.5, seg_colour, 0.5, 0)
    
    if obstacle_mask.any():
        blended = cv2.addWeighted(out, 0.45, _OBS_COLOR_IMG, 0.55, 0)
        cv2.copyTo(blended, obstacle_mask, out)

print(f"Optimised full render: {(time.time() - t0)/100 * 1000:.2f} ms")
