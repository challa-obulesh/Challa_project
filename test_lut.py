import numpy as np
import cv2
import time

label_map = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)
palette = np.random.randint(0, 255, (19, 3), dtype=np.uint8)

cv_lut = np.zeros((256, 1, 3), dtype=np.uint8)
cv_lut[:19, 0, :] = palette

t0 = time.time()
for _ in range(100):
    res2 = cv2.applyColorMap(label_map, cv_lut)
print(f"cv2.applyColorMap: {(time.time()-t0)*10:.2f} ms")
