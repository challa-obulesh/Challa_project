import time
import torch
import psutil
import os
import numpy as np
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

device = torch.device('cpu')

# DeepLabV3 Benchmark
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights).to(device).eval()

dummy_input = torch.rand(1, 3, 480, 640).to(device)

print("Warming up DeepLabV3...")
for _ in range(5):
    with torch.no_grad():
        _ = model(dummy_input)

print("Benchmarking DeepLabV3...")
latencies = []
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1e6

for _ in range(20):
    start_t = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    latencies.append((time.perf_counter() - start_t) * 1000)

end_mem = psutil.Process(os.getpid()).memory_info().rss / 1e6

mean_lat = np.mean(latencies)
fps = 1000.0 / mean_lat
print(f"DeepLabV3 -> FPS: {fps:.2f}, Latency: {mean_lat:.2f}ms, RAM: {end_mem:.2f}MB")
