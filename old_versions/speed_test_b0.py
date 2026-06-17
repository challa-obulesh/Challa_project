import cv2
import time
import torch

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

cap = cv2.VideoCapture("/home/sdv/zed_navigation_3min.mp4")

ret, frame = cap.read()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

start = time.time()

for _ in range(20):

    inputs = processor(
        images=rgb,
        return_tensors="pt"
    )

    with torch.no_grad():
        _ = model(**inputs)

elapsed = time.time() - start

print("FPS =", 20 / elapsed)

cap.release()
