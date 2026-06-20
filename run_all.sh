#!/bin/bash
videos=("00a0f008-a315437f.mov" "00a1176f-5121b501.mov" "01c4035b-bcaeb067.mov")
for vid in "${videos[@]}"; do
    base="${vid%%.*}"
    echo "Processing $vid with CCW rotation..."
    python3 src/segformer_yolo_navigation_v5.py --source "$vid" --output "raw_${base}.mp4" --rotate ccw --no-show
    echo "Converting to H264 playable format..."
    ffmpeg -y -i "raw_${base}.mp4" -c:v libx264 -pix_fmt yuv420p -preset fast "final_ccw_${base}.mp4"
done
echo "All done!"
