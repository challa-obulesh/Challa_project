#!/bin/bash
videos=("00a0f008-a315437f.mov" "00a1176f-5121b501.mov" "01c4035b-bcaeb067.mov")
for vid in "${videos[@]}"; do
    base="${vid%%.*}"
    echo "Processing $vid with v6..."
    python3 src/segformer_yolo_navigation_v6.py --source "$vid" --output "raw_v6_${base}.mp4" --rotate ccw --no-show
    echo "Converting to H264 playable format..."
    ffmpeg -y -i "raw_v6_${base}.mp4" -c:v libx264 -pix_fmt yuv420p -preset fast "final_v6_${base}.mp4"
done
echo "All done!"
