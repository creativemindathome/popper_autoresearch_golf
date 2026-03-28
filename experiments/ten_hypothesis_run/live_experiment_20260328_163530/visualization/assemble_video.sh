#!/bin/bash
# Assemble frames into video using ffmpeg
# Run this script from the experiment directory

FPS=30
INPUT_PATTERN="visualization/frames/frame_%05d.png"
OUTPUT="evolution_timelapse.mp4"

echo "Assembling time-lapse video..."
echo "  FPS: $FPS"
echo "  Input: $INPUT_PATTERN"
echo "  Output: $OUTPUT"

ffmpeg -y \
    -framerate $FPS \
    -i "$INPUT_PATTERN" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -crf 23 \
    -preset medium \
    "$OUTPUT"

echo "✓ Video created: $OUTPUT"
