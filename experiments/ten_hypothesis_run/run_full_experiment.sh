#!/bin/bash
# Run the multi-hypothesis experiment (Stage 1 + optional Stage 2), then optional viz.
#
# Primary runner: run_full_live_experiment.py (creates live_run_YYYYMMDD_HHMMSS/).
# Legacy Stage-1-only path: run_10_hypotheses.py (creates run_YYYYMMDD_HHMMSS/).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MULTI-HYPOTHESIS EXPERIMENT (full live pipeline)"
echo "=========================================="
echo ""

if [ -f ../../.env ]; then
    echo "Loading environment from .env..."
    set -a
    source ../../.env
    set +a
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "✗ ANTHROPIC_API_KEY not set."
    echo "  export ANTHROPIC_API_KEY='...'  or add it to ../../.env"
    exit 1
fi

echo "✓ Environment loaded"
echo ""

# Default: 10 hypotheses; pass extra args to the Python runner, e.g. --disable-stage2
python3 run_full_live_experiment.py --num-hypotheses 10 "$@"

LATEST_RUN="$(ls -td live_run_*/ 2>/dev/null | head -1 | sed 's|/$||')"
if [ -z "$LATEST_RUN" ] || [ ! -d "$LATEST_RUN" ]; then
    echo "✗ No live_run_* directory found after the experiment."
    exit 1
fi

echo ""
echo "✓ Hypothesis run complete: $LATEST_RUN"
echo ""

# Step 2: Generate visualization frames
echo "=========================================="
echo "STEP 2: Generating visualization frames (optional)"
echo "=========================================="
echo ""

if [ -f "$LATEST_RUN/visualization/visualization_data.json" ]; then
    python3 visualization/generate_frames.py \
        --viz-data "$LATEST_RUN/visualization/visualization_data.json" \
        --output "$LATEST_RUN/visualization/frames" \
        --max-frames 100
    echo ""
    echo "✓ Frames generated"
else
    echo "⚠ No visualization data found; skipping frame generation"
fi

# Step 3: Assemble video clip
echo ""
echo "=========================================="
echo "STEP 3: Assembling video clip (optional)"
echo "=========================================="
echo ""

if [ -d "$LATEST_RUN/visualization/frames" ] && [ -n "$(ls -A "$LATEST_RUN/visualization/frames" 2>/dev/null)" ]; then
    python3 visualization/assemble_clip.py \
        --frames "$LATEST_RUN/visualization/frames" \
        --output "$LATEST_RUN/evolution_timelapse.mp4" \
        --method auto
    echo ""
    echo "✓ Video clip created (if ffmpeg/moviepy available)"
else
    echo "⚠ No frames directory; skip video assembly. Use HTML viewer if present."
fi

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="
echo ""
echo "Results: $LATEST_RUN/"
echo "  • logs/run.log — execution log"
echo "  • summary.json — aggregate stats"
echo "  • visualization/visualization_data.json — data for frames / viewer"
echo ""

rm -f latest
ln -s "$LATEST_RUN" latest
echo "Symlink: latest -> $LATEST_RUN"
