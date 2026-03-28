#!/bin/bash
# Run the complete 10-hypothesis experiment with visualization

set -e  # Exit on error

echo "=========================================="
echo "10 HYPOTHESIS EXPERIMENT"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f ../../.env ]; then
    echo "Loading environment from .env..."
    set -a
    source ../../.env
    set +a
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "✗ ANTHROPIC_API_KEY not set!"
    echo "Please set it with: export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

echo "✓ Environment loaded"
echo "✓ API key configured"
echo ""

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="run_${TIMESTAMP}"

echo "Creating run directory: $RUN_DIR"
mkdir -p "$RUN_DIR"/{logs,visualization,graph_snapshots,output}

# Step 1: Run 10 hypotheses
echo ""
echo "=========================================="
echo "STEP 1: Generating 10 Hypotheses"
echo "=========================================="
echo ""

python3 run_10_hypotheses.py 2>&1 | tee "$RUN_DIR/logs/console.log"

# Find the actual run directory created by the script
LATEST_RUN=$(ls -td run_*/ | head -1 | sed 's|/$||')

if [ ! -d "$LATEST_RUN" ]; then
    echo "✗ Run failed - no output directory found"
    exit 1
fi

echo ""
echo "✓ Hypothesis run complete: $LATEST_RUN"
echo ""

# Step 2: Generate visualization frames
echo "=========================================="
echo "STEP 2: Generating Visualization Frames"
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
    echo "⚠ No visualization data found, skipping frame generation"
fi

# Step 3: Assemble video clip
echo ""
echo "=========================================="
echo "STEP 3: Assembling Video Clip"
echo "=========================================="
echo ""

if [ -d "$LATEST_RUN/visualization/frames" ]; then
    python3 visualization/assemble_clip.py \
        --frames "$LATEST_RUN/visualization/frames" \
        --output "$LATEST_RUN/evolution_timelapse.mp4" \
        --method auto

    echo ""
    echo "✓ Video clip created"
else
    echo "⚠ No frames found, creating HTML viewer instead"
    python3 visualization/assemble_clip.py \
        --frames "$LATEST_RUN/visualization/frames" \
        --output "$LATEST_RUN/evolution_timelapse.mp4" \
        --method auto || true
fi

# Summary
echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="
echo ""
echo "Results in: $LATEST_RUN/"
echo ""
echo "Files generated:"
echo "  • logs/run.log - Detailed execution log"
echo "  • logs/console.log - Console output"
echo "  • summary.json - Final statistics"
echo "  • graph_snapshots/ - Graph state snapshots"
echo "  • visualization/visualization_data.json - Raw visualization data"
echo "  • visualization/frames/ - Frame images for time-lapse"
echo "  • evolution_timelapse.mp4 - Video clip (if ffmpeg available)"
echo "  • viewer.html - Interactive HTML viewer"
echo ""
echo "To view results:"
echo "  1. Open $LATEST_RUN/viewer.html in a browser"
echo "  2. Or run: open $LATEST_RUN/evolution_timelapse.mp4"
echo ""

# Create symlink to latest
rm -f latest
ln -s "$LATEST_RUN" latest
echo "Created symlink: latest -> $LATEST_RUN"
