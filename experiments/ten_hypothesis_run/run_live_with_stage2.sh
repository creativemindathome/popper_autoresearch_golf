#!/bin/bash
# FULL LIVE EXPERIMENT with Stage 2 Anthropic Sonnet
# Generates 10 hypotheses, runs Stage 1 gates, then adversarial Stage 2 falsifier

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     FULL LIVE EXPERIMENT - STAGE 2 ANTHROPIC (SONNET)            ║"
echo "║                                                                  ║"
echo "║     Pipeline: Ideator → Stage 1 → Stage 2 → Knowledge Graph     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
echo "Loading environment..."
if [ -f ../../.env ]; then
    set -a
    source ../../.env
    set +a
    echo "✓ Environment loaded from .env"
else
    echo "⚠ No .env file found, relying on existing environment"
fi

# Check API key
echo ""
echo "Checking API configuration..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "✗ ANTHROPIC_API_KEY not set!"
    echo ""
    echo "To set it:"
    echo "  export ANTHROPIC_API_KEY='<your-anthropic-api-key>'"
    echo ""
    echo "Or create .env from ../../.env.example and edit locally (never commit .env)"
    exit 1
fi

echo "✓ ANTHROPIC_API_KEY configured"
echo ""

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="live_run_${TIMESTAMP}"

echo "Creating run directory: $RUN_DIR"
mkdir -p "$RUN_DIR"

# Run the experiment
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  RUNNING FULL EXPERIMENT                                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Generate 10 novel architecture ideas (Anthropic Sonnet)"
echo "  2. Run Stage 1 gates (T2-T7) on each"
echo "  3. Run Stage 2 adversarial falsifier on survivors"
echo "  4. Track knowledge graph evolution"
echo ""
echo "Estimated time: 30-50 minutes"
echo ""

python3 run_full_live_experiment.py \
    --num-hypotheses 10 \
    --output-dir "$RUN_DIR" \
    --stage2-model "claude-sonnet-4-20250514" \
    2>&1 | tee "$RUN_DIR/console.log"

# Check if run succeeded
if [ ! -f "$RUN_DIR/summary.json" ]; then
    echo ""
    echo "✗ Experiment failed - no summary.json found"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GENERATING VISUALIZATION                                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Generate frames
if [ -f "$RUN_DIR/visualization/visualization_data.json" ]; then
    echo "Generating visualization frames..."
    python3 visualization/generate_frames.py \
        --viz-data "$RUN_DIR/visualization/visualization_data.json" \
        --output "$RUN_DIR/visualization/frames" \
        --max-frames 150

    echo ""
    echo "✓ Frames generated: $RUN_DIR/visualization/frames/"
else
    echo "⚠ No visualization data found"
fi

# Assemble video or create HTML viewer
echo ""
echo "Creating video/HTML viewer..."

if [ -d "$RUN_DIR/visualization/frames" ]; then
    python3 visualization/assemble_clip.py \
        --frames "$RUN_DIR/visualization/frames" \
        --output "$RUN_DIR/evolution_timelapse.mp4" \
        --method auto || true
fi

# Create symlink to latest
rm -f latest_live
ln -s "$RUN_DIR" latest_live

# Final summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT COMPLETE                                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory: $RUN_DIR/"
echo ""
echo "Key files:"
echo "  • $RUN_DIR/summary.json           - Final statistics"
echo "  • $RUN_DIR/logs/run.log            - Detailed execution log"
echo "  • $RUN_DIR/console.log             - Console output"
echo "  • $RUN_DIR/graph_snapshots/        - Graph evolution snapshots"
echo "  • $RUN_DIR/visualization/          - Visualization data"
echo "  • $RUN_DIR/evolution_timelapse.mp4 - Video (if ffmpeg available)"
echo "  • $RUN_DIR/viewer.html             - Interactive HTML viewer"
echo ""

# Show summary
echo "Quick stats:"
cat "$RUN_DIR/summary.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Total hypotheses: {data.get('total_hypotheses', 0)}\")
print(f\"  Completed: {data.get('completed', 0)}/{data.get('total_hypotheses', 0)}\")
print(f\"  Runtime: {data.get('total_time_minutes', 0):.1f} minutes\")
stats = data.get('statistics', {})
print(f\"  Stage 1 passed: {stats.get('stage1_passed', 0)}/{stats.get('total_generated', 0)}\")
if stats.get('stage2_triggered', 0) > 0:
    print(f\"  Stage 2 triggered: {stats.get('stage2_triggered', 0)}\")
    print(f\"  Stage 2 passed: {stats.get('stage2_passed', 0)}/{stats.get('stage2_triggered', 0)}\")
print(f\"  Snapshots: {data.get('snapshots_captured', 0)}\")
"

echo ""
echo "To view results:"
echo "  1. Video: open $RUN_DIR/evolution_timelapse.mp4"
echo "  2. HTML viewer: open $RUN_DIR/viewer.html"
echo "  3. Logs: less $RUN_DIR/logs/run.log"
echo ""
echo "Symlink created: latest_live -> $RUN_DIR"
echo ""
echo "✓ Full live experiment with Stage 2 Anthropic complete!"
