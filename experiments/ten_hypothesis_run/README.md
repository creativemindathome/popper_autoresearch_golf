# 10 Hypothesis Experiment Runner

Complete infrastructure for running 10 architecture hypotheses through the falsifier pipeline, with real-time graph evolution tracking and visualization for 2-hour time-lapse clips.

## Quick Start

```bash
# 1. Ensure API key is set
cd /Users/curiousmind/Desktop/null_fellow_hackathon
set -a && source .env && set +a

# 2. Run the full experiment
cd experiments/ten_hypothesis_run
bash run_full_experiment.sh

# 3. View results
open latest/viewer.html  # Interactive viewer
# or
open latest/evolution_timelapse.mp4  # Video (if generated)
```

## Architecture

```
experiments/ten_hypothesis_run/
├── run_10_hypotheses.py          # Main experiment runner
├── run_full_experiment.sh        # Master orchestration script
├── visualization/
│   ├── generate_frames.py        # Frame generation for time-lapse
│   └── assemble_clip.py          # Video assembly
└── README.md                     # This file

run_YYYYMMDD_HHMMSS/              # Generated per experiment
├── logs/
│   ├── run.log                   # Detailed execution log
│   └── console.log               # Console output
├── graph_snapshots/              # Knowledge graph at each step
│   ├── snapshot_0000.json
│   ├── snapshot_0001.json
│   └── ...
├── visualization/
│   ├── visualization_data.json   # Raw data for frames
│   └── frames/                   # Generated frames
│       ├── frame_00000.png
│       ├── frame_00001.png
│       └── ...
├── output/                       # Individual run results
│   ├── run_001.json
│   └── ...
├── summary.json                  # Final statistics
├── evolution_timelapse.mp4       # Video clip (if ffmpeg)
└── viewer.html                   # Interactive HTML viewer
```

## Pipeline Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   ANTHROPIC     │────▶│   FALSIFIER      │────▶│  KNOWLEDGE      │
│   (Generate 10  │     │   (Stage 1 + 2)  │     │  GRAPH          │
│    ideas)       │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │  Capture    │       │  Capture    │       │  Capture    │
  │  Snapshot   │       │  Snapshot   │       │  Snapshot   │
  └─────────────┘       └─────────────┘       └─────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                           │
                           ▼
              ┌─────────────────────┐
              │  Generate Frames    │
              │  (Interpolation)    │
              └─────────────────────┘
                           │
                           ▼
              ┌─────────────────────┐
              │  Assemble Video     │
              │  or HTML Viewer     │
              └─────────────────────┘
```

## Visualization Options

### 1. HTML Viewer (Always Available)

Interactive web-based viewer with:
- Frame-by-frame navigation
- Play/Pause controls
- Live statistics display
- Export instructions

```bash
open latest/viewer.html
```

### 2. Video Clip (Requires ffmpeg)

2-hour time-lapse compressed to video:

```bash
# Auto-assemble with best available method
python3 visualization/assemble_clip.py \
    --frames latest/visualization/frames \
    --output latest/evolution_timelapse.mp4

# Or use script
bash latest/assemble_video.sh
```

### 3. Frame Sequence

Access individual PNG frames for custom editing:

```bash
ls latest/visualization/frames/frame_*.png
```

## Configuration

### Environment Setup

```bash
# Create .env file in project root
cat > ../../.env << 'EOF'
ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
EOF
```

### Frame Generation Options

```bash
# Generate more frames (slower but smoother)
python3 visualization/generate_frames.py \
    --viz-data run_*/visualization/visualization_data.json \
    --output run_*/visualization/frames \
    --max-frames 300  # vs default 100

# Generate all frames for full 2-hour time-lapse
python3 visualization/generate_frames.py \
    --viz-data run_*/visualization/visualization_data.json \
    --output run_*/visualization/frames
    # (omit --max-frames for full generation)
```

### Video Assembly Options

```bash
# Force specific method
python3 visualization/assemble_clip.py \
    --frames run_*/visualization/frames \
    --output run_*/evolution_timelapse.mp4 \
    --method ffmpeg  # or moviepy

# Custom ffmpeg command
ffmpeg -framerate 30 -i run_*/visualization/frames/frame_%05d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    run_*/evolution_timelapse.mp4
```

## What Gets Measured

### Knowledge Graph Evolution
- **Total Ideas**: Cumulative hypotheses generated
- **Approved**: Ideas passing reviewer
- **Falsified**: Ideas killed by falsifier
- **Stage 1 Passed**: Survived T2-T7 gates
- **Stage 2 Passed**: Survived adversarial evaluation

### Per-Hypothesis Tracking
- Generation timestamp
- Falsifier verdict
- Which gate killed it (if applicable)
- Execution time
- Error messages (if failed)

### Time-Series Data
- Graph state captured at each pipeline stage
- Interpolated frames for smooth animation
- Timeline of hypothesis lifecycle

## Extending the Infrastructure

### Adding Custom Metrics

Edit `run_10_hypotheses.py`:

```python
def capture_graph_snapshot(self) -> GraphSnapshot:
    # Add your custom metrics
    custom_metric = self.calculate_custom_metric()

    snapshot = GraphSnapshot(
        # ... existing fields ...
        custom_metric=custom_metric,
    )
    return snapshot
```

### Custom Frame Rendering

Edit `visualization/generate_frames.py`:

```python
def generate_matplotlib_frame(self, frame_data: FrameData, frame_number: int):
    # Add your custom subplot
    ax = axes[2, 0]  # New subplot position
    ax.plot(your_data)
    ax.set_title('Your Metric')
```

### Integration with External Tools

The visualization data is JSON:

```bash
# Load into Python
cat run_*/visualization/visualization_data.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Timeline events: {len(data[\"timeline\"])}')
print(f'Evolution frames: {len(data[\"evolution\"])}')
"

# Load into JavaScript/D3
# Use viewer.html as template for custom visualizations
```

## Troubleshooting

### API Key Issues
```bash
# Test API key
python3 -c "
import os
key = os.environ.get('ANTHROPIC_API_KEY')
print(f'Key: {key[:20]}...' if key else 'Not set')
"

# Reload environment
set -a && source ../../.env && set +a
```

### Missing Dependencies
```bash
# Install matplotlib for frame generation
pip3 install matplotlib numpy

# Install moviepy for video assembly (alternative to ffmpeg)
pip3 install moviepy

# Install ffmpeg (macOS)
brew install ffmpeg
```

### Frame Generation Slow
```bash
# Reduce frame count for faster generation
python3 visualization/generate_frames.py --max-frames 50 ...
```

### Video Assembly Fails
```bash
# Check ffmpeg availability
which ffmpeg

# Use HTML viewer instead (always works)
open latest/viewer.html
```

## Performance Expectations

### Timing (on M-series Mac)
- **Idea Generation**: ~5-10s per hypothesis (API latency)
- **Stage 1 Gates**: ~30-60s per hypothesis (local computation)
- **Stage 2**: ~2-5 min per hypothesis (if enabled)
- **Frame Generation**: ~0.5s per frame
- **Video Assembly**: ~1-2 min for 100 frames

### Total Runtime
- **10 Hypotheses**: ~15-30 minutes
- **Frame Generation (100)**: ~1 minute
- **Video Assembly**: ~1-2 minutes
- **Total**: ~20-35 minutes

## Output Examples

### Summary JSON
```json
{
  "total_hypotheses": 10,
  "completed": 10,
  "failed": 0,
  "total_time_seconds": 1200.5,
  "verdicts": {
    "STAGE_1_KILLED": 7,
    "STAGE_2_KILLED": 2,
    "SURVIVED": 1
  },
  "snapshots_captured": 21
}
```

### Frame Data Structure
```json
{
  "frame": 42,
  "timestamp": 450.2,
  "total_ideas": 5,
  "approved": 5,
  "falsified": 3,
  "stage1_passed": 2,
  "stage2_passed": 0
}
```

## Credits

This infrastructure extends the Falsifier pipeline with:
- Anthropic Claude integration for idea generation
- Real-time knowledge graph snapshotting
- Matplotlib-based frame generation
- HTML5/JavaScript interactive viewer
- ffmpeg video assembly

Designed for 2-hour time-lapse visualization of research evolution.
