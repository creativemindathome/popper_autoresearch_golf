# Pipeline Integration Guide: Auto-Generating Evolution Movies

This guide shows how to automatically generate MP4 evolution videos every time your pipeline runs.

## Quick Start

### Option 1: Manual (One-time generation)
```bash
python3 knowledge_graph/visuals/auto_visualize.py \
    --experiment-dir experiments/ten_hypothesis_run/live_run_20260328_170317 \
    --fps 1
```

### Option 2: Automatic (Add to your pipeline)
Add this to the **end** of your pipeline script:

```python
# At the end of your pipeline run script
from knowledge_graph.visuals.auto_visualize import generate_all_visualizations

results = generate_all_visualizations(
    experiment_dir=experiment_output_dir,
    seed_graph="knowledge_graph/seed_parameter_golf_kg.json",
    output_dir="knowledge_graph/visuals",
    fps=1  # 1 frame per second
)

print(f"Evolution movie: {results['files'].get('evolution_movie')}")
```

## What Gets Generated

After each pipeline run, you'll automatically get:

| File | Description | Size |
|------|-------------|------|
| `evolution_movie.mp4` | **Animated timeline** showing hypotheses being added | ~500KB |
| `evolution_timeline.png` | Static timeline view | ~300KB |
| `kg_with_dead_ends.png` | Knowledge graph with refuted hypotheses marked | ~1MB |
| `kg_executive_with_research.png` | Executive presentation view | ~750KB |
| `registry_{run_id}.json` | Structured history of this run | ~5KB |
| `visualization_manifest.json` | Index of all generated files | ~1KB |

## The Evolution Movie

The MP4 shows:
1. **Frame 0**: Baseline knowledge graph (3 pillars)
2. **Frames 1..N**: Each hypothesis appears as it's added
3. **Color coding**:
   - Blue = Data Pipeline pillar
   - Red = Neural Network pillar  
   - Green = Training pillar
   - Red box with ✗ = Refuted hypothesis
   - Green box = Verified hypothesis
   - Amber = Unknown/Error
4. **Final frame**: Complete view with all 10 hypotheses and their fates

### Watching the Evolution

```bash
# macOS
open knowledge_graph/visuals/evolution_movie.mp4

# Linux
vlc knowledge_graph/visuals/evolution_movie.mp4

# Or copy to your phone/projector
```

## Integration Examples

### For your `run_pipeline.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# ... your pipeline code ...

def main():
    # Your existing pipeline
    experiment_dir = run_experiment()  # Your function

    # Auto-generate visualizations at the end
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    from knowledge_graph.visuals.auto_visualize import generate_all_visualizations

    viz_results = generate_all_visualizations(
        experiment_dir=str(experiment_dir),
        fps=1  # 1 frame per second
    )

    if viz_results["success"]:
        movie_path = viz_results["files"].get("evolution_movie")
        print(f"\n🎬 Evolution movie ready: {movie_path}")
        print(f"📊 View the animated history of your run!")
    else:
        print("\n⚠️  Some visualizations failed")

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### For Symphony orchestration:

Add to your `orchestrator.py` or pipeline runner:

```python
async def after_run_hook(experiment_dir: Path):
    """Called after each experiment run completes."""
    from knowledge_graph.visuals.auto_visualize import generate_all_visualizations

    results = generate_all_visualizations(
        experiment_dir=str(experiment_dir),
        fps=1
    )

    # Return paths for notification/logging
    return {
        "movie": results["files"].get("evolution_movie"),
        "timeline": results["files"].get("evolution_timeline"),
        "registry": results["files"].get("hypothesis_registry"),
    }
```

## Customization

### Change Frame Rate
```python
generate_all_visualizations(
    experiment_dir="...",
    fps=2  # 2 frames per second = faster video
)
```

### Higher Quality
Edit `generate_evolution_movie.py` and change:
```python
dpi = 300  # Instead of 150 for higher resolution
```

### Only Generate Movie
```python
from generate_evolution_movie import generate_movie

generate_movie(
    seed_path=Path("knowledge_graph/seed_parameter_golf_kg.json"),
    experiment_dir=Path("experiments/..."),
    output_path=Path("my_movie.mp4"),
    fps=1,
    dpi=150
)
```

## Requirements

```bash
# macOS
brew install graphviz ffmpeg

# Ubuntu/Debian  
sudo apt-get install graphviz ffmpeg

# Verify
which dot   # Should show path
which ffmpeg # Should show path
```

## File Structure After Run

```
knowledge_graph/
├── visuals/
│   ├── evolution_movie.mp4           # 🎬 The animated timeline
│   ├── evolution_timeline.png        # Static version
│   ├── evolution_summary_frame.png   # Final state
│   ├── kg_with_dead_ends.png        # Graph showing failures
│   ├── kg_executive_with_research.png
│   └── visualization_manifest.json    # Index of files
│
└── history/
    ├── hypothesis_registry.json       # Master registry
    └── registry_{run_id}.json       # Per-run history
```

## How It Works

1. **Load baseline** knowledge graph (3 pillars, ~90 nodes)
2. **Parse experiment** visualization_data.json for timeline
3. **Generate frames** showing progressive addition of hypotheses
4. **Render with Graphviz** each frame as PNG
5. **Stitch with ffmpeg** into MP4 video
6. **Save registry** of all tried hypotheses with verdicts

## Tips

### Viewing the Movie
- **VLC** (free, all platforms) - good for slow playback
- **QuickTime** (macOS) - smooth playback
- **Browser** - drag MP4 into any browser tab

### Sharing Results
The MP4 is small (~500KB) and can be:
- Emailed to collaborators
- Embedded in presentations
- Uploaded to Slack/Discord
- Played in team meetings

### Pipeline Integration Checklist

- [ ] Add `from knowledge_graph.visuals.auto_visualize import generate_all_visualizations`
- [ ] Call it after your experiment completes
- [ ] Pass the correct `experiment_dir` path
- [ ] (Optional) Print the movie path for easy access
- [ ] Install `ffmpeg` if not already installed

## Troubleshooting

### "ffmpeg not found"
```bash
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu
```

### "dot not found"
```bash
brew install graphviz  # macOS
sudo apt-get install graphviz  # Ubuntu
```

### Movie is too fast/slow
Change `--fps`:
- `--fps 0.5` = 2 seconds per frame (slower, longer video)
- `--fps 2` = 2 frames per second (faster, shorter video)

### Frames look wrong
Check that `visualization/visualization_data.json` exists in your experiment directory.

## Example Output

After running, you'll see:

```
============================================================
Generating Knowledge Graph Visualizations
============================================================

1. Creating evolution movie...
   ✓ knowledge_graph/visuals/evolution_movie.mp4

2. Creating evolution timeline...
   ✓ knowledge_graph/visuals/evolution_timeline.png

3. Updating hypothesis history...
   ✓ knowledge_graph/history/registry_live_run_20260328_170317.json

4. Creating knowledge graph with dead ends...
   ✓ knowledge_graph/visuals/kg_with_dead_ends.png

5. Creating executive visualizations...
   ✓ knowledge_graph/visuals/kg_executive_with_research.png

============================================================
Visualization Generation Complete
============================================================
Generated 5 files:
  - evolution_movie: knowledge_graph/visuals/evolution_movie.mp4
  - evolution_timeline: knowledge_graph/visuals/evolution_timeline.png
  - hypothesis_registry: knowledge_graph/history/registry_live_run_20260328_170317.json
  - kg_with_dead_ends: knowledge_graph/visuals/kg_with_dead_ends.png
  - executive: knowledge_graph/visuals/kg_executive_with_research.png

✓ All visualizations generated successfully

Manifest saved: knowledge_graph/visuals/visualization_manifest.json
```

Then just:
```bash
open knowledge_graph/visuals/evolution_movie.mp4
```

And watch your hypotheses evolve from baseline to final verdicts!
