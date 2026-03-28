#!/usr/bin/env python3
"""
Generate Evolution Movie (MP4) of Knowledge Graph

Creates an animated visualization showing:
- Baseline knowledge graph
- Hypotheses being added over time
- Status changes (pending → verified/refuted)
- Final state with all learnings

Usage:
    python3 knowledge_graph/visuals/generate_evolution_movie.py
    python3 knowledge_graph/visuals/generate_evolution_movie.py --experiment-dir /path/to/run
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import os


def load_seed_graph(path: Path) -> tuple[list, list]:
    """Load the baseline knowledge graph."""
    with open(path) as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses_from_run(experiment_dir: Path) -> List[Dict]:
    """Load hypotheses from a specific experiment run."""
    viz_data_path = experiment_dir / "visualization" / "visualization_data.json"
    if not viz_data_path.exists():
        return []

    with open(viz_data_path) as f:
        data = json.load(f)

    # Build timeline of hypotheses
    hypotheses = []
    timeline = data.get("timeline", [])

    for event in timeline:
        if event.get("event") == "complete":
            hypotheses.append({
                "id": event.get("theory_id", f"hyp_{event.get('hypothesis', 0)}"),
                "hypothesis_num": event.get("hypothesis", 0),
                "verdict": event.get("verdict", "UNKNOWN"),
                "time": event.get("time", 0),
                "killed_by": event.get("stage1_verdict") if event.get("verdict") == "REFUTED" else None,
            })

    return sorted(hypotheses, key=lambda x: x["time"])


def load_all_historical_hypotheses(outbox_dir: Path) -> List[Dict]:
    """Load all historical hypotheses from outbox."""
    hypotheses = []

    # Try to load from falsifier results
    falsifier_dir = outbox_dir / "falsifier"
    if falsifier_dir.exists():
        for result_file in falsifier_dir.glob("*_result.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                theory_id = data.get("theory_id", "")
                verdict = data.get("verdict", "PENDING")
                killed_by = data.get("killed_by", "")

                if theory_id:
                    hypotheses.append({
                        "id": theory_id,
                        "title": theory_id.replace("-", " ").title(),
                        "verdict": verdict,
                        "killed_by": killed_by,
                        "time": 0,
                    })
            except Exception:
                continue

    return hypotheses


def build_frame_dot(
    nodes: list,
    edges: list,
    hypotheses: List[Dict],
    frame_number: int,
    total_frames: int,
    show_up_to_idx: int,
    title_suffix: str = "",
) -> str:
    """Build a single frame of the animation."""

    node_map = {n["id"]: n for n in nodes if n.get("id")}

    # Colors
    colors = {
        "root": {"data": "#2563eb", "nn": "#dc2626", "train": "#16a34a"},
        "branch": {"data": "#60a5fa", "nn": "#f87171", "train": "#4ade80"},
        "leaf": {"data": "#bfdbfe", "nn": "#fecaca", "train": "#bbf7d0"},
    }

    status_colors = {
        "pending": ("#fef3c7", "#f59e0b"),     # Amber
        "verified": ("#d1fae5", "#10b981"),    # Green
        "refuted": ("#fee2e2", "#ef4444"),     # Red
        "unknown": ("#f3f4f6", "#6b7280"),     # Gray
    }

    lines = []
    lines.append("digraph EvolutionFrame {")
    lines.append('  graph [bgcolor="#fafafa", fontname="Helvetica Neue",')
    lines.append('         fontsize=20, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.5, ranksep=1.2, pad=1.0,')
    lines.append('         margin=0, fontcolor="#1f2937"];')
    lines.append('  node [fontname="Helvetica Neue", shape=box,')
    lines.append('        style="rounded,filled", penwidth=2];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.6, penwidth=1.2];')
    lines.append("")

    progress = f"Frame {frame_number}/{total_frames}"
    if title_suffix:
        title = f"Knowledge Evolution - {title_suffix} ({progress})"
    else:
        title = f"Knowledge Evolution ({progress})"
    lines.append(f'  label="{title}";')
    lines.append("")

    # Build pillar structure (simplified)
    pillars = [
        ("node_root_data_pipeline", "Data Pipeline", "data"),
        ("node_root_neural_network", "Neural Network", "nn"),
        ("node_root_training_eval", "Training & Evaluation", "train"),
    ]

    children = {}
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            if src not in children:
                children[src] = []
            children[src].append(tgt)

    # Draw pillars
    for root_id, root_label, cat in pillars:
        if root_id not in node_map:
            continue

        c = colors["root"][cat]

        # Count relevant hypotheses for this pillar
        hyp_count = sum(1 for h in hypotheses[:show_up_to_idx]
                       if is_hypothesis_for_pillar(h, cat))

        if hyp_count > 0:
            label = f"{root_label}\\n({hyp_count} tested)"
            fillcolor = f"{c}80"  # 50% opacity
        else:
            label = root_label
            fillcolor = c

        lines.append(
            f'  "{root_id}" [label="{label}", fillcolor="{fillcolor}", '
            f'fontcolor="white", width=2.5, height=0.7, fontsize=12, penwidth=3];'
        )

        # Add some branches
        for branch_id in children.get(root_id, [])[:4]:
            if branch_id not in node_map:
                continue

            branch = node_map[branch_id]
            branch_label = branch.get("label", branch_id)

            lines.append(
                f'  "{branch_id}" [label="{branch_label}", fillcolor="{colors["branch"][cat]}", '
                f'fontcolor="white", width=2.0, height=0.5, fontsize=10];'
            )
            lines.append(f'  "{root_id}" -> "{branch_id}";')

    # Add hypotheses up to current frame
    if show_up_to_idx > 0 and hypotheses:
        lines.append("")
        lines.append("  // Hypotheses")

        for i, hyp in enumerate(hypotheses[:show_up_to_idx]):
            hyp_id = f"hyp_{i}"
            verdict = hyp.get("verdict") or "unknown"
            status = verdict.lower()
            fill, stroke = status_colors.get(status, status_colors["unknown"])

            short_id = (hyp.get("id") or f"H{i}")[:25]
            label = f"{short_id}\\n({status.upper()})"

            # Style based on status
            if status == "refuted":
                style = "filled,rounded,dashed"
                shape = "box"
            elif status == "verified":
                style = "filled,rounded"
                shape = "box"
            else:
                style = "filled,rounded,dashed"
                shape = "ellipse"

            lines.append(
                f'  "{hyp_id}" [label="{label}", fillcolor="{fill}", '
                f'color="{stroke}", style="{style}", fontcolor="{stroke}", '
                f'fontsize=9, width=2.0, height=0.5, penwidth=2];'
            )

            # Connect to pillar
            cat = categorize_hypothesis(hyp)
            if cat == "nn":
                lines.append(f'  "{hyp_id}" -> "node_root_neural_network" [color="{stroke}60", style="dashed", constraint=false];')
            elif cat == "data":
                lines.append(f'  "{hyp_id}" -> "node_root_data_pipeline" [color="{stroke}60", style="dashed", constraint=false];')
            elif cat == "train":
                lines.append(f'  "{hyp_id}" -> "node_root_training_eval" [color="{stroke}60", style="dashed", constraint=false];')

    lines.append("}")
    return "\n".join(lines)


def is_hypothesis_for_pillar(hyp: Dict, pillar: str) -> bool:
    """Check if a hypothesis belongs to a specific pillar."""
    cat = categorize_hypothesis(hyp)
    return cat == pillar


def categorize_hypothesis(hyp: Dict) -> str:
    """Categorize a hypothesis into a pillar."""
    hyp_id = (hyp.get("id") or "").lower()

    attention_keywords = ["attention", "mlp", "layer", "embed", "transformer", "depth", "sparse", "moe"]
    training_keywords = ["optim", "loss", "train", "precision", "grad", "checkpoint"]
    data_keywords = ["data", "token", "sequence", "curriculum", "augment"]

    if any(kw in hyp_id for kw in attention_keywords):
        return "nn"
    elif any(kw in hyp_id for kw in training_keywords):
        return "train"
    elif any(kw in hyp_id for kw in data_keywords):
        return "data"
    else:
        return "nn"  # Default


def render_frame(dot_text: str, output_path: Path, dpi: int = 150) -> bool:
    """Render a single frame to PNG."""
    try:
        # Write dot to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_text)
            dot_path = f.name

        # Render with dot
        result = subprocess.run(
            ["dot", f"-Gdpi={dpi}", "-Tpng", dot_path, "-o", str(output_path)],
            capture_output=True,
            text=True
        )

        os.unlink(dot_path)
        return result.returncode == 0

    except Exception as e:
        print(f"Error rendering frame: {e}")
        return False


def create_mp4_from_frames(frame_dir: Path, output_path: Path, fps: int = 2) -> bool:
    """Combine frames into MP4 using ffmpeg."""
    try:
        # Use ffmpeg to create MP4
        frame_pattern = str(frame_dir / "frame_%04d.png")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Quality (lower is better)
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False

        return True

    except FileNotFoundError:
        print("ffmpeg not found. Install with: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"Error creating MP4: {e}")
        return False


def generate_movie(
    seed_path: Path,
    experiment_dir: Path,
    output_path: Path,
    fps: int = 2,
    dpi: int = 150,
) -> bool:
    """Generate the full evolution movie."""

    print(f"Loading seed graph from {seed_path}...")
    nodes, edges = load_seed_graph(seed_path)

    print(f"Loading hypotheses from {experiment_dir}...")
    hypotheses = load_hypotheses_from_run(experiment_dir)

    if not hypotheses:
        print("No hypotheses found in experiment data")
        return False

    print(f"Found {len(hypotheses)} hypotheses")

    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)

        # Generate frames
        total_frames = len(hypotheses) + 2  # +2 for intro and outro

        # Frame 0: Baseline only
        print("Generating frame 0: Baseline...")
        dot = build_frame_dot(nodes, edges, hypotheses, 0, total_frames, 0, "Baseline")
        render_frame(dot, frame_dir / "frame_0000.png", dpi)

        # Frames 1..n: Add hypotheses one by one
        for i in range(len(hypotheses)):
            frame_num = i + 1
            hyp = hypotheses[i]
            print(f"Generating frame {frame_num}: Adding {hyp['id'][:30]}...")

            dot = build_frame_dot(nodes, edges, hypotheses, frame_num, total_frames, i + 1,
                                  f"Hypothesis {i+1}")
            render_frame(dot, frame_dir / f"frame_{frame_num:04d}.png", dpi)

        # Final frame: All hypotheses
        print(f"Generating frame {total_frames-1}: Final state...")
        dot = build_frame_dot(nodes, edges, hypotheses, total_frames - 1, total_frames,
                              len(hypotheses), "Complete")
        render_frame(dot, frame_dir / f"frame_{total_frames-1:04d}.png", dpi)

        # Create MP4
        print(f"Creating MP4 at {fps} fps...")
        if create_mp4_from_frames(frame_dir, output_path, fps):
            print(f"✓ Generated evolution movie: {output_path}")
            return True
        else:
            print("✗ Failed to create MP4")
            return False


def main():
    parser = argparse.ArgumentParser(description="Generate knowledge graph evolution movie")
    parser.add_argument("--seed", default="knowledge_graph/seed_parameter_golf_kg.json",
                        help="Path to seed knowledge graph")
    parser.add_argument("--experiment-dir",
                        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
                        help="Path to experiment run directory")
    parser.add_argument("--output", default="knowledge_graph/visuals/evolution_movie.mp4",
                        help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=2,
                        help="Frames per second in output video")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution of frames")
    args = parser.parse_args()

    seed_path = Path(args.seed)
    experiment_dir = Path(args.experiment_dir)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = generate_movie(seed_path, experiment_dir, output_path, args.fps, args.dpi)

    if success:
        # Also create a summary image
        print("\nGenerating summary frame...")
        nodes, edges = load_seed_graph(seed_path)
        hypotheses = load_hypotheses_from_run(experiment_dir)

        dot = build_frame_dot(nodes, edges, hypotheses, 0, 1, len(hypotheses), "Final Summary")
        summary_path = output_path.parent / "evolution_summary_frame.png"
        render_frame(dot, summary_path, dpi=300)
        print(f"✓ Summary frame: {summary_path}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
