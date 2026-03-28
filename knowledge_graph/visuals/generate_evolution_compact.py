#!/usr/bin/env python3
"""
Generate COMPACT but READABLE Evolution Visualization

Optimized for 100+ hypotheses with:
- Horizontal layout (saves vertical space)
- Grid arrangement within categories
- Large readable fonts
- Color coding by status
"""

import argparse
import json
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Dict
import os


def load_seed_graph(path: Path) -> tuple[list, list]:
    with open(path) as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses_from_run(experiment_dir: Path) -> List[Dict]:
    viz_data_path = experiment_dir / "visualization" / "visualization_data.json"
    if not viz_data_path.exists():
        return []

    with open(viz_data_path) as f:
        data = json.load(f)

    hyp_events = {}
    for event in data.get("timeline", []):
        hyp_num = event.get("hypothesis", 0)
        if hyp_num not in hyp_events:
            hyp_events[hyp_num] = {"events": []}
        hyp_events[hyp_num]["events"].append(event)

    hypotheses = []
    for hyp_num, hdata in hyp_events.items():
        start_event = None
        complete_event = None
        for e in hdata["events"]:
            if e.get("event") == "start":
                start_event = e
            elif e.get("event") == "complete":
                complete_event = e

        if complete_event:
            theory_id = start_event.get("theory_id") if start_event else None
            if not theory_id:
                theory_id = f"hyp_{hyp_num}"

            hypotheses.append({
                "id": theory_id,
                "hypothesis_num": hyp_num,
                "verdict": complete_event.get("verdict", "UNKNOWN"),
                "time": complete_event.get("time", 0),
                "killed_by": complete_event.get("stage1_verdict") if complete_event.get("verdict") == "REFUTED" else None,
            })

    return sorted(hypotheses, key=lambda x: x["time"])


def find_parent_category(hyp: Dict) -> str:
    hyp_id = (hyp.get("id") or "").lower()

    if any(kw in hyp_id for kw in ["attention", "mlp", "layer", "embed", "transformer", "depth", "sparse", "moe", "ffn"]):
        return "neural"
    elif any(kw in hyp_id for kw in ["optim", "loss", "train", "precision", "grad", "checkpoint", "lr ", "scheduler", "batch"]):
        return "training"
    elif any(kw in hyp_id for kw in ["data", "token", "sequence", "curriculum", "augment", "sample", "clean", "dedup"]):
        return "data"
    else:
        return "neural"


def build_compact_dot(
    hypotheses: List[Dict],
    title: str = "Hypothesis Evolution",
) -> str:
    """Build a compact, grid-based DOT graph for many hypotheses."""

    # Group by category and status
    categories = {
        "data": {"label": "📊 Data Pipeline", "color": "#2563eb", "fill": "#dbeafe", "hyps": []},
        "neural": {"label": "🧠 Neural Network", "color": "#dc2626", "fill": "#fee2e2", "hyps": []},
        "training": {"label": "⚙️ Training", "color": "#16a34a", "fill": "#dcfce7", "hyps": []},
    }

    for hyp in hypotheses:
        cat = find_parent_category(hyp)
        if cat in categories:
            categories[cat]["hyps"].append(hyp)

    lines = []
    lines.append("digraph Evolution {")
    # Horizontal layout, very readable
    lines.append('  graph [bgcolor="#ffffff", fontname="Helvetica Neue",')
    lines.append('         fontsize=28, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.4, ranksep=1.5, pad=1.0, margin=2.0,')
    lines.append('         fontcolor="#1f2937", rankdir=TB];')
    lines.append('  node [fontname="Helvetica Neue", shape=box, style="rounded,filled",')
    lines.append('        penwidth=2, fontsize=11, margin="0.15,0.1", width=3.5, height=0.6];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.7, penwidth=1.5];')
    lines.append("")

    # Main title
    refuted = sum(1 for h in hypotheses if h.get("verdict") == "REFUTED")
    verified = sum(1 for h in hypotheses if h.get("verdict") == "VERIFIED")
    unknown = len(hypotheses) - refuted - verified

    title_str = f"{title}\\n{len(hypotheses)} Hypotheses | {refuted} ❌ | {verified} ✓ | {unknown} ?"
    lines.append(f'  label="{title_str}";')
    lines.append("")

    # Status colors
    status_colors = {
        "refuted": ("#fee2e2", "#ef4444", "❌"),
        "verified": ("#d1fae5", "#10b981", "✓"),
        "unknown": ("#f3f4f6", "#6b7280", "?"),
    }

    # Create each category section
    for cat_key, cat_data in categories.items():
        if not cat_data["hyps"]:
            continue

        # Section header node
        header_id = f"header_{cat_key}"
        count = len(cat_data["hyps"])
        lines.append(
            f'  "{header_id}" [label="{cat_data["label"]}\\n({count} hypotheses)", '
            f'fillcolor="{cat_data["fill"]}", color="{cat_data["color"]}", '
            f'fontcolor="{cat_data["color"]}", fontsize=20, penwidth=3, '
            f'width=6.0, height=0.8, style="rounded,filled,bold"];'
        )
        lines.append("")

        # Arrange hypotheses in a grid (3 per row)
        hyps = cat_data["hyps"]
        cols = 3  # Number of columns

        for i, hyp in enumerate(hyps):
            row = i // cols
            col = i % cols

            hyp_node_id = f"hyp_{cat_key}_{i}"
            verdict = hyp.get("verdict") or "unknown"
            status = verdict.lower()
            fill, stroke, icon = status_colors.get(status, status_colors["unknown"])

            # Truncate ID for display
            short_id = (hyp.get("id") or f"H{i}")[:32]

            if status == "refuted" and hyp.get("killed_by"):
                label = f"{icon} {short_id}\\nKilled: {hyp['killed_by']}"
            else:
                label = f"{icon} {short_id}"

            # Create node
            lines.append(
                f'  "{hyp_node_id}" [label="{label}", fillcolor="{fill}", '
                f'color="{stroke}", fontcolor="{stroke}", '
                f'fontsize=11, penwidth=2, width=3.5, height=0.6];'
            )

            # Connect to header or previous in row
            if row == 0:
                # First row connects to header
                lines.append(
                    f'  "{header_id}" -> "{hyp_node_id}" [color="{stroke}", '
                    f'style=dashed, penwidth=1.5, weight=1];'
                )
            else:
                # Connect to node above (previous row, same column)
                above_id = f"hyp_{cat_key}_{i - cols}"
                lines.append(
                    f'  "{above_id}" -> "{hyp_node_id}" [style=invis, weight=10];'
                )

            # Connect horizontally within row for ordering
            if col > 0:
                prev_id = f"hyp_{cat_key}_{i-1}"
                lines.append(
                    f'  "{prev_id}" -> "{hyp_node_id}" [style=invis, weight=5];'
                )

        lines.append("")

        # Keep all nodes in this category at same rank per row
        for row in range((len(hyps) + cols - 1) // cols):
            row_nodes = [f"hyp_{cat_key}_{row * cols + c}" for c in range(cols)
                        if row * cols + c < len(hyps)]
            if row_nodes:
                lines.append("  { rank=same; " + "; ".join(f'"{n}"' for n in row_nodes) + "; }")

        lines.append("")

    lines.append("}")
    return "\n".join(lines)


def render_dot(dot_text: str, output_path: Path, dpi: int = 300) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_text)
            dot_path = f.name

        result = subprocess.run(
            ["dot", f"-Gdpi={dpi}", "-Tpng", dot_path, "-o", str(output_path)],
            capture_output=True,
            text=True
        )

        os.unlink(dot_path)

        if result.returncode != 0:
            print(f"  Render error: {result.stderr[:500]}")
            return False

        return True

    except Exception as e:
        print(f"Error rendering: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate compact evolution visualization")
    parser.add_argument("--experiment-dir",
                        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
                        help="Path to experiment run directory")
    parser.add_argument("--output", default="knowledge_graph/visuals/evolution_compact.png",
                        help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution (300 for screen, 600 for print)")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading hypotheses from {experiment_dir}...")
    hypotheses = load_hypotheses_from_run(experiment_dir)

    if not hypotheses:
        print("No hypotheses found!")
        return 1

    print(f"Found {len(hypotheses)} hypotheses")

    categories = {"data": 0, "neural": 0, "training": 0}
    for hyp in hypotheses:
        cat = find_parent_category(hyp)
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  Data Pipeline: {categories['data']}")
    print(f"  Neural Network: {categories['neural']}")
    print(f"  Training: {categories['training']}")

    print(f"\nGenerating compact visualization at {args.dpi} DPI...")
    dot = build_compact_dot(hypotheses, "Knowledge Evolution")

    if render_dot(dot, output_path, args.dpi):
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Generated: {output_path} ({size_mb:.1f} MB)")
        print(f"  Layout: Grid (3 columns per category)")
        print(f"  Scales to: 100+ hypotheses")
        return 0
    else:
        print("✗ Failed to generate")
        return 1


if __name__ == "__main__":
    exit(main())
