#!/usr/bin/env python3
"""
Generate READABLE Evolution Visualization

Designed to scale to 100+ hypotheses with:
- Large, readable fonts
- Proper node sizing
- Clustered layout by parent node
- High resolution output
"""

import argparse
import json
import subprocess
import tempfile
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
    """Categorize hypothesis into a high-level pillar."""
    hyp_id = (hyp.get("id") or "").lower()

    if any(kw in hyp_id for kw in ["attention", "mlp", "layer", "embed", "transformer", "depth", "sparse", "moe", "ffn"]):
        return "neural"
    elif any(kw in hyp_id for kw in ["optim", "loss", "train", "precision", "grad", "checkpoint", "lr ", "scheduler", "batch"]):
        return "training"
    elif any(kw in hyp_id for kw in ["data", "token", "sequence", "curriculum", "augment", "sample", "clean", "dedup"]):
        return "data"
    else:
        return "neural"  # Default


def build_readable_dot(
    hypotheses: List[Dict],
    title: str = "Hypothesis Evolution",
) -> str:
    """Build a highly readable DOT graph optimized for many hypotheses."""

    # Group hypotheses by category
    categories = {
        "data": {"label": "Data Pipeline", "color": "#2563eb", "fill": "#dbeafe", "hyps": []},
        "neural": {"label": "Neural Network", "color": "#dc2626", "fill": "#fee2e2", "hyps": []},
        "training": {"label": "Training & Eval", "color": "#16a34a", "fill": "#dcfce7", "hyps": []},
    }

    for hyp in hypotheses:
        cat = find_parent_category(hyp)
        if cat in categories:
            categories[cat]["hyps"].append(hyp)

    lines = []
    lines.append("digraph Evolution {")
    # Much larger, readable graph
    lines.append('  graph [bgcolor="#ffffff", fontname="Helvetica Neue",')
    lines.append('         fontsize=24, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.8, ranksep=2.0, pad=1.0, margin=2.0,');
    lines.append('         fontcolor="#1f2937", rankdir=TB];')
    lines.append('  node [fontname="Helvetica Neue", shape=box, style="rounded,filled",')
    lines.append('        penwidth=2, fontsize=14, margin="0.2,0.15", width=0, height=0];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.8, penwidth=2];')
    lines.append("")
    lines.append(f'  label="{title} - {len(hypotheses)} Hypotheses";')
    lines.append("")

    # Create pillar nodes at top
    pillar_nodes = []
    for cat_key, cat_data in categories.items():
        if cat_data["hyps"]:
            node_id = f"pillar_{cat_key}"
            count = len(cat_data["hyps"])
            label = f"{cat_data['label']}\\n({count} tested)"
            lines.append(
                f'  "{node_id}" [label="{label}", fillcolor="{cat_data["fill"]}", '
                f'color="{cat_data["color"]}", fontcolor="{cat_data["color"]}", '
                f'fontsize=18, penwidth=3, width=3.0, height=1.0];'
            )
            pillar_nodes.append(node_id)

    # Keep pillars at same rank
    if pillar_nodes:
        lines.append("  { rank=same; " + "; ".join(f'"{p}"' for p in pillar_nodes) + "; }")
        for left, right in zip(pillar_nodes, pillar_nodes[1:]):
            lines.append(f'  "{left}" -> "{right}" [style="invis"];')
    lines.append("")

    # Status colors
    status_colors = {
        "refuted": ("#fee2e2", "#ef4444"),     # Red
        "verified": ("#d1fae5", "#10b981"),    # Green
        "unknown": ("#f3f4f6", "#6b7280"),     # Gray
    }

    # Add hypotheses grouped by category in clusters
    for cat_key, cat_data in categories.items():
        if not cat_data["hyps"]:
            continue

        pillar_id = f"pillar_{cat_key}"

        # Start subgraph cluster
        lines.append(f'  subgraph cluster_{cat_key} {{')
        lines.append(f'    label="{cat_data["label"]} Hypotheses";')
        lines.append(f'    style=filled;')
        lines.append(f'    fillcolor="{cat_data["fill"]}30";')  # 30% opacity
        lines.append(f'    color="{cat_data["color"]}80";')
        lines.append(f'    fontsize=16;')
        lines.append(f'    fontcolor="{cat_data["color"]}";')
        lines.append(f'    penwidth=2;')
        lines.append(f'    margin=20;')
        lines.append("")

        # Add hypotheses in this category
        for i, hyp in enumerate(cat_data["hyps"]):
            hyp_node_id = f"hyp_{cat_key}_{i}"
            verdict = hyp.get("verdict") or "unknown"
            status = verdict.lower()
            fill, stroke = status_colors.get(status, status_colors["unknown"])

            short_id = (hyp.get("id") or f"H{i}")[:35]

            if status == "refuted":
                killed_by = hyp.get("killed_by", "")
                label = f"{short_id}\\n❌ REFUTED\\n({killed_by})"
            elif status == "verified":
                label = f"{short_id}\\n✓ VERIFIED"
            else:
                label = f"{short_id}\\n? {status.upper()}"

            lines.append(
                f'    "{hyp_node_id}" [label="{label}", fillcolor="{fill}", '
                f'color="{stroke}", fontcolor="{stroke}", '
                f'fontsize=12, penwidth=2, width=4.0, height=0.8];'
            )

            # Connect to pillar
            lines.append(
                f'    "{pillar_id}" -> "{hyp_node_id}" [color="{stroke}", '
                f'style=dashed, penwidth=2, arrowsize=1.0];'
            )

            # Connect sequentially within category for layout
            if i > 0:
                prev_id = f"hyp_{cat_key}_{i-1}"
                lines.append(
                    f'    "{prev_id}" -> "{hyp_node_id}" [style=invis, weight=10];'
                )

        lines.append("  }")
        lines.append("")

    lines.append("}")
    return "\n".join(lines)


def render_dot(dot_text: str, output_path: Path, dpi: int = 300) -> bool:
    """Render DOT to PNG with high quality."""
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
    parser = argparse.ArgumentParser(description="Generate readable evolution visualization")
    parser.add_argument("--experiment-dir",
                        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
                        help="Path to experiment run directory")
    parser.add_argument("--output", default="knowledge_graph/visuals/evolution_readable.png",
                        help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution (default 300, use 600 for print)")
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

    # Count by category
    categories = {"data": 0, "neural": 0, "training": 0}
    for hyp in hypotheses:
        cat = find_parent_category(hyp)
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  Data Pipeline: {categories['data']}")
    print(f"  Neural Network: {categories['neural']}")
    print(f"  Training: {categories['training']}")

    print(f"\nGenerating readable visualization at {args.dpi} DPI...")
    dot = build_readable_dot(hypotheses, f"Knowledge Evolution - {len(hypotheses)} Hypotheses")

    if render_dot(dot, output_path, args.dpi):
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Generated: {output_path} ({size_mb:.1f} MB)")
        print(f"  Dimensions: {args.dpi} DPI")
        print(f"  Readable at: zoom 100-200% for detail")
        return 0
    else:
        print("✗ Failed to generate")
        return 1


if __name__ == "__main__":
    exit(main())
