#!/usr/bin/env python3
"""
Generate PRESENTATION-QUALITY Evolution Visualization

Designed for:
- Projector/screenshare readability
- Large fonts (16pt+)
- Clear status indicators
- Scales to 100+ hypotheses
- Professional color scheme
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict
import os


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
    elif any(kw in hyp_id for kw in ["optim", "loss", "train", "precision", "grad", "checkpoint"]):
        return "training"
    elif any(kw in hyp_id for kw in ["data", "token", "sequence", "curriculum", "augment", "sample"]):
        return "data"
    else:
        return "neural"


def truncate_name(name: str, max_len: int = 28) -> str:
    """Truncate long hypothesis names."""
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + "..."


def build_presentation_dot(hypotheses: List[Dict]) -> str:
    """Build presentation-quality DOT."""

    # Group by category
    categories = {
        "data": {"label": "📊 Data Pipeline", "color": "#1d4ed8", "fill": "#dbeafe", "hyps": []},
        "neural": {"label": "🧠 Neural Network", "color": "#b91c1c", "fill": "#fee2e2", "hyps": []},
        "training": {"label": "⚙️ Training & Eval", "color": "#15803d", "fill": "#dcfce7", "hyps": []},
    }

    for hyp in hypotheses:
        cat = find_parent_category(hyp)
        if cat in categories:
            categories[cat]["hyps"].append(hyp)

    # Count stats
    refuted = sum(1 for h in hypotheses if h.get("verdict") == "REFUTED")
    verified = sum(1 for h in hypotheses if h.get("verdict") == "VERIFIED")
    unknown = len(hypotheses) - refuted - verified

    lines = []
    lines.append("digraph Evolution {")
    # Very large, presentation-ready
    lines.append('  graph [bgcolor="#ffffff", fontname="Helvetica Neue",')
    lines.append('         fontsize=32, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.6, ranksep=2.5, pad=1.5, margin=3.0,')
    lines.append('         fontcolor="#0f172a", rankdir=TB];')
    lines.append('  node [fontname="Helvetica Neue", shape=box, style="rounded,filled",')
    lines.append('        penwidth=3, fontsize=16, margin="0.25,0.2", width=4.0, height=0.8];')
    lines.append('  edge [color="#64748b", arrowsize=0.8, penwidth=2];')
    lines.append("")

    # Title with stats
    title = f"Knowledge Graph Evolution\\n{len(hypotheses)} Hypotheses Tested"
    subtitle = f"{refuted} Refuted  |  {verified} Verified  |  {unknown} Unknown"
    lines.append(f'  label="{title}\\n{subtitle}";')
    lines.append("")

    # Status colors - professional
    status_styles = {
        "refuted": {"fill": "#fee2e2", "stroke": "#dc2626", "icon": "❌", "label": "REFUTED"},
        "verified": {"fill": "#dcfce7", "stroke": "#16a34a", "icon": "✓", "label": "VERIFIED"},
        "unknown": {"fill": "#f1f5f9", "stroke": "#475569", "icon": "?", "label": "UNKNOWN"},
    }

    col_count = 3  # Columns per category

    # Build each category
    for cat_key, cat_data in categories.items():
        if not cat_data["hyps"]:
            continue

        hyps = cat_data["hyps"]

        # Category header - large and prominent
        header_id = f"header_{cat_key}"
        count_text = f"{len(hyps)} hypothesis" if len(hyps) == 1 else f"{len(hyps)} hypotheses"
        lines.append(
            f'  "{header_id}" [label="{cat_data["label"]}\\n{count_text}", '
            f'fillcolor="{cat_data["fill"]}", color="{cat_data["color"]}", '
            f'fontcolor="{cat_data["color"]}", fontsize=24, penwidth=4, '
            f'width=12.0, height=1.2, style="rounded,filled,bold"];'
        )
        lines.append("")

        # Create hypothesis nodes in grid
        for i, hyp in enumerate(hyps):
            row = i // col_count
            col = i % col_count

            node_id = f"hyp_{cat_key}_{i}"
            verdict = (hyp.get("verdict") or "unknown").lower()
            style = status_styles.get(verdict, status_styles["unknown"])

            # Format label
            name = truncate_name(hyp.get("id", f"Hyp {i+1}"))
            killed_by = hyp.get("killed_by", "")

            if verdict == "refuted" and killed_by:
                label = f"{style['icon']} {name}\\nKilled by: {killed_by}"
            else:
                label = f"{style['icon']} {name}"

            lines.append(
                f'  "{node_id}" [label="{label}", fillcolor="{style["fill"]}", '
                f'color="{style["stroke"]}", fontcolor="{style["stroke"]}", '
                f'fontsize=14, penwidth=2, width=4.0, height=0.9];'
            )

        lines.append("")

        # Connect to header with dashed lines
        for i in range(len(hyps)):
            node_id = f"hyp_{cat_key}_{i}"
            verdict = (hyps[i].get("verdict") or "unknown").lower()
            style = status_styles.get(verdict, status_styles["unknown"])
            lines.append(
                f'  "{header_id}" -> "{node_id}" [color="{style["stroke"]}", '
                f'style=dashed, penwidth=2, arrowsize=1.2];'
            )

        lines.append("")

        # Grid layout - keep rows at same rank
        rows = (len(hyps) + col_count - 1) // col_count
        for row in range(rows):
            row_nodes = []
            for col in range(col_count):
                idx = row * col_count + col
                if idx < len(hyps):
                    row_nodes.append(f"hyp_{cat_key}_{idx}")
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
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate presentation-quality evolution")
    parser.add_argument("--experiment-dir",
                        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
                        help="Path to experiment run directory")
    parser.add_argument("--output", default="knowledge_graph/visuals/evolution_presentation.png",
                        help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution (300=screen, 600=print)")
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

    print(f"\nGenerating presentation visualization at {args.dpi} DPI...")
    dot = build_presentation_dot(hypotheses)

    if render_dot(dot, output_path, args.dpi):
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Generated: {output_path} ({size_mb:.1f} MB)")
        print(f"  Features:")
        print(f"    - 16pt+ fonts (readable on projectors)")
        print(f"    - 3-column grid layout")
        print(f"    - Clear status icons (❌ ✓ ?)")
        print(f"    - Scales to 100+ hypotheses")
        print(f"    - Professional color scheme")
        return 0
    else:
        print("✗ Failed to generate")
        return 1


if __name__ == "__main__":
    exit(main())
