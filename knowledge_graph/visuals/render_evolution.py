#!/usr/bin/env python3
"""
Render hypothesis evolution timeline visualization.

Shows the progression of hypotheses through the pipeline with verdicts.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict


def load_visualization_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_timeline_dot(data: dict) -> str:
    timeline = data.get("timeline", [])

    # Group by hypothesis
    by_hypothesis = defaultdict(list)
    for event in timeline:
        hyp = event.get("hypothesis", 0)
        by_hypothesis[hyp].append(event)

    lines = []
    lines.append("digraph Evolution {")
    lines.append('  graph [bgcolor="#fafafa", fontname="Helvetica",')
    lines.append('         fontsize=16, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.4, ranksep=1.5, pad=1.0];')
    lines.append('  node [fontname="Helvetica", shape=box, style="rounded,filled", penwidth=1.5];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.7, penwidth=1.2];')
    lines.append("")
    lines.append('  label="Hypothesis Evolution Timeline";')
    lines.append("")

    # Create nodes for each hypothesis
    colors = {
        "start": "#3b82f6",      # Blue
        "complete": "#22c55e",     # Green
        "REFUTED": "#ef4444",      # Red
        "UNKNOWN": "#f59e0b",      # Amber
    }

    for hyp_num, events in sorted(by_hypothesis.items()):
        if hyp_num == 0:
            continue

        # Find start and complete events
        start = next((e for e in events if e["event"] == "start"), None)
        complete = next((e for e in events if e["event"] == "complete"), None)

        if not start:
            continue

        theory_id = start.get("theory_id", f"hyp_{hyp_num}")
        label = theory_id.replace("-", " ").title()[:25]

        # Start node
        start_id = f"h{hyp_num}_start"
        lines.append(
            f'  "{start_id}" [label="H{hyp_num}: {label}\\n(start)", '
            f'fillcolor="{colors["start"]}", fontcolor="white", width=2.0, height=0.6];'
        )

        # Complete node if exists
        if complete:
            verdict = complete.get("verdict", "UNKNOWN")
            color = colors.get(verdict, colors["UNKNOWN"])

            complete_id = f"h{hyp_num}_complete"
            time_taken = complete.get("time", 0) - start.get("time", 0)

            lines.append(
                f'  "{complete_id}" [label="{verdict}\\n({time_taken:.1f}s)", '
                f'fillcolor="{color}", fontcolor="white", width=1.2, height=0.5];'
            )
            lines.append(f'  "{start_id}" -> "{complete_id}";')

        # Chain hypotheses
        if hyp_num > 1:
            prev_complete = f"h{hyp_num-1}_complete"
            if any(f'h{hyp_num-1}' in line for line in lines):
                lines.append(
                    f'  "{prev_complete}" -> "{start_id}" '
                    f'[style="dashed", color="#94a3b860", constraint=false];'
                )

    lines.append("}")
    return "\n".join(lines)


def main():
    viz_path = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/experiments/ten_hypothesis_run/live_run_20260328_170317/visualization/visualization_data.json")
    out_dir = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/knowledge_graph/visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_visualization_data(viz_path)
    dot_text = build_timeline_dot(data)

    dot_path = out_dir / "evolution_timeline.dot"
    png_path = out_dir / "evolution_timeline.png"
    svg_path = out_dir / "evolution_timeline.svg"

    dot_path.write_text(dot_text)

    try:
        subprocess.run(
            ["dot", "-Gdpi=300", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True, capture_output=True,
        )
        print(f"✓ Generated evolution timeline:")
        print(f"  {png_path}")
        print(f"  {svg_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"DOT saved to: {dot_path}")


if __name__ == "__main__":
    main()
