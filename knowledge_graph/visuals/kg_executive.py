#!/usr/bin/env python3
"""
Executive/Pitch-Deck Knowledge Graph Visualizer

Compact, focused visualization for executive presentations.
- Vertical layout (better for slide decks)
- Three clear columns
- Icon-style nodes with labels
- Clean typography
- Emphasizes the three pillars
- Shows depth without overwhelming

Usage:
    python3 knowledge_graph/visuals/kg_executive.py
    python3 knowledge_graph/visuals/kg_executive.py --with-hypotheses
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


# Executive color palette
COLORS = {
    "bg": "#ffffff",
    "text": "#111827",
    "subtext": "#6b7280",
    "border": "#e5e7eb",

    "data": {
        "primary": "#2563eb",    # Blue 600
        "secondary": "#3b82f6",  # Blue 500
        "light": "#dbeafe",      # Blue 100
        "pale": "#eff6ff",       # Blue 50
    },
    "nn": {
        "primary": "#dc2626",    # Red 600
        "secondary": "#ef4444",  # Red 500
        "light": "#fee2e2",      # Red 100
        "pale": "#fef2f2",       # Red 50
    },
    "train": {
        "primary": "#16a34a",    # Green 600
        "secondary": "#22c55e",  # Green 500
        "light": "#dcfce7",      # Green 100
        "pale": "#f0fdf4",       # Green 50
    },
}

HYPOTHESIS = {
    "pending": ("#fffbeb", "#f59e0b"),   # Amber
    "verified": ("#ecfdf5", "#10b981"),  # Green
    "refuted": ("#fef2f2", "#ef4444"),  # Red
}


def load_graph(path: Path):
    data = json.loads(path.read_text())
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses(outbox: Path):
    hypotheses = []
    falsifier_dir = outbox / "falsifier"
    ideator_dir = outbox / "ideator"

    verdicts = {}
    if falsifier_dir.exists():
        for f in falsifier_dir.glob("*_result.json"):
            try:
                data = json.loads(f.read_text())
                verdicts[data.get("theory_id", "")] = data.get("verdict", "PENDING").lower()
            except:
                pass

    if ideator_dir.exists():
        for f in ideator_dir.glob("*.json"):
            if "_train_gpt" in f.name or f.name == "latest.json":
                continue
            try:
                data = json.loads(f.read_text())
                idea_id = data.get("idea_id", f.stem)
                hypotheses.append({
                    "id": idea_id,
                    "title": data.get("title", "Unknown"),
                    "summary": data.get("novelty_summary", "")[:200],
                    "status": verdicts.get(idea_id, "pending"),
                })
            except:
                pass

    return hypotheses


def _escape(text: str) -> str:
    """Escape special characters for DOT HTML labels."""
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def build_exec_dot(
    nodes: list,
    edges: list,
    hypotheses: list | None = None,
    title: str = "Knowledge Architecture",
) -> str:
    """Build executive-style DOT graph."""

    node_map = {n["id"]: n for n in nodes if n.get("id")}

    # Build relationships
    children = defaultdict(list)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            children[src].append(tgt)

    # Root info
    pillars = [
        ("node_root_data_pipeline", "Data Pipeline", "data"),
        ("node_root_neural_network", "Neural Network", "nn"),
        ("node_root_training_eval", "Training &amp; Evaluation", "train"),
    ]

    lines = []
    lines.append("digraph ExecutiveKG {")
    lines.append(f'  graph [bgcolor="{COLORS["bg"]}", fontname="SF Pro Display,Inter,Helvetica",')
    lines.append(f'         fontsize=28, labelloc="t", labeljust="center",')
    lines.append(f'         nodesep=0.8, ranksep=2.5, splines="ortho",')
    lines.append(f'         overlap="scale", pad=2.0, margin=0,' )
    lines.append(f'         fontcolor="{COLORS["text"]}", rankdir=TB];')
    lines.append(f'  node [fontname="SF Pro Display,Inter,Helvetica", shape=box,')
    lines.append(f'        style="rounded,filled", penwidth=2,')
    lines.append(f'        fontcolor="{COLORS["text"]}", margin="0.25,0.15"];')
    lines.append(f'  edge [color="{COLORS["border"]}", arrowsize=0.7, penwidth=2,' )
    lines.append(f'        dir="forward", arrowhead="normal"];')
    lines.append("")

    if title:
        title_escaped = _escape(title)
        lines.append(f'  label=<<FONT POINT-SIZE="32" FACE="SF Pro Display,Inter,Helvetica"><B>{title_escaped}</B></FONT>>;')
        lines.append("")

    # Create three subgraphs side by side
    for root_id, root_label, cat in pillars:
        if root_id not in node_map:
            continue

        c = COLORS[cat]

        lines.append(f'  subgraph cluster_{cat} {{')
        lines.append(f'    label=<<FONT POINT-SIZE="18" COLOR="{c["primary"]}"><B>{root_label}</B></FONT>>;')
        lines.append(f'    style="rounded,filled";')
        lines.append(f'    fillcolor="{c["pale"]}";')
        lines.append(f'    color="{c["primary"]}40";')
        lines.append(f'    penwidth=2;')
        lines.append("")

        # Root node
        root_label_escaped = _escape(root_label)
        lines.append(
            f'    "{root_id}" [label=<<B>{root_label_escaped}</B>>, fillcolor="{c["primary"]}", '
            f'fontcolor="#ffffff", width=3.0, height=0.8, fontsize=14, penwidth=3];'
        )

        # Get branches (direct children)
        branches = children.get(root_id, [])[:8]  # Limit to 8 for clarity

        for i, branch_id in enumerate(branches):
            if branch_id not in node_map:
                continue

            branch = node_map[branch_id]
            branch_label = _escape(branch.get("label", branch_id))

            lines.append(
                f'    "{branch_id}" [label="{branch_label}", fillcolor="{c["secondary"]}", '
                f'fontcolor="#ffffff", width=2.8, height=0.6, fontsize=11, penwidth=2];'
            )
            lines.append(f'    "{root_id}" -> "{branch_id}" [color="{c["primary"]}60"];')

            # Get leaves for this branch (max 4 per branch)
            leaves = children.get(branch_id, [])[:4]
            for j, leaf_id in enumerate(leaves):
                if leaf_id not in node_map:
                    continue

                leaf = node_map[leaf_id]
                leaf_label = _escape(leaf.get("label", leaf_id))

                # Compact leaf representation
                lines.append(
                    f'    "{leaf_id}" [label="{leaf_label}", fillcolor="{c["light"]}", '
                    f'fontcolor="{c["primary"]}", width=2.6, height=0.4, fontsize=9, penwidth=1];'
                )
                lines.append(f'    "{branch_id}" -> "{leaf_id}" [color="{c["secondary"]}40", penwidth=1];')

        lines.append("  }")
        lines.append("")

    # Add hypotheses floating at bottom
    if hypotheses:
        lines.append("  // Research Hypotheses")
        lines.append('  { rank=same; ')

        for i, hyp in enumerate(hypotheses[:5]):
            hyp_id = f"hyp_{i}"
            fill, stroke = HYPOTHESIS.get(hyp["status"], HYPOTHESIS["pending"])
            style = "filled,rounded,dashed" if hyp["status"] == "pending" else "filled,rounded"

            short_title = _escape(hyp["title"][:30])
            if len(hyp["title"]) > 30:
                short_title += "..."

            lines.append(
                f'    "{hyp_id}" [label="{short_title}", fillcolor="{fill}", '
                f'color="{stroke}", style="{style}", fontcolor="{stroke}", '
                f'fontsize=10, width=2.0, height=0.5, penwidth=2];'
            )

        lines.append("}")

        # Connect hypotheses
        for i, hyp in enumerate(hypotheses[:5]):
            hyp_id = f"hyp_{i}"
            fill, stroke = HYPOTHESIS.get(hyp["status"], HYPOTHESIS["pending"])

            # Determine which pillar
            title_lower = hyp["title"].lower()
            if any(w in title_lower for w in ["attention", "mlp", "layer", "embed", "transformer", "rank", "factor"]):
                target = "node_root_neural_network"
            elif any(w in title_lower for w in ["optim", "loss", "train", "precision", "grad"]):
                target = "node_root_training_eval"
            else:
                target = "node_root_data_pipeline"

            lines.append(
                f'  "{hyp_id}" -> "{target}" [color="{stroke}", style="dashed", '
                f'penwidth=1.5, constraint=false, arrowhead="none"];'
            )

    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Executive knowledge graph visualizer")
    parser.add_argument("--seed", default="knowledge_graph/seed_parameter_golf_kg.json")
    parser.add_argument("--outbox", default="knowledge_graph/outbox")
    parser.add_argument("--out-dir", default="knowledge_graph/visuals")
    parser.add_argument("--basename", default="kg_executive")
    parser.add_argument("--with-hypotheses", action="store_true")
    parser.add_argument("--title", default="Parameter Golf: Knowledge Architecture")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    seed_path = Path(args.seed)
    outbox_path = Path(args.outbox)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_graph(seed_path)
    hypotheses = load_hypotheses(outbox_path) if args.with_hypotheses else []

    dot_text = build_exec_dot(nodes, edges, hypotheses if hypotheses else None, title=args.title)

    suffix = "_with_research" if args.with_hypotheses else ""
    dot_path = out_dir / f"{args.basename}{suffix}.dot"
    png_path = out_dir / f"{args.basename}{suffix}.png"
    svg_path = out_dir / f"{args.basename}{suffix}.svg"

    dot_path.write_text(dot_text)

    try:
        subprocess.run(
            ["dot", f"-Gdpi={args.dpi}", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True, capture_output=True,
        )

        print(f"✓ Generated executive visualization:")
        print(f"  {png_path}")
        print(f"  {svg_path}")
        if hypotheses:
            print(f"  Included {len(hypotheses)} research hypothesis(es)")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"  DOT saved: {dot_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
