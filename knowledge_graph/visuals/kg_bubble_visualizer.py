#!/usr/bin/env python3
"""
Bubble-Cluster Knowledge Graph Visualizer

Creates a visualization matching the reference image style:
- Three large colored circles (Data=blue, Architecture=red, Optimization=green)
- Medium circles attached to roots (branches)
- Small circles attached to branches (leaves)
- Dashed connections for new hypotheses (H1 style)
- Clean, minimal aesthetic with good spacing

Usage:
    python3 knowledge_graph/visuals/kg_bubble_visualizer.py
    python3 knowledge_graph/visuals/kg_bubble_visualizer.py --with-hypotheses
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# Color Scheme (from reference image)
# ============================================================================

ROOT_COLORS = {
    "node_root_data_pipeline": {
        "fill": "#3b82f6",      # Blue-500
        "stroke": "#2563eb",    # Blue-600
    },
    "node_root_neural_network": {
        "fill": "#ef4444",      # Red-500
        "stroke": "#dc2626",    # Red-600
    },
    "node_root_training_eval": {
        "fill": "#22c55e",      # Green-500
        "stroke": "#16a34a",    # Green-600
    },
}

BRANCH_COLORS = [
    ("#2dd4bf", "#14b8a6"),   # Teal (from ref image - Data branch)
    ("#fbbf24", "#f59e0b"),   # Amber (from ref image - Architecture branch)
    ("#fb923c", "#f97316"),   # Orange (from ref image - Optimization branch)
]

LEAF_COLORS = [
    ["#a5b4fc", "#c4b5fd"],   # Indigo/Violet leaves
    ["#fda4af", "#fdb4b6"],   # Rose/Pink leaves
    ["#86efac", "#bbf7d0"],   # Light green leaves
]

HYPOTHESIS_STYLE = {
    "pending": {
        "fill": "#e0f2fe",      # Light blue fill
        "stroke": "#3b82f6",    # Blue dashed border
        "stroke_style": "dashed",
    },
    "verified": {
        "fill": "#dcfce7",      # Light green fill
        "stroke": "#22c55e",    # Green solid border
        "stroke_style": "solid",
    },
    "refuted": {
        "fill": "#fee2e2",      # Light red fill
        "stroke": "#ef4444",    # Red solid border
        "stroke_style": "solid",
    },
}


def _escape(text: str) -> str:
    return text.replace('"', '\\"').replace("\n", " ")


def _short_label(text: str, max_len: int = 20) -> str:
    """Shorten label for display in bubble."""
    words = text.replace("/", " ").split()
    # Try to extract meaningful short name
    if len(words) <= 2:
        return text[:max_len]

    # Skip common prefixes
    skip_words = {"the", "a", "an", "of", "for", "in", "to", "and", "with", "on", "at"}
    meaningful = [w for w in words if w.lower() not in skip_words]

    if len(meaningful) >= 2:
        return (meaningful[0] + meaningful[1])[:max_len]
    return text[:max_len]


# ============================================================================
# Data Loading
# ============================================================================

def load_graph(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text())
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses(outbox: Path) -> list[dict]:
    """Load hypotheses from falsifier results and ideator outputs."""
    hypotheses = []

    # Load falsifier results for verdicts
    falsifier_dir = outbox / "falsifier"
    verdicts = {}
    if falsifier_dir.exists():
        for f in falsifier_dir.glob("*_result.json"):
            try:
                data = json.loads(f.read_text())
                verdicts[data.get("theory_id", "")] = {
                    "verdict": data.get("verdict", "PENDING").lower(),
                    "reason": data.get("kill_reason", ""),
                }
            except:
                pass

    # Load ideator ideas
    ideator_dir = outbox / "ideator"
    if ideator_dir.exists():
        for f in ideator_dir.glob("*.json"):
            if "_train_gpt" in f.name or f.name == "latest.json":
                continue
            try:
                data = json.loads(f.read_text())
                idea_id = data.get("idea_id", f.stem)
                verdict_info = verdicts.get(idea_id, {})

                hypotheses.append({
                    "id": idea_id,
                    "title": data.get("title", "Unknown"),
                    "summary": data.get("novelty_summary", "")[:100],
                    "status": verdict_info.get("verdict", "pending"),
                    "reason": verdict_info.get("reason", ""),
                })
            except:
                pass

    return hypotheses


def build_structure(nodes: list[dict], edges: list[dict]) -> dict:
    """Build hierarchical structure from flat nodes/edges."""

    # Build relationships
    children = defaultdict(list)
    parents = defaultdict(list)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            children[src].append(tgt)
            parents[tgt].append(src)

    # Find roots
    root_ids = ["node_root_data_pipeline", "node_root_neural_network", "node_root_training_eval"]

    structure = {}
    node_lookup = {n["id"]: n for n in nodes if n.get("id")}

    for root_id in root_ids:
        if root_id not in node_lookup:
            continue

        root_node = node_lookup[root_id]
        structure[root_id] = {
            "id": root_id,
            "label": root_node.get("label", root_id),
            "type": "RootBox",
            "branches": [],
        }

        # Find direct branch children
        for branch_id in children.get(root_id, []):
            if branch_id not in node_lookup:
                continue
            branch_node = node_lookup[branch_id]
            branch_data = {
                "id": branch_id,
                "label": branch_node.get("label", branch_id),
                "type": "Branch",
                "leaves": [],
            }

            # Find leaf children of this branch
            for leaf_id in children.get(branch_id, []):
                if leaf_id not in node_lookup:
                    continue
                # Check if it's a leaf (has no children or only one level deep)
                leaf_children = children.get(leaf_id, [])
                is_leaf = not leaf_children or all(
                    node_lookup.get(cid, {}).get("type") == "Leaf" for cid in leaf_children
                )

                leaf_node = node_lookup[leaf_id]
                leaf_data = {
                    "id": leaf_id,
                    "label": leaf_node.get("label", leaf_id),
                    "type": "Leaf",
                    "children": [],
                }

                # Add sub-leaves if any
                for sub_id in leaf_children:
                    if sub_id in node_lookup:
                        sub_node = node_lookup[sub_id]
                        leaf_data["children"].append({
                            "id": sub_id,
                            "label": sub_node.get("label", sub_id),
                            "type": "Leaf",
                        })

                branch_data["leaves"].append(leaf_data)

            structure[root_id]["branches"].append(branch_data)

    return structure


def build_bubble_dot(
    structure: dict,
    hypotheses: list[dict] | None = None,
    *,
    title: str = "Knowledge Graph",
) -> str:
    """Build DOT with bubble-cluster layout."""

    lines = []
    lines.append("digraph KG {")
    lines.append('  graph [bgcolor="white", fontname="Helvetica Neue",')
    lines.append('         fontsize=16, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.4, ranksep=1.2, splines="curved",')
    lines.append('         overlap="scale", pad=0.8];')
    lines.append('  node [fontname="Helvetica Neue", shape=circle,')
    lines.append('        fixedsize=true, style="filled", penwidth=2];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.6, penwidth=1.5];')
    lines.append("")

    if title:
        lines.append(f'  label="{_escape(title)}";')
        lines.append("")

    # Root positions (manually placed for visual balance)
    root_positions = {
        "node_root_data_pipeline": (2, 5),
        "node_root_neural_network": (8, 5),
        "node_root_training_eval": (14, 5),
    }

    root_ids = list(structure.keys())

    # Place roots at same rank
    lines.append("  // Roots (main clusters)")
    lines.append("  { rank=same; ")
    for root_id in root_ids:
        x, y = root_positions.get(root_id, (0, 0))
        root = structure[root_id]
        colors = ROOT_COLORS.get(root_id, {"fill": "#666", "stroke": "#333"})

        lines.append(
            f'    "{root_id}" [pos="{x},{y}!", width=1.2, height=1.2, '
            f'fillcolor="{colors["fill"]}", color="{colors["stroke"]}", '
            f'fontsize=11, fontcolor="white", label="{_escape(root["label"][:15])}"];'
        )
    lines.append("  }")
    lines.append("")

    # Add branches radiating from roots
    lines.append("  // Branches (attached to roots)")
    branch_idx = 0
    for root_id in root_ids:
        root = structure[root_id]
        rx, ry = root_positions.get(root_id, (0, 0))

        for i, branch in enumerate(root["branches"][:3]):  # Limit to 3 branches per root for clarity
            branch_id = branch["id"]
            color_pair = BRANCH_COLORS[i % len(BRANCH_COLORS)]
            fill, stroke = color_pair

            # Position branch below root, offset horizontally
            bx = rx + (i - 1) * 2.5
            by = ry - 2.5

            lines.append(
                f'    "{branch_id}" [pos="{bx},{by}!", width=0.7, height=0.7, '
                f'fillcolor="{fill}", color="{stroke}", '
                f'fontsize=9, fontcolor="#1e293b", label="{_short_label(branch["label"])}"];'
            )
            lines.append(f'    "{root_id}" -> "{branch_id}" [penwidth=2, color="{stroke}55"];')

            # Add leaves radiating from branches
            leaf_colors = LEAF_COLORS[i % len(LEAF_COLORS)]
            for j, leaf in enumerate(branch["leaves"][:4]):  # Limit leaves per branch
                leaf_id = leaf["id"]
                fill = leaf_colors[j % len(leaf_colors)]

                # Position leaf below branch, fan out
                angle = (j - 1.5) * 0.8
                lx = bx + angle * 1.2
                ly = by - 1.8

                lines.append(
                    f'    "{leaf_id}" [pos="{lx},{ly}!", width=0.35, height=0.35, '
                    f'fillcolor="{fill}", color="{fill}", '
                    f'fontsize=0, label="", tooltip="{_escape(leaf["label"])}"];'
                )
                lines.append(
                    f'    "{branch_id}" -> "{leaf_id}" [penwidth=1, color="#cbd5e155", '
                    f'arrowhead="none"];'
                )

        branch_idx += 1

    lines.append("")

    # Add hypotheses as floating nodes with dashed connections
    if hypotheses:
        lines.append("  // Hypotheses (H1 style nodes)")

        # Position hypotheses below the main structure
        for i, hyp in enumerate(hypotheses[:5]):  # Limit to 5 visible hypotheses
            hyp_id = f'hyp_{i}'
            style = HYPOTHESIS_STYLE.get(hyp["status"], HYPOTHESIS_STYLE["pending"])

            # Position in a row below
            hx = 4 + i * 3
            hy = 0.5

            short_title = _short_label(hyp["title"], 15)
            lines.append(
                f'    "{hyp_id}" [pos="{hx},{hy}!", width=0.5, height=0.5, '
                f'fillcolor="{style["fill"]}", color="{style["stroke"]}", '
                f'style="filled,{style["stroke_style"]}", '
                f'fontsize=8, fontcolor="#0f172a", label="H{i+1}", '
                f'tooltip="{_escape(hyp["title"])}\n{_escape(hyp["summary"])}"];'
            )

            # Dashed connections to related roots
            if i % 3 == 0 and "node_root_data_pipeline" in root_ids:
                lines.append(
                    f'    "{hyp_id}" -> "node_root_data_pipeline" '
                    f'[color="{style["stroke"]}", style="dashed", penwidth=1.5, '
                    f'constraint=false];'
                )
            elif i % 3 == 1 and "node_root_neural_network" in root_ids:
                lines.append(
                    f'    "{hyp_id}" -> "node_root_neural_network" '
                    f'[color="{style["stroke"]}", style="dashed", penwidth=1.5, '
                    f'constraint=false];'
                )
            elif i % 3 == 2 and "node_root_training_eval" in root_ids:
                lines.append(
                    f'    "{hyp_id}" -> "node_root_training_eval" '
                    f'[color="{style["stroke"]}", style="dashed", penwidth=1.5, '
                    f'constraint=false];'
                )

    lines.append("")
    lines.append("}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bubble-cluster knowledge graph visualizer"
    )
    parser.add_argument(
        "--seed",
        default="knowledge_graph/seed_parameter_golf_kg.json",
    )
    parser.add_argument(
        "--outbox",
        default="knowledge_graph/outbox",
    )
    parser.add_argument(
        "--out-dir",
        default="knowledge_graph/visuals",
    )
    parser.add_argument(
        "--basename",
        default="kg_bubble",
    )
    parser.add_argument(
        "--with-hypotheses",
        action="store_true",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz engine (neato, fdp for force-directed)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    seed_path = Path(args.seed)
    outbox_path = Path(args.outbox)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    nodes, edges = load_graph(seed_path)
    structure = build_structure(nodes, edges)
    hypotheses = load_hypotheses(outbox_path) if args.with_hypotheses else []

    # Build DOT
    dot_text = build_bubble_dot(
        structure,
        hypotheses if hypotheses else None,
        title="Knowledge Graph" + (" + Active Research" if hypotheses else ""),
    )

    # Output
    dot_path = out_dir / f"{args.basename}.dot"
    png_path = out_dir / f"{args.basename}.png"
    svg_path = out_dir / f"{args.basename}.svg"

    dot_path.write_text(dot_text)

    try:
        subprocess.run(
            [args.engine, f"-Gdpi={args.dpi}", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [args.engine, "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            capture_output=True,
        )
        print(f"Generated:")
        print(f"  {dot_path}")
        print(f"  {png_path}")
        print(f"  {svg_path}")
        if hypotheses:
            print(f"  Included {len(hypotheses)} hypothesis(es)")
    except subprocess.CalledProcessError as e:
        print(f"Render error: {e}")
        print(f"DOT saved: {dot_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
