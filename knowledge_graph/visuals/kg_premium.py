#!/usr/bin/env python3
"""
Premium Knowledge Graph Visualizer

Presentation-quality visualization with:
- Large, properly sized nodes for clear labels
- Hierarchical tree layout with proper spacing
- Publication-ready typography
- Distinct visual hierarchy (root > branch > leaf)
- Optional research hypotheses overlay
- Light and dark themes

Suitable for: Research papers, conference presentations, investor pitches
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


# Professional color palette
COLORS = {
    "light": {
        "bg": "#ffffff",
        "text": "#1f2937",
        "text_light": "#6b7280",
        "edge": "#d1d5db",
        "roots": {
            "data": "#1e40af",     # Blue 800
            "nn": "#991b1b",       # Red 800
            "train": "#166534",    # Green 800
        },
        "branches": {
            "data": "#3b82f6",     # Blue 500
            "nn": "#f59e0b",       # Amber 500
            "train": "#10b981",    # Emerald 500
        },
        "leaves": {
            "data": "#93c5fd",     # Blue 300
            "nn": "#fcd34d",       # Amber 300
            "train": "#6ee7b7",    # Emerald 300
        },
    },
    "dark": {
        "bg": "#111827",
        "text": "#f9fafb",
        "text_light": "#9ca3af",
        "edge": "#374151",
        "roots": {
            "data": "#60a5fa",     # Blue 400
            "nn": "#f87171",       # Red 400
            "train": "#4ade80",    # Green 400
        },
        "branches": {
            "data": "#3b82f6",     # Blue 500
            "nn": "#fbbf24",       # Amber 400
            "train": "#34d399",    # Emerald 400
        },
        "leaves": {
            "data": "#1e40af",     # Blue 800
            "nn": "#b45309",       # Amber 700
            "train": "#065f46",    # Emerald 800
        },
    },
}

HYPOTHESIS = {
    "pending": ("#fef3c7", "#f59e0b"),   # Amber
    "verified": ("#d1fae5", "#10b981"),  # Emerald
    "refuted": ("#fee2e2", "#ef4444"),  # Red
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


def build_tree_structure(nodes: list, edges: list):
    """Build hierarchical tree from flat nodes."""
    node_map = {n["id"]: n for n in nodes if n.get("id")}
    children = defaultdict(list)
    parents = {}

    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            children[src].append(tgt)
            parents[tgt] = src

    # Find roots
    root_ids = [
        "node_root_data_pipeline",
        "node_root_neural_network",
        "node_root_training_eval",
    ]

    categories = {
        "node_root_data_pipeline": "data",
        "node_root_neural_network": "nn",
        "node_root_training_eval": "train",
    }

    # Build tree for each root
    trees = {}
    for root_id in root_ids:
        if root_id not in node_map:
            continue

        cat = categories.get(root_id, "data")
        root_node = node_map[root_id]

        def build_subtree(node_id, depth=0):
            node = node_map.get(node_id, {})
            node_type = node.get("type", "")
            label = node.get("label", node_id)

            subtree = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "depth": depth,
                "category": cat,
                "children": [],
            }

            for child_id in children.get(node_id, []):
                subtree["children"].append(build_subtree(child_id, depth + 1))

            return subtree

        trees[root_id] = build_subtree(root_id)

    return trees, node_map, children


def build_dot(
    trees: dict,
    node_map: dict,
    children: dict,
    hypotheses: list | None = None,
    theme: str = "light",
    title: str = "Parameter Golf Knowledge Graph",
) -> str:
    """Build premium DOT graph."""

    colors = COLORS[theme]
    bg = colors["bg"]
    text = colors["text"]

    lines = []
    lines.append("digraph KG {")
    lines.append(f'  graph [bgcolor="{bg}", fontname="Inter",')
    lines.append(f'         fontsize=20, labelloc="t", labeljust="center",')
    lines.append(f'         nodesep=0.6, ranksep=2.0, splines="ortho",')
    lines.append(f'         overlap="scale", pad=1.5, margin=0,' )
    lines.append(f'         fontcolor="{text}", bgcolor="{bg}"];')
    lines.append(f'  node [fontname="Inter", shape=box,')
    lines.append(f'        style="rounded,filled", penwidth=2,')
    lines.append(f'        fontcolor="{text}"];')
    lines.append(f'  edge [color="{colors["edge"]}", arrowsize=0.8, penwidth=1.5,' )
    lines.append(f'        dir="forward", arrowhead="normal"];')
    lines.append("")

    if title:
        lines.append(f'  label=<<FONT POINT-SIZE="24" FACE="Inter"><B>{title}</B></FONT>>;')
        lines.append("")

    # Process each tree
    for root_id, tree in trees.items():
        cat = tree["category"]

        def add_nodes(node, lines):
            nid = node["id"]
            label = node["label"]
            depth = node["depth"]
            ntype = node["type"]

            # Style based on type
            if ntype == "RootBox":
                color = colors["roots"][cat]
                width = 2.0
                height = 0.8
                fontsize = 14
                penwidth = 3
            elif ntype == "Branch":
                color = colors["branches"][cat]
                width = 1.8
                height = 0.6
                fontsize = 11
                penwidth = 2
            else:  # Leaf
                color = colors["leaves"][cat]
                width = 1.5
                height = 0.4
                fontsize = 9
                penwidth = 1

            fontcolor = "#ffffff" if theme == "light" and ntype in ["RootBox", "Branch"] else text
            if theme == "dark" and ntype == "Leaf":
                fontcolor = colors["text_light"]

            label = label.replace('"', '\\"')
            lines.append(
                f'  "{nid}" [label="{label}", fillcolor="{color}", '
                f'width={width}, height={height}, fontsize={fontsize}, '
                f'fontcolor="{fontcolor}", penwidth={penwidth}];'
            )

            for child in node["children"]:
                add_nodes(child, lines)
                lines.append(f'  "{nid}" -> "{child["id"]}";')

        add_nodes(tree, lines)
        lines.append("")

    # Add hypotheses
    if hypotheses:
        lines.append("  // Hypotheses")

        for i, hyp in enumerate(hypotheses[:6]):
            hyp_id = f"hyp_{i}"
            fill, stroke = HYPOTHESIS.get(hyp["status"], HYPOTHESIS["pending"])
            style = "filled,rounded,dashed" if hyp["status"] == "pending" else "filled,rounded"

            short_title = hyp["title"][:35] + "..." if len(hyp["title"]) > 35 else hyp["title"]
            short_title = short_title.replace('"', '\\"')

            lines.append(
                f'  "{hyp_id}" [label="{short_title}", fillcolor="{fill}", '
                f'color="{stroke}", style="{style}", fontcolor="{stroke}", '
                f'fontsize=10, width=2.5, height=0.5, penwidth=2];'
            )

            # Connect to relevant root
            roots_map = {
                "data": "node_root_data_pipeline",
                "nn": "node_root_neural_network",
                "train": "node_root_training_eval",
            }

            # Simple keyword matching to connect
            title_lower = hyp["title"].lower()
            target_root = None
            if any(w in title_lower for w in ["data", "token", "sequence"]):
                target_root = roots_map["data"]
            elif any(w in title_lower for w in ["attention", "mlp", "layer", "embed", "transformer"]):
                target_root = roots_map["nn"]
            elif any(w in title_lower for w in ["optim", "loss", "train", "precision"]):
                target_root = roots_map["train"]
            else:
                target_root = roots_map["nn"]  # Default

            lines.append(
                f'  "{hyp_id}" -> "{target_root}" [color="{stroke}", '
                f'style="dashed", penwidth=1.5, constraint=false];'
            )

    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Premium knowledge graph visualizer")
    parser.add_argument("--seed", default="knowledge_graph/seed_parameter_golf_kg.json")
    parser.add_argument("--outbox", default="knowledge_graph/outbox")
    parser.add_argument("--out-dir", default="knowledge_graph/visuals")
    parser.add_argument("--basename", default="kg_premium")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--with-hypotheses", action="store_true")
    parser.add_argument("--title", default="Parameter Golf: Knowledge Architecture")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    seed_path = Path(args.seed)
    outbox_path = Path(args.outbox)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_graph(seed_path)
    trees, node_map, children_map = build_tree_structure(nodes, edges)
    hypotheses = load_hypotheses(outbox_path) if args.with_hypotheses else []

    dot_text = build_dot(
        trees, node_map, children_map,
        hypotheses if hypotheses else None,
        theme=args.theme,
        title=args.title,
    )

    suffix = f"_{args.theme}" if args.theme == "dark" else ""
    if args.with_hypotheses:
        suffix += "_with_research"

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

        print(f"✓ Generated premium visualization:")
        print(f"  {png_path} ({args.dpi} DPI)")
        print(f"  {svg_path}")
        if hypotheses:
            print(f"  Included {len(hypotheses)} hypothesis(es)")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"  DOT saved: {dot_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
