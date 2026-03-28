#!/usr/bin/env python3
"""
Unified Knowledge Graph Visualizer

Creates a hierarchical, circular visualization that:
- Shows the seed knowledge graph (RootBox → Branch → Leaf)
- Extends with new autoresearch hypotheses from outbox/
- Uses colors to distinguish the three main clusters

Usage:
    python3 knowledge_graph/visuals/knowledge_graph_visualizer.py
    python3 knowledge_graph/visuals/knowledge_graph_visualizer.py --with-hypotheses
    python3 knowledge_graph/visuals/knowledge_graph_visualizer.py --labels all --dpi 300
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# Color Palette (matching the reference image style)
# ============================================================================

ROOT_COLORS = {
    "node_root_data_pipeline": "#3b82f6",      # Blue
    "node_root_neural_network": "#ef4444",    # Red
    "node_root_training_eval": "#22c55e",     # Green
}

BRANCH_COLORS = {
    "data": ["#60a5fa", "#93c5fd", "#bfdbfe"],      # Blue shades
    "nn": ["#f87171", "#fca5a5", "#fecaca"],        # Red shades
    "train": ["#4ade80", "#86efac", "#bbf7d0"],     # Green shades
}

LEAF_COLORS = {
    "data": ["#818cf8", "#a5b4fc", "#c7d2fe"],      # Indigo-purple shades
    "nn": ["#fb923c", "#fdba74", "#fed7aa"],        # Orange shades
    "train": ["#c084fc", "#d8b4fe", "#e9d5ff"],     # Purple shades
}

HYPOTHESIS_COLORS = {
    "pending": "#fbbf24",     # Amber
    "verified": "#10b981",    # Emerald
    "refuted": "#ef4444",     # Red
    "connection": "#3b82f6",   # Blue dashed
}


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dot_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "graph"


def _hsl_to_hex(hue_deg: float, saturation: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# ============================================================================
# Graph Loading
# ============================================================================

def load_seed_graph(path: Path) -> tuple[list[dict], list[dict]]:
    """Load the seed knowledge graph."""
    data = json.loads(path.read_text())
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses(outbox_dir: Path) -> list[dict]:
    """Load hypothesis ideas from outbox/ideator."""
    hypotheses = []
    ideator_dir = outbox_dir / "ideator"

    if not ideator_dir.exists():
        return hypotheses

    # Look for idea JSON files
    for json_file in ideator_dir.glob("*.json"):
        if "_train_gpt" in json_file.name or json_file.name == "latest.json":
            continue

        try:
            data = json.loads(json_file.read_text())
            if "idea_id" in data or "title" in data:
                hypotheses.append({
                    "id": data.get("idea_id", json_file.stem),
                    "title": data.get("title", "Unknown Idea"),
                    "summary": data.get("novelty_summary", ""),
                    "source_file": str(json_file),
                    "type": "hypothesis",
                    "status": "pending",
                    "connects_to": _extract_connections(data),
                })
        except Exception:
            continue

    # Also load falsifier results to get verdicts
    falsifier_dir = outbox_dir / "falsifier"
    if falsifier_dir.exists():
        verdicts = {}
        for result_file in falsifier_dir.glob("*_result.json"):
            try:
                data = json.loads(result_file.read_text())
                theory_id = data.get("theory_id", "")
                verdict = data.get("verdict", "PENDING")
                if theory_id:
                    verdicts[theory_id] = verdict.lower()
            except Exception:
                continue

        # Update hypothesis statuses
        for h in hypotheses:
            if h["id"] in verdicts:
                h["status"] = verdicts[h["id"]]

    return hypotheses


def _extract_connections(data: dict) -> list[str]:
    """Extract which knowledge graph nodes this hypothesis connects to."""
    connections = []
    text = json.dumps(data).lower()

    # Map keywords to node IDs
    keyword_map = {
        "attention": ["node_transformer_attention", "node_attn_projections"],
        "mlp": ["node_transformer_mlp", "node_mlp_variant"],
        "embedding": ["node_nn_embeddings", "node_nn_embedding_strategy"],
        "optimizer": ["node_train_optimizer", "node_optimizer_state_strategy"],
        "loss": ["node_train_loss"],
        "normalization": ["node_transformer_norm"],
        "dropout": ["node_transformer_dropout"],
        "kv cache": ["node_attn_kv_cache"],
        "positional": ["node_nn_positional_encoding"],
        "data": ["node_root_data_pipeline"],
        "tokenization": ["node_data_tokenization"],
        "precision": ["node_train_precision"],
    }

    for keyword, nodes in keyword_map.items():
        if keyword in text:
            connections.extend(nodes)

    return list(set(connections))


# ============================================================================
# Graph Building
# ============================================================================

def build_clustered_dot(
    nodes: list[dict],
    edges: list[dict],
    hypotheses: list[dict] | None = None,
    *,
    labels: str = "roots-branches",
    title: str | None = "Parameter Golf Knowledge Graph",
    include_legend: bool = True,
) -> str:
    """
    Build a DOT graph with:
    - Three main clusters (Data Pipeline, Neural Network, Training & Evaluation)
    - Hierarchical node sizing (Root > Branch > Leaf)
    - Optional hypothesis nodes connected to relevant leaves
    """

    # Build node lookup and type mapping
    node_by_id = {n["id"]: n for n in nodes if n.get("id")}
    node_type = {n["id"]: n.get("type", "") for n in nodes if n.get("id")}

    # Build parent-child relationships
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            children[str(src)].append(str(tgt))
            parents[str(tgt)].append(str(src))

    # Assign nodes to roots
    root_ids = ["node_root_data_pipeline", "node_root_neural_network", "node_root_training_eval"]
    root_categories = {
        "node_root_data_pipeline": "data",
        "node_root_neural_network": "nn",
        "node_root_training_eval": "train",
    }

    node_to_root: dict[str, str] = {}
    for root_id in root_ids:
        reachable = _get_reachable(root_id, children)
        for node_id in reachable:
            node_to_root[node_id] = root_id

    # Assign colors
    node_color = {}
    for node_id, root_id in node_to_root.items():
        cat = root_categories.get(root_id, "data")
        node_type_val = node_type.get(node_id, "")

        if node_type_val == "RootBox":
            node_color[node_id] = ROOT_COLORS.get(root_id, "#666666")
        elif node_type_val == "Branch":
            # Assign branch color based on parent root
            branches = BRANCH_COLORS.get(cat, ["#999999"])
            idx = hash(node_id) % len(branches)
            node_color[node_id] = branches[idx]
        else:  # Leaf
            leaves = LEAF_COLORS.get(cat, ["#cccccc"])
            idx = hash(node_id) % len(leaves)
            node_color[node_id] = leaves[idx]

    # DOT generation
    lines: list[str] = []
    lines.append("digraph KnowledgeGraph {")
    lines.append('  graph [rankdir=TB, bgcolor="white", fontname="Helvetica",')
    lines.append('         fontsize=14, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.25, ranksep=0.6, splines="true",')
    lines.append('         overlap="scalexy", pad=0.5, bgcolor="#fafafa"];')
    lines.append('  node [fontname="Helvetica", fontsize=9, shape=circle,')
    lines.append('        fixedsize=true, style="filled", penwidth=1.5];')
    lines.append('  edge [color="#64748b", arrowsize=0.7, penwidth=1.2,')
    lines.append('        dir="forward", arrowhead="normal"];')
    lines.append("")

    if title:
        lines.append(f'  label=<<B>{_html_escape(title)}</B>>;')
        lines.append("")

    # Create subgraph clusters for each root
    for root_id in root_ids:
        root_node = node_by_id.get(root_id, {})
        root_label = root_node.get("label", root_id)
        cat = root_categories.get(root_id, "data")

        # Get all nodes in this cluster
        cluster_nodes = [n for n in nodes if node_to_root.get(n["id"]) == root_id]

        lines.append(f'  subgraph cluster_{cat} {{')
        lines.append(f'    label=<<B>{_html_escape(root_label)}</B>>;')
        lines.append(f'    style="rounded,filled";')
        lines.append(f'    fillcolor="#f1f5f9";')
        lines.append(f'    color="{ROOT_COLORS.get(root_id, "#cccccc")}";')
        lines.append(f'    penwidth=2;')
        lines.append("")

        # Add nodes in hierarchy order
        hierarchy = _build_hierarchy(root_id, children)
        for level_nodes in hierarchy:
            rank_line = "    { rank=same; "
            for node_id in level_nodes:
                node = node_by_id.get(node_id, {})
                label = node.get("label", node_id)
                ntype = node_type.get(node_id, "")
                color = node_color.get(node_id, "#999999")

                # Size based on type
                if ntype == "RootBox":
                    size = 0.6
                    fontsize = 11
                    penwidth = 3
                elif ntype == "Branch":
                    size = 0.4
                    fontsize = 9
                    penwidth = 2
                else:  # Leaf
                    size = 0.25
                    fontsize = 8
                    penwidth = 1.5

                # Determine if we show label
                show_label = (
                    (ntype == "RootBox" and labels in ["roots", "roots-branches", "all"]) or
                    (ntype == "Branch" and labels in ["roots-branches", "all"]) or
                    (ntype == "Leaf" and labels == "all")
                )

                if show_label:
                    label_attr = f'xlabel="{_dot_escape(label)}" fontcolor="#334155" fontsize={fontsize}'
                else:
                    label_attr = f'tooltip="{_dot_escape(label)}"'

                lines.append(
                    f'    "{node_id}" [width={size}, height={size}, fillcolor="{color}", '
                    f'penwidth={penwidth}, {label_attr}];'
                )
                rank_line += f'"{node_id}"; '

            rank_line += "}"
            lines.append(rank_line)
            lines.append("")

        lines.append("  }")
        lines.append("")

    # Add edges (only within clusters, not between)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            # Skip if going between different roots
            if node_to_root.get(src) == node_to_root.get(tgt):
                lines.append(f'  "{src}" -> "{tgt}" [color="#64748b55", penwidth=1];')

    # Add hypothesis nodes if provided
    if hypotheses:
        lines.append("")
        lines.append("  // Hypothesis nodes")

        for i, hyp in enumerate(hypotheses):
            hyp_id = f'hypothesis_{i}'
            status = hyp.get("status", "pending")
            color = HYPOTHESIS_COLORS.get(status, HYPOTHESIS_COLORS["pending"])

            # Style based on status
            if status == "verified":
                style = 'filled,rounded'
                shape = 'box'
            elif status == "refuted":
                style = 'dashed,filled'
                shape = 'diamond'
            else:
                style = 'dashed,filled'
                shape = 'ellipse'

            title = hyp.get("title", "Unknown")
            lines.append(
                f'  "{hyp_id}" [label="{_dot_escape(title[:30])}", '
                f'shape={shape}, fillcolor="{color}", style="{style}", '
                f'fontcolor="white", fontsize=9, width=0.8, height=0.4];'
            )

            # Add edges to connected nodes
            for conn in hyp.get("connects_to", []):
                if conn in node_by_id:
                    lines.append(
                        f'  "{hyp_id}" -> "{conn}" [color="{color}", '
                        f'style="dashed", penwidth=1.5, arrowhead="vee"];'
                    )

    # Add legend if requested
    if include_legend:
        lines.append("")
        lines.append("  // Legend")
        lines.append('  subgraph cluster_legend {')
        lines.append('    label="Legend";')
        lines.append('    style="rounded";')
        lines.append('    fillcolor="white";')
        lines.append('    color="#94a3b8";')
        lines.append('    node [shape=box, style=filled, fontsize=8, width=0.3, height=0.2];')
        lines.append('    edge [style=invis];')
        lines.append("")
        lines.append(f'    "legend_root" [label="Root", fillcolor="{ROOT_COLORS["node_root_neural_network"]}", shape=circle, width=0.3];')
        lines.append(f'    "legend_branch" [label="Branch", fillcolor="{BRANCH_COLORS["nn"][0]}", shape=circle, width=0.25];')
        lines.append(f'    "legend_leaf" [label="Leaf", fillcolor="{LEAF_COLORS["nn"][0]}", shape=circle, width=0.15];')
        lines.append('    "legend_hyp" [label="Hypothesis", fillcolor="' + HYPOTHESIS_COLORS["pending"] + '", style=dashed, shape=ellipse];')
        lines.append('    "legend_verified" [label="Verified", fillcolor="' + HYPOTHESIS_COLORS["verified"] + '", shape=box, style=rounded];')
        lines.append('    "legend_refuted" [label="Refuted", fillcolor="' + HYPOTHESIS_COLORS["refuted"] + '", style=dashed, shape=diamond];')
        lines.append('    "legend_root" -> "legend_branch" -> "legend_leaf" -> "legend_hyp" -> "legend_verified" -> "legend_refuted";')
        lines.append('  }')

    lines.append("}")
    return "\n".join(lines) + "\n"


def _get_reachable(start: str, children: dict[str, list[str]]) -> set[str]:
    """Get all nodes reachable from start."""
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def _build_hierarchy(root_id: str, children: dict[str, list[str]]) -> list[list[str]]:
    """Build hierarchy levels (BFS)."""
    levels = [[root_id]]
    seen = {root_id}

    current_level = [root_id]
    while current_level:
        next_level = []
        for node_id in current_level:
            for child in children.get(node_id, []):
                if child not in seen:
                    seen.add(child)
                    next_level.append(child)
        if next_level:
            levels.append(next_level)
        current_level = next_level

    return levels


# ============================================================================
# Rendering
# ============================================================================

def render_graphviz(
    dot_path: Path,
    out_png: Path,
    out_svg: Path,
    engine: str,
    dpi: int,
) -> None:
    """Render DOT to PNG and SVG."""
    subprocess.run(
        [engine, f"-Gdpi={dpi}", "-Tpng", str(dot_path), "-o", str(out_png)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [engine, "-Tsvg", str(dot_path), "-o", str(out_svg)],
        check=True,
        capture_output=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Knowledge Graph Visualizer - hierarchical circular layout"
    )
    parser.add_argument(
        "--seed",
        default="knowledge_graph/seed_parameter_golf_kg.json",
        help="Path to the seed knowledge graph JSON.",
    )
    parser.add_argument(
        "--outbox",
        default="knowledge_graph/outbox",
        help="Path to outbox directory with hypotheses.",
    )
    parser.add_argument(
        "--out-dir",
        default="knowledge_graph/visuals",
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--basename",
        default="kg_unified",
        help="Base filename for output.",
    )
    parser.add_argument(
        "--with-hypotheses",
        action="store_true",
        help="Include hypothesis nodes from outbox.",
    )
    parser.add_argument(
        "--labels",
        choices=["none", "roots", "roots-branches", "all"],
        default="roots-branches",
        help="Which node labels to display.",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz engine (dot, neato, fdp, sfdp, twopi, circo).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Omit the legend from the graph.",
    )
    args = parser.parse_args()

    # Paths
    seed_path = Path(args.seed)
    outbox_path = Path(args.outbox)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    nodes, edges = load_seed_graph(seed_path)
    hypotheses = load_hypotheses(outbox_path) if args.with_hypotheses else []

    # Build DOT
    dot_text = build_clustered_dot(
        nodes,
        edges,
        hypotheses if hypotheses else None,
        labels=args.labels,
        title="Parameter Golf Knowledge Graph" + (" + Research" if hypotheses else ""),
        include_legend=not args.no_legend,
    )

    # Write and render
    dot_path = out_dir / f"{args.basename}.dot"
    png_path = out_dir / f"{args.basename}.png"
    svg_path = out_dir / f"{args.basename}.svg"

    dot_path.write_text(dot_text)

    try:
        render_graphviz(dot_path, png_path, svg_path, args.engine, args.dpi)
        print(f"Generated:")
        print(f"  DOT: {dot_path}")
        print(f"  PNG: {png_path}")
        print(f"  SVG: {svg_path}")
        if hypotheses:
            print(f"  Included {len(hypotheses)} hypothesis node(s)")
    except subprocess.CalledProcessError as e:
        print(f"Error rendering graph: {e}")
        print(f"DOT file saved to: {dot_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
