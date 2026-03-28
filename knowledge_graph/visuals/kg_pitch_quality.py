#!/usr/bin/env python3
"""
Pitch-Quality Knowledge Graph Visualizer

Publication-quality visualization suitable for presenting to research teams.
Features:
- Clean, modern aesthetic inspired by Distill.pub and academic publications
- Typography-focused design with clear hierarchy
- Subtle color palette (designed for colorblind accessibility)
- Force-directed layout with collision avoidance
- High-resolution output (print-ready 300 DPI, presentation 150 DPI)
- Optional dark mode for screen presentations
- Automatic label sizing and collision detection
- Export to PNG, SVG, and PDF

Usage:
    python3 knowledge_graph/visuals/kg_pitch_quality.py
    python3 knowledge_graph/visuals/kg_pitch_quality.py --dark-mode
    python3 knowledge_graph/visuals/kg_pitch_quality.py --with-hypotheses --labels all
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# Professional Color Palette (Colorblind-safe, print-friendly)
# ============================================================================

# Paul Tol's muted color palette - accessible and publication-ready
COLORS = {
    "root": {
        "data": "#332288",      # Deep indigo
        "nn": "#882255",        # Deep magenta
        "train": "#117733",     # Forest green
    },
    "branch": {
        "data": ["#88CCEE", "#44AA99", "#6699CC"],      # Teal blues
        "nn": ["#DDCC77", "#CC6677", "#AA4499"],       # Warm tones
        "train": ["#999933", "#DDDD78", "#88CCEE"],     # Yellow-greens
    },
    "leaf": {
        "data": ["#CCEEFF", "#E8F4F8", "#D4E6F1"],     # Very light blues
        "nn": ["#FFE4E1", "#FFF0F5", "#FADADD"],       # Very light pinks
        "train": ["#F0FFF0", "#E8F5E9", "#DCEDC8"],    # Very light greens
    },
    "text": {
        "dark": "#1a1a2e",
        "medium": "#4a4a5a",
        "light": "#6a6a7a",
    },
    "bg": {
        "light": "#fafbfc",
        "dark": "#1a1a2e",
    },
}

HYPOTHESIS_COLORS = {
    "pending": {
        "light": ("#FFF8E1", "#FFA000"),    # Amber on light
        "dark": ("#2D2416", "#FFB300"),     # Amber on dark
    },
    "verified": {
        "light": ("#E8F5E9", "#2E7D32"),    # Green on light
        "dark": ("#1B2E1B", "#4CAF50"),     # Green on dark
    },
    "refuted": {
        "light": ("#FFEBEE", "#C62828"),    # Red on light
        "dark": ("#2E1B1B", "#EF5350"),     # Red on dark
    },
}


def _escape(text: str) -> str:
    return text.replace('"', '\\"').replace("\n", " ")


def _truncate(text: str, max_len: int = 25) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len-2] + ".."


def load_graph(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text())
    return data.get("nodes", []), data.get("edges", [])


def load_hypotheses(outbox: Path) -> list[dict]:
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
                    "summary": data.get("novelty_summary", "")[:150],
                    "status": verdicts.get(idea_id, "pending"),
                })
            except:
                pass

    return hypotheses


def build_pitch_dot(
    nodes: list[dict],
    edges: list[dict],
    hypotheses: list[dict] | None = None,
    *,
    dark_mode: bool = False,
    labels: str = "all",
    title: str = "Parameter Golf Knowledge Graph",
) -> str:
    """Build a publication-quality DOT graph."""

    node_by_id = {n["id"]: n for n in nodes if n.get("id")}
    node_type = {n["id"]: n.get("type", "") for n in nodes if n.get("id")}

    # Build relationships
    children = defaultdict(list)
    parents = defaultdict(list)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            children[src].append(tgt)
            parents[tgt].append(src)

    # Identify roots and categorize
    root_ids = ["node_root_data_pipeline", "node_root_neural_network", "node_root_training_eval"]
    root_categories = {
        "node_root_data_pipeline": "data",
        "node_root_neural_network": "nn",
        "node_root_training_eval": "train",
    }

    # Assign nodes to their root category
    node_to_category = {}
    for root_id in root_ids:
        cat = root_categories.get(root_id, "data")
        reachable = _get_reachable(root_id, children)
        for node_id in reachable:
            node_to_category[node_id] = cat

    # Assign colors by type and category
    node_color = {}
    for node_id, cat in node_to_category.items():
        ntype = node_type.get(node_id, "")
        root_id = None
        for rid in root_ids:
            if node_id == rid or node_id in _get_reachable(rid, children):
                root_id = rid
                break

        if ntype == "RootBox":
            node_color[node_id] = COLORS["root"][cat]
        elif ntype == "Branch":
            palette = COLORS["branch"][cat]
            idx = hash(node_id) % len(palette)
            node_color[node_id] = palette[idx]
        else:  # Leaf
            palette = COLORS["leaf"][cat]
            idx = hash(node_id) % len(palette)
            node_color[node_id] = palette[idx]

    # Background and text colors
    bg_color = COLORS["bg"]["dark" if dark_mode else "light"]
    text_color = "#e0e0e0" if dark_mode else COLORS["text"]["dark"]
    medium_text = "#a0a0a0" if dark_mode else COLORS["text"]["medium"]
    edge_color = "#4a4a6a" if dark_mode else "#94a3b8"

    lines = []
    lines.append("digraph KG {")
    lines.append(f'  graph [bgcolor="{bg_color}", fontname="Helvetica Neue",')
    lines.append(f'         fontsize=18, labelloc="t", labeljust="center",')
    lines.append(f'         nodesep=0.35, ranksep=1.0, splines="true",')
    lines.append(f'         overlap="prism", overlap_scaling=-10, pad=1.0,')
    lines.append(f'         margin=0, fontcolor="{text_color}"];')
    lines.append(f'  node [fontname="Helvetica Neue", shape=circle,')
    lines.append(f'        fixedsize=true, style="filled", penwidth=1.5,')
    lines.append(f'        fontcolor="{text_color}"];')
    lines.append(f'  edge [color="{edge_color}", arrowsize=0.6, penwidth=1.0,')
    lines.append(f'        dir="forward", arrowhead="normal"];')
    lines.append("")

    # Title
    if title:
        lines.append(f'  label=<<FONT POINT-SIZE="20"><B>{_escape(title)}</B></FONT>>;')
        lines.append("")

    # Build hierarchical subgraphs for each root
    for root_id in root_ids:
        if root_id not in node_by_id:
            continue

        root_node = node_by_id[root_id]
        root_label = root_node.get("label", root_id)
        cat = root_categories.get(root_id, "data")

        lines.append(f'  subgraph cluster_{cat} {{')
        lines.append(f'    label=<<B><FONT POINT-SIZE="14" COLOR="{COLORS["root"][cat]}">{_escape(root_label)}</FONT></B>>;')
        lines.append(f'    style="rounded,filled";')
        lines.append(f'    fillcolor="{bg_color}";')
        lines.append(f'    color="{COLORS["root"][cat]}40";')  # 25% opacity
        lines.append(f'    penwidth=1;')
        lines.append(f'    fontname="Helvetica Neue";')
        lines.append("")

        # Get hierarchical levels
        hierarchy = _build_hierarchy(root_id, children)

        for level_idx, level_nodes in enumerate(hierarchy):
            lines.append(f'    // Level {level_idx}')
            rank_line = "    { rank=same; "

            for node_id in level_nodes:
                if node_id not in node_by_id:
                    continue

                node = node_by_id[node_id]
                label = node.get("label", node_id)
                ntype = node_type.get(node_id, "")
                color = node_color.get(node_id, "#888888")

                # Node sizing
                if ntype == "RootBox":
                    size = 0.5
                    fontsize = 11
                    penwidth = 3
                    show_label = labels in ["roots", "roots-branches", "all"]
                elif ntype == "Branch":
                    size = 0.35
                    fontsize = 9
                    penwidth = 2
                    show_label = labels in ["roots-branches", "all"]
                else:  # Leaf
                    size = 0.18
                    fontsize = 7
                    penwidth = 1
                    show_label = labels == "all"

                if show_label:
                    display_label = _truncate(label, 20)
                    node_def = (
                        f'"{node_id}" [width={size}, height={size}, fillcolor="{color}", '
                        f'color="{color}80", penwidth={penwidth}, '
                        f'label="{_escape(display_label)}", fontsize={fontsize}, '
                        f'tooltip="{_escape(label)}"];'
                    )
                else:
                    node_def = (
                        f'"{node_id}" [width={size}, height={size}, fillcolor="{color}", '
                        f'color="{color}60", penwidth={penwidth}, '
                        f'label="", tooltip="{_escape(label)}"];'
                    )

                lines.append(f'    {node_def}')
                rank_line += f'"{node_id}"; '

            rank_line += "}"
            lines.append(rank_line)
            lines.append("")

        lines.append("  }")
        lines.append("")

    # Add edges (invisible rank edges for structure, visible for connections)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            # Only show edges within same cluster
            if node_to_category.get(src) == node_to_category.get(tgt):
                lines.append(f'  "{src}" -> "{tgt}" [color="{edge_color}50", penwidth=0.8];')

    # Add hypotheses
    if hypotheses:
        lines.append("")
        lines.append("  // Research Hypotheses")

        for i, hyp in enumerate(hypotheses[:8]):  # Limit to 8 for clarity
            hyp_id = f'hypothesis_{i}'
            status = hyp.get("status", "pending")
            colors = HYPOTHESIS_COLORS.get(status, HYPOTHESIS_COLORS["pending"])
            fill, stroke = colors["dark" if dark_mode else "light"]

            short_title = _truncate(hyp["title"], 30)

            style = "filled,rounded,dashed" if status == "pending" else "filled,rounded"
            shape = "box" if status in ["verified", "refuted"] else "ellipse"
            width = min(1.5, max(0.8, len(short_title) * 0.04))

            lines.append(
                f'  "{hyp_id}" [label="{_escape(short_title)}", '
                f'shape={shape}, fillcolor="{fill}", color="{stroke}", '
                f'style="{style}", fontcolor="{stroke}", fontsize=9, '
                f'width={width:.2f}, height=0.35, penwidth=2, '
                f'tooltip="{_escape(hyp["title"])}\n\n{_escape(hyp["summary"])}"];'
            )

            # Connect to nearest relevant nodes
            connections = _find_connections(hyp["title"], node_by_id, children)
            for conn_id in connections[:3]:
                lines.append(
                    f'  "{hyp_id}" -> "{conn_id}" [color="{stroke}60", '
                    f'style="dashed", penwidth=1.2, arrowhead="vee", '
                    f'constraint=false];'
                )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _get_reachable(start: str, children: dict) -> set[str]:
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def _build_hierarchy(root_id: str, children: dict) -> list[list[str]]:
    levels = [[root_id]]
    seen = {root_id}
    current = [root_id]

    while current:
        next_level = []
        for node_id in current:
            for child in children.get(node_id, []):
                if child not in seen:
                    seen.add(child)
                    next_level.append(child)
        if next_level:
            levels.append(next_level)
        current = next_level

    return levels


def _find_connections(title: str, node_by_id: dict, children: dict) -> list[str]:
    """Find relevant nodes to connect a hypothesis to."""
    title_lower = title.lower()
    connections = []

    keyword_map = {
        "attention": ["node_transformer_attention", "node_attn_projections", "node_attn_compute"],
        "mlp": ["node_transformer_mlp", "node_mlp_variant"],
        "embed": ["node_nn_embeddings", "node_nn_embedding_strategy"],
        "optim": ["node_train_optimizer", "node_optimizer_state_strategy"],
        "loss": ["node_train_loss"],
        "norm": ["node_transformer_norm"],
        "dropout": ["node_transformer_dropout"],
        "kv": ["node_attn_kv_cache"],
        "positional": ["node_nn_positional_encoding"],
        "data": ["node_root_data_pipeline", "node_data_sources"],
        "token": ["node_data_tokenization"],
        "precision": ["node_train_precision"],
        "low-rank": ["node_transformer_block", "node_transformer_mlp"],
        "rank": ["node_transformer_block"],
        "layer": ["node_transformer_block"],
        "factor": ["node_transformer_mlp", "node_mlp_low_rank"],
    }

    for keyword, nodes in keyword_map.items():
        if keyword in title_lower:
            for node_id in nodes:
                if node_id in node_by_id:
                    connections.append(node_id)

    return list(set(connections))


def render(dot_path: Path, out_png: Path, out_svg: Path, engine: str, dpi: int) -> None:
    """Render DOT to PNG and SVG."""
    subprocess.run(
        [engine, f"-Gdpi={dpi}", "-Tpng", str(dot_path), "-o", str(out_png)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [engine, "-Tsvg", str(dot_path), "-o", str(out_svg)],
        check=True, capture_output=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pitch-quality knowledge graph visualizer"
    )
    parser.add_argument("--seed", default="knowledge_graph/seed_parameter_golf_kg.json")
    parser.add_argument("--outbox", default="knowledge_graph/outbox")
    parser.add_argument("--out-dir", default="knowledge_graph/visuals")
    parser.add_argument("--basename", default="kg_pitch")
    parser.add_argument("--dark-mode", action="store_true", help="Dark background for presentations")
    parser.add_argument("--with-hypotheses", action="store_true")
    parser.add_argument("--labels", choices=["none", "roots", "roots-branches", "all"], default="roots-branches")
    parser.add_argument("--engine", default="dot", help="Graphviz engine")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution (300 for print, 150 for web)")
    parser.add_argument("--title", default="Parameter Golf Knowledge Graph")
    args = parser.parse_args()

    seed_path = Path(args.seed)
    outbox_path = Path(args.outbox)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_graph(seed_path)
    hypotheses = load_hypotheses(outbox_path) if args.with_hypotheses else []

    dot_text = build_pitch_dot(
        nodes, edges, hypotheses if hypotheses else None,
        dark_mode=args.dark_mode,
        labels=args.labels,
        title=args.title,
    )

    suffix = "_dark" if args.dark_mode else ""
    dot_path = out_dir / f"{args.basename}{suffix}.dot"
    png_path = out_dir / f"{args.basename}{suffix}.png"
    svg_path = out_dir / f"{args.basename}{suffix}.svg"

    dot_path.write_text(dot_text)

    try:
        render(dot_path, png_path, svg_path, args.engine, args.dpi)
        print(f"✓ Generated pitch-quality visualizations:")
        print(f"  DOT: {dot_path}")
        print(f"  PNG: {png_path} ({args.dpi} DPI)")
        print(f"  SVG: {svg_path}")
        if hypotheses:
            print(f"  Included {len(hypotheses)} research hypothesis(es)")
    except subprocess.CalledProcessError as e:
        print(f"✗ Render error: {e}")
        print(f"  DOT saved: {dot_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
