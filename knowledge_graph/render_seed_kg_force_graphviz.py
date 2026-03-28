#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dot_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _hsl_to_hex(hue_deg: int, saturation: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _stable_hue(key: str) -> int:
    hue = 0
    for ch in key:
        hue = (hue * 31 + ord(ch)) % 360
    return int(hue)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "graph"


def _load_nodes_edges(input_path: Path) -> tuple[list[dict], list[dict]]:
    graph = json.loads(input_path.read_text())
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Expected {nodes:[...], edges:[...]} JSON.")
    return nodes, edges


def _find_root_nodes(nodes: list[dict], edges: list[dict]) -> list[str]:
    typed_roots = [str(n.get("id")) for n in nodes if n.get("type") == "RootBox" and n.get("id")]
    if typed_roots:
        return typed_roots

    indegree: dict[str, int] = {}
    for n in nodes:
        node_id = n.get("id")
        if node_id:
            indegree[str(node_id)] = 0
    for e in edges:
        target = e.get("target")
        if target is None:
            continue
        target_id = str(target)
        if target_id in indegree:
            indegree[target_id] += 1

    return [node_id for node_id, deg in indegree.items() if deg == 0]


def _reachable_nodes(root_id: str, edges: list[dict]) -> set[str]:
    children: dict[str, list[str]] = {}
    for e in edges:
        source = e.get("source")
        target = e.get("target")
        if source is None or target is None:
            continue
        children.setdefault(str(source), []).append(str(target))

    seen: set[str] = {root_id}
    queue: list[str] = [root_id]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
    return seen


def _infer_branch_ancestor(
    node_id: str,
    *,
    parents: dict[str, list[str]],
    branch_ids: set[str],
) -> str:
    seen: set[str] = {node_id}
    current = node_id
    while True:
        if current in branch_ids:
            return current
        incoming = parents.get(current)
        if not incoming:
            return current
        current = incoming[0]
        if current in seen:
            return current
        seen.add(current)


def _assign_colors(nodes: list[dict], edges: list[dict]) -> dict[str, str]:
    parents: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        source = e.get("source")
        target = e.get("target")
        if source is None or target is None:
            continue
        parents[str(target)].append(str(source))

    node_type: dict[str, str] = {str(n.get("id")): str(n.get("type") or "") for n in nodes if n.get("id")}
    branch_ids = {node_id for node_id, t in node_type.items() if t == "Branch"}

    branch_color: dict[str, str] = {}
    for branch_id in sorted(branch_ids):
        branch_color[branch_id] = _hsl_to_hex(_stable_hue(branch_id), 0.75, 0.50)

    root_color = {
        "node_root_data_pipeline": "#3b82f6",
        "node_root_neural_network": "#ef4444",
        "node_root_training_eval": "#22c55e",
    }

    node_color: dict[str, str] = {}
    for node_id, t in node_type.items():
        if t == "RootBox":
            node_color[node_id] = root_color.get(node_id, "#111827")
            continue

        branch_ancestor = _infer_branch_ancestor(node_id, parents=parents, branch_ids=branch_ids)
        node_color[node_id] = branch_color.get(branch_ancestor, "#9ca3af")

    return node_color


def _render_graphviz(
    dot_path: Path,
    *,
    out_png: Path,
    out_svg: Path,
    engine: str,
    dpi: int,
) -> None:
    subprocess.run(
        [engine, f"-Gdpi={dpi}", "-Tpng", str(dot_path), "-o", str(out_png)],
        check=True,
    )
    subprocess.run(
        [engine, "-Tsvg", str(dot_path), "-o", str(out_svg)],
        check=True,
    )


def build_force_dot(
    nodes: list[dict],
    edges: list[dict],
    *,
    labels: str,
    title: str | None,
) -> str:
    node_type: dict[str, str] = {str(n.get("id")): str(n.get("type") or "") for n in nodes if n.get("id")}
    node_label: dict[str, str] = {
        str(n.get("id")): str(n.get("label") or "") for n in nodes if n.get("id")
    }
    node_color = _assign_colors(nodes, edges)

    show_roots = labels in {"roots", "roots-branches", "all"}
    show_branches = labels in {"roots-branches", "all"}
    show_all = labels == "all"

    lines: list[str] = []
    lines.append("graph KnowledgeGraph {")
    lines.append(
        '  graph [layout=sfdp, overlap=prism, bgcolor="white", outputorder="edgesfirst", '
        'splines=true, pad=0.4, K=1.35, repulsiveforce=1.2];'
    )
    lines.append('  node [shape=circle, fixedsize=true, label="", penwidth=0];')
    lines.append('  edge [color="#00000022", penwidth=1.0];')
    lines.append("")

    if title:
        lines.append(f'  label="{_html_escape(title)}";')
        lines.append('  labelloc="t";')
        lines.append('  labeljust="l";')
        lines.append("")

    for node_id, t in node_type.items():
        label = node_label.get(node_id, "")
        color = node_color.get(node_id, "#9ca3af")

        size = 0.08
        if t == "Leaf":
            size = 0.07
        elif t == "Branch":
            size = 0.12
        elif t == "RootBox":
            size = 0.18

        attrs: list[str] = [
            'style="filled"',
            f'color="{color}"',
            f'fillcolor="{color}"',
            f"width={size}",
            f"height={size}",
            f'tooltip="{_dot_escape(label)}"',
        ]

        should_label = (t == "RootBox" and show_roots) or (t == "Branch" and show_branches) or show_all
        if should_label and label:
            fontsize = 12 if t == "RootBox" else 10 if t == "Branch" else 8
            attrs.extend(
                [
                    f'xlabel="{_dot_escape(label)}"',
                    'fontname="Helvetica"',
                    f"fontsize={fontsize}",
                    'fontcolor="#111827"',
                ]
            )

        lines.append(f'  "{node_id}" [' + ", ".join(attrs) + "];")

    lines.append("")

    for e in edges:
        source = str(e.get("source"))
        target = str(e.get("target"))
        if not source or not target:
            continue

        c = node_color.get(source, "#9ca3af")
        source_t = node_type.get(source, "")

        length = 1.0 if source_t == "RootBox" else 0.9 if source_t == "Branch" else 0.8
        weight = 4 if source_t == "RootBox" else 2 if source_t == "Branch" else 1
        penwidth = 1.4 if source_t == "RootBox" else 1.0

        lines.append(
            f'  "{source}" -- "{target}" '
            f'[color="{c}66", len={length}, weight={weight}, penwidth={penwidth}];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the seed Parameter Golf knowledge graph to a Gephi-like force-directed visualization."
    )
    parser.add_argument(
        "--input",
        default="knowledge_graph/seed_parameter_golf_kg.json",
        help="Path to the Nodes+Edges JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default="knowledge_graph/visuals",
        help="Output directory for .dot/.png/.svg.",
    )
    parser.add_argument(
        "--basename",
        default="seed_parameter_golf_kg_force",
        help="Base filename (without extension).",
    )
    parser.add_argument(
        "--mode",
        choices=["overview", "split-roots", "both"],
        default="overview",
        help="Render just the overview graph, per-root graphs, or both.",
    )
    parser.add_argument(
        "--labels",
        choices=["none", "roots", "roots-branches", "all"],
        default="none",
        help="Which labels to render (SVG tooltips always contain full labels).",
    )
    parser.add_argument(
        "--engine",
        default="sfdp",
        help="Graphviz layout engine to use (recommend: sfdp).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI for PNG output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = _load_nodes_edges(input_path)
    root_ids = _find_root_nodes(nodes, edges)
    node_by_id = {str(n.get("id")): n for n in nodes if n.get("id")}

    generated: list[Path] = []

    def render_one(dot_text: str, out_base: str, *, title: str | None) -> None:
        dot_path = out_dir / f"{out_base}.dot"
        png_path = out_dir / f"{out_base}.png"
        svg_path = out_dir / f"{out_base}.svg"

        dot_path.write_text(dot_text)
        _render_graphviz(
            dot_path,
            out_png=png_path,
            out_svg=svg_path,
            engine=str(args.engine),
            dpi=int(args.dpi),
        )
        generated.extend([dot_path, png_path, svg_path])

    if args.mode in {"overview", "both"}:
        dot_text = build_force_dot(nodes, edges, labels=str(args.labels), title=None)
        render_one(dot_text, str(args.basename), title=None)

    if args.mode in {"split-roots", "both"} and root_ids:
        for root_id in root_ids:
            root_label = str((node_by_id.get(root_id) or {}).get("label") or root_id)
            suffix = _slugify(root_label)
            allowed = _reachable_nodes(root_id, edges)
            sub_nodes = [n for n in nodes if str(n.get("id")) in allowed]
            sub_edges = [
                e
                for e in edges
                if str(e.get("source")) in allowed and str(e.get("target")) in allowed
            ]

            dot_text = build_force_dot(sub_nodes, sub_edges, labels=str(args.labels), title=root_label)
            render_one(dot_text, f"{args.basename}_{suffix}", title=root_label)

    for path in generated:
        print(str(path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
