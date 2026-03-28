#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_params(n_params: int) -> str:
    return f"{n_params:,}"


def _format_mb(mb: float) -> str:
    if mb == 0:
        return "0"
    if mb < 10:
        return f"{mb:.2f}".rstrip("0").rstrip(".")
    return f"{mb:.1f}".rstrip("0").rstrip(".")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "graph"


def _node_style(node_type: str) -> dict[str, str]:
    if node_type == "RootBox":
        return {
            "shape": "box",
            "style": "rounded,filled,bold",
            "fillcolor": "#fff9db",
            "color": "#e67700",
            "fontcolor": "#212529",
            "penwidth": "2",
        }
    if node_type == "Branch":
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#e7f5ff",
            "color": "#1971c2",
            "fontcolor": "#212529",
            "penwidth": "1.4",
        }
    if node_type == "Leaf":
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#f8f9fa",
            "color": "#495057",
            "fontcolor": "#212529",
            "penwidth": "1",
        }
    return {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#ffffff",
        "color": "#495057",
        "fontcolor": "#212529",
    }


def _dot_attr_list(attrs: dict[str, str | None]) -> str:
    parts: list[str] = []
    for key, value in attrs.items():
        if value is None:
            continue
        if key == "label" and value.startswith("<<") and value.endswith(">>"):
            parts.append(f"{key}={value}")
            continue
        parts.append(f'{key}="{value}"')
    return ", ".join(parts)


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


def _node_label_html(node: dict) -> str:
    node_label = _html_escape(str(node.get("label", "")))
    params = int(node.get("parameter_estimate", 0) or 0)
    mem_mb = float(node.get("memory_estimate_mb", 0.0) or 0.0)

    metrics: list[str] = []
    if params > 0:
        metrics.append(f"P:{_format_params(params)}")
    if mem_mb > 0:
        metrics.append(f"Mem:{_format_mb(mem_mb)}MB")

    if not metrics:
        return f"<<B>{node_label}</B>>"

    return (
        f'<<B>{node_label}</B><BR/><FONT POINT-SIZE="9">'
        f'{" | ".join(metrics)}</FONT>>'
    )


def build_dot(
    nodes: list[dict],
    edges: list[dict],
    *,
    root_order: list[str] | None = None,
    title: str | None = None,
) -> str:

    lines: list[str] = []
    lines.append("digraph KnowledgeGraph {")
    lines.append(
        '  graph [rankdir=LR, bgcolor="#ffffff", fontname="Helvetica", fontsize=10, '
        'labelloc="t", labeljust="l", nodesep="0.15", ranksep="0.25", splines="ortho", '
        'overlap="false", pad="0.1"];'
    )
    lines.append('  node [fontname="Helvetica", fontsize=10, margin="0.10,0.07"];')
    lines.append('  edge [color="#adb5bd", arrowsize="0.6", penwidth="1"];')
    lines.append("")

    if title:
        lines.append(f'  label="{_html_escape(title)}";')
        lines.append("")

    if root_order and len(root_order) > 1:
        lines.append("  { rank = same; " + "; ".join(f'"{r}"' for r in root_order) + "; }")
        for left, right in zip(root_order, root_order[1:]):
            lines.append(f'  "{left}" -> "{right}" [style="invis"];')
        lines.append("")

    for node in nodes:
        node_id = node["id"]
        tooltip = _html_escape(str(node.get("mechanism_description", "")))

        attrs = _node_style(str(node.get("type", "")))
        attrs.update(
            {
                "label": _node_label_html(node),
                "tooltip": tooltip,
            }
        )
        lines.append(f'  "{node_id}" [{_dot_attr_list(attrs)}];')

    lines.append("")

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relationship = str(edge.get("relationship", ""))
        edge_attrs: dict[str, str | None] = {}
        if relationship and relationship != "sub_component_of":
            edge_attrs["label"] = _html_escape(relationship)
            edge_attrs["fontsize"] = "8"
            edge_attrs["fontcolor"] = "#495057"
        lines.append(f'  "{source}" -> "{target}" [{_dot_attr_list(edge_attrs)}];')

    lines.append("}")
    return "\n".join(lines) + "\n"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a knowledge graph JSON ({nodes:[...], edges:[...]}) to DOT + PNG/SVG using Graphviz."
        )
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
        default="seed_parameter_golf_kg",
        help="Base filename (without extension).",
    )
    parser.add_argument(
        "--mode",
        choices=["overview", "split-roots", "both"],
        default="both",
        help="Render just the overview graph, per-root graphs, or both.",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz layout engine to use (e.g. dot, sfdp, neato, twopi).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PNG output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = _load_nodes_edges(input_path)
    root_ids = _find_root_nodes(nodes, edges)

    generated: list[Path] = []

    if args.mode in {"overview", "both"}:
        dot_text = build_dot(
            nodes,
            edges,
            root_order=root_ids,
            title="Knowledge Graph",
        )

        dot_path = out_dir / f"{args.basename}.dot"
        png_path = out_dir / f"{args.basename}.png"
        svg_path = out_dir / f"{args.basename}.svg"

        dot_path.write_text(dot_text)
        _render_graphviz(
            dot_path,
            out_png=png_path,
            out_svg=svg_path,
            engine=str(args.engine),
            dpi=int(args.dpi),
        )
        generated.extend([dot_path, png_path, svg_path])

    if args.mode in {"split-roots", "both"} and root_ids:
        node_by_id = {str(n.get("id")): n for n in nodes if n.get("id")}
        for root_id in root_ids:
            root_node = node_by_id.get(root_id, {})
            root_label = str(root_node.get("label") or root_id)
            suffix = _slugify(root_label)

            allowed = _reachable_nodes(root_id, edges)
            sub_nodes = [n for n in nodes if str(n.get("id")) in allowed]
            sub_edges = [
                e
                for e in edges
                if str(e.get("source")) in allowed and str(e.get("target")) in allowed
            ]

            dot_text = build_dot(
                sub_nodes,
                sub_edges,
                root_order=[root_id],
                title=root_label,
            )

            dot_path = out_dir / f"{args.basename}_{suffix}.dot"
            png_path = out_dir / f"{args.basename}_{suffix}.png"
            svg_path = out_dir / f"{args.basename}_{suffix}.svg"

            dot_path.write_text(dot_text)
            _render_graphviz(
                dot_path,
                out_png=png_path,
                out_svg=svg_path,
                engine=str(args.engine),
                dpi=int(args.dpi),
            )
            generated.extend([dot_path, png_path, svg_path])

    for path in generated:
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
