#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
        return f"{mb:.2f}"
    return f"{mb:.1f}"


def _node_style(node_type: str) -> dict[str, str]:
    if node_type == "RootBox":
        return {
            "shape": "box",
            "style": "rounded,filled,bold",
            "fillcolor": "#fff3bf",
            "color": "#b08900",
            "fontcolor": "#2b2b2b",
            "penwidth": "2",
        }
    if node_type == "Branch":
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#e7f5ff",
            "color": "#1c7ed6",
            "fontcolor": "#2b2b2b",
            "penwidth": "1.2",
        }
    if node_type == "Leaf":
        return {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#f8f9fa",
            "color": "#495057",
            "fontcolor": "#2b2b2b",
            "penwidth": "1",
        }
    return {
        "shape": "box",
        "style": "filled",
        "fillcolor": "#ffffff",
        "color": "#495057",
        "fontcolor": "#2b2b2b",
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


def build_dot(input_path: Path) -> str:
    graph = json.loads(input_path.read_text())
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    lines: list[str] = []
    lines.append("digraph KnowledgeGraph {")
    lines.append(
        '  graph [rankdir=LR, bgcolor="white", fontname="Helvetica", fontsize=10, '
        'labelloc="t", nodesep="0.25", ranksep="0.50", splines="spline", overlap="false"];'
    )
    lines.append('  node [fontname="Helvetica", fontsize=10, margin="0.07,0.05"];')
    lines.append('  edge [color="#868e96", arrowsize="0.6"];')
    lines.append("")

    for node in nodes:
        node_id = node["id"]
        node_label = _html_escape(str(node["label"]))
        params = int(node.get("parameter_estimate", 0))
        mem_mb = float(node.get("memory_estimate_mb", 0.0))
        tooltip = _html_escape(str(node.get("mechanism_description", "")))

        label = (
            f'<<B>{node_label}</B><BR/><FONT POINT-SIZE="9">'
            f'P:{_format_params(params)} | Mem:{_format_mb(mem_mb)}MB</FONT>>'
        )
        attrs = _node_style(str(node.get("type", "")))
        attrs.update(
            {
                "label": label,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the seed Parameter Golf knowledge graph (JSON) to DOT + PNG/SVG using Graphviz."
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
        "--dpi",
        type=int,
        default=200,
        help="DPI for PNG output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_text = build_dot(input_path)

    dot_path = out_dir / f"{args.basename}.dot"
    png_path = out_dir / f"{args.basename}.png"
    svg_path = out_dir / f"{args.basename}.svg"

    dot_path.write_text(dot_text)

    subprocess.run(
        ["dot", f"-Gdpi={args.dpi}", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    subprocess.run(
        ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )

    print(str(png_path))
    print(str(svg_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

