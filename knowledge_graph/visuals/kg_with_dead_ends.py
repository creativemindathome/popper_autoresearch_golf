#!/usr/bin/env python3
"""
Knowledge Graph with Dead Ends Visualization

Shows the knowledge graph with refuted hypotheses marked as red X nodes.
This visualizes what has been tried and failed.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict


def load_seed_graph(path: Path):
    with open(path) as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])


def load_hypothesis_registry(path: Path):
    with open(path) as f:
        return json.load(f)


def build_dot_with_dead_ends(
    nodes: list,
    edges: list,
    registry: dict,
    title: str = "Knowledge Graph: Tried, Failed, and Untested"
) -> str:
    """Build graph showing dead ends."""

    node_map = {n["id"]: n for n in nodes if n.get("id")}

    lines = []
    lines.append("digraph KGWithDeadEnds {")
    lines.append('  graph [bgcolor="#fafafa", fontname="Helvetica",')
    lines.append('         fontsize=18, labelloc="t", labeljust="center",')
    lines.append('         nodesep=0.5, ranksep=1.2, pad=1.5];')
    lines.append('  node [fontname="Helvetica", shape=box, style="rounded,filled", penwidth=2];')
    lines.append('  edge [color="#94a3b8", arrowsize=0.7, penwidth=1.5];')
    lines.append("")
    lines.append(f'  label="{title}";')
    lines.append("")

    # Color scheme
    colors = {
        "root": {"data": "#2563eb", "nn": "#dc2626", "train": "#16a34a"},
        "branch": {"data": "#60a5fa", "nn": "#f87171", "train": "#4ade80"},
        "leaf": {"data": "#bfdbfe", "nn": "#fecaca", "train": "#bbf7d0"},
    }

    refuted_color = "#7f1d1d"  # Dark red for dead ends
    untested_color = "#f3f4f6"  # Light gray for untested

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
        ("node_root_training_eval", "Training & Evaluation", "train"),
    ]

    # Count refuted hypotheses per pillar based on keywords
    refuted_counts = {"data": 0, "nn": 0, "train": 0}
    for hyp in registry.get("hypotheses", []):
        if hyp.get("verdict") == "REFUTED":
            category = hyp.get("category", "")
            if category in ["attention", "architecture"]:
                refuted_counts["nn"] += 1
            elif category in ["mlp"]:
                refuted_counts["nn"] += 1
            elif category in ["data", "tokenization"]:
                refuted_counts["data"] += 1
            elif category in ["optimizer", "training", "precision"]:
                refuted_counts["train"] += 1
            else:
                refuted_counts["nn"] += 1  # Default to NN

    # Draw pillars with refuted counts
    for root_id, root_label, cat in pillars:
        if root_id not in node_map:
            continue

        c = colors["root"][cat]
        refuted = refuted_counts.get(cat, 0)

        # Root with refuted count badge
        if refuted > 0:
            label = f"{root_label}\\n({refuted} refuted)"
            fillcolor = f"{c}80"  # 50% opacity
        else:
            label = root_label
            fillcolor = c

        lines.append(
            f'  "{root_id}" [label="{label}", fillcolor="{fillcolor}", '
            f'fontcolor="white", width=2.5, height=0.8, fontsize=12, penwidth=3];'
        )

        # Add branches and leaves
        for branch_id in children.get(root_id, [])[:6]:
            if branch_id not in node_map:
                continue

            branch = node_map[branch_id]
            branch_label = branch.get("label", branch_id)

            lines.append(
                f'  "{branch_id}" [label="{branch_label}", fillcolor="{colors["branch"][cat]}", '
                f'fontcolor="white", width=2.0, height=0.5, fontsize=10];'
            )
            lines.append(f'  "{root_id}" -> "{branch_id}";')

            for leaf_id in children.get(branch_id, [])[:3]:
                if leaf_id not in node_map:
                    continue

                leaf = node_map[leaf_id]
                leaf_label = leaf.get("label", leaf_id)

                lines.append(
                    f'  "{leaf_id}" [label="{leaf_label}", fillcolor="{colors["leaf"][cat]}", '
                    f'fontcolor="#1f2937", width=1.8, height=0.35, fontsize=8];'
                )
                lines.append(f'  "{branch_id}" -> "{leaf_id}";')

    # Add dead ends (refuted hypotheses) as X-shaped nodes
    lines.append("")
    lines.append("  // Refuted Hypotheses (Dead Ends)")

    y_offset = 0
    for hyp in registry.get("hypotheses", [])[:10]:
        if hyp.get("verdict") != "REFUTED":
            continue

        hyp_id = f"dead_{hyp['id']}"
        short_name = hyp['id'].replace("-", " ")[:20]

        lines.append(
            f'  "{hyp_id}" [label="✗ {short_name}", fillcolor="{refuted_color}", '
            f'fontcolor="white", width=1.5, height=0.4, fontsize=9, '
            f'shape="box", style="rounded,filled,dashed", penwidth=2];'
        )

        # Connect to relevant root based on category
        cat = hyp.get("category", "")
        if cat in ["attention", "mlp", "architecture"]:
            lines.append(f'  "{hyp_id}" -> "node_root_neural_network" [color="{refuted_color}60", style="dashed"];')
        elif cat in ["data", "tokenization"]:
            lines.append(f'  "{hyp_id}" -> "node_root_data_pipeline" [color="{refuted_color}60", style="dashed"];')
        else:
            lines.append(f'  "{hyp_id}" -> "node_root_neural_network" [color="{refuted_color}60", style="dashed"];')

        y_offset += 1

    # Add legend
    lines.append("")
    lines.append("  // Legend")
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    style="rounded";')
    lines.append('    fillcolor="white";')
    lines.append('    color="#9ca3af";')
    lines.append('    node [shape=box, width=0.3, height=0.2, fontsize=8];')
    lines.append('    edge [style=invis];')
    lines.append("")
    lines.append(f'    "leg_root" [label="Root", fillcolor="{colors["root"]["nn"]}", fontcolor="white"];')
    lines.append(f'    "leg_branch" [label="Branch", fillcolor="{colors["branch"]["nn"]}", fontcolor="white"];')
    lines.append(f'    "leg_leaf" [label="Leaf", fillcolor="{colors["leaf"]["nn"]}", fontcolor="#1f2937"];')
    lines.append(f'    "leg_dead" [label="✗ Dead End", fillcolor="{refuted_color}", fontcolor="white", style="dashed"];')
    lines.append('    "leg_root" -> "leg_branch" -> "leg_leaf" -> "leg_dead";')
    lines.append('  }')

    lines.append("}")
    return "\n".join(lines)


def main():
    seed_path = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/knowledge_graph/seed_parameter_golf_kg.json")
    registry_path = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/knowledge_graph/history/hypothesis_registry.json")
    out_dir = Path("/Users/curiousmind/Desktop/null_fellow_hackathon/knowledge_graph/visuals")

    nodes, edges = load_seed_graph(seed_path)
    registry = load_hypothesis_registry(registry_path)

    dot_text = build_dot_with_dead_ends(nodes, edges, registry)

    dot_path = out_dir / "kg_with_dead_ends.dot"
    png_path = out_dir / "kg_with_dead_ends.png"
    svg_path = out_dir / "kg_with_dead_ends.svg"

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
        print(f"✓ Generated knowledge graph with dead ends:")
        print(f"  {png_path}")
        print(f"  {svg_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"DOT saved: {dot_path}")


if __name__ == "__main__":
    main()
