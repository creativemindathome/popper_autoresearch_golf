#!/usr/bin/env python3
"""
Generate Evolution Movie (MP4) - Version 2

Creates an animated visualization that:
1. Starts with the FULL original knowledge graph structure
2. Shows hypotheses being added with proper connections to parent nodes
3. Maintains the original styling (RootBox=yellow, Branch=blue, Leaf=gray)
4. Animates the growth from baseline through all hypotheses

Usage:
    python3 knowledge_graph/visuals/generate_evolution_movie_v2.py
    python3 knowledge_graph/visuals/generate_evolution_movie_v2.py --experiment-dir /path/to/run
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Set
import os

from hypothesis_sources import (
    load_enriched_hypotheses_single_run,
    load_hypotheses_chronological_by_run,
    load_hypotheses_from_run,
    load_merged_hypotheses,
)


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
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


def _node_style(node_type: str) -> dict[str, str]:
    """Original styling from render_seed_kg_graphviz.py"""
    if node_type == "RootBox":
        return {
            "shape": "box",
            "style": "rounded,filled,bold",
            "fillcolor": "#fff9db",  # Yellow
            "color": "#e67700",
            "fontcolor": "#212529",
            "penwidth": "2",
        }
    if node_type == "Branch":
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#e7f5ff",  # Blue
            "color": "#1971c2",
            "fontcolor": "#212529",
            "penwidth": "1.4",
        }
    if node_type == "Leaf":
        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#f8f9fa",  # Gray
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


def _hypothesis_routing_id(hyp: Dict) -> str:
    """Text used to map a hypothesis to a seed parent (keywords). Prefer routing_id when set."""
    return str(hyp.get("routing_id") or hyp.get("id") or "")


def _node_label_html(node: dict) -> str:
    """Original label formatting from render_seed_kg_graphviz.py"""
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


def load_seed_graph(path: Path) -> tuple[list, list]:
    """Load the baseline knowledge graph."""
    with open(path) as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])


def _direct_parent_of(target_id: str, edges: list) -> str | None:
    for e in edges:
        if e.get("target") == target_id:
            return str(e.get("source"))
    return None


def _default_branch_under_root(
    root_id: str,
    node_map: dict,
    node_ids: set,
    edges: list,
) -> str:
    """Pick a concrete Branch hub under each pillar root (blue nodes in the evolution viz)."""
    hub = {
        "node_root_neural_network": "node_nn_transformer",
        "node_root_data_pipeline": "node_data_sources",
        "node_root_training_eval": "node_train_optimizer",
    }
    hid = hub.get(root_id)
    if hid and hid in node_ids:
        return hid
    for e in edges:
        if e.get("source") == root_id:
            cid = e.get("target")
            if cid in node_ids and (node_map.get(cid) or {}).get("type") == "Branch":
                return str(cid)
    return root_id


def _resolve_parent_to_branch(
    candidate_id: str,
    *,
    node_map: dict,
    edges: list,
    node_ids: set,
) -> str:
    """
    Attach hypotheses to a Branch (blue) node: walk up from Leaf to parent Branch;
    replace RootBox (yellow) with the pillar's default Branch hub.
    """
    if candidate_id not in node_ids:
        return candidate_id
    nid: str | None = candidate_id
    seen: set[str] = set()
    while nid and nid not in seen:
        seen.add(nid)
        meta = node_map.get(nid) or {}
        typ = str(meta.get("type", ""))
        if typ == "Branch":
            return nid
        if typ == "RootBox":
            return _default_branch_under_root(nid, node_map, node_ids, edges)
        par = _direct_parent_of(nid, edges)
        if not par:
            break
        nid = par
    return candidate_id


def find_best_parent_for_hypothesis(hyp: Dict, nodes: list, edges: list) -> str:
    """
    Find the most appropriate parent node in the knowledge graph for a hypothesis.
    Prioritizes specific nodes; result is normalized to a Branch (blue) parent when possible.
    """
    hyp_id = (hyp.get("id") or "").lower()

    # Build set of existing node IDs
    node_ids = {n.get("id") for n in nodes if n.get("id")}
    node_map = {n.get("id"): n for n in nodes if n.get("id")}

    # SPECIFIC node mappings — IDs must exist in seed_parameter_golf_kg.json
    keyword_to_specific_nodes = {
        # Neural Network - Attention related
        "attention": ["node_transformer_attention", "node_nn_transformer"],
        "routing": ["node_transformer_attention", "node_nn_transformer"],
        "gates": ["node_transformer_attention", "node_nn_transformer"],
        "sparse": ["node_transformer_attention", "node_nn_transformer"],

        # Neural Network - MLP/Layer related
        "mlp": ["node_transformer_mlp", "node_nn_transformer"],
        "moe": ["node_transformer_mlp", "node_nn_transformer"],
        "ffn": ["node_transformer_mlp", "node_nn_transformer"],
        "layer": ["node_transformer_block", "node_nn_transformer"],
        "transformer": ["node_nn_transformer", "node_transformer_block"],
        "depth": ["node_transformer_block", "node_nn_transformer"],
        "temporal": ["node_transformer_block", "node_nn_transformer"],

        # Neural Network - Embeddings / position
        "embed": ["node_nn_embeddings", "node_nn_embedding_strategy"],
        "positional": ["node_nn_positional_encoding"],
        "position": ["node_nn_positional_encoding", "node_nn_transformer", "node_transformer_block"],

        # Neural Network - Other
        "norm": ["node_transformer_norm"],
        "residual": ["node_transformer_residual"],
        "dropout": ["node_transformer_dropout"],
        "head": ["node_nn_output_head"],
        "output": ["node_nn_output_head"],

        # Training & Evaluation
        "optim": ["node_train_optimizer", "node_optimizer_state_strategy"],
        "adam": ["node_train_optimizer", "node_optimizer_state_strategy"],
        "sgd": ["node_train_optimizer"],
        "loss": ["node_train_loss", "node_loss_cross_entropy"],
        "objective": ["node_train_loss"],
        "precision": ["node_train_precision"],
        "grad": ["node_train_optimizer"],
        "checkpoint": ["node_train_activation_storage"],
        "8bit": ["node_train_precision"],
        "4bit": ["node_train_precision"],
        "eval": ["node_train_evaluation"],
        "budget": ["node_train_budget_accounting"],

        # Data Pipeline (valid seed IDs)
        "token": ["node_data_tokenization"],
        "tokenizer": ["node_data_tokenization"],
        "sequence": ["node_data_sequence_construction"],
        "data": ["node_data_sources"],
        "curriculum": ["node_data_curriculum_mixing"],
        "augment": ["node_data_augmentation"],
        "clean": ["node_data_cleaning"],
        "dedup": ["node_data_dedup"],
    }

    candidate: str | None = None

    # First: most specific keyword → seed node (may be Leaf; resolved to Branch below)
    best_matches: list[str] = []
    for keyword, specific_nodes in keyword_to_specific_nodes.items():
        if keyword in hyp_id:
            best_matches.extend(specific_nodes)

    for node_id in best_matches:
        if node_id in node_ids:
            candidate = node_id
            break

    # Second: pillar scores → prefer Branch list for that pillar
    if candidate is None:
        pillar_keywords = {
            "nn": [
                "neural",
                "network",
                "layer",
                "model",
                "activation",
                "norm",
                "residual",
                "attention",
                "transformer",
                "embedding",
                "positional",
                "position",
                "gpt",
                "mixer",
                "block",
                "mlp",
                "ffn",
                "causal",
                "softmax",
                "logit",
                "dropout",
                "rope",
                "alibi",
                "kv",
                "nano",
            ],
            "train": ["train", "eval", "benchmark", "budget", "memory", "compute", "optimizer", "adamw", "sgd"],
            "data": ["source", "text", "code", "book", "synthetic", "sample", "batch", "corpus", "dataset"],
        }

        pillar_scores = {pillar: 0 for pillar in pillar_keywords}
        for pillar, keywords in pillar_keywords.items():
            for kw in keywords:
                if kw in hyp_id:
                    pillar_scores[pillar] += 1

        best_pillar = max(pillar_scores, key=pillar_scores.get)
        if pillar_scores[best_pillar] > 0:
            if best_pillar == "nn":
                branch_options = [
                    "node_nn_transformer",
                    "node_transformer_block",
                    "node_transformer_attention",
                    "node_transformer_mlp",
                    "node_nn_embeddings",
                    "node_nn_positional_encoding",
                    "node_nn_output_head",
                ]
            elif best_pillar == "train":
                branch_options = [
                    "node_train_optimizer",
                    "node_train_loss",
                    "node_train_precision",
                    "node_train_evaluation",
                ]
            else:
                branch_options = [
                    "node_data_sources",
                    "node_data_cleaning",
                    "node_data_tokenization",
                    "node_data_sequence_construction",
                ]

            for branch_id in branch_options:
                if branch_id in node_ids:
                    candidate = branch_id
                    break

    # Last resort: pillar roots → resolved to default Branch hubs
    if candidate is None:
        if any(
            kw in hyp_id
            for kw in ["optim", "loss", "train", "precision", "grad", "checkpoint", "8bit", "4bit"]
        ):
            candidate = "node_root_training_eval"
        elif any(
            kw in hyp_id
            for kw in ["data", "token", "sequence", "curriculum", "augment", "clean", "dedup"]
        ):
            candidate = "node_root_data_pipeline"
        else:
            candidate = "node_root_neural_network"

    return _resolve_parent_to_branch(
        candidate,
        node_map=node_map,
        edges=edges,
        node_ids=node_ids,
    )


def build_evolution_dot(
    nodes: list,
    edges: list,
    hypotheses: List[Dict],
    frame_number: int,
    total_frames: int,
    show_up_to_idx: int,
    title_suffix: str = "",
) -> str:
    """Build a single frame showing the full knowledge graph + hypotheses."""

    # Hypothesis status colors
    status_colors = {
        "pending": ("#fef3c7", "#f59e0b", "dashed"),     # Amber, dashed
        "verified": ("#d1fae5", "#10b981", "filled"),    # Green, solid
        "refuted": ("#fee2e2", "#ef4444", "dashed"),     # Red, dashed
        "unknown": ("#f3f4f6", "#6b7280", "dashed"),     # Gray, dashed
    }

    lines = []
    lines.append("digraph EvolutionFrame {")
    # Use same layout as original
    lines.append(
        '  graph [rankdir=LR, bgcolor="#ffffff", fontname="Helvetica", fontsize=14, '
        'labelloc="t", labeljust="center", nodesep="0.15", ranksep="0.25", splines="ortho", '
        'overlap="false", pad="0.3", margin=0];'
    )
    lines.append('  node [fontname="Helvetica", fontsize=10, margin="0.10,0.07", shape=box];')
    lines.append('  edge [color="#adb5bd", arrowsize="0.6", penwidth="1"];')
    lines.append("")

    # Title with progress
    progress = f"Frame {frame_number}/{total_frames}"
    if title_suffix:
        title = f"{title_suffix} ({progress})"
    else:
        title = f"Knowledge Evolution ({progress})"
    lines.append(f'  label="{_html_escape(title)}";')
    lines.append("")

    # Find root nodes for layout
    root_ids = [n.get("id") for n in nodes if n.get("type") == "RootBox" and n.get("id")]

    # Keep roots at same rank
    if root_ids:
        lines.append("  { rank = same; " + "; ".join(f'"{r}"' for r in root_ids) + "; }")
        for left, right in zip(root_ids, root_ids[1:]):
            lines.append(f'  "{left}" -> "{right}" [style="invis"];')
        lines.append("")

    # Build node map
    node_map = {n.get("id"): n for n in nodes if n.get("id")}

    # Add all knowledge graph nodes
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        node_type = str(node.get("type", ""))
        attrs = _node_style(node_type)

        # Check if this node has hypotheses targeting it
        hyp_count = sum(
            1
            for h in hypotheses[:show_up_to_idx]
            if find_best_parent_for_hypothesis(
                {"id": _hypothesis_routing_id(h)}, nodes, edges
            )
            == node_id
        )

        label = _node_label_html(node)
        tooltip = _html_escape(str(node.get("mechanism_description", "")))

        # If node has hypotheses, add indicator
        if hyp_count > 0:
            # Modify label to show count
            label_str = str(node.get("label", ""))
            metrics_str = ""
            params = int(node.get("parameter_estimate", 0) or 0)
            mem_mb = float(node.get("memory_estimate_mb", 0.0) or 0.0)
            if params > 0:
                metrics_str += f"P:{_format_params(params)}"
            if mem_mb > 0:
                if metrics_str:
                    metrics_str += " | "
                metrics_str += f"Mem:{_format_mb(mem_mb)}MB"

            if metrics_str:
                new_label = f'<<B>{_html_escape(label_str)}</B><BR/><FONT POINT-SIZE="9">{metrics_str} | <FONT COLOR="#e67700">{hyp_count} tested</FONT></FONT>>'
            else:
                new_label = f'<<B>{_html_escape(label_str)}</B><BR/><FONT POINT-SIZE="9" COLOR="#e67700">({hyp_count} tested)</FONT>>'
            label = new_label

            # Make border thicker to indicate activity
            attrs["penwidth"] = "3"

        attrs.update({
            "label": label,
            "tooltip": tooltip,
        })

        lines.append(f'  "{node_id}" [{_dot_attr_list(attrs)}];')

    lines.append("")

    # Add all original edges
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue

        relationship = str(edge.get("relationship", ""))
        edge_attrs: dict[str, str | None] = {}
        if relationship and relationship != "sub_component_of":
            edge_attrs["label"] = _html_escape(relationship)
            edge_attrs["fontsize"] = "8"
            edge_attrs["fontcolor"] = "#495057"

        lines.append(f'  "{source}" -> "{target}" [{_dot_attr_list(edge_attrs)}];')

    # Add hypotheses as nodes connected to their parent
    if show_up_to_idx > 0 and hypotheses:
        lines.append("")
        lines.append("  // Hypotheses")

        for i, hyp in enumerate(hypotheses[:show_up_to_idx]):
            hyp_node_id = f"hyp_{i}"
            verdict = hyp.get("verdict") or "unknown"
            status = str(verdict).lower()
            fill, stroke, style_type = status_colors.get(status, status_colors["unknown"])

            # Get hypothesis display info (display_id = short label; routing_id = parent matching)
            short_id = (hyp.get("display_id") or hyp.get("id") or f"H{i}")[:40]

            # Determine if refuted and why
            if status == "refuted":
                killed_by = hyp.get("killed_by", "")
                label = f'<<B>{_html_escape(short_id)}</B><BR/><FONT POINT-SIZE="8">❌ {status.upper()}</FONT><BR/><FONT POINT-SIZE="7">Killed: {_html_escape(str(killed_by))[:20]}</FONT>>'
            else:
                label = f'<<B>{_html_escape(short_id)}</B><BR/><FONT POINT-SIZE="8">{status.upper()}</FONT>>'

            # Hypothesis node styling
            lines.append(
                f'  "{hyp_node_id}" [label={label}, fillcolor="{fill}", '
                f'color="{stroke}", fontcolor="{stroke}", style="rounded,filled,{style_type}", '
                f'shape=box, fontsize=9, width=0, height=0, penwidth=2, margin="0.08,0.05"];'
            )

            # Connect to parent - constraint=false allows free placement
            parent_id = find_best_parent_for_hypothesis(
                {"id": _hypothesis_routing_id(hyp)}, nodes, edges
            )

            # Get parent label for edge label
            parent_label = ""
            for n in nodes:
                if n.get('id') == parent_id:
                    parent_label = n.get('label', parent_id)
                    break

            # Truncate parent label for edge
            short_parent = parent_label[:15] + "..." if len(parent_label) > 15 else parent_label

            lines.append(
                f'  "{parent_id}" -> "{hyp_node_id}" [color="{stroke}", style="dashed", '
                f'penwidth=2.0, arrowsize=0.9, constraint=false, '
                f'label="{short_parent}", fontcolor="{stroke}", fontsize=8];'
            )

    lines.append("}")
    return "\n".join(lines)


def render_frame(dot_text: str, output_path: Path, dpi: int = 150) -> bool:
    """Render a single frame to PNG."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_text)
            dot_path = f.name

        result = subprocess.run(
            ["dot", f"-Gdpi={dpi}", "-Tpng", dot_path, "-o", str(output_path)],
            capture_output=True,
            text=True
        )

        os.unlink(dot_path)
        return result.returncode == 0

    except Exception as e:
        print(f"Error rendering frame: {e}")
        return False


def create_mp4_from_frames(frame_dir: Path, output_path: Path, fps: float = 1) -> bool:
    """Combine frames into MP4 using ffmpeg."""
    try:
        frame_pattern = str(frame_dir / "frame_%04d.png")

        # Verify frames exist
        frames = list(frame_dir.glob("frame_*.png"))
        print(f"  Found {len(frames)} frames in {frame_dir}")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", frame_pattern,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure dimensions divisible by 2
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ffmpeg stderr: {result.stderr}")
            return False

        return True

    except FileNotFoundError:
        print("ffmpeg not found. Install with: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"Error creating MP4: {e}")
        return False


def _evolution_frame_title(
    *,
    phase: str,
    segments: list[tuple[str, int, int]] | None,
    hyp_index: int | None,
    total_h: int,
    frame_num: int,
    total_frames: int,
) -> str:
    prog = f"Frame {frame_num}/{total_frames}"
    if segments is None:
        if phase == "baseline":
            return f"Baseline Knowledge Graph ({prog})"
        if phase == "add" and hyp_index is not None:
            return f"Hypothesis {hyp_index + 1}/{total_h} ({prog})"
        if phase == "final":
            return f"Complete — {total_h} hypotheses ({prog})"
        return prog

    if phase == "baseline":
        return f"All efforts — seed ontology ({prog})"
    if phase == "add" and hyp_index is not None:
        i = hyp_index
        for name, a, b in segments:
            if a <= i < b:
                short = name.replace("live_run_", "").replace("live_experiment_", "exp_")[:36]
                return f"{short} · +{i - a + 1}/{b - a} · total {i + 1}/{total_h} ({prog})"
        return f"Hypothesis {i + 1}/{total_h} ({prog})"
    if phase == "final":
        return f"All efforts — complete ({total_h} hypotheses) ({prog})"
    return prog


def generate_movie(
    seed_path: Path,
    output_path: Path,
    fps: float = 1.0,
    dpi: int = 150,
    *,
    experiment_dir: Path | None = None,
    hypotheses: List[Dict] | None = None,
    segments: list[tuple[str, int, int]] | None = None,
) -> bool:
    """Generate the full evolution movie (baseline → add hypotheses one by one → final)."""

    print(f"Loading seed graph from {seed_path}...")
    nodes, edges = load_seed_graph(seed_path)

    print(f"Found {len(nodes)} nodes, {len(edges)} edges in baseline graph")

    if hypotheses is None:
        if experiment_dir is None:
            print("Either experiment_dir or hypotheses= is required")
            return False
        print(f"Loading hypotheses from {experiment_dir}...")
        hypotheses = load_enriched_hypotheses_single_run(experiment_dir)
        if not hypotheses:
            hypotheses = load_hypotheses_from_run(experiment_dir)
            hypotheses = [
                {
                    "idea_id": str(h["id"]),
                    "routing_id": str(h["id"]).lower(),
                    "display_id": str(h["id"])[:48],
                    "verdict": str(h.get("verdict") or "UNKNOWN").upper(),
                    "killed_by": str(h.get("killed_by") or ""),
                }
                for h in hypotheses
            ]

    if not hypotheses:
        print("No hypotheses found")
        return False

    print(f"Found {len(hypotheses)} hypotheses (fps={fps})")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = Path(tmpdir)

        total_frames = len(hypotheses) + 2

        # Frame 0: Baseline only (FULL graph)
        print("Generating frame 0: Baseline knowledge graph...")
        t0 = _evolution_frame_title(
            phase="baseline",
            segments=segments,
            hyp_index=None,
            total_h=len(hypotheses),
            frame_num=0,
            total_frames=total_frames,
        )
        dot = build_evolution_dot(nodes, edges, hypotheses, 0, total_frames, 0, t0)
        render_frame(dot, frame_dir / "frame_0000.png", dpi)

        # Frames 1..n: Add hypotheses one by one
        for i in range(len(hypotheses)):
            frame_num = i + 1
            hyp = hypotheses[i]
            _lab = hyp.get("display_id") or hyp.get("idea_id") or hyp.get("id") or "?"
            print(f"Generating frame {frame_num}: Adding hypothesis {i+1}/{len(hypotheses)}: {_lab[:40]}...")

            tstep = _evolution_frame_title(
                phase="add",
                segments=segments,
                hyp_index=i,
                total_h=len(hypotheses),
                frame_num=frame_num,
                total_frames=total_frames,
            )
            dot = build_evolution_dot(nodes, edges, hypotheses, frame_num, total_frames, i + 1, tstep)
            render_frame(dot, frame_dir / f"frame_{frame_num:04d}.png", dpi)

        # Final frame: All hypotheses
        print(f"Generating frame {total_frames-1}: Final state with all hypotheses...")
        tf = _evolution_frame_title(
            phase="final",
            segments=segments,
            hyp_index=None,
            total_h=len(hypotheses),
            frame_num=total_frames - 1,
            total_frames=total_frames,
        )
        dot = build_evolution_dot(nodes, edges, hypotheses, total_frames - 1, total_frames,
                                  len(hypotheses), tf)
        render_frame(dot, frame_dir / f"frame_{total_frames-1:04d}.png", dpi)

        # Create MP4
        print(f"Creating MP4 at {fps} fps (~{len(hypotheses) + 2} frames)...")
        if create_mp4_from_frames(frame_dir, output_path, fps):
            print(f"✓ Generated evolution movie: {output_path}")
            return True
        else:
            print("✗ Failed to create MP4")
            return False


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate knowledge graph evolution movie v2")
    parser.add_argument("--seed", default=str(repo / "knowledge_graph/seed_parameter_golf_kg.json"),
                        help="Path to seed knowledge graph")
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Single live run directory (ignored if --merged)",
    )
    parser.add_argument("--merged", action="store_true",
                        help="Use all ideas from merged viz + snapshots under --snapshots-root")
    parser.add_argument(
        "--all-efforts",
        action="store_true",
        help="Chronological order by experiment folder; on-screen titles show which run each idea came from",
    )
    parser.add_argument("--snapshots-root", type=Path,
                        default=repo / "experiments" / "ten_hypothesis_run",
                        help="Root folder for --merged / --all-efforts")
    parser.add_argument("--output", default=str(repo / "knowledge_graph/visuals/evolution_movie_v2.mp4"),
                        help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second (overrides --duration-seconds if both set)")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Target clip length in seconds (e.g. 10); sets fps from frame count")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Resolution of frames (lower = faster render)")
    args = parser.parse_args()

    seed_path = Path(args.seed).resolve()
    output_path = Path(args.output).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    hypotheses = None
    experiment_dir = Path(args.experiment_dir).resolve() if args.experiment_dir else None

    segments: list[tuple[str, int, int]] | None = None

    if args.all_efforts:
        hypotheses, segments = load_hypotheses_chronological_by_run(
            args.snapshots_root.resolve(), seed_path
        )
        if not hypotheses:
            print("No hypotheses found under", args.snapshots_root)
            return 1
        print("Effort segments (folder → index range):")
        for name, a, b in segments:
            print(f"  {name}: [{a}, {b}) → {b - a} ideas")
    elif args.merged:
        hypotheses = load_merged_hypotheses(args.snapshots_root.resolve(), seed_path)
        if not hypotheses:
            print("No merged hypotheses found under", args.snapshots_root)
            return 1
    elif experiment_dir is None:
        print("Provide --experiment-dir, --merged, or --all-efforts")
        return 1
    else:
        hypotheses = None

    if hypotheses is not None:
        n_frames = len(hypotheses) + 2
    else:
        hy = load_enriched_hypotheses_single_run(experiment_dir)  # type: ignore
        if not hy:
            hy = load_hypotheses_from_run(experiment_dir)  # type: ignore
        n_frames = len(hy) + 2

    fps: float
    if args.fps is not None:
        fps = float(args.fps)
    elif args.duration_seconds is not None and n_frames > 0:
        fps = max(0.25, n_frames / float(args.duration_seconds))
    else:
        fps = 1.0

    success = generate_movie(
        seed_path,
        output_path,
        fps=fps,
        dpi=args.dpi,
        experiment_dir=experiment_dir,
        hypotheses=hypotheses,
        segments=segments,
    )

    if success:
        print("\nGenerating high-quality summary frame...")
        nodes, edges = load_seed_graph(seed_path)
        if hypotheses is None and experiment_dir is not None:
            hy_final = load_enriched_hypotheses_single_run(experiment_dir)
            if not hy_final:
                hy_final = load_hypotheses_from_run(experiment_dir)
                hy_final = [
                    {
                        "idea_id": str(h["id"]),
                        "routing_id": str(h["id"]).lower(),
                        "display_id": str(h["id"])[:48],
                        "verdict": str(h.get("verdict") or "UNKNOWN").upper(),
                        "killed_by": str(h.get("killed_by") or ""),
                    }
                    for h in hy_final
                ]
        else:
            hy_final = hypotheses

        dot = build_evolution_dot(nodes, edges, hy_final, 0, 1, len(hy_final),
                                  f"Final State - {len(hy_final)} Hypotheses")
        summary_path = output_path.parent / "evolution_summary_v2.png"
        render_frame(dot, summary_path, dpi=300)
        print(f"✓ Summary frame: {summary_path}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
