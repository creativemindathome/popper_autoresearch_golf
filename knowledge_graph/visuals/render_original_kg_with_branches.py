#!/usr/bin/env python3
"""
Render the full seed parameter-golf ontology (original knowledge graph) plus experiment
hypotheses as nodes that branch off their mapped parent concepts — same layout engine as
generate_evolution_movie_v2.build_evolution_dot (Graphviz).

Data sources:
  --experiment-dir   Only that run’s visualization_data.json (often just a few hypotheses).
  --source merged (default)   All ideas: every live_run_*/visualization/visualization_data.json
                    (runs[].idea_json) merged with graph_snapshots, deduped by idea_id.
  --source viz-all  Only merged visualization runs (no snapshots).
  --source snapshots Only graph_snapshots union.

Usage:
  python3 knowledge_graph/visuals/render_original_kg_with_branches.py
  python3 knowledge_graph/visuals/render_original_kg_with_branches.py --source merged --svg
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from generate_evolution_movie_v2 import (
    build_evolution_dot,
    load_seed_graph,
    render_frame,
)
from hypothesis_sources import (
    collect_hypotheses_from_all_visualization_runs,
    collect_hypotheses_from_snapshots,
    load_base_ids,
    load_enriched_hypotheses_single_run,
    load_merged_hypotheses,
    merge_hypotheses_by_idea_id,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Full seed KG + experiment branches (Graphviz PNG/SVG)"
    )
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument(
        "--seed",
        type=Path,
        default=repo / "knowledge_graph" / "seed_parameter_golf_kg.json",
    )
    ap.add_argument(
        "--snapshots-root",
        type=Path,
        default=repo / "experiments" / "ten_hypothesis_run",
        help="Scans live_run_*/graph_snapshots/",
    )
    ap.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="If set, load only this run's visualization_data.json (overrides --source)",
    )
    ap.add_argument(
        "--source",
        choices=("merged", "viz-all", "snapshots"),
        default="merged",
        help="merged=all viz runs + snapshots (dedupe); viz-all=every visualization_data.json; snapshots=graph_snapshots only",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=repo / "knowledge_graph" / "visuals" / "original_kg_with_branches.png",
    )
    ap.add_argument("--dpi", type=int, default=120, help="Lower if the PNG is huge or slow")
    ap.add_argument("--svg", action="store_true", help="Also write .svg next to the PNG")
    args = ap.parse_args()

    seed_path = args.seed.resolve()
    nodes, edges = load_seed_graph(seed_path)
    base_ids = load_base_ids(seed_path)

    root = args.snapshots_root.resolve()

    if args.experiment_dir is not None:
        exp = args.experiment_dir.resolve()
        hyps = load_enriched_hypotheses_single_run(exp)
        if not hyps:
            print("No hypotheses in", exp)
            return 1
        title = f"Seed ontology + branches — run {exp.name} ({len(hyps)} ideas)"
    elif args.source == "snapshots":
        hyps = collect_hypotheses_from_snapshots(root, base_ids)
        title = f"Seed ontology + branches — snapshots only ({len(hyps)} ideas)"
    elif args.source == "viz-all":
        hyps = collect_hypotheses_from_all_visualization_runs(root)
        title = f"Seed ontology + branches — all visualization runs ({len(hyps)} ideas)"
    else:
        hyps = load_merged_hypotheses(root, seed_path)
        title = f"Seed ontology + branches — merged viz + snapshots ({len(hyps)} ideas)"

    if not hyps:
        print("No experiment hypotheses found under", root)
        return 1

    dot = build_evolution_dot(
        nodes,
        edges,
        hyps,
        frame_number=1,
        total_frames=1,
        show_up_to_idx=len(hyps),
        title_suffix=title,
    )

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not render_frame(dot, out, dpi=args.dpi):
        print("Graphviz render failed (is `dot` installed?)")
        return 1
    print(f"Wrote {out} ({len(hyps)} hypothesis nodes)")

    if args.svg:
        svg_path = out.with_suffix(".svg")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
                f.write(dot)
                dot_path = f.name
            subprocess.run(
                ["dot", "-Tsvg", dot_path, "-o", str(svg_path)],
                check=True,
                capture_output=True,
            )
            os.unlink(dot_path)
            print(f"Wrote {svg_path}")
        except (subprocess.CalledProcessError, OSError) as e:
            print("SVG export failed:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
