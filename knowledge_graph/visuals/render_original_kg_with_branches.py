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
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_evolution_movie_v2 import (
    build_evolution_dot,
    load_seed_graph,
    render_frame,
    load_hypotheses_from_run,
)


def _parse_time(data: dict[str, Any]) -> datetime | None:
    iso = data.get("iso_time")
    if iso:
        try:
            t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return t
        except ValueError:
            pass
    ts = data.get("timestamp")
    if ts is not None:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return None


def load_base_ids(seed_path: Path) -> set[str]:
    nodes, _ = load_seed_graph(seed_path)
    return {n["id"] for n in nodes if n.get("id")}


def hypothesis_dict_from_snapshot_node(n: dict[str, Any]) -> dict[str, Any]:
    idea = str(n.get("idea_id") or n.get("node_id") or "").strip()
    title = str(n.get("title") or "").strip()
    why = str(n.get("what_and_why") or "").strip()
    routing = f"{idea} {title} {why}".strip().lower()
    display = idea or title[:48] or "hypothesis"
    st = str(n.get("status") or "").upper()
    verdict = "UNKNOWN"
    if st == "REFUTED":
        verdict = "REFUTED"
    elif st == "VERIFIED":
        verdict = "VERIFIED"
    killed = ""
    if st == "REFUTED":
        fal = n.get("falsification") or {}
        killed = str(fal.get("killed_by") or fal.get("kill_reason") or "")[:40]
    return {
        "idea_id": idea,
        "routing_id": routing,
        "display_id": display[:48],
        "verdict": verdict,
        "killed_by": killed,
    }


def hypothesis_dict_from_viz_run(run: dict[str, Any], idea: dict[str, Any]) -> dict[str, Any]:
    idea_id = str(idea.get("idea_id") or idea.get("theory_id") or "").strip()
    title = str(idea.get("title") or "").strip()
    nov = str(idea.get("novelty_summary") or idea.get("what_and_why") or "").strip()
    routing = f"{idea_id} {title} {nov}".strip().lower()
    verdict = run.get("verdict")
    if verdict is None:
        verdict = "UNKNOWN"
    else:
        verdict = str(verdict).upper()
    killed = ""
    if verdict == "REFUTED":
        killed = str(run.get("stage1_verdict") or "")[:40]
    return {
        "idea_id": idea_id,
        "routing_id": routing,
        "display_id": (idea_id or title)[:48] or "hypothesis",
        "verdict": verdict,
        "killed_by": killed,
    }


def collect_hypotheses_from_all_visualization_runs(snapshots_root: Path) -> list[dict[str, Any]]:
    """Merge runs[].idea_json from every live_run_*/visualization/visualization_data.json."""
    by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for viz in sorted(snapshots_root.glob("live_run_*/visualization/visualization_data.json")):
        try:
            mtime = viz.stat().st_mtime
            data = json.loads(viz.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for run in data.get("runs") or []:
            idea = run.get("idea_json")
            if not idea or not isinstance(idea, dict):
                continue
            idea_id = str(idea.get("idea_id") or idea.get("theory_id") or "").strip()
            if not idea_id:
                continue
            h = hypothesis_dict_from_viz_run(run, idea)
            if idea_id not in by_id or mtime >= by_id[idea_id][0]:
                by_id[idea_id] = (mtime, h)
    return [by_id[k][1] for k in sorted(by_id)]


def collect_hypotheses_from_snapshots(
    snapshots_root: Path,
    base_ids: set[str],
) -> list[dict[str, Any]]:
    """Latest snapshot row per experiment idea_id."""
    best: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for snap in sorted(snapshots_root.glob("live_run_*/graph_snapshots/snapshot_*.json")):
        try:
            data = json.loads(snap.read_text())
        except json.JSONDecodeError:
            continue
        t = _parse_time(data)
        if t is None:
            continue
        for n in data.get("nodes") or []:
            nid = n.get("node_id") or n.get("idea_id")
            if not nid or nid in base_ids:
                continue
            h = hypothesis_dict_from_snapshot_node(n)
            if nid not in best or t >= best[nid][0]:
                best[nid] = (t, h)
    return [best[k][1] for k in sorted(best, key=lambda x: best[x][0])]


def merge_hypotheses_by_idea_id(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer entries with longer routing_id (richer keyword match)."""
    best: dict[str, dict[str, Any]] = {}
    for group in groups:
        for h in group:
            iid = str(h.get("idea_id") or h.get("display_id") or "").strip()
            if not iid:
                continue
            prev = best.get(iid)
            if prev is None:
                best[iid] = h
                continue
            if len(h.get("routing_id") or "") > len(prev.get("routing_id") or ""):
                best[iid] = h
    return [best[k] for k in sorted(best)]


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
        hyps = load_hypotheses_from_run(exp)
        if not hyps:
            print("No hypotheses in", exp)
            return 1
        # Enrich with routing_id/display_id for short theory_id-only entries
        enriched: list[dict[str, Any]] = []
        viz_path = exp / "visualization" / "visualization_data.json"
        runs_by_num: dict[int, dict] = {}
        if viz_path.exists():
            try:
                vd = json.loads(viz_path.read_text())
                for run in vd.get("runs") or []:
                    n = int(run.get("hypothesis_number") or 0)
                    runs_by_num[n] = run
            except (json.JSONDecodeError, OSError):
                pass
        for h in hyps:
            num = int(h.get("hypothesis_num") or 0)
            run = runs_by_num.get(num)
            idea = (run or {}).get("idea_json") if run else None
            if idea and isinstance(idea, dict):
                enriched.append(hypothesis_dict_from_viz_run(run, idea))
            else:
                tid = str(h.get("id") or "")
                enriched.append(
                    {
                        "idea_id": tid,
                        "routing_id": tid.lower(),
                        "display_id": tid[:48],
                        "verdict": str(h.get("verdict") or "UNKNOWN").upper(),
                        "killed_by": str(h.get("killed_by") or ""),
                    }
                )
        hyps = enriched
        title = f"Seed ontology + branches — run {exp.name} ({len(hyps)} ideas)"
    elif args.source == "snapshots":
        hyps = collect_hypotheses_from_snapshots(root, base_ids)
        title = f"Seed ontology + branches — snapshots only ({len(hyps)} ideas)"
    elif args.source == "viz-all":
        hyps = collect_hypotheses_from_all_visualization_runs(root)
        title = f"Seed ontology + branches — all visualization runs ({len(hyps)} ideas)"
    else:
        viz_h = collect_hypotheses_from_all_visualization_runs(root)
        snap_h = collect_hypotheses_from_snapshots(root, base_ids)
        hyps = merge_hypotheses_by_idea_id(viz_h, snap_h)
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
        import subprocess
        import tempfile
        import os

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
