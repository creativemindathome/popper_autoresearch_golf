"""
Shared hypothesis loading for KG visuals (merged viz + snapshots, single-run enrichment).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_hypotheses_from_run(experiment_dir: Path) -> list[dict[str, Any]]:
    """Load hypotheses from visualization_data.json timeline (theory ids, verdicts)."""
    viz_data_path = experiment_dir / "visualization" / "visualization_data.json"
    if not viz_data_path.exists():
        return []

    data = json.loads(viz_data_path.read_text())
    hypotheses: list[dict[str, Any]] = []
    timeline = data.get("timeline", [])
    hyp_events: dict[int, dict] = {}
    for event in timeline:
        hyp_num = int(event.get("hypothesis", 0))
        if hyp_num not in hyp_events:
            hyp_events[hyp_num] = {"events": []}
        hyp_events[hyp_num]["events"].append(event)

    for hyp_num, hdata in hyp_events.items():
        start_event = None
        complete_event = None
        for e in hdata["events"]:
            if e.get("event") == "start":
                start_event = e
            elif e.get("event") == "complete":
                complete_event = e

        if complete_event:
            theory_id = start_event.get("theory_id") if start_event else None
            if not theory_id:
                theory_id = f"hyp_{hyp_num}"

            hypotheses.append(
                {
                    "id": theory_id,
                    "hypothesis_num": hyp_num,
                    "verdict": complete_event.get("verdict", "UNKNOWN"),
                    "time": complete_event.get("time", 0),
                    "killed_by": complete_event.get("stage1_verdict")
                    if complete_event.get("verdict") == "REFUTED"
                    else None,
                }
            )

    return sorted(hypotheses, key=lambda x: x["time"])


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
    data = json.loads(seed_path.read_text())
    return {n["id"] for n in data.get("nodes", []) if n.get("id")}


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


def load_merged_hypotheses(snapshots_root: Path, seed_path: Path) -> list[dict[str, Any]]:
    """All unique ideas from viz runs + graph snapshots (dedupe by idea_id)."""
    base_ids = load_base_ids(seed_path)
    viz_h = collect_hypotheses_from_all_visualization_runs(snapshots_root)
    snap_h = collect_hypotheses_from_snapshots(snapshots_root, base_ids)
    return merge_hypotheses_by_idea_id(viz_h, snap_h)


def load_hypotheses_chronological_by_run(
    snapshots_root: Path,
    seed_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int]]]:
    """
    Ideas in chronological order of experiment folders (sorted path names), one segment per
    run that contributed idea_json rows. First time an idea_id appears wins; later duplicates
    skipped. Remaining nodes only found in graph_snapshots append in one tail segment.

    Returns:
        hypotheses flat list, and segments as (folder_name, start_index, end_index) half-open.
    """
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    segments: list[tuple[str, int, int]] = []

    for viz in sorted(snapshots_root.glob("*/visualization/visualization_data.json")):
        run_name = viz.parent.parent.name
        try:
            data = json.loads(viz.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        start = len(ordered)
        for run in data.get("runs") or []:
            idea = run.get("idea_json")
            if not idea or not isinstance(idea, dict):
                continue
            idea_id = str(idea.get("idea_id") or idea.get("theory_id") or "").strip()
            if not idea_id or idea_id in seen:
                continue
            seen.add(idea_id)
            ordered.append(hypothesis_dict_from_viz_run(run, idea))
        end = len(ordered)
        if end > start:
            segments.append((run_name, start, end))

    base_ids = load_base_ids(seed_path)
    snap = collect_hypotheses_from_snapshots(snapshots_root, base_ids)
    start = len(ordered)
    for h in snap:
        iid = str(h.get("idea_id") or h.get("display_id") or "").strip()
        if not iid or iid in seen:
            continue
        seen.add(iid)
        ordered.append(h)
    end = len(ordered)
    if end > start:
        segments.append(("graph_snapshots_only", start, end))

    return ordered, segments


def load_enriched_hypotheses_single_run(experiment_dir: Path) -> list[dict[str, Any]]:
    """Timeline order with full idea_json text for parent routing."""
    hyps = load_hypotheses_from_run(experiment_dir)
    if not hyps:
        return []
    viz_path = experiment_dir / "visualization" / "visualization_data.json"
    runs_by_num: dict[int, dict] = {}
    if viz_path.exists():
        try:
            vd = json.loads(viz_path.read_text())
            for run in vd.get("runs") or []:
                n = int(run.get("hypothesis_number") or 0)
                runs_by_num[n] = run
        except (json.JSONDecodeError, OSError):
            pass
    enriched: list[dict[str, Any]] = []
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
    return enriched
