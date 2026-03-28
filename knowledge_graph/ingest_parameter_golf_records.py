#!/usr/bin/env python3
"""Merge OpenAI Parameter Golf leaderboard folders under records/ into graph.json.

Each records/<track>/<run>/submission.json becomes a node with status OFFICIAL_RECORD
and source.type parameter-golf-record. Idempotent: re-run updates the same node_ids.

Usage:
  python3 knowledge_graph/ingest_parameter_golf_records.py          # dry-run (summary only)
  python3 knowledge_graph/ingest_parameter_golf_records.py --apply  # write graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from falsifier.graph.locking import atomic_read_json, atomic_write_json  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _node_id_for(track: str, run_dir: str) -> str:
    safe = run_dir.replace(" ", "_")
    return f"pg-official-{track}-{safe}"


def _load_submission(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_node(
    track: str,
    run_name: str,
    sub: dict[str, Any],
    rel_readme: str,
) -> dict[str, Any]:
    nid = _node_id_for(track, run_name)
    title = str(sub.get("name") or run_name)
    blurb = str(sub.get("blurb") or "")
    now = time.time()
    measured = {
        k: sub[k]
        for k in ("val_bpb", "val_loss", "bytes_total", "bytes_code")
        if k in sub
    }
    return {
        "node_id": nid,
        "idea_id": nid,
        "title": title,
        "theory_type": "benchmark_reference",
        "status": "OFFICIAL_RECORD",
        "status_history": [
            {
                "status": "OFFICIAL_RECORD",
                "timestamp": now,
                "actor": "ingest_parameter_golf_records",
                "metadata": {"schema_version": "parameter-golf.submission.v1"},
            }
        ],
        "source": {
            "type": "parameter-golf-record",
            "track": track,
            "record_dir": run_name,
            "readme_relative": rel_readme,
            "author": sub.get("author"),
            "github_id": sub.get("github_id"),
            "date": sub.get("date"),
        },
        "what_and_why": blurb,
        "expected_metric_change": "",
        "novelty_claims": [],
        "parameter_estimate": "",
        "parent_architecture": "OpenAI Parameter Golf (train_gpt.py in record folder)",
        "risk_factors": [],
        "implementation": {
            "train_gpt_code": "",
            "falsifier_smoke_tests": [],
            "implementation_steps": [],
            "record_train_gpt_path": f"records/{track}/{run_name}/train_gpt.py",
        },
        "review": {
            "novelty_summary": blurb[:500],
            "expected_metric_change": "",
            "decision": None,
            "feedback": None,
            "novelty_claims": [],
        },
        "measured_metrics": measured,
        "tags": ["parameter-golf", "leaderboard", track],
        "change_types": [],
        "falsification": None,
        "parents": [],
        "total_wall_seconds": 0.0,
        "total_gpu_seconds": 0.0,
        "_created": now,
        "_created_by": "ingest_parameter_golf_records",
    }


def collect_record_nodes(records_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    nodes: list[dict[str, Any]] = []
    skipped: list[str] = []
    for track_dir in sorted(records_root.iterdir()):
        if not track_dir.is_dir() or track_dir.name.startswith("."):
            continue
        if not track_dir.name.startswith("track_"):
            continue
        track = track_dir.name
        for run in sorted(track_dir.iterdir()):
            if not run.is_dir():
                continue
            sub_path = run / "submission.json"
            if not sub_path.is_file():
                skipped.append(f"{track}/{run.name} (no submission.json)")
                continue
            sub = _load_submission(sub_path)
            if not sub:
                skipped.append(f"{track}/{run.name} (invalid submission.json)")
                continue
            rel_readme = f"records/{track}/{run.name}/README.md"
            nodes.append(_build_node(track, run.name, sub, rel_readme))
    return nodes, skipped


def merge_into_graph(graph_path: Path, record_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    data = atomic_read_json(graph_path, {"nodes": {}, "edges": [], "metadata": {}})
    if "nodes" not in data:
        data["nodes"] = {}
    if not isinstance(data["nodes"], dict):
        raise TypeError("graph.json nodes must be a dict")
    if "edges" not in data:
        data["edges"] = []
    if "metadata" not in data:
        data["metadata"] = {}

    for node in record_nodes:
        nid = node["node_id"]
        data["nodes"][nid] = node

    meta = data["metadata"]
    meta["parameter_golf_records"] = {
        "ingested_at": time.time(),
        "count": len(record_nodes),
        "records_root": "records/",
    }
    meta["last_modified"] = time.time()
    meta["last_modified_by"] = "ingest_parameter_golf_records"
    meta["node_count"] = len(data["nodes"])

    atomic_write_json(data, graph_path)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=_repo_root() / "records",
        help="Path to records/ (default: repo records/)",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=_repo_root() / "knowledge_graph" / "graph.json",
        help="Path to graph.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write merged nodes to graph.json (default: print plan only)",
    )
    args = parser.parse_args()

    records_root = args.records.resolve()
    if not records_root.is_dir():
        print(f"records directory not found: {records_root}")
        return 1

    nodes, skipped = collect_record_nodes(records_root)
    print(f"Found {len(nodes)} submission.json entries under {records_root}")
    for line in skipped[:20]:
        print(f"  skip: {line}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")

    nodes.sort(
        key=lambda n: (
            (n.get("measured_metrics") or {}).get("val_bpb") is None,
            n.get("title") or "",
        )
    )
    for n in nodes[:5]:
        m = n.get("measured_metrics") or {}
        bpb = m.get("val_bpb")
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "?"
        print(f"  - {n['node_id']}: val_bpb={bpb_s} ({n.get('title')})")
    if len(nodes) > 5:
        print(f"  ... and {len(nodes) - 5} more")

    if args.apply:
        merge_into_graph(args.graph.resolve(), nodes)
        print(f"Wrote {len(nodes)} nodes to {args.graph}")
    else:
        print("Dry-run only. Pass --apply to update knowledge_graph/graph.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
