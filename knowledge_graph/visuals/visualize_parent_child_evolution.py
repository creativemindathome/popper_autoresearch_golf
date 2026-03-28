#!/usr/bin/env python3
"""
Plot how experiment (child) hypotheses attach to base-knowledge-graph parents over time.

- Loads the seed ontology from knowledge_graph/seed_parameter_golf_kg.json (edges define
  parent=source → child=target).
- For each graph snapshot, maps every non-seed node (hypothesis idea) to a seed parent
  using the same keyword routing as generate_evolution_movie_v2.find_best_parent_for_hypothesis.
- Aggregates by root pillar (Data / Neural Network / Training) and by whether the
  mapped parent is a Leaf vs Branch/Root in the seed graph.

Usage:
  python3 knowledge_graph/visuals/visualize_parent_child_evolution.py
  python3 knowledge_graph/visuals/visualize_parent_child_evolution.py \\
    --snapshots-root experiments/ten_hypothesis_run \\
    --output knowledge_graph/visuals/parent_child_evolution.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_evolution_movie_v2 import find_best_parent_for_hypothesis


ROOT_LABELS = {
    "node_root_data_pipeline": "Data pipeline",
    "node_root_neural_network": "Neural network",
    "node_root_training_eval": "Training & eval",
}


def load_seed(seed_path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(seed_path.read_text())
    return data["nodes"], data["edges"]


def build_children_map(edges: list[dict]) -> dict[str, list[str]]:
    ch: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s and t:
            ch[str(s)].append(str(t))
    return dict(ch)


def build_node_to_root(children: dict[str, list[str]]) -> dict[str, str]:
    roots = [
        "node_root_data_pipeline",
        "node_root_neural_network",
        "node_root_training_eval",
    ]
    node_to_root: dict[str, str] = {}
    for root in roots:
        stack = [root]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node_to_root[nid] = root
            for c in children.get(nid, []):
                stack.append(c)
    return node_to_root


def node_text_for_routing(n: dict[str, Any]) -> str:
    parts = [
        str(n.get("node_id") or n.get("idea_id") or ""),
        str(n.get("title") or ""),
        str(n.get("what_and_why") or ""),
    ]
    return " ".join(parts).lower()


def parse_snapshot_time(data: dict[str, Any]) -> datetime | None:
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


def collect_snapshots(snapshots_root: Path) -> list[tuple[datetime, Path, dict]]:
    rows: list[tuple[datetime, Path, dict]] = []
    for snap in sorted(snapshots_root.glob("live_run_*/graph_snapshots/snapshot_*.json")):
        try:
            data = json.loads(snap.read_text())
        except json.JSONDecodeError:
            continue
        t = parse_snapshot_time(data)
        if t is None:
            continue
        rows.append((t, snap, data))
    rows.sort(key=lambda x: x[0])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Parent vs child (base KG) evolution over time")
    ap.add_argument(
        "--seed",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "knowledge_graph" / "seed_parameter_golf_kg.json",
    )
    ap.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "ten_hypothesis_run",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "parent_child_evolution.png",
    )
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return 1

    seed_nodes, seed_edges = load_seed(args.seed.resolve())
    base_ids = {n["id"] for n in seed_nodes if n.get("id")}
    node_map = {n["id"]: n for n in seed_nodes if n.get("id")}
    children = build_children_map(seed_edges)
    node_to_root = build_node_to_root(children)

    snapshots = collect_snapshots(args.snapshots_root.resolve())
    if not snapshots:
        print("No graph snapshots found under", args.snapshots_root)
        return 1

    times: list[datetime] = []
    pillar_counts: dict[str, list[int]] = {k: [] for k in ROOT_LABELS}
    leaf_parent_counts: list[int] = []
    branch_parent_counts: list[int] = []

    for t, _path, data in snapshots:
        times.append(t)
        nodes = data.get("nodes") or []
        per_pillar = {k: 0 for k in ROOT_LABELS}
        leaf_n = 0
        branch_n = 0

        for n in nodes:
            nid = n.get("node_id") or n.get("idea_id")
            if not nid or nid in base_ids:
                continue
            hyp = {"id": node_text_for_routing(n)}
            parent_id = find_best_parent_for_hypothesis(hyp, seed_nodes, seed_edges)
            root = node_to_root.get(parent_id, parent_id)
            if root in per_pillar:
                per_pillar[root] += 1
            ptype = (node_map.get(parent_id) or {}).get("type") or ""
            if ptype == "Leaf":
                leaf_n += 1
            else:
                branch_n += 1

        for k in ROOT_LABELS:
            pillar_counts[k].append(per_pillar[k])
        leaf_parent_counts.append(leaf_n)
        branch_parent_counts.append(branch_n)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    ax1, ax2 = axes

    # Stacked area: hypotheses per pillar (mapped parent → root)
    ax1.stackplot(
        times,
        [pillar_counts[k] for k in ROOT_LABELS],
        labels=[ROOT_LABELS[k] for k in ROOT_LABELS],
        colors=["#38bdf8", "#a78bfa", "#fbbf24"],
        alpha=0.9,
    )
    ax1.set_title("Experiment hypotheses by base-KG pillar (mapped parent → root)")
    ax1.set_ylabel("Hypothesis nodes in snapshot")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax1.grid(True, alpha=0.25)

    # Leaf vs non-leaf mapped parent (stacked)
    ax2.stackplot(
        times,
        [leaf_parent_counts, branch_parent_counts],
        labels=["Mapped to Leaf parent", "Mapped to Branch/Root parent"],
        colors=["#34d399", "#818cf8"],
        alpha=0.9,
    )
    ax2.set_title(
        "Child hypotheses vs parent tier in seed graph (Leaf vs Branch/Root of mapped parent)"
    )
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        "Evolution: experiment nodes relative to seed ontology parents (UTC)",
        fontsize=12,
        fontweight="bold",
    )

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({len(snapshots)} snapshots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
