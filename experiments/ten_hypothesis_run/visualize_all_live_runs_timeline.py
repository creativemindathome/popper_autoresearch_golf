#!/usr/bin/env python3
"""
Aggregate every live_run_* directory and trace_run_*.log under this folder into one
wall-clock timeline: run spans, knowledge-graph evolution frames, and trace markers.

Usage:
  cd experiments/ten_hypothesis_run
  python3 visualize_all_live_runs_timeline.py
  python3 visualize_all_live_runs_timeline.py --output ../../knowledge_graph/visuals/all_live_experiments_timeline.png

Requires: matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_live_run_dt(name: str) -> datetime | None:
    m = re.fullmatch(r"live_run_(\d{8})_(\d{6})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _parse_trace_log_dt(name: str) -> datetime | None:
    m = re.fullmatch(r"trace_run_(\d{8})_(\d{6})\.log", name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


@dataclass
class RunRecord:
    path: Path
    run_id: str
    start: datetime | None
    end: datetime | None
    total_hypotheses: int
    completed: int
    failed: int
    total_time_seconds: float | None
    snapshots_captured: int


@dataclass
class EvolutionPoint:
    t: datetime
    run_id: str
    total_ideas: int
    falsified: int
    approved: int
    frame: int


def _load_summary(p: Path) -> dict[str, Any] | None:
    s = p / "summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def _load_viz(p: Path) -> dict[str, Any] | None:
    v = p / "visualization" / "visualization_data.json"
    if not v.exists():
        return None
    return json.loads(v.read_text())


def discover_live_runs(base: Path) -> list[RunRecord]:
    out: list[RunRecord] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        rid = child.name
        if not rid.startswith("live_run_"):
            continue
        summary = _load_summary(child)
        viz = _load_viz(child)
        start: datetime | None = None
        end: datetime | None = None
        if viz and "start_time" in viz:
            st = float(viz["start_time"])
            start = datetime.fromtimestamp(st, tz=timezone.utc)
        else:
            start = _parse_live_run_dt(rid)

        total_h = 0
        completed = 0
        failed = 0
        total_sec: float | None = None
        snaps = 0
        if summary:
            total_h = int(summary.get("total_hypotheses", 0))
            completed = int(summary.get("completed", 0))
            failed = int(summary.get("failed", 0))
            total_sec = summary.get("total_time_seconds")
            if total_sec is not None:
                total_sec = float(total_sec)
            snaps = int(summary.get("snapshots_captured", 0))

        if start and total_sec is not None:
            from datetime import timedelta

            end = start + timedelta(seconds=total_sec)
        elif start and viz and viz.get("timeline"):
            last_t = max(
                float(e.get("time", 0) or 0) for e in viz["timeline"]
            )
            from datetime import timedelta

            end = start + timedelta(seconds=last_t)
        else:
            end = start

        out.append(
            RunRecord(
                path=child,
                run_id=rid,
                start=start,
                end=end,
                total_hypotheses=total_h,
                completed=completed,
                failed=failed,
                total_time_seconds=total_sec,
                snapshots_captured=snaps,
            )
        )
    return out


def collect_evolution_points(runs: list[RunRecord]) -> list[EvolutionPoint]:
    pts: list[EvolutionPoint] = []
    for r in runs:
        viz = _load_viz(r.path)
        if not viz or not viz.get("start_time"):
            continue
        st = float(viz["start_time"])
        base = datetime.fromtimestamp(st, tz=timezone.utc)
        for ev in viz.get("evolution") or []:
            off = float(ev.get("timestamp", 0) or 0)
            t = datetime.fromtimestamp(st + off, tz=timezone.utc)
            pts.append(
                EvolutionPoint(
                    t=t,
                    run_id=r.run_id,
                    total_ideas=int(ev.get("total_ideas", 0)),
                    falsified=int(ev.get("falsified", 0)),
                    approved=int(ev.get("approved", 0)),
                    frame=int(ev.get("frame", 0)),
                )
            )
    pts.sort(key=lambda x: x.t)
    return pts


def collect_snapshot_points(base: Path, runs: list[RunRecord]) -> list[tuple[datetime, str, int, int]]:
    """(time, run_id, total_ideas, n_nodes) from graph_snapshots when no evolution array."""
    out: list[tuple[datetime, str, int, int]] = []
    for r in runs:
        snap_dir = r.path / "graph_snapshots"
        if not snap_dir.is_dir():
            continue
        for f in sorted(snap_dir.glob("snapshot_*.json")):
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            iso = data.get("iso_time")
            ts = data.get("timestamp")
            if iso:
                try:
                    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            elif ts is not None:
                t = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            else:
                continue
            ti = int(data.get("total_ideas", 0))
            nodes = data.get("nodes")
            n_nodes = len(nodes) if isinstance(nodes, list) else 0
            out.append((t, r.run_id, ti, n_nodes))
    out.sort(key=lambda x: x[0])
    return out


def discover_trace_logs(base: Path) -> list[datetime]:
    times: list[datetime] = []
    for f in base.glob("trace_run_*.log"):
        dt = _parse_trace_log_dt(f.name)
        if dt:
            times.append(dt)
    times.sort()
    return times


def build_figure(
    runs: list[RunRecord],
    evo_pts: list[EvolutionPoint],
    snap_pts: list[tuple[datetime, str, int, int]],
    trace_times: list[datetime],
    title: str,
) -> Any:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        height_ratios=[1.1, 1.4, 0.45],
        constrained_layout=True,
    )
    ax_gantt, ax_evo, ax_trace = axes

    # --- Gantt: run wall time ---
    gantt_runs = [r for r in runs if r.start is not None and r.end is not None]
    y_labels = [r.run_id.replace("live_run_", "")[:15] for r in gantt_runs]
    n = len(gantt_runs)
    for i, r in enumerate(gantt_runs):
        ax_gantt.barh(
            i,
            mdates.date2num(r.end) - mdates.date2num(r.start),
            left=mdates.date2num(r.start),
            height=0.65,
            color="#3b82f6",
            alpha=0.85,
            edgecolor="#1e3a8a",
        )
    ax_gantt.set_yticks(range(n))
    ax_gantt.set_yticklabels(y_labels, fontsize=7)
    ax_gantt.set_title("Live experiment runs (wall time)")
    ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_gantt.grid(True, axis="x", alpha=0.3)
    ax_gantt.set_xlabel("UTC time")

    # --- Evolution: total_ideas over time ---
    if evo_pts:
        by_run: dict[str, list[EvolutionPoint]] = {}
        for p in evo_pts:
            by_run.setdefault(p.run_id, []).append(p)
        cmap = plt.colormaps["tab20"]
        for j, (rid, seq) in enumerate(sorted(by_run.items(), key=lambda x: x[0])):
            xs = [p.t for p in seq]
            ys = [p.total_ideas for p in seq]
            c = cmap(j % 20)
            short = rid.replace("live_run_", "")[:18]
            ax_evo.plot(xs, ys, "o-", color=c, label=short, markersize=4, linewidth=1.2)
        ax_evo.set_ylabel("total_ideas (evolution frames)")
    elif snap_pts:
        by_run: dict[str, list[tuple[datetime, int]]] = {}
        for t, rid, ti, _ in snap_pts:
            by_run.setdefault(rid, []).append((t, ti))
        cmap = plt.colormaps["tab20"]
        for j, (rid, seq) in enumerate(sorted(by_run.items(), key=lambda x: x[0])):
            xs = [a[0] for a in seq]
            ys = [a[1] for a in seq]
            c = cmap(j % 20)
            short = rid.replace("live_run_", "")[:18]
            ax_evo.plot(xs, ys, "s-", color=c, label=short, markersize=4, linewidth=1.2)
        ax_evo.set_ylabel("total_ideas (graph snapshots)")
    else:
        ax_evo.text(
            0.5,
            0.5,
            "No evolution frames or graph snapshots found.",
            ha="center",
            va="center",
            transform=ax_evo.transAxes,
        )
    ax_evo.set_title("Knowledge graph evolution vs time")
    ax_evo.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_evo.grid(True, alpha=0.3)
    if evo_pts or snap_pts:
        ax_evo.legend(loc="upper left", fontsize=6, ncol=2, framealpha=0.9)

    # --- Trace log markers ---
    bounds: list[datetime] = []
    for r in runs:
        if r.start:
            bounds.append(r.start)
        if r.end:
            bounds.append(r.end)
    for p in evo_pts:
        bounds.append(p.t)
    for t, _, _, _ in snap_pts:
        bounds.append(t)
    bounds.extend(trace_times)
    if bounds:
        ax_trace.set_xlim(min(bounds), max(bounds))
    for tt in trace_times:
        ax_trace.axvline(tt, color="#f97316", alpha=0.7, linewidth=1.0, linestyle="--")
    ax_trace.set_yticks([])
    ax_trace.set_title(f"Trace runs (n={len(trace_times)}) — vertical lines")
    ax_trace.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#3b82f6", edgecolor="#1e3a8a", label="Live run span"),
        Patch(facecolor="#f97316", label="trace_run_*.log time"),
    ]
    ax_gantt.legend(handles=legend_elements, loc="upper right", fontsize=8)

    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description="Timeline of all live experiments and traces")
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing live_run_* folders and trace logs",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "knowledge_graph"
        / "visuals"
        / "all_live_experiments_timeline.png",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--title",
        default="All live experiments & traces — evolution over time (UTC)",
    )
    args = ap.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return 1

    base = args.base.resolve()
    runs = discover_live_runs(base)
    evo_pts = collect_evolution_points(runs)
    snap_pts = collect_snapshot_points(base, runs)
    trace_times = discover_trace_logs(base)

    import matplotlib.pyplot as plt

    fig = build_figure(runs, evo_pts, snap_pts, trace_times, args.title)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  live runs: {len(runs)}, evolution points: {len(evo_pts)}, snapshots: {len(snap_pts)}, traces: {len(trace_times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
