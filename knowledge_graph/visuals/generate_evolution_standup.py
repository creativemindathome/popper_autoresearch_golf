#!/usr/bin/env python3
"""
Standup-quality slide: horizontal timeline of knowledge evolution.

- Graph title (top): headline + run id + coarse stats
- One left-to-right chain: baseline → each hypothesis in time order → epilogue
- Projector-friendly type (auto-shrinks if many steps)
- Outputs PNG (+ optional SVG) and .dot

Usage:
  python3 knowledge_graph/visuals/generate_evolution_standup.py \\
    --experiment-dir experiments/.../live_run_XXX --svg
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_hypotheses_from_run(experiment_dir: Path) -> list[dict[str, Any]]:
    viz = experiment_dir / "visualization" / "visualization_data.json"
    if not viz.exists():
        return []
    data = json.loads(viz.read_text())
    hyp_events: dict[int, dict[str, Any]] = {}
    for event in data.get("timeline", []):
        n = int(event.get("hypothesis", 0))
        hyp_events.setdefault(n, {"events": []})
        hyp_events[n]["events"].append(event)

    out: list[dict[str, Any]] = []
    for hyp_num, hdata in hyp_events.items():
        start_event = complete_event = None
        for e in hdata["events"]:
            if e.get("event") == "start":
                start_event = e
            elif e.get("event") == "complete":
                complete_event = e
        if not complete_event:
            continue
        theory_id = start_event.get("theory_id") if start_event else None
        if not theory_id:
            theory_id = f"hyp_{hyp_num}"
        verdict = complete_event.get("verdict")
        if verdict is None:
            verdict = "UNKNOWN"
        out.append(
            {
                "id": theory_id,
                "hypothesis_num": hyp_num,
                "verdict": str(verdict).upper(),
                "time": float(complete_event.get("time", 0) or 0),
                "killed_by": (complete_event.get("stage1_verdict") or "")
                if str(verdict).upper() == "REFUTED"
                else None,
            }
        )
    return sorted(out, key=lambda x: x["time"])


def load_summary(experiment_dir: Path) -> dict[str, Any] | None:
    p = experiment_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def verdict_style(v: str) -> tuple[str, str, str, str]:
    u = (v or "UNKNOWN").upper()
    if u == "REFUTED":
        return "#fef2f2", "#b91c1c", "Refuted", "✗"
    if u == "VERIFIED":
        return "#ecfdf5", "#047857", "Verified", "✓"
    return "#f8fafc", "#334155", "Unknown", "?"


def build_standup_dot(
    hypotheses: list[dict[str, Any]],
    *,
    graph_title: str,
    run_id: str,
    minutes: float | None,
) -> str:
    n = len(hypotheses)
    if n <= 8:
        fs_big, fs_mid, fs_small = 18, 14, 11
        hyp_width = 2.5
    elif n <= 14:
        fs_big, fs_mid, fs_small = 15, 12, 10
        hyp_width = 2.1
    else:
        fs_big, fs_mid, fs_small = 12, 10, 8
        hyp_width = 1.75

    refuted = sum(1 for h in hypotheses if h.get("verdict") == "REFUTED")
    verified = sum(1 for h in hypotheses if h.get("verdict") == "VERIFIED")
    unknown = n - refuted - verified

    line3 = (
        f"{n} steps • {refuted} refuted • {verified} verified • {unknown} unknown"
        + (f" • ~{minutes:.1f} min wall" if minutes is not None else "")
    )
    graph_lbl = "\\n".join(
        [dot_escape(graph_title), f"Run: {dot_escape(run_id)}", dot_escape(line3)]
    )

    lines: list[str] = []
    lines.append("digraph StandupEvolution {")
    lines.append(
        "  graph [rankdir=LR, bgcolor=\"#fafafa\", fontname=\"Helvetica Neue\","
        f' label="{graph_lbl}",'
        " labelloc=\"t\", labeljust=\"c\", fontsize=20, fontcolor=\"#0f172a\","
        " nodesep=0.55, ranksep=1.0, splines=line, pad=0.9, margin=0.4,"
        " concentrate=false];"
    )
    lines.append(
        "  node [fontname=\"Helvetica Neue\", style=\"rounded,filled\", penwidth=2.5];"
    )
    lines.append(
        "  edge [color=\"#94a3b8\", penwidth=5, arrowsize=1.05, minlen=1.8];"
    )
    lines.append("")

    bl = (
        f'<<TABLE BORDER="0" CELLSPACING="3">'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_small}" COLOR="#64748b">t₀</FONT></TD></TR>'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_big}" COLOR="#1e293b"><B>Seed KG</B></FONT></TD></TR>'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_small}" COLOR="#475569">Baseline concepts</FONT></TD></TR>'
        f"</TABLE>>"
    )
    lines.append(
        f'  baseline [label={bl}, fillcolor="#e2e8f0", color="#475569", '
        f'width=2.5, height=1.35, penwidth=3];'
    )

    for i, h in enumerate(hypotheses):
        vid = f"h{i}"
        v = str(h.get("verdict", "UNKNOWN")).upper()
        fill, stroke, vlab, icon = verdict_style(v)
        name = str(h.get("id", ""))
        if len(name) > 40:
            name = name[:37] + "…"
        kb = h.get("killed_by") or ""
        extra = ""
        if v == "REFUTED" and kb:
            extra = (
                f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{max(1, fs_small - 1)}" COLOR="{stroke}">'
                f"{html_escape(str(kb)[:24])}</FONT></TD></TR>"
            )

        lbl = (
            f'<<TABLE BORDER="0" CELLSPACING="2">'
            f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_small}" COLOR="#64748b">'
            f"t{i + 1} · {html_escape(vlab)}</FONT></TD></TR>"
            f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_big}" COLOR="{stroke}"><B>'
            f"{html_escape(icon)} {i + 1}</B></FONT></TD></TR>"
            f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_mid}" COLOR="#0f172a"><B>'
            f"{html_escape(name)}</B></FONT></TD></TR>"
            f"{extra}"
            f"</TABLE>>"
        )
        lines.append(
            f'  {vid} [label={lbl}, fillcolor="{fill}", color="{stroke}", '
            f'width={hyp_width}, height=1.25, penwidth=3];'
        )

    ep = (
        f'<<TABLE BORDER="0" CELLSPACING="3">'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_small}" COLOR="#64748b">t_end</FONT></TD></TR>'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_big}" COLOR="#312e81"><B>Current knowledge</B></FONT></TD></TR>'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="{fs_small}" COLOR="#4338ca">'
        f"{html_escape(str(n))} probes accumulated</FONT></TD></TR>"
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="10" COLOR="#64748b">'
        f"Legend: slate = baseline • red = refuted • green = verified • grey = unknown"
        f"</FONT></TD></TR>"
        f"</TABLE>>"
    )
    lines.append(
        f'  epilogue [label={ep}, fillcolor="#eef2ff", color="#4338ca", '
        f'width=2.7, height=1.45, penwidth=3];'
    )

    lines.append("")
    chain = ["baseline"] + [f"h{i}" for i in range(n)] + ["epilogue"]
    for a, b in zip(chain, chain[1:]):
        lines.append(f"  {a} -> {b};")
    lines.append("}")
    return "\n".join(lines)


def render_dot(dot: str, out_png: Path, dpi: int, out_svg: Path | None) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
            f.write(dot)
            dp = f.name
        ok = True
        r = subprocess.run(
            ["dot", f"-Gdpi={dpi}", "-Tpng", str(dp), "-o", str(out_png)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr[:1200])
            ok = False
        if out_svg:
            s = subprocess.run(
                ["dot", "-Tsvg", str(dp), "-o", str(out_svg)],
                capture_output=True,
                text=True,
            )
            if s.returncode != 0:
                print(s.stderr[:800])
                ok = False
        os.unlink(dp)
        return ok
    except OSError as e:
        print(e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Standup timeline: knowledge evolution")
    ap.add_argument(
        "--experiment-dir",
        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
    )
    ap.add_argument(
        "--output",
        default="knowledge_graph/visuals/evolution_standup.png",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--svg", action="store_true")
    ap.add_argument("--title", default="Knowledge evolution over time")
    args = ap.parse_args()

    exp = Path(args.experiment_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    hyps = load_hypotheses_from_run(exp)
    if not hyps:
        print("No hypotheses; need visualization/visualization_data.json")
        return 1

    summary = load_summary(exp)
    minutes = None
    if summary and summary.get("total_time_minutes") is not None:
        minutes = float(summary["total_time_minutes"])

    dot = build_standup_dot(
        hyps,
        graph_title=args.title,
        run_id=exp.name,
        minutes=minutes,
    )
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot)

    svg = out.with_suffix(".svg") if args.svg else None
    if render_dot(dot, out, args.dpi, svg):
        print(f"Wrote {out}")
        print(f"DOT: {dot_path}")
        if svg and svg.exists():
            print(f"SVG: {svg}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
