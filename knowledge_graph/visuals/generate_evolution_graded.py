#!/usr/bin/env python3
"""
Graded evolution visualization: grey baseline (seed KG) + colour gradient for new hypotheses.

- Baseline: seed graph root nodes (and optional pill labels) in neutral grey.
- Hypotheses: blend from grey → verdict hue by run order (early = paler, later = stronger).
- Legend: HTML table top-right explaining baseline, age gradient, verdict mapping.
- Scales to many hypotheses via adjustable columns (--cols).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return rgb_to_hex(
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def load_seed_graph(path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(path.read_text())
    return data.get("nodes") or [], data.get("edges") or []


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
        verdict = complete_event.get("verdict") or "UNKNOWN"
        out.append(
            {
                "id": theory_id,
                "hypothesis_num": hyp_num,
                "verdict": verdict,
                "time": float(complete_event.get("time", 0) or 0),
                "killed_by": complete_event.get("stage1_verdict")
                if verdict == "REFUTED"
                else None,
            }
        )
    return sorted(out, key=lambda x: x["time"])


def find_parent_category(hyp: dict[str, Any]) -> str:
    hid = (hyp.get("id") or "").lower()
    if any(
        kw in hid
        for kw in [
            "attention",
            "mlp",
            "layer",
            "embed",
            "transformer",
            "depth",
            "sparse",
            "moe",
            "ffn",
        ]
    ):
        return "neural"
    if any(
        kw in hid
        for kw in [
            "optim",
            "loss",
            "train",
            "precision",
            "grad",
            "checkpoint",
        ]
    ):
        return "training"
    if any(
        kw in hid
        for kw in [
            "data",
            "token",
            "sequence",
            "curriculum",
            "augment",
            "sample",
        ]
    ):
        return "data"
    return "neural"


def verdict_palette() -> dict[str, tuple[str, str, str]]:
    """verdict -> (fill strong, stroke strong, short label)."""
    return {
        "REFUTED": ("#fecaca", "#b91c1c", "Refuted"),
        "VERIFIED": ("#bbf7d0", "#15803d", "Verified"),
        "UNKNOWN": ("#e2e8f0", "#475569", "Unknown / failed"),
    }


def hypothesis_colours(
    hyp: dict[str, Any],
    order_index: int,
    n_hyps: int,
    baseline_grey_fill: str,
    baseline_grey_stroke: str,
) -> tuple[str, str, float]:
    """
    Return (fill, stroke, age_t). age_t in [0,1]: 0 = earliest idea, 1 = latest.
    Blends baseline greys toward verdict colour by age (later = stronger).
    """
    pal = verdict_palette()
    v = (hyp.get("verdict") or "UNKNOWN").upper()
    if v not in pal:
        v = "UNKNOWN"
    strong_fill, strong_stroke, _ = pal[v]

    if n_hyps <= 1:
        age_t = 1.0
    else:
        age_t = order_index / (n_hyps - 1)

    # Early hypotheses stay closer to baseline grey; later ones approach verdict colour.
    # Quadratic easing so mid-run is visibly coloured but not fully saturated.
    blend = age_t**0.85

    gf = hex_to_rgb(baseline_grey_fill)
    gs = hex_to_rgb(baseline_grey_stroke)
    ff = hex_to_rgb(strong_fill)
    sf = hex_to_rgb(strong_stroke)

    fill = lerp_rgb(gf, ff, blend)
    stroke = lerp_rgb(gs, sf, blend)
    return fill, stroke, age_t


def build_legend_html(
    baseline_grey_fill: str,
    baseline_grey_stroke: str,
) -> str:
    """Top-right legend as Graphviz HTML-like label."""
    pal = verdict_palette()
    # Gradient swatches for refuted path (example)
    g0 = lerp_rgb(hex_to_rgb(baseline_grey_fill), hex_to_rgb(pal["REFUTED"][0]), 0.15)
    g1 = lerp_rgb(hex_to_rgb(baseline_grey_fill), hex_to_rgb(pal["REFUTED"][0]), 0.55)
    g2 = lerp_rgb(hex_to_rgb(baseline_grey_fill), hex_to_rgb(pal["REFUTED"][0]), 1.0)

    rows = []
    rows.append(
        f'<TR><TD COLSPAN="3" ALIGN="LEFT" BGCOLOR="#f8fafc"><B>Legend</B></TD></TR>'
    )
    rows.append(
        f'<TR><TD WIDTH="28" HEIGHT="18" BGCOLOR="{baseline_grey_fill}" BORDER="1" COLOR="{baseline_grey_stroke}"></TD>'
        f'<TD COLSPAN="2" ALIGN="LEFT"> Baseline (seed knowledge graph)</TD></TR>'
    )
    rows.append(
        f'<TR><TD BORDER="0"></TD><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="11">'
        f"<B>New hypotheses:</B> run order → stronger fill (same verdict family)</FONT></TD></TR>"
    )
    rows.append(
        f'<TR>'
        f'<TD BGCOLOR="{g0}" BORDER="1" COLOR="{pal["REFUTED"][1]}"></TD>'
        f'<TD BGCOLOR="{g1}" BORDER="1" COLOR="{pal["REFUTED"][1]}"></TD>'
        f'<TD BGCOLOR="{g2}" BORDER="1" COLOR="{pal["REFUTED"][1]}"></TD>'
        f"</TR>"
    )
    rows.append(
        '<TR><TD COLSPAN="3" ALIGN="LEFT"><FONT POINT-SIZE="10">Early  ·  mid-run  ·  latest (example: refuted family)</FONT></TD></TR>'
    )
    for key in ("REFUTED", "VERIFIED", "UNKNOWN"):
        ff, ss, lab = pal[key]
        icon = "❌" if key == "REFUTED" else ("✓" if key == "VERIFIED" else "?")
        rows.append(
            f'<TR><TD BGCOLOR="{ff}" BORDER="1" COLOR="{ss}"></TD>'
            f'<TD COLSPAN="2" ALIGN="LEFT"> {html_escape(icon + " " + lab)}</TD></TR>'
        )

    table = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="6" ALIGN="LEFT">'
        + "".join(rows)
        + "</TABLE>>"
    )
    return table


def build_dot(
    roots: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    *,
    cols: int,
    title: str,
) -> str:
    baseline_fill = "#e5e7eb"
    baseline_stroke = "#6b7280"

    lines: list[str] = []
    lines.append("digraph GradedEvolution {")
    lines.append(
        '  graph [bgcolor="#ffffff", fontname="Helvetica Neue", labelloc="t", labeljust="center",'
        ' rankdir=TB, nodesep=0.55, ranksep=1.4, pad=0.8, margin=0.4];'
    )
    lines.append(
        '  node [fontname="Helvetica Neue", fontsize=13, margin="0.2,0.15"];'
    )
    lines.append('  edge [color="#94a3b8", penwidth=1.2, arrowsize=0.7];')
    lines.append("")

    # Title node + legend on same rank (legend to the right)
    refuted = sum(1 for h in hypotheses if h.get("verdict") == "REFUTED")
    verified = sum(1 for h in hypotheses if h.get("verdict") == "VERIFIED")
    unk = len(hypotheses) - refuted - verified
    subtitle = (
        f"{len(hypotheses)} new hypotheses  •  {refuted} refuted  •  {verified} verified  •  {unk} unknown"
    )
    title_block = (
        f'<<TABLE BORDER="0" CELLSPACING="4">'
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="22"><B>{html_escape(title)}</B></FONT></TD></TR>'
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="14" COLOR="#475569">{html_escape(subtitle)}</FONT></TD></TR>'
        f"</TABLE>>"
    )
    lines.append(f'  title_main [shape=plaintext, label={title_block}];')
    lines.append(f'  legend_box [shape=plaintext, label={build_legend_html(baseline_fill, baseline_stroke)}];')
    # Widen invisible spacer so the HTML legend sits toward the top-right
    lines.append('  title_pad [shape=plaintext, label="", width=9.0, height=0.01];')
    lines.append("  { rank=same; title_main; title_pad; legend_box; }")
    lines.append('  title_main -> title_pad -> legend_box [style=invis, weight=10];')
    lines.append("")

    # Baseline cluster
    lines.append('  subgraph cluster_baseline {')
    lines.append('    label="Baseline ideas (seed knowledge graph)";')
    lines.append('    style="rounded,filled";')
    lines.append(f'    fillcolor="#f9fafb";')
    lines.append(f'    color="{baseline_stroke}";')
    lines.append('    fontname="Helvetica Neue"; fontsize=15; penwidth=2;')
    lines.append("    margin=16;")
    for i, r in enumerate(roots):
        rid = f"baseline_root_{i}"
        lab = html_escape(str(r.get("label", r.get("id", ""))))
        lines.append(
            f'    "{rid}" [label="{lab}", shape=box, style="rounded,filled", '
            f'fillcolor="{baseline_fill}", color="{baseline_stroke}", '
            f'fontcolor="#374151", fontsize=14, penwidth=2, width=2.8, height=0.65];'
        )
    if roots:
        lines.append("    { rank=same; " + "; ".join(f'\"baseline_root_{i}\"' for i in range(len(roots))) + "; }")
    lines.append("  }")
    lines.append("")

    if roots:
        lines.append('  title_main -> baseline_root_0 [style=invis, weight=100];')
    lines.append("")

    n = len(hypotheses)
    lines.append('  subgraph cluster_new {')
    lines.append('    label="New hypotheses (chronological order • colour = age × verdict)";')
    lines.append('    style="rounded";')
    lines.append('    color="#334155";')
    lines.append('    fontname="Helvetica Neue"; fontsize=15; penwidth=2;')
    lines.append("    margin=20;")

    col_count = max(1, cols)
    rows_n = math.ceil(n / col_count) if n else 0

    for idx, hyp in enumerate(hypotheses):
        fill, stroke, age_t = hypothesis_colours(
            hyp, idx, n, baseline_fill, baseline_stroke
        )
        nid = f"hyp_{idx}"
        name = str(hyp.get("id", ""))
        if len(name) > 36:
            name = name[:33] + "..."
        cat = find_parent_category(hyp)
        cat_tag = {"neural": "NN", "training": "Tr", "data": "Data"}.get(cat, "")
        v = (hyp.get("verdict") or "?").upper()
        kb = hyp.get("killed_by")
        line2 = html_escape(f"{v}" + (f" • {kb}" if kb else ""))
        label = (
            f'<<TABLE BORDER="0" CELLSPACING="2">'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11" COLOR="#475569">'
            f'#{idx + 1} · {html_escape(cat_tag)} · run position {age_t:.0%}</FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="15" COLOR="#0f172a"><B>{html_escape(name)}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="{stroke}"><B>{line2}</B></FONT></TD></TR>'
            f"</TABLE>>"
        )
        lines.append(
            f'    "{nid}" [label={label}, shape=box, style="rounded,filled", '
            f'fillcolor="{fill}", color="{stroke}", penwidth=2.5];'
        )

    # Grid ranks
    for r in range(rows_n):
        row_nodes = [f"hyp_{r * col_count + c}" for c in range(col_count) if r * col_count + c < n]
        if row_nodes:
            lines.append("    { rank=same; " + "; ".join(f'\"{x}\"' for x in row_nodes) + "; }")

    # Invisible row edges
    for r in range(rows_n):
        for c in range(col_count):
            i = r * col_count + c
            if i >= n:
                break
            if c > 0:
                lines.append(f'    hyp_{i - 1} -> hyp_{i} [style=invis, weight=10];')
            if r > 0 and i - col_count >= 0:
                lines.append(f'    hyp_{i - col_count} -> hyp_{i} [style=invis, weight=10];')

    lines.append("  }")

    def _stem_baseline_index(rlist: list[dict[str, Any]]) -> int:
        for i, r in enumerate(rlist):
            rid = str(r.get("id", "")).lower()
            if "neural" in rid:
                return i
        return max(0, len(rlist) // 2)

    if roots and n:
        stem = _stem_baseline_index(roots)
        lines.append(
            f'  baseline_root_{stem} -> hyp_0 [style=dashed, color="#64748b", penwidth=2, '
            f'constraint=false, label="new ideas", fontcolor="#64748b", fontsize=10];'
        )
    elif n:
        lines.append("  title_main -> hyp_0 [style=invis];")

    lines.append("}")
    return "\n".join(lines)


def render_dot(
    dot: str,
    out_png: Path,
    dpi: int,
    *,
    out_svg: Path | None = None,
) -> bool:
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
            print(r.stderr[:800])
            ok = False
        if out_svg is not None:
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
    ap = argparse.ArgumentParser(description="Graded evolution PNG (baseline grey + hypothesis gradient)")
    ap.add_argument(
        "--seed",
        default="knowledge_graph/seed_parameter_golf_kg.json",
        help="Seed KG JSON",
    )
    ap.add_argument(
        "--experiment-dir",
        default="experiments/ten_hypothesis_run/live_run_20260328_170317",
    )
    ap.add_argument(
        "--output",
        default="knowledge_graph/visuals/evolution_graded.png",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Grid columns (use 3–6; more columns = shorter image for many hyps)",
    )
    ap.add_argument(
        "--title",
        default="Knowledge evolution — baseline vs new ideas",
    )
    ap.add_argument(
        "--svg",
        action="store_true",
        help="Also write .svg (infinite zoom, good for 100+ hypotheses)",
    )
    args = ap.parse_args()

    seed_path = Path(args.seed)
    exp = Path(args.experiment_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    nodes, _edges = load_seed_graph(seed_path)
    roots = [n for n in nodes if n.get("type") == "RootBox" and n.get("id")]
    roots.sort(key=lambda x: str(x.get("id", "")))

    hyps = load_hypotheses_from_run(exp)
    if not hyps:
        print("No hypotheses found; check --experiment-dir")
        return 1

    dot = build_dot(roots, hyps, cols=args.cols, title=args.title)
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot)

    svg_path = out.with_suffix(".svg") if args.svg else None
    if render_dot(dot, out, args.dpi, out_svg=svg_path):
        mb = out.stat().st_size / (1024 * 1024)
        print(f"Wrote {out} ({mb:.2f} MiB)")
        print(f"DOT source: {dot_path}")
        if svg_path and svg_path.exists():
            print(f"SVG (zoom-friendly): {svg_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
