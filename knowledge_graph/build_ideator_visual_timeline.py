#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _truncate(text: str | None, *, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)].rstrip() + "…"


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def _canonical_idea_node_id(idea_id: str) -> str:
    idea_id = (idea_id or "").strip()
    if not idea_id:
        return "idea_unknown"
    safe = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", idea_id)
    return f"idea_{safe}"


KEYWORD_TO_SEED_NODE_IDS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\\b(low[- ]?rank|lrf|svd)\\b", re.I), ["node_mlp_low_rank", "node_embed_factorized"]),
    (re.compile(r"\\b(factoriz(e|ed|ation)|factorized)\\b", re.I), ["node_embed_factorized", "node_head_adaptive_softmax"]),
    (re.compile(r"\\b(quantiz(e|ation)|quant|int8|8[- ]?bit|codebook|lookup table)\\b", re.I), ["node_kv_cache_int8"]),
    (re.compile(r"\\b(moe|mixture[- ]of[- ]experts|sparse experts)\\b", re.I), ["node_mlp_moe"]),
    (re.compile(r"\\b(optimizer|adamw|adafactor|moments)\\b", re.I), ["node_optimizer_state_strategy", "node_opt_adamw_8bit_state", "node_opt_adafactor"]),
]


def infer_seed_links(text: str, *, seed_node_ids: set[str]) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()

    for pat, node_ids in KEYWORD_TO_SEED_NODE_IDS:
        if not pat.search(text):
            continue
        for node_id in node_ids:
            if node_id in seed_node_ids and node_id not in seen:
                picked.append(node_id)
                seen.add(node_id)

    return picked


@dataclass(frozen=True)
class TimelineEvent:
    kind: str
    generated_at: datetime
    idea_id: str
    title: str
    novelty_summary: str
    decision: str | None
    novelty_score: int | None
    primary_reasons: list[str]
    revision_instructions: str | None
    similar_to_knowledge: list[str]
    source_path: str


def _load_review_failures(review_failures_dir: Path) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for path in _iter_json_files(review_failures_dir):
        try:
            obj = _read_json(path)
        except Exception:
            continue
        if str(obj.get("schema_version") or "") != "ideator.review_failure.v1":
            continue
        generated_at = _parse_iso8601(str(obj.get("generated_at") or "")) or datetime.fromtimestamp(path.stat().st_mtime)

        previous_idea = obj.get("previous_idea") if isinstance(obj.get("previous_idea"), dict) else {}
        reviewer_feedback = obj.get("reviewer_feedback") if isinstance(obj.get("reviewer_feedback"), dict) else {}

        idea_id = str(previous_idea.get("idea_id") or "")
        title = str(previous_idea.get("title") or idea_id or "Untitled idea")
        novelty_summary = str(previous_idea.get("novelty_summary") or "")

        decision = reviewer_feedback.get("decision")
        if decision is not None:
            decision = str(decision)

        novelty_score = reviewer_feedback.get("novelty_score")
        try:
            novelty_score_int = int(novelty_score) if novelty_score is not None else None
        except Exception:
            novelty_score_int = None

        primary_reasons_raw = reviewer_feedback.get("primary_reasons")
        primary_reasons: list[str] = []
        if isinstance(primary_reasons_raw, list):
            primary_reasons = [str(x) for x in primary_reasons_raw if str(x).strip()]

        similar_to_raw = reviewer_feedback.get("similar_to_knowledge")
        similar_to_knowledge: list[str] = []
        if isinstance(similar_to_raw, list):
            similar_to_knowledge = [str(x) for x in similar_to_raw if str(x).strip()]

        revision_instructions = reviewer_feedback.get("revision_instructions")
        revision_instructions = str(revision_instructions) if revision_instructions else None

        events.append(
            TimelineEvent(
                kind="review_failure",
                generated_at=generated_at,
                idea_id=idea_id,
                title=title,
                novelty_summary=novelty_summary,
                decision=decision,
                novelty_score=novelty_score_int,
                primary_reasons=primary_reasons,
                revision_instructions=revision_instructions,
                similar_to_knowledge=similar_to_knowledge,
                source_path=str(path.as_posix()),
            )
        )
    return events


def _load_approved_ideas(outbox_dir: Path) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for path in _iter_json_files(outbox_dir):
        if path.name.startswith("latest"):
            continue
        if path.name.endswith("_review.json"):
            continue
        try:
            obj = _read_json(path)
        except Exception:
            continue
        schema_version = str(obj.get("schema_version") or "")
        if not schema_version.startswith("ideator.idea."):
            continue

        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        reviewer = meta.get("reviewer") if isinstance(meta.get("reviewer"), dict) else {}
        decision = reviewer.get("decision")
        if decision is None:
            continue
        decision = str(decision)

        generated_at = _parse_iso8601(str(meta.get("generated_at") or "")) or datetime.fromtimestamp(path.stat().st_mtime)
        idea_id = str(obj.get("idea_id") or "")
        title = str(obj.get("title") or idea_id or "Untitled idea")
        novelty_summary = str(obj.get("novelty_summary") or "")

        novelty_score = reviewer.get("novelty_score")
        try:
            novelty_score_int = int(novelty_score) if novelty_score is not None else None
        except Exception:
            novelty_score_int = None

        primary_reasons_raw = reviewer.get("primary_reasons")
        primary_reasons: list[str] = []
        if isinstance(primary_reasons_raw, list):
            primary_reasons = [str(x) for x in primary_reasons_raw if str(x).strip()]

        events.append(
            TimelineEvent(
                kind="reviewed_idea",
                generated_at=generated_at,
                idea_id=idea_id,
                title=title,
                novelty_summary=novelty_summary,
                decision=decision,
                novelty_score=novelty_score_int,
                primary_reasons=primary_reasons,
                revision_instructions=None,
                similar_to_knowledge=[],
                source_path=str(path.as_posix()),
            )
        )
    return events


def _load_background_ideas(outbox_dir: Path, *, include_idea_ids: set[str]) -> list[dict[str, Any]]:
    background: list[dict[str, Any]] = []
    for path in _iter_json_files(outbox_dir):
        if path.name.startswith("latest"):
            continue
        if path.name.endswith("_review.json"):
            continue
        try:
            obj = _read_json(path)
        except Exception:
            continue
        schema_version = str(obj.get("schema_version") or "")
        if not schema_version.startswith("ideator.idea."):
            continue
        idea_id = str(obj.get("idea_id") or "")
        if not idea_id or idea_id not in include_idea_ids:
            continue
        title = str(obj.get("title") or idea_id)
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        generated_at = str(meta.get("generated_at") or "")
        background.append(
            {
                "idea_id": idea_id,
                "node_id": _canonical_idea_node_id(idea_id),
                "label": title,
                "status": "KNOWN",
                "generated_at": generated_at,
                "source_path": str(path.as_posix()),
                "novelty_summary": _truncate(str(obj.get("novelty_summary") or ""), max_chars=220),
            }
        )
    return background


def build_timeline(
    *,
    seed_graph_path: Path,
    outbox_dir: Path,
    max_events: int,
) -> dict[str, Any]:
    seed = _read_json(seed_graph_path)
    nodes_raw = seed.get("nodes") if isinstance(seed.get("nodes"), list) else []
    edges_raw = seed.get("edges") if isinstance(seed.get("edges"), list) else []

    seed_nodes: list[dict[str, Any]] = []
    seed_node_ids: set[str] = set()
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        node_id = n.get("id")
        if not node_id:
            continue
        node_id = str(node_id)
        seed_node_ids.add(node_id)
        seed_nodes.append(
            {
                "id": node_id,
                "label": str(n.get("label") or node_id),
                "type": str(n.get("type") or "Node"),
                "status": str(n.get("status") or "BASE_KNOWLEDGE"),
            }
        )

    seed_edges: list[dict[str, Any]] = []
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        source = e.get("source")
        target = e.get("target")
        if source is None or target is None:
            continue
        seed_edges.append({"source": str(source), "target": str(target), "kind": "seed"})

    review_failures = _load_review_failures(outbox_dir / "review_failures")
    reviewed_ideas = _load_approved_ideas(outbox_dir)
    all_events = sorted([*review_failures, *reviewed_ideas], key=lambda ev: ev.generated_at)
    if max_events > 0:
        all_events = all_events[-max_events:]

    event_idea_ids = {ev.idea_id for ev in all_events if ev.idea_id}
    referenced_background_idea_ids: set[str] = set()
    for ev in all_events:
        for ref in ev.similar_to_knowledge:
            if ref and ref not in event_idea_ids:
                referenced_background_idea_ids.add(ref)

    background_ideas = _load_background_ideas(outbox_dir, include_idea_ids=referenced_background_idea_ids)

    steps: list[dict[str, Any]] = []

    root_ids = [n["id"] for n in seed_nodes if n.get("type") == "RootBox"]
    steps.append(
        {
            "title": "Load knowledge graph",
            "subtitle": f"Seed: {len(seed_nodes)} nodes, {len(seed_edges)} edges",
            "duration_ms": 1700,
            "actions": [{"type": "highlight", "ids": root_ids, "style": "pulse"}] if root_ids else [],
        }
    )

    if background_ideas:
        actions: list[dict[str, Any]] = []
        for idea in background_ideas:
            actions.append(
                {
                    "type": "add_node",
                    "node": {
                        "id": idea["node_id"],
                        "label": idea["label"],
                        "type": "Idea",
                        "status": "KNOWN",
                        "meta": {
                            "idea_id": idea["idea_id"],
                            "generated_at": idea.get("generated_at"),
                            "source_path": idea.get("source_path"),
                            "novelty_summary": idea.get("novelty_summary"),
                        },
                    },
                }
            )
            links = infer_seed_links(
                (idea.get("label") or "") + "\n" + (idea.get("novelty_summary") or ""),
                seed_node_ids=seed_node_ids,
            )
            for node_id in links:
                actions.append(
                    {"type": "add_edge", "edge": {"source": idea["node_id"], "target": node_id, "kind": "mentions"}}
                )

        steps.append(
            {
                "title": "Existing ideas in the knowledge graph",
                "subtitle": ", ".join([x["label"] for x in background_ideas[:3]]) + ("…" if len(background_ideas) > 3 else ""),
                "duration_ms": 2200,
                "actions": actions,
            }
        )

    prev_event_node_id: str | None = None
    prev_event_decision: str | None = None

    for ev in all_events:
        idea_node_id = _canonical_idea_node_id(ev.idea_id)
        idea_text = f"{ev.title}\n{ev.novelty_summary}"
        linked_seed_nodes = infer_seed_links(idea_text, seed_node_ids=seed_node_ids)

        highlight_ids: list[str] = []
        highlight_ids.extend(linked_seed_nodes)
        for similar in ev.similar_to_knowledge:
            highlight_ids.append(_canonical_idea_node_id(similar))

        steps.append(
            {
                "title": "Scan: retrieve relevant concepts",
                "subtitle": ev.title,
                "duration_ms": 1600,
                "actions": [{"type": "highlight", "ids": sorted(set(highlight_ids)), "style": "glow"}]
                if highlight_ids
                else [],
            }
        )

        actions: list[dict[str, Any]] = [
            {
                "type": "add_node",
                "node": {
                    "id": idea_node_id,
                    "label": ev.title,
                    "type": "Idea",
                    "status": "PENDING_REVIEW",
                    "meta": {
                        "idea_id": ev.idea_id,
                        "source_path": ev.source_path,
                        "generated_at": ev.generated_at.isoformat(),
                        "novelty_summary": _truncate(ev.novelty_summary, max_chars=280),
                    },
                },
            }
        ]

        for node_id in linked_seed_nodes:
            actions.append({"type": "add_edge", "edge": {"source": idea_node_id, "target": node_id, "kind": "builds_on"}})

        for similar in ev.similar_to_knowledge:
            actions.append(
                {
                    "type": "add_edge",
                    "edge": {
                        "source": idea_node_id,
                        "target": _canonical_idea_node_id(similar),
                        "kind": "similar_to",
                    },
                }
            )

        if prev_event_node_id and prev_event_decision in {"revise", "reject"} and ev.decision == "pass":
            actions.append(
                {
                    "type": "add_edge",
                    "edge": {"source": prev_event_node_id, "target": idea_node_id, "kind": "revision"},
                }
            )

        steps.append(
            {
                "title": "Ideate: propose new node",
                "subtitle": ev.idea_id,
                "duration_ms": 2200,
                "actions": actions,
            }
        )

        status = "PENDING_REVIEW"
        if ev.decision == "pass":
            status = "APPROVED"
        elif ev.decision in {"revise", "reject"}:
            status = "REVISE"

        steps.append(
            {
                "title": f"Review: {ev.decision or 'unknown'}",
                "subtitle": f"Novelty score: {ev.novelty_score}" if ev.novelty_score is not None else "",
                "duration_ms": 2400,
                "actions": [
                    {"type": "set_status", "id": idea_node_id, "status": status},
                    {"type": "highlight", "ids": [idea_node_id], "style": "pulse"},
                ],
                "notes": {
                    "primary_reasons": ev.primary_reasons[:5],
                    "revision_instructions": _truncate(ev.revision_instructions, max_chars=320) if ev.revision_instructions else None,
                },
            }
        )

        prev_event_node_id = idea_node_id
        prev_event_decision = ev.decision

    return {
        "schema_version": "knowledge_graph.ideator_visual_timeline.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": {"nodes": seed_nodes, "edges": seed_edges},
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an animation timeline for ideator→knowledge-graph updates.")
    parser.add_argument(
        "--seed-graph",
        default="knowledge_graph/seed_parameter_golf_kg.json",
        help="Path to seed knowledge graph JSON (default: knowledge_graph/seed_parameter_golf_kg.json)",
    )
    parser.add_argument(
        "--outbox-dir",
        default="knowledge_graph/outbox/ideator",
        help="Ideator outbox directory (default: knowledge_graph/outbox/ideator)",
    )
    parser.add_argument(
        "--out",
        default="knowledge_graph/visuals/ideator_kg_update/timeline.json",
        help="Output timeline JSON (default: knowledge_graph/visuals/ideator_kg_update/timeline.json)",
    )
    parser.add_argument(
        "--inline-js",
        default=None,
        help="Optional output JS file that assigns `window.__IDEATOR_TIMELINE__` (useful for opening index.html directly)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=6,
        help="Maximum number of ideator events to include (default: 6; 0 means all)",
    )
    args = parser.parse_args()

    seed_graph_path = Path(args.seed_graph)
    outbox_dir = Path(args.outbox_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline = build_timeline(seed_graph_path=seed_graph_path, outbox_dir=outbox_dir, max_events=args.max_events)
    out_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.inline_js:
        js_path = Path(args.inline_js)
        js_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))
        js_path.write_text(
            "/* Auto-generated by knowledge_graph/build_ideator_visual_timeline.py */\n"
            "window.__IDEATOR_TIMELINE__ = "
            + payload
            + ";\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
