from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def choose_knowledge_dir(explicit: Optional[Path], *, cwd: Path) -> Optional[Path]:
    if explicit is not None:
        return explicit

    kdir = cwd / "knowledge_graph"
    if kdir.exists() and kdir.is_dir():
        return kdir
    return None


def get_working_graph_path(knowledge_dir: Path) -> Path:
    """Get the path to the working graph (graph.json).

    The working graph is where all falsification results are stored.
    This is separate from the seed graph which is read-only.
    """
    return knowledge_dir / "graph.json"


def get_seed_graph_path(knowledge_dir: Path) -> Path:
    """Get the path to the seed graph (seed_parameter_golf_kg.json).

    The seed graph contains the base knowledge hierarchy and is read-only.
    It should never be modified by falsification runs.
    """
    return knowledge_dir / "seed_parameter_golf_kg.json"


@dataclass(frozen=True)
class KnowledgeContext:
    summary: str
    raw: str


def load_knowledge_context(knowledge_dir: Path, *, max_chars: int = 18_000) -> str:
    """Load knowledge context from the working graph (graph.json).

    The working graph accumulates all falsification results and is the
    source of truth for what has been tried and what failed. The seed
    graph (seed_parameter_golf_kg.json) is read-only and provides the
    base knowledge hierarchy.

    Priority:
    1. graph.json - Working graph with all falsification results
    2. outbox/ideator/ - Generated ideas
    3. outbox/falsifier/ - Falsification outputs
    4. seed_parameter_golf_kg.json - Base knowledge (read-only)
    """
    if not knowledge_dir.exists():
        return ""

    structured_lines: List[str] = []
    raw_chunks: List[str] = []

    # Priority 1: Working graph (graph.json) - contains falsification results
    working_graph = knowledge_dir / "graph.json"
    if working_graph.exists():
        text = _safe_read_text(working_graph)
        if text.strip() and text != '{"nodes": {}, "edges": [], "version": "1.0"}':
            # Parse working graph and summarize nodes with falsification results
            try:
                graph = json.loads(text)
                nodes = graph.get("nodes", {})
                # Prioritize REFUTED and PASSED nodes - these are most informative
                for node_id, node in sorted(nodes.items(),
                    key=lambda x: (0 if x[1].get("status") in ("REFUTED", "STAGE_2_PASSED") else 1)):
                    status = node.get("status", "UNKNOWN")
                    idea_id = node.get("idea_id", node_id)
                    title = node.get("title", "")

                    # Build rich summary for failed/passed ideas
                    if status in ("REFUTED", "STAGE_2_PASSED", "STAGE_1_PASSED"):
                        falsification = node.get("falsification", {})
                        killed_by = falsification.get("killed_by", "")
                        kill_reason = falsification.get("kill_reason", "")
                        outcome = falsification.get("outcome", "")

                        # Include metrics if available
                        metrics = falsification.get("metrics", {})
                        bpb = metrics.get("bits_per_byte")
                        if bpb is None:
                            bpb = metrics.get("measured_bpb")
                        loss = metrics.get("loss_at_100")

                        summary_parts = [f"- {idea_id}: {title} [{status}]"]
                        if killed_by:
                            summary_parts.append(f"killed_by={killed_by}")
                        if outcome:
                            summary_parts.append(f"outcome={outcome}")
                        if bpb:
                            summary_parts.append(f"bpb={bpb:.2f}")
                        if kill_reason:
                            # Truncate kill reason
                            reason = kill_reason[:60] + "..." if len(kill_reason) > 60 else kill_reason
                            summary_parts.append(f"reason='{reason}'")

                        structured_lines.append(" ".join(summary_parts))
                    elif status == "OFFICIAL_RECORD":
                        mm = node.get("measured_metrics") or {}
                        bpb = mm.get("val_bpb")
                        track = (node.get("source") or {}).get("track", "")
                        bpb_s = f" val_bpb={bpb:.4f}" if isinstance(bpb, (int, float)) else ""
                        tr = f" [{track}]" if track else ""
                        structured_lines.append(
                            f"- {idea_id}: {title} [OFFICIAL_RECORD]{tr}{bpb_s}"
                        )
                    else:
                        # Simple summary for other statuses
                        structured_lines.append(f"- {idea_id}: {title} [{status}]")
            except json.JSONDecodeError:
                pass

    # Priority 2: Seed graph for base knowledge
    seed_graph = knowledge_dir / "seed_parameter_golf_kg.json"
    if seed_graph.exists():
        text = _safe_read_text(seed_graph)
        parsed = _try_summarize_json("seed_parameter_golf_kg.json", text)
        if parsed:
            structured_lines.append("\n## Base Knowledge (Seed Graph)")
            structured_lines.extend(parsed)

    # Priority 3: Other files (outbox, etc.) - but skip empty graph.json
    for file in _iter_knowledge_files(knowledge_dir):
        rel = file.relative_to(knowledge_dir)

        # Skip files we've already processed
        if file.name == "graph.json" or file.name == "seed_parameter_golf_kg.json":
            continue

        text = _safe_read_text(file)
        if not text.strip():
            continue

        parsed_summary = _try_summarize_json(rel.as_posix(), text)
        if parsed_summary:
            structured_lines.extend(parsed_summary)
        else:
            raw_chunks.append(f"### {rel.as_posix()}\n{text.strip()}\n")

    summary = "\n".join(structured_lines).strip()
    raw = "\n".join(raw_chunks).strip()

    combined = ""
    if summary:
        combined += "## Knowledge Graph (Working Graph with Falsification Results)\n" + summary + "\n"
    if raw:
        combined += "## Knowledge Graph (Raw Snippets)\n" + raw + "\n"

    if len(combined) <= max_chars:
        return combined.strip()
    return (combined[: max_chars - 20] + "\n...[truncated]").strip()


def _iter_knowledge_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.name.lower() in ("readme.md", "readme.txt"):
            continue
        if "visuals" in {part.lower() for part in p.parts}:
            continue
        if p.suffix.lower() not in (".json", ".jsonl", ".md", ".txt"):
            continue
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".bin"):
            continue
        files.append(p)
    return files


def _safe_read_text(path: Path, *, max_bytes: int = 120_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _try_summarize_json(source_name: str, text: str) -> Optional[List[str]]:
    s = text.strip()
    if not s:
        return None

    if source_name.endswith(".jsonl"):
        lines = [ln for ln in s.splitlines() if ln.strip()]
        objs: List[Any] = []
        for ln in lines[-80:]:
            try:
                objs.append(json.loads(ln))
            except Exception:
                continue
        if not objs:
            return None
        return _summarize_records(source_name, objs)

    if not source_name.endswith(".json"):
        return None

    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj):
        nodes = obj.get("nodes") if isinstance(obj.get("nodes"), list) else []
        edges = obj.get("edges") if isinstance(obj.get("edges"), list) else []
        lines = [f"- {source_name}: graph with {len(nodes)} nodes, {len(edges)} edges"]
        lines.extend(_summarize_records(source_name, nodes[-40:]))
        return lines

    if isinstance(obj, list):
        return _summarize_records(source_name, obj[-60:])

    if isinstance(obj, dict):
        # Heuristic: treat as one record.
        return _summarize_records(source_name, [obj])

    return None


def _summarize_records(source_name: str, records: List[Any]) -> List[str]:
    lines: List[str] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        title = r.get("title") or r.get("name") or r.get("idea_title") or ""
        rid = r.get("id") or r.get("idea_id") or r.get("node_id") or ""
        status = r.get("status") or r.get("stage") or ""
        score = r.get("score") or r.get("metric") or r.get("bits_per_byte") or None
        components = r.get("components") or r.get("tags") or []
        comp_str = ""
        if isinstance(components, list) and components:
            comp_str = f" components={components[:8]}"
        score_str = ""
        if isinstance(score, (int, float, str)) and str(score):
            score_str = f" score={score}"
        head = " - ".join([x for x in (str(rid).strip(), str(title).strip(), str(status).strip()) if x])
        if not head:
            continue
        lines.append(f"- {source_name}: {head}{score_str}{comp_str}")
    return lines
