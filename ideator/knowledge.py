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


@dataclass(frozen=True)
class KnowledgeContext:
    summary: str
    raw: str


def load_knowledge_context(knowledge_dir: Path, *, max_chars: int = 18_000) -> str:
    if not knowledge_dir.exists():
        return ""

    structured_lines: List[str] = []
    raw_chunks: List[str] = []

    for file in _iter_knowledge_files(knowledge_dir):
        rel = file.relative_to(knowledge_dir)
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
        combined += "## Knowledge Graph (structured)\n" + summary + "\n"
    if raw:
        combined += "## Knowledge Graph (raw snippets)\n" + raw + "\n"

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
