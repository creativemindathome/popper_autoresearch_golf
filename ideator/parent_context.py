"""Resolve a human-readable parent binding for ideator prompts.

Links the on-disk parent train_gpt.py to knowledge graph nodes (e.g. OFFICIAL_RECORD)
so the model edits a specific baseline instead of inventing a generic codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .knowledge import get_working_graph_path
from .parent_code import ParentCode, sha256_text


def _norm_rel(p: str) -> str:
    return p.replace("\\", "/").strip()


def _relpath_or_none(repo_root: Path, path: Path) -> str | None:
    try:
        return _norm_rel(str(path.resolve().relative_to(repo_root.resolve())))
    except ValueError:
        return None


def lookup_official_record_node(
    repo_root: Path,
    graph_path: Path,
    parent_py: Path,
) -> dict[str, Any] | None:
    """If parent_py matches a records/ submission ingested into graph.json, return that node."""
    if not graph_path.is_file():
        return None
    try:
        import json

        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    nodes = data.get("nodes") or {}
    rel = _relpath_or_none(repo_root, parent_py)
    if not rel:
        return None
    rel_n = _norm_rel(rel)
    best: dict[str, Any] | None = None
    for _nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("status") != "OFFICIAL_RECORD":
            continue
        impl = node.get("implementation") or {}
        if not isinstance(impl, dict):
            continue
        rec_path = impl.get("record_train_gpt_path")
        if isinstance(rec_path, str) and _norm_rel(rec_path) == rel_n:
            best = node
            break
    return best


def _resolve_parent_py(
    parent: ParentCode,
    *,
    save_root: Path | None,
    parent_file_path: str,
) -> Path | None:
    ref = parent.ref
    if ref.kind == "local" and ref.local_path:
        p = Path(ref.local_path)
        return p if p.is_file() else None
    if ref.kind == "run" and ref.run_id and save_root:
        p = save_root / "runs" / ref.run_id / Path(parent_file_path).name
        return p if p.is_file() else None
    return None


def build_parent_binding(
    *,
    repo_root: Path,
    parent: ParentCode,
    knowledge_dir: Path | None,
    save_root: Path | None = None,
    parent_file_path: str = "train_gpt.py",
    explicit_graph_node_id: str | None = None,
) -> dict[str, Any]:
    """Return extra fields merged into ideator ``parent_code_ref`` (JSON-safe)."""
    ref = parent.ref
    resolved = _resolve_parent_py(parent, save_root=save_root, parent_file_path=parent_file_path)
    rel = _relpath_or_none(repo_root, resolved) if resolved else None

    graph_node: dict[str, Any] | None = None
    graph_path = get_working_graph_path(knowledge_dir) if knowledge_dir else None
    if explicit_graph_node_id and graph_path and graph_path.is_file():
        graph_node = _load_node_by_id(graph_path, explicit_graph_node_id)
    elif graph_path and resolved and resolved.is_file():
        graph_node = lookup_official_record_node(repo_root, graph_path, resolved)

    kind = ref.kind
    sha_short = parent.sha256[:16]

    if graph_node:
        nid = graph_node.get("node_id") or graph_node.get("idea_id")
        title = str(graph_node.get("title") or "")
        src = graph_node.get("source") or {}
        track = src.get("track") if isinstance(src, dict) else None
        mm = graph_node.get("measured_metrics") or {}
        bpb = mm.get("val_bpb")
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "n/a"
        identity = (
            f"Knowledge graph OFFICIAL_RECORD “{title}” (node {nid}, {track or 'unknown track'}, "
            f"val_bpb={bpb_s}). Parent bytes are this record's train_gpt.py."
        )
        return {
            "parent_identity": identity,
            "graph_parent_node_id": str(nid) if nid else None,
            "graph_parent_title": title or None,
            "graph_parent_track": str(track) if track else None,
            "graph_parent_val_bpb": float(bpb) if isinstance(bpb, (int, float)) else None,
            "parent_train_gpt_relative": rel,
            "parent_sha256_prefix": sha_short,
            "binding_kind": "official_record",
        }

    if kind == "github":
        identity = (
            f"Upstream parent: {ref.repo_url} @ {ref.git_ref} ({ref.file_path}); "
            f"sha256={parent.sha256[:16]}… — implement as a minimal patch to this file, not a new project."
        )
        return {
            "parent_identity": identity,
            "graph_parent_node_id": None,
            "parent_train_gpt_relative": None,
            "parent_sha256_prefix": sha_short,
            "binding_kind": "github",
        }

    if kind == "run" and ref.run_id:
        identity = (
            f"Prior ideator run parent: run_id={ref.run_id}, file={ref.file_path} "
            f"(sha256={sha_short}…). Extend this artifact; do not swap in unrelated code."
        )
        return {
            "parent_identity": identity,
            "graph_parent_node_id": None,
            "parent_train_gpt_relative": rel,
            "parent_sha256_prefix": sha_short,
            "binding_kind": "run",
        }

    if kind == "local" and rel:
        identity = (
            f"Local parent file `{rel}` (sha256={sha_short}…). "
            "All implementation_steps must be anchored to this file; do not substitute another template."
        )
        return {
            "parent_identity": identity,
            "graph_parent_node_id": None,
            "parent_train_gpt_relative": rel,
            "parent_sha256_prefix": sha_short,
            "binding_kind": "local_file",
        }

    identity = (
        f"Parent code loaded ({kind}, sha256={sha_short}…). "
        "Describe changes as a delta to this parent only."
    )
    return {
        "parent_identity": identity,
        "graph_parent_node_id": None,
        "parent_train_gpt_relative": rel,
        "parent_sha256_prefix": sha_short,
        "binding_kind": kind,
    }


def _load_node_by_id(graph_path: Path, node_id: str) -> dict[str, Any] | None:
    try:
        import json

        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    nodes = data.get("nodes") or {}
    n = nodes.get(node_id)
    return n if isinstance(n, dict) else None


def build_parent_binding_from_path(
    *,
    repo_root: Path,
    parent_py: Path,
    knowledge_dir: Path | None,
    explicit_graph_node_id: str | None = None,
) -> dict[str, Any]:
    """Binding when only a path is known (e.g. live experiment), without a ParentCode object."""
    parent_py = parent_py.resolve()
    if not parent_py.is_file():
        return {
            "parent_identity": f"Configured parent `{parent_py}` is missing — fix IDEATOR_PARENT_TRAIN_GPT or clone parameter-golf.",
            "graph_parent_node_id": None,
            "parent_train_gpt_relative": None,
            "parent_sha256_prefix": None,
            "binding_kind": "missing",
        }
    content = parent_py.read_text(encoding="utf-8", errors="replace")
    sha = sha256_text(content)
    rel = _relpath_or_none(repo_root, parent_py)

    graph_node = None
    graph_path = get_working_graph_path(knowledge_dir) if knowledge_dir else None
    if explicit_graph_node_id and graph_path:
        graph_node = _load_node_by_id(graph_path, explicit_graph_node_id)
    elif graph_path:
        graph_node = lookup_official_record_node(repo_root, graph_path, parent_py)

    if graph_node:
        nid = graph_node.get("node_id") or graph_node.get("idea_id")
        title = str(graph_node.get("title") or "")
        src = graph_node.get("source") or {}
        track = src.get("track") if isinstance(src, dict) else None
        mm = graph_node.get("measured_metrics") or {}
        bpb = mm.get("val_bpb")
        bpb_s = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "n/a"
        identity = (
            f"Knowledge graph OFFICIAL_RECORD “{title}” (node {nid}, {track or 'unknown track'}, "
            f"val_bpb={bpb_s}). You MUST edit this parent file only: `{rel}`."
        )
        return {
            "parent_identity": identity,
            "graph_parent_node_id": str(nid) if nid else None,
            "graph_parent_title": title or None,
            "graph_parent_track": str(track) if track else None,
            "graph_parent_val_bpb": float(bpb) if isinstance(bpb, (int, float)) else None,
            "parent_train_gpt_relative": rel,
            "parent_sha256_prefix": sha[:16],
            "binding_kind": "official_record",
        }

    identity = (
        f"Parent implementation file: `{rel or parent_py}` (sha256={sha[:16]}…). "
        "Propose ONLY a delta against this file; keep Parameter Golf eval constraints."
    )
    return {
        "parent_identity": identity,
        "graph_parent_node_id": None,
        "parent_train_gpt_relative": rel,
        "parent_sha256_prefix": sha[:16],
        "binding_kind": "local_file",
    }


def format_binding_for_prompt(binding: dict[str, Any]) -> str:
    """Short block inserted into long-form (Anthropic) system prompts."""
    ident = binding.get("parent_identity") or ""
    gid = binding.get("graph_parent_node_id")
    lines = [
        ident,
        "",
        "Rules:",
        "- parent_implementation in your JSON MUST name this same parent (repo/git_ref from the binding below, primary_file train_gpt.py).",
        "- implementation_steps must be anchored edits (search strings / line-level deltas), not a greenfield reimplementation.",
        "- Do not point parent_implementation at nanoGPT, Hugging Face demos, or unrelated repos unless this binding explicitly says so.",
    ]
    if gid:
        lines.insert(2, f"- Include graph_parent_node_id: \"{gid}\" inside parent_implementation.")
    return "\n".join(lines)
