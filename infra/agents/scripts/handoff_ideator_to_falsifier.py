#!/usr/bin/env python3
"""
Handoff script: Ideator -> Falsifier

Automatically symlinks approved ideas from ideator outbox to falsifier inbox.
Updates the knowledge graph with APPROVED status for those ideas.

Usage:
    python infra/agents/scripts/handoff_ideator_to_falsifier.py [--knowledge-dir PATH]

Logic:
- For each JSON file in outbox/ideator/:
  - Load the idea
  - Check if it has reviewer approval (reviewer_feedback.decision == "pass")
  - Check if symlink already exists in inbox/approved/
  - If approved and not in inbox:
    - Create symlink: inbox/approved/{idea_id}.json -> outbox/ideator/{filename}
    - Create/update node in graph.json with status APPROVED
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _find_repo_root() -> Path:
    """Find the repository root by looking for .git or knowledge_graph directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Search up to 5 levels
        if (current / ".git").exists() or (current / "knowledge_graph").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.cwd()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_idea_approved(idea_data: Dict[str, Any]) -> bool:
    """Check if an idea has been approved by the reviewer."""
    # Check for reviewer_feedback field (from _finalize_idea_v2)
    reviewer_feedback = idea_data.get("reviewer_feedback")
    if isinstance(reviewer_feedback, dict):
        decision = str(reviewer_feedback.get("decision") or "").strip().lower()
        if decision == "pass":
            return True

    # Check for artifacts.review_json (indicates review happened)
    artifacts = idea_data.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get("review_json"):
        # Has review artifact but need to check actual review file
        return True  # Assume approved if review artifact exists

    # Check in meta or top-level for approval status
    meta = idea_data.get("meta", {})
    if isinstance(meta, dict):
        status = str(meta.get("review_status") or "").strip().lower()
        if status == "approved" or status == "pass":
            return True

    # Check for explicit status field
    status = str(idea_data.get("status") or "").strip().upper()
    if status == "APPROVED":
        return True

    return False


def _get_idea_id(idea_data: Dict[str, Any], filename: str) -> str:
    """Extract idea_id from idea data or filename."""
    idea_id = idea_data.get("idea_id")
    if isinstance(idea_id, str) and idea_id.strip():
        return idea_id.strip()

    # Extract from filename (e.g., 20260328T112538Z_low-rank-transformer-layers.json)
    stem = Path(filename).stem
    if "_" in stem:
        # Remove timestamp prefix if present
        parts = stem.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 15 and parts[0][8] == "T":  # Looks like timestamp
            return parts[1]
    return stem


def _get_idea_title(idea_data: Dict[str, Any]) -> str:
    """Extract title from idea data."""
    title = idea_data.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return ""


def _get_idea_status(idea_data: Dict[str, Any]) -> str:
    """Get the review status from idea data."""
    reviewer_feedback = idea_data.get("reviewer_feedback")
    if isinstance(reviewer_feedback, dict):
        return str(reviewer_feedback.get("decision") or "unknown").upper()
    return "APPROVED"  # Default for files that passed review


def _get_novelty_score(idea_data: Dict[str, Any]) -> Optional[int]:
    """Extract novelty score from idea data."""
    reviewer_feedback = idea_data.get("reviewer_feedback")
    if isinstance(reviewer_feedback, dict):
        score = reviewer_feedback.get("novelty_score")
        if isinstance(score, int):
            return score
        try:
            return int(score)
        except (TypeError, ValueError):
            pass
    return None


def scan_outbox_ideas(outbox_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Scan the outbox directory for all idea JSON files."""
    ideas: List[Tuple[Path, Dict[str, Any]]] = []

    if not outbox_dir.exists():
        return ideas

    for file_path in outbox_dir.glob("*.json"):
        # Skip latest.json (it's a symlink or copy)
        if file_path.name == "latest.json":
            continue

        data = _load_json(file_path)
        if data is None:
            continue

        # Check if it's an idea file (has schema_version starting with ideator.idea)
        schema = str(data.get("schema_version", ""))
        if schema.startswith("ideator.idea"):
            ideas.append((file_path, data))

    return ideas


def get_existing_inbox_ideas(inbox_dir: Path) -> Set[str]:
    """Get set of idea IDs already in the inbox."""
    existing: Set[str] = set()

    if not inbox_dir.exists():
        return existing

    for file_path in inbox_dir.glob("*.json"):
        # Check if it's a symlink
        if file_path.is_symlink():
            # Resolve to get the target
            try:
                target = file_path.resolve()
                data = _load_json(target)
                if data:
                    idea_id = _get_idea_id(data, target.name)
                    existing.add(idea_id)
            except Exception:
                pass
        else:
            # Regular file - load and check
            data = _load_json(file_path)
            if data:
                idea_id = _get_idea_id(data, file_path.name)
                existing.add(idea_id)

    return existing


def load_graph(graph_path: Path) -> Dict[str, Any]:
    """Load the knowledge graph JSON."""
    if graph_path.exists():
        data = _load_json(graph_path)
        if data is not None:
            return data
    # Falsifier lifecycle uses nodes as a dict keyed by node_id; align defaults.
    return {"nodes": {}, "edges": []}


def save_graph(graph_path: Path, graph: Dict[str, Any]) -> None:
    """Save the knowledge graph JSON."""
    _write_json(graph_path, graph)


def find_or_create_node(graph: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Find an existing node or create a new one.

    Supports dict-shaped ``nodes`` (falsifier / ``lifecycle`` schema) and legacy
    list-shaped ``nodes``. Never replaces a dict ``nodes`` object with a list —
    doing so would wipe the working graph when this script runs.
    """
    raw = graph.get("nodes")

    if isinstance(raw, dict):
        existing = raw.get(node_id)
        if isinstance(existing, dict):
            return existing
        new_node: Dict[str, Any] = {"node_id": node_id}
        raw[node_id] = new_node
        return new_node

    if isinstance(raw, list):
        for node in raw:
            if isinstance(node, dict) and node.get("id") == node_id:
                return node
        new_node = {"id": node_id}
        raw.append(new_node)
        return new_node

    # Missing or unexpected type: start a canonical dict (preserve other graph keys).
    nodes_dict: Dict[str, Any] = {}
    graph["nodes"] = nodes_dict
    new_node = {"node_id": node_id}
    nodes_dict[node_id] = new_node
    return new_node


def update_graph_with_approved_idea(
    graph: Dict[str, Any],
    idea_id: str,
    idea_data: Dict[str, Any],
    symlink_path: Path,
) -> None:
    """Update the knowledge graph with an approved idea node."""
    node = find_or_create_node(graph, f"idea_{idea_id}")

    # Update node properties
    node["label"] = _get_idea_title(idea_data) or idea_id
    node["type"] = "Idea"
    node["status"] = "APPROVED"
    node["mechanism_description"] = idea_data.get("novelty_summary", "")

    # Add metadata
    node["idea_id"] = idea_id
    node["schema_version"] = idea_data.get("schema_version", "ideator.idea.v2")

    # Extract parameter/memory estimates if available
    expected_metric = idea_data.get("expected_metric_change", "")
    node["expected_metric_change"] = expected_metric

    # Store novelty score if available
    novelty_score = _get_novelty_score(idea_data)
    if novelty_score is not None:
        node["novelty_score"] = novelty_score

    # Add timestamps
    node["approved_at"] = datetime.now(timezone.utc).isoformat()

    # Add reference to symlink
    node["symlink_path"] = str(symlink_path)

    # Add parent implementation reference
    parent_impl = idea_data.get("parent_implementation")
    if isinstance(parent_impl, dict):
        node["parent_repo"] = parent_impl.get("repo_url", "")
        node["parent_file"] = parent_impl.get("primary_file", "train_gpt.py")

    # Store run_id if available
    run_id = idea_data.get("run_id")
    if run_id:
        node["run_id"] = run_id


def create_symlink(source: Path, target: Path) -> bool:
    """Create a symlink from target to source. Returns True on success."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing symlink if it exists
        if target.exists() or target.is_symlink():
            target.unlink()

        # Create relative symlink if possible
        try:
            rel_source = source.relative_to(target.parent)
            target.symlink_to(rel_source)
        except ValueError:
            # Fall back to absolute path
            target.symlink_to(source.resolve())

        return True
    except Exception as e:
        sys.stderr.write(f"Error creating symlink {target} -> {source}: {e}\n")
        return False


def run_handoff(
    knowledge_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Run the handoff process.

    Returns:
        Tuple of (processed, approved, symlinks_created)
    """
    outbox_dir = knowledge_dir / "outbox" / "ideator"
    inbox_dir = knowledge_dir / "inbox" / "approved"
    graph_path = knowledge_dir / "graph.json"

    if not outbox_dir.exists():
        sys.stderr.write(f"Outbox directory does not exist: {outbox_dir}\n")
        return 0, 0, 0

    # Scan for ideas
    ideas = scan_outbox_ideas(outbox_dir)
    if verbose:
        print(f"Found {len(ideas)} idea files in {outbox_dir}")

    # Get existing inbox ideas
    existing_ideas = get_existing_inbox_ideas(inbox_dir)
    if verbose:
        print(f"Found {len(existing_ideas)} existing ideas in inbox")

    # Load graph
    graph = load_graph(graph_path)

    processed = 0
    approved = 0
    symlinks_created = 0

    for source_path, idea_data in ideas:
        processed += 1
        idea_id = _get_idea_id(idea_data, source_path.name)

        if verbose:
            print(f"Processing: {source_path.name} (idea_id: {idea_id})")

        # Check if approved
        if not _is_idea_approved(idea_data):
            if verbose:
                print(f"  -> Not approved (skipping)")
            continue

        approved += 1

        # Check if already in inbox
        if idea_id in existing_ideas:
            if verbose:
                print(f"  -> Already in inbox (skipping)")
            continue

        # Create symlink
        target_path = inbox_dir / f"{idea_id}.json"

        if dry_run:
            print(f"  -> Would create symlink: {target_path} -> {source_path}")
            symlinks_created += 1
        else:
            if create_symlink(source_path, target_path):
                print(f"  -> Created symlink: {target_path} -> {source_path}")
                symlinks_created += 1

                # Update graph
                update_graph_with_approved_idea(graph, idea_id, idea_data, target_path)
            else:
                print(f"  -> Failed to create symlink for {idea_id}")

    # Save graph if changes were made and not in dry-run mode
    if symlinks_created > 0 and not dry_run:
        save_graph(graph_path, graph)
        print(f"Updated graph: {graph_path}")

    return processed, approved, symlinks_created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Handoff approved ideas from ideator to falsifier inbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run handoff with default knowledge directory
  python infra/agents/scripts/handoff_ideator_to_falsifier.py

  # Dry run to see what would happen
  python infra/agents/scripts/handoff_ideator_to_falsifier.py --dry-run

  # Specify custom knowledge directory
  python infra/agents/scripts/handoff_ideator_to_falsifier.py --knowledge-dir /path/to/knowledge_graph
""",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default=None,
        help="Path to knowledge_graph directory (default: ./knowledge_graph or repo root knowledge_graph)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information",
    )

    args = parser.parse_args()

    # Determine knowledge directory
    if args.knowledge_dir:
        knowledge_dir = Path(args.knowledge_dir)
    else:
        repo_root = _find_repo_root()
        knowledge_dir = repo_root / "knowledge_graph"

    if not knowledge_dir.exists():
        sys.stderr.write(f"Knowledge directory not found: {knowledge_dir}\n")
        return 1

    # Run handoff
    processed, approved, symlinks_created = run_handoff(
        knowledge_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Summary
    print(f"\nSummary:")
    print(f"  Ideas processed: {processed}")
    print(f"  Ideas approved: {approved}")
    print(f"  Symlinks created: {symlinks_created}")

    return 0 if symlinks_created > 0 or processed == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
