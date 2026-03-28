#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re


DEPENDENCY_SECTION = re.compile(r"^## Dependencies\s*$", re.MULTILINE)
STATUS_SECTION = re.compile(r"^## Status\s*$", re.MULTILINE)
HEADING_SECTION = re.compile(r"^##\s+", re.MULTILINE)
VALID_STATES = {"todo", "in_progress", "backlog", "done"}


def extract_dependencies(text: str) -> list[str]:
    match = DEPENDENCY_SECTION.search(text)
    if not match:
        return []
    section_start = match.end()
    next_heading = HEADING_SECTION.search(text, section_start)
    section = text[section_start : next_heading.start() if next_heading else len(text)]
    dependencies: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line in {"- none", "* none", "none"}:
            continue
        normalized = line.lstrip("-* ").strip().strip("`")
        if normalized.endswith(".md"):
            dependencies.append(normalized)
    return dependencies


def extract_status(text: str) -> str | None:
    match = STATUS_SECTION.search(text)
    if not match:
        return None
    section_start = match.end()
    next_heading = HEADING_SECTION.search(text, section_start)
    section = text[section_start : next_heading.start() if next_heading else len(text)]
    for raw_line in section.splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        normalized = line.lstrip("-* ").strip()
        if ":" in normalized:
            _, normalized = normalized.split(":", 1)
            normalized = normalized.strip()
        normalized = normalized.replace(" ", "_")
        if normalized in VALID_STATES:
            return normalized
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    handoff_dir = repo_root / "infra" / "agents" / "handoffs"
    issue_paths = sorted(p for p in handoff_dir.glob("*.md") if p.name != "README.md")
    dependency_map = {path.name: extract_dependencies(path.read_text()) for path in issue_paths}
    status_map = {path.name: extract_status(path.read_text()) for path in issue_paths}
    issue_names = set(dependency_map)
    todo = []
    in_progress = []
    backlog = []
    done = []
    for issue_name, dependencies in dependency_map.items():
        declared_status = status_map.get(issue_name)
        if declared_status == "done":
            done.append(issue_name)
            continue
        if declared_status == "in_progress":
            in_progress.append(issue_name)
            continue
        if declared_status == "todo":
            todo.append(issue_name)
            continue
        if declared_status == "backlog":
            backlog.append(issue_name)
            continue
        filtered = [dependency for dependency in dependencies if dependency in issue_names]
        blocked = [dependency for dependency in filtered if status_map.get(dependency) != "done"]
        if blocked:
            backlog.append(issue_name)
        else:
            todo.append(issue_name)
    queue = {
        "todo": sorted(todo),
        "in_progress": sorted(in_progress),
        "backlog": sorted(backlog),
        "done": sorted(done),
        "dependencies": dependency_map,
        "declared_status": status_map,
        "note": "Items with local markdown dependencies remain in backlog until their prerequisite issue files are considered terminal.",
    }
    print(json.dumps(queue, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
