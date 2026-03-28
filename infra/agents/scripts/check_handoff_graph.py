#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


REQUIRED_HEADINGS = [
    "## Goal",
    "## Context",
    "## Scope",
    "## Acceptance Criteria",
    "## Validation",
    "## Delivery",
    "## Dependencies",
    "## Ownership / Non-Overlap",
    "## Constraints",
    "## References",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    handoff_dir = repo_root / "infra" / "agents" / "handoffs"
    issue_files = sorted(p for p in handoff_dir.glob("*.md") if p.name != "README.md")
    failures: list[str] = []

    for path in issue_files:
        text = path.read_text()
        missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing:
            failures.append(f"{path}: missing headings: {', '.join(missing)}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"handoff graph ok ({len(issue_files)} issue files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
