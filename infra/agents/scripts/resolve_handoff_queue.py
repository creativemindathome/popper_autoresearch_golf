#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    handoff_dir = repo_root / "infra" / "agents" / "handoffs"
    issues = sorted(p.name for p in handoff_dir.glob("*.md") if p.name != "README.md")
    queue = {
        "todo": issues,
        "backlog": [],
        "note": "Dependency parsing is not implemented yet; all checked-in issue files are surfaced as todo candidates.",
    }
    print(json.dumps(queue, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
