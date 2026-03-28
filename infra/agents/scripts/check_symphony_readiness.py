#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    symphony_env = parse_env(repo_root / "infra" / "agents" / "env" / ".env.symphony")
    required = [
        "LINEAR_API_KEY",
        "SYMPHONY_LINEAR_PROJECT_SLUG",
        "SYMPHONY_EXECUTION_PROJECT_SLUG",
        "SOURCE_REPO_URL",
        "SYMPHONY_CODEX_COMMAND",
    ]
    missing = [key for key in required if not symphony_env.get(key)]

    if missing:
        print(f"missing required symphony env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("symphony readiness check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
