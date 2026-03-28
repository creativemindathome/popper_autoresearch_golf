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
    infra_root = repo_root / "infra" / "agents"
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

    buildout_slug = symphony_env["SYMPHONY_LINEAR_PROJECT_SLUG"]
    execution_slug = symphony_env["SYMPHONY_EXECUTION_PROJECT_SLUG"]
    if buildout_slug == execution_slug:
        print("buildout and execution project slugs must differ", file=sys.stderr)
        return 1

    workflow_path = infra_root / "symphony" / "WORKFLOW.hermes.md"
    runtime_path = infra_root / "symphony" / "elixir" / "mix.exs"
    helper_path = infra_root / "scripts" / "lib" / "common.sh"
    issue_map_path = infra_root / "handoffs" / "buildout_issue_map.json"
    sync_back_path = infra_root / "scripts" / "sync_workspace_back.sh"
    archive_path = infra_root / "scripts" / "archive_workspace_changes.sh"
    validate_delivery_path = infra_root / "scripts" / "validate_delivery_paths.sh"
    handoff_drift_path = infra_root / "scripts" / "check_handoff_linear_drift.py"
    missing_paths = [
        str(path)
        for path in (
            workflow_path,
            runtime_path,
            helper_path,
            issue_map_path,
            sync_back_path,
            archive_path,
            validate_delivery_path,
            handoff_drift_path,
        )
        if not path.exists()
    ]
    if missing_paths:
        print(f"missing required symphony repo paths: {', '.join(missing_paths)}", file=sys.stderr)
        return 1

    global_codex_auth = Path.home() / ".codex" / "auth.json"
    if not global_codex_auth.exists():
        print(f"missing Codex auth file: {global_codex_auth}", file=sys.stderr)
        return 1

    print("symphony readiness check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
