#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--reopen-missing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise SystemExit("repo root does not exist")

    print(f"delivery reconciliation stub for {repo_root}")
    if args.reopen_missing:
        print("reopen_missing requested; implement Linear reconciliation here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
