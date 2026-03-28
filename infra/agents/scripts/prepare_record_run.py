#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track_10min_16mb")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = repo_root / "records" / args.track / f"{datetime.now().date()}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    readme = run_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Candidate Run\n\n"
            f"- Track: `{args.track}`\n"
            f"- Name: `{args.name}`\n"
            "- Fill in architecture, artifact accounting, logs, and reproduction steps.\n"
        )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
