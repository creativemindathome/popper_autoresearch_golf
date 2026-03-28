#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, UTC
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.probe_library import architecture_profile, quantization_profile, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Optional path to a torch checkpoint such as final_model.pt")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    out_dir = repo_root / "research" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    architecture = architecture_profile(checkpoint_path=args.checkpoint, repo_root=repo_root)
    quantization = quantization_profile(checkpoint_path=args.checkpoint, repo_root=repo_root)
    payload = {
        "profile_id": f"baseline-{timestamp}",
        "created_at": timestamp,
        "status": "ok",
        "checkpoint_path": args.checkpoint,
        "architecture": architecture,
        "quantization": quantization,
    }
    out_path = out_dir / f"{payload['profile_id']}.json"
    save_json(out_path, payload)
    latest = out_dir / "latest_baseline_profile.json"
    save_json(latest, payload)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
