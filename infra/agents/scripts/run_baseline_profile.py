#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, UTC
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from falsifier.calibration_lite import build_calibration_lite_payload, validate_calibration_lite
from falsifier.stage1.init_aggregates import compute_minimal_init_aggregates
from research.baseline_micro_train import run_micro_train_summary
from research.probe_library import architecture_profile, quantization_profile, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Optional path to a torch checkpoint such as final_model.pt")
    parser.add_argument(
        "--skip-micro-train",
        action="store_true",
        help="Omit micro_train_100_step from calibration_lite (faster; for CI smoke only).",
    )
    parser.add_argument("--micro-train-steps", type=int, default=100, help="Steps for baseline CPU micro-train summary.")
    parser.add_argument("--micro-train-seed", type=int, default=42, help="RNG seed for micro-train.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    out_dir = repo_root / "research" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    architecture = architecture_profile(checkpoint_path=args.checkpoint, repo_root=repo_root)
    quantization = quantization_profile(checkpoint_path=args.checkpoint, repo_root=repo_root)
    micro_train = None
    if not args.skip_micro_train:
        micro_train = run_micro_train_summary(
            repo_root,
            steps=args.micro_train_steps,
            seed=args.micro_train_seed,
        )
    minimal_init = compute_minimal_init_aggregates(repo_root / "train_gpt.py", seed=42)
    calibration_lite = build_calibration_lite_payload(
        architecture,
        quantization,
        micro_train_100_step=micro_train,
        checkpoint_weight_profile={"used_checkpoint": bool(args.checkpoint)},
        minimal_init_baseline=minimal_init,
    )
    ok, reasons = validate_calibration_lite(calibration_lite)
    if not ok:
        print("calibration_lite validation failed:", reasons, file=sys.stderr)
        return 1
    payload = {
        "profile_id": f"baseline-{timestamp}",
        "created_at": timestamp,
        "status": "ok",
        "checkpoint_path": args.checkpoint,
        "architecture": architecture,
        "quantization": quantization,
        "calibration_lite": calibration_lite,
    }
    out_path = out_dir / f"{payload['profile_id']}.json"
    save_json(out_path, payload)
    latest = out_dir / "latest_baseline_profile.json"
    save_json(latest, payload)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
