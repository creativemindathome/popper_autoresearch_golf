from __future__ import annotations

import json
import math
from pathlib import Path

from falsifier.calibration_lite import extract_calibration_lite_from_profile
from falsifier.types import CandidatePackage, MicroTrainDiagnostics
from research.baseline_micro_train import run_micro_train_summary

T4_MICRO = "T4"


def _baseline_micro_train(repo_root: Path) -> dict | None:
    path = repo_root / "research" / "profiles" / "latest_baseline_profile.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text())
    cl = extract_calibration_lite_from_profile(profile)
    if not cl:
        return None
    mt = cl.get("micro_train_100_step")
    return mt if isinstance(mt, dict) else None


def evaluate_micro_train(
    candidate: CandidatePackage,
    repo_root: Path,
    *,
    steps: int = 100,
    seed: int = 42,
) -> tuple[MicroTrainDiagnostics, bool, list[str]]:
    raw = run_micro_train_summary(
        repo_root,
        train_gpt_path=Path(candidate.train_gpt_path),
        steps=steps,
        seed=seed,
    )
    loss_first = float(raw["loss_first"])
    loss_last = float(raw["loss_last"])
    loss_drop = float(raw["loss_drop"])
    reasons: list[str] = []
    if not all(math.isfinite(x) for x in (loss_first, loss_last, loss_drop)):
        diag = MicroTrainDiagnostics(
            steps=int(raw["steps"]),
            loss_first=loss_first,
            loss_last=loss_last,
            loss_drop=loss_drop,
            throughput_steps_per_sec=float(raw["throughput_steps_per_sec"]),
            ok=False,
        )
        return diag, False, ["non-finite loss during micro-train"]

    baseline_mt = _baseline_micro_train(repo_root)
    ok = True
    if baseline_mt is not None:
        base_drop = float(baseline_mt.get("loss_drop", 0.0))
        floor = max(1e-12, base_drop * 0.05)
        if loss_drop < floor:
            ok = False
            reasons.append(
                f"micro-train loss_drop {loss_drop:.6f} below calibration floor {floor:.6f} (5% of baseline drop)"
            )
    else:
        if loss_drop <= 0.0:
            ok = False
            reasons.append("micro-train did not reduce loss (no calibration baseline for relative check)")

    diag = MicroTrainDiagnostics(
        steps=int(raw["steps"]),
        loss_first=loss_first,
        loss_last=loss_last,
        loss_drop=loss_drop,
        throughput_steps_per_sec=float(raw["throughput_steps_per_sec"]),
        ok=ok,
    )
    return diag, ok, reasons
