from __future__ import annotations

import json
from pathlib import Path

from falsifier.calibration_lite import extract_calibration_lite_from_profile
from falsifier.types import CandidatePackage, InitGateDiagnostics

from .init_aggregates import compute_minimal_init_aggregates, within_band

T5_INIT = "T5"


def _load_minimal_init_baseline(repo_root: Path) -> dict | None:
    path = repo_root / "research" / "profiles" / "latest_baseline_profile.json"
    if not path.is_file():
        return None
    profile = json.loads(path.read_text())
    cl = extract_calibration_lite_from_profile(profile)
    if not cl:
        return None
    mi = cl.get("minimal_init_baseline")
    return mi if isinstance(mi, dict) else None


def evaluate_init_gate(
    candidate: CandidatePackage,
    repo_root: Path,
    *,
    seed: int = 42,
) -> tuple[InitGateDiagnostics, bool, list[str]]:
    """
    T5: compare minimal-model init statistics to calibration `minimal_init_baseline`.
    If baseline is missing, pass (no gate).
    """
    baseline = _load_minimal_init_baseline(repo_root)
    cand = compute_minimal_init_aggregates(
        candidate.train_gpt_path,
        env_overrides=candidate.env_overrides,
        seed=seed,
    )
    reasons: list[str] = []
    if baseline is None:
        diag = InitGateDiagnostics(
            candidate_kurtosis_mean=float(cand["weight_kurtosis_mean"]),
            candidate_effective_rank_mean=float(cand["effective_rank_mean"]),
            baseline_kurtosis_mean=None,
            baseline_effective_rank_mean=None,
            ok=True,
            skipped=True,
        )
        return diag, True, []

    bk = float(baseline.get("weight_kurtosis_mean", 0.0))
    br = float(baseline.get("effective_rank_mean", 0.0))
    ck = float(cand["weight_kurtosis_mean"])
    cr = float(cand["effective_rank_mean"])

    ok_k = within_band(ck, bk)
    ok_r = within_band(cr, br)
    ok = ok_k and ok_r
    if not ok_k:
        reasons.append(
            f"T5 init kurtosis mean {ck:.6f} outside log-band vs baseline {bk:.6f}"
        )
    if not ok_r:
        reasons.append(
            f"T5 init effective_rank mean {cr:.6f} outside log-band vs baseline {br:.6f}"
        )

    diag = InitGateDiagnostics(
        candidate_kurtosis_mean=ck,
        candidate_effective_rank_mean=cr,
        baseline_kurtosis_mean=bk,
        baseline_effective_rank_mean=br,
        ok=ok,
        skipped=False,
    )
    return diag, ok, reasons
