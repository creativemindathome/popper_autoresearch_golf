from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from falsifier.calibration_lite import (
    extract_calibration_lite_from_profile,
    validate_calibration_lite,
)

# Defaults when no profile or no calibration_lite (local / transient workspaces).
DEFAULT_BASELINE_CONFIG: dict[str, Any] = {
    "vocab_size": 1024,
    "num_layers": 9,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2,
    "tie_embeddings": True,
    "iterations": 20_000,
    "train_batch_tokens": 524_288,
    "train_seq_len": 1024,
}
DEFAULT_ARTIFACT_LIMIT_BYTES = 16_777_216 - 200_000
DEFAULT_MAX_FLOPS_RATIO = 1.5


@dataclass(frozen=True)
class Stage1Thresholds:
    baseline_config: dict[str, Any]
    artifact_limit_bytes: int
    max_flops_ratio: float
    profile_path: Path | None
    source: str  # "calibration_lite" | "defaults"


@lru_cache(maxsize=32)
def load_stage1_thresholds_cached(repo_root_resolved: str) -> Stage1Thresholds:
    repo_root = Path(repo_root_resolved)
    path = repo_root / "research" / "profiles" / "latest_baseline_profile.json"
    if not path.is_file():
        return Stage1Thresholds(
            baseline_config=dict(DEFAULT_BASELINE_CONFIG),
            artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
            max_flops_ratio=DEFAULT_MAX_FLOPS_RATIO,
            profile_path=None,
            source="defaults",
        )
    profile = json.loads(path.read_text())
    cl = extract_calibration_lite_from_profile(profile)
    if cl is None:
        return Stage1Thresholds(
            baseline_config=dict(DEFAULT_BASELINE_CONFIG),
            artifact_limit_bytes=DEFAULT_ARTIFACT_LIMIT_BYTES,
            max_flops_ratio=DEFAULT_MAX_FLOPS_RATIO,
            profile_path=path,
            source="defaults",
        )
    ok, reasons = validate_calibration_lite(cl)
    if not ok:
        raise ValueError("invalid calibration_lite in profile: " + "; ".join(reasons))
    bb = cl["budget_baseline"]
    hp = bb["hyperparameters"]
    baseline_config = {
        "vocab_size": int(hp["vocab_size"]),
        "num_layers": int(hp["num_layers"]),
        "model_dim": int(hp["model_dim"]),
        "num_heads": int(hp["num_heads"]),
        "num_kv_heads": int(hp["num_kv_heads"]),
        "mlp_mult": int(hp["mlp_mult"]),
        "tie_embeddings": bool(hp.get("tie_embeddings", True)),
        "iterations": int(hp.get("iterations", 20_000)),
        "train_batch_tokens": int(hp["train_batch_tokens"]),
        "train_seq_len": int(hp["train_seq_len"]),
    }
    return Stage1Thresholds(
        baseline_config=baseline_config,
        artifact_limit_bytes=int(bb["artifact_limit_bytes"]),
        max_flops_ratio=float(bb["max_flops_ratio_vs_baseline"]),
        profile_path=path,
        source="calibration_lite",
    )


def load_stage1_thresholds(repo_root: Path) -> Stage1Thresholds:
    return load_stage1_thresholds_cached(str(repo_root.resolve()))
