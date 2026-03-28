from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from falsifier.utils.config_parser import count_parameters, estimate_flops

CALIBRATION_LITE_SCHEMA_VERSION = "1"

# Mirrors stage-1 static budget gate defaults; consumed by threshold loading (wave 2).
DEFAULT_ARTIFACT_LIMIT_BYTES = 16_777_216 - 200_000
DEFAULT_MAX_FLOPS_RATIO = 1.5


def _budget_from_hyperparameters(hp: dict[str, Any]) -> dict[str, Any]:
    config = {
        "vocab_size": int(hp["vocab_size"]),
        "num_layers": int(hp["num_layers"]),
        "model_dim": int(hp["model_dim"]),
        "num_heads": int(hp["num_heads"]),
        "num_kv_heads": int(hp["num_kv_heads"]),
        "mlp_mult": int(hp["mlp_mult"]),
        "tie_embeddings": bool(hp.get("tie_embeddings", True)),
        "iterations": int(hp.get("iterations", 20_000)),
        "train_batch_tokens": int(hp.get("train_batch_tokens", 524_288)),
        "train_seq_len": int(hp.get("train_seq_len", 1024)),
    }
    param_counts = count_parameters(config)
    total_params = sum(param_counts.values())
    flops = estimate_flops(config)
    return {
        "hyperparameters": config,
        "param_count_estimate": total_params,
        "param_count_by_component": {k: int(v) for k, v in param_counts.items()},
        "flops_estimate_per_step": flops,
        "artifact_limit_bytes": DEFAULT_ARTIFACT_LIMIT_BYTES,
        "max_flops_ratio_vs_baseline": DEFAULT_MAX_FLOPS_RATIO,
    }


def _random_init_baseline(architecture: dict[str, Any]) -> dict[str, Any]:
    wk = architecture.get("weight_kurtosis") or {}
    er = architecture.get("effective_rank") or {}
    kvals = [float(v) for v in wk.values() if isinstance(v, (int, float))]
    rvals = [float(v) for v in er.values() if isinstance(v, (int, float))]
    return {
        "weight_kurtosis_mean": sum(kvals) / len(kvals) if kvals else 0.0,
        "weight_kurtosis_max": max(kvals) if kvals else 0.0,
        "effective_rank_mean": sum(rvals) / len(rvals) if rvals else 0.0,
        "effective_rank_max": max(rvals) if rvals else 0.0,
        "tensor_count": len(architecture.get("tensor_stats") or {}),
    }


def build_calibration_lite_payload(
    architecture: dict[str, Any],
    quantization: dict[str, Any],
    micro_train_100_step: dict[str, Any] | None,
    checkpoint_weight_profile: dict[str, Any] | None = None,
    minimal_init_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hp = architecture["hyperparameters"]
    out: dict[str, Any] = {
        "schema_version": CALIBRATION_LITE_SCHEMA_VERSION,
        "budget_baseline": _budget_from_hyperparameters(hp),
        "random_init_baseline": _random_init_baseline(architecture),
        "quantization_mse_baseline": {
            "by_group": (quantization.get("quantization_mse") or {}).get("by_group", {}),
        },
    }
    if micro_train_100_step is not None:
        out["micro_train_100_step"] = micro_train_100_step
    if checkpoint_weight_profile is not None:
        out["checkpoint_weight_profile"] = checkpoint_weight_profile
    if minimal_init_baseline is not None:
        out["minimal_init_baseline"] = minimal_init_baseline
    return out


def validate_calibration_lite(obj: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(obj, dict):
        return False, ["calibration_lite must be an object"]
    if obj.get("schema_version") != CALIBRATION_LITE_SCHEMA_VERSION:
        reasons.append(
            f"unsupported calibration_lite.schema_version: {obj.get('schema_version')!r}; "
            f"expected {CALIBRATION_LITE_SCHEMA_VERSION!r}"
        )
    bb = obj.get("budget_baseline")
    if not isinstance(bb, dict):
        reasons.append("calibration_lite.budget_baseline missing or not an object")
    else:
        for key in ("flops_estimate_per_step", "artifact_limit_bytes", "max_flops_ratio_vs_baseline"):
            if key not in bb:
                reasons.append(f"calibration_lite.budget_baseline.{key} missing")
    ri = obj.get("random_init_baseline")
    if not isinstance(ri, dict):
        reasons.append("calibration_lite.random_init_baseline missing or not an object")
    return (len(reasons) == 0, reasons)


def load_profile_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_calibration_lite_from_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    cl = profile.get("calibration_lite")
    if cl is None:
        return None
    if isinstance(cl, dict):
        return cl
    return None
