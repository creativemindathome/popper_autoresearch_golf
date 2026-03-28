from pathlib import Path

from falsifier.calibration_lite import (
    build_calibration_lite_payload,
    extract_calibration_lite_from_profile,
    validate_calibration_lite,
)


def test_validate_calibration_lite_accepts_built_payload():
    arch = {
        "hyperparameters": {
            "vocab_size": 64,
            "num_layers": 2,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_mult": 2,
            "tie_embeddings": True,
            "iterations": 1,
            "train_batch_tokens": 8,
            "train_seq_len": 8,
        },
        "weight_kurtosis": {"a": 1.0},
        "effective_rank": {"a": 10.0},
        "tensor_stats": {"a": {}},
    }
    quant = {"quantization_mse": {"by_group": {"attention": 1e-6}}}
    cl = build_calibration_lite_payload(arch, quant, micro_train_100_step={"steps": 1, "loss_last": 1.0})
    ok, reasons = validate_calibration_lite(cl)
    assert ok, reasons


def test_extract_from_profile_roundtrip():
    profile = {
        "calibration_lite": {
            "schema_version": "1",
            "budget_baseline": {
                "flops_estimate_per_step": 1.0,
                "artifact_limit_bytes": 100,
                "max_flops_ratio_vs_baseline": 1.5,
            },
            "random_init_baseline": {},
        }
    }
    assert extract_calibration_lite_from_profile(profile) is not None


def test_latest_profile_has_calibration_lite_when_present():
    root = Path(__file__).resolve().parents[2]
    latest = root / "research" / "profiles" / "latest_baseline_profile.json"
    if not latest.exists():
        return
    import json

    profile = json.loads(latest.read_text())
    cl = extract_calibration_lite_from_profile(profile)
    if cl is None:
        return
    ok, _ = validate_calibration_lite(cl)
    assert ok
