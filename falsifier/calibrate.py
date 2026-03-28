"""
Calibration runner: produce Calibration dataclass with Baseline100.

Run once per SOTA change. On Mac/CPU, uses minimal-env PyTorch models.
With MLX, uses full model for faster training.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import Baseline100, Calibration

# Default paths
DEFAULT_PROFILE_DIR = Path("research/profiles")
DEFAULT_PROFILE_NAME = "latest_baseline_profile.json"


def calibrate(
    train_gpt_path: Path | str,
    sota_checkpoint_path: Path | str | None = None,
    val_data_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Calibration:
    """Produce all calibration data. Run once per SOTA change.
    
    Args:
        train_gpt_path: Path to train_gpt.py (PyTorch version)
        sota_checkpoint_path: Optional path to trained SOTA checkpoint
        val_data_path: Optional path to FineWeb validation data
        output_path: Where to save the calibration profile
    
    Returns:
        Calibration dataclass with baseline measurements
    """
    repo_root = Path(train_gpt_path).resolve().parent
    
    print(f"[calibrate] Starting calibration for {train_gpt_path}")
    
    # Try to load checkpoint profile if available
    checkpoint_profile = _load_checkpoint_profile(sota_checkpoint_path) if sota_checkpoint_path else {}
    
    # ══ 100-step baseline (5 seeds) ═══════════════════════════════════════════
    print("[calibrate] Running 100-step baseline (5 seeds)...")
    loss_drops: list[float] = []
    loss_1s: list[float] = []
    loss_100s: list[float] = []
    grad_norm_means: list[float] = []
    grad_norm_stds: list[float] = []
    tokens_per_sec: list[float] = []
    
    for seed in range(5):
        print(f"[calibrate] Seed {seed}...")
        metrics = _run_micro_train(repo_root, train_gpt_path, steps=100, seed=seed)
        
        loss_drops.append(metrics["loss_drop"])
        loss_1s.append(metrics["loss_first"])
        loss_100s.append(metrics["loss_last"])
        grad_norm_means.append(metrics["grad_norm_mean"])
        grad_norm_stds.append(metrics["grad_norm_std"])
        tokens_per_sec.append(metrics["tokens_per_second"])
    
    # Compute statistics
    ld_mean = sum(loss_drops) / len(loss_drops)
    ld_std = math.sqrt(sum((d - ld_mean) ** 2 for d in loss_drops) / len(loss_drops))
    
    baseline_100 = Baseline100(
        loss_drop_mean=ld_mean,
        loss_drop_std=ld_std,
        loss_at_1_mean=sum(loss_1s) / len(loss_1s),
        loss_at_100_mean=sum(loss_100s) / len(loss_100s),
        gradient_norm_mean=sum(grad_norm_means) / len(grad_norm_means),
        gradient_norm_std=sum(grad_norm_stds) / len(grad_norm_stds),
        tokens_per_second_mean=sum(tokens_per_sec) / len(tokens_per_sec),
        # Derived thresholds per PRD
        learning_ratio_kill=max(0.15, 1.0 - 3.0 * ld_std / ld_mean) if ld_mean > 0 else 0.15,
        learning_ratio_tag=max(0.30, 1.0 - 2.0 * ld_std / ld_mean) if ld_mean > 0 else 0.30,
    )
    
    # ══ 500-step baseline (1 seed) ════════════════════════════════════════════
    print("[calibrate] Running 500-step baseline (seed 42)...")
    metrics_500 = _run_micro_train(repo_root, train_gpt_path, steps=500, seed=42)
    baseline_100.loss_at_500 = metrics_500["loss_last"]
    baseline_100.loss_drop_500_mean = metrics_500["loss_drop"]
    
    # Build Calibration
    cal = Calibration(
        baseline_100=baseline_100,
        # Use checkpoint values if available, else defaults
        sota_layer_gradient_norms=checkpoint_profile.get("layer_gradient_norms", {}),
        sota_layer_activation_norms=checkpoint_profile.get("layer_activation_norms", {}),
        sota_attention_entropy_per_head=checkpoint_profile.get("attention_entropy", {}),
        sota_component_gradient_norms=checkpoint_profile.get("component_gradient_norms", {}),
        sota_mlp_effective_ranks=checkpoint_profile.get("mlp_effective_ranks", {}),
        sota_attn_effective_ranks=checkpoint_profile.get("attn_effective_ranks", {}),
        sota_init_logit_max=checkpoint_profile.get("init_logit_max", 10.0),
        sota_gradient_norm_ratio=checkpoint_profile.get("gradient_norm_ratio", 10.0),
        sota_output_entropy=checkpoint_profile.get("output_entropy", 0.0),
        sota_param_count=checkpoint_profile.get("param_count", 0),
        sota_artifact_bytes=checkpoint_profile.get("artifact_bytes", 0),
        sota_tokens_per_second=baseline_100.tokens_per_second_mean,
    )
    
    # Save to profile
    if output_path:
        save_path = Path(output_path)
    else:
        save_path = repo_root / DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build profile JSON
    profile = _build_profile_json(cal, train_gpt_path)
    save_path.write_text(json.dumps(profile, indent=2, default=str))
    print(f"[calibrate] Saved profile to {save_path}")
    
    return cal


def _run_micro_train(
    repo_root: Path,
    train_gpt_path: Path | str,
    steps: int,
    seed: int,
) -> dict[str, float]:
    """Run micro-training and return metrics."""
    # Try MLX first if available
    try:
        from .adapters.mlx_adapter import mlx_available, run_mlx_micro_train_summary
        
        if mlx_available():
            result = run_mlx_micro_train_summary(
                repo_root,
                config_overrides={},
                steps=steps,
                seed=seed,
            )
            return {
                "loss_first": result["loss_first"],
                "loss_last": result["loss_last"],
                "loss_drop": result["loss_drop"],
                "grad_norm_mean": result.get("grad_norm_mean", 0.0),
                "grad_norm_std": result.get("grad_norm_std", 0.0),
                "tokens_per_second": result["throughput_steps_per_sec"],
            }
    except Exception as e:
        print(f"[calibrate] MLX failed, falling back to PyTorch: {e}")
    
    # Fall back to PyTorch minimal-env
    from research.baseline_micro_train import run_micro_train_summary
    
    result = run_micro_train_summary(
        repo_root,
        train_gpt_path=Path(train_gpt_path),
        steps=steps,
        seed=seed,
    )
    
    # Calculate gradient stats if available
    grad_norm_mean = 0.0
    grad_norm_std = 0.0
    if "gradient_norms" in result:
        gns = result["gradient_norms"]
        if gns:
            grad_norm_mean = sum(gns) / len(gns)
            grad_norm_std = math.sqrt(sum((g - grad_norm_mean) ** 2 for g in gns) / len(gns))
    
    return {
        "loss_first": float(result["loss_first"]),
        "loss_last": float(result["loss_last"]),
        "loss_drop": float(result["loss_drop"]),
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "tokens_per_second": float(result["throughput_steps_per_sec"]),
    }


def _load_checkpoint_profile(checkpoint_path: Path | str | None) -> dict[str, Any]:
    """Load profile from checkpoint if available."""
    if not checkpoint_path:
        return {}
    
    # This would need actual checkpoint loading logic
    # For now, return empty dict (graceful degradation)
    return {}


def _build_profile_json(cal: Calibration, train_gpt_path: Path | str) -> dict[str, Any]:
    """Build the JSON profile format."""
    from .calibration_lite import build_calibration_lite_payload
    
    # Get architecture info
    from .utils.config_parser import extract_model_config, count_parameters
    
    source = Path(train_gpt_path).read_text()
    config = extract_model_config(source)
    param_counts = count_parameters(config)
    
    # Build calibration-lite payload
    calibration_lite = build_calibration_lite_payload(
        architecture={
            "hyperparameters": config,
            "param_count_by_component": param_counts,
            "param_count_estimate": sum(param_counts.values()),
        },
        quantization={"scheme": "none"},
        micro_train_100_step={
            "loss_drop": cal.baseline_100.loss_drop_mean,
            "loss_first": cal.baseline_100.loss_at_1_mean,
            "loss_last": cal.baseline_100.loss_at_100_mean,
            "throughput_steps_per_sec": cal.baseline_100.tokens_per_second_mean,
            "gradient_norm_mean": cal.baseline_100.gradient_norm_mean,
            "gradient_norm_std": cal.baseline_100.gradient_norm_std,
        },
        checkpoint_weight_profile=None,
        minimal_init_baseline=None,
    )
    
    # Add the full calibration data
    return {
        "calibration_lite": calibration_lite,
        "full_calibration": {
            "baseline_100": asdict(cal.baseline_100),
            "sota_layer_gradient_norms": cal.sota_layer_gradient_norms,
            "sota_layer_activation_norms": cal.sota_layer_activation_norms,
            "sota_attention_entropy_per_head": cal.sota_attention_entropy_per_head,
            "sota_component_gradient_norms": cal.sota_component_gradient_norms,
            "sota_init_logit_max": cal.sota_init_logit_max,
            "sota_gradient_norm_ratio": cal.sota_gradient_norm_ratio,
            "sota_output_entropy": cal.sota_output_entropy,
        },
    }


def load_calibration(repo_root: Path | str) -> Calibration:
    """Load calibration from profile or return defaults."""
    profile_path = Path(repo_root) / DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME
    
    if not profile_path.exists():
        print(f"[calibrate] No profile at {profile_path}, using defaults")
        return Calibration()
    
    try:
        profile = json.loads(profile_path.read_text())
        fc = profile.get("full_calibration", {})
        bb = fc.get("baseline_100", {})
        
        return Calibration(
            baseline_100=Baseline100(
                loss_drop_mean=bb.get("loss_drop_mean", 0.0),
                loss_drop_std=bb.get("loss_drop_std", 0.0),
                loss_at_1_mean=bb.get("loss_at_1_mean", 0.0),
                loss_at_100_mean=bb.get("loss_at_100_mean", 0.0),
                gradient_norm_mean=bb.get("gradient_norm_mean", 0.0),
                gradient_norm_std=bb.get("gradient_norm_std", 0.0),
                tokens_per_second_mean=bb.get("tokens_per_second_mean", 0.0),
                learning_ratio_kill=bb.get("learning_ratio_kill", 0.15),
                learning_ratio_tag=bb.get("learning_ratio_tag", 0.30),
                loss_at_500=bb.get("loss_at_500"),
                loss_drop_500_mean=bb.get("loss_drop_500_mean"),
            ),
            sota_layer_gradient_norms=fc.get("sota_layer_gradient_norms", {}),
            sota_layer_activation_norms=fc.get("sota_layer_activation_norms", {}),
            sota_attention_entropy_per_head=fc.get("sota_attention_entropy_per_head", {}),
            sota_component_gradient_norms=fc.get("sota_component_gradient_norms", {}),
            sota_init_logit_max=fc.get("sota_init_logit_max", 10.0),
            sota_gradient_norm_ratio=fc.get("sota_gradient_norm_ratio", 10.0),
            sota_output_entropy=fc.get("sota_output_entropy", 0.0),
        )
    except Exception as e:
        print(f"[calibrate] Error loading profile: {e}, using defaults")
        return Calibration()
