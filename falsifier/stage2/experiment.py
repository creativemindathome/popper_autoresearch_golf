"""Convert hypotheses to experiments and evaluate them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import Calibration, KillHypothesis


@dataclass
class ExperimentSpec:
    """An executable experiment specification."""
    
    name: str
    source: str  # "theory", "ablation_{change}", "baseline"
    steps: int
    seed: int = 42
    dense_logging: bool = False
    metric: str = "loss"
    threshold: float = 0.0
    comparator: str = ">"
    needs_ablation: bool = False
    ablation_target: str | None = None
    component_hooks: list[str] = None  # type: ignore


@dataclass
class ExperimentResult:
    """Result of executing an experiment."""
    
    triggered: bool
    measured_value: float
    threshold: float
    detail: str


def build_experiment(h: KillHypothesis, inp: Any) -> ExperimentSpec:
    """Convert a kill hypothesis to an executable experiment."""
    spec = h.experiment_spec
    
    step = spec.get("step", 500)
    needs_ablation = spec.get("needs_ablation", False)
    ablation_target = spec.get("ablation_target")
    
    # Determine source
    if needs_ablation and ablation_target:
        source = f"ablation_{ablation_target}"
    else:
        source = "theory"
    
    return ExperimentSpec(
        name=h.hypothesis_id,
        source=source,
        steps=step,
        seed=42,
        dense_logging=True,
        metric=spec.get("metric", "loss"),
        threshold=float(spec.get("threshold", 0.0)),
        comparator=spec.get("comparator", ">"),
        needs_ablation=needs_ablation,
        ablation_target=ablation_target,
        component_hooks=spec.get("component_to_instrument", []),
    )


def evaluate_experiment(
    spec: ExperimentSpec,
    run_data: dict[str, Any],
    calibration: Calibration | None,
) -> ExperimentResult:
    """Evaluate an experiment against run data."""
    
    # Get the metric value from run data
    losses = run_data.get("losses", [])
    if not losses:
        return ExperimentResult(
            triggered=False,
            measured_value=0.0,
            threshold=spec.threshold,
            detail="No loss data available",
        )
    
    # Determine which metric to check
    metric = spec.metric
    measured: float = 0.0
    
    if metric == "loss":
        measured = losses[-1] if losses else 0.0
    elif metric == "loss_delta":
        measured = losses[-1] - losses[0] if len(losses) > 1 else 0.0
    elif metric == "grad_norm_max":
        grad_norms = run_data.get("grad_norms", [])
        measured = max(grad_norms) if grad_norms else 0.0
    elif metric == "learning_ratio":
        if calibration and calibration.baseline_100.loss_drop_mean > 0:
            loss_drop = losses[0] - losses[-1] if len(losses) > 1 else 0.0
            measured = loss_drop / calibration.baseline_100.loss_drop_mean
        else:
            measured = 0.0
    else:
        # Fallback to last loss
        measured = losses[-1] if losses else 0.0
    
    # Compare against threshold
    triggered = _compare(measured, spec.threshold, spec.comparator)
    
    detail = f"{metric}={measured:.4f}, threshold={spec.threshold}, comparator={spec.comparator}"
    
    return ExperimentResult(
        triggered=triggered,
        measured_value=measured,
        threshold=spec.threshold,
        detail=detail,
    )


def _compare(value: float, threshold: float, comparator: str) -> bool:
    """Compare value against threshold."""
    if comparator == ">":
        return value > threshold
    elif comparator == ">=":
        return value >= threshold
    elif comparator == "<":
        return value < threshold
    elif comparator == "<=":
        return value <= threshold
    elif comparator == "==":
        return abs(value - threshold) < 1e-6
    else:
        return value > threshold
