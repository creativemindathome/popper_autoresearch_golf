"""Tag resolution: convert Stage 1 tags into Stage 2 experiments.

Tag resolution map from PRD lines 1817-1831.
"""

from __future__ import annotations

from typing import Any

from .experiment import ExperimentSpec

# Tag resolution map: Stage 1 tag -> Stage 2 experiment
TAG_RESOLUTION: dict[str, dict[str, Any]] = {
    "T4_gradient_ratio": {
        "metric": "gradient_norm_ratio",
        "step": 500,
        "threshold": 3.0,
        "comparator": ">",
        "description": "Gradient norm ratio exceeds 3x baseline at step 500",
    },
    "T4_low_output_entropy": {
        "metric": "output_entropy",
        "step": 500,
        "threshold": 0.2,
        "comparator": "<",
        "description": "Output entropy below 20% of max at step 500",
    },
    "T7_entropy_collapse": {
        "metric": "output_entropy",
        "step": 300,
        "threshold": 0.5,
        "comparator": "<",
        "description": "Entropy collapses below 50% of initial by step 300",
    },
    "T5_extreme_logits": {
        "metric": "max_logit",
        "step": 500,
        "threshold": 100.0,
        "comparator": ">",
        "description": "Max logit exceeds 100 at step 500",
    },
    "T5_weight_symmetry": {
        "metric": "weight_row_cosine",
        "step": 200,
        "threshold": 0.3,
        "comparator": ">",
        "description": "Weight symmetry (cosine) still > 0.3 at step 200",
    },
    "T7_component_speed_imbalance": {
        "metric": "component_speed_ratio",
        "step": 500,
        "threshold": 200.0,
        "comparator": ">",
        "description": "Component speed ratio exceeds 200x at step 500",
    },
    "T7_slow_learning": {
        "metric": "learning_ratio",
        "step": 500,
        "threshold": 0.50,
        "comparator": "<",
        "description": "Learning ratio below 50% of baseline at step 500",
    },
    "T7_low_throughput": {
        "metric": "projected_full_run",
        "step": 500,
        "threshold": 660.0,
        "comparator": ">",
        "description": "Projected full run exceeds 660 seconds",
    },
    "T6_false_premise": {
        "metric": None,
        "step": None,
        "description": "Informational only - no specific experiment",
    },
    "T6_high_information_cost": {
        "metric": "loss_delta",
        "step": 500,
        "threshold": 0.0,
        "comparator": ">",
        "description": "Loss not recovered below baseline by step 500",
    },
}


def build_tag_experiments(tags: list[Any]) -> list[ExperimentSpec]:
    """Convert Stage 1 tags into Stage 2 experiments."""
    experiments: list[ExperimentSpec] = []
    
    for tag in tags:
        tag_id = tag.tag_id if hasattr(tag, "tag_id") else str(tag)
        
        if tag_id not in TAG_RESOLUTION:
            continue
        
        spec = TAG_RESOLUTION[tag_id]
        
        # Skip if no metric (informational only)
        if spec.get("metric") is None:
            continue
        
        experiments.append(ExperimentSpec(
            name=f"tag_{tag_id}",
            source="theory",
            steps=spec.get("step", 500),
            seed=42,
            dense_logging=True,
            metric=spec["metric"],
            threshold=float(spec.get("threshold", 0.0)),
            comparator=spec.get("comparator", ">"),
        ))
    
    return experiments
