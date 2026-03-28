"""Generate feedback for the Ideator."""

from __future__ import annotations

from typing import Any

from ..types import FalsifierInput, FalsifierOutput, Feedback, S2Result, Tag


def generate_feedback(
    inp: FalsifierInput,
    out: FalsifierOutput,
    s2: S2Result | None,
) -> Feedback:
    """Generate feedback for the Ideator.
    
    LLM-generated for Stage 2, structured for Stage 1.
    """
    if out.verdict in ("REFUTED", "REJECTED"):
        return _generate_failure_feedback(inp, out)
    
    if out.verdict == "STAGE_2_PASSED":
        return _generate_success_feedback(inp, out)
    
    # IMPLEMENTATION_FAIL
    return _generate_error_feedback(inp, out)


def _generate_failure_feedback(
    inp: FalsifierInput,
    out: FalsifierOutput,
) -> Feedback:
    """Generate failure feedback."""
    killed_by = out.killed_by or "unknown"
    kill_reason = out.kill_reason or "No reason provided"
    
    # Determine stage reached (streamlined: T2→T3→{T4,T5}→T7)
    if killed_by.startswith("T2"):
        stage_reached = 0
    elif killed_by.startswith("T3") or killed_by.startswith("T4") or killed_by.startswith("T5"):
        stage_reached = 1
    elif killed_by.startswith("T7"):
        stage_reached = 3
    elif killed_by.startswith("S2"):
        stage_reached = 4
    elif killed_by == "COMPOUND_TAGS" or killed_by.startswith("CORRELATED_TAGS"):
        stage_reached = 3  # Reached end of Stage 1
    else:
        stage_reached = 0
    
    # Build one-liner
    one_line = f"Killed by {killed_by}: {kill_reason[:100]}"
    
    # Extract key measurements
    key_measurements: dict[str, Any] = {}
    
    if out.t3_compilation:
        key_measurements["forward_ms"] = out.t3_compilation.forward_ms
        key_measurements["backward_ms"] = out.t3_compilation.backward_ms
    
    if out.t4_signal:
        key_measurements["gradient_norm_ratio"] = out.t4_signal.gradient_norm_ratio
        key_measurements["output_entropy"] = out.t4_signal.output_entropy
    
    if out.t5_init:
        key_measurements["logit_max"] = out.t5_init.logit_max
        key_measurements["effective_rank_mean"] = out.t5_init.effective_rank_mean
    
    if out.t7_microtrain:
        key_measurements["loss_drop"] = out.t7_microtrain.loss_drop
        key_measurements["learning_ratio"] = out.t7_microtrain.learning_ratio
    
    # Suggested directions
    directions: list[str] = []
    
    # Check tags for specific suggestions
    for tag in out.tags:
        suggestion = _tag_to_suggestion(tag)
        if suggestion and suggestion not in directions:
            directions.append(suggestion)
    
    # Generic suggestions if none specific
    if not directions:
        if killed_by.startswith("T2"):
            directions.append("Reduce parameter count or optimize for speed")
        elif killed_by.startswith("T3"):
            directions.append("Fix compilation errors and ensure gradient connectivity")
        elif killed_by.startswith("T4"):
            directions.append("Check initialization and normalization strategies")
        elif killed_by.startswith("T5"):
            directions.append("Review weight initialization and ensure sufficient diversity")
        elif killed_by.startswith("T7"):
            directions.append("Improve learning dynamics and training stability")
        else:
            directions.append("Review theory and address underlying issues")
    
    return Feedback(
        one_line=one_line,
        stage_reached=stage_reached,
        failure_analysis=None,  # LLM-generated in full implementation
        suggested_directions=directions,
        key_measurements=key_measurements,
    )


def _generate_success_feedback(
    inp: FalsifierInput,
    out: FalsifierOutput,
) -> Feedback:
    """Generate success feedback."""
    return Feedback(
        one_line="Theory survived Stage 1 + Stage 2! Ready for full training run.",
        stage_reached=4,
        failure_analysis=None,
        suggested_directions=[
            "Monitor training for convergence beyond 10K steps",
            "Watch for long-term instability not detected in 500-step evaluation",
        ],
        key_measurements={
            "loss_drop_100": out.t7_microtrain.loss_drop if out.t7_microtrain else None,
            "loss_drop_500": out.s2_results.trend_verification.get("actual_loss_500") if out.s2_results else None,
        } if out.s2_results else {},
    )


def _generate_error_feedback(
    inp: FalsifierInput,
    out: FalsifierOutput,
) -> Feedback:
    """Generate error feedback."""
    return Feedback(
        one_line="Implementation error prevented full evaluation",
        stage_reached=0,
        failure_analysis=out.kill_reason,
        suggested_directions=[
            "Fix code errors and resubmit",
            "Ensure train_gpt.py is syntactically valid Python",
            "Verify all imported modules are available",
        ],
        key_measurements={},
    )


def _tag_to_suggestion(tag: Tag) -> str | None:
    """Convert a tag to a suggestion."""
    suggestions = {
        "T2_tight_budget": "Optimize for smaller model size or faster execution",
        "T2_high_flops": "Reduce computational complexity",
        "T3_disconnected_params": "Ensure all model parameters are connected to the computation graph",
        "T4_gradient_ratio": "Check layer-wise gradient scaling and normalization",
        "T4_low_output_entropy": "Investigate output layer initialization and temperature",
        "T4_extreme_activation_norm": "Add or adjust normalization layers",
        "T5_extreme_logits": "Reduce weight initialization scale or add output scaling",
        "T5_low_effective_rank": "Ensure sufficient weight matrix rank through initialization",
        "T5_high_condition_number": "Improve conditioning through regularization or initialization",
        "T5_weight_symmetry": "Break symmetry through initialization diversity",
        "T5_rank_deficient": "Review weight initialization to ensure full rank matrices",
        "T5_poor_capacity_utilization": "Improve capacity utilization through better initialization",
        "T7_slow_learning": "Improve learning dynamics through better optimization or architecture",
        "T7_gradient_instability": "Add gradient clipping or adjust learning rate",
        "T7_component_speed_imbalance": "Balance component contributions through gating or scaling",
        "T7_entropy_collapse": "Prevent entropy collapse through attention mechanisms",
        "T7_low_throughput": "Optimize for faster training",
        "T7_learning_plateau": "Address learning plateau through architecture changes",
    }
    
    return suggestions.get(tag.tag_id)
