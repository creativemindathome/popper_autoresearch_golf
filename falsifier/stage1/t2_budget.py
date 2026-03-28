"""T2: Parameter Budget gate.

Evaluates whether the proposed model fits within the parameter budget:
- Total artifact size limit (16MB - 200KB safety margin)
- Training time budget (600 seconds with 20% tolerance)
- Tags for tight budgets and high FLOPS
"""

from __future__ import annotations

import time
from pathlib import Path

from falsifier.types import Calibration, FalsifierInput, Tag, T2Result, TestStatus
from falsifier.utils.config_parser import (
    count_parameters,
    estimate_artifact_bytes,
    estimate_flops,
    estimate_flops_per_component,
    extract_model_config,
)

# Budget constants
LIMIT_BYTES = 16_777_216  # 16 MB
SAFETY_MARGIN_BYTES = 200_000  # 200 KB safety margin
EFFECTIVE_LIMIT = LIMIT_BYTES - SAFETY_MARGIN_BYTES
TIME_BUDGET_SECONDS = 600  # 10 minutes
TIME_TOLERANCE = 1.2  # 20% tolerance


def _get_baseline_tokens_per_second(calibration: Calibration | None) -> float:
    """Get baseline tokens per second from calibration or use default."""
    if calibration and calibration.baseline_100:
        return calibration.baseline_100.tokens_per_second_mean
    # Default fallback for minimal env model
    return 1000.0


def _estimate_training_seconds(
    flops: float,
    tokens_per_second: float,
    config: dict,
    calibration: Calibration | None,
) -> float:
    """Estimate training time in seconds.

    Uses calibration data if available, otherwise estimates from FLOPS.
    """
    if calibration and calibration.sota_tokens_per_second > 0:
        # Use SOTA throughput as baseline
        tokens_per_step = int(config.get("train_batch_tokens", 524_288))
        iterations = int(config.get("iterations", 20_000))
        total_tokens = tokens_per_step * iterations
        # Scale by FLOPS ratio
        sota_flops = calibration.sota_param_count * 2  # rough proxy
        flops_ratio = flops / max(sota_flops, 1.0)
        est_seconds = (total_tokens / calibration.sota_tokens_per_second) * flops_ratio
        return est_seconds

    # Fallback: rough estimation
    # Assume ~1ms per 1M FLOPS on CPU
    return flops / 1e6 * 0.001


def run_t2(inp: FalsifierInput) -> T2Result:
    """Run T2: Parameter Budget test.

    Args:
        inp: FalsifierInput with proposed_train_gpt and calibration

    Returns:
        T2Result with budget status, estimates, and tags
    """
    start_time = time.perf_counter()

    try:
        # Extract config and compute estimates
        config = extract_model_config(inp.proposed_train_gpt)
        param_counts = count_parameters(config)
        estimated_params = sum(param_counts.values())

        # Estimate artifact size
        total_artifact, remaining = estimate_artifact_bytes(
            inp.proposed_train_gpt,
            inp.auxiliary_files,
            bits_per_param=8.0,
            limit_bytes=LIMIT_BYTES,
        )

        # Estimate FLOPS
        estimated_flops = estimate_flops(config)

        # Estimate per-component FLOPs for architectural balance analysis
        flops_breakdown = estimate_flops_per_component(config)
        attn_flops_ratio = flops_breakdown["attn_ratio"]
        mlp_flops_ratio = flops_breakdown["mlp_ratio"]
        embed_flops_ratio = flops_breakdown["embed_ratio"]

        # Calculate architectural balance score
        # Ideal: attn_ratio ≈ 0.25-0.35, mlp_ratio ≈ 0.60-0.70
        # Score based on distance from ideal ranges
        ideal_attn_min, ideal_attn_max = 0.25, 0.35
        ideal_mlp_min, ideal_mlp_max = 0.60, 0.70

        attn_distance = max(0.0, ideal_attn_min - attn_flops_ratio, attn_flops_ratio - ideal_attn_max)
        mlp_distance = max(0.0, ideal_mlp_min - mlp_flops_ratio, mlp_flops_ratio - ideal_mlp_max)

        # Balance score: 1.0 = perfect balance, 0.0 = completely unbalanced
        # Penalize by distance from ideal ranges
        architectural_balance_score = max(0.0, 1.0 - 5.0 * (attn_distance + mlp_distance))

        # Get baseline from calibration for FLOPS ratio
        baseline_flops = 0.0
        if inp.calibration:
            # Use stored SOTA FLOPS or estimate from param count
            baseline_flops = getattr(inp.calibration, "sota_flops_estimate", 0.0)
        if baseline_flops <= 0:
            # Estimate baseline from default config
            baseline_config = extract_model_config("")
            baseline_flops = estimate_flops(baseline_config)

        flops_ratio = estimated_flops / max(baseline_flops, 1.0)

        # Estimate training time
        tokens_per_second = _get_baseline_tokens_per_second(inp.calibration)
        est_train_secs = _estimate_training_seconds(
            estimated_flops,
            tokens_per_second,
            config,
            inp.calibration,
        )

        # Budget utilization
        budget_utilization = total_artifact / LIMIT_BYTES

        # Determine status and tags
        tags: list[Tag] = []
        status: TestStatus = "PASS"
        kill_reason: str | None = None

        # FATAL: artifact exceeds limit
        if total_artifact > EFFECTIVE_LIMIT:
            status = "FAIL_FATAL"
            kill_reason = (
                f"Artifact size {total_artifact:,} bytes exceeds limit "
                f"{EFFECTIVE_LIMIT:,} bytes ({LIMIT_BYTES:,} - {SAFETY_MARGIN_BYTES:,} safety)"
            )

        # FATAL: training time exceeds budget with tolerance
        time_limit_with_tolerance = TIME_BUDGET_SECONDS * TIME_TOLERANCE
        if est_train_secs > time_limit_with_tolerance:
            status = "FAIL_FATAL"
            kill_reason = (
                f"Estimated training time {est_train_secs:.1f}s exceeds "
                f"limit {time_limit_with_tolerance:.1f}s ({TIME_BUDGET_SECONDS}s * {TIME_TOLERANCE})"
            )

        # TAG: tight budget (remaining < 200KB)
        if remaining < SAFETY_MARGIN_BYTES:
            tags.append(
                Tag(
                    tag_id="T2_tight_budget",
                    test_id="T2",
                    category="speed_pathology",
                    description=f"Only {remaining:,} bytes remaining ({remaining/1024:.1f}KB < 200KB safety margin)",
                )
            )

        # TAG: high FLOPS ratio
        if flops_ratio > 1.5:
            tags.append(
                Tag(
                    tag_id="T2_high_flops",
                    test_id="T2",
                    category="speed_pathology",
                    description=f"FLOPS ratio {flops_ratio:.2f} exceeds 1.5x baseline",
                )
            )

        # TAG: unbalanced architecture
        # Flag when attn/mlp ratio > 2.0 or < 0.5 (i.e., one dominates the other)
        if mlp_flops_ratio > 0:
            attn_mlp_ratio = attn_flops_ratio / mlp_flops_ratio
            if attn_mlp_ratio > 2.0 or attn_mlp_ratio < 0.5:
                tags.append(
                    Tag(
                        tag_id="T2_unbalanced_architecture",
                        test_id="T2",
                        category="capacity_pathology",
                        description=(
                            f"Attention FLOPs {attn_flops_ratio:.1%} vs MLP FLOPs {mlp_flops_ratio:.1%} "
                            f"(ratio {attn_mlp_ratio:.2f} outside 0.5-2.0 range)"
                        ),
                    )
                )

        wall_seconds = time.perf_counter() - start_time

        return T2Result(
            status=status,
            test_id="T2",
            estimated_params=estimated_params,
            estimated_artifact_bytes=total_artifact,
            budget_remaining_bytes=remaining,
            budget_utilization=budget_utilization,
            flops_ratio=flops_ratio,
            estimated_training_seconds=est_train_secs,
            kill_reason=kill_reason,
            tags=tags,
            wall_seconds=wall_seconds,
            attn_flops_ratio=attn_flops_ratio,
            mlp_flops_ratio=mlp_flops_ratio,
            embed_flops_ratio=embed_flops_ratio,
            architectural_balance_score=architectural_balance_score,
        )

    except Exception as e:
        wall_seconds = time.perf_counter() - start_time
        return T2Result(
            status="FAIL_FATAL",
            test_id="T2",
            estimated_params=0,
            estimated_artifact_bytes=0,
            budget_remaining_bytes=0,
            budget_utilization=0.0,
            flops_ratio=1.0,
            estimated_training_seconds=0.0,
            kill_reason=f"T2 error: {e}",
            tags=[],
            wall_seconds=wall_seconds,
            attn_flops_ratio=0.0,
            mlp_flops_ratio=0.0,
            embed_flops_ratio=0.0,
            architectural_balance_score=0.0,
        )
