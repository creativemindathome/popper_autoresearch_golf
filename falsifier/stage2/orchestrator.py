"""Stage 2 orchestrator: adversarial LLM prosecution of theories."""

from __future__ import annotations

import time
from typing import Any

from ..types import (
    Calibration,
    FalsifierInput,
    FalsifierOutput,
    Feedback,
    KillHypothesis,
    S2Result,
    Verdict,
)
from .ablation import build_ablation_source
from .experiment import ExperimentResult, ExperimentSpec, build_experiment, evaluate_experiment
from .feedback import generate_feedback
from .hypothesis_gen import generate_kill_hypotheses, validate_hypothesis
from .run_executor import execute_training_run
from .run_planner import RunPlan, RunSpec, optimize_run_plan
from .tag_resolver import build_tag_experiments
from .trend_verifier import verify_trends


def run_stage_2(
    inp: FalsifierInput,
    stage1_output: FalsifierOutput,
) -> FalsifierOutput:
    """Execute Stage 2: adversarial prosecution.
    
    Returns: FalsifierOutput with verdict, s2_results, and complete feedback.
    """
    start_time = time.time()
    
    # Collect Stage 1 results
    stage1_results = _collect_stage1_results(stage1_output)
    
    # ══ Phase 1: Generate hypotheses ════════════════════════════════════════════
    print("[Stage 2] Phase 1: Generating kill hypotheses...")
    hypotheses = generate_kill_hypotheses(inp, stage1_results)
    
    # Filter to valid hypotheses
    valid_hypotheses = [h for h in hypotheses if validate_hypothesis(h, inp)]
    
    if not valid_hypotheses:
        print("[Stage 2] No valid hypotheses generated, skipping Stage 2")
        return _stage2_skipped(stage1_output, start_time)
    
    print(f"[Stage 2] Generated {len(valid_hypotheses)} valid hypotheses")
    
    # ══ Phase 2: Convert to experiments ═══════════════════════════════════════
    print("[Stage 2] Phase 2: Building experiments...")
    experiments: list[ExperimentSpec] = []
    
    # Add hypothesis experiments
    for h in valid_hypotheses:
        exp = build_experiment(h, inp)
        experiments.append(exp)
    
    # Add tag resolution experiments
    tag_experiments = build_tag_experiments(stage1_output.tags)
    experiments.extend(tag_experiments)
    
    print(f"[Stage 2] {len(experiments)} experiments total")
    
    # ══ Phase 3: Optimize runs ════════════════════════════════════════════════
    print("[Stage 2] Phase 3: Optimizing runs...")
    run_plan = optimize_run_plan(experiments, inp)
    
    print(f"[Stage 2] {len(run_plan.unique_runs)} unique runs to execute")
    
    # ══ Phase 4: Execute training runs ════════════════════════════════════════
    print("[Stage 2] Phase 4: Executing training runs...")
    run_results: dict[str, Any] = {}
    
    for run_spec in run_plan.unique_runs:
        print(f"[Stage 2] Running {run_spec.name} ({run_spec.steps} steps)...")
        result = execute_training_run(run_spec, inp)
        run_results[run_spec.name] = result
        
        # Check for early termination
        if result.diverged:
            print(f"[Stage 2] Run {run_spec.name} diverged at step {result.diverged_step}")
            wall_seconds = time.time() - start_time
            return _build_refuted_output(
                stage1_output,
                valid_hypotheses,
                experiments,
                result,
                "Run diverged during training",
                wall_seconds,
            )
    
    # ══ Phase 5: Evaluate experiments ═════════════════════════════════════════
    print("[Stage 2] Phase 5: Evaluating experiments...")
    calibration = inp.calibration or Calibration()
    
    hypothesis_results: list[dict[str, Any]] = []
    killed_by: str | None = None
    kill_reason: str | None = None
    
    for i, exp in enumerate(experiments):
        run_name = run_plan.experiment_to_run.get(exp.name, "theory_run")
        run_data = run_results.get(run_name, {})
        
        eval_result = evaluate_experiment(exp, run_data, calibration)
        
        hypothesis_results.append({
            "hypothesis_id": exp.name,
            "triggered": eval_result.triggered,
            "measured_value": eval_result.measured_value,
            "threshold": eval_result.threshold,
            "comparator": exp.comparator,
            "detail": eval_result.detail,
        })
        
        if eval_result.triggered and killed_by is None:
            killed_by = f"S2_H{i+1}"
            kill_reason = f"Experiment {exp.name} triggered: {eval_result.detail}"
            print(f"[Stage 2] Kill hypothesis triggered: {exp.name}")
    
    # ══ Phase 6: Verify trends ════════════════════════════════════════════════
    print("[Stage 2] Phase 6: Verifying trends...")
    trend_result = None
    
    if "theory_run" in run_results and stage1_output.t7_microtrain:
        theory_run = run_results["theory_run"]
        trend = verify_trends(theory_run, stage1_output.t7_microtrain, calibration)
        
        trend_result = {
            "broken": trend.broken,
            "detail": trend.detail,
            "loss_deviation": trend.loss_deviation,
            "actual_loss_500": trend.actual_loss_500,
        }
        
        if trend.broken and killed_by is None:
            killed_by = "S2_TREND"
            kill_reason = f"Trend verification failed: {trend.detail}"
            print(f"[Stage 2] Trend verification failed")
    
    # Build final result
    wall_seconds = time.time() - start_time
    
    s2_result = S2Result(
        verdict="REFUTED" if killed_by else "STAGE_2_PASSED",
        killed_by=killed_by,
        kill_reason=kill_reason,
        hypotheses=[{"hypothesis_id": h.hypothesis_id, "failure_mode": h.failure_mode} for h in valid_hypotheses],
        hypothesis_results=hypothesis_results,
        tag_results=[],  # Populated if needed
        trend_verification=trend_result,
    )
    
    # Build output
    out = FalsifierOutput(
        theory_id=stage1_output.theory_id,
        verdict="REFUTED" if killed_by else "STAGE_2_PASSED",
        killed_by=killed_by,
        kill_reason=kill_reason,
        t0_novelty=stage1_output.t0_novelty,
        t1_precedent=stage1_output.t1_precedent,
        t2_budget=stage1_output.t2_budget,
        t3_compilation=stage1_output.t3_compilation,
        t4_signal=stage1_output.t4_signal,
        t5_init=stage1_output.t5_init,
        t6_checkpoint=stage1_output.t6_checkpoint,
        t7_microtrain=stage1_output.t7_microtrain,
        s2_results=s2_result,
        tags=stage1_output.tags,
        feedback=generate_feedback(inp, stage1_output, s2_result),
        total_wall_seconds=stage1_output.total_wall_seconds + wall_seconds,
        total_gpu_seconds=stage1_output.total_gpu_seconds,
    )
    
    # Update feedback with Stage 2 context
    out.feedback.stage_reached = 4 if out.verdict == "STAGE_2_PASSED" else _infer_stage(killed_by)
    
    return out


def _collect_stage1_results(stage1_output: FalsifierOutput) -> dict[str, Any]:
    """Collect Stage 1 results into a dict."""
    return {
        "T0": stage1_output.t0_novelty,
        "T1": stage1_output.t1_precedent,
        "T2": stage1_output.t2_budget,
        "T3": stage1_output.t3_compilation,
        "T4": stage1_output.t4_signal,
        "T5": stage1_output.t5_init,
        "T6": stage1_output.t6_checkpoint,
        "T7": stage1_output.t7_microtrain,
    }


def _stage2_skipped(stage1_output: FalsifierOutput, start_time: float) -> FalsifierOutput:
    """Stage 2 skipped (no valid hypotheses or no API key)."""
    out = FalsifierOutput(
        theory_id=stage1_output.theory_id,
        verdict="STAGE_1_PASSED",  # Not killed, but not fully passed either
        killed_by=None,
        kill_reason=None,
        t0_novelty=stage1_output.t0_novelty,
        t1_precedent=stage1_output.t1_precedent,
        t2_budget=stage1_output.t2_budget,
        t3_compilation=stage1_output.t3_compilation,
        t4_signal=stage1_output.t4_signal,
        t5_init=stage1_output.t5_init,
        t6_checkpoint=stage1_output.t6_checkpoint,
        t7_microtrain=stage1_output.t7_microtrain,
        tags=stage1_output.tags,
        feedback=Feedback(
            one_line="Stage 1 passed. Stage 2 skipped (no LLM hypotheses).",
            stage_reached=3,
            failure_analysis=None,
            suggested_directions=["Re-run with ANTHROPIC_API_KEY for full Stage 2 evaluation"],
        ),
        total_wall_seconds=stage1_output.total_wall_seconds + (time.time() - start_time),
        total_gpu_seconds=stage1_output.total_gpu_seconds,
    )
    return out


def _build_refuted_output(
    stage1_output: FalsifierOutput,
    hypotheses: list[KillHypothesis],
    experiments: list[ExperimentSpec],
    run_result: Any,
    reason: str,
    wall_seconds: float,
) -> FalsifierOutput:
    """Build output for refuted case during run execution."""
    s2 = S2Result(
        verdict="REFUTED",
        killed_by="S2_DIVERGE",
        kill_reason=f"Training run diverged: {reason}",
        hypotheses=[{"hypothesis_id": h.hypothesis_id, "failure_mode": h.failure_mode} for h in hypotheses],
        hypothesis_results=[],
        tag_results=[],
        trend_verification=None,
    )
    
    out = FalsifierOutput(
        theory_id=stage1_output.theory_id,
        verdict="REFUTED",
        killed_by="S2_DIVERGE",
        kill_reason=f"Training run diverged: {reason}",
        t0_novelty=stage1_output.t0_novelty,
        t1_precedent=stage1_output.t1_precedent,
        t2_budget=stage1_output.t2_budget,
        t3_compilation=stage1_output.t3_compilation,
        t4_signal=stage1_output.t4_signal,
        t5_init=stage1_output.t5_init,
        t6_checkpoint=stage1_output.t6_checkpoint,
        t7_microtrain=stage1_output.t7_microtrain,
        s2_results=s2,
        tags=stage1_output.tags,
        feedback=generate_feedback(FalsifierInput(theory_id=stage1_output.theory_id, what_and_why=""), stage1_output, s2),
        total_wall_seconds=stage1_output.total_wall_seconds + wall_seconds,
        total_gpu_seconds=stage1_output.total_gpu_seconds,
    )
    
    out.feedback.stage_reached = 4
    return out


def _infer_stage(killed_by: str | None) -> int:
    """Infer stage reached from killed_by."""
    if killed_by is None:
        return 4
    if killed_by.startswith("S2"):
        return 4
    return 3  # Stage 1
