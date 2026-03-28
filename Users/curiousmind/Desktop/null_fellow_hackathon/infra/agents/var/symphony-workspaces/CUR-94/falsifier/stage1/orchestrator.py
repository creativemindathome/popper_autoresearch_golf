from __future__ import annotations

from pathlib import Path

from ..adapters.parameter_golf import run_smoke_diagnostics
from ..precedent import graph_aware_precedent
from ..types import BudgetCheck, CandidatePackage, Stage1Result
from ..utils.config_parser import count_parameters, estimate_artifact_bytes, estimate_flops, extract_model_config
from ..validation import validate_candidate_package


NOVELTY_PROMOTE_THRESHOLD = 0.55
ARTIFACT_LIMIT_BYTES = 16_777_216 - 200_000
BASELINE_CONFIG = {
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
MAX_FLOPS_RATIO = 1.5
T1_PRECEDENT = "T1"
T2_BUDGET = "T2"
T3_SMOKE = "T3"


def run_budget_gate(candidate: CandidatePackage) -> BudgetCheck:
    source = Path(candidate.train_gpt_path).read_text()
    config = extract_model_config(source)
    param_counts = count_parameters(config)
    total_artifact, remaining = estimate_artifact_bytes(source, {})
    param_estimate = sum(param_counts.values())
    flops = estimate_flops(config)
    baseline_flops = estimate_flops(BASELINE_CONFIG)
    within_budget = total_artifact <= ARTIFACT_LIMIT_BYTES and flops <= baseline_flops * MAX_FLOPS_RATIO
    return BudgetCheck(
        total_artifact_bytes=total_artifact,
        remaining_budget_bytes=remaining,
        param_count_estimate=param_estimate,
        flops_estimate=flops,
        within_budget=within_budget,
    )


def run_stage1(candidate: CandidatePackage) -> Stage1Result:
    validation = validate_candidate_package(candidate)
    if not validation.ok:
        return Stage1Result(
            theory_id=candidate.theory_id,
            verdict="reject",
            validation=validation,
            novelty_score=0.0,
            novelty_reason="validation failed before novelty scoring",
            stage_reached="validation",
            reasons=list(validation.reasons),
        )

    budget = run_budget_gate(candidate)
    if not budget.within_budget:
        reasons = []
        if budget.total_artifact_bytes > ARTIFACT_LIMIT_BYTES:
            reasons.append(
                f"artifact budget exceeded: {budget.total_artifact_bytes:,} > {ARTIFACT_LIMIT_BYTES:,}"
            )
        if budget.flops_estimate > estimate_flops(BASELINE_CONFIG) * MAX_FLOPS_RATIO:
            reasons.append("estimated flops exceed deterministic stage-1 threshold")
        return Stage1Result(
            theory_id=candidate.theory_id,
            verdict="refute",
            validation=validation,
            novelty_score=0.0,
            novelty_reason=f"candidate failed {T2_BUDGET} budget gate before smoke testing",
            budget=budget,
            stage_reached=T2_BUDGET,
            reasons=reasons,
        )

    try:
        signature, smoke = run_smoke_diagnostics(
            candidate.train_gpt_path,
            env_overrides=candidate.env_overrides,
        )
    except Exception as exc:
        return Stage1Result(
            theory_id=candidate.theory_id,
            verdict="implementation_fail",
            validation=validation,
            novelty_score=0.0,
            novelty_reason=f"{T3_SMOKE} train_gpt smoke test failed",
            budget=budget,
            stage_reached=T3_SMOKE,
            reasons=[f"train_gpt smoke test failed: {exc}"],
        )

    if not smoke.backward_ok:
        return Stage1Result(
            theory_id=candidate.theory_id,
            verdict="implementation_fail",
            validation=validation,
            novelty_score=0.0,
            novelty_reason=f"{T3_SMOKE} train_gpt smoke test found disconnected gradients",
            model_signature=signature,
            budget=budget,
            smoke=smoke,
            stage_reached=T3_SMOKE,
            reasons=[f"parameters without gradients: {', '.join(smoke.params_without_grad[:5])}"],
        )

    novelty, novelty_reason, precedent_evidence = graph_aware_precedent(
        candidate.what_and_why,
        theory_history=candidate.theory_history,
        reference_texts=candidate.reference_theories,
    )
    if novelty < NOVELTY_PROMOTE_THRESHOLD:
        return Stage1Result(
            theory_id=candidate.theory_id,
            verdict="refute",
            validation=validation,
            novelty_score=novelty,
            novelty_reason=novelty_reason,
            precedent_evidence=precedent_evidence,
            model_signature=signature,
            budget=budget,
            smoke=smoke,
            stage_reached=T1_PRECEDENT,
            reasons=[f"novelty below threshold: {novelty:.2f} < {NOVELTY_PROMOTE_THRESHOLD:.2f}"],
        )

    return Stage1Result(
        theory_id=candidate.theory_id,
        verdict="promote",
        validation=validation,
        novelty_score=novelty,
        novelty_reason=novelty_reason,
        precedent_evidence=precedent_evidence,
        model_signature=signature,
        budget=budget,
        smoke=smoke,
        stage_reached="promote",
        reasons=[],
    )


def run_stage_1(candidate: CandidatePackage) -> Stage1Result:
    return run_stage1(candidate)
