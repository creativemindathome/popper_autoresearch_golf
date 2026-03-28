from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Verdict = Literal["reject", "refute", "promote", "implementation_fail"]
TheoryRecordVerdict = Literal["refuted", "surviving", "reference"]


@dataclass
class TheoryHistoryRecord:
    theory_id: str
    verdict: TheoryRecordVerdict
    theory_text: str
    failure_context: str = ""
    mechanism_tags: list[str] = field(default_factory=list)
    related_theory_ids: list[str] = field(default_factory=list)


@dataclass
class PrecedentEvidence:
    query_mode: str
    matched_theory_id: str | None
    matched_verdict: TheoryRecordVerdict | None
    matched_fields: list[str] = field(default_factory=list)
    supporting_theory_ids: list[str] = field(default_factory=list)
    overlap_score: float = 0.0
    explanation: str = ""


@dataclass
class CandidatePackage:
    theory_id: str
    train_gpt_path: str | Path
    what_and_why: str
    reference_theories: list[str] = field(default_factory=list)
    theory_history: list[TheoryHistoryRecord] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSignature:
    param_count: int
    trainable_param_count: int
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    tie_embeddings: bool
    smoke_loss: float | None = None


@dataclass
class BudgetCheck:
    total_artifact_bytes: int
    remaining_budget_bytes: int
    param_count_estimate: int
    flops_estimate: float
    within_budget: bool


@dataclass
class SmokeDiagnostics:
    forward_ok: bool
    backward_ok: bool
    loss_is_finite: bool
    params_without_grad: list[str] = field(default_factory=list)


@dataclass
class Stage1Result:
    theory_id: str
    verdict: Verdict
    validation: ValidationResult
    novelty_score: float
    novelty_reason: str
    t1_mode: str = "graph_aware_precedent"
    precedent_evidence: PrecedentEvidence | None = None
    model_signature: ModelSignature | None = None
    budget: BudgetCheck | None = None
    smoke: SmokeDiagnostics | None = None
    stage_reached: str = "validation"
    reasons: list[str] = field(default_factory=list)
