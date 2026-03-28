"""
Falsifier types: complete PRD dataclass schema.

This module defines all data structures for the two-stage falsifier pipeline:
- Stage 1: Fixed battery of 8 tests (T0-T7) with dependency graph
- Stage 2: Adversarial LLM agent that generates kill hypotheses

All dataclasses are frozen where possible for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ═══════════════════════════════════════════════════════════════════════════════
# Status and verdict literals
# ═══════════════════════════════════════════════════════════════════════════════

TestStatus = Literal["PASS", "FAIL_TAG", "FAIL_FATAL", "SKIP"]
"""Stage 1 test result status.

- PASS: Test passed, no issues
- FAIL_TAG: Test passed but with warnings (accumulate tags)
- FAIL_FATAL: Test failed, theory is refuted at this gate
- SKIP: Test was skipped (theory-type routing or dependency not met)
"""

Verdict = Literal["REJECTED", "REFUTED", "IMPLEMENTATION_FAIL", "STAGE_1_PASSED", "STAGE_2_PASSED"]
"""Final falsifier verdict.

- REJECTED: Validation failed, not tested
- REFUTED: Theory is wrong (killed by specific test)
- IMPLEMENTATION_FAIL: Code broke, idea untested — requeue for repair
- STAGE_1_PASSED: Survived Stage 1, needs Stage 2
- STAGE_2_PASSED: Survived both stages, ready for full training run
"""

TheoryType = Literal["architectural", "training", "data", "quantization", "hybrid"]
"""Theory classification for test routing."""

Confidence = Literal["high", "medium", "low"]
"""Confidence level for Stage 2 kill hypotheses."""

# ═══════════════════════════════════════════════════════════════════════════════
# Tag system
# ═══════════════════════════════════════════════════════════════════════════════

TAG_CATEGORIES: dict[str, list[str]] = {
    "gradient_pathology": [
        "T4_gradient_ratio",
        "T4_gradient_imbalance",
        "T7_gradient_instability",
    ],
    "entropy_pathology": ["T4_low_output_entropy", "T7_entropy_collapse"],
    "scale_pathology": [
        "T4_extreme_activation_norm",
        "T4_low_signal_to_noise",
        "T5_extreme_logits",
        "T5_high_condition_number",
    ],
    "capacity_pathology": [
        "T4_dead_neurons",
        "T5_weight_symmetry",
        "T5_low_effective_rank",
        "T5_rank_deficient",
        "T5_poor_capacity_utilization",
        "T7_component_speed_imbalance",
        "T7_component_imbalance",
    ],
    "mechanism_pathology": ["T6_false_premise", "T6_high_information_cost", "T6_diminishing_returns"],
    "speed_pathology": ["T2_high_flops", "T7_slow_learning", "T7_low_throughput", "T7_learning_plateau"],
    "stability_pathology": ["T7_divergent_training", "T7_high_variance"],
}
"""Tag categories for compound kill rules.

A correlated kill happens when >= 2 tags from different tests appear in the same category.
"""


@dataclass(frozen=True)
class Tag:
    """A non-fatal warning tag from a test."""

    tag_id: str
    test_id: str
    category: str
    description: str


# ═══════════════════════════════════════════════════════════════════════════════
# Input structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParentRef:
    """Reference to an existing graph node."""

    node_id: str
    relationship: str  # "builds_on", "learns_from"
    what_changed: str  # free text describing the delta


@dataclass
class ComponentSpec:
    """Novel code component that can't be expressed as config delta."""

    name: str
    code: str
    injection_point: str  # "after_attention", "before_mlp", etc.
    init_gate: float  # 0.0 = gated to zero at init


@dataclass
class KnowledgeGraph:
    """Graph of all theory nodes and their relationships.

    Backed by JSON file storage. Provides query methods for T1 (precedent)
    and T6d (interpolation).
    """

    graph_path: Path | None = None
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def get_nodes(self, status: str | None = None) -> list[dict[str, Any]]:
        """Query nodes by status (e.g., 'REFUTED', 'STAGE_2_PASSED')."""
        if status is None:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n.get("status") == status]

    def get_measurement_history(self, key: str) -> list[tuple[float, float]]:
        """Get historical (value, result) pairs for a config key.

        Returns list of (proposed_value, measured_bpb) for interpolation.
        """
        history: list[tuple[float, float]] = []
        for node in self.nodes.values():
            delta = node.get("config_delta", {})
            if key in delta and "measured_bpb" in node:
                history.append((float(delta[key]), float(node["measured_bpb"])))
        return sorted(history)

    def find_relevant_graph_nodes(
        self, theory_type: str, change_types: set[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find relevant nodes for Stage 2 context building."""
        relevant: list[tuple[dict[str, Any], int]] = []
        for node in self.nodes.values():
            score = 0
            if node.get("theory_type") == theory_type:
                score += 1
            if node.get("change_types") & change_types:
                score += len(node.get("change_types", set()) & change_types)
            if score > 0:
                relevant.append((node, score))
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in relevant[:limit]]


@dataclass
class FalsifierInput:
    """Everything the Falsifier receives."""

    # ══ The Theory (from Ideator) ════════════════════════════════════════════
    theory_id: str
    what_and_why: str
    # One paragraph. What you're changing and why it should work.
    # Must reference observable properties of the system.
    config_delta: dict[str, Any] | None = None
    new_components: list[ComponentSpec] | None = None
    parents: list[ParentRef] = field(default_factory=list)
    theory_type: TheoryType = "architectural"

    # ══ The Code ═════════════════════════════════════════════════════════════
    # Either source string OR file path (adapter resolves to source)
    proposed_train_gpt: str = ""
    train_gpt_path: Path | str = ""

    auxiliary_files: dict[str, str] = field(default_factory=dict)

    # ══ The Environment ═════════════════════════════════════════════════════
    sota_train_gpt: str = ""  # current SOTA source code
    sota_checkpoint_path: Path | str = ""  # path to trained SOTA model weights
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    val_data_path: Path | str = ""  # path to FineWeb validation data

    # ══ Pre-computed Calibration ═════════════════════════════════════════════
    calibration: Calibration | None = None  # set below after Calibration class def

    def __post_init__(self):
        # Resolve train_gpt_path to source if path provided
        if not self.proposed_train_gpt and self.train_gpt_path:
            path = Path(self.train_gpt_path)
            if path.exists():
                object.__setattr__(self, "proposed_train_gpt", path.read_text())


# Convenience alias for backward compatibility
CandidatePackage = FalsifierInput


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Baseline100:
    """100-step baseline from 5-seed runs."""

    loss_drop_mean: float = 0.0
    loss_drop_std: float = 0.0
    loss_at_1_mean: float = 0.0
    loss_at_100_mean: float = 0.0
    gradient_norm_mean: float = 0.0
    gradient_norm_std: float = 0.0
    tokens_per_second_mean: float = 0.0

    # Derived thresholds
    learning_ratio_kill: float = 0.15
    learning_ratio_tag: float = 0.30

    # Extended baseline (single 500-step run)
    loss_at_500: float | None = None
    loss_drop_500_mean: float | None = None


@dataclass
class Calibration:
    """Pre-computed baselines. Run once per SOTA change."""

    # ══ Trained checkpoint profile (when checkpoint available) ═══════════════
    sota_layer_gradient_norms: dict[str, float] = field(default_factory=dict)
    sota_layer_activation_norms: dict[str, float] = field(default_factory=dict)
    sota_attention_entropy_per_head: dict[str, float] = field(default_factory=dict)
    sota_component_gradient_norms: dict[str, float] = field(default_factory=dict)
    sota_mlp_effective_ranks: dict[str, float] = field(default_factory=dict)
    sota_attn_effective_ranks: dict[str, float] = field(default_factory=dict)
    sota_mlp_singular_values: dict[str, list[float]] = field(default_factory=dict)
    sota_attn_singular_values: dict[str, list[float]] = field(default_factory=dict)
    sota_init_logit_max: float = 10.0  # default fallback
    sota_gradient_norm_ratio: float = 10.0  # default fallback
    sota_output_entropy: float = 0.0

    # ══ Budget stats ═════════════════════════════════════════════════════════
    sota_artifact_bytes: int = 0
    sota_param_count: int = 0
    sota_tokens_per_second: float = 0.0

    # ══ 100-step baseline (5-seed runs) ══════════════════════════════════════
    baseline_100: Baseline100 = field(default_factory=Baseline100)


# Patch forward reference in FalsifierInput
FalsifierInput.__annotations__["calibration"] = Calibration | None


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 test results (Streamlined: T2→T3→{T4,T5}→T7, T0/T1/T6 removed)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class T2Result:
    """T2: Parameter Budget."""

    status: TestStatus = "PASS"
    test_id: str = "T2"
    estimated_params: int = 0
    estimated_artifact_bytes: int = 0
    budget_remaining_bytes: int = 0
    budget_utilization: float = 0.0
    flops_ratio: float = 1.0
    estimated_training_seconds: float = 0.0
    kill_reason: str | None = None
    tags: list[Tag] = field(default_factory=list)
    wall_seconds: float = 0.0

    # Per-component FLOPs breakdown
    attn_flops_ratio: float = 0.0
    mlp_flops_ratio: float = 0.0
    embed_flops_ratio: float = 0.0
    architectural_balance_score: float = 0.0


@dataclass
class T3Result:
    """T3: Compilation & Construction."""

    status: TestStatus = "PASS"
    test_id: str = "T3"
    actual_params: int = 0
    output_shape: list[int] = field(default_factory=list)
    has_nan: bool = False
    has_inf: bool = False
    grad_nan: bool = False
    grad_inf: bool = False
    params_no_grad: list[str] = field(default_factory=list)
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    gpu_memory: int = 0
    kill_reason: str | None = None
    tags: list[Tag] = field(default_factory=list)
    wall_seconds: float = 0.0

    # ══ Construction Diagnostics (enhanced T3) ═════════════════════════════════
    layer_shapes_consistent: bool = True
    forward_backward_consistent: bool = True
    init_scale_reasonable: bool = True
    construction_diagnostics: dict = field(default_factory=dict)


@dataclass
class T4Result:
    """T4: Signal Propagation."""

    status: TestStatus = "PASS"
    test_id: str = "T4"
    layer_activation_norms: dict[str, float] = field(default_factory=dict)
    layer_gradient_norms: dict[str, float] = field(default_factory=dict)
    gradient_norm_ratio: float = 1.0
    gradient_max_layer: str = ""
    gradient_min_layer: str = ""
    output_entropy: float = 0.0
    entropy_ratio: float = 0.0
    loss_at_init: float = 0.0
    kill_reason: str | None = None
    tags: list[Tag] = field(default_factory=list)
    wall_seconds: float = 0.0

    # Comparative signal analysis
    gradient_flow_health: float = 0.0
    """Ratio of layers with healthy gradients vs baseline (0-1)."""

    dead_neuron_ratio: float = 0.0
    """Percentage of "dead" activations (near-zero outputs, < 0.01)."""

    activation_distribution_shift: dict[str, float] = field(default_factory=dict)
    """Per-layer KL divergence from baseline activation norms (if calibration available)."""

    signal_to_noise_ratio: float = 0.0
    """Mean activation / std of activations (higher is better signal propagation)."""

    per_layer_snr: dict[str, float] = field(default_factory=dict)
    """Signal-to-noise ratio per layer for detailed diagnostics."""

    dead_neurons_per_layer: dict[str, float] = field(default_factory=dict)
    """Dead neuron ratio per layer for targeted diagnostics."""


@dataclass
class T5Result:
    """T5: Initialization Diagnostics."""

    status: TestStatus = "PASS"
    test_id: str = "T5"
    logit_max: float = 0.0
    logit_std: float = 0.0
    effective_ranks: dict[str, float] = field(default_factory=dict)
    condition_numbers: dict[str, float] = field(default_factory=dict)
    weight_symmetry: dict[str, float] = field(default_factory=dict)
    kurtosis_mean: float = 0.0
    effective_rank_mean: float = 0.0
    # Weight spectrum analysis (new)
    weight_spectrum_percentiles: dict[str, float] = field(default_factory=dict)
    init_symmetry_score: float = 0.0
    capacity_utilization: float = 0.0
    rank_deficiency_ratio: float = 0.0
    kill_reason: str | None = None
    tags: list[Tag] = field(default_factory=list)
    wall_seconds: float = 0.0


@dataclass
class T7Result:
    """T7: Micro-Training (100 Steps)."""

    status: TestStatus = "PASS"
    test_id: str = "T7"
    framework: str = "unknown"
    """Framework used for training: 'mlx', 'pytorch', or 'unknown'."""
    loss_trajectory: list[float] = field(default_factory=list)
    loss_at_1: float = 0.0
    loss_at_100: float = 0.0
    loss_drop: float = 0.0
    learning_ratio: float = 0.0
    gradient_norms: list[float] = field(default_factory=list)
    gradient_mean: float = 0.0
    gradient_stability: float = 0.0
    entropy_trajectory: list[float] = field(default_factory=list)
    component_speeds: dict[str, float] = field(default_factory=dict)
    component_speed_ratio: float = 1.0
    tokens_per_second: float = 0.0
    projected_full_run_seconds: float = 0.0
    kill_reason: str | None = None
    tags: list[Tag] = field(default_factory=list)
    wall_seconds: float = 0.0

    # ══ Learning Curve Analysis (Enhanced T7) ════════════════════════════════════
    learning_curve_shape: str = "unknown"
    """Shape of learning curve: monotonic, plateau, sawtooth, divergent, noisy."""

    convergence_trajectory: str = "unknown"
    """Convergence behavior: stable, oscillating, unstable, improving, degrading."""

    component_learning_order: dict[str, int] = field(default_factory=dict)
    """First step where each component's grad norm drops significantly: {"attn": 10, "mlp": 25}."""

    stepwise_instability_flags: list[int] = field(default_factory=list)
    """Step indices where gradient spiked >3x mean."""

    # ══ Projected Metrics (Enhanced T7) ══════════════════════════════════════════
    projected_convergence_step: int | None = None
    """Extrapolated step when loss would reach baseline (None if already there)."""

    projected_final_loss: float | None = None
    """Extrapolated loss at full convergence."""

    # ══ Per-step Component Tracking (Enhanced T7) ════════════════════════════════
    component_grad_norms_per_step: dict[str, list[float]] = field(default_factory=dict)
    """Per-component gradient norms at each step: {"attn": [0.1, 0.09, ...], ...}."""


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KillHypothesis:
    """A generated kill hypothesis from the LLM."""

    hypothesis_id: str
    confidence: Confidence
    failure_mode: str
    mechanism: str
    experiment_type: str  # isolation|temporal|component|interaction|absolute|relative
    experiment_spec: dict[str, Any]
    evidence: str


@dataclass
class S2Result:
    """Stage 2 adversarial prosecution result."""

    verdict: str = ""  # REFUTED or STAGE_2_PASSED
    killed_by: str | None = None
    kill_reason: str | None = None
    hypotheses: list[dict] = field(default_factory=list)
    hypothesis_results: list[dict] = field(default_factory=list)
    tag_results: list[dict] = field(default_factory=list)
    trend_verification: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Output structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Feedback:
    """Structured feedback for the Ideator."""

    one_line: str = ""
    stage_reached: int = 0  # 0=T2, 1=T3-T5, 2=T7, 3=Stage2, 4=FullPass
    failure_analysis: str | None = None  # LLM-written for S2
    suggested_directions: list[str] = field(default_factory=list)
    key_measurements: dict[str, Any] = field(default_factory=dict)


@dataclass
class FalsifierOutput:
    """Complete falsifier output."""

    theory_id: str = ""
    verdict: Verdict = "REJECTED"
    killed_by: str | None = None  # T0-T7, S2_H1, COMPOUND_TAGS, CORRELATED_TAGS_{cat}
    kill_reason: str | None = None

    # ══ Per-Test Results (Streamlined: T2→T3→{T4,T5}→T7, T0/T1/T6 removed) ═══
    t2_budget: T2Result | None = None
    t3_compilation: T3Result | None = None
    t4_signal: T4Result | None = None
    t5_init: T5Result | None = None
    t7_microtrain: T7Result | None = None
    s2_results: S2Result | None = None

    # ══ Aggregate ═══════════════════════════════════════════════════════════
    tags: list[Tag] = field(default_factory=list)
    feedback: Feedback = field(default_factory=Feedback)
    total_wall_seconds: float = 0.0
    total_gpu_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy compatibility structures (for migration)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Legacy validation result (pre-T0)."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSignature:
    """Legacy model signature."""

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
    """Legacy budget check."""

    total_artifact_bytes: int
    remaining_budget_bytes: int
    param_count_estimate: int
    flops_estimate: float
    within_budget: bool


@dataclass
class SmokeDiagnostics:
    """Legacy smoke diagnostics."""

    forward_ok: bool
    backward_ok: bool
    loss_is_finite: bool
    params_without_grad: list[str] = field(default_factory=list)


@dataclass
class MicroTrainDiagnostics:
    """Legacy micro-train diagnostics."""

    steps: int
    loss_first: float
    loss_last: float
    loss_drop: float
    throughput_steps_per_sec: float
    ok: bool


@dataclass
class InitGateDiagnostics:
    """Legacy init gate diagnostics."""

    candidate_kurtosis_mean: float
    candidate_effective_rank_mean: float
    baseline_kurtosis_mean: float | None
    baseline_effective_rank_mean: float | None
    ok: bool
    skipped: bool = False


@dataclass
class CitationGateDiagnostics:
    """Legacy citation gate diagnostics."""

    claims_checked: int
    mismatches: list[str]
    ok: bool
    skipped: bool = False


@dataclass
class Stage1Result:
    """Legacy Stage 1 result (for backward compatibility)."""

    theory_id: str
    verdict: Literal["reject", "refute", "promote", "implementation_fail"]
    validation: ValidationResult
    model_signature: ModelSignature | None = None
    budget: BudgetCheck | None = None
    smoke: SmokeDiagnostics | None = None
    stage_reached: str = "validation"
    reasons: list[str] = field(default_factory=list)
    comment: str = ""
    micro_train: MicroTrainDiagnostics | None = None
    init_gate: InitGateDiagnostics | None = None
    citation_gate: CitationGateDiagnostics | None = None
