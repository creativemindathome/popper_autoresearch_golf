"""
Structured context for human/agent review: what Stage 1 actually measures,
why gates are ordered as they are, minimal-env contract, and pytest coverage.

Stage 1 is intentionally **cheap, deterministic evidence** — not end-task quality.
This bundle makes that explicit so reviewers do not over-interpret promote/refute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from falsifier.adapters.parameter_golf import MINIMAL_TRAIN_GPT_ENV
from falsifier.types import CandidatePackage

# Must stay aligned with `instantiate_minimal_model` in parameter_golf.py.
_STAGE1_GPT_KWARGS: tuple[str, ...] = (
    "vocab_size",
    "num_layers",
    "model_dim",
    "num_heads",
    "num_kv_heads",
    "mlp_mult",
    "tie_embeddings",
    "tied_embed_init_std",
    "logit_softcap",
    "rope_base",
    "qk_gain_init",
)

# Gate id -> pytest node ids (repo-relative) that exercise that failure mode.
_PYTEST_GATE_COVERAGE: dict[str, list[str]] = {
    "validation": [
        "tests/falsifier/test_stage1.py::test_stage1_rejects_invalid_candidate",
        "tests/test_falsifier_core.py::FalsifierCoreTests::test_stage1_failure_includes_comment",
    ],
    "T2": ["tests/falsifier/test_stage1.py::test_stage1_refutes_over_budget_candidate"],
    "T3": [
        "tests/falsifier/test_stage1.py::test_stage1_fails_import_broken_candidate_at_t3",
        "tests/falsifier/test_stage1.py::test_stage1_fails_construction_broken_candidate_at_t3",
        "tests/test_falsifier_core.py::FalsifierCoreTests::test_smoke_diagnostics_runs_backward",
    ],
    "T4": ["tests/test_falsifier_core.py::FalsifierCoreTests::test_stage1_promotes_baseline_candidate"],
    "T5": ["tests/falsifier/test_t5_t6.py::test_t5_skipped_when_no_minimal_init_in_profile"],
    "T6": [
        "tests/falsifier/test_t5_t6.py::test_t6_refutes_on_bad_calibration_claim",
        "tests/falsifier/test_t5_t6.py::test_t6_passes_matching_claim",
    ],
    "promote": [
        "tests/falsifier/test_stage1.py::test_stage1_promotes_baseline_candidate_with_novel_explanation",
        "tests/falsifier/test_stage1.py::test_cli_writes_verdict_artifact",
    ],
}


def _gates_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "validation",
            "placement": "First: reject bad metadata before any import or torch work.",
            "measures": ["Candidate JSON sanity", "theory_id non-empty", "what_and_why length", "path is train_gpt.py"],
            "does_not_measure": ["Code quality", "Scientific truth of the theory", "Novelty vs prior work"],
            "information_value": "high_for_admission_plumbing",
            "low_signal_warning": "Long what_and_why can still be vacuous — only length is checked.",
        },
        {
            "id": "T2",
            "placement": "Before import: static budget from source + FLOPs estimate vs calibration.",
            "measures": ["Artifact size", "Estimated FLOPs vs baseline ratio"],
            "does_not_measure": ["Wall-clock cost", "Actual training FLOPs on full config", "Data pipeline"],
            "information_value": "medium_high_for_cost_explosion",
            "low_signal_warning": "Estimates can diverge from real runs; catches egregious blow-ups, not fine tuning.",
        },
        {
            "id": "T3",
            "placement": "After budget: minimal-env import, GPT build, forward/backward on toy batch.",
            "measures": ["Import works", "Constructor matches adapter contract", "Gradients reach parameters"],
            "does_not_measure": ["Full-scale hyperparameters", "Downstream accuracy", "Distributed correctness"],
            "information_value": "high_for_integration_breaks",
            "low_signal_warning": "Uses MINIMAL_TRAIN_GPT_ENV — passing T3 does not validate your default FineWeb run.",
        },
        {
            "id": "T5",
            "placement": "After smoke, before long micro-train: init statistics vs baseline (or skip if no baseline).",
            "measures": ["Weight kurtosis / effective-rank band vs calibration minimal-init snapshot"],
            "does_not_measure": ["Full-model init", "Whether theory's mechanism is sound"],
            "information_value": "medium_when_not_skipped",
            "low_signal_warning": "Skipped when minimal_init_baseline missing — promote does not imply init checked.",
        },
        {
            "id": "T4",
            "placement": "After T5: short CPU micro-train vs calibration loss-drop floor.",
            "measures": ["Loss decreases over N steps under same minimal policy as baseline profile"],
            "does_not_measure": ["Convergence quality", "Generalization", "Optimal LR"],
            "information_value": "medium_high_for_learning_plumbing",
            "low_signal_warning": "A promote only means 'learns a bit' in the harness — not SOTA.",
        },
        {
            "id": "T6",
            "placement": "Last: optional numeric claims vs calibration_lite (T6a).",
            "measures": ["Claimed calibration numbers match profile when claims present"],
            "does_not_measure": ["Whether claims are scientifically meaningful"],
            "information_value": "high_when_claims_present",
            "low_signal_warning": "Empty calibration_claims skips T6 — no citation enforcement by default.",
        },
    ]


def _ordering_rationale() -> str:
    return (
        "validation → T2 → T3 → T5 → T4 → T6: fail fast on cheap checks (metadata, static budget) "
        "before importing torch; smoke before expensive micro-train; init check before long run; "
        "citations last so they apply to a candidate that already executes."
    )


def build_agent_review_prompt(candidate: CandidatePackage, repo_root: Path) -> str:
    tg = Path(candidate.train_gpt_path).resolve()
    kwargs = ", ".join(_STAGE1_GPT_KWARGS)
    return f"""You are reviewing a candidate change to `train_gpt.py` for alignment with the repo falsifier Stage 1 harness.

**Theory (author intent)**
theory_id: {candidate.theory_id}
what_and_why:
{candidate.what_and_why}

**File under review**
{tg}

**Structural contract (Stage 1 smoke / micro-train)**
- Module must define `Hyperparameters` and `GPT` as loaded by the falsifier adapter.
- `GPT.__init__` must accept these keyword arguments (minimal run): {kwargs}
- Stage 1 runs with a **minimal environment** (see `review_bundle.minimal_env_for_stage1` in JSON output), not your default FineWeb-scale env.

**Your tasks**
1. Check whether the **code change** plausibly implements what `what_and_why` claims (architecture, routing, init, etc.).
2. Flag any change that would break the constructor contract or the minimal-env assumptions.
3. List which **pytest** nodes from `review_bundle.pytest_gates_matrix` should be run after edits; prefer `uv run pytest tests/falsifier/test_stage1.py tests/test_falsifier_core.py tests/falsifier/test_t5_t6.py -q`.

**Do not** treat Stage 1 `promote` as proof of benchmark wins — it only validates the harness signals in `review_bundle.gates`.
"""


def build_stage1_review_bundle(candidate: CandidatePackage, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root if repo_root is not None else Path(candidate.train_gpt_path).resolve().parent
    train_path = str(Path(candidate.train_gpt_path).resolve())
    return {
        "schema_version": "1",
        "summary": (
            "Stage 1 adds **targeted, low-dimensional** evidence (budget, smoke, optional init/calibration claims, "
            "short micro-train). It is **not** a substitute for task metrics or peer review of the theory. "
            "Use this bundle to place that evidence correctly and drive agent/human review of `train_gpt.py` vs tests."
        ),
        "ordering_rationale": _ordering_rationale(),
        "gates": _gates_catalog(),
        "minimal_env_for_stage1": dict(MINIMAL_TRAIN_GPT_ENV),
        "gpt_constructor_keywords_stage1": list(_STAGE1_GPT_KWARGS),
        "pytest_gates_matrix": _PYTEST_GATE_COVERAGE,
        "recommended_pytest_cmd": (
            "uv run pytest tests/falsifier/test_stage1.py tests/test_falsifier_core.py "
            "tests/falsifier/test_t5_t6.py -q"
        ),
        "candidate": {
            "theory_id": candidate.theory_id,
            "train_gpt_path": train_path,
            "repo_root": str(root),
            "what_and_why": candidate.what_and_why,
            "has_calibration_claims": bool(getattr(candidate, "calibration_claims", None)),
        },
        "agent_review_prompt": build_agent_review_prompt(candidate, root),
    }
