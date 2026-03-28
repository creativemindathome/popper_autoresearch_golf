"""
Stage 1 orchestrator: streamlined dependency graph (T0/T1/T6 removed).

Dependency graph:
T2 (Budget) ──→ T3 (Compilation)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼
   T4 (Signal)    T5 (Init)
        │               │
        └───────────────┘
                │
                ▼
          T7 (Micro-Train)

Rationale for removals:
- T0 (Novelty): Low-boldness theories fail T3-T7 anyway; adds no unique signal
- T1 (Precedent): Graph lookup belongs in ideator, not falsifier
- T6 (Citation): Moved to Stage 2 for LLM context building

Theory-type routing (should_skip):
- T4, T5: skip for "training", "data" (no model structure changes)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..types import (
    TAG_CATEGORIES,
    Calibration,
    FalsifierInput,
    FalsifierOutput,
    Feedback,
    KnowledgeGraph,
    Tag,
    T2Result,
    T3Result,
    T4Result,
    T5Result,
    T7Result,
    TestStatus,
    Verdict,
)
from ..validation import validate_candidate_package
from .t2_budget import run_t2
from .t3_compilation import run_t3
from .t4_signal import run_t4
from .t5_init import run_t5
from .t7_microtrain import run_t7

# Streamlined test execution order and dependencies (T0/T1/T6 removed)
TEST_SCHEDULE: list[tuple[str, Any, list[str]]] = [
    ("T2", run_t2, []),
    ("T3", run_t3, ["T2"]),
    ("T4", run_t4, ["T3"]),
    ("T5", run_t5, ["T3"]),
    ("T7", run_t7, ["T3", "T4|TAG", "T5|TAG"]),
    # T7 runs if T4/T5 are PASS or TAG (not FATAL or SKIP)
]


def should_skip(test_id: str, inp: FalsifierInput) -> tuple[bool, str]:
    """Theory-type routing. Returns (should_skip, reason)."""
    theory_type = inp.theory_type
    
    # T4, T5: skip for training-only or data-only theories (no model structure changes)
    if test_id in ("T4", "T5") and theory_type in ("training", "data"):
        return True, f"theory_type='{theory_type}' has no architectural changes"
    
    return False, ""


def dependencies_met(test_id: str, deps: list[str], results: dict[str, Any]) -> tuple[bool, str]:
    """Check if dependencies are satisfied.
    
    Dep syntax: "T4" (must be PASS), "T4|TAG" (PASS or FAIL_TAG allowed)
    """
    for dep in deps:
        allow_tag = dep.endswith("|TAG")
        dep_id = dep.replace("|TAG", "")
        
        if dep_id not in results:
            return False, f"dependency {dep_id} not yet run"
        
        result = results[dep_id]
        status = getattr(result, "status", "SKIP")
        
        if status == "FAIL_FATAL":
            return False, f"dependency {dep_id} failed"
        
        if status == "SKIP" and not allow_tag:
            return False, f"dependency {dep_id} skipped"
    
    return True, ""


def _infer_stage(killed_by: str | None) -> int:
    """Infer stage reached from killed_by test ID."""
    if killed_by is None:
        return 4  # Stage 2 passed
    if killed_by.startswith("S2_"):
        return 4
    if killed_by.startswith("T2"):
        return 0
    if killed_by.startswith("T3") or killed_by.startswith("T4") or killed_by.startswith("T5"):
        return 1
    if killed_by.startswith("T7"):
        return 3
    return 0


def run_stage_1(inp: FalsifierInput) -> FalsifierOutput:
    """Execute Stage 1: fixed battery of 8 tests with dependency graph."""
    start_time = time.time()
    results: dict[str, Any] = {}
    tags: list[Tag] = []
    total_gpu_seconds = 0.0
    
    # Pre-validation
    from ..types import CandidatePackage
    from ..validation import validate_candidate_package
    
    # Convert to CandidatePackage for validation
    if isinstance(inp, CandidatePackage) or hasattr(inp, 'theory_id'):
        validation = validate_candidate_package(inp)
    else:
        validation = validate_candidate_package(inp)
    
    if not validation.ok:
        return FalsifierOutput(
            theory_id=inp.theory_id,
            verdict="REJECTED",
            killed_by="VALIDATION",
            kill_reason="; ".join(validation.reasons),
            feedback=Feedback(
                one_line="Validation failed: " + "; ".join(validation.reasons),
                stage_reached=0,
            ),
            total_wall_seconds=time.time() - start_time,
        )
    
    # Load calibration if not provided
    calibration = inp.calibration
    if calibration is None:
        calibration = _load_or_default_calibration(inp)
    
    # Execute tests in order
    for test_id, test_fn, deps in TEST_SCHEDULE:
        # Check skip conditions
        skip, skip_reason = should_skip(test_id, inp)
        if skip:
            from dataclasses import fields
            # Create empty result with SKIP status
            result_type = _get_result_type(test_id)
            result_data = {f.name: None for f in fields(result_type) if f.name not in ("status", "test_id", "wall_seconds")}
            result_data["status"] = "SKIP"
            result_data["test_id"] = test_id
            result_data["wall_seconds"] = 0.0
            results[test_id] = result_type(**result_data)
            continue
        
        # Check dependencies
        deps_ok, deps_msg = dependencies_met(test_id, deps, results)
        if not deps_ok:
            from dataclasses import fields
            result_type = _get_result_type(test_id)
            result_data = {f.name: None for f in fields(result_type) if f.name not in ("status", "test_id", "wall_seconds")}
            result_data["status"] = "SKIP"
            result_data["test_id"] = test_id
            result_data["wall_seconds"] = 0.0
            results[test_id] = result_type(**result_data)
            continue
        
        # Run the test
        try:
            result = test_fn(inp)
            results[test_id] = result
            
            # Accumulate tags
            if hasattr(result, "tags"):
                tags.extend(result.tags)
            
            # Check for FATAL
            if result.status == "FAIL_FATAL":
                wall_seconds = time.time() - start_time
                return _build_refuted_output(
                    inp, results, tags, test_id, result.kill_reason or f"Failed at {test_id}",
                    wall_seconds, total_gpu_seconds
                )
                
        except Exception as e:
            wall_seconds = time.time() - start_time
            return _build_refuted_output(
                inp, results, tags, test_id, f"Exception at {test_id}: {e}",
                wall_seconds, total_gpu_seconds
            )
    
    # Compound kill: >= 3 tags from any tests
    if len(tags) >= 3:
        wall_seconds = time.time() - start_time
        return _build_refuted_output(
            inp, results, tags, "COMPOUND_TAGS",
            f"{len(tags)} tags accumulated across tests (>=3 threshold)",
            wall_seconds, total_gpu_seconds
        )
    
    # Correlated kill: >= 2 tags from different tests in same category
    for category, member_ids in TAG_CATEGORIES.items():
        hit_tests: set[str] = set()
        for tag in tags:
            if tag.tag_id in member_ids:
                hit_tests.add(tag.test_id)
        if len(hit_tests) >= 2:
            wall_seconds = time.time() - start_time
            return _build_refuted_output(
                inp, results, tags, f"CORRELATED_TAGS_{category}",
                f"{len(hit_tests)} tests in category '{category}' have correlated tags",
                wall_seconds, total_gpu_seconds
            )
    
    # Stage 1 passed
    wall_seconds = time.time() - start_time
    return FalsifierOutput(
        theory_id=inp.theory_id,
        verdict="STAGE_1_PASSED",
        killed_by=None,
        kill_reason=None,
        t2_budget=results.get("T2"),
        t3_compilation=results.get("T3"),
        t4_signal=results.get("T4"),
        t5_init=results.get("T5"),
        t7_microtrain=results.get("T7"),
        tags=tags,
        feedback=Feedback(
            one_line="Stage 1 passed all gates",
            stage_reached=3,
        ),
        total_wall_seconds=wall_seconds,
        total_gpu_seconds=total_gpu_seconds,
    )


def _get_result_type(test_id: str) -> type:
    """Get the result dataclass type for a test ID."""
    mapping = {
        "T2": T2Result,
        "T3": T3Result,
        "T4": T4Result,
        "T5": T5Result,
        "T7": T7Result,
    }
    return mapping.get(test_id, T2Result)


def _build_refuted_output(
    inp: FalsifierInput,
    results: dict[str, Any],
    tags: list[Tag],
    killed_by: str,
    kill_reason: str,
    wall_seconds: float,
    gpu_seconds: float,
) -> FalsifierOutput:
    """Build a REFUTED output."""
    stage = _infer_stage(killed_by)
    
    return FalsifierOutput(
        theory_id=inp.theory_id,
        verdict="REFUTED",
        killed_by=killed_by,
        kill_reason=kill_reason,
        t2_budget=results.get("T2"),
        t3_compilation=results.get("T3"),
        t4_signal=results.get("T4"),
        t5_init=results.get("T5"),
        t7_microtrain=results.get("T7"),
        tags=tags,
        feedback=Feedback(
            one_line=kill_reason,
            stage_reached=stage,
            failure_analysis=None,
            suggested_directions=[],
            key_measurements={},
        ),
        total_wall_seconds=wall_seconds,
        total_gpu_seconds=gpu_seconds,
    )


def _load_or_default_calibration(inp: FalsifierInput) -> Calibration:
    """Load calibration from profile or return defaults."""
    # Try to load from profile
    if inp.train_gpt_path:
        repo_root = Path(inp.train_gpt_path).resolve().parent
        profile_path = repo_root / "research" / "profiles" / "latest_baseline_profile.json"
        if profile_path.exists():
            try:
                import json
                from ..calibration_lite import extract_calibration_lite_from_profile
                
                profile = json.loads(profile_path.read_text())
                cl = extract_calibration_lite_from_profile(profile)
                
                if cl and "micro_train_100_step" in cl:
                    mt = cl["micro_train_100_step"]
                    return Calibration(
                        baseline_100=Baseline100(
                            loss_drop_mean=mt.get("loss_drop", 0.0),
                            loss_at_1_mean=mt.get("loss_first", 0.0),
                            loss_at_100_mean=mt.get("loss_last", 0.0),
                            tokens_per_second_mean=mt.get("throughput_steps_per_sec", 0.0),
                        ),
                    )
            except Exception:
                pass
    
    # Return defaults
    return Calibration()


# Backward compatibility
run_stage1 = run_stage_1
