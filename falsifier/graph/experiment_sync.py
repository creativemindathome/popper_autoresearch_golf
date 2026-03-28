"""
Experiment-Knowledge Graph sync utilities.

This module provides lifecycle helpers for the autonomous experiment loop
to align with the CLI falsifier's graph update patterns.

Key functions:
- create_node_from_experiment_idea: Initialize a node before falsification
- build_falsifier_output_from_results: Convert experiment results to FalsifierOutput
- sync_experiment_results: Canonical graph update using lifecycle helpers
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from falsifier.graph.lifecycle import (
    AtomicGraphUpdate,
    update_node_status,
    update_node_with_falsification_results,
    find_node_by_idea_id,
)
from falsifier.types import (
    FalsifierOutput,
    FalsifierInput,
    T2Result,
    T3Result,
    T4Result,
    T5Result,
    T7Result,
    S2Result,
    Tag,
    Feedback,
)


def _idea_with_canonical_id(idea: dict, theory_id: str) -> dict:
    """Node key and FalsifierOutput always use ``theory_id``; idea JSON may omit ``idea_id``."""
    tid = theory_id.strip()
    return {**idea, "idea_id": tid, "theory_id": tid}


def create_node_from_experiment_idea(
    idea: dict,
    graph_path: Path,
    model: str = "unknown",
) -> str:
    """
    Create a new node from an experiment-generated idea.

    This aligns with create_node_from_ideator_idea and uses the CLI ideator
    schema (ideator.idea.v2) as the source of truth.

    Args:
        idea: The experiment idea dict with keys like idea_id, title,
              novelty_summary, train_gpt_code, etc. (ideator.idea.v2 schema)
        graph_path: Path to the graph.json file
        model: The model used for generation (e.g., "claude-sonnet-4-20250514")

    Returns:
        The node_id (equal to idea_id)
    """
    idea_id = idea["idea_id"]

    node_id = idea_id
    title = idea.get("title", idea_id)

    # Build node data matching the canonical schema using v2 fields
    node_data = {
        "node_id": node_id,
        "idea_id": idea_id,
        "title": title,
        "theory_type": "architectural",  # Experiment generates architectural theories
        "status": "GENERATED",
        "status_history": [
            {
                "status": "GENERATED",
                "timestamp": time.time(),
                "actor": "anthropic-ideator",
                "metadata": {
                    "model": model,
                    "source": "experiment",
                    "schema_version": "ideator.idea.v2",
                }
            }
        ],
        "source": {
            "type": "anthropic-ideator",
            "ideator_model": model,
            "generated_at": time.time(),
            "schema_version": "ideator.idea.v2",
        },
        "what_and_why": idea.get("novelty_summary", ""),  # v2 field
        "expected_metric_change": idea.get("expected_metric_change", ""),  # v2 field
        "novelty_claims": [],  # v2 uses implementation_steps instead
        "parameter_estimate": "",  # Included in expected_metric_change in v2
        "parent_architecture": idea.get("parent_implementation", {}).get("primary_file", "GPT-2 small (modified)"),
        "risk_factors": idea.get("falsifier_smoke_tests", []),  # v2 field
        "implementation": {
            "train_gpt_code": idea.get("train_gpt_code", ""),  # v2 field (inline)
            "falsifier_smoke_tests": idea.get("falsifier_smoke_tests", []),
            "implementation_steps": idea.get("implementation_steps", []),
        },
        "review": {
            "novelty_summary": idea.get("novelty_summary", ""),
            "expected_metric_change": idea.get("expected_metric_change", ""),
            "decision": None,
            "feedback": None,
            "novelty_claims": [],
        },
        "measured_metrics": {},
        "tags": [],
        "change_types": [],
        "falsification": None,
        "parents": [],
        "total_wall_seconds": 0.0,
        "total_gpu_seconds": 0.0,
    }

    # Atomically create the node
    atomic = AtomicGraphUpdate(graph_path)

    # Check if node already exists
    graph = atomic.read_graph()
    if node_id in graph.get("nodes", {}):
        # Node already exists, just return the id
        return node_id

    # Create the node
    atomic.create_node(node_id, node_data)

    return node_id


def init_node_for_falsification(
    node_id: str,
    graph_path: Path,
    actor: str = "falsifier",
) -> None:
    """
    Mark a node as IN_FALSIFICATION before running tests.

    This aligns with the CLI's behavior in load_from_ideator_inbox.

    Args:
        node_id: The node identifier
        graph_path: Path to the graph.json file
        actor: Who is running falsification (default: "falsifier")
    """
    update_node_status(
        node_id=node_id,
        new_status="IN_FALSIFICATION",
        graph_path=graph_path,
        actor=actor,
        metadata={
            "started_at": time.time(),
            "source": "experiment",
        }
    )


def _rebuild_dataclass(cls, data: dict | None) -> Any:
    """Helper to rebuild a dataclass from a dict, filtering to valid fields."""
    if not data or not isinstance(data, dict):
        return None
    try:
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        valid = {k: v for k, v in data.items() if k in field_names}
        return cls(**valid)
    except Exception:
        return None


def _rebuild_tags(tags_data: list[dict] | None) -> list[Tag]:
    """Rebuild Tag objects from serialized tag dicts."""
    if not tags_data:
        return []
    tags = []
    for t in tags_data:
        if isinstance(t, dict) and "tag_id" in t:
            try:
                tags.append(Tag(
                    tag_id=t["tag_id"],
                    test_id=t.get("test_id", ""),
                    category=t.get("category", ""),
                    description=t.get("description", t.get("detail", "")),
                ))
            except Exception:
                pass
    return tags


def build_falsifier_output_from_results(
    theory_id: str,
    stage1_result: dict | None,
    stage2_result: dict | None,
    feedback: Feedback | None = None,
    total_wall_seconds: float = 0.0,
    total_gpu_seconds: float = 0.0,
) -> FalsifierOutput:
    """
    Build a proper FalsifierOutput from experiment stage results.

    This converts the serialized dict format (from run_stage1) back to
    typed dataclasses for use with update_node_with_falsification_results.

    Args:
        theory_id: The theory identifier
        stage1_result: Dict from run_stage1 (serialized via asdict)
        stage2_result: Dict from run_stage2 (optional)
        feedback: Structured feedback object (optional)
        total_wall_seconds: Total wall clock time
        total_gpu_seconds: Total GPU time

    Returns:
        FalsifierOutput ready for lifecycle update
    """
    if stage1_result is None:
        stage1_result = {}

    # Rebuild Stage 1 test results
    t2_budget = _rebuild_dataclass(T2Result, stage1_result.get("t2_budget"))
    t3_compilation = _rebuild_dataclass(T3Result, stage1_result.get("t3_compilation"))
    t4_signal = _rebuild_dataclass(T4Result, stage1_result.get("t4_signal"))
    t5_init = _rebuild_dataclass(T5Result, stage1_result.get("t5_init"))
    t7_microtrain = _rebuild_dataclass(T7Result, stage1_result.get("t7_microtrain"))

    # Rebuild tags
    tags = _rebuild_tags(stage1_result.get("tags"))

    # Determine verdict
    verdict = stage1_result.get("verdict", "UNKNOWN")
    killed_by = stage1_result.get("killed_by")
    kill_reason = stage1_result.get("kill_reason")

    # Build S2Result if Stage 2 was run
    s2_results = None
    if stage2_result:
        s2_verdict = stage2_result.get("verdict", verdict)
        s2_killed_by = stage2_result.get("killed_by")
        s2_kill_reason = stage2_result.get("kill_reason")

        # If Stage 2 killed it, update the verdict
        if s2_verdict in ("REFUTED", "REJECTED"):
            verdict = s2_verdict
            killed_by = s2_killed_by or killed_by
            kill_reason = s2_kill_reason or kill_reason

        s2_results = S2Result(
            verdict=s2_verdict,
            killed_by=s2_killed_by,
            kill_reason=s2_kill_reason,
            hypotheses=[],  # Not stored in stage2_result dict
            hypothesis_results=[],  # Not stored in stage2_result dict
        )

    return FalsifierOutput(
        theory_id=theory_id,
        verdict=verdict,
        killed_by=killed_by,
        kill_reason=kill_reason,
        t2_budget=t2_budget,
        t3_compilation=t3_compilation,
        t4_signal=t4_signal,
        t5_init=t5_init,
        t7_microtrain=t7_microtrain,
        s2_results=s2_results,
        tags=tags,
        feedback=feedback or Feedback(),
        total_wall_seconds=total_wall_seconds,
        total_gpu_seconds=total_gpu_seconds,
    )


def sync_experiment_results(
    theory_id: str,
    idea: dict,
    stage1_result: dict | None,
    stage2_result: dict | None,
    graph_path: Path,
    start_time: float,
    end_time: float | None = None,
    falsifier_feedback: dict | None = None,
    model: str = "unknown",
) -> None:
    """
    Sync experiment results to the knowledge graph using canonical lifecycle.

    This is the main entry point - it handles the full flow:
    1. Ensures node exists (creates if needed)
    2. Builds FalsifierOutput from results
    3. Calls update_node_with_falsification_results for consistent schema

    Args:
        theory_id: The theory/node identifier
        idea: The original idea dict (for creating node if needed)
        stage1_result: Results from Stage 1 (optional for validation failures)
        stage2_result: Results from Stage 2 (optional)
        graph_path: Path to the graph.json file
        start_time: Experiment start timestamp
        end_time: Experiment end timestamp (defaults to now)
        falsifier_feedback: Dict with feedback fields (optional)
        model: Model used for generation
    """
    end_time = end_time or time.time()
    total_wall_seconds = end_time - start_time

    # Build feedback object if provided
    feedback = None
    if falsifier_feedback:
        feedback = Feedback(
            one_line=falsifier_feedback.get("one_line", ""),
            stage_reached=falsifier_feedback.get("stage_reached", 0),
            failure_analysis=falsifier_feedback.get("failure_analysis"),
            suggested_directions=falsifier_feedback.get("suggested_directions", []),
            key_measurements=falsifier_feedback.get("key_measurements", {}),
        )

    # Build FalsifierOutput
    output = build_falsifier_output_from_results(
        theory_id=theory_id,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
        feedback=feedback,
        total_wall_seconds=total_wall_seconds,
        total_gpu_seconds=0.0,  # Experiment doesn't track GPU separately
    )

    # Ensure node exists
    atomic = AtomicGraphUpdate(graph_path)
    graph = atomic.read_graph()

    idea_canon = _idea_with_canonical_id(idea, theory_id)
    if theory_id not in graph.get("nodes", {}):
        create_node_from_experiment_idea(idea_canon, graph_path, model)

    # Use the canonical lifecycle helper for the update
    update_node_with_falsification_results(theory_id, output, graph_path)


def get_or_create_node(
    theory_id: str,
    idea: dict,
    graph_path: Path,
    model: str = "unknown",
) -> str:
    """
    Get existing node_id or create a new node for the experiment.

    Args:
        theory_id: The theory identifier
        idea: The idea dict for creating node if needed
        graph_path: Path to the graph.json file
        model: Model used for generation

    Returns:
        The node_id (equal to theory_id)
    """
    # Try to find existing node
    existing = find_node_by_idea_id(graph_path, theory_id)
    if existing:
        return existing

    return create_node_from_experiment_idea(
        _idea_with_canonical_id(idea, theory_id),
        graph_path,
        model,
    )
