"""
Knowledge graph node lifecycle management.

This module handles:
1. Creating nodes from ideator outputs
2. Updating node status with history tracking
3. Updating nodes with falsification results
4. Finding nodes by various criteria
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from falsifier.graph.locking import AtomicGraphUpdate, lock_context, atomic_read_json, atomic_write_json
from falsifier.types import FalsifierOutput, Tag


def create_node_from_ideator_idea(
    idea: dict,
    graph_path: Path,
    knowledge_dir: Path,
) -> str:
    """
    Create a new node from an ideator output.

    Creates a node with status=GENERATED, including all ideator fields.
    The node_id is derived from the idea_id.

    Node schema includes:
    - node_id: unique id
    - idea_id: the idea identifier
    - title: from ideator
    - status: current state
    - status_history: list of state transitions
    - source: ideator metadata
    - implementation: steps, parent_implementation, etc.
    - review: novelty score, decision, feedback
    - falsification: test results, metrics, kill info, timing

    Args:
        idea: The ideator output dictionary (schema_version: ideator.idea.v1)
        graph_path: Path to the graph.json file
        knowledge_dir: Path to the knowledge_graph directory

    Returns:
        The node_id for the created node
    """
    # Extract fields from ideator output
    idea_id = idea.get("idea_id", f"idea_{uuid.uuid4().hex[:8]}")
    node_id = idea_id

    title = idea.get("title", "")
    novelty_summary = idea.get("novelty_summary", "")
    expected_metric_change = idea.get("expected_metric_change", "")

    # Build parent references from parent_implementation
    parents = []
    parent_impl = idea.get("parent_implementation", {})
    if parent_impl:
        parents.append({
            "node_id": parent_impl.get("primary_file", "unknown"),
            "relationship": "builds_on",
            "what_changed": parent_impl.get("repo_url", ""),
        })

    # Build node data matching the plan schema
    node_data = {
        "node_id": node_id,
        "idea_id": idea_id,
        "title": title,
        "theory_type": "architectural",  # Default from ideator output
        "status": "GENERATED",
        "status_history": [
            {
                "status": "GENERATED",
                "timestamp": time.time(),
                "actor": "ideator",
                "metadata": {
                    "model": idea.get("meta", {}).get("model", "unknown"),
                    "generated_at": idea.get("meta", {}).get("generated_at"),
                }
            }
        ],
        "source": {
            "ideator_model": idea.get("meta", {}).get("model", "unknown"),
            "generated_at": idea.get("meta", {}).get("generated_at"),
            "schema_version": idea.get("schema_version", "ideator.idea.v1"),
        },
        "implementation": {
            "steps": idea.get("implementation_steps", []),
            "parent_implementation": parent_impl,
            "falsifier_smoke_tests": idea.get("falsifier_smoke_tests", []),
        },
        "review": {
            "novelty_summary": novelty_summary,
            "expected_metric_change": expected_metric_change,
            "decision": None,
            "feedback": None,
        },
        "what_and_why": novelty_summary,
        "expected_metric_change": expected_metric_change,
        "config_delta": {},  # To be extracted later from implementation_steps
        "new_components": [],  # To be populated later
        "parents": parents,
        "tags": [],
        "change_types": [],
        "measured_metrics": {},
        "falsification": None,
        "total_wall_seconds": 0.0,
        "total_gpu_seconds": 0.0,
    }

    # Atomically create the node using AtomicGraphUpdate
    atomic = AtomicGraphUpdate(graph_path)

    # Check if node already exists
    graph = atomic.read_graph()
    if node_id in graph.get("nodes", {}):
        # Node already exists, return existing node_id
        return node_id

    # Create the node
    atomic.create_node(node_id, node_data)

    return node_id


def update_node_status(
    node_id: str,
    new_status: str,
    graph_path: Path,
    actor: str = "system",
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Update a node's status with history tracking.

    Appends to status_history and updates the status field.
    Uses locking from falsifier.graph.locking for thread safety.

    Args:
        node_id: The node identifier
        new_status: The new status (e.g., "APPROVED", "IN_FALSIFICATION", "REFUTED")
        graph_path: Path to the graph.json file
        actor: Who/what made the change (default: "system")
        metadata: Optional additional metadata for this status change

    Raises:
        ValueError: If node does not exist
        TimeoutError: If lock cannot be acquired
    """
    atomic = AtomicGraphUpdate(graph_path)

    # Read current node data
    graph = atomic.read_graph()
    node = graph.get("nodes", {}).get(node_id)

    if node is None:
        raise ValueError(f"Node {node_id} not found in graph")

    # Build status history entry
    history_entry = {
        "status": new_status,
        "timestamp": time.time(),
        "actor": actor,
        "metadata": metadata or {},
    }

    # Get existing history or create new
    status_history = node.get("status_history", [])
    status_history.append(history_entry)

    # Update the node
    updates = {
        "status": new_status,
        "status_history": status_history,
    }

    atomic.update_node(node_id, updates)


def update_node_with_falsification_results(
    node_id: str,
    output: FalsifierOutput,
    graph_path: Path,
) -> None:
    """
    Update a node with falsification results.

    Updates the falsification section with all test results, metrics, kill info.
    Calculates bits_per_byte from loss_at_100 if available.

    Args:
        node_id: The node identifier
        output: The FalsifierOutput with all test results
        graph_path: Path to the graph.json file

    Raises:
        ValueError: If node does not exist
        TimeoutError: If lock cannot be acquired
    """
    atomic = AtomicGraphUpdate(graph_path)

    # Read current node data
    graph = atomic.read_graph()
    node = graph.get("nodes", {}).get(node_id)

    if node is None:
        raise ValueError(f"Node {node_id} not found in graph")

    # Build measured metrics from all test results
    measured_metrics: dict[str, Any] = {}

    if output.t2_budget:
        measured_metrics["estimated_params"] = output.t2_budget.estimated_params
        measured_metrics["budget_utilization"] = output.t2_budget.budget_utilization
        measured_metrics["flops_ratio"] = output.t2_budget.flops_ratio

    if output.t3_compilation:
        measured_metrics["actual_params"] = output.t3_compilation.actual_params
        measured_metrics["forward_ms"] = output.t3_compilation.forward_ms
        measured_metrics["backward_ms"] = output.t3_compilation.backward_ms
        measured_metrics["gpu_memory"] = output.t3_compilation.gpu_memory

    if output.t4_signal:
        measured_metrics["gradient_norm_ratio"] = output.t4_signal.gradient_norm_ratio
        measured_metrics["output_entropy"] = output.t4_signal.output_entropy
        measured_metrics["layer_activation_norms"] = output.t4_signal.layer_activation_norms
        measured_metrics["layer_gradient_norms"] = output.t4_signal.layer_gradient_norms
        measured_metrics["dead_neuron_ratio"] = output.t4_signal.dead_neuron_ratio
        measured_metrics["signal_to_noise_ratio"] = output.t4_signal.signal_to_noise_ratio

    if output.t5_init:
        measured_metrics["logit_max"] = output.t5_init.logit_max
        measured_metrics["effective_rank_mean"] = output.t5_init.effective_rank_mean
        measured_metrics["condition_numbers"] = output.t5_init.condition_numbers
        measured_metrics["capacity_utilization"] = output.t5_init.capacity_utilization

    if output.t7_microtrain:
        measured_metrics["loss_at_1"] = output.t7_microtrain.loss_at_1
        measured_metrics["loss_at_100"] = output.t7_microtrain.loss_at_100
        measured_metrics["loss_drop"] = output.t7_microtrain.loss_drop
        measured_metrics["learning_ratio"] = output.t7_microtrain.learning_ratio
        measured_metrics["tokens_per_second"] = output.t7_microtrain.tokens_per_second
        measured_metrics["gradient_mean"] = output.t7_microtrain.gradient_mean
        measured_metrics["gradient_stability"] = output.t7_microtrain.gradient_stability
        measured_metrics["learning_curve_shape"] = output.t7_microtrain.learning_curve_shape
        measured_metrics["convergence_trajectory"] = output.t7_microtrain.convergence_trajectory

    # Calculate bits_per_byte from loss_at_100 if available
    # bits_per_byte = loss * ln(2) = loss * 1.4427
    measured_bpb = None
    if output.t7_microtrain and output.t7_microtrain.loss_at_100:
        measured_bpb = output.t7_microtrain.loss_at_100 * 1.4427

    # Convert tags to serializable format
    tags_data = [
        {
            "tag_id": tag.tag_id,
            "test_id": tag.test_id,
            "category": tag.category,
            "description": tag.description,
        }
        for tag in output.tags
    ]

    # Build change types from tags
    change_types: set[str] = set()
    for tag in output.tags:
        if tag.category:
            change_types.add(tag.category)

    # Build falsification section
    falsification = {
        "outcome": output.verdict,
        "killed_by": output.killed_by,
        "kill_reason": output.kill_reason,
        "test_results": {
            "t2_budget": _serialize_test_result(output.t2_budget),
            "t3_compilation": _serialize_test_result(output.t3_compilation),
            "t4_signal": _serialize_test_result(output.t4_signal),
            "t5_init": _serialize_test_result(output.t5_init),
            "t7_microtrain": _serialize_test_result(output.t7_microtrain),
        },
        "metrics": {
            "measured_bpb": measured_bpb,
            "tokens_per_second": measured_metrics.get("tokens_per_second"),
            "loss_at_100": measured_metrics.get("loss_at_100"),
            "learning_ratio": measured_metrics.get("learning_ratio"),
        },
        "timing": {
            "total_wall_seconds": output.total_wall_seconds,
            "total_gpu_seconds": output.total_gpu_seconds,
        },
        "failure_analysis": None,
        "lessons": [],
        "suggested_alternatives": [],
    }

    # Add failure analysis if theory was refuted
    if output.verdict in ("REFUTED", "REJECTED", "IMPLEMENTATION_FAIL"):
        falsification["failure_analysis"] = {
            "root_cause": output.kill_reason or "Unknown",
            "failure_pattern": output.killed_by or "unknown",
            "lessons": output.feedback.suggested_directions if output.feedback else [],
            "suggested_alternatives": output.feedback.suggested_directions if output.feedback else [],
        }

    # Add stage 2 results if available
    if output.s2_results:
        falsification["stage_2"] = {
            "verdict": output.s2_results.verdict,
            "killed_by": output.s2_results.killed_by,
            "kill_reason": output.s2_results.kill_reason,
            "hypotheses_tested": len(output.s2_results.hypotheses),
        }

    # Build the complete update
    updates = {
        "status": output.verdict,
        "measured_metrics": measured_metrics,
        "measured_bpb": measured_bpb,
        "tags": tags_data,
        "change_types": list(change_types),
        "falsification": falsification,
        "total_wall_seconds": output.total_wall_seconds,
        "total_gpu_seconds": output.total_gpu_seconds,
    }

    # Update status history with final status
    status_history = node.get("status_history", [])
    status_history.append({
        "status": output.verdict,
        "timestamp": time.time(),
        "actor": "falsifier",
        "metadata": {
            "killed_by": output.killed_by,
            "total_wall_seconds": output.total_wall_seconds,
        }
    })
    updates["status_history"] = status_history

    # Apply the update
    atomic.update_node(node_id, updates)


def _serialize_test_result(result: Any) -> dict[str, Any] | None:
    """Helper to serialize a test result dataclass to dict."""
    if result is None:
        return None
    if hasattr(result, "__dataclass_fields__"):
        return {
            k: _serialize_value(v)
            for k, v in result.__dict__.items()
        }
    return dict(result)


def _serialize_value(value: Any) -> Any:
    """Helper to serialize a value for JSON storage."""
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _serialize_test_result(value)
    return value


def find_node_by_idea_id(graph_path: Path, idea_id: str) -> str | None:
    """
    Find the most recent node_id for a given idea_id.

    Args:
        graph_path: Path to the graph.json file
        idea_id: The idea_id to search for

    Returns:
        The node_id if found, None otherwise
    """
    if not graph_path.exists():
        return None

    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    nodes = graph.get("nodes", {})

    # First try exact match on node_id (which equals idea_id for new nodes)
    if idea_id in nodes:
        return idea_id

    # Otherwise search for nodes with matching idea_id field
    matching_nodes = []
    for node_id, node in nodes.items():
        if node.get("idea_id") == idea_id:
            # Get timestamp from status history for recency
            history = node.get("status_history", [])
            if history:
                last_timestamp = history[-1].get("timestamp", 0)
            else:
                last_timestamp = 0
            matching_nodes.append((node_id, last_timestamp))

    if not matching_nodes:
        return None

    # Return the most recent matching node
    matching_nodes.sort(key=lambda x: x[1], reverse=True)
    return matching_nodes[0][0]


def get_node_status(graph_path: Path, node_id: str) -> str | None:
    """
    Get the current status of a node.

    Args:
        graph_path: Path to the graph.json file
        node_id: The node identifier

    Returns:
        The status string if found, None otherwise
    """
    if not graph_path.exists():
        return None

    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    node = graph.get("nodes", {}).get(node_id)
    if node is None:
        return None

    return node.get("status")


def find_nodes_by_status(graph_path: Path, status: str) -> list[dict[str, Any]]:
    """
    Find all nodes with a given status.

    Args:
        graph_path: Path to the graph.json file
        status: The status to filter by

    Returns:
        List of node dictionaries with the given status
    """
    if not graph_path.exists():
        return []

    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    nodes = graph.get("nodes", {})
    return [node for node in nodes.values() if node.get("status") == status]


def get_node_full(graph_path: Path, node_id: str) -> dict[str, Any] | None:
    """
    Get the complete data for a node.

    Args:
        graph_path: Path to the graph.json file
        node_id: The node identifier

    Returns:
        The node dictionary if found, None otherwise
    """
    if not graph_path.exists():
        return None

    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    return graph.get("nodes", {}).get(node_id)
