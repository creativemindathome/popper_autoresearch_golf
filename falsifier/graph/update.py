"""
Knowledge graph update utilities.

Handles appending nodes to the graph JSON after falsifier verdict.
Referenced in PRD lines 1948-1952.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..types import FalsifierInput, FalsifierOutput, Tag


def update_graph_after_verdict(
    graph_path: Path,
    output: FalsifierOutput,
    inp: FalsifierInput,
) -> None:
    """
    Append a new node to the knowledge graph after falsifier verdict.

    Creates a node with:
    - theory_id, status=verdict
    - config_delta from input
    - measured metrics from output
    - kill_reason if applicable
    - parents from input
    - tags from output

    Args:
        graph_path: Path to graph JSON file (created if doesn't exist)
        output: FalsifierOutput with verdict and results
        inp: FalsifierInput with theory details
    """
    # Load or create graph
    if graph_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        if not isinstance(graph, dict):
            graph = {"nodes": [], "edges": []}
    else:
        graph = {"nodes": [], "edges": []}

    # Build node ID
    theory_id = getattr(inp, "theory_id", None) or output.theory_id
    node_id = theory_id or f"theory_{uuid.uuid4().hex[:8]}"

    # Extract measured metrics from results
    measured_metrics: dict[str, Any] = {}

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

    if output.t5_init:
        measured_metrics["logit_max"] = output.t5_init.logit_max
        measured_metrics["effective_rank_mean"] = output.t5_init.effective_rank_mean
        measured_metrics["condition_numbers"] = output.t5_init.condition_numbers

    if output.t7_microtrain:
        measured_metrics["loss_at_1"] = output.t7_microtrain.loss_at_1
        measured_metrics["loss_at_100"] = output.t7_microtrain.loss_at_100
        measured_metrics["loss_drop"] = output.t7_microtrain.loss_drop
        measured_metrics["learning_ratio"] = output.t7_microtrain.learning_ratio
        measured_metrics["tokens_per_second"] = output.t7_microtrain.tokens_per_second

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

    # Build parent references
    parents = getattr(inp, "parents", []) or []
    parents_data = [
        {
            "node_id": parent.node_id,
            "relationship": parent.relationship,
            "what_changed": parent.what_changed,
        }
        for parent in parents
    ]

    # Build change types from tags
    change_types: set[str] = set()
    for tag in output.tags:
        if tag.category:
            change_types.add(tag.category)

    # Create node
    node = {
        "node_id": node_id,
        "theory_id": theory_id,
        "theory_type": getattr(inp, "theory_type", "architectural"),
        "status": output.verdict,
        "config_delta": getattr(inp, "config_delta", None) or {},
        "new_components": [
            {
                "name": c.name,
                "injection_point": c.injection_point,
                "init_gate": c.init_gate,
            }
            for c in (getattr(inp, "new_components", None) or [])
        ],
        "measured_metrics": measured_metrics,
        "measured_bpb": measured_metrics.get("loss_at_100", 0.0) * 1.4427 if measured_metrics.get("loss_at_100") else None,
        "kill_reason": output.kill_reason,
        "killed_by": output.killed_by,
        "parents": parents_data,
        "tags": tags_data,
        "change_types": list(change_types),
        "what_and_why": getattr(inp, "what_and_why", ""),
        "total_wall_seconds": output.total_wall_seconds,
        "total_gpu_seconds": output.total_gpu_seconds,
    }

    nodes_obj = graph.get("nodes")
    if isinstance(nodes_obj, dict):
        nodes_obj[node_id] = node
    else:
        if not isinstance(nodes_obj, list):
            nodes_obj = []
            graph["nodes"] = nodes_obj

        # Use the existing ideator-style node id if present; otherwise canonicalize.
        list_node_id = node_id if str(node_id).startswith("idea_") else f"idea_{node_id}"

        existing: dict[str, Any] | None = None
        for n in nodes_obj:
            if isinstance(n, dict) and (n.get("id") == list_node_id or n.get("node_id") == list_node_id):
                existing = n
                break
        if existing is None:
            existing = {"id": list_node_id, "type": "Idea", "label": node_id}
            nodes_obj.append(existing)

        # Merge falsifier fields onto the ideator-style node.
        existing.update(node)
        existing.setdefault("id", list_node_id)

    # Create edges from parents
    for parent in parents:
        edge = {
            "source": parent.node_id,
            "target": node_id if isinstance(graph.get("nodes"), dict) else (node_id if str(node_id).startswith("idea_") else f"idea_{node_id}"),
            "relationship": parent.relationship,
            "what_changed": parent.what_changed,
        }
        if isinstance(graph.get("edges"), list):
            graph["edges"].append(edge)
        else:
            graph["edges"] = [edge]

    # Save graph
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, default=str)
