"""
Knowledge graph query utilities.

Provides functions to load and query the falsifier knowledge graph
for precedent checking and interpolation analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from falsifier.types import KnowledgeGraph


def load_graph(graph_path: Path) -> KnowledgeGraph:
    """
    Load a KnowledgeGraph from a JSON file.

    Args:
        graph_path: Path to the JSON file containing graph data

    Returns:
        KnowledgeGraph instance populated with nodes and edges

    Raises:
        FileNotFoundError: If graph_path does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {graph_path}")

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def _node_key(node: Any, idx: int) -> str:
        if isinstance(node, dict):
            raw = node.get("node_id") or node.get("id")
            if raw:
                return str(raw)
        return str(idx)

    # Handle both flat and nested JSON structures
    if isinstance(data, dict):
        if "nodes" in data:
            raw_nodes = data["nodes"]
            if isinstance(raw_nodes, dict):
                nodes = {str(k): v for k, v in raw_nodes.items() if isinstance(v, dict)}
            elif isinstance(raw_nodes, list):
                nodes = {_node_key(n, i): n for i, n in enumerate(raw_nodes) if isinstance(n, dict)}
        else:
            # Assume flat structure where keys are node_ids
            nodes = {str(k): v for k, v in data.items() if isinstance(v, dict)}

        if "edges" in data:
            raw_edges = data["edges"]
            if isinstance(raw_edges, list):
                edges = raw_edges
    elif isinstance(data, list):
        # Assume list of nodes
        nodes = {_node_key(n, i): n for i, n in enumerate(data) if isinstance(n, dict)}

    return KnowledgeGraph(graph_path=graph_path, nodes=nodes, edges=edges)


def save_graph(graph: KnowledgeGraph, graph_path: Path | None = None) -> None:
    """
    Save a KnowledgeGraph to a JSON file.

    Args:
        graph: KnowledgeGraph to save
        graph_path: Optional path to save to (defaults to graph.graph_path)
    """
    target_path = graph_path or graph.graph_path
    if target_path is None:
        raise ValueError("No graph_path specified for saving")

    data = {
        "nodes": list(graph.nodes.values()),
        "edges": graph.edges,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_node(
    graph: KnowledgeGraph,
    node_id: str,
    status: str,
    what_and_why: str,
    theory_type: str,
    config_delta: dict[str, Any] | None = None,
    failure_reason: str | None = None,
    measured_bpb: float | None = None,
    change_types: set[str] | None = None,
) -> None:
    """
    Add a new node to the knowledge graph.

    Args:
        graph: KnowledgeGraph to modify
        node_id: Unique identifier for the node
        status: Node status (e.g., "REFUTED", "STAGE_2_PASSED")
        what_and_why: Theory description
        theory_type: Type of theory (architectural, training, data, etc.)
        config_delta: Optional config changes
        failure_reason: Optional reason for refutation
        measured_bpb: Optional measured bits per byte
        change_types: Optional set of change type strings
    """
    node: dict[str, Any] = {
        "node_id": node_id,
        "status": status,
        "what_and_why": what_and_why,
        "theory_type": theory_type,
    }

    if config_delta is not None:
        node["config_delta"] = config_delta
    if failure_reason is not None:
        node["failure_reason"] = failure_reason
    if measured_bpb is not None:
        node["measured_bpb"] = measured_bpb
    if change_types is not None:
        node["change_types"] = list(change_types)

    graph.nodes[node_id] = node
