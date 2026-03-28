"""Regression: handoff must not wipe dict-shaped graph.json (falsifier lifecycle)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "infra" / "agents" / "scripts"))

from handoff_ideator_to_falsifier import (  # noqa: E402
    find_or_create_node,
    update_graph_with_approved_idea,
    load_graph,
)


def test_find_or_create_preserves_dict_nodes() -> None:
    graph: dict = {
        "nodes": {
            "prior": {
                "node_id": "prior",
                "idea_id": "prior",
                "status": "REFUTED",
                "falsification": {"outcome": "REFUTED"},
            },
        },
        "edges": [{"source": "a", "target": "b"}],
    }
    prior_keys = set(graph["nodes"].keys())

    node = find_or_create_node(graph, "idea_fresh")
    node["status"] = "APPROVED"

    assert set(graph["nodes"].keys()) == prior_keys | {"idea_fresh"}
    assert graph["nodes"]["prior"]["status"] == "REFUTED"
    assert isinstance(graph["nodes"], dict)
    assert graph["edges"] == [{"source": "a", "target": "b"}]


def test_update_approved_merges_into_dict_without_wipe() -> None:
    graph: dict = {
        "nodes": {
            "keep-me": {"node_id": "keep-me", "idea_id": "keep-me", "status": "STAGE_2_PASSED"},
        },
        "edges": [],
    }
    idea_data = {
        "schema_version": "ideator.idea.v1",
        "idea_id": "approved-x",
        "title": "Title",
        "novelty_summary": "Summary",
        "reviewer_feedback": {"decision": "pass"},
    }
    update_graph_with_approved_idea(graph, "approved-x", idea_data, Path("/tmp/inbox/approved-x.json"))

    assert "keep-me" in graph["nodes"]
    assert graph["nodes"]["keep-me"]["status"] == "STAGE_2_PASSED"
    assert "idea_approved-x" in graph["nodes"]
    assert graph["nodes"]["idea_approved-x"]["status"] == "APPROVED"
    assert graph["nodes"]["idea_approved-x"]["idea_id"] == "approved-x"


def test_legacy_list_nodes_still_append() -> None:
    graph = {"nodes": [{"id": "idea_a", "label": "A"}], "edges": []}
    find_or_create_node(graph, "idea_b")
    assert isinstance(graph["nodes"], list)
    assert len(graph["nodes"]) == 2


def test_load_graph_empty_default_is_dict_nodes(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent_graph.json"
    assert not missing.exists()
    g = load_graph(missing)
    assert isinstance(g["nodes"], dict)
    assert g["nodes"] == {}


def test_measured_bpb_in_knowledge_context() -> None:
    from ideator.knowledge import load_knowledge_context

    with tempfile.TemporaryDirectory() as tmp:
        kg = Path(tmp) / "knowledge_graph"
        kg.mkdir()
        (kg / "graph.json").write_text(
            json.dumps(
                {
                    "nodes": {
                        "n1": {
                            "idea_id": "n1",
                            "title": "t",
                            "status": "REFUTED",
                            "falsification": {
                                "killed_by": "T7",
                                "kill_reason": "r",
                                "outcome": "REFUTED",
                                "metrics": {"measured_bpb": 2.25},
                            },
                        },
                    },
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        text = load_knowledge_context(kg, max_chars=50_000)
    assert "bpb=2.25" in text
