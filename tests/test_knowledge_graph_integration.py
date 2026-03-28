"""Integration test for knowledge graph lifecycle.

Tests: GENERATED → APPROVED → IN_FALSIFICATION → REFUTED/PASSED
"""

import json
import tempfile
from pathlib import Path

import pytest

from falsifier.adapters import load_ideator_idea, adapt_ideator_to_falsifier
from falsifier.graph.lifecycle import (
    create_node_from_ideator_idea,
    update_node_status,
    find_node_by_idea_id,
    get_node_status,
)
from falsifier.graph.locking import AtomicGraphUpdate


@pytest.fixture
def mock_ideator_idea():
    """Create a mock ideator output."""
    return {
        "schema_version": "ideator.idea.v1",
        "idea_id": "test-low-rank",
        "title": "Test Low-Rank Factorization",
        "novelty_summary": "Test idea for low-rank factorization",
        "parent_implementation": {
            "repo_url": "https://github.com/openai/parameter-golf",
            "primary_file": "train_gpt.py",
            "run_command": "python train_gpt.py",
        },
        "implementation_steps": [
            {
                "step_id": "step_1",
                "file": "train_gpt.py",
                "locate": "class MLP",
                "change": "Use low-rank factorization",
            }
        ],
        "falsifier_smoke_tests": ["python train_gpt.py --test"],
        "meta": {
            "generated_at": "2026-03-28T12:00:00Z",
            "model": "gemini-2.5-flash",
        },
    }


@pytest.fixture
def temp_knowledge_dir():
    """Create temporary knowledge graph directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = Path(tmpdir) / "knowledge_graph"
        (kg / "outbox" / "ideator").mkdir(parents=True)
        (kg / "inbox" / "approved").mkdir(parents=True)
        (kg / "work" / "in_falsification").mkdir(parents=True)

        # Create mock train_gpt.py file
        (kg / "outbox" / "ideator" / "test-low-rank_train_gpt.py").write_text(
            "# Mock train_gpt.py\nprint('test')"
        )

        yield kg


class TestKnowledgeGraphLifecycle:
    """Test full node lifecycle."""

    def test_create_node_from_ideator(self, mock_ideator_idea, temp_knowledge_dir):
        """Test: Ideator creates node with status=GENERATED."""
        graph_path = temp_knowledge_dir / "graph.json"

        # Create node from ideator idea
        node_id = create_node_from_ideator_idea(
            mock_ideator_idea, graph_path, temp_knowledge_dir
        )

        # Verify node created (node_id uses just idea_id in current implementation)
        assert node_id == "test-low-rank"

        # Load and verify node
        updater = AtomicGraphUpdate(graph_path)
        graph = updater.read_graph()

        assert node_id in graph["nodes"]
        node = graph["nodes"][node_id]

        assert node["idea_id"] == "test-low-rank"
        assert node["title"] == "Test Low-Rank Factorization"
        assert node["status"] == "GENERATED"
        assert len(node["status_history"]) == 1
        assert node["status_history"][0]["status"] == "GENERATED"
        assert "generated_at" in node["source"]

    def test_update_status_to_approved(self, mock_ideator_idea, temp_knowledge_dir):
        """Test: Reviewer updates status to APPROVED."""
        graph_path = temp_knowledge_dir / "graph.json"

        # Create node
        node_id = create_node_from_ideator_idea(
            mock_ideator_idea, graph_path, temp_knowledge_dir
        )

        # Update to APPROVED
        update_node_status(
            node_id,
            "APPROVED",
            graph_path,
            actor="reviewer",
            metadata={"novelty_score": 7},
        )

        # Verify
        status = get_node_status(graph_path, node_id)
        assert status == "APPROVED"

        # Verify history
        updater = AtomicGraphUpdate(graph_path)
        graph = updater.read_graph()
        node = graph["nodes"][node_id]

        assert len(node["status_history"]) == 2
        assert node["status_history"][1]["status"] == "APPROVED"
        assert node["status_history"][1]["actor"] == "reviewer"
        assert node["status_history"][1]["metadata"]["novelty_score"] == 7

    def test_find_node_by_idea_id(self, mock_ideator_idea, temp_knowledge_dir):
        """Test: Finding nodes by idea_id."""
        graph_path = temp_knowledge_dir / "graph.json"

        # Create node
        create_node_from_ideator_idea(mock_ideator_idea, graph_path, temp_knowledge_dir)

        # Find node
        found_id = find_node_by_idea_id(graph_path, "test-low-rank")
        assert found_id == "test-low-rank"

        # Non-existent idea
        not_found = find_node_by_idea_id(graph_path, "non-existent")
        assert not_found is None

    def test_adapt_ideator_to_falsifier(self, mock_ideator_idea, temp_knowledge_dir):
        """Test: Converting ideator format to FalsifierInput."""
        from falsifier.types import FalsifierInput

        # Adapt
        inp = adapt_ideator_to_falsifier(mock_ideator_idea, temp_knowledge_dir)

        # Verify
        assert isinstance(inp, FalsifierInput)
        assert inp.theory_id == "test-low-rank"
        assert inp.what_and_why == "Test idea for low-rank factorization"
        assert inp.theory_type == "architectural"
        assert inp.train_gpt_path == "train_gpt.py"
        # The adapter reads the actual train_gpt.py file content
        assert "Mock train_gpt.py" in inp.proposed_train_gpt


class TestLockFileMechanism:
    """Test lock file creation and management."""

    def test_lock_file_created_and_removed(self, temp_knowledge_dir):
        """Test: Lock files track in-progress falsification."""
        work_dir = temp_knowledge_dir / "work" / "in_falsification"
        lock_file = work_dir / "test-idea.lock"

        # Create lock
        lock_data = {
            "lock_type": "falsifier_in_progress",
            "node_id": "test-idea",
            "idea_id": "test-idea",
            "started_at": "2026-03-28T12:00:00Z",
            "pid": 12345,
        }
        lock_file.write_text(json.dumps(lock_data, indent=2))

        # Verify created
        assert lock_file.exists()

        # Remove lock
        lock_file.unlink()
        assert not lock_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
