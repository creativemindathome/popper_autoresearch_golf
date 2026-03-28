"""
Falsifier CLI entry point.

Usage:
    python -m falsifier.main --candidate-json candidate.json --output-json output.json
    python -m falsifier.main --calibrate --train-gpt train_gpt.py
    python -m falsifier.main --idea-id my_idea --knowledge-dir knowledge_graph --output-json output.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .adapters import load_and_adapt_ideator_idea
from .calibrate import calibrate, load_calibration
from .graph.lifecycle import (
    find_node_by_idea_id,
    update_node_status,
    update_node_with_falsification_results,
)
from .graph.locking import acquire_lock, release_lock
from .graph.update import update_graph_after_verdict
from .stage1.orchestrator import run_stage_1
from .stage2.orchestrator import run_stage_2
from .types import FalsifierInput, FalsifierOutput, KnowledgeGraph, ParentRef, Feedback
from .validation import validate_candidate_package


def _load_candidate_json(path: Path) -> FalsifierInput:
    """Load candidate from JSON file."""
    data = json.loads(path.read_text())
    
    # Convert parents from dict to ParentRef
    parents = [ParentRef(**p) for p in data.get("parents", [])]
    
    # Convert graph dict to KnowledgeGraph if provided
    graph_data = data.get("graph", {})
    if isinstance(graph_data, dict):
        graph = KnowledgeGraph(
            graph_path=Path(graph_data.get("graph_path", "")),
            nodes=graph_data.get("nodes", {}),
            edges=graph_data.get("edges", []),
        )
    else:
        graph = KnowledgeGraph()
    
    # Convert theory_type
    theory_type = data.get("theory_type", "architectural")
    if theory_type not in ("architectural", "training", "data", "quantization", "hybrid"):
        theory_type = "architectural"
    
    return FalsifierInput(
        theory_id=data.get("theory_id", "unknown"),
        what_and_why=data.get("what_and_why", "No description provided"),
        config_delta=data.get("config_delta") or data.get("hyperparameters_delta"),
        parents=parents,
        theory_type=theory_type,  # type: ignore
        proposed_train_gpt=data.get("proposed_train_gpt", ""),
        train_gpt_path=data.get("train_gpt_path", ""),
        sota_train_gpt=data.get("sota_train_gpt", ""),
        sota_checkpoint_path=data.get("sota_checkpoint_path", ""),
        graph=graph,
        val_data_path=data.get("val_data_path", ""),
    )


def load_from_ideator_inbox(
    idea_id: str,
    knowledge_dir: Path,
    graph_path: Path | None = None,
) -> tuple[FalsifierInput, str | None]:
    """
    Load and adapt an ideator idea from the inbox/approved directory.

    Args:
        idea_id: The ID of the idea to load
        knowledge_dir: Path to the knowledge_graph directory
        graph_path: Optional path to the graph JSON file

    Returns:
        Tuple of (FalsifierInput, node_id) where node_id is the graph node ID
        if a graph path was provided and the node was found/created.

    Raises:
        FileNotFoundError: If the idea file doesn't exist in inbox/approved/
    """
    idea_path = knowledge_dir / "inbox" / "approved" / f"{idea_id}.json"
    
    if not idea_path.exists():
        raise FileNotFoundError(
            f"Idea file not found: {idea_path}\n"
            f"Expected idea at knowledge_graph/inbox/approved/{idea_id}.json"
        )
    
    # Load and adapt using the adapter
    inp = load_and_adapt_ideator_idea(idea_path, knowledge_dir)
    
    node_id = None
    
    # If graph path provided, find or create the node
    if graph_path:
        node_id = find_node_by_idea_id(graph_path, idea_id)
        if node_id:
            # Update node status to IN_FALSIFICATION
            update_node_status(
                node_id=node_id,
                new_status="IN_FALSIFICATION",
                graph_path=graph_path,
                actor="falsifier",
                metadata={
                    "started_at": time.time(),
                    "pid": os.getpid(),
                }
            )
    
    return inp, node_id


def _create_work_lock(
    idea_id: str,
    node_id: str | None,
    knowledge_dir: Path,
) -> Path:
    """
    Create a lock file in knowledge_graph/work/in_falsification/.

    Args:
        idea_id: The idea ID being processed
        node_id: The graph node ID (if available)
        knowledge_dir: Path to the knowledge_graph directory

    Returns:
        Path to the created lock file
    """
    lock_dir = knowledge_dir / "work" / "in_falsification"
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    lock_path = lock_dir / f"{idea_id}.lock"
    
    lock_data = {
        "pid": os.getpid(),
        "timestamp": time.time(),
        "idea_id": idea_id,
        "node_id": node_id,
    }
    
    lock_path.write_text(json.dumps(lock_data, indent=2))
    
    return lock_path


def _remove_work_lock(lock_path: Path) -> None:
    """Remove the work lock file if it exists."""
    if lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            pass


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Falsifier: Stage 1 + Stage 2 theory prosecution")
    
    # Mode selection
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration mode instead of falsification",
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--candidate-json",
        type=Path,
        help="Path to candidate JSON file",
    )
    input_group.add_argument(
        "--idea-id",
        type=str,
        help="Load idea from knowledge_graph/inbox/approved/{idea_id}.json",
    )
    
    # Knowledge graph directory
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path("knowledge_graph"),
        help="Path to knowledge_graph directory (default: knowledge_graph)",
    )
    
    # Output paths
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("falsifier_output.json"),
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--train-gpt",
        type=Path,
        help="Path to train_gpt.py (for calibration mode)",
    )
    parser.add_argument(
        "--sota-checkpoint",
        type=Path,
        help="Path to SOTA checkpoint (optional)",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        help="Path to FineWeb validation data (optional)",
    )
    parser.add_argument(
        "--stage-only",
        choices=["1", "2"],
        help="Run only Stage 1 or Stage 2 (default: both)",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        help="Path to knowledge graph JSON file",
    )
    
    parsed = parser.parse_args(args)
    
    # ══ Calibration mode ═══════════════════════════════════════════════════════
    if parsed.calibrate:
        if not parsed.train_gpt:
            print("Error: --train-gpt required for calibration mode")
            return 1
        
        print(f"[main] Running calibration on {parsed.train_gpt}")
        cal = calibrate(
            train_gpt_path=parsed.train_gpt,
            sota_checkpoint_path=parsed.sota_checkpoint,
            val_data_path=parsed.val_data,
        )
        print(f"[main] Calibration complete")
        print(f"  Baseline 100-step loss drop: {cal.baseline_100.loss_drop_mean:.4f} ± {cal.baseline_100.loss_drop_std:.4f}")
        return 0
    
    # ══ Falsification mode ═════════════════════════════════════════════════
    # Validate input source
    if not parsed.candidate_json and not parsed.idea_id:
        print("Error: Either --candidate-json or --idea-id required for falsification mode (or use --calibrate)")
        return 1
    
    # Resolve knowledge_dir to absolute path
    knowledge_dir = parsed.knowledge_dir.resolve()
    
    # Resolve graph path if not provided but using idea-id
    graph_path = parsed.graph_path
    if not graph_path and parsed.idea_id:
        # Default graph path is knowledge_dir/graph.json
        graph_path = knowledge_dir / "graph.json"
    if graph_path:
        graph_path = graph_path.resolve()
    
    # Load candidate from appropriate source
    inp: FalsifierInput
    node_id: str | None = None
    idea_id: str | None = None
    lock_path: Path | None = None
    
    try:
        if parsed.idea_id:
            # Load from ideator inbox
            idea_id = parsed.idea_id
            print(f"[main] Loading idea {idea_id} from {knowledge_dir}/inbox/approved/")
            
            inp, node_id = load_from_ideator_inbox(
                idea_id=idea_id,
                knowledge_dir=knowledge_dir,
                graph_path=graph_path if graph_path and graph_path.exists() else None,
            )
            
            # Create lock file
            lock_path = _create_work_lock(idea_id, node_id, knowledge_dir)
            print(f"[main] Created lock file: {lock_path}")
            
        else:
            # Load from candidate JSON
            print(f"[main] Loading candidate from {parsed.candidate_json}")
            inp = _load_candidate_json(parsed.candidate_json)
            idea_id = inp.theory_id
            
            # Create lock file even for non-graph mode (for tracking)
            lock_path = _create_work_lock(idea_id, None, knowledge_dir)
        
        # Ensure lock is cleaned up even on error
        try:
            # Load calibration
            repo_root = Path(inp.train_gpt_path).resolve().parent if inp.train_gpt_path else Path.cwd()
            inp.calibration = load_calibration(repo_root)
            
            # Load graph if path provided
            if graph_path and graph_path.exists():
                from .graph.query import load_graph
                inp.graph = load_graph(graph_path)
            
            # Validate
            print(f"[main] Validating candidate {inp.theory_id}...")
            validation = validate_candidate_package(inp)
            if not validation.ok:
                print(f"[main] Validation failed: {'; '.join(validation.reasons)}")
                output = FalsifierOutput(
                    theory_id=inp.theory_id,
                    verdict="REJECTED",
                    killed_by="VALIDATION",
                    kill_reason="; ".join(validation.reasons),
                    feedback=Feedback(
                        one_line="Validation failed: " + "; ".join(validation.reasons),
                        stage_reached=0,
                    ),
                )
                _save_output(output, parsed.output_json, inp.graph, graph_path, node_id)
                return 1
            
            # ══ Stage 1 ═══════════════════════════════════════════════════
            stage1_output = None
            if parsed.stage_only != "2":
                print(f"[main] Running Stage 1...")
                stage1_output = run_stage_1(inp)
                
                print(f"[main] Stage 1: {stage1_output.verdict}")
                if stage1_output.killed_by:
                    print(f"  Killed by: {stage1_output.killed_by}")
                    print(f"  Reason: {stage1_output.kill_reason}")
                else:
                    print(f"  Tags accumulated: {len(stage1_output.tags)}")
                
                # Early exit if killed in Stage 1
                if stage1_output.verdict in ("REJECTED", "REFUTED"):
                    _save_output(stage1_output, parsed.output_json, inp.graph, graph_path, node_id)
                    return 0
            else:
                # Load Stage 1 output from somewhere (not implemented, would need path)
                print("Error: --stage-only=2 requires pre-existing Stage 1 output")
                return 1
            
            # ══ Stage 2 ═══════════════════════════════════════════════════
            final_output = None
            if parsed.stage_only != "1":
                print(f"[main] Running Stage 2...")
                final_output = run_stage_2(inp, stage1_output)
                
                print(f"[main] Final verdict: {final_output.verdict}")
                if final_output.killed_by:
                    print(f"  Killed by: {final_output.killed_by}")
                    print(f"  Reason: {final_output.kill_reason}")
                else:
                    print(f"  Stage 2 passed!")
            else:
                final_output = stage1_output
            
            # Save output and update graph
            _save_output(final_output, parsed.output_json, inp.graph, graph_path, node_id)
            
            print(f"[main] Output saved to {parsed.output_json}")
            return 0 if final_output.verdict in ("STAGE_1_PASSED", "STAGE_2_PASSED") else 1
            
        finally:
            # Always remove lock file
            if lock_path:
                _remove_work_lock(lock_path)
                print(f"[main] Removed lock file: {lock_path}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during falsification: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _output_to_dict(out: FalsifierOutput) -> dict[str, Any]:
    """Convert FalsifierOutput to serializable dict."""
    from dataclasses import asdict
    
    d = asdict(out)
    
    # Remove non-serializable items
    def clean(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    
    return clean(d)


def _save_output(
    out: FalsifierOutput,
    output_path: Path,
    graph: KnowledgeGraph,
    graph_path: Path | None,
    node_id: str | None,
) -> None:
    """Save output to JSON and update graph."""
    # Save output JSON
    output_path.write_text(json.dumps(_output_to_dict(out), indent=2))
    
    # Update graph if path provided
    if graph_path:
        if node_id:
            # Use the lifecycle module for full node update with falsification results
            try:
                update_node_with_falsification_results(node_id, out, graph_path)
                print(f"[main] Updated graph node {node_id} with falsification results")
            except Exception as e:
                print(f"[main] Warning: Could not update node with falsification results: {e}")
                # Fall back to basic graph update
                from .types import FalsifierInput
                from dataclasses import dataclass
                
                @dataclass
                class MockInput:
                    theory_id: str
                    config_delta: dict | None
                    parents: list
                    
                mock = MockInput(
                    theory_id=out.theory_id,
                    config_delta=None,
                    parents=[],
                )
                update_graph_after_verdict(graph_path, out, mock)
        else:
            # No node_id, use basic update
            from .types import FalsifierInput
            from dataclasses import dataclass
            
            @dataclass
            class MockInput:
                theory_id: str
                config_delta: dict | None
                parents: list
                
            mock = MockInput(
                theory_id=out.theory_id,
                config_delta=None,
                parents=[],
            )
            update_graph_after_verdict(graph_path, out, mock)


if __name__ == "__main__":
    sys.exit(main())
