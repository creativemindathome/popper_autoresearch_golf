"""
Falsifier CLI entry point.

Usage:
    python -m falsifier.main --candidate-json candidate.json --output-json output.json
    python -m falsifier.main --calibrate --train-gpt train_gpt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .calibrate import calibrate, load_calibration
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


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Falsifier: Stage 1 + Stage 2 theory prosecution")
    
    # Mode selection
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration mode instead of falsification",
    )
    
    # Input/output paths
    parser.add_argument(
        "--candidate-json",
        type=Path,
        help="Path to candidate JSON file",
    )
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
    
    # ══ Falsification mode ═════════════════════════════════════════════════════
    if not parsed.candidate_json:
        print("Error: --candidate-json required for falsification mode (or use --calibrate)")
        return 1
    
    # Load candidate
    print(f"[main] Loading candidate from {parsed.candidate_json}")
    inp = _load_candidate_json(parsed.candidate_json)
    
    # Load calibration
    repo_root = Path(inp.train_gpt_path).resolve().parent if inp.train_gpt_path else Path.cwd()
    inp.calibration = load_calibration(repo_root)
    
    # Load graph if path provided
    if parsed.graph_path and parsed.graph_path.exists():
        from .graph.query import load_graph
        inp.graph = load_graph(parsed.graph_path)
    
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
        parsed.output_json.write_text(json.dumps(_output_to_dict(output), indent=2))
        return 1
    
    # ══ Stage 1 ═══════════════════════════════════════════════════════════════
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
            _save_output(stage1_output, parsed.output_json, inp.graph, parsed.graph_path)
            return 0
    else:
        # Load Stage 1 output from somewhere (not implemented, would need path)
        print("Error: --stage-only=2 requires pre-existing Stage 1 output")
        return 1
    
    # ══ Stage 2 ═══════════════════════════════════════════════════════════════
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
    
    # Save output
    _save_output(final_output, parsed.output_json, inp.graph, parsed.graph_path)
    
    print(f"[main] Output saved to {parsed.output_json}")
    return 0 if final_output.verdict in ("STAGE_1_PASSED", "STAGE_2_PASSED") else 1


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
) -> None:
    """Save output to JSON and update graph."""
    # Save output JSON
    output_path.write_text(json.dumps(_output_to_dict(out), indent=2))
    
    # Update graph if path provided
    if graph_path:
        # Need to reload input for graph update
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
