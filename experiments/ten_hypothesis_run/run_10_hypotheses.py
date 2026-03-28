#!/usr/bin/env python3
"""Generate and falsify 10 hypotheses, tracking graph evolution.

This script:
1. Generates 10 novel architecture ideas using Anthropic Claude
2. Runs each through Stage 1 falsifier gates (T2-T7)
3. Tracks knowledge graph evolution with snapshots
4. Creates visualization data for time-lapse
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ideator.anthropic_client import AnthropicClient, get_anthropic_api_key
from falsifier.main import run_falsifier
from falsifier.types import FalsifierInput, Calibration
from falsifier.validation import validate_falsifier_input


@dataclass
class HypothesisRun:
    """Track a single hypothesis through the pipeline."""
    run_id: str
    hypothesis_number: int
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # pending, generating, stage1, stage2, complete, failed
    idea_json: Optional[Dict] = None
    stage1_result: Optional[Dict] = None
    stage2_result: Optional[Dict] = None
    verdict: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GraphSnapshot:
    """Snapshot of knowledge graph state at a point in time."""
    timestamp: float
    iso_time: str
    total_ideas: int
    approved_ideas: int
    falsified_ideas: int
    passed_stage1: int
    passed_stage2: int
    nodes: List[Dict]
    edges: List[Dict]


class TenHypothesisRunner:
    """Orchestrate 10 hypothesis generation and falsification."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.snapshots_dir = output_dir / "graph_snapshots"
        self.viz_dir = output_dir / "visualization"

        # Create directories
        for d in [self.logs_dir, self.snapshots_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State
        self.runs: List[HypothesisRun] = []
        self.snapshots: List[GraphSnapshot] = []
        self.start_time = time.time()
        self.client: Optional[AnthropicClient] = None

        # Load API key
        api_key = get_anthropic_api_key()
        if not api_key:
            # Try loading from .env
            env_file = project_root / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.strip().startswith("ANTHROPIC_API_KEY"):
                            key = line.split("=", 1)[1].strip().strip('"')
                            os.environ["ANTHROPIC_API_KEY"] = key
                            api_key = key
                            break

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env")

        self.client = AnthropicClient(api_key=api_key)
        print(f"✓ Anthropic client initialized")

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)

        # Append to log file
        log_file = self.logs_dir / "run.log"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    def generate_idea(self, hypothesis_num: int) -> Dict:
        """Generate a novel architecture idea using Anthropic."""
        self.log(f"Generating hypothesis #{hypothesis_num}...")

        system_prompt = """You are an expert ML researcher specializing in efficient transformer architectures.
Your task is to generate a novel, bold, and potentially breakthrough idea for a new neural network architecture.

The idea should:
1. Be technically grounded (not science fiction)
2. Challenge conventional wisdom
3. Have a clear mechanism for why it could work
4. Be falsifiable (we can test it with micro-training)

Respond with a JSON object containing:
- theory_id: unique identifier
- what_and_why: 2-3 paragraphs explaining the idea and why it might work
- train_gpt_code: Python code implementing the architecture (complete, runnable)
- parent_architecture: what existing work this builds on
- novelty_claims: list of specific novel aspects

Be creative but rigorous. The best ideas often come from questioning fundamental assumptions."""

        user_prompt = f"Generate hypothesis #{hypothesis_num} for a novel efficient transformer architecture. Think boldly but keep it grounded in what we know about neural networks and training dynamics."

        try:
            response = self.client.generate_idea(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=4096,
                model="claude-sonnet-4-20250514",
            )

            # Parse JSON response
            idea = json.loads(response)
            self.log(f"✓ Generated hypothesis #{hypothesis_num}: {idea.get('theory_id', 'unknown')}")
            return idea

        except json.JSONDecodeError as e:
            self.log(f"✗ Failed to parse JSON for hypothesis #{hypothesis_num}: {e}", "ERROR")
            # Try to extract JSON from markdown code block
            try:
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    idea = json.loads(json_match.group(1))
                    self.log(f"✓ Extracted JSON from markdown for hypothesis #{hypothesis_num}")
                    return idea
            except:
                pass
            raise
        except Exception as e:
            self.log(f"✗ Failed to generate hypothesis #{hypothesis_num}: {e}", "ERROR")
            raise

    def save_idea_file(self, idea: Dict, run: HypothesisRun) -> Path:
        """Save idea to inbox for falsifier."""
        idea_file = self.output_dir / "output" / f"{run.run_id}.json"
        idea_file.parent.mkdir(parents=True, exist_ok=True)

        with open(idea_file, "w") as f:
            json.dump(idea, f, indent=2)

        return idea_file

    def run_falsifier_on_idea(self, idea: Dict, run: HypothesisRun) -> Dict:
        """Run the falsifier pipeline on an idea."""
        self.log(f"Running falsifier on hypothesis #{run.hypothesis_number}...")

        # Extract train_gpt code and save to temp file
        train_code = idea.get("train_gpt_code", "")
        if not train_code:
            raise ValueError("No train_gpt_code in idea")

        # Save code to temp file
        temp_code_file = self.output_dir / "output" / f"{run.run_id}_train_gpt.py"
        with open(temp_code_file, "w") as f:
            f.write(train_code)

        # Create falsifier input
        inp = FalsifierInput(
            train_gpt_path=str(temp_code_file),
            theory_id=idea.get("theory_id", run.run_id),
            what_and_why=idea.get("what_and_why", "No description provided"),
            parent_train_gpt_path=str(project_root / "parameter-golf" / "train_gpt.py"),
            val_data_path=str(project_root / "data" / "fineweb" / "sample.jsonl"),
        )

        # Validate input
        is_valid, errors = validate_falsifier_input(inp)
        if not is_valid:
            self.log(f"✗ Validation failed: {errors}", "ERROR")
            raise ValueError(f"Invalid falsifier input: {errors}")

        # Run falsifier
        calibration = Calibration()
        output = run_falsifier(inp, calibration)

        # Convert to dict for storage
        result = {
            "verdict": output.verdict,
            "killed_by": output.killed_by,
            "summary": output.summary,
            "tags": [asdict(t) for t in output.tags],
            "t2_budget": asdict(output.t2_budget) if output.t2_budget else None,
            "t3_compilation": asdict(output.t3_compilation) if output.t3_compilation else None,
            "t4_signal": asdict(output.t4_signal) if output.t4_signal else None,
            "t5_init": asdict(output.t5_init) if output.t5_init else None,
            "t7_microtrain": asdict(output.t7_microtrain) if output.t7_microtrain else None,
            "stage2": asdict(output.stage2) if output.stage2 else None,
        }

        self.log(f"✓ Falsifier complete for hypothesis #{run.hypothesis_number}: {output.verdict}")
        return result

    def capture_graph_snapshot(self) -> GraphSnapshot:
        """Capture current state of knowledge graph."""
        kg_dir = project_root / "knowledge_graph"

        nodes = []
        edges = []

        # Load all nodes from graph.json
        graph_file = kg_dir / "graph.json"
        if graph_file.exists():
            try:
                with open(graph_file) as f:
                    graph_data = json.load(f)
                    nodes = graph_data.get("nodes", [])
                    edges = graph_data.get("edges", [])
            except:
                pass

        # Count by status
        total = len(nodes)
        approved = sum(1 for n in nodes if n.get("status") == "APPROVED")
        falsified = sum(1 for n in nodes if n.get("status") == "FALSIFIED")
        passed_s1 = sum(1 for n in nodes if n.get("stage1_passed", False))
        passed_s2 = sum(1 for n in nodes if n.get("stage2_passed", False))

        snapshot = GraphSnapshot(
            timestamp=time.time(),
            iso_time=datetime.now().isoformat(),
            total_ideas=total,
            approved_ideas=approved,
            falsified_ideas=falsified,
            passed_stage1=passed_s1,
            passed_stage2=passed_s2,
            nodes=nodes,
            edges=edges,
        )

        # Save snapshot
        snapshot_file = self.snapshots_dir / f"snapshot_{len(self.snapshots):04d}.json"
        with open(snapshot_file, "w") as f:
            json.dump(asdict(snapshot), f, indent=2)

        self.snapshots.append(snapshot)
        return snapshot

    def run_single_hypothesis(self, hypothesis_num: int) -> HypothesisRun:
        """Run one hypothesis through the full pipeline."""
        run_id = f"h{hypothesis_num:03d}_{uuid.uuid4().hex[:8]}"
        run = HypothesisRun(
            run_id=run_id,
            hypothesis_number=hypothesis_num,
            start_time=time.time(),
        )
        self.runs.append(run)

        try:
            # Stage 1: Generate idea
            run.status = "generating"
            idea = self.generate_idea(hypothesis_num)
            run.idea_json = idea
            run.status = "stage1"

            # Capture snapshot after generation
            self.capture_graph_snapshot()

            # Stage 2: Run falsifier
            result = self.run_falsifier_on_idea(idea, run)
            run.stage1_result = result
            run.verdict = result.get("verdict")
            run.status = "complete"

            # Capture snapshot after falsification
            self.capture_graph_snapshot()

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            self.log(f"✗ Hypothesis #{hypothesis_num} failed: {e}", "ERROR")

        run.end_time = time.time()
        return run

    def run_all(self, num_hypotheses: int = 10):
        """Run all hypotheses."""
        self.log(f"Starting 10-hypothesis run...")
        self.log(f"Output directory: {self.output_dir}")

        for i in range(1, num_hypotheses + 1):
            self.log(f"\n{'='*60}")
            self.log(f"HYPOTHESIS {i}/{num_hypotheses}")
            self.log(f"{'='*60}")

            run = self.run_single_hypothesis(i)

            # Save run data
            run_file = self.output_dir / "output" / f"run_{i:03d}.json"
            with open(run_file, "w") as f:
                json.dump(asdict(run), f, indent=2, default=str)

            # Brief pause between runs
            if i < num_hypotheses:
                self.log("Pausing 2 seconds before next hypothesis...")
                time.sleep(2)

        # Final summary
        self.generate_summary()
        self.generate_visualization_data()

    def generate_summary(self):
        """Generate final summary report."""
        self.log("\n" + "="*60)
        self.log("FINAL SUMMARY")
        self.log("="*60)

        total_time = time.time() - self.start_time
        completed = sum(1 for r in self.runs if r.status == "complete")
        failed = sum(1 for r in self.runs if r.status == "failed")

        verdicts = {}
        for run in self.runs:
            if run.verdict:
                verdicts[run.verdict] = verdicts.get(run.verdict, 0) + 1

        summary = {
            "total_hypotheses": len(self.runs),
            "completed": completed,
            "failed": failed,
            "total_time_seconds": total_time,
            "verdicts": verdicts,
            "snapshots_captured": len(self.snapshots),
        }

        self.log(f"Total time: {total_time:.1f}s")
        self.log(f"Completed: {completed}/{len(self.runs)}")
        self.log(f"Failed: {failed}/{len(self.runs)}")
        self.log(f"Verdicts: {verdicts}")
        self.log(f"Snapshots: {len(self.snapshots)}")

        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"\n✓ Summary saved to: {summary_file}")

    def generate_visualization_data(self):
        """Generate data for visualization tools."""
        self.log("\nGenerating visualization data...")

        # Timeline data
        timeline = []
        for run in self.runs:
            if run.idea_json:
                timeline.append({
                    "time": run.start_time - self.start_time,
                    "hypothesis": run.hypothesis_number,
                    "event": "start",
                    "theory_id": run.idea_json.get("theory_id", run.run_id),
                })
            if run.end_time:
                timeline.append({
                    "time": run.end_time - self.start_time,
                    "hypothesis": run.hypothesis_number,
                    "event": "complete",
                    "verdict": run.verdict,
                    "status": run.status,
                })

        # Evolution data for time-lapse
        evolution = []
        for i, snapshot in enumerate(self.snapshots):
            evolution.append({
                "frame": i,
                "timestamp": snapshot.timestamp - self.start_time,
                "total_ideas": snapshot.total_ideas,
                "approved": snapshot.approved_ideas,
                "falsified": snapshot.falsified_ideas,
                "stage1_passed": snapshot.passed_stage1,
                "stage2_passed": snapshot.passed_stage2,
            })

        viz_data = {
            "timeline": timeline,
            "evolution": evolution,
            "start_time": self.start_time,
            "runs": [asdict(r) for r in self.runs],
        }

        # Save visualization data
        viz_file = self.viz_dir / "visualization_data.json"
        with open(viz_file, "w") as f:
            json.dump(viz_data, f, indent=2, default=str)

        self.log(f"✓ Visualization data saved to: {viz_file}")
        self.log(f"  - Timeline: {len(timeline)} events")
        self.log(f"  - Evolution frames: {len(evolution)}")


def main():
    """Main entry point."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / f"run_{timestamp}"

    print("="*60)
    print("10 HYPOTHESIS EXPERIMENT")
    print("="*60)
    print(f"Output: {output_dir}")
    print()

    try:
        runner = TenHypothesisRunner(output_dir)
        runner.run_all(num_hypotheses=10)

        print("\n" + "="*60)
        print("✓ EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Results in: {output_dir}")
        print(f"  - logs/run.log: Detailed execution log")
        print(f"  - graph_snapshots/: Graph state at each step")
        print(f"  - visualization/: Data for time-lapse animation")
        print(f"  - output/: Individual run results")

    except Exception as e:
        print(f"\n✗ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
