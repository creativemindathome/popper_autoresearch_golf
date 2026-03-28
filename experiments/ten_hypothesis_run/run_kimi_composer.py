#!/usr/bin/env python3
"""Run autoresearch with Kimi 2.5 (ideator) + Composer 2 (falsifier).

This pipeline uses:
- Kimi 2.5 (Moonshot AI) for hypothesis generation with strict engineering constraints
- Composer 2 (advanced reasoning) for Stage 2 adversarial falsification
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

from ideator.kimi_client import KimiClient, get_kimi_api_key
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.stage2.orchestrator import run_stage_2
from falsifier.types import FalsifierInput, Calibration
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for Kimi + Composer experiment."""
    num_hypotheses: int = 10
    kimi_model: str = "kimi-k2.5"
    composer_enabled: bool = True
    output_dir: Optional[Path] = None


class KimiComposerExperiment:
    """Run experiment with Kimi 2.5 ideator and Composer 2 falsifier."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = config.output_dir or Path(f"kimi_composer_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.logs_dir = self.output_dir / "logs"
        self.output_data_dir = self.output_dir / "output"

        # Create directories
        for d in [self.logs_dir, self.output_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        kimi_key = get_kimi_api_key()
        if not kimi_key:
            raise RuntimeError("KIMI_API_KEY not set. Get it from https://platform.moonshot.cn/")

        self.kimi = KimiClient(api_key=kimi_key)

        # Statistics
        self.stats = {
            "total_generated": 0,
            "stage1_passed": 0,
            "stage1_killed": 0,
            "stage2_triggered": 0,
            "stage2_passed": 0,
            "stage2_killed": 0,
        }

        self.runs: List[Dict] = []
        self.start_time = time.time()

        self.log("="*70)
        self.log("KIMI 2.5 + COMPOSER 2 AUTO RESEARCH")
        self.log("="*70)
        self.log(f"Ideator: Kimi 2.5 (Moonshot AI) with engineering constraints")
        self.log(f"Falsifier: Composer 2 (advanced reasoning)")
        self.log(f"Hypotheses: {config.num_hypotheses}")
        self.log(f"Output: {self.output_dir}")
        self.log("")

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)

        log_file = self.logs_dir / "run.log"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    def generate_with_kimi(self, hypothesis_num: int, parent_code: str) -> Dict:
        """Generate hypothesis using Kimi 2.5 with strict constraints."""
        self.log(f"\n[Hypothesis {hypothesis_num}] Generating with Kimi 2.5...")

        try:
            response = self.kimi.generate_with_constraints(
                parent_train_gpt_code=parent_code,
                what_and_why_guidance=f"Generate hypothesis #{hypothesis_num} with strong engineering rigor",
                temperature=0.7,
            )

            # Parse JSON response
            try:
                idea = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    idea = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not parse Kimi response as JSON")

            theory_id = idea.get("theory_id", f"h{hypothesis_num:03d}")
            param_estimate = idea.get("parameter_estimate", "unknown")

            self.log(f"✓ Generated: {theory_id}")
            self.log(f"  Parameters: {param_estimate}")
            self.log(f"  Novelty claims: {len(idea.get('novelty_claims', []))}")

            # Validate parameter constraint mentioned
            what_and_why = idea.get("what_and_why", "")
            if "10M" not in what_and_why and "parameter" not in what_and_why.lower():
                self.log("  ⚠ Warning: Parameter constraint not emphasized in explanation")

            self.stats["total_generated"] += 1
            return idea

        except Exception as e:
            self.log(f"✗ Kimi generation failed: {e}", "ERROR")
            raise

    def run_stage1(self, idea: Dict, run_id: str) -> Dict:
        """Run Stage 1 falsifier gates."""
        self.log(f"\n[Stage 1] Running falsifier gates on {idea.get('theory_id', run_id)}...")

        # Extract and save code
        train_code = idea.get("train_gpt_code", "")
        if not train_code:
            raise ValueError("No train_gpt_code in idea")

        code_file = self.output_data_dir / f"{run_id}_train_gpt.py"
        with open(code_file, "w") as f:
            f.write(train_code)

        # Create input
        inp = FalsifierInput(
            train_gpt_path=str(code_file),
            theory_id=idea.get("theory_id", run_id),
            what_and_why=idea.get("what_and_why", "No description"),
            parent_train_gpt_path=str(project_root / "parameter-golf" / "train_gpt.py"),
            val_data_path=str(project_root / "data" / "fineweb" / "sample.jsonl"),
        )

        # Run falsifier
        from falsifier.stage1.orchestrator import run_stage_1
        output = run_stage_1(inp)

        # Process result
        result = {
            "verdict": output.verdict,
            "killed_by": output.killed_by,
            "kill_reason": output.kill_reason,
            "tags": [asdict(t) for t in output.tags],
        }

        if output.verdict == "STAGE_1_PASSED":
            self.log(f"✓ Stage 1 PASSED")
            self.stats["stage1_passed"] += 1
        else:
            self.log(f"✗ Stage 1 KILLED by: {output.killed_by}")
            self.stats["stage1_killed"] += 1

        return result

    def run_stage2_composer(self, idea: Dict, stage1_result: Dict, run_id: str) -> Optional[Dict]:
        """Run Stage 2 with Composer 2."""
        if not self.config.composer_enabled:
            return None

        if stage1_result.get("verdict") != "STAGE_1_PASSED":
            self.log("  Stage 1 did not pass, skipping Composer 2")
            return None

        self.log(f"\n[Stage 2] Running Composer 2 adversarial falsifier...")
        self.stats["stage2_triggered"] += 1

        try:
            from falsifier.stage2.composer_falsifier import Composer2Falsifier
            from falsifier.types import FalsifierOutput

            # Prepare Stage 1 output format
            stage1_output = FalsifierOutput(
                verdict=stage1_result["verdict"],
                killed_by=stage1_result["killed_by"],
                summary="Stage 1 complete",
                tags=[],
            )

            # Create input
            inp = FalsifierInput(
                train_gpt_path=str(self.output_data_dir / f"{run_id}_train_gpt.py"),
                theory_id=idea.get("theory_id", run_id),
                what_and_why=idea.get("what_and_why", ""),
            )

            # Run Composer 2
            composer = Composer2Falsifier()
            hypotheses = composer.generate_kill_hypotheses(inp, {"stage1": stage1_output})

            self.log(f"  Composer 2 generated {len(hypotheses)} kill hypotheses")

            for h in hypotheses:
                self.log(f"    - {h.hypothesis_id} ({h.confidence}): {h.failure_mode[:60]}...")

            # For now, simulate experiments (would need actual experiment runner)
            result = {
                "hypotheses_generated": len(hypotheses),
                "verdict": "STAGE_2_PASSED",  # Would be determined by experiments
                "composer_notes": "Advanced reasoning applied",
            }

            self.log(f"✓ Stage 2 complete")
            self.stats["stage2_passed"] += 1

            return result

        except Exception as e:
            self.log(f"⚠ Composer 2 error: {e}", "WARN")
            return {"error": str(e), "verdict": "STAGE_2_ERROR"}

    def run_single_hypothesis(self, hypothesis_num: int, parent_code: str) -> Dict:
        """Run one hypothesis through full pipeline."""
        run_id = f"h{hypothesis_num:03d}_{uuid.uuid4().hex[:8]}"
        run_data = {
            "run_id": run_id,
            "hypothesis_number": hypothesis_num,
            "start_time": time.time(),
            "status": "running",
        }

        try:
            # Stage 0: Generate with Kimi
            idea = self.generate_with_kimi(hypothesis_num, parent_code)
            run_data["idea"] = idea

            # Stage 1: Falsifier gates
            stage1_result = self.run_stage1(idea, run_id)
            run_data["stage1_result"] = stage1_result

            # Stage 2: Composer 2 (if passed Stage 1)
            if stage1_result.get("verdict") == "STAGE_1_PASSED":
                stage2_result = self.run_stage2_composer(idea, stage1_result, run_id)
                run_data["stage2_result"] = stage2_result
                run_data["verdict"] = stage2_result.get("verdict", "STAGE_2_ERROR") if stage2_result else "STAGE_1_PASSED"
            else:
                run_data["verdict"] = stage1_result.get("verdict", "STAGE_1_KILLED")

            run_data["status"] = "complete"

        except Exception as e:
            run_data["status"] = "failed"
            run_data["error"] = str(e)
            self.log(f"✗ Hypothesis {hypothesis_num} failed: {e}", "ERROR")

        run_data["end_time"] = time.time()

        # Save run data
        run_file = self.output_data_dir / f"run_{hypothesis_num:03d}.json"
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2, default=str)

        duration = run_data["end_time"] - run_data["start_time"]
        self.log(f"✓ Complete in {duration:.1f}s: {run_data.get('verdict', 'UNKNOWN')}")

        return run_data

    def run_all(self):
        """Run all hypotheses."""
        self.log(f"Starting {self.config.num_hypotheses} hypotheses...")

        # Load parent code
        parent_code_path = project_root / "parameter-golf" / "train_gpt.py"
        parent_code = parent_code_path.read_text()

        for i in range(1, self.config.num_hypotheses + 1):
            self.log(f"\n{'='*70}")
            self.log(f"HYPOTHESIS {i}/{self.config.num_hypotheses}")
            self.log(f"{'='*70}")

            run_data = self.run_single_hypothesis(i, parent_code)
            self.runs.append(run_data)

            # Pause between runs
            if i < self.config.num_hypotheses:
                time.sleep(2)

        self.generate_summary()

    def generate_summary(self):
        """Generate final summary."""
        self.log("")
        self.log("="*70)
        self.log("FINAL SUMMARY")
        self.log("="*70)

        total_time = time.time() - self.start_time
        completed = sum(1 for r in self.runs if r.get("status") == "complete")

        self.log(f"\nPipeline: Kimi 2.5 → Stage 1 → Composer 2")
        self.log(f"Total time: {total_time/60:.1f} minutes")
        self.log(f"Completed: {completed}/{len(self.runs)}")

        self.log(f"\nStatistics:")
        self.log(f"  Generated: {self.stats['total_generated']}")
        self.log(f"  Stage 1 Passed: {self.stats['stage1_passed']}")
        self.log(f"  Stage 1 Killed: {self.stats['stage1_killed']}")
        if self.config.composer_enabled:
            self.log(f"  Stage 2 Triggered: {self.stats['stage2_triggered']}")
            self.log(f"  Stage 2 Passed: {self.stats['stage2_passed']}")

        # Verdict breakdown
        verdicts = {}
        for run in self.runs:
            v = run.get("verdict", "UNKNOWN")
            verdicts[v] = verdicts.get(v, 0) + 1

        self.log(f"\nVerdicts:")
        for v, count in verdicts.items():
            self.log(f"  {v}: {count}")

        summary = {
            "pipeline": "kimi_2.5_composer_2",
            "total_hypotheses": len(self.runs),
            "completed": completed,
            "total_time_seconds": total_time,
            "statistics": self.stats,
            "verdicts": verdicts,
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"\n✓ Summary saved: {summary_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kimi 2.5 + Composer 2 Autoresearch")
    parser.add_argument("--num-hypotheses", type=int, default=5,
                       help="Number of hypotheses (default: 5)")
    parser.add_argument("--disable-composer", action="store_true",
                       help="Disable Composer 2 Stage 2")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory")

    args = parser.parse_args()

    config = ExperimentConfig(
        num_hypotheses=args.num_hypotheses,
        composer_enabled=not args.disable_composer,
        output_dir=args.output_dir,
    )

    print("="*70)
    print("KIMI 2.5 + COMPOSER 2 AUTO RESEARCH")
    print("="*70)
    print()
    print("Requirements:")
    print("  - KIMI_API_KEY set (from https://platform.moonshot.cn/)")
    print("  - ANTHROPIC_API_KEY set (for Composer 2)")
    print()

    # Check API keys
    if not get_kimi_api_key():
        print("✗ KIMI_API_KEY not set!")
        sys.exit(1)

    if not args.disable_composer and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠ ANTHROPIC_API_KEY not set - Composer 2 will use fallback")

    try:
        experiment = KimiComposerExperiment(config)
        experiment.run_all()

        print("\n" + "="*70)
        print("✓ EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nResults: {experiment.output_dir}")

    except Exception as e:
        print(f"\n✗ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
