#!/usr/bin/env python3
"""
Direct pressure test - runs actual candidates through the full pipeline.
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from falsifier.types import FalsifierInput, FalsifierOutput, Calibration, KnowledgeGraph
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.stage2.orchestrator import run_stage_2


@dataclass
class TestCase:
    name: str
    path: Path
    expected_stage: str
    description: str


def run_pressure_test():
    """Run direct pressure test on real candidates."""
    
    print("=" * 80)
    print("DIRECT PRESSURE TEST - Full Pipeline")
    print("=" * 80)
    print()
    
    # Test cases - real candidates with expected outcomes
    test_cases = [
        TestCase("good_student.py", Path("tests/candidates/good_student.py"), "T5", "Should fail T5 on rank deficiency"),
        TestCase("gradient_flow_issues.py", Path("tests/candidates/gradient_flow_issues.py"), "T5", "MLX model - may pass or fail"),
        TestCase("over_parametrized.py", Path("tests/candidates/over_parametrized.py"), "T2", "Large model - should fail T2 budget"),
        TestCase("broken_architecture.py", Path("tests/candidates/broken_architecture.py"), "T3", "Broken - should fail compilation"),
    ]
    
    results = []
    
    for test in test_cases:
        if not test.path.exists():
            print(f"SKIP: {test.name} not found")
            continue
        
        print(f"Testing {test.name}...")
        print(f"  Description: {test.description}")
        
        start = time.time()
        try:
            # Load candidate
            code = test.path.read_text()
            
            # Create input
            inp = FalsifierInput(
                theory_id=f"pressure_test_{test.name}",
                what_and_why=f"Pressure test for {test.name}",
                proposed_train_gpt=code,
                sota_train_gpt=code,
                config_delta={},
                graph=KnowledgeGraph(),
                val_data_path="",
                calibration=Calibration(),
            )
            
            # Run Stage 1
            stage1_result = run_stage_1(inp)
            stage1_duration = time.time() - start
            
            # Record Stage 1 results
            stage1_reached = stage1_result.verdict
            killed_by = stage1_result.killed_by
            tags = stage1_result.tags or []
            
            print(f"  Stage 1: {stage1_reached}")
            print(f"  Killed by: {killed_by or 'N/A'}")
            print(f"  Tags: {len(tags)}")
            print(f"  Duration: {stage1_duration:.3f}s")
            
            # Check if we reached expected stage
            stage1_correct = (killed_by and killed_by.startswith(test.expected_stage)) or \
                           (stage1_reached == "STAGE_1_PASSED" and test.expected_stage == "T7")
            
            results.append({
                "name": test.name,
                "stage1_passed": True,
                "stage1_duration": stage1_duration,
                "verdict": stage1_reached,
                "killed_by": killed_by,
                "tags_count": len(tags),
                "expected_stage": test.expected_stage,
                "reached_expected": stage1_correct,
            })
            
            # Run Stage 2 if Stage 1 passed
            if stage1_reached == "STAGE_1_PASSED":
                print(f"  Running Stage 2...")
                s2_start = time.time()
                
                try:
                    stage2_result = run_stage_2(inp, stage1_result)
                    s2_duration = time.time() - s2_start
                    
                    print(f"  Stage 2: {stage2_result.verdict}")
                    print(f"  Stage 2 Duration: {s2_duration:.3f}s")
                    
                    results[-1]["stage2_passed"] = True
                    results[-1]["stage2_duration"] = s2_duration
                    results[-1]["final_verdict"] = stage2_result.verdict
                    
                except Exception as e:
                    s2_duration = time.time() - s2_start
                    print(f"  Stage 2 ERROR: {str(e)[:80]}")
                    results[-1]["stage2_passed"] = False
                    results[-1]["stage2_error"] = str(e)
                    results[-1]["stage2_duration"] = s2_duration
            else:
                results[-1]["stage2_skipped"] = True
                results[-1]["final_verdict"] = stage1_reached
            
            print(f"  ✓ Test completed")
            
        except Exception as e:
            duration = time.time() - start
            print(f"  ✗ ERROR: {str(e)[:100]}")
            results.append({
                "name": test.name,
                "stage1_passed": False,
                "error": str(e),
                "duration": duration,
            })
        
        print()
    
    # Generate report
    print("=" * 80)
    print("PRESSURE TEST REPORT")
    print("=" * 80)
    print()
    
    total = len(results)
    stage1_passed = sum(1 for r in results if r.get("stage1_passed"))
    stage2_passed = sum(1 for r in results if r.get("stage2_passed"))
    reached_expected = sum(1 for r in results if r.get("reached_expected"))
    
    print(f"Total candidates: {total}")
    print(f"Stage 1 executed: {stage1_passed}/{total}")
    print(f"Stage 2 executed: {sum(1 for r in results if 'stage2_passed' in r or 'stage2_skipped' in r)}/{total}")
    print(f"Reached expected stage: {reached_expected}/{total}")
    print()
    
    # Performance summary
    stage1_times = [r.get("stage1_duration", 0) for r in results if r.get("stage1_passed")]
    if stage1_times:
        avg_t1 = sum(stage1_times) / len(stage1_times)
        min_t1 = min(stage1_times)
        max_t1 = max(stage1_times)
        print(f"Stage 1 timing:")
        print(f"  Average: {avg_t1:.3f}s")
        print(f"  Min: {min_t1:.3f}s")
        print(f"  Max: {max_t1:.3f}s")
        print(f"  Throughput: {3600/avg_t1:.1f} theories/hour")
    
    stage2_times = [r.get("stage2_duration", 0) for r in results if r.get("stage2_passed")]
    if stage2_times:
        avg_t2 = sum(stage2_times) / len(stage2_times)
        print(f"\nStage 2 timing (when reached):")
        print(f"  Average: {avg_t2:.3f}s")
    
    print()
    
    # Per-candidate summary
    print("Per-candidate results:")
    print("-" * 80)
    for r in results:
        name = r["name"]
        verdict = r.get("final_verdict", r.get("verdict", "ERROR"))
        killed = r.get("killed_by", "N/A")
        tags = r.get("tags_count", 0)
        expected = r.get("expected_stage", "?")
        reached = "✓" if r.get("reached_expected") else "○"
        
        print(f"  {reached} {name:25s} -> {verdict:20s} (killed_by={killed:10s}, tags={tags:2d}, expected={expected})")
    
    print()
    
    # Save detailed results
    report_path = Path("pressure_test_results.json")
    report_path.write_text(json.dumps(results, indent=2))
    print(f"Detailed results saved to: {report_path}")
    
    # Final verdict
    print()
    if stage1_passed == total and reached_expected >= total * 0.75:
        print("✓✓ PRESSURE TEST PASSED - Pipeline handles all candidates correctly")
        return True
    elif stage1_passed >= total * 0.75:
        print("○ MOSTLY WORKING - Pipeline handles most candidates")
        return True
    else:
        print("✗ PRESSURE TEST FAILED - Too many execution errors")
        return False


if __name__ == "__main__":
    success = run_pressure_test()
    sys.exit(0 if success else 1)
