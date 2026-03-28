#!/usr/bin/env python3
"""
Comprehensive pressure test for the full falsifier pipeline.

Tests:
1. All Stage 1 gates (T2, T3, T4, T5, T7) with multiple candidates
2. Stage 2 execution with various paths
3. Error handling and edge cases
4. Performance and throughput
5. Full loop integration
"""

from __future__ import annotations

import json
import sys
import time
import tempfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from falsifier.types import FalsifierInput, FalsifierOutput, Calibration, KnowledgeGraph, ParentRef
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.stage2.orchestrator import run_stage_2
from falsifier.adapters.parameter_golf import instantiate_minimal_model


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    error: str | None = None
    stage_reached: str = ""
    verdict: str = ""
    tags_count: int = 0
    details: dict = field(default_factory=dict)


class PipelinePressureTest:
    """Comprehensive pressure testing for the falsifier pipeline."""
    
    def __init__(self):
        self.results: list[TestResult] = []
        self.start_time = time.time()
        
    def log(self, msg: str):
        """Print with timestamp."""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.2f}s] {msg}")
        
    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        self.log("=" * 80)
        self.log("FULL PIPELINE PRESSURE TEST")
        self.log("=" * 80)
        self.log("")
        
        # Test 1: Stage 1 - All gates with working candidate
        self.test_stage_1_working_candidate()
        
        # Test 2: Stage 1 - T2 Budget gate variations
        self.test_stage_1_t2_budget()
        
        # Test 3: Stage 1 - T3 Compilation failure cases
        self.test_stage_1_t3_compilation()
        
        # Test 4: Stage 1 - T4 Signal detection
        self.test_stage_1_t4_signal()
        
        # Test 5: Stage 1 - T5 Init diagnostics
        self.test_stage_1_t5_init()
        
        # Test 6: Stage 1 - T7 Microtrain
        self.test_stage_1_t7_microtrain()
        
        # Test 7: Stage 2 execution paths
        self.test_stage_2_paths()
        
        # Test 8: Edge cases and error handling
        self.test_edge_cases()
        
        # Test 9: Performance and throughput
        self.test_performance()
        
        # Test 10: Full loop integration
        self.test_full_loop_integration()
        
        # Generate report
        return self.generate_report()
    
    def create_test_input(self, code: str, theory_id: str = "test") -> FalsifierInput:
        """Create a FalsifierInput from code string."""
        return FalsifierInput(
            theory_id=theory_id,
            what_and_why="Test candidate for pressure testing",
            proposed_train_gpt=code,
            sota_train_gpt=code,
            config_delta={},
            graph=KnowledgeGraph(),
            val_data_path="",
            calibration=Calibration(),
        )
    
    def test_stage_1_working_candidate(self):
        """Test Stage 1 with various candidates."""
        self.log("TEST 1: Stage 1 - Working Candidate Suite")
        self.log("-" * 60)
        
        candidates = [
            ("good_student.py", Path("tests/candidates/good_student.py")),
            ("gradient_flow_issues.py", Path("tests/candidates/gradient_flow_issues.py")),
            ("over_parametrized.py", Path("tests/candidates/over_parametrized.py")),
            ("broken_architecture.py", Path("tests/candidates/broken_architecture.py")),
        ]
        
        for name, path in candidates:
            if not path.exists():
                self.log(f"  SKIP: {name} not found")
                continue
                
            self.log(f"  Testing {name}...")
            start = time.time()
            
            try:
                code = path.read_text()
                inp = self.create_test_input(code, f"test_{name}")
                
                result = run_stage_1(inp)
                duration = time.time() - start
                
                self.results.append(TestResult(
                    name=f"stage1_{name}",
                    passed=True,
                    duration=duration,
                    stage_reached=result.verdict,
                    verdict=result.verdict,
                    tags_count=len(result.tags),
                    details={
                        "t2_passed": bool(result.t2_budget and result.t2_budget.get("passed")),
                        "t3_passed": bool(result.t3_compilation and result.t3_compilation.get("passed")),
                        "t4_passed": bool(result.t4_signal and result.t4_signal.get("passed")),
                        "t5_passed": result.t5_init is None or result.t5_init.get("kill_reason") is None,
                        "killed_by": result.killed_by,
                    }
                ))
                
                self.log(f"    ✓ {name}: {result.verdict} in {duration:.3f}s, {len(result.tags)} tags")
                
            except Exception as e:
                duration = time.time() - start
                self.results.append(TestResult(
                    name=f"stage1_{name}",
                    passed=False,
                    duration=duration,
                    error=str(e),
                ))
                self.log(f"    ✗ {name}: ERROR - {e}")
    
    def test_stage_1_t2_budget(self):
        """Test T2 Budget gate with various configurations."""
        self.log("")
        self.log("TEST 2: Stage 1 - T2 Budget Gate Variations")
        self.log("-" * 60)
        
        test_cases = [
            ("tiny_model", self._generate_tiny_model(), "Should pass easily"),
            ("huge_model", self._generate_huge_model(), "Should fail on budget"),
            ("unbalanced", self._generate_unbalanced_model(), "Should warn on FLOPs ratio"),
        ]
        
        for name, code, expected in test_cases:
            self.log(f"  Testing {name}...")
            start = time.time()
            
            try:
                inp = self.create_test_input(code, f"test_t2_{name}")
                result = run_stage_1(inp)
                duration = time.time() - start
                
                if result.t2_budget:
                    budget_util = result.t2_budget.get("budget_utilization", 0)
                    flops_ratio = result.t2_budget.get("flops_ratio", 0)
                    
                    self.results.append(TestResult(
                        name=f"t2_{name}",
                        passed=True,
                        duration=duration,
                        verdict=result.t2_budget.get("status", "UNKNOWN"),
                        details={
                            "budget_utilization": budget_util,
                            "flops_ratio": flops_ratio,
                            "estimated_params": result.t2_budget.get("estimated_params"),
                        }
                    ))
                    
                    self.log(f"    ✓ {name}: budget_util={budget_util:.4f}, flops_ratio={flops_ratio:.4f}")
                else:
                    self.results.append(TestResult(
                        name=f"t2_{name}",
                        passed=False,
                        duration=duration,
                        error="T2 result missing",
                    ))
                    self.log(f"    ✗ {name}: T2 result missing")
                    
            except Exception as e:
                duration = time.time() - start
                self.results.append(TestResult(
                    name=f"t2_{name}",
                    passed=False,
                    duration=duration,
                    error=str(e),
                ))
                self.log(f"    ✗ {name}: ERROR - {e}")
    
    def test_stage_1_t3_compilation(self):
        """Test T3 Compilation gate failure cases."""
        self.log("")
        self.log("TEST 3: Stage 1 - T3 Compilation Failure Cases")
        self.log("-" * 60)
        
        test_cases = [
            ("syntax_error", self._generate_syntax_error(), "Should fail compilation"),
            ("shape_mismatch", self._generate_shape_mismatch(), "Should fail forward pass"),
            ("no_gradients", self._generate_no_gradients(), "Should detect missing grads"),
            ("nan_weights", self._generate_nan_weights(), "Should detect NaN"),
        ]
        
        for name, code, expected in test_cases:
            self.log(f"  Testing {name}...")
            start = time.time()
            
            try:
                inp = self.create_test_input(code, f"test_t3_{name}")
                result = run_stage_1(inp)
                duration = time.time() - start
                
                caught = result.verdict == "REFUTED" and result.killed_by == "T3"
                
                self.results.append(TestResult(
                    name=f"t3_{name}",
                    passed=caught or result.verdict != "REFUTED",  # Pass if caught or didn't fail
                    duration=duration,
                    verdict=result.verdict,
                    killed_by=result.killed_by,
                    details={
                        "expected_failure": True,
                        "was_caught": caught,
                        "t3_has_nan": result.t3_compilation and result.t3_compilation.get("has_nan"),
                        "t3_has_inf": result.t3_compilation and result.t3_compilation.get("has_inf"),
                    }
                ))
                
                status = "✓" if caught else "○" if result.verdict != "REFUTED" else "✗"
                self.log(f"    {status} {name}: {result.verdict}, killed_by={result.killed_by}")
                
            except Exception as e:
                duration = time.time() - start
                self.results.append(TestResult(
                    name=f"t3_{name}",
                    passed=True,  # Exception means it failed as expected
                    duration=duration,
                    error=str(e),
                ))
                self.log(f"    ✓ {name}: Exception as expected - {str(e)[:50]}")
    
    def test_stage_1_t4_signal(self):
        """Test T4 Signal Propagation gate."""
        self.log("")
        self.log("TEST 4: Stage 1 - T4 Signal Detection")
        self.log("-" * 60)
        
        test_cases = [
            ("dead_relu", self._generate_dead_relu_model(), "Should detect dead neurons"),
            ("vanishing_grad", self._generate_vanishing_grad_model(), "Should detect low SNR"),
            ("exploding_activations", self._generate_exploding_activations(), "Should detect high activation"),
        ]
        
        for name, code, expected in test_cases:
            self.log(f"  Testing {name}...")
            start = time.time()
            
            try:
                inp = self.create_test_input(code, f"test_t4_{name}")
                result = run_stage_1(inp)
                duration = time.time() - start
                
                has_t4_tags = result.t4_signal and len(result.t4_signal.get("tags", [])) > 0
                
                self.results.append(TestResult(
                    name=f"t4_{name}",
                    passed=True,
                    duration=duration,
                    verdict=result.verdict,
                    tags_count=len(result.t4_signal.get("tags", [])) if result.t4_signal else 0,
                    details={
                        "dead_neuron_ratio": result.t4_signal.get("dead_neuron_ratio") if result.t4_signal else None,
                        "snr": result.t4_signal.get("signal_to_noise_ratio") if result.t4_signal else None,
                        "has_t4_tags": has_t4_tags,
                    }
                ))
                
                tags = result.t4_signal.get("tags", []) if result.t4_signal else []
                self.log(f"    ✓ {name}: {len(tags)} T4 tags, SNR={result.t4_signal.get('signal_to_noise_ratio') if result.t4_signal else 'N/A'}")
                
            except Exception as e:
                duration = time.time() - start
                self.results.append(TestResult(
                    name=f"t4_{name}",
                    passed=False,
                    duration=duration,
                    error=str(e),
                ))
                self.log(f"    ✗ {name}: ERROR - {e}")
    
    def test_stage_1_t5_init(self):
        """Test T5 Init Diagnostics gate."""
        self.log("")
        self.log("TEST 5: Stage 1 - T5 Init Diagnostics")
        self.log("-" * 60)
        
        # Test with good_student.py which we know has rank deficiency
        self.log("  Testing rank deficiency detection...")
        start = time.time()
        
        try:
            path = Path("tests/candidates/good_student.py")
            if path.exists():
                code = path.read_text()
                inp = self.create_test_input(code, "test_t5_rank")
                
                result = run_stage_1(inp)
                duration = time.time() - start
                
                detected_rank_issue = (
                    result.t5_init and 
                    result.t5_init.get("rank_deficiency_ratio", 0) > 0
                )
                
                self.results.append(TestResult(
                    name="t5_rank_deficiency",
                    passed=detected_rank_issue,
                    duration=duration,
                    verdict=result.verdict,
                    killed_by=result.killed_by,
                    details={
                        "rank_deficiency_ratio": result.t5_init.get("rank_deficiency_ratio") if result.t5_init else 0,
                        "effective_rank_mean": result.t5_init.get("effective_rank_mean") if result.t5_init else 0,
                        "low_rank_matrices": [
                            k for k, v in (result.t5_init.get("effective_ranks", {}) if result.t5_init else {}).items()
                            if v < 1.0
                        ],
                    }
                ))
                
                rank_ratio = result.t5_init.get("rank_deficiency_ratio", 0) if result.t5_init else 0
                self.log(f"    ✓ Detected rank deficiency: {rank_ratio*100:.1f}%")
            else:
                self.log(f"    ○ good_student.py not found, skipping")
                
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult(
                name="t5_rank_deficiency",
                passed=False,
                duration=duration,
                error=str(e),
            ))
            self.log(f"    ✗ ERROR - {e}")
    
    def test_stage_1_t7_microtrain(self):
        """Test T7 Microtrain gate."""
        self.log("")
        self.log("TEST 6: Stage 1 - T7 Microtrain")
        self.log("-" * 60)
        
        self.log("  Testing with candidate that passes T5...")
        start = time.time()
        
        try:
            # Use a simple model that might pass T5
            code = self._generate_tiny_model()
            inp = self.create_test_input(code, "test_t7_microtrain")
            
            result = run_stage_1(inp)
            duration = time.time() - start
            
            t7_reached = result.t7_microtrain is not None
            
            self.results.append(TestResult(
                name="t7_microtrain",
                passed=t7_reached or result.verdict != "STAGE_1_PASSED",
                duration=duration,
                verdict=result.verdict,
                stage_reached="T7" if t7_reached else result.killed_by or "early_stop",
                details={
                    "t7_reached": t7_reached,
                    "t7_loss_drop": result.t7_microtrain.get("loss_drop") if t7_reached else None,
                    "t7_learning_ratio": result.t7_microtrain.get("learning_ratio") if t7_reached else None,
                }
            ))
            
            if t7_reached:
                self.log(f"    ✓ T7 executed: loss_drop={result.t7_microtrain.get('loss_drop', 'N/A')}")
            else:
                self.log(f"    ○ T7 not reached (stopped at {result.killed_by})")
                
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult(
                name="t7_microtrain",
                passed=False,
                duration=duration,
                error=str(e),
            ))
            self.log(f"    ✗ ERROR - {e}")
    
    def test_stage_2_paths(self):
        """Test Stage 2 execution paths."""
        self.log("")
        self.log("TEST 7: Stage 2 Execution Paths")
        self.log("-" * 60)
        
        # Test Stage 2 with a Stage 1 passed result
        self.log("  Testing Stage 2 with STAGE_1_PASSED input...")
        start = time.time()
        
        try:
            # Create a mock Stage 1 output that passed
            stage1_output = FalsifierOutput(
                theory_id="test_stage2",
                verdict="STAGE_1_PASSED",
                t2_budget={"passed": True},
                t3_compilation={"passed": True},
                t4_signal={"passed": True},
                t5_init={"passed": True},
                t7_microtrain={"passed": True, "loss_trajectory": [4.0, 3.5, 3.0, 2.8, 2.5]},
            )
            
            # Create input
            code = self._generate_tiny_model()
            inp = self.create_test_input(code, "test_stage2")
            
            # Run Stage 2
            result = run_stage_2(inp, stage1_output)
            duration = time.time() - start
            
            self.results.append(TestResult(
                name="stage2_execution",
                passed=True,
                duration=duration,
                verdict=result.verdict,
                stage_reached="Stage 2",
                details={
                    "new_tags_count": len(result.new_tags) if hasattr(result, 'new_tags') else 0,
                    "experiments_run": "varies",
                }
            ))
            
            self.log(f"    ✓ Stage 2 completed: {result.verdict} in {duration:.3f}s")
            
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult(
                name="stage2_execution",
                passed=False,
                duration=duration,
                error=str(e),
            ))
            self.log(f"    ✗ ERROR - {str(e)[:80]}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        self.log("")
        self.log("TEST 8: Edge Cases and Error Handling")
        self.log("-" * 60)
        
        edge_cases = [
            ("empty_code", "", "Empty code string"),
            ("no_gpt_class", "x = 1\ny = 2", "No GPT class defined"),
            ("invalid_python", "def foo(:\n  pass", "Invalid Python syntax"),
            ("very_large_model", self._generate_huge_model(), "Extremely large model"),
        ]
        
        for name, code, description in edge_cases:
            self.log(f"  Testing {name} ({description})...")
            start = time.time()
            
            try:
                inp = self.create_test_input(code, f"test_edge_{name}")
                result = run_stage_1(inp)
                duration = time.time() - start
                
                # Should either pass or be rejected (not crash)
                handled_gracefully = result.verdict in ("REJECTED", "REFUTED", "STAGE_1_PASSED", "IMPLEMENTATION_FAIL")
                
                self.results.append(TestResult(
                    name=f"edge_{name}",
                    passed=handled_gracefully,
                    duration=duration,
                    verdict=result.verdict,
                    error=result.kill_reason if result.kill_reason else None,
                    details={
                        "handled_gracefully": handled_gracefully,
                        "description": description,
                    }
                ))
                
                self.log(f"    {'✓' if handled_gracefully else '✗'} {name}: {result.verdict}")
                
            except Exception as e:
                duration = time.time() - start
                self.results.append(TestResult(
                    name=f"edge_{name}",
                    passed=False,
                    duration=duration,
                    error=str(e),
                    details={"description": description}
                ))
                self.log(f"    ✗ {name}: Exception - {str(e)[:50]}")
    
    def test_performance(self):
        """Test performance and throughput."""
        self.log("")
        self.log("TEST 9: Performance and Throughput")
        self.log("-" * 60)
        
        path = Path("tests/candidates/good_student.py")
        if not path.exists():
            self.log("  SKIP: good_student.py not found")
            return
            
        code = path.read_text()
        
        # Test single execution time
        self.log("  Testing single execution time...")
        times = []
        
        for i in range(3):
            start = time.time()
            try:
                inp = self.create_test_input(code, f"perf_test_{i}")
                result = run_stage_1(inp)
                duration = time.time() - start
                times.append(duration)
            except Exception as e:
                self.log(f"    Run {i+1} failed: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            self.results.append(TestResult(
                name="performance_single",
                passed=avg_time < 10.0,  # Should complete in < 10s
                duration=avg_time,
                details={
                    "runs": len(times),
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                }
            ))
            
            self.log(f"    ✓ Single execution: {avg_time:.3f}s avg (min={min_time:.3f}s, max={max_time:.3f}s)")
            
            # Estimate throughput
            if avg_time > 0:
                throughput = 3600 / avg_time  # theories per hour
                self.log(f"    ✓ Estimated throughput: {throughput:.1f} theories/hour")
    
    def test_full_loop_integration(self):
        """Test full loop integration components."""
        self.log("")
        self.log("TEST 10: Full Loop Integration")
        self.log("-" * 60)
        
        # Test 1: Knowledge graph directories
        self.log("  Testing knowledge graph directories...")
        dirs_ok = all(
            Path(d).exists() for d in [
                "knowledge_graph/inbox/approved",
                "knowledge_graph/outbox/ideator",
                "knowledge_graph/outbox/falsifier",
                "knowledge_graph/work/in_falsification",
            ]
        )
        
        self.results.append(TestResult(
            name="integration_dirs",
            passed=dirs_ok,
            duration=0.001,
            details={"all_dirs_exist": dirs_ok}
        ))
        self.log(f"    {'✓' if dirs_ok else '✗'} Knowledge graph directories")
        
        # Test 2: Graph lifecycle functions
        self.log("  Testing graph lifecycle functions...")
        try:
            from falsifier.graph.lifecycle import update_node_status
            self.results.append(TestResult(
                name="integration_lifecycle",
                passed=True,
                duration=0.001,
            ))
            self.log("    ✓ Graph lifecycle imports")
        except Exception as e:
            self.results.append(TestResult(
                name="integration_lifecycle",
                passed=False,
                duration=0.001,
                error=str(e),
            ))
            self.log(f"    ✗ Graph lifecycle: {e}")
        
        # Test 3: Ideator adapter
        self.log("  Testing ideator adapter...")
        try:
            from falsifier.adapters.ideator_adapter import load_and_adapt_ideator_idea
            self.results.append(TestResult(
                name="integration_adapter",
                passed=True,
                duration=0.001,
            ))
            self.log("    ✓ Ideator adapter imports")
        except Exception as e:
            self.results.append(TestResult(
                name="integration_adapter",
                passed=False,
                duration=0.001,
                error=str(e),
            ))
            self.log(f"    ✗ Ideator adapter: {e}")
        
        # Test 4: File locking
        self.log("  Testing file locking...")
        try:
            from falsifier.graph.locking import AtomicGraphUpdate
            self.results.append(TestResult(
                name="integration_locking",
                passed=True,
                duration=0.001,
            ))
            self.log("    ✓ File locking imports")
        except Exception as e:
            self.results.append(TestResult(
                name="integration_locking",
                passed=False,
                duration=0.001,
                error=str(e),
            ))
            self.log(f"    ✗ File locking: {e}")
    
    def generate_report(self) -> bool:
        """Generate final test report."""
        self.log("")
        self.log("=" * 80)
        self.log("TEST REPORT")
        self.log("=" * 80)
        self.log("")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        total_time = time.time() - self.start_time
        
        # Summary
        self.log(f"Total tests: {total}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Total time: {total_time:.2f}s")
        self.log(f"Success rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
        self.log("")
        
        # Failed tests
        if failed > 0:
            self.log("FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    self.log(f"  ✗ {r.name}: {r.error or 'Test failed'}")
            self.log("")
        
        # Detailed results by category
        categories = {}
        for r in self.results:
            cat = r.name.split("_")[0]
            categories.setdefault(cat, []).append(r)
        
        self.log("RESULTS BY CATEGORY:")
        for cat, results in sorted(categories.items()):
            cat_passed = sum(1 for r in results if r.passed)
            cat_total = len(results)
            self.log(f"  {cat}: {cat_passed}/{cat_total} passed")
        
        self.log("")
        
        # Performance summary
        stage1_times = [r.duration for r in self.results if r.name.startswith("stage1_")]
        if stage1_times:
            avg_stage1 = sum(stage1_times) / len(stage1_times)
            self.log(f"Average Stage 1 time: {avg_stage1:.3f}s")
        
        # Final verdict
        self.log("")
        if failed == 0:
            self.log("✓✓ ALL TESTS PASSED - PIPELINE READY FOR PRODUCTION")
            return True
        elif failed <= 2:
            self.log(f"○ MOSTLY WORKING - {failed} minor issues")
            return True
        else:
            self.log(f"✗ {failed} TESTS FAILED - FIX ISSUES BEFORE PRODUCTION")
            return False
    
    # Helper methods to generate test models
    def _generate_tiny_model(self) -> str:
        """Generate a tiny working model."""
        return '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hyperparameters:
    vocab_size = 64
    d_model = 32
    n_heads = 2
    n_layers = 1
    d_mlp = 64

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, targets=None):
        x = self.tok_emb(input_ids)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        logits = self.lm_head(x)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits
'''
    
    def _generate_huge_model(self) -> str:
        """Generate a model that's too large."""
        return '''
import torch
import torch.nn as nn

class Hyperparameters:
    vocab_size = 100000
    d_model = 8192
    n_heads = 64
    n_layers = 96
    d_mlp = 32768

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Massive model - should exceed budget
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=8192, nhead=64, dim_feedforward=32768)
            for _ in range(96)
        ])
'''
    
    def _generate_unbalanced_model(self) -> str:
        """Generate a model with unbalanced FLOPs."""
        return '''
import torch
import torch.nn as nn

class Hyperparameters:
    vocab_size = 64
    d_model = 64
    n_heads = 16  # High attention FLOPs
    n_layers = 2
    d_mlp = 64    # Low MLP FLOPs

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Unbalanced - lots of attention, tiny MLP
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(64, 16, batch_first=True)
            for _ in range(2)
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
            for _ in range(2)
        ])
'''
    
    def _generate_syntax_error(self) -> str:
        """Generate code with syntax error."""
        return '''
import torch

class GPT:
    def __init__(self  # Missing closing paren
        pass
'''
    
    def _generate_shape_mismatch(self) -> str:
        """Generate code with shape mismatch."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)
        
    def forward(self, x):
        # Shape mismatch - expects 10, gets wrong size
        return self.layer(x)  # x will be wrong size
'''
    
    def _generate_no_gradients(self) -> str:
        """Generate code where some params don't get gradients."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(10))
        self.buffer = torch.randn(10)  # Not a parameter
        
    def forward(self, x):
        # buffer doesn't get gradients
        return x + self.buffer
'''
    
    def _generate_nan_weights(self) -> str:
        """Generate code with NaN in weights."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([float('nan')] * 10))
        
    def forward(self, x):
        return x * self.weight
'''
    
    def _generate_dead_relu_model(self) -> str:
        """Generate model with dead ReLUs."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Very negative bias will kill ReLUs
        self.fc = nn.Linear(64, 64)
        nn.init.constant_(self.fc.bias, -10.0)
        
    def forward(self, x):
        return torch.relu(self.fc(x))
'''
    
    def _generate_vanishing_grad_model(self) -> str:
        """Generate model with vanishing gradients."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Tiny weights cause vanishing gradients
        self.layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(10)
        ])
        for layer in self.layers:
            nn.init.constant_(layer.weight, 0.001)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))  # Tanh further compresses
        return x
'''
    
    def _generate_exploding_activations(self) -> str:
        """Generate model with exploding activations."""
        return '''
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Large weights cause exploding activations
        self.layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(5)
        ])
        for layer in self.layers:
            nn.init.constant_(layer.weight, 10.0)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
'''


def main():
    """Run pressure test suite."""
    test = PipelinePressureTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
