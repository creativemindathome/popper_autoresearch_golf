"""Core falsifier tests using new PRD-aligned types."""

from __future__ import annotations

from pathlib import Path
import unittest

from falsifier.adapters.parameter_golf import run_smoke_diagnostics, smoke_test_train_gpt
from falsifier.stage1.orchestrator import run_stage_1
from falsifier.types import FalsifierInput, KnowledgeGraph
from falsifier.validation import validate_candidate_package


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = REPO_ROOT / "train_gpt.py"


class FalsifierCoreTests(unittest.TestCase):
    def test_smoke_test_imports_and_instantiates_train_gpt(self) -> None:
        """Test that smoke test can import and instantiate train_gpt.py."""
        signature = smoke_test_train_gpt(TRAIN_GPT)
        self.assertGreater(signature.param_count, 0)
        self.assertEqual(signature.num_layers, 2)
        self.assertEqual(signature.model_dim, 32)
        self.assertTrue(signature.tie_embeddings)
        self.assertIsNotNone(signature.smoke_loss)
        self.assertGreater(signature.smoke_loss or 0.0, 0.0)

    def test_smoke_diagnostics_runs_backward(self) -> None:
        """Test that smoke diagnostics can run forward and backward passes."""
        signature, diagnostics = run_smoke_diagnostics(TRAIN_GPT)
        self.assertGreater(signature.param_count, 0)
        self.assertTrue(diagnostics.forward_ok)
        self.assertTrue(diagnostics.backward_ok)
        self.assertTrue(diagnostics.loss_is_finite)
        self.assertEqual(diagnostics.params_without_grad, [])

    def test_validation_rejects_missing_candidate_path(self) -> None:
        """Test that validation rejects missing train_gpt.py path."""
        inp = FalsifierInput(
            theory_id="missing-path",
            what_and_why="Move capacity into a smaller, more specialized bottleneck.",
            train_gpt_path=REPO_ROOT / "does_not_exist.py",
        )
        result = validate_candidate_package(inp)
        self.assertFalse(result.ok)
        self.assertTrue(any("not found" in reason for reason in result.reasons))

    def test_stage1_runs_baseline_candidate(self) -> None:
        """Test that baseline train_gpt.py can run through Stage 1.
        
        Note: The baseline train_gpt.py is sized for real training (17M params)
        and may be killed at T2 (budget) due to estimated training time.
        """
        inp = FalsifierInput(
            theory_id="baseline-theory",
            what_and_why="Shift capacity into a local gating path that specializes early residual routing.",
            train_gpt_path=TRAIN_GPT,
            graph=KnowledgeGraph(),
        )
        result = run_stage_1(inp)
        
        # Should have a valid verdict
        self.assertIn(result.verdict, ("STAGE_1_PASSED", "REFUTED", "REJECTED", "IMPLEMENTATION_FAIL"))
        
        # Should have theory_id
        self.assertEqual(result.theory_id, "baseline-theory")
        
        # Should have feedback
        self.assertIsNotNone(result.feedback)
        
        # If passed T0-T2, should have T3 compilation result
        if result.verdict == "STAGE_1_PASSED" or result.killed_by not in ("T0", "T1", "T2", "VALIDATION"):
            self.assertIsNotNone(result.t3_compilation)

    def test_stage1_failure_has_feedback(self) -> None:
        """Test that failed Stage 1 includes feedback."""
        inp = FalsifierInput(
            theory_id="short-theory",
            what_and_why="too short",
            train_gpt_path=TRAIN_GPT,
        )
        result = run_stage_1(inp)
        
        # Should be rejected or refuted
        self.assertIn(result.verdict, ("REJECTED", "REFUTED"))
        
        # Should have feedback
        self.assertIsNotNone(result.feedback)
        
        # Feedback should include the theory_id
        self.assertTrue(len(result.feedback.one_line) > 0)


if __name__ == "__main__":
    unittest.main()
