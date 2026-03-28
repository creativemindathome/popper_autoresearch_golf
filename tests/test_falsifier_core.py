from __future__ import annotations

from pathlib import Path
import unittest

from falsifier.adapters.parameter_golf import run_smoke_diagnostics, smoke_test_train_gpt
from falsifier.stage1.orchestrator import run_stage1
from falsifier.types import CandidatePackage
from falsifier.validation import validate_candidate_package


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = REPO_ROOT / "train_gpt.py"


class FalsifierCoreTests(unittest.TestCase):
    def test_smoke_test_imports_and_instantiates_train_gpt(self) -> None:
        signature = smoke_test_train_gpt(TRAIN_GPT)
        self.assertGreater(signature.param_count, 0)
        self.assertEqual(signature.num_layers, 2)
        self.assertEqual(signature.model_dim, 32)
        self.assertTrue(signature.tie_embeddings)
        self.assertIsNotNone(signature.smoke_loss)
        self.assertGreater(signature.smoke_loss or 0.0, 0.0)

    def test_smoke_diagnostics_runs_backward(self) -> None:
        signature, diagnostics = run_smoke_diagnostics(TRAIN_GPT)
        self.assertGreater(signature.param_count, 0)
        self.assertTrue(diagnostics.forward_ok)
        self.assertTrue(diagnostics.backward_ok)
        self.assertTrue(diagnostics.loss_is_finite)
        self.assertEqual(diagnostics.params_without_grad, [])

    def test_validation_rejects_missing_candidate_path(self) -> None:
        candidate = CandidatePackage(
            theory_id="missing-path",
            train_gpt_path=REPO_ROOT / "does_not_exist.py",
            what_and_why="Move capacity into a smaller, more specialized bottleneck.",
        )
        result = validate_candidate_package(candidate)
        self.assertFalse(result.ok)
        self.assertTrue(any("not found" in reason for reason in result.reasons))

    def test_stage1_promotes_novel_candidate(self) -> None:
        candidate = CandidatePackage(
            theory_id="novel-theory",
            train_gpt_path=TRAIN_GPT,
            what_and_why="Shift capacity into a local gating path that specializes early residual routing.",
            reference_theories=[
                "Use deeper layers to recover information after the embedding stage.",
            ],
        )
        result = run_stage1(candidate)
        self.assertEqual(result.verdict, "promote")
        self.assertGreaterEqual(result.novelty_score, 0.55)
        self.assertIsNotNone(result.model_signature)
        self.assertIsNotNone(result.smoke)
        self.assertTrue(result.smoke.backward_ok)

    def test_stage1_refutes_near_duplicate_theory(self) -> None:
        text = "Shift capacity into a local gating path that specializes early residual routing."
        candidate = CandidatePackage(
            theory_id="duplicate-theory",
            train_gpt_path=TRAIN_GPT,
            what_and_why=text,
            reference_theories=[text],
        )
        result = run_stage1(candidate)
        self.assertEqual(result.verdict, "refute")
        self.assertLess(result.novelty_score, 0.55)
        self.assertTrue(result.reasons)


if __name__ == "__main__":
    unittest.main()
