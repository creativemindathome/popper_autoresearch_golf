from pathlib import Path
import json
import subprocess
import sys

from falsifier.stage1.orchestrator import run_stage_1
from falsifier.types import CandidatePackage, TheoryHistoryRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_GPT = REPO_ROOT / "train_gpt.py"


def test_stage1_rejects_invalid_candidate():
    candidate = CandidatePackage(
        theory_id="",
        train_gpt_path=REPO_ROOT / "missing.py",
        what_and_why="too short",
    )
    out = run_stage_1(candidate)
    assert out.verdict == "reject"
    assert out.validation.ok is False


def test_stage1_promotes_baseline_candidate_with_novel_explanation():
    candidate = CandidatePackage(
        theory_id="baseline-novel",
        train_gpt_path=TRAIN_GPT,
        what_and_why="Redistribute early residual routing so the architecture can separate local processing from later global mixing.",
        reference_theories=["Focus depth increases after the embedding stage."],
    )
    out = run_stage_1(candidate)
    assert out.verdict == "promote"
    assert out.model_signature is not None
    assert out.budget is not None
    assert out.budget.within_budget is True
    assert out.smoke is not None
    assert out.stage_reached == "promote"


def test_stage1_refutes_near_duplicate_candidate():
    text = "Redistribute early residual routing so the architecture can separate local processing from later global mixing."
    candidate = CandidatePackage(
        theory_id="baseline-duplicate",
        train_gpt_path=TRAIN_GPT,
        what_and_why=text,
        theory_history=[
            TheoryHistoryRecord(
                theory_id="prior-duplicate",
                verdict="refuted",
                theory_text=text,
            )
        ],
    )
    out = run_stage_1(candidate)
    assert out.verdict == "refute"
    assert out.novelty_score < 0.55
    assert out.stage_reached == "T1"
    assert out.precedent_evidence is not None
    assert out.precedent_evidence.matched_theory_id == "prior-duplicate"
    assert out.t1_mode == "graph_aware_precedent"


def test_stage1_refutes_candidate_from_failure_context_history():
    candidate = CandidatePackage(
        theory_id="baseline-history-context",
        train_gpt_path=TRAIN_GPT,
        what_and_why="Stabilize qk gain saturation with a softer logit clamp in the attention path.",
        theory_history=[
            TheoryHistoryRecord(
                theory_id="prior-attention-instability",
                verdict="refuted",
                theory_text="Tune the attention path to improve optimization stability.",
                failure_context="Refuted after qk gain saturation caused unstable logits during training.",
                related_theory_ids=["prior-softcap-neighbor"],
            ),
            TheoryHistoryRecord(
                theory_id="prior-softcap-neighbor",
                verdict="surviving",
                theory_text="Attention softcap changes can contain extreme logits.",
                mechanism_tags=["logit clamp", "qk gain", "attention saturation"],
            ),
        ],
    )
    out = run_stage_1(candidate)
    assert out.verdict == "refute"
    assert out.precedent_evidence is not None
    assert out.precedent_evidence.matched_theory_id == "prior-attention-instability"
    assert "graph_context" in out.precedent_evidence.matched_fields


def test_stage1_refutes_over_budget_candidate(tmp_path):
    candidate_file = tmp_path / "train_gpt.py"
    candidate_file.write_text(
        """
class Hyperparameters:
    vocab_size = int(__import__("os").environ.get("VOCAB_SIZE", 1000000))
    num_layers = int(__import__("os").environ.get("NUM_LAYERS", 1024))
    model_dim = int(__import__("os").environ.get("MODEL_DIM", 4096))
    num_heads = int(__import__("os").environ.get("NUM_HEADS", 16))
    num_kv_heads = int(__import__("os").environ.get("NUM_KV_HEADS", 8))
    mlp_mult = int(__import__("os").environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(__import__("os").environ.get("TIE_EMBEDDINGS", "1")))
    iterations = int(__import__("os").environ.get("ITERATIONS", 20000))
    train_batch_tokens = int(__import__("os").environ.get("TRAIN_BATCH_TOKENS", 524288))
    train_seq_len = int(__import__("os").environ.get("TRAIN_SEQ_LEN", 1024))
"""
    )
    candidate = CandidatePackage(
        theory_id="over-budget",
        train_gpt_path=candidate_file,
        what_and_why="Scale the architecture up so aggressively that the static budget gate rejects it before runtime smoke testing.",
    )
    out = run_stage_1(candidate)
    assert out.verdict == "refute"
    assert out.stage_reached == "T2"
    assert out.budget is not None
    assert out.budget.within_budget is False


def test_stage1_fails_import_broken_candidate_at_t3(tmp_path):
    candidate_file = tmp_path / "train_gpt.py"
    candidate_file.write_text(
        """
import definitely_missing_package

class Hyperparameters:
    vocab_size = 64
    num_layers = 2
    model_dim = 32
    num_heads = 4
    num_kv_heads = 2
    mlp_mult = 2
    tie_embeddings = True
    tied_embed_init_std = 0.5
    logit_softcap = 30.0
    rope_base = 10000
    qk_gain_init = 1.0
"""
    )
    candidate = CandidatePackage(
        theory_id="import-broken",
        train_gpt_path=candidate_file,
        what_and_why="Probe a radically different routing design with a deliberately broken candidate import surface.",
    )
    out = run_stage_1(candidate)
    assert out.verdict == "implementation_fail"
    assert out.stage_reached == "T3"
    assert any("smoke test failed" in reason for reason in out.reasons)


def test_stage1_fails_construction_broken_candidate_at_t3(tmp_path):
    candidate_file = tmp_path / "train_gpt.py"
    candidate_file.write_text(
        """
class Hyperparameters:
    vocab_size = 64
    num_layers = 2
    model_dim = 32
    num_heads = 4
    num_kv_heads = 2
    mlp_mult = 2
    tie_embeddings = True
    tied_embed_init_std = 0.5
    logit_softcap = 30.0
    rope_base = 10000
    qk_gain_init = 1.0

class GPT:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("construction exploded")
"""
    )
    candidate = CandidatePackage(
        theory_id="construction-broken",
        train_gpt_path=candidate_file,
        what_and_why="Probe a radically different routing design with a deliberately broken candidate constructor.",
    )
    out = run_stage_1(candidate)
    assert out.verdict == "implementation_fail"
    assert out.stage_reached == "T3"
    assert any("construction exploded" in reason for reason in out.reasons)


def test_cli_writes_verdict_artifact(tmp_path):
    input_path = tmp_path / "candidate.json"
    output_path = tmp_path / "verdict.json"
    input_path.write_text(
        json.dumps(
            {
                "theory_id": "cli-smoke",
                "train_gpt_path": str(TRAIN_GPT),
                "what_and_why": "Redistribute early residual routing so the architecture can separate local processing from later global mixing.",
                "theory_history": [
                    {
                        "theory_id": "prior-depth",
                        "verdict": "surviving",
                        "theory_text": "Focus depth increases after the embedding stage.",
                    }
                ],
            }
        )
    )
    subprocess.run(
        [sys.executable, "-m", "falsifier.main", "--input", str(input_path), "--output", str(output_path)],
        check=True,
    )
    payload = json.loads(output_path.read_text())
    assert payload["theory_id"] == "cli-smoke"
    assert payload["verdict"] == "promote"
    assert payload["stage_reached"] == "promote"
    assert payload["t1_mode"] == "graph_aware_precedent"
    assert payload["precedent_evidence"]["query_mode"] == "graph_aware_precedent"
