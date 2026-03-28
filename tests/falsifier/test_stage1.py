"""Tests for Stage 1 orchestrator using new PRD-aligned types."""

from pathlib import Path
import json
import subprocess
import sys

from falsifier.stage1.orchestrator import run_stage_1
from falsifier.types import FalsifierInput, KnowledgeGraph


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_GPT = REPO_ROOT / "train_gpt.py"


def test_stage1_rejects_invalid_candidate():
    """Invalid candidate should be rejected at validation."""
    inp = FalsifierInput(
        theory_id="",
        what_and_why="too short",
        train_gpt_path=REPO_ROOT / "missing.py",
    )
    out = run_stage_1(inp)
    assert out.verdict == "REJECTED"
    assert out.killed_by == "VALIDATION"


def test_stage1_runs_baseline_candidate_with_streamlined_gates():
    """Baseline train_gpt.py should pass streamlined Stage 1 or be refuted at specific gate.
    
    Note: The baseline train_gpt.py is sized for real training (17M params)
    and may be killed at T2 (budget) due to estimated training time.
    Streamlined gates: T2 -> T3 -> {T4,T5} -> T7 (T0/T1/T6 removed)
    """
    inp = FalsifierInput(
        theory_id="baseline-novel",
        what_and_why="Redistribute early residual routing so the architecture can separate local processing from later global mixing.",
        train_gpt_path=TRAIN_GPT,
        graph=KnowledgeGraph(),
    )
    out = run_stage_1(inp)
    
    # Should pass Stage 1 or be refuted at specific gate
    assert out.theory_id == "baseline-novel"
    assert out.verdict in ("STAGE_1_PASSED", "REFUTED")
    
    # Check that streamlined gates ran (T0/T1/T6 removed)
    assert out.t2_budget is not None  # Always runs first
    
    # T2 should have enhanced per-component FLOPs analysis
    assert out.t2_budget.attn_flops_ratio > 0
    assert out.t2_budget.mlp_flops_ratio > 0
    
    # If passed T2, should have T3 compilation result
    if out.killed_by not in ("T2", "VALIDATION"):
        assert out.t3_compilation is not None
        # T3 should have construction diagnostics
        assert hasattr(out.t3_compilation, 'layer_shapes_consistent')
    
    # Feedback should be provided
    assert out.feedback is not None


def test_stage1_refutes_over_budget_candidate(tmp_path):
    """Over-budget candidate should be refuted at T2."""
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
    inp = FalsifierInput(
        theory_id="over-budget",
        what_and_why="Scale the architecture up so aggressively that the static budget gate rejects it before runtime smoke testing.",
        train_gpt_path=candidate_file,
        graph=KnowledgeGraph(),
    )
    out = run_stage_1(inp)
    
    # Should be refuted at T2 (budget)
    assert out.verdict == "REFUTED"
    assert out.killed_by == "T2"
    assert out.t2_budget is not None


def test_stage1_fails_import_broken_candidate_at_t3(tmp_path):
    """Import-broken candidate should be refuted at T3."""
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
    inp = FalsifierInput(
        theory_id="import-broken",
        what_and_why="Probe a radically different routing design with a deliberately broken candidate import surface.",
        train_gpt_path=candidate_file,
        graph=KnowledgeGraph(),
    )
    out = run_stage_1(inp)
    
    # Should be refuted at T3 (compilation)
    assert out.verdict == "REFUTED"
    assert out.killed_by == "T3"
    assert "import" in (out.kill_reason or "").lower()


def test_stage1_fails_construction_broken_candidate_at_t3(tmp_path):
    """Construction-broken candidate should be refuted at T3."""
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
    inp = FalsifierInput(
        theory_id="construction-broken",
        what_and_why="Probe a radically different routing design with a deliberately broken candidate constructor.",
        train_gpt_path=candidate_file,
        graph=KnowledgeGraph(),
    )
    out = run_stage_1(inp)
    
    # Should be refuted at T3 (compilation)
    assert out.verdict == "REFUTED"
    assert out.killed_by == "T3"


def test_cli_writes_verdict_artifact(tmp_path):
    """CLI should write verdict artifact."""
    input_path = tmp_path / "candidate.json"
    output_path = tmp_path / "verdict.json"
    input_path.write_text(
        json.dumps(
            {
                "theory_id": "cli-smoke",
                "train_gpt_path": str(TRAIN_GPT),
                "what_and_why": "Redistribute early residual routing so the architecture can separate local processing from later global mixing.",
            }
        )
    )
    subprocess.run(
        [sys.executable, "-m", "falsifier.main", "--candidate-json", str(input_path), "--output-json", str(output_path)],
        check=True,
    )
    payload = json.loads(output_path.read_text())
    assert payload["theory_id"] == "cli-smoke"
    assert payload["verdict"] in ("STAGE_1_PASSED", "REFUTED")
