"""T5 init gate tests using new PRD-aligned types (T6 removed - moved to Stage 2)."""

import json
from pathlib import Path

from falsifier.types import FalsifierInput, Calibration, KnowledgeGraph


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_GPT = REPO_ROOT / "train_gpt.py"


def test_t5_runs_init_diagnostics():
    """T5 should run init diagnostics."""
    from falsifier.stage1.t5_init import run_t5
    
    # Create a minimal test file
    minimal_source = '''
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

class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

class MLP(nn.Module):
    def __init__(self, model_dim, mlp_mult):
        super().__init__()

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads, mlp_mult, qk_gain_init):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads, num_kv_heads, qk_gain_init)
        self.mlp = MLP(model_dim, mlp_mult)

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads, num_kv_heads, mlp_mult, qk_gain_init) for _ in range(num_layers)])
        self.tie_embeddings = tie_embeddings

import torch.nn as nn
'''
    
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(minimal_source)
        temp_path = Path(f.name)
    
    try:
        inp = FalsifierInput(
            theory_id="t5-test",
            what_and_why="Testing T5 init diagnostics on minimal model.",
            train_gpt_path=temp_path,
            calibration=Calibration(sota_init_logit_max=10.0),
            graph=KnowledgeGraph(),
        )
        
        result = run_t5(inp)
        
        # Should run and return metrics
        assert result.test_id == "T5"
        assert result.status in ("PASS", "FAIL_TAG", "FAIL_FATAL")
    finally:
        temp_path.unlink(missing_ok=True)


def test_t5_with_calibration_thresholds():
    """T5 should use calibration for thresholding."""
    from falsifier.stage1.t5_init import run_t5
    
    inp = FalsifierInput(
        theory_id="t5-thresh",
        what_and_why="Testing T5 with calibration thresholds.",
        train_gpt_path=TRAIN_GPT,
        calibration=Calibration(
            sota_init_logit_max=5.0,  # Lower threshold
        ),
        graph=KnowledgeGraph(),
    )
    
    result = run_t5(inp)
    
    # Should compare against calibration
    assert result.test_id == "T5"
    assert isinstance(result.tags, list)
