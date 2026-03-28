from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ..types import ModelSignature, SmokeDiagnostics


MINIMAL_TRAIN_GPT_ENV = {
    "NUM_LAYERS": "2",
    "MODEL_DIM": "32",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "2",
    "VOCAB_SIZE": "64",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_SEQ_LEN": "8",
    "VAL_BATCH_SIZE": "8",
    "TRAIN_BATCH_TOKENS": "8",
    "ITERATIONS": "1",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "0",
}


@contextmanager
def _patched_env(overrides: dict[str, str] | None):
    merged = dict(MINIMAL_TRAIN_GPT_ENV)
    if overrides:
        merged.update({key: str(value) for key, value in overrides.items()})

    previous: dict[str, str | None] = {}
    for key, value in merged.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_train_gpt_module(train_gpt_path: str | Path, env_overrides: dict[str, str] | None = None) -> ModuleType:
    path = Path(train_gpt_path).resolve()
    module_name = f"falsifier_train_gpt_{uuid.uuid4().hex}"
    with _patched_env(env_overrides):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load train_gpt.py from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def instantiate_minimal_model(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[ModuleType, torch.nn.Module]:
    module = load_train_gpt_module(train_gpt_path, env_overrides=env_overrides)
    args = module.Hyperparameters()
    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )
    return module, model


def model_signature(model: torch.nn.Module, smoke_loss: float | None = None) -> ModelSignature:
    return ModelSignature(
        param_count=sum(int(param.numel()) for param in model.parameters()),
        trainable_param_count=sum(int(param.numel()) for param in model.parameters() if param.requires_grad),
        num_layers=len(model.blocks),
        model_dim=int(model.tok_emb.embedding_dim),
        num_heads=int(model.blocks[0].attn.num_heads),
        num_kv_heads=int(model.blocks[0].attn.num_kv_heads),
        tie_embeddings=bool(model.tie_embeddings),
        smoke_loss=smoke_loss,
    )


def smoke_test_train_gpt(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> ModelSignature:
    signature, _ = run_smoke_diagnostics(train_gpt_path, env_overrides=env_overrides)
    return signature


def run_smoke_diagnostics(
    train_gpt_path: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[ModelSignature, SmokeDiagnostics]:
    _, model = instantiate_minimal_model(train_gpt_path, env_overrides=env_overrides)
    seq_len = 8
    vocab_size = int(model.tok_emb.num_embeddings)
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0) % vocab_size
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    loss = model(input_ids, target_ids)
    if not torch.isfinite(loss):
        raise ValueError("train_gpt smoke test produced a non-finite loss")
    loss.backward()
    params_without_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    diagnostics = SmokeDiagnostics(
        forward_ok=True,
        backward_ok=not params_without_grad,
        loss_is_finite=True,
        params_without_grad=params_without_grad,
    )
    return model_signature(model, smoke_loss=float(loss.item())), diagnostics
