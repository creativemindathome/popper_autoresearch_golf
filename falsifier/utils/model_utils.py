"""
Model utilities - thin wrappers around parameter_golf adapter.

Used by T4, T5, T6, and Stage 2 for model instantiation, loading,
and optimizer setup.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch.optim import AdamW

from ..adapters.parameter_golf import instantiate_minimal_model


def instantiate_model(
    source: str,
    env_overrides: dict[str, Any] | None = None,
    block_imports: list[str] | None = None,
) -> tuple[ModuleType, torch.nn.Module]:
    """
    Instantiate a minimal model from train_gpt.py source.

    Thin wrapper around parameter_golf.instantiate_minimal_model.

    Args:
        source: Path to train_gpt.py file or source string (handled by adapter)
        env_overrides: Optional environment variable overrides for model config
        block_imports: List of module names to block (e.g., ["sentencepiece"])

    Returns:
        Tuple of (module, model) where module contains Hyperparameters class
    """
    return instantiate_minimal_model(
        source,
        env_overrides=env_overrides,
        block_imports=block_imports or ["sentencepiece", "spm"],
    )


def load_model(
    source: str,
    checkpoint_path: Path | None = None,
) -> tuple[ModuleType, torch.nn.Module]:
    """
    Load model with optional checkpoint weights.

    Args:
        source: Path to train_gpt.py file
        checkpoint_path: Optional path to checkpoint to load (skip if None)

    Returns:
        Tuple of (module, model) with loaded weights if checkpoint provided
    """
    module, model = instantiate_model(source)

    if checkpoint_path is not None and checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    return module, model


def setup_optimizer_from_source(
    model: torch.nn.Module,
    source: str,
) -> AdamW:
    """
    Extract optimizer config from Hyperparameters and return AdamW.

    Args:
        model: The model to optimize
        source: Path to train_gpt.py file to extract Hyperparameters from

    Returns:
        AdamW optimizer configured from Hyperparameters
    """
    # Load the module to get Hyperparameters
    module, _ = instantiate_model(source)
    hparams = module.Hyperparameters()

    # Extract optimizer settings from Hyperparameters
    lr = getattr(hparams, "matrix_lr", 0.04)
    beta1 = getattr(hparams, "beta1", 0.9)
    beta2 = getattr(hparams, "beta2", 0.95)
    eps = getattr(hparams, "adam_eps", 1e-8)

    return AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )
