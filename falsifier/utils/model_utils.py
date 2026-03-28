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
from torch.optim import Adam, AdamW

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


def _detect_optimizer_type(source_code: str) -> dict[str, Any]:
    """Detect optimizer type and patterns from source code.

    Args:
        source_code: The train_gpt.py source code

    Returns:
        Dict with detected patterns:
        - has_muon: bool - Muon class is defined
        - has_split_optimizers: bool - SplitOptimizers class is defined
        - has_muon_hparams: bool - Muon hyperparameters exist
        - has_lr_groups: bool - Multiple learning rate parameters exist
        - optimizer_style: str - "muon", "split_optimizers", "adamw", etc.
    """
    patterns = {
        "has_muon": False,
        "has_split_optimizers": False,
        "has_muon_hparams": False,
        "has_lr_groups": False,
        "optimizer_style": "adamw",
    }

    # Check for Muon optimizer class definition
    if "class Muon" in source_code or "class Muon(torch.optim.Optimizer)" in source_code:
        patterns["has_muon"] = True

    # Check for SplitOptimizers class (MLX style)
    if "class SplitOptimizers" in source_code:
        patterns["has_split_optimizers"] = True

    # Check for Muon hyperparameters
    muon_hp_patterns = [
        "muon_momentum",
        "muon_backend_steps",
        "muon_momentum_warmup",
    ]
    patterns["has_muon_hparams"] = any(p in source_code for p in muon_hp_patterns)

    # Check for learning rate groups
    lr_patterns = ["embed_lr", "head_lr", "matrix_lr", "scalar_lr", "tied_embed_lr"]
    lr_count = sum(1 for p in lr_patterns if p in source_code)
    patterns["has_lr_groups"] = lr_count >= 2

    # Determine optimizer style
    if patterns["has_split_optimizers"]:
        patterns["optimizer_style"] = "split_optimizers"
    elif patterns["has_muon"] and patterns["has_muon_hparams"]:
        patterns["optimizer_style"] = "muon"
    elif patterns["has_lr_groups"]:
        patterns["optimizer_style"] = "lr_groups"

    return patterns


def _get_control_tensor_patterns() -> list[str]:
    """Get control tensor name patterns for parameter classification."""
    return ["skip_scale", "qk_gain", "logit_scale"]


def _classify_pytorch_params(
    model: torch.nn.Module,
    hparams: Any,
) -> dict[str, list[torch.nn.Parameter]]:
    """Classify model parameters into groups for PyTorch optimizer setup.

    Args:
        model: PyTorch model
        hparams: Hyperparameters instance

    Returns:
        Dict with parameter lists for each group
    """
    control_patterns = _get_control_tensor_patterns()

    # Get parameters from transformer blocks if available
    matrix_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []

    # Try to get block parameters
    if hasattr(model, "blocks"):
        block_named_params = list(model.blocks.named_parameters())
        matrix_params = [
            p
            for name, p in block_named_params
            if p.ndim == 2 and not any(pattern in name for pattern in control_patterns)
        ]
        scalar_params = [
            p
            for name, p in block_named_params
            if p.ndim < 2 or any(pattern in name for pattern in control_patterns)
        ]
    else:
        # Fallback: classify all parameters by dimension
        for p in model.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                scalar_params.append(p)

    # Handle skip_weights if present
    if hasattr(model, "skip_weights") and model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    # Get embedding and head parameters
    tok_emb_param: torch.nn.Parameter | None = None
    lm_head_param: torch.nn.Parameter | None = None

    if hasattr(model, "tok_emb"):
        tok_emb_param = model.tok_emb.weight if hasattr(model.tok_emb, "weight") else None
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm_head_param = model.lm_head.weight if hasattr(model.lm_head, "weight") else None

    return {
        "matrix": matrix_params,
        "scalar": scalar_params,
        "tok_emb": tok_emb_param,
        "lm_head": lm_head_param,
    }


def _create_pytorch_split_optimizers(
    model: torch.nn.Module,
    module: ModuleType,
    hparams: Any,
) -> list[torch.optim.Optimizer]:
    """Create split optimizers for PyTorch (Muon + Adam combination).

    Args:
        model: PyTorch model
        module: train_gpt module containing optimizer classes
        hparams: Hyperparameters instance

    Returns:
        List of optimizer instances
    """
    # Extract hyperparameters with defaults
    beta1 = getattr(hparams, "beta1", 0.9)
    beta2 = getattr(hparams, "beta2", 0.95)
    eps = getattr(hparams, "adam_eps", 1e-8)
    matrix_lr = getattr(hparams, "matrix_lr", 0.04)
    scalar_lr = getattr(hparams, "scalar_lr", 0.04)
    tied_embed_lr = getattr(hparams, "tied_embed_lr", 0.05)
    embed_lr = getattr(hparams, "embed_lr", 0.6)
    head_lr = getattr(hparams, "head_lr", 0.008)

    # Determine if embeddings are tied
    tie_embeddings = getattr(hparams, "tie_embeddings", True)
    token_lr = tied_embed_lr if tie_embeddings else embed_lr

    # Classify parameters
    param_groups = _classify_pytorch_params(model, hparams)

    optimizers: list[torch.optim.Optimizer] = []

    # Token embedding optimizer (Adam)
    if param_groups["tok_emb"] is not None:
        optimizer_tok = Adam(
            [{"params": [param_groups["tok_emb"]], "lr": token_lr, "base_lr": token_lr}],
            betas=(beta1, beta2),
            eps=eps,
            fused=True,
        )
        optimizers.append(optimizer_tok)

    # Head optimizer (Adam) - only if embeddings not tied
    if param_groups["lm_head"] is not None and not tie_embeddings:
        optimizer_head = Adam(
            [{"params": [param_groups["lm_head"]], "lr": head_lr, "base_lr": head_lr}],
            betas=(beta1, beta2),
            eps=eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # Matrix parameters optimizer (Muon if available, else AdamW)
    if param_groups["matrix"]:
        if hasattr(module, "Muon"):
            # Use Muon optimizer from module
            muon_momentum = getattr(hparams, "muon_momentum", 0.95)
            muon_backend_steps = getattr(hparams, "muon_backend_steps", 5)

            optimizer_muon = module.Muon(
                param_groups["matrix"],
                lr=matrix_lr,
                momentum=muon_momentum,
                backend_steps=muon_backend_steps,
            )
            # Add base_lr for lr scheduling compatibility
            for group in optimizer_muon.param_groups:
                group["base_lr"] = matrix_lr
            optimizers.append(optimizer_muon)
        else:
            # Fallback to AdamW for matrix params
            optimizer_matrix = AdamW(
                [{"params": param_groups["matrix"], "lr": matrix_lr, "base_lr": matrix_lr}],
                betas=(beta1, beta2),
                eps=eps,
            )
            optimizers.append(optimizer_matrix)

    # Scalar parameters optimizer (Adam)
    if param_groups["scalar"]:
        optimizer_scalar = Adam(
            [{"params": param_groups["scalar"], "lr": scalar_lr, "base_lr": scalar_lr}],
            betas=(beta1, beta2),
            eps=eps,
            fused=True,
        )
        optimizers.append(optimizer_scalar)

    return optimizers


def setup_optimizer_from_source(
    model: Any,
    source_code: str,
    framework: str = "pytorch",
) -> Any:
    """Setup optimizer from train_gpt.py source.

    Parses the source code to detect optimizer type (AdamW, Muon, Adam, etc.),
    extracts learning rates per parameter group from the Hyperparameters class,
    and returns an appropriate optimizer instance for the specified framework.

    Args:
        model: The model to optimize (PyTorch nn.Module or MLX nn.Module)
        source_code: Path to train_gpt.py file or the source code string
        framework: "pytorch" or "mlx"

    Returns:
        Optimizer instance:
        - For PyTorch with Muon: list of optimizers [Adam(tok_emb), Muon(matrix), Adam(scalar)]
        - For PyTorch AdamW only: single AdamW optimizer
        - For MLX: SplitOptimizers instance

    Examples:
        >>> # PyTorch with Muon optimizer
        >>> optimizers = setup_optimizer_from_source(model, "train_gpt.py", "pytorch")
        >>> for opt in optimizers:
        ...     opt.zero_grad()
        >>> loss.backward()
        >>> for opt in optimizers:
        ...     opt.step()

        >>> # MLX with SplitOptimizers
        >>> opt = setup_optimizer_from_source(model, "train_gpt_mlx.py", "mlx")
        >>> loss, grads = mx.value_and_grad(model)(x, y)
        >>> opt.step(model, grads, step=0, lr_mul=1.0)
    """
    # Determine if source_code is a file path or actual source
    source_path = Path(source_code)
    if source_path.exists():
        source = source_path.read_text()
        file_path = str(source_path)
    else:
        source = source_code
        file_path = None

    # Detect optimizer patterns
    patterns = _detect_optimizer_type(source)

    # Load the module to get Hyperparameters
    if file_path:
        module, _ = instantiate_model(file_path)
    else:
        # If source code string provided without file, we need to parse it
        # This is a fallback - ideally file_path is provided
        raise ValueError("File path required for optimizer setup. Please provide path to train_gpt.py")

    hparams = module.Hyperparameters()

    if framework == "mlx":
        # MLX: Use SplitOptimizers from the module
        if patterns["has_split_optimizers"] and hasattr(module, "SplitOptimizers"):
            return module.SplitOptimizers(model, hparams)
        else:
            # Fallback: use standard MLX Adam
            try:
                import mlx.optimizers as optim

                lr = getattr(hparams, "matrix_lr", 0.04)
                beta1 = getattr(hparams, "beta1", 0.9)
                beta2 = getattr(hparams, "beta2", 0.95)
                eps = getattr(hparams, "adam_eps", 1e-8)
                return optim.Adam(learning_rate=lr, betas=[beta1, beta2], eps=eps)
            except ImportError:
                raise RuntimeError("MLX not available for optimizer creation")

    else:  # pytorch
        # PyTorch: Create split optimizers
        if patterns["has_lr_groups"] or patterns["has_muon"]:
            # Use split optimizer approach (Muon + Adam)
            return _create_pytorch_split_optimizers(model, module, hparams)
        else:
            # Simple AdamW optimizer
            lr = getattr(hparams, "matrix_lr", 0.04)
            beta1 = getattr(hparams, "beta1", 0.9)
            beta2 = getattr(hparams, "beta2", 0.95)
            eps = getattr(hparams, "adam_eps", 1e-8)
            return AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
