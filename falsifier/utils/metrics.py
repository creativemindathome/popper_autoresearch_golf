"""
Metrics computation utilities for falsifier testing.

Used by T4 (Signal Propagation), T5 (Init Diagnostics), T6 (Checkpoint),
T7 (Micro-Training), and Stage 2 adversarial testing.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F


def compute_val_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """
    Compute cross-entropy loss on validation batch.

    Args:
        model: The model to evaluate
        batch: Dict with 'input_ids' and 'target_ids' tensors

    Returns:
        Cross-entropy loss as float
    """
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    with torch.no_grad():
        logits = model(input_ids)

        # Flatten for cross-entropy
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

        loss = F.cross_entropy(logits, target_ids, reduction="mean")

    return float(loss.item())


def compute_attention_entropy_all_heads(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute per-head attention entropy for all layers.

    Args:
        model: The model to evaluate
        batch: Dict with 'input_ids' tensor

    Returns:
        Dict mapping layer.head to entropy value
    """
    entropies: dict[str, float] = {}
    input_ids = batch["input_ids"]

    # Storage for attention weights from hooks
    attention_weights: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(module, input, output):
            # Assuming attention output includes attention weights
            # or we can capture them from the attention module
            if hasattr(module, "attn_weights"):
                attention_weights[layer_name] = module.attn_weights.detach()
        return hook

    # Register hooks on attention modules
    handles = []
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model(input_ids)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Compute entropy for each head
    for layer_name, attn in attention_weights.items():
        # attn shape: [batch, heads, seq, seq]
        if attn.dim() == 4:
            num_heads = attn.size(1)
            for h in range(num_heads):
                head_attn = attn[:, h, :, :]  # [batch, seq, seq]
                # Entropy: -sum(p * log(p), dim=-1)
                # Average over batch and seq
                entropy = -torch.sum(
                    head_attn * torch.log(head_attn + 1e-10),
                    dim=-1,
                )
                mean_entropy = entropy.mean().item()
                entropies[f"{layer_name}.head_{h}"] = mean_entropy

    return entropies


def compute_global_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute global gradient norm across all parameters.

    sqrt(sum(p.grad.norm()**2 for all parameters with gradients))

    Args:
        model: The model with gradients

    Returns:
        Global gradient norm as float
    """
    total_norm_sq = 0.0

    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            param_norm = param.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2

    return math.sqrt(total_norm_sq)


def classify_component(param_name: str) -> str:
    """
    Classify parameter by component type based on its name.

    Args:
        param_name: Full parameter name (e.g., 'blocks.0.attn.q_proj.weight')

    Returns:
        One of: "attn", "mlp", "embed", "norm", "other"
    """
    name_lower = param_name.lower()

    # Attention patterns
    attn_patterns = [
        "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value", "wq", "wk", "wv", "wo",
    ]
    if any(p in name_lower for p in attn_patterns):
        return "attn"

    # MLP patterns
    mlp_patterns = [
        "mlp", "ffn", "feedforward", "fc", "up_proj", "down_proj",
        "gate_proj", "w1", "w2", "w3",
    ]
    if any(p in name_lower for p in mlp_patterns):
        return "mlp"

    # Embedding patterns
    embed_patterns = [
        "emb", "embed", "token", "position", "pos_emb", "tok_emb",
        "word_embedding", "input_embedding",
    ]
    if any(p in name_lower for p in embed_patterns):
        return "embed"

    # Normalization patterns
    norm_patterns = [
        "norm", "ln", "layernorm", "rmsnorm", "rms_norm", "bn", "batchnorm",
        "scale", "gain", "bias_norm",
    ]
    if any(p in name_lower for p in norm_patterns):
        return "norm"

    return "other"


def is_transformer_layer_output(module_name: str) -> bool:
    """
    Check if module is a transformer layer output for hooks.

    Args:
        module_name: Full module name

    Returns:
        True if this is a layer output that should be captured
    """
    name_lower = module_name.lower()

    # Patterns indicating layer outputs
    layer_output_patterns = [
        r"blocks\.\d+$",
        r"layers\.\d+$",
        r"transformer\.h\.\d+$",
        r"encoder\.layer\.\d+$",
        r"decoder\.layer\.\d+$",
        r"\.\d+\.output$",
        r"\.\d+\.add$",
        r"\.\d+\.residual$",
    ]

    for pattern in layer_output_patterns:
        if re.search(pattern, name_lower):
            return True

    # Specific module types that indicate layer output
    output_indicators = [
        "final", "output", "add_norm", "resid", "residual",
    ]

    # Check if it's a block/layer container
    block_patterns = [
        r"blocks\.[\d_]+",
        r"layers\.[\d_]+",
        r"h\.[\d_]+",
    ]

    for pattern in block_patterns:
        if re.search(pattern, name_lower):
            # And has output indicator or is the end of the block
            if any(ind in name_lower for ind in output_indicators):
                return True
            # Or if it's directly the block module
            if re.match(rf"{pattern}$", name_lower):
                return True

    return False
