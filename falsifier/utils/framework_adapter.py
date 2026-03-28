"""Framework adapter for unified PyTorch and MLX support.

Provides framework-agnostic functions that work with both PyTorch and MLX models,
enabling the falsifier to handle models from either framework transparently.
"""

from __future__ import annotations

import math
from typing import Any, Union, Callable

# Framework availability checks
_torch_available = False
_mlx_available = False

# Public exports for framework availability
TORCH_AVAILABLE = False
MLX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn_torch
    import torch.nn.functional as F_torch

    _torch_available = True
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn_torch = None  # type: ignore
    F_torch = None  # type: ignore

try:
    import mlx.core as mx
    import mlx.nn as nn_mlx
    from mlx.utils import tree_flatten

    _mlx_available = True
    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    nn_mlx = None  # type: ignore
    tree_flatten = None  # type: ignore

# Type alias for models that could be from either framework
FrameworkModel = Any
FrameworkTensor = Any


def detect_framework(model: FrameworkModel) -> str:
    """Detect whether a model is PyTorch or MLX-based.

    Args:
        model: The model instance to check

    Returns:
        "pytorch" if PyTorch model, "mlx" if MLX model

    Raises:
        ValueError: If model type cannot be determined
    """
    if _torch_available and isinstance(model, nn_torch.Module):
        return "pytorch"

    if _mlx_available and isinstance(model, nn_mlx.Module):
        return "mlx"

    # Try duck-typing based on common attributes
    if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
        # PyTorch models have parameters() method
        if hasattr(model, "named_parameters"):
            return "pytorch"

    if hasattr(model, "state") and hasattr(model, "update"):
        # MLX models have state dict-like interface
        return "mlx"

    raise ValueError(f"Cannot detect framework for model type: {type(model).__name__}")


def get_parameters(model: FrameworkModel) -> list:
    """Get all parameters from a model, regardless of framework.

    Args:
        model: PyTorch or MLX model

    Returns:
        List of parameter tensors/arrays
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        return list(model.parameters())
    else:  # mlx
        # MLX uses tree_flatten to get all parameters
        from mlx.utils import tree_flatten

        return [p for _, p in tree_flatten(model.parameters())]


def get_named_parameters(model: FrameworkModel) -> list[tuple[str, Any]]:
    """Get named parameters from a model.

    Args:
        model: PyTorch or MLX model

    Returns:
        List of (name, parameter) tuples
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        return list(model.named_parameters())
    else:  # mlx
        from mlx.utils import tree_flatten

        params = model.parameters()
        flattened = tree_flatten(params)
        return [(name, param) for name, param in flattened]


def get_param_count(model: FrameworkModel) -> int:
    """Get total parameter count for a model.

    Args:
        model: PyTorch or MLX model

    Returns:
        Total number of parameters
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        return sum(int(p.numel()) for p in model.parameters())
    else:  # mlx
        from mlx.utils import tree_flatten

        params = tree_flatten(model.parameters())
        return sum(int(p.size) for _, p in params)


def get_trainable_param_count(model: FrameworkModel) -> int:
    """Get trainable parameter count for a model.

    Args:
        model: PyTorch or MLX model

    Returns:
        Number of trainable parameters
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    else:  # mlx
        # MLX doesn't have explicit requires_grad - all params are trainable by default
        return get_param_count(model)


def model_forward(model: FrameworkModel, input_ids: FrameworkTensor, target_ids: FrameworkTensor | None = None) -> FrameworkTensor:
    """Run forward pass on a model.

    Args:
        model: PyTorch or MLX model
        input_ids: Input token IDs
        target_ids: Target token IDs (optional, for loss computation)

    Returns:
        Output tensor (loss if target_ids provided, else logits)
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        if target_ids is not None:
            return model(input_ids, target_ids)
        else:
            return model(input_ids)
    else:  # mlx
        if target_ids is not None:
            return model.loss(input_ids, target_ids)
        else:
            return model(input_ids)


def model_eval(model: FrameworkModel) -> None:
    """Set model to evaluation mode.

    Args:
        model: PyTorch or MLX model
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        model.eval()
    # MLX doesn't have explicit eval mode - training state is handled differently


def model_train(model: FrameworkModel) -> None:
    """Set model to training mode.

    Args:
        model: PyTorch or MLX model
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        model.train()
    # MLX training is controlled via gradient computation context


def compute_loss(logits: FrameworkTensor, targets: FrameworkTensor, vocab_size: int | None = None) -> FrameworkTensor:
    """Compute cross-entropy loss.

    Args:
        logits: Model output logits
        targets: Target token IDs
        vocab_size: Vocabulary size (optional, for MLX shape inference)

    Returns:
        Loss tensor
    """
    # Try to detect framework from tensor type
    if _torch_available and isinstance(logits, torch.Tensor):
        return nn_torch.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
    elif _mlx_available and hasattr(logits, 'dtype') and 'mlx' in str(type(logits)):
        # MLX cross entropy computation
        import mlx.core as mx

        # Flatten logits and targets
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)

        # Compute log softmax
        log_probs = mx.log_softmax(logits_flat.astype(mx.float32), axis=-1)

        # Gather log probs for targets
        batch_size = log_probs.shape[0]
        indices = targets_flat.astype(mx.int32)

        # Compute negative log likelihood
        nll = -log_probs[mx.arange(batch_size), indices]
        return nll.mean()
    else:
        raise ValueError(f"Cannot determine framework for loss computation: {type(logits)}")


def check_finite(tensor: FrameworkTensor) -> bool:
    """Check if all values in a tensor are finite (no NaN or Inf).

    Args:
        tensor: PyTorch or MLX tensor/array

    Returns:
        True if all values are finite, False otherwise
    """
    if _torch_available and isinstance(tensor, torch.Tensor):
        return bool(torch.isfinite(tensor).all())
    elif _mlx_available and hasattr(tensor, 'dtype') and 'mlx' in str(type(tensor)):
        import mlx.core as mx

        # Check for NaN (not equal to itself) and Inf
        finite_mask = mx.isfinite(tensor)
        return bool(finite_mask.all().item())
    else:
        raise ValueError(f"Cannot check finiteness for tensor type: {type(tensor)}")


def has_nan(tensor: FrameworkTensor) -> bool:
    """Check if tensor contains NaN values.

    Args:
        tensor: PyTorch or MLX tensor/array

    Returns:
        True if tensor contains NaN, False otherwise
    """
    if _torch_available and isinstance(tensor, torch.Tensor):
        return bool(torch.isnan(tensor).any())
    elif _mlx_available and hasattr(tensor, 'dtype') and 'mlx' in str(type(tensor)):
        import mlx.core as mx

        return bool(mx.isnan(tensor).any().item())
    else:
        raise ValueError(f"Cannot check NaN for tensor type: {type(tensor)}")


def has_inf(tensor: FrameworkTensor) -> bool:
    """Check if tensor contains Inf values.

    Args:
        tensor: PyTorch or MLX tensor/array

    Returns:
        True if tensor contains Inf, False otherwise
    """
    if _torch_available and isinstance(tensor, torch.Tensor):
        return bool(torch.isinf(tensor).any())
    elif _mlx_available and hasattr(tensor, 'dtype') and 'mlx' in str(type(tensor)):
        import mlx.core as mx

        return bool(mx.isinf(tensor).any().item())
    else:
        raise ValueError(f"Cannot check Inf for tensor type: {type(tensor)}")


def tensor_mean(tensor: FrameworkTensor) -> float:
    """Get mean value of a tensor.

    Args:
        tensor: PyTorch or MLX tensor/array

    Returns:
        Mean value as float
    """
    if _torch_available and isinstance(tensor, torch.Tensor):
        return float(tensor.mean().item())
    elif _mlx_available and hasattr(tensor, 'dtype') and 'mlx' in str(type(tensor)):
        import mlx.core as mx

        return float(mx.mean(tensor).item())
    else:
        raise ValueError(f"Cannot compute mean for tensor type: {type(tensor)}")


def tensor_std(tensor: FrameworkTensor) -> float:
    """Get standard deviation of a tensor.

    Args:
        tensor: PyTorch or MLX tensor/array

    Returns:
        Standard deviation as float
    """
    if _torch_available and isinstance(tensor, torch.Tensor):
        return float(tensor.std().item())
    elif _mlx_available and hasattr(tensor, 'dtype') and 'mlx' in str(type(tensor)):
        import mlx.core as mx

        return float(mx.std(tensor).item())
    else:
        raise ValueError(f"Cannot compute std for tensor type: {type(tensor)}")


def create_random_input(vocab_size: int, batch_size: int, seq_len: int, framework: str = "pytorch") -> FrameworkTensor:
    """Create random input tensor for testing.

    Args:
        vocab_size: Vocabulary size
        batch_size: Batch size
        seq_len: Sequence length
        framework: "pytorch" or "mlx"

    Returns:
        Random input tensor of shape (batch_size, seq_len)
    """
    if framework == "pytorch":
        if torch is None:
            raise RuntimeError("PyTorch not available")
        return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    else:  # mlx
        if mx is None:
            raise RuntimeError("MLX not available")
        return mx.random.randint(0, vocab_size, (batch_size, seq_len))


def create_rolled_targets(input_ids: FrameworkTensor) -> FrameworkTensor:
    """Create target IDs by rolling input IDs (for next-token prediction).

    Args:
        input_ids: Input token IDs tensor

    Returns:
        Rolled target tensor
    """
    if _torch_available and isinstance(input_ids, torch.Tensor):
        return torch.roll(input_ids, shifts=-1, dims=1)
    elif _mlx_available and hasattr(input_ids, 'dtype') and 'mlx' in str(type(input_ids)):
        import mlx.core as mx

        # Roll the array: take elements from position 1 to end, then append first element
        rolled = mx.concatenate([input_ids[:, 1:], input_ids[:, :1]], axis=1)
        return rolled
    else:
        raise ValueError(f"Cannot roll tensor type: {type(input_ids)}")


def requires_grad_context(enabled: bool = False):
    """Get a context manager for gradient computation.

    Args:
        enabled: Whether gradients should be enabled

    Returns:
        Context manager (torch.no_grad() or nullcontext)
    """
    if _torch_available:
        if enabled:
            return torch.enable_grad()
        else:
            return torch.no_grad()
    else:
        from contextlib import nullcontext

        return nullcontext()


def backward_pass(loss: FrameworkTensor) -> None:
    """Run backward pass on a loss tensor.

    Args:
        loss: Loss tensor
    """
    if _torch_available and isinstance(loss, torch.Tensor):
        loss.backward()
    elif _mlx_available and hasattr(loss, 'dtype') and 'mlx' in str(type(loss)):
        # MLX uses value_and_grad pattern, backward is handled differently
        # This is a placeholder - actual MLX backward needs model state management
        raise NotImplementedError("MLX backward pass requires value_and_grad compilation. Use model-specific training loops.")
    else:
        raise ValueError(f"Cannot run backward for tensor type: {type(loss)}")


def get_gradients(model: FrameworkModel) -> dict[str, Any]:
    """Get gradients from model parameters.

    Args:
        model: PyTorch or MLX model

    Returns:
        Dict mapping parameter names to their gradients
    """
    framework = detect_framework(model)

    if framework == "pytorch":
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
        return grads
    else:  # mlx
        # MLX stores gradients differently - need to use value_and_grad
        # Return empty dict as placeholder
        return {}


def get_model_info(model: FrameworkModel) -> dict[str, Any]:
    """Get general information about a model.

    Args:
        model: PyTorch or MLX model

    Returns:
        Dict with model metadata (num_layers, embedding_dim, vocab_size, etc.)
    """
    framework = detect_framework(model)
    info: dict[str, Any] = {"framework": framework}

    if framework == "pytorch":
        # Extract common attributes
        if hasattr(model, "blocks"):
            info["num_layers"] = len(model.blocks)
        if hasattr(model, "tok_emb"):
            tok_emb = model.tok_emb
            if hasattr(tok_emb, "num_embeddings"):
                info["vocab_size"] = int(tok_emb.num_embeddings)
            if hasattr(tok_emb, "embedding_dim"):
                info["embedding_dim"] = int(tok_emb.embedding_dim)
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            block = model.blocks[0]
            if hasattr(block, "attn"):
                attn = block.attn
                if hasattr(attn, "num_heads"):
                    info["num_heads"] = int(attn.num_heads)
                if hasattr(attn, "num_kv_heads"):
                    info["num_kv_heads"] = int(attn.num_kv_heads)
    else:  # mlx
        # Extract MLX model attributes
        if hasattr(model, "blocks"):
            info["num_layers"] = len(model.blocks)
        if hasattr(model, "tok_emb"):
            tok_emb = model.tok_emb
            if hasattr(tok_emb, "weight"):
                weight_shape = tok_emb.weight.shape
                info["vocab_size"] = int(weight_shape[0])
                info["embedding_dim"] = int(weight_shape[1])
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            block = model.blocks[0]
            if hasattr(block, "attn"):
                attn = block.attn
                if hasattr(attn, "num_heads"):
                    info["num_heads"] = int(attn.num_heads)
                if hasattr(attn, "num_kv_heads"):
                    info["num_kv_heads"] = int(attn.num_kv_heads)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# T4 Signal Propagation - Specific Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gradient_norms_pytorch(
    model: Any,
    input_ids: Any,
    target_ids: Any,
    vocab_size: int,
) -> dict[str, float]:
    """Compute per-layer gradient norms for PyTorch model.

    Args:
        model: PyTorch model
        input_ids: Input token IDs (torch.Tensor)
        target_ids: Target token IDs (torch.Tensor)
        vocab_size: Vocabulary size for loss computation

    Returns:
        Dict mapping layer names to gradient norms
    """
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch not available")

    model.zero_grad()

    # Forward pass - try different calling conventions
    try:
        # Try calling with return_logits (some models support this)
        logits = model(input_ids, target_ids, return_logits=True)
        loss = F_torch.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )
    except TypeError:
        # Model doesn't support return_logits
        # Try standard call - some models return loss directly
        output = model(input_ids, target_ids)
        if output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1):
            # Output is a scalar loss
            loss = output
        else:
            # Output is logits
            loss = F_torch.cross_entropy(
                output.view(-1, vocab_size),
                target_ids.view(-1)
            )

    # Backward pass
    loss.backward()

    # Collect per-layer gradient norms
    layer_grad_norms: dict[str, float] = {}

    for idx, block in enumerate(model.blocks):
        # Attention gradients
        attn_grad_norms = []
        for name, param in block.attn.named_parameters():
            if param.grad is not None:
                attn_grad_norms.append(float(param.grad.norm().item()))
        if attn_grad_norms:
            layer_grad_norms[f"layer_{idx}_attn"] = sum(attn_grad_norms)

        # MLP gradients
        mlp_module = getattr(block, "ffn", getattr(block, "mlp", None))
        if mlp_module:
            mlp_grad_norms = []
            for name, param in mlp_module.named_parameters():
                if param.grad is not None:
                    mlp_grad_norms.append(float(param.grad.norm().item()))
            if mlp_grad_norms:
                layer_grad_norms[f"layer_{idx}_mlp"] = sum(mlp_grad_norms)

    return layer_grad_norms


def compute_gradient_norms_mlx(
    model: Any,
    input_ids: Any,
    target_ids: Any,
    vocab_size: int,
) -> dict[str, float]:
    """Compute per-layer gradient norms for MLX model.

    Args:
        model: MLX model
        input_ids: Input token IDs (mx.array)
        target_ids: Target token IDs (mx.array)
        vocab_size: Vocabulary size

    Returns:
        Dict mapping layer names to gradient norms
    """
    if not MLX_AVAILABLE or mx is None or tree_flatten is None:
        raise RuntimeError("MLX not available")

    # Define loss function for value_and_grad
    def loss_fn(model: Any, x: Any, y: Any) -> Any:
        return model.loss(x, y)

    # Compute loss and gradients
    loss, grads = nn_mlx.value_and_grad(model, loss_fn)(model, input_ids, target_ids)

    # Flatten gradient tree
    flat_grads = dict(tree_flatten(grads))

    # Collect per-layer gradient norms
    layer_grad_norms: dict[str, float] = {}

    for idx, block in enumerate(model.blocks):
        # Attention gradients
        attn_grad_sq = 0.0
        for name, grad in flat_grads.items():
            if f"blocks.{idx}.attn" in name:
                grad_sq = float(mx.sum(grad * grad).item())
                attn_grad_sq += grad_sq
        if attn_grad_sq > 0:
            layer_grad_norms[f"layer_{idx}_attn"] = math.sqrt(attn_grad_sq)

        # MLP gradients
        mlp_grad_sq = 0.0
        mlp_name = f"blocks.{idx}.ffn" if hasattr(block, "ffn") else f"blocks.{idx}.mlp"
        for name, grad in flat_grads.items():
            if mlp_name in name:
                grad_sq = float(mx.sum(grad * grad).item())
                mlp_grad_sq += grad_sq
        if mlp_grad_sq > 0:
            layer_grad_norms[f"layer_{idx}_mlp"] = math.sqrt(mlp_grad_sq)

    return layer_grad_norms


def compute_activation_norm_pytorch(tensor: Any) -> float:
    """Compute activation norm for PyTorch tensor."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    return float(tensor.norm().item())


def compute_activation_norm_mlx(tensor: Any) -> float:
    """Compute activation norm for MLX array."""
    if not MLX_AVAILABLE or mx is None:
        raise RuntimeError("MLX not available")
    # Compute L2 norm: sqrt(sum(x^2))
    sq_sum = float(mx.sum(tensor * tensor).item())
    return math.sqrt(sq_sum)


def compute_activation_stats_pytorch(tensor: Any) -> dict[str, float]:
    """Compute activation statistics for PyTorch tensor.

    Returns dict with: mean, std, snr, dead_ratio
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    mean_act = float(tensor.mean().item())
    std_act = float(tensor.std().item())
    snr = abs(mean_act) / (std_act + 1e-12)

    # Dead neuron detection (outputs < 0.01)
    dead_mask = tensor.abs() < 0.01
    dead_ratio = float(dead_mask.float().mean().item())

    return {
        "mean": mean_act,
        "std": std_act,
        "snr": snr,
        "dead_ratio": dead_ratio,
    }


def compute_activation_stats_mlx(tensor: Any) -> dict[str, float]:
    """Compute activation statistics for MLX array.

    Returns dict with: mean, std, snr, dead_ratio
    """
    if not MLX_AVAILABLE or mx is None:
        raise RuntimeError("MLX not available")

    mean_act = float(mx.mean(tensor).item())

    # Compute std: sqrt(E[x^2] - E[x]^2)
    sq_mean = float(mx.mean(tensor * tensor).item())
    variance = sq_mean - mean_act ** 2
    std_act = math.sqrt(max(0.0, variance))

    snr = abs(mean_act) / (std_act + 1e-12)

    # Dead neuron detection (outputs < 0.01)
    abs_tensor = mx.abs(tensor)
    dead_mask = abs_tensor < 0.01
    dead_count = float(mx.sum(dead_mask).item())
    total_count = tensor.size
    dead_ratio = dead_count / total_count if total_count > 0 else 0.0

    return {
        "mean": mean_act,
        "std": std_act,
        "snr": snr,
        "dead_ratio": dead_ratio,
    }


def compute_output_entropy_pytorch(logits: Any) -> float:
    """Compute output entropy from PyTorch logits.

    Args:
        logits: Raw logits tensor

    Returns:
        Mean entropy across batch
    """
    if not TORCH_AVAILABLE or torch is None or F_torch is None:
        raise RuntimeError("PyTorch not available")

    vocab_size = logits.shape[-1]
    probs = F_torch.softmax(logits.detach().view(-1, vocab_size), dim=-1)
    entropy = float(-(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item())
    return entropy


def compute_output_entropy_mlx(logits: Any) -> float:
    """Compute output entropy from MLX logits.

    Args:
        logits: Raw logits array

    Returns:
        Mean entropy across batch
    """
    if not MLX_AVAILABLE or mx is None:
        raise RuntimeError("MLX not available")

    # Compute softmax probabilities
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)

    # Entropy: -sum(p * log(p))
    log_probs = mx.log(probs + 1e-12)
    entropy_per_token = -mx.sum(probs * log_probs, axis=-1)
    mean_entropy = float(mx.mean(entropy_per_token).item())

    return mean_entropy


def create_activation_hook_pytorch(
    activation_norms: dict[str, float],
    activation_stats: dict[str, dict[str, float]],
    layer_name: str,
) -> Callable:
    """Create a PyTorch forward hook for activation analysis.

    Args:
        activation_norms: Dict to store norm values
        activation_stats: Dict to store detailed stats
        layer_name: Name for this layer

    Returns:
        Hook function for register_forward_hook
    """
    def hook(module: Any, input: Any, output: Any) -> None:
        tensor = None
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            tensor = output[0]

        if tensor is not None:
            activation_norms[layer_name] = compute_activation_norm_pytorch(tensor)
            stats = compute_activation_stats_pytorch(tensor)
            stats["tensor_shape"] = list(tensor.shape)
            activation_stats[layer_name] = stats

    return hook


def make_random_input_pytorch(vocab_size: int, seq_len: int, seed: int = 42) -> tuple:
    """Create random input tensors for PyTorch.

    Returns:
        Tuple of (input_ids, target_ids)
    """
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch not available")

    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    return input_ids, target_ids


def make_random_input_mlx(vocab_size: int, seq_len: int, seed: int = 42) -> tuple:
    """Create random input arrays for MLX.

    Returns:
        Tuple of (input_ids, target_ids)
    """
    if not MLX_AVAILABLE or mx is None:
        raise RuntimeError("MLX not available")

    mx.random.seed(seed)
    input_ids = mx.random.randint(0, vocab_size, (1, seq_len))
    target_ids = mx.roll(input_ids, shift=-1, axis=1)
    return input_ids, target_ids
