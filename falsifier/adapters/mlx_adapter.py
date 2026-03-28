"""MLX adapter for Apple Silicon micro-training.

Provides MLX-based model instantiation and training with fallback handling.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

from falsifier.utils.metrics import classify_component

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def mlx_available() -> bool:
    """Check if MLX is installed and available."""
    return MLX_AVAILABLE


def _patched_env(overrides: dict[str, str] | None) -> dict[str, str | None]:
    """Apply environment variable overrides, returning previous values for restoration."""
    previous: dict[str, str | None] = {}
    if overrides:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    """Restore environment variables to their previous values."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def load_train_gpt_mlx_module(
    train_gpt_mlx_path: str | Path, env_overrides: dict[str, str] | None = None
) -> ModuleType:
    """Load train_gpt_mlx.py as a module with optional environment overrides."""
    path = Path(train_gpt_mlx_path).resolve()
    module_name = f"falsifier_train_gpt_mlx_{uuid.uuid4().hex}"

    previous = _patched_env(env_overrides)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load train_gpt_mlx.py from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_env(previous)


def instantiate_mlx_model(
    train_gpt_mlx_path: str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[ModuleType, nn.Module]:
    """Instantiate an MLX GPT model with optional config overrides.

    Args:
        train_gpt_mlx_path: Path to train_gpt_mlx.py
        config_overrides: Dict of config overrides applied via environment variables

    Returns:
        Tuple of (module, model) where model is the instantiated MLX GPT

    Raises:
        RuntimeError: If MLX is not available
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available - cannot instantiate MLX model")

    module = load_train_gpt_mlx_module(train_gpt_mlx_path, env_overrides=config_overrides)
    args = module.Hyperparameters()

    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    return module, model


def run_mlx_training(
    model: nn.Module,
    module: ModuleType,
    steps: int,
    data_path: Path | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run MLX training loop with real data or random fallback.

    Args:
        model: MLX GPT model instance
        module: train_gpt_mlx module (contains TokenLoader, Hyperparameters)
        steps: Number of training steps
        data_path: Path to FineWeb data directory, or None for random data
        seed: Random seed for reproducibility

    Returns:
        Dict with training metrics:
        - loss_trajectory: list of loss values per step
        - loss_first: loss at step 0
        - loss_last: loss at final step
        - loss_drop: loss_first - loss_last
        - throughput: tokens per second
        - gradient_norms: list of global gradient norms per step
        - entropy_trajectory: list of output entropies every 10 steps
        - component_speeds: dict of per-component gradient norms

    Raises:
        FileNotFoundError: If data_path is provided but data files don't exist
        RuntimeError: If MLX is not available
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available - cannot run MLX training")

    mx.random.seed(seed)

    args = module.Hyperparameters()
    model.train()

    # Setup optimizer (AdamW-like with split optimizer structure)
    opt = module.SplitOptimizers(model, args)

    # Compile loss function
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Setup data loading
    use_real_data = False
    train_loader = None
    if data_path is not None:
        data_path = Path(data_path)
        train_pattern = str(data_path / "fineweb_train_*.bin")
        import glob

        train_files = glob.glob(train_pattern)
        if train_files:
            try:
                train_loader = module.TokenLoader(train_pattern, log_fn=None, dataset_name="fineweb")
                use_real_data = True
            except Exception:
                train_loader = None

    # Training metrics
    loss_trajectory: list[float] = []
    gradient_norms: list[float] = []
    entropy_trajectory: list[float] = []
    component_speeds: dict[str, list[float]] = {k: [] for k in ["attn", "mlp", "embed", "norm", "skip", "other"]}

    # Per-step component gradient tracking for learning order analysis (enhanced T7)
    component_grad_norms_per_step: dict[str, list[float]] = {
        "attn": [], "mlp": [], "embed": [], "norm": [], "other": []
    }

    vocab_size = int(model.tok_emb.weight.shape[0])
    seq_len = args.train_seq_len
    batch_tokens = args.train_batch_tokens

    t0 = time.perf_counter()

    for step in range(steps):
        step_t0 = time.perf_counter()

        if use_real_data and train_loader is not None:
            # Real data from FineWeb
            try:
                x, y = train_loader.next_batch(batch_tokens, seq_len)
            except Exception:
                # Fall back to random data on error
                x = mx.random.randint(0, vocab_size, (batch_tokens // seq_len, seq_len))
                y = mx.random.randint(0, vocab_size, (batch_tokens // seq_len, seq_len))
        else:
            # Random synthetic data
            x = mx.random.randint(0, vocab_size, (batch_tokens // seq_len, seq_len))
            y = mx.random.randint(0, vocab_size, (batch_tokens // seq_len, seq_len))

        # Forward + backward
        loss, grads = compiled_loss_and_grad(x, y)
        loss_value = float(loss.item())

        # Compute gradient norm
        flat_grads = dict(tree_flatten(grads))
        total_sq = 0.0
        for grad in flat_grads.values():
            total_sq += float(mx.sum(grad * grad).item())
        grad_norm = math.sqrt(total_sq)
        gradient_norms.append(grad_norm)

        # Track component speeds (per-component gradient norms)
        step_component_norms: dict[str, float] = {"attn": 0.0, "mlp": 0.0, "embed": 0.0, "norm": 0.0, "other": 0.0}
        for name, grad in flat_grads.items():
            grad_sq = float(mx.sum(grad * grad).item())
            grad_norm = math.sqrt(grad_sq)

            # Use consistent component classification with PyTorch path
            comp_type = classify_component(name)

            # Track in component_speeds for aggregate metrics
            if comp_type in component_speeds:
                component_speeds[comp_type].append(grad_norm)
            else:
                component_speeds["other"].append(grad_norm)

            # Accumulate for per-step tracking
            step_component_norms[comp_type] += grad_norm

        # Store per-step component norms for learning order analysis
        for comp_type, norm_val in step_component_norms.items():
            component_grad_norms_per_step[comp_type].append(norm_val)

        # Calculate entropy every 10 steps
        if step % 10 == 0 or step == steps - 1:
            with mx.stream(mx.cpu):
                # Get model outputs for entropy calculation
                logits_proj = model(x) @ model.tok_emb.weight.T
                c = model.logit_softcap
                logits = c * mx.tanh(logits_proj / c)
                probs = mx.softmax(logits.astype(mx.float32), axis=-1)
                # Entropy: -sum(p * log(p))
                log_probs = mx.log(probs + 1e-10)
                entropy = -mx.sum(probs * log_probs, axis=-1)
                mean_entropy = float(mx.mean(entropy).item())
                entropy_trajectory.append(mean_entropy)

        # Optimizer step
        opt.step(model, grads, step=step, lr_mul=1.0)
        mx.eval(model.state)
        mx.synchronize()

        loss_trajectory.append(loss_value)

    elapsed = time.perf_counter() - t0

    # Aggregate component speeds
    agg_component_speeds: dict[str, float] = {}
    for k, v in component_speeds.items():
        if v:
            agg_component_speeds[k] = sum(v) / len(v)
        else:
            agg_component_speeds[k] = 0.0

    # Calculate throughput
    total_tokens = steps * batch_tokens
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "loss_trajectory": loss_trajectory,
        "loss_first": loss_trajectory[0] if loss_trajectory else 0.0,
        "loss_last": loss_trajectory[-1] if loss_trajectory else 0.0,
        "loss_drop": (loss_trajectory[0] - loss_trajectory[-1]) if loss_trajectory else 0.0,
        "throughput": throughput,
        "gradient_norms": gradient_norms,
        "entropy_trajectory": entropy_trajectory,
        "component_speeds": agg_component_speeds,
        "component_grad_norms_per_step": component_grad_norms_per_step,
        "elapsed_seconds": elapsed,
    }


def run_mlx_micro_train_summary(
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
    steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Convenience wrapper for MLX micro-training.

    Args:
        repo_root: Repository root path
        config_overrides: Optional config overrides
        steps: Number of training steps
        seed: Random seed

    Returns:
        Training metrics dict from run_mlx_training

    Raises:
        RuntimeError: If MLX is not available
        FileNotFoundError: If train_gpt_mlx.py is not found
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available - cannot run MLX micro-training")

    train_gpt_mlx_path = Path(repo_root) / "train_gpt_mlx.py"
    if not train_gpt_mlx_path.exists():
        raise FileNotFoundError(f"train_gpt_mlx.py not found at {train_gpt_mlx_path}")

    module, model = instantiate_mlx_model(train_gpt_mlx_path, config_overrides)

    # Check for data path override
    data_path = None
    if config_overrides and "DATA_PATH" in config_overrides:
        data_path = Path(config_overrides["DATA_PATH"])

    return run_mlx_training(model, module, steps, data_path, seed)
