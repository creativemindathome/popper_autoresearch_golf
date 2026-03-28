"""Execute training runs for Stage 2.

Refactored to handle diverse parameter-golf submission patterns:
- Unified model interface via framework_adapter
- Proper temp file handling for source code
- Both PyTorch and MLX training paths
- Automatic model signature detection
- Progress logging with per-step divergence detection
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..adapters.parameter_golf import instantiate_minimal_model
from ..utils.model_utils import setup_optimizer_from_source
from ..utils.framework_adapter import (
    detect_framework,
    model_forward,
    model_train,
    create_random_input,
    create_rolled_targets,
    compute_output_entropy_pytorch,
    get_model_info,
    TORCH_AVAILABLE,
)
from ..utils.metrics import classify_component


@dataclass
class RunResult:
    """Result of a training run."""

    name: str
    losses: list[float]
    grad_norms: list[float]
    entropies: list[float]
    component_metrics: dict[int, dict[str, float]]
    tokens_per_second: float
    diverged: bool = False
    diverged_step: int | None = None
    wall_seconds: float = 0.0


def _detect_framework_from_source(source: str) -> str:
    """Detect if source code is MLX or PyTorch based."""
    if "import mlx" in source or "from mlx" in source:
        return "mlx"
    return "pytorch"


def execute_training_run(
    spec: Any,
    inp: Any,
) -> RunResult:
    """Execute a training run.

    Uses MLX if available for speed, falls back to PyTorch minimal-env.
    """
    # Detect framework from source
    source = _get_source_from_spec(spec, inp)
    framework = _detect_framework_from_source(source)

    # Try MLX first if framework is MLX and MLX is available
    if framework == "mlx":
        try:
            from ..adapters.mlx_adapter import mlx_available

            if mlx_available():
                return _execute_mlx_run(spec, inp)
        except Exception:
            pass

    # Fall back to PyTorch minimal-env
    return _execute_pytorch_run(spec, inp)


def _get_source_from_spec(spec: Any, inp: Any) -> str:
    """Extract source code based on spec.source type."""
    if spec.source == "theory":
        return inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else ""
    elif spec.source.startswith("ablation_"):
        from .ablation import build_ablation_source
        return build_ablation_source(
            inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else "",
            inp.sota_train_gpt if hasattr(inp, "sota_train_gpt") else "",
            inp.config_delta if hasattr(inp, "config_delta") else {},
            spec.source.replace("ablation_", ""),
        )
    elif spec.source == "baseline":
        return inp.sota_train_gpt if hasattr(inp, "sota_train_gpt") else ""
    else:
        return inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else ""


def _ensure_source_path(source: str) -> tuple[str, bool]:
    """Ensure source is a valid file path, creating temp file if needed.

    Returns:
        Tuple of (source_path, is_temp_file)
    """
    import tempfile

    if not source:
        raise ValueError("Empty source code provided")

    # Check if source is already a valid path
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        return str(source_path), False

    # Source is actual code content - write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        return f.name, True


def _detect_model_signature(model: Any, framework: str) -> str:
    """Detect if model returns logits or loss from forward pass."""
    if framework != "pytorch":
        return "unknown"

    import inspect

    # Check forward method signature
    forward_sig = inspect.signature(model.forward)
    params = list(forward_sig.parameters.keys())

    # If forward accepts 'targets' or 'labels', it likely returns loss
    if any(p in params for p in ["targets", "labels", "target_ids"]):
        return "loss_returning"

    # Try a test forward pass to detect output type
    try:
        test_input = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            output = model(test_input)

        # Check output dimensions
        if output.dim() == 0 or (output.dim() == 1 and output.shape[0] == 1):
            return "loss_returning"
        elif output.dim() >= 2:
            return "logits_returning"
    except Exception:
        pass

    return "unknown"


def _compute_loss_safe(
    model: Any,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
    signature: str,
) -> torch.Tensor:
    """Compute loss handling both logits and loss-returning models."""
    if signature == "loss_returning":
        # Model returns loss directly
        try:
            loss = model(input_ids, target_ids)
            if loss.dim() > 0:
                return loss.mean()
            return loss
        except TypeError:
            # Try with keyword args
            try:
                loss = model(input_ids, targets=target_ids)
                if loss.dim() > 0:
                    return loss.mean()
                return loss
            except TypeError:
                pass

    # Default: model returns logits
    output = model(input_ids)
    return F.cross_entropy(
        output.reshape(-1, output.size(-1)),
        target_ids.reshape(-1),
    )


def _log_progress(
    step: int,
    total_steps: int,
    loss: float,
    grad_norm: float,
    start_time: float,
    diverged: bool = False,
) -> None:
    """Log training progress."""
    elapsed = time.time() - start_time
    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
    eta = (total_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0.0

    status = "DIVERGED" if diverged else f"{step + 1}/{total_steps}"
    print(
        f"[RunExecutor] Step {status} | "
        f"Loss: {loss:.4f} | Grad: {grad_norm:.4f} | "
        f"Speed: {steps_per_sec:.1f} steps/s | ETA: {eta:.1f}s"
    )


def _execute_pytorch_run(spec: Any, inp: Any) -> RunResult:
    """Execute training run with PyTorch minimal-env using unified interface."""
    import tempfile
    import os

    torch.manual_seed(spec.seed)

    # Get source code
    source = _get_source_from_spec(spec, inp)

    # Ensure source is a valid file path
    source_path, is_temp = _ensure_source_path(source)

    try:
        # Instantiate model via parameter_golf adapter
        # Returns (module, model) tuple
        module, model = instantiate_minimal_model(source_path)

        # Set model to training mode using unified interface
        model_train(model)

        # Detect framework and model signature
        framework = detect_framework(model)
        signature = _detect_model_signature(model, framework)
        print(f"[RunExecutor] Model signature: {signature} (framework: {framework})")

        # Get model info for vocab size
        model_info = get_model_info(model)
        vocab_size = model_info.get("vocab_size", 64)

        # Setup optimizer (handles both single optimizer and list of optimizers)
        optimizer_result = setup_optimizer_from_source(model, source_path, framework=framework)
        # Normalize to list for consistent handling
        if isinstance(optimizer_result, list):
            optimizers = optimizer_result
        else:
            optimizers = [optimizer_result]

        # Training loop
        losses, grad_norms, entropies, comp_metrics = [], [], [], {}
        t0 = time.time()

        # Log start
        print(f"[RunExecutor] Starting PyTorch training: {spec.name} ({spec.steps} steps)")

        for step in range(spec.steps):
            # Create random input using unified interface
            batch = create_random_input(vocab_size, 1, 8, framework="pytorch")
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward pass with signature-aware loss computation
            loss = _compute_loss_safe(model, input_ids, target_ids, vocab_size, signature)

            # Backward pass - zero gradients for all optimizers
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()

            # Track gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norms.append(grad_norm)

            # Store loss
            loss_val = loss.item()
            losses.append(loss_val)

            # Check for divergence at every step
            if loss_val > 100 or math.isnan(loss_val) or math.isinf(loss_val):
                print(f"[RunExecutor] DIVERGENCE DETECTED at step {step}: loss={loss_val}")
                return RunResult(
                    name=spec.name,
                    losses=losses,
                    grad_norms=grad_norms,
                    entropies=entropies,
                    component_metrics=comp_metrics,
                    tokens_per_second=0.0,
                    diverged=True,
                    diverged_step=step,
                    wall_seconds=time.time() - t0,
                )

            # Entropy calculation every 10 steps
            if step % 10 == 0 or step == spec.steps - 1:
                with torch.no_grad():
                    # Get logits for entropy
                    try:
                        output = model(input_ids)
                        if output.dim() > 1:
                            probs = F.softmax(output[:, -1, :], dim=-1)
                            ent = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
                            entropies.append(ent)
                    except Exception:
                        pass

            # Dense logging for component metrics
            if spec.dense_logging and step % 10 == 0:
                cm: dict[str, list[float]] = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        comp = classify_component(name)
                        cm.setdefault(comp, []).append(p.grad.float().norm().item())
                comp_metrics[step] = {k: sum(v) / len(v) for k, v in cm.items()}

            # Progress logging every 20 steps
            if step % 20 == 0 or step == spec.steps - 1:
                _log_progress(step, spec.steps, loss_val, grad_norm, t0)

            # Gradient clipping and optimizer step for all optimizers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()

        elapsed = time.time() - t0
        tokens = spec.steps * 8  # Approximate
        tps = tokens / elapsed if elapsed > 0 else 0.0

        print(f"[RunExecutor] Completed {spec.name}: {spec.steps} steps in {elapsed:.1f}s ({tps:.1f} tok/s)")

        return RunResult(
            name=spec.name,
            losses=losses,
            grad_norms=grad_norms,
            entropies=entropies,
            component_metrics=comp_metrics,
            tokens_per_second=tps,
            diverged=False,
            wall_seconds=elapsed,
        )

    finally:
        # Clean up temp file if created
        if is_temp and os.path.exists(source_path):
            try:
                os.unlink(source_path)
            except Exception:
                pass


def _execute_mlx_run(spec: Any, inp: Any) -> RunResult:
    """Execute training run with MLX."""
    from ..adapters.mlx_adapter import run_mlx_training, instantiate_mlx_model

    # Get source code
    source = _get_source_from_spec(spec, inp)

    # Ensure source is a valid file path
    source_path, is_temp = _ensure_source_path(source)

    try:
        # Build model with proper source
        module, model = instantiate_mlx_model(source_path)

        # Get data path
        data_path = Path(inp.val_data_path) if hasattr(inp, "val_data_path") and inp.val_data_path else None

        print(f"[RunExecutor] Starting MLX training: {spec.name} ({spec.steps} steps)")

        # Run training
        result = run_mlx_training(
            model=model,
            module=module,
            steps=spec.steps,
            data_path=data_path,
            seed=spec.seed,
        )

        print(f"[RunExecutor] Completed {spec.name}: {result.get('throughput', 0):.1f} tok/s")

        # Convert to RunResult format
        return RunResult(
            name=spec.name,
            losses=result.get("loss_trajectory", []),
            grad_norms=result.get("gradient_norms", []),
            entropies=result.get("entropy_trajectory", []),
            component_metrics={},  # MLX adapter doesn't provide per-step component metrics yet
            tokens_per_second=result.get("throughput", 0.0),
            diverged=result.get("loss_first", 0) < result.get("loss_last", 0),
            wall_seconds=result.get("elapsed_seconds", 0.0),
        )

    finally:
        # Clean up temp file if created
        if is_temp:
            import os
            if os.path.exists(source_path):
                try:
                    os.unlink(source_path)
                except Exception:
                    pass
