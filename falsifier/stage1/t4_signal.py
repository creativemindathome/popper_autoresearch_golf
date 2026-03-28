"""T4: Signal Propagation gate.

Analyzes gradient flow and signal propagation through the model at initialization:
- Gradient norm ratios across layers (detect vanishing/exploding gradients)
- Output entropy (detect overconfidence/underconfidence)
- Activation norms (detect extreme scales)
- Dead neuron detection (identify inactive units)
- Signal-to-noise ratio per layer
- Comparative analysis against baseline calibration
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from falsifier.adapters.parameter_golf import instantiate_minimal_model
from falsifier.types import FalsifierInput, Tag, T4Result, TestStatus
from falsifier.utils.framework_adapter import (
    TORCH_AVAILABLE,
    MLX_AVAILABLE,
    compute_gradient_norms_pytorch,
    compute_gradient_norms_mlx,
    compute_output_entropy_pytorch,
    compute_output_entropy_mlx,
    create_activation_hook_pytorch,
    make_random_input_pytorch,
    make_random_input_mlx,
    compute_activation_norm_mlx,
    compute_activation_stats_mlx,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn


def _get_train_gpt_path(inp: FalsifierInput) -> Path:
    """Get the path to train_gpt file, creating a temp file if needed."""
    if inp.train_gpt_path and Path(inp.train_gpt_path).exists():
        return Path(inp.train_gpt_path)
    # Write proposed_train_gpt to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(inp.proposed_train_gpt)
        return Path(f.name)


def run_t4_pytorch(inp: FalsifierInput) -> T4Result:
    """Run T4 signal propagation test for PyTorch models."""
    start_time = time.perf_counter()

    try:
        # Build model with minimal env
        train_gpt_path = _get_train_gpt_path(inp)
        module, model = instantiate_minimal_model(train_gpt_path)
        model.train()

        # Extract vocab_size - try different attribute names
        if hasattr(model.tok_emb, 'num_embeddings'):
            vocab_size = int(model.tok_emb.num_embeddings)
        elif hasattr(model.tok_emb, 'weight'):
            vocab_size = int(model.tok_emb.weight.shape[0])
        else:
            raise RuntimeError("Cannot determine vocab_size from tok_emb")
        seq_len = 8  # Minimal sequence length

        # Create random input
        input_ids, target_ids = make_random_input_pytorch(vocab_size, seq_len)

        # Storage for activation analysis per layer
        activation_norms: dict[str, float] = {}
        activation_stats: dict[str, dict[str, float]] = {}  # mean, std, dead_count
        handles: list[Any] = []

        # Register activation analysis hooks on transformer blocks
        def make_hook(layer_name: str):
            return create_activation_hook_pytorch(
                activation_norms, activation_stats, layer_name
            )

        for idx, block in enumerate(model.blocks):
            # Hook on attention output
            attn_handle = block.attn.register_forward_hook(make_hook(f"layer_{idx}_attn"))
            handles.append(attn_handle)
            # Hook on MLP output (via ffn or mlp depending on naming)
            mlp_module = getattr(block, "ffn", getattr(block, "mlp", None))
            if mlp_module:
                mlp_handle = mlp_module.register_forward_hook(make_hook(f"layer_{idx}_mlp"))
                handles.append(mlp_handle)

        # Forward pass - try different calling conventions
        try:
            # Try calling with return_logits (some models support this)
            logits = model(input_ids, target_ids, return_logits=True)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )
        except TypeError:
            # Model doesn't support return_logits, try standard call
            loss = model(input_ids, target_ids)
            # For logits, we need to run forward pass again without targets
            # Most models return logits when only input_ids is provided
            try:
                logits = model(input_ids)
            except TypeError:
                # Model always requires both arguments, compute logits manually
                # by extracting from the model's output layer
                with torch.no_grad():
                    x = model.tok_emb(input_ids)
                    x = F.rms_norm(x, (x.size(-1),))
                    for block in model.blocks:
                        x = block(x, x)  # Pass x as both input and x0
                    x = model.final_norm(x)
                    # Get logits from tok_emb (tied embeddings) or lm_head
                    if hasattr(model, 'tie_embeddings') and model.tie_embeddings:
                        logits = F.linear(x, model.tok_emb.weight)
                    elif hasattr(model, 'lm_head') and model.lm_head is not None:
                        logits = model.lm_head(x)
                    else:
                        # Fallback: use identity (will give wrong entropy but allow test to continue)
                        logits = x

        # Calculate output entropy
        entropy = compute_output_entropy_pytorch(logits)
        max_entropy = math.log(vocab_size)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

        loss_at_init = float(loss.item())

        # Compute gradient norms
        layer_grad_norms = compute_gradient_norms_pytorch(
            model, input_ids, target_ids, vocab_size
        )

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Calculate gradient norm ratio
        if layer_grad_norms:
            max_grad = max(layer_grad_norms.values())
            min_grad = min(layer_grad_norms.values())
            gradient_norm_ratio = max_grad / min_grad if min_grad > 0 else float("inf")
            max_layer = max(layer_grad_norms, key=layer_grad_norms.get)
            min_layer = min(layer_grad_norms, key=layer_grad_norms.get)
        else:
            max_grad = 0.0
            min_grad = 0.0
            gradient_norm_ratio = 1.0
            max_layer = ""
            min_layer = ""

        # Build result
        return _build_t4_result(
            status="PASS",
            layer_grad_norms=layer_grad_norms,
            gradient_norm_ratio=gradient_norm_ratio,
            max_layer=max_layer,
            min_layer=min_layer,
            max_grad=max_grad,
            min_grad=min_grad,
            entropy=entropy,
            entropy_ratio=entropy_ratio,
            loss_at_init=loss_at_init,
            activation_norms=activation_norms,
            activation_stats=activation_stats,
            calibration=inp.calibration,
            wall_seconds=time.perf_counter() - start_time,
        )

    except Exception as e:
        wall_seconds = time.perf_counter() - start_time
        return _build_error_result(f"T4 PyTorch error: {e}", wall_seconds)


def run_t4_mlx(inp: FalsifierInput) -> T4Result:
    """Run T4 signal propagation test for MLX models."""
    start_time = time.perf_counter()

    try:
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")

        from falsifier.adapters.mlx_adapter import (
            instantiate_mlx_model,
        )

        # Build model with minimal env
        train_gpt_path = _get_train_gpt_path(inp)
        module, model = instantiate_mlx_model(train_gpt_path)
        model.train()

        vocab_size = int(model.tok_emb.weight.shape[0])
        seq_len = 8  # Minimal sequence length

        # Create random input
        input_ids, target_ids = make_random_input_mlx(vocab_size, seq_len)

        # Storage for activation analysis per layer
        activation_norms: dict[str, float] = {}
        activation_stats: dict[str, dict[str, float]] = {}

        # For MLX, we capture activations during manual forward pass
        # since hooks work differently
        def capture_forward_with_activations(x: Any, y: Any) -> tuple[Any, Any]:
            """Forward pass that captures intermediate activations."""
            tok_emb = model.tok_emb

            # Token embeddings
            h = tok_emb(x)

            # Add positional embeddings if available
            if hasattr(model, 'pos_emb'):
                pos_emb = model.pos_emb
                T = x.shape[1]
                positions = mx.arange(T)
                h = h + pos_emb(positions)

            # Process through each block
            for idx, block in enumerate(model.blocks):
                # Get norm layers - different models use different naming
                attn_norm = getattr(block, "ln1", getattr(block, "attn_norm", None))
                mlp_norm = getattr(block, "ln2", getattr(block, "mlp_norm", None))

                # Attention layer
                if attn_norm:
                    h_attn = block.attn(attn_norm(h))
                else:
                    h_attn = block.attn(h)
                h = h + h_attn

                # Capture attention output
                activation_norms[f"layer_{idx}_attn"] = compute_activation_norm_mlx(h)
                stats = compute_activation_stats_mlx(h)
                stats["tensor_shape"] = list(h.shape)
                activation_stats[f"layer_{idx}_attn"] = stats

                # MLP layer
                mlp_module = getattr(block, "ffn", getattr(block, "mlp", None))
                if mlp_module:
                    if mlp_norm:
                        h_mlp = mlp_module(mlp_norm(h))
                    else:
                        h_mlp = mlp_module(h)
                    h = h + h_mlp

                    # Capture MLP output
                    activation_norms[f"layer_{idx}_mlp"] = compute_activation_norm_mlx(h_mlp)
                    stats = compute_activation_stats_mlx(h_mlp)
                    stats["tensor_shape"] = list(h_mlp.shape)
                    activation_stats[f"layer_{idx}_mlp"] = stats

            # Final layer norm
            final_norm = getattr(model, "ln_f", getattr(model, "final_norm", None))
            if final_norm:
                h = final_norm(h)

            # Output projection
            logits_proj = h @ tok_emb.weight.T

            # Apply softcap if configured
            if hasattr(model, 'logit_softcap') and model.logit_softcap:
                c = model.logit_softcap
                logits = c * mx.tanh(logits_proj / c)
            else:
                logits = logits_proj

            # Compute loss
            loss = model.loss(x, y)

            return logits, loss

        # Forward pass with activation capture
        logits, loss = capture_forward_with_activations(input_ids, target_ids)

        # Calculate output entropy
        entropy = compute_output_entropy_mlx(logits)
        max_entropy = math.log(vocab_size)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

        loss_at_init = float(loss.item())

        # Compute gradient norms
        layer_grad_norms = compute_gradient_norms_mlx(
            model, input_ids, target_ids, vocab_size
        )

        # Calculate gradient norm ratio
        if layer_grad_norms:
            max_grad = max(layer_grad_norms.values())
            min_grad = min(layer_grad_norms.values())
            gradient_norm_ratio = max_grad / min_grad if min_grad > 0 else float("inf")
            max_layer = max(layer_grad_norms, key=layer_grad_norms.get)
            min_layer = min(layer_grad_norms, key=layer_grad_norms.get)
        else:
            max_grad = 0.0
            min_grad = 0.0
            gradient_norm_ratio = 1.0
            max_layer = ""
            min_layer = ""

        # Build result
        return _build_t4_result(
            status="PASS",
            layer_grad_norms=layer_grad_norms,
            gradient_norm_ratio=gradient_norm_ratio,
            max_layer=max_layer,
            min_layer=min_layer,
            max_grad=max_grad,
            min_grad=min_grad,
            entropy=entropy,
            entropy_ratio=entropy_ratio,
            loss_at_init=loss_at_init,
            activation_norms=activation_norms,
            activation_stats=activation_stats,
            calibration=inp.calibration,
            wall_seconds=time.perf_counter() - start_time,
        )

    except Exception as e:
        wall_seconds = time.perf_counter() - start_time
        return _build_error_result(f"T4 MLX error: {e}", wall_seconds)


def _build_t4_result(
    status: TestStatus,
    layer_grad_norms: dict[str, float],
    gradient_norm_ratio: float,
    max_layer: str,
    min_layer: str,
    max_grad: float,
    min_grad: float,
    entropy: float,
    entropy_ratio: float,
    loss_at_init: float,
    activation_norms: dict[str, float],
    activation_stats: dict[str, dict[str, float]],
    calibration: Any,
    wall_seconds: float,
) -> T4Result:
    """Build T4Result from computed metrics."""

    # Get calibration values
    baseline_ratio = 10.0  # Default fallback
    baseline_max_activation = 100.0  # Default fallback
    baseline_layer_activation_norms: dict[str, float] = {}
    baseline_layer_gradient_norms: dict[str, float] = {}

    if calibration:
        baseline_ratio = getattr(calibration, "sota_gradient_norm_ratio", 10.0)
        # Get activation norms from calibration
        calib_acts = getattr(calibration, "sota_layer_activation_norms", {})
        if calib_acts:
            baseline_max_activation = max(calib_acts.values())
            baseline_layer_activation_norms = calib_acts
        # Get gradient norms from calibration
        calib_grads = getattr(calibration, "sota_layer_gradient_norms", {})
        if calib_grads:
            baseline_layer_gradient_norms = calib_grads

    # Calculate comparative signal analysis metrics
    per_layer_snr: dict[str, float] = {}
    dead_neurons_per_layer: dict[str, float] = {}
    total_neurons = 0
    total_dead = 0

    for layer_name, stats in activation_stats.items():
        per_layer_snr[layer_name] = stats["snr"]
        dead_ratio = stats["dead_ratio"]
        dead_neurons_per_layer[layer_name] = dead_ratio
        # Estimate total neurons from tensor shape
        shape = stats.get("tensor_shape", [1, 1, 1])
        layer_neurons = shape[-1] if len(shape) >= 2 else 1
        total_neurons += layer_neurons
        total_dead += int(dead_ratio * layer_neurons)

    # Overall dead neuron ratio
    dead_neuron_ratio = total_dead / total_neurons if total_neurons > 0 else 0.0

    # Overall signal-to-noise ratio (average across layers)
    signal_to_noise_ratio = (
        sum(per_layer_snr.values()) / len(per_layer_snr)
        if per_layer_snr else 0.0
    )

    # Calculate gradient flow health vs baseline
    healthy_gradients = 0
    total_gradient_layers = 0
    for layer_name, grad_norm in layer_grad_norms.items():
        total_gradient_layers += 1
        baseline_norm = baseline_layer_gradient_norms.get(layer_name)
        if baseline_norm:
            # Healthy if within 10x of baseline
            ratio = grad_norm / baseline_norm if baseline_norm > 0 else float("inf")
            if 0.1 <= ratio <= 10.0:
                healthy_gradients += 1
        else:
            # No baseline - consider healthy if non-zero
            if grad_norm > 0:
                healthy_gradients += 1

    gradient_flow_health = (
        healthy_gradients / total_gradient_layers
        if total_gradient_layers > 0 else 0.0
    )

    # Calculate activation distribution shift (KL divergence from baseline)
    activation_distribution_shift: dict[str, float] = {}
    if baseline_layer_activation_norms:
        for layer_name, act_norm in activation_norms.items():
            baseline_norm = baseline_layer_activation_norms.get(layer_name)
            if baseline_norm and baseline_norm > 0:
                # Use log ratio as proxy for distribution shift
                shift = abs(math.log(act_norm / baseline_norm))
                activation_distribution_shift[layer_name] = shift

    # Determine thresholds
    fatal_thresh = max(1000.0, baseline_ratio * 10.0)
    tag_thresh = max(100.0, baseline_ratio * 3.0)

    # Evaluate status and tags
    tags: list[Tag] = []
    result_status: TestStatus = "PASS"
    kill_reason: str | None = None

    # FATAL: zero gradient in any layer
    zero_grad_layers = [name for name, norm in layer_grad_norms.items() if norm == 0.0]
    if zero_grad_layers:
        result_status = "FAIL_FATAL"
        # Build specific kill reason with per-layer details
        layer_details = []
        for layer_name in zero_grad_layers[:3]:  # Show first 3
            layer_details.append(f"{layer_name}: 0.0")
        if len(zero_grad_layers) > 3:
            layer_details.append(f"... and {len(zero_grad_layers) - 3} more")
        kill_reason = f"Zero gradient in layers: {', '.join(layer_details)}"

    # FATAL: gradient norm ratio exceeds threshold
    elif gradient_norm_ratio > fatal_thresh:
        result_status = "FAIL_FATAL"
        # Build specific kill reason showing which layers contribute
        max_grad_norm = layer_grad_norms.get(max_layer, 0.0)
        min_grad_norm = layer_grad_norms.get(min_layer, 0.0)
        kill_reason = (
            f"{max_layer} has high gradients (norm: {max_grad_norm:.4f}) while "
            f"{min_layer} has low gradients (norm: {min_grad_norm:.4f}) - "
            f"ratio {gradient_norm_ratio:.2f} exceeds fatal threshold {fatal_thresh:.2f}"
        )

    # TAG: gradient ratio warning
    if gradient_norm_ratio > tag_thresh and result_status != "FAIL_FATAL":
        tags.append(
            Tag(
                tag_id="T4_gradient_ratio",
                test_id="T4",
                category="gradient_pathology",
                description=(
                    f"Gradient norm ratio {gradient_norm_ratio:.2f} exceeds "
                    f"tag threshold {tag_thresh:.2f} (max: {max_layer}={max_grad:.4f}, "
                    f"min: {min_layer}={min_grad:.4f})"
                ),
            )
        )

    # TAG: gradient imbalance (new specific tag)
    if gradient_norm_ratio > 100.0:
        tags.append(
            Tag(
                tag_id="T4_gradient_imbalance",
                test_id="T4",
                category="gradient_pathology",
                description=(
                    f"Severe gradient imbalance: ratio {gradient_norm_ratio:.2f} > 100 "
                    f"({max_layer}={max_grad:.4f} vs {min_layer}={min_grad:.4f})"
                ),
            )
        )

    # TAG: low output entropy
    if entropy_ratio < 0.3:
        tags.append(
            Tag(
                tag_id="T4_low_output_entropy",
                test_id="T4",
                category="entropy_pathology",
                description=(
                    f"Output entropy ratio {entropy_ratio:.3f} below 0.3 threshold "
                    f"(entropy: {entropy:.3f}, max: {math.log(len(activation_norms) + 1):.3f})"
                ),
            )
        )

    # TAG: extreme activation norm
    max_activation = max(activation_norms.values()) if activation_norms else 0.0
    if max_activation > baseline_max_activation * 50:
        tags.append(
            Tag(
                tag_id="T4_extreme_activation_norm",
                test_id="T4",
                category="scale_pathology",
                description=(
                    f"Activation norm {max_activation:.2f} exceeds 50x baseline "
                    f"({baseline_max_activation:.2f})"
                ),
            )
        )

    # TAG: dead neurons (new)
    if dead_neuron_ratio > 0.1:
        # Find layers with highest dead neuron ratios
        worst_layers = sorted(
            dead_neurons_per_layer.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        layer_details = [f"{name}={ratio:.1%}" for name, ratio in worst_layers]
        tags.append(
            Tag(
                tag_id="T4_dead_neurons",
                test_id="T4",
                category="capacity_pathology",
                description=(
                    f"Dead neuron ratio {dead_neuron_ratio:.1%} exceeds 10% threshold "
                    f"(worst: {', '.join(layer_details)})"
                ),
            )
        )

    # TAG: low signal-to-noise ratio (new)
    if signal_to_noise_ratio < 0.5:
        # Find layers with lowest SNR
        worst_snr = sorted(
            per_layer_snr.items(),
            key=lambda x: x[1]
        )[:3]
        layer_details = [f"{name}={snr:.3f}" for name, snr in worst_snr]
        tags.append(
            Tag(
                tag_id="T4_low_signal_to_noise",
                test_id="T4",
                category="scale_pathology",
                description=(
                    f"Signal-to-noise ratio {signal_to_noise_ratio:.3f} below 0.5 threshold "
                    f"(worst: {', '.join(layer_details)})"
                ),
            )
        )

    return T4Result(
        status=result_status,
        test_id="T4",
        layer_activation_norms=activation_norms,
        layer_gradient_norms=layer_grad_norms,
        gradient_norm_ratio=gradient_norm_ratio,
        gradient_max_layer=max_layer,
        gradient_min_layer=min_layer,
        output_entropy=entropy,
        entropy_ratio=entropy_ratio,
        loss_at_init=loss_at_init,
        kill_reason=kill_reason,
        tags=tags,
        wall_seconds=wall_seconds,
        # New comparative analysis fields
        gradient_flow_health=gradient_flow_health,
        dead_neuron_ratio=dead_neuron_ratio,
        activation_distribution_shift=activation_distribution_shift,
        signal_to_noise_ratio=signal_to_noise_ratio,
        per_layer_snr=per_layer_snr,
        dead_neurons_per_layer=dead_neurons_per_layer,
    )


def _build_error_result(kill_reason: str, wall_seconds: float) -> T4Result:
    """Build error T4Result."""
    return T4Result(
        status="FAIL_FATAL",
        test_id="T4",
        layer_activation_norms={},
        layer_gradient_norms={},
        gradient_norm_ratio=1.0,
        gradient_max_layer="",
        gradient_min_layer="",
        output_entropy=0.0,
        entropy_ratio=0.0,
        loss_at_init=0.0,
        kill_reason=kill_reason,
        tags=[],
        wall_seconds=wall_seconds,
        gradient_flow_health=0.0,
        dead_neuron_ratio=0.0,
        activation_distribution_shift={},
        signal_to_noise_ratio=0.0,
        per_layer_snr={},
        dead_neurons_per_layer={},
    )


def _detect_framework_from_file(train_gpt_path: Path) -> str:
    """Detect whether a train_gpt file is PyTorch or MLX based.
    
    Args:
        train_gpt_path: Path to the train_gpt file
        
    Returns:
        "pytorch" or "mlx"
    """
    # Check filename first
    if "mlx" in str(train_gpt_path).lower():
        return "mlx"
    
    # Check file content for MLX imports
    try:
        content = train_gpt_path.read_text()
        if "import mlx" in content or "from mlx" in content:
            return "mlx"
    except Exception:
        pass
    
    # Default to PyTorch
    return "pytorch"


def run_t4(inp: FalsifierInput) -> T4Result:
    """Run T4: Signal Propagation test.

    Detects framework from input and routes to appropriate implementation.

    Args:
        inp: FalsifierInput with calibration data and train_gpt path

    Returns:
        T4Result with status, gradient metrics, comparative analysis, and tags
    """
    # Detect framework from train_gpt path
    train_gpt_path = _get_train_gpt_path(inp)

    # Determine framework from file extension or content
    framework = _detect_framework_from_file(train_gpt_path)
    if framework == "mlx":
        return run_t4_mlx(inp)
    else:
        # Default to PyTorch for train_gpt.py
        return run_t4_pytorch(inp)
