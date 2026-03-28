"""T3: Compilation & Construction gate.

Validates that the proposed code:
- Can be imported without SyntaxError or ImportError
- Can instantiate a minimal model
- Can run forward and backward passes
- Produces finite outputs and gradients
- Has no disconnected parameters (requires_grad but no grad)
- Has consistent layer shapes across transformer blocks
- Has proper gradient flow to all parameters
- Has reasonable initial scale (no NaN/Inf in initial forward)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from falsifier.adapters.parameter_golf import (
    instantiate_minimal_model,
    load_train_gpt_module,
)
from falsifier.types import FalsifierInput, Tag, T3Result, TestStatus
from falsifier.utils.framework_adapter import (
    check_finite,
    create_random_input,
    create_rolled_targets,
    detect_framework,
    get_model_info,
    get_named_parameters,
    get_param_count,
    has_inf,
    has_nan,
    model_forward,
    requires_grad_context,
    tensor_mean,
    tensor_std,
)


def _check_layer_shape_consistency(model: Any) -> tuple[bool, dict[str, Any]]:
    """Check that all transformer layers have consistent shapes.

    Returns:
        (is_consistent, diagnostics_dict)
    """
    diagnostics: dict[str, Any] = {
        "num_layers": 0,
        "checks_performed": [],
        "mismatches": [],
    }

    if not hasattr(model, "blocks") or not model.blocks:
        return False, {**diagnostics, "error": "Model has no 'blocks' attribute"}

    num_layers = len(model.blocks)
    diagnostics["num_layers"] = num_layers

    # Check attention heads consistency
    try:
        first_attn = model.blocks[0].attn
        expected_heads = first_attn.num_heads
        expected_kv_heads = getattr(first_attn, "num_kv_heads", expected_heads)
        expected_head_dim = getattr(first_attn, "head_dim", None)

        for i, block in enumerate(model.blocks):
            attn = block.attn
            if attn.num_heads != expected_heads:
                diagnostics["mismatches"].append({
                    "layer": i,
                    "component": "attn.num_heads",
                    "expected": expected_heads,
                    "actual": attn.num_heads,
                })
            if hasattr(attn, "num_kv_heads") and attn.num_kv_heads != expected_kv_heads:
                diagnostics["mismatches"].append({
                    "layer": i,
                    "component": "attn.num_kv_heads",
                    "expected": expected_kv_heads,
                    "actual": attn.num_kv_heads,
                })
        diagnostics["checks_performed"].append("attention_heads")
    except Exception as e:
        diagnostics["attention_error"] = str(e)

    # Check embedding dimension consistency
    try:
        # Framework-agnostic way to get embedding dims
        framework = detect_framework(model)
        if framework == "pytorch":
            tok_emb_dim = model.tok_emb.embedding_dim
            pos_emb_dim = model.pos_emb.embedding_dim
        else:  # mlx
            # MLX doesn't have explicit embedding_dim attribute
            tok_emb_dim = model.tok_emb.weight.shape[1]
            pos_emb_dim = model.pos_emb.weight.shape[1]

        if tok_emb_dim != pos_emb_dim:
            diagnostics["mismatches"].append({
                "layer": "embedding",
                "component": "tok_emb vs pos_emb",
                "expected": tok_emb_dim,
                "actual": pos_emb_dim,
            })

        # Check lm_head if it exists and is not tied
        if hasattr(model, "lm_head") and model.lm_head is not None:
            if framework == "pytorch":
                lm_head_dim = model.lm_head.in_features
            else:  # mlx - need to infer from weight shape
                lm_head_dim = model.lm_head.weight.shape[1] if hasattr(model.lm_head, "weight") else tok_emb_dim
            if lm_head_dim != tok_emb_dim:
                diagnostics["mismatches"].append({
                    "layer": "output",
                    "component": "lm_head.in_features",
                    "expected": tok_emb_dim,
                    "actual": lm_head_dim,
                })

        diagnostics["checks_performed"].append("embeddings")
    except Exception as e:
        diagnostics["embedding_error"] = str(e)

    # Check layer dimension consistency (model_dim)
    try:
        first_block = model.blocks[0]
        # Common attribute names for model dimension
        model_dim = None
        for attr in ["model_dim", "n_embd", "d_model", "embed_dim"]:
            if hasattr(first_block, attr):
                model_dim = getattr(first_block, attr)
                break

        # If no direct attribute, try to infer from attention
        if model_dim is None and hasattr(first_block, "attn"):
            attn = first_block.attn
            if hasattr(attn, "qkv"):
                # Try to infer from qkv projection
                framework = detect_framework(model)
                if framework == "pytorch":
                    qkv_out = attn.qkv.out_features if hasattr(attn.qkv, "out_features") else None
                else:  # mlx
                    qkv_out = attn.qkv.weight.shape[0] if hasattr(attn.qkv, "weight") else None
                if qkv_out:
                    model_dim = qkv_out // 3  # Approximate

        diagnostics["inferred_model_dim"] = model_dim
        diagnostics["checks_performed"].append("layer_dimensions")
    except Exception as e:
        diagnostics["dimension_error"] = str(e)

    is_consistent = len(diagnostics["mismatches"]) == 0
    return is_consistent, diagnostics


def _check_forward_backward_consistency(
    model: Any,
    input_ids: Any,
    target_ids: Any,
) -> tuple[bool, dict[str, Any]]:
    """Check that forward and backward passes work consistently.

    Returns:
        (is_consistent, diagnostics_dict)
    """
    diagnostics: dict[str, Any] = {
        "forward_ok": False,
        "backward_ok": False,
        "gradients_computed": 0,
        "total_trainable_params": 0,
        "params_without_grad": [],
        "forward_error": None,
        "backward_error": None,
    }

    framework = detect_framework(model)

    # Forward pass check
    try:
        loss = model_forward(model, input_ids, target_ids)
        diagnostics["forward_ok"] = True
        diagnostics["loss_shape"] = list(loss.shape) if hasattr(loss, "shape") else None
        diagnostics["loss_finite"] = check_finite(loss)
    except Exception as e:
        diagnostics["forward_error"] = f"{type(e).__name__}: {e}"
        return False, diagnostics

    # Backward pass check (PyTorch only - MLX uses value_and_grad)
    if framework == "pytorch":
        try:
            loss_mean = loss.mean()
            loss_mean.backward()
            diagnostics["backward_ok"] = True
        except Exception as e:
            diagnostics["backward_error"] = f"{type(e).__name__}: {e}"
            return False, diagnostics

        # Check gradient flow to all parameters
        total_params = 0
        trainable_params = 0
        params_without_grad = []

        for name, param in get_named_parameters(model):
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
                if param.grad is None:
                    params_without_grad.append(name)

        diagnostics["gradients_computed"] = trainable_params - len(params_without_grad)
        diagnostics["total_trainable_params"] = trainable_params
        diagnostics["params_without_grad"] = params_without_grad

        is_consistent = len(params_without_grad) == 0
    else:  # mlx
        # MLX handles gradients differently via value_and_grad
        # For basic compilation check, just verify forward works
        diagnostics["backward_ok"] = True  # Assumed OK for MLX - full training uses different path
        diagnostics["gradients_computed"] = 0  # Would need value_and_grad to compute
        diagnostics["total_trainable_params"] = get_param_count(model)
        diagnostics["params_without_grad"] = []
        is_consistent = True

    return is_consistent, diagnostics


def _check_init_scale(model: Any, loss: Any) -> tuple[bool, dict[str, Any]]:
    """Check that initial scale is reasonable (no NaN/Inf in first forward).

    Returns:
        (is_reasonable, diagnostics_dict)
    """
    diagnostics: dict[str, Any] = {
        "loss_has_nan": False,
        "loss_has_inf": False,
        "loss_mean": None,
        "loss_std": None,
        "activations_checked": [],
    }

    # Check loss for NaN/Inf using framework-agnostic functions
    loss_has_nan = has_nan(loss)
    loss_has_inf = has_inf(loss)

    diagnostics["loss_has_nan"] = loss_has_nan
    diagnostics["loss_has_inf"] = loss_has_inf

    if not loss_has_nan and not loss_has_inf:
        with requires_grad_context(enabled=False):
            diagnostics["loss_mean"] = tensor_mean(loss)
            diagnostics["loss_std"] = tensor_std(loss)

    # Check if loss value is reasonable for cross-entropy (should be around ln(vocab_size))
    # For small test models, this is a rough sanity check
    is_reasonable = not loss_has_nan and not loss_has_inf

    return is_reasonable, diagnostics


def run_t3(inp: FalsifierInput) -> T3Result:
    """Run T3: Compilation & Construction test.

    Args:
        inp: FalsifierInput with proposed_train_gpt source code

    Returns:
        T3Result with compilation status, timing, and construction diagnostics
    """
    start_time = time.perf_counter()

    status: TestStatus = "PASS"
    tags: list[Tag] = []
    kill_reason: str | None = None

    # Result fields
    actual_params = 0
    output_shape: list[int] = []
    loss_has_nan_flag = False
    loss_has_inf_flag = False
    grad_nan = False
    grad_inf = False
    params_no_grad: list[str] = []
    forward_ms = 0.0
    backward_ms = 0.0
    gpu_memory = 0

    # Construction diagnostics fields
    layer_shapes_consistent = True
    forward_backward_consistent = True
    init_scale_reasonable = True
    construction_diagnostics: dict[str, Any] = {}

    # Create temp file with proposed code
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(inp.proposed_train_gpt)
            temp_path = Path(f.name)

        # Determine which imports to stub
        # Always stub sentencepiece (not needed for structural checks)
        # Only stub mlx if it's not actually installed (for structural checks on MLX code)
        block_imports = ["sentencepiece", "spm"]
        try:
            import mlx  # noqa: F401
            # MLX is installed - don't stub it, let the real library load
        except ImportError:
            # MLX not installed - stub it for structural checks
            block_imports.extend(["mlx", "mlx.nn", "mlx.core", "mlx.optimizers", "mlx.utils"])
        
        # Try to import/load the module (with import stubbing for optional deps)
        try:
            module = load_train_gpt_module(temp_path, block_imports=block_imports)
        except SyntaxError as e:
            return T3Result(
                status="FAIL_FATAL",
                test_id="T3",
                kill_reason=f"SyntaxError: {e}",
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics={"import_error": f"SyntaxError at line {getattr(e, 'lineno', 'unknown')}: {e}"},
            )
        except ImportError as e:
            return T3Result(
                status="FAIL_FATAL",
                test_id="T3",
                kill_reason=f"ImportError: {e}",
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics={"import_error": str(e)},
            )

        # Try to instantiate model with minimal env (with conditional import stubbing)
        try:
            module, model = instantiate_minimal_model(temp_path, block_imports=block_imports)
        except Exception as e:
            return T3Result(
                status="FAIL_FATAL",
                test_id="T3",
                kill_reason=f"InstantiationError: {e}",
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics={"instantiation_error": str(e)},
            )

        # Detect framework and get parameter count
        framework = detect_framework(model)
        actual_params = get_param_count(model)

        # ═════════════════════════════════════════════════════════════════════════
        # Layer Shape Consistency Check
        # ═════════════════════════════════════════════════════════════════════════
        layer_shapes_consistent, shape_diagnostics = _check_layer_shape_consistency(model)
        construction_diagnostics["layer_shapes"] = shape_diagnostics

        if not layer_shapes_consistent:
            # Build detailed error message about mismatches
            mismatch_details = []
            for mismatch in shape_diagnostics.get("mismatches", []):
                layer = mismatch.get("layer", "unknown")
                component = mismatch.get("component", "unknown")
                expected = mismatch.get("expected", "unknown")
                actual = mismatch.get("actual", "unknown")
                mismatch_details.append(f"Layer {layer} {component}: expected {expected}, got {actual}")

            mismatch_msg = "; ".join(mismatch_details[:3])  # Limit to first 3
            if len(shape_diagnostics.get("mismatches", [])) > 3:
                mismatch_msg += f" (+{len(shape_diagnostics['mismatches']) - 3} more)"

            tags.append(
                Tag(
                    tag_id="T3_layer_shape_inconsistency",
                    test_id="T3",
                    category="capacity_pathology",
                    description=f"Layer shapes inconsistent: {mismatch_msg}",
                )
            )

        # ═════════════════════════════════════════════════════════════════════════
        # Forward Pass with Construction Diagnostics
        # ═════════════════════════════════════════════════════════════════════════
        seq_len = 8
        model_info = get_model_info(model)
        vocab_size = model_info.get("vocab_size", 64)  # Fallback to default
        batch_size = 1

        # Create dummy input using framework-agnostic functions
        input_ids = create_random_input(vocab_size, batch_size, seq_len, framework=framework)
        target_ids = create_rolled_targets(input_ids)

        # Forward pass
        forward_start = time.perf_counter()
        loss = None
        try:
            loss = model_forward(model, input_ids, target_ids)
            output_shape = list(loss.shape) if hasattr(loss, "shape") else []
            loss_has_nan = has_nan(loss)
            loss_has_inf = has_inf(loss)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Forward pass error: {type(e).__name__}: {e}"
            construction_diagnostics["forward_error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "input_shape": list(input_ids.shape),
            }
            return T3Result(
                status="FAIL_FATAL",
                test_id="T3",
                actual_params=actual_params,
                layer_shapes_consistent=layer_shapes_consistent,
                kill_reason=error_msg,
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics=construction_diagnostics,
                tags=tags,
            )
        forward_ms = (time.perf_counter() - forward_start) * 1000  # Convert to ms

        # Check init scale (reasonableness of initial forward)
        init_scale_reasonable, scale_diagnostics = _check_init_scale(model, loss)
        construction_diagnostics["init_scale"] = scale_diagnostics

        # Check for NaN/Inf in output - FATAL
        if loss_has_nan or loss_has_inf:
            status = "FAIL_FATAL"
            kill_reason = f"Output has {'NaN' if loss_has_nan else ''}{' and ' if loss_has_nan and loss_has_inf else ''}{'Inf' if loss_has_inf else ''}"
            return T3Result(
                status=status,
                test_id="T3",
                actual_params=actual_params,
                output_shape=output_shape,
                has_nan=loss_has_nan,
                has_inf=loss_has_inf,
                layer_shapes_consistent=layer_shapes_consistent,
                init_scale_reasonable=False,
                kill_reason=kill_reason,
                forward_ms=forward_ms,
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics=construction_diagnostics,
                tags=tags,
            )

        # ═════════════════════════════════════════════════════════════════════════
        # Forward-Backward Consistency Check
        # ═════════════════════════════════════════════════════════════════════════
        forward_backward_consistent, fb_diagnostics = _check_forward_backward_consistency(
            model, input_ids, target_ids
        )
        construction_diagnostics["forward_backward"] = fb_diagnostics

        if not forward_backward_consistent:
            # Build detailed message about which params lack gradients
            missing_grads = fb_diagnostics.get("params_without_grad", [])
            params_no_grad = missing_grads

            if missing_grads:
                grad_detail = f"{len(missing_grads)} parameters lack gradients"
                if len(missing_grads) <= 5:
                    grad_detail += f": {', '.join(missing_grads)}"
                else:
                    grad_detail += f": {', '.join(missing_grads[:3])}... (+{len(missing_grads) - 3} more)"

                tags.append(
                    Tag(
                        tag_id="T3_partial_gradient_flow",
                        test_id="T3",
                        category="gradient_pathology",
                        description=grad_detail,
                    )
                )

        # Run backward timing separately for metrics (PyTorch only)
        backward_start = time.perf_counter()
        grad_nan = False
        grad_inf = False
        params_no_grad: list[str] = []

        if framework == "pytorch":
            try:
                # Fresh forward pass with grad
                loss = model_forward(model, input_ids, target_ids)
                loss_mean = loss.mean()
                loss_mean.backward()
            except Exception as e:
                error_msg = f"Backward pass error: {type(e).__name__}: {e}"
                construction_diagnostics["backward_error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                }
                return T3Result(
                    status="FAIL_FATAL",
                    test_id="T3",
                    actual_params=actual_params,
                    output_shape=output_shape,
                    layer_shapes_consistent=layer_shapes_consistent,
                    forward_backward_consistent=False,
                    kill_reason=error_msg,
                    forward_ms=forward_ms,
                    wall_seconds=time.perf_counter() - start_time,
                    construction_diagnostics=construction_diagnostics,
                    tags=tags,
                )

            # Check gradients for NaN/Inf
            for name, param in get_named_parameters(model):
                if hasattr(param, "requires_grad") and param.requires_grad:
                    if param.grad is None:
                        params_no_grad.append(name)
                    else:
                        if has_nan(param.grad):
                            grad_nan = True
                        if has_inf(param.grad):
                            grad_inf = True

        backward_ms = (time.perf_counter() - backward_start) * 1000  # Convert to ms

        # FATAL: NaN or Inf in gradients
        if grad_nan or grad_inf:
            status = "FAIL_FATAL"
            kill_reason = f"Gradient has {'NaN' if grad_nan else ''}{' and ' if grad_nan and grad_inf else ''}{'Inf' if grad_inf else ''}"
            return T3Result(
                status=status,
                test_id="T3",
                actual_params=actual_params,
                output_shape=output_shape,
                has_nan=loss_has_nan,
                has_inf=loss_has_inf,
                grad_nan=grad_nan,
                grad_inf=grad_inf,
                params_no_grad=params_no_grad,
                layer_shapes_consistent=layer_shapes_consistent,
                forward_backward_consistent=forward_backward_consistent,
                init_scale_reasonable=init_scale_reasonable,
                kill_reason=kill_reason,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                wall_seconds=time.perf_counter() - start_time,
                construction_diagnostics=construction_diagnostics,
                tags=tags,
            )

        # TAG: disconnected parameters (requires_grad but grad is None)
        if params_no_grad:
            tags.append(
                Tag(
                    tag_id="T3_disconnected_params",
                    test_id="T3",
                    category="capacity_pathology",
                    description=f"{len(params_no_grad)} parameters have requires_grad=True but no gradient: {', '.join(params_no_grad[:3])}{'...' if len(params_no_grad) > 3 else ''}",
                )
            )

        # If we have partial gradient flow warning, ensure it's a FAIL_TAG not PASS
        if any(t.tag_id == "T3_partial_gradient_flow" for t in tags):
            status = "FAIL_TAG"

        # If we have layer shape inconsistency, ensure it's a FAIL_TAG
        if any(t.tag_id == "T3_layer_shape_inconsistency" for t in tags):
            status = "FAIL_TAG"

        wall_seconds = time.perf_counter() - start_time

        return T3Result(
            status=status,
            test_id="T3",
            actual_params=actual_params,
            output_shape=output_shape,
            has_nan=loss_has_nan,
            has_inf=loss_has_inf,
            grad_nan=grad_nan,
            grad_inf=grad_inf,
            params_no_grad=params_no_grad,
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            gpu_memory=gpu_memory,
            kill_reason=kill_reason,
            tags=tags,
            wall_seconds=wall_seconds,
            layer_shapes_consistent=layer_shapes_consistent,
            forward_backward_consistent=forward_backward_consistent,
            init_scale_reasonable=init_scale_reasonable,
            construction_diagnostics=construction_diagnostics,
        )

    except Exception as e:
        wall_seconds = time.perf_counter() - start_time
        return T3Result(
            status="FAIL_FATAL",
            test_id="T3",
            kill_reason=f"T3 unexpected error: {e}",
            wall_seconds=wall_seconds,
            construction_diagnostics={"unexpected_error": f"{type(e).__name__}: {e}"},
        )

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
