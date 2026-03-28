"""T5: Initialization Diagnostics gate.

Analyzes the model at initialization for pathological conditions:
- Extreme logit values (overconfidence)
- Low effective rank (redundant capacity)
- High condition numbers (numerical instability)
- Weight symmetry (initialization redundancy)
- Kurtosis and effective rank aggregates

Supports both PyTorch and MLX frameworks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from falsifier.adapters.mlx_adapter import mlx_available
from falsifier.adapters.parameter_golf import instantiate_minimal_model
from falsifier.types import FalsifierInput, Tag, T5Result, TestStatus


def _detect_framework(model: Any) -> str:
    """Detect whether model is PyTorch or MLX based.

    Returns:
        "pytorch" if model is a torch.nn.Module
        "mlx" if model is an mlx.nn.Module
    """
    # Check for PyTorch first
    try:
        import torch.nn as nn_torch

        if isinstance(model, nn_torch.Module):
            return "pytorch"
    except ImportError:
        pass

    # Check for MLX
    try:
        import mlx.nn as nn_mlx

        if isinstance(model, nn_mlx.Module):
            return "mlx"
    except ImportError:
        pass

    # Fallback: check for framework-specific attributes
    if hasattr(model, "state") and hasattr(model, "update"):
        # MLX models have .state() and .update() methods
        return "mlx"
    if hasattr(model, "parameters") and hasattr(model, "named_parameters"):
        # PyTorch models typically have these methods
        return "pytorch"

    raise RuntimeError(f"Cannot detect framework for model type: {type(model)}")


def _to_numpy(tensor: Any, framework: str) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        numpy array
    """
    if framework == "pytorch":
        return tensor.detach().cpu().numpy()
    else:  # mlx
        # Convert to float32 first to handle bfloat16 and other special dtypes
        # that numpy doesn't directly support
        import mlx.core as mx
        return np.array(tensor.astype(mx.float32))


def _get_tensor_shape(tensor: Any, framework: str) -> tuple[int, ...]:
    """Get tensor shape.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        shape tuple
    """
    if framework == "pytorch":
        return tuple(tensor.shape)
    else:  # mlx
        return tuple(tensor.shape)


def _get_tensor_ndim(tensor: Any, framework: str) -> int:
    """Get tensor ndim.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        number of dimensions
    """
    if framework == "pytorch":
        return tensor.ndim
    else:  # mlx
        return tensor.ndim


def _is_floating_point(tensor: Any, framework: str) -> bool:
    """Check if tensor is floating point.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        True if floating point
    """
    if framework == "pytorch":
        return tensor.is_floating_point()
    else:  # mlx
        import mlx.core as mx

        return mx.issubdtype(tensor.dtype, mx.floating)


def _tensor_numel(tensor: Any, framework: str) -> int:
    """Get number of elements in tensor.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        number of elements
    """
    shape = _get_tensor_shape(tensor, framework)
    return int(np.prod(shape))


def _tensor_item(tensor: Any, framework: str) -> float:
    """Get scalar value from tensor.

    Args:
        tensor: PyTorch Tensor or MLX array (scalar)
        framework: "pytorch" or "mlx"

    Returns:
        scalar value as float
    """
    if framework == "pytorch":
        return float(tensor.item())
    else:  # mlx
        return float(tensor.item())


def _tensor_max(tensor: Any, framework: str) -> Any:
    """Get max value from tensor.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        scalar tensor
    """
    if framework == "pytorch":
        import torch

        return tensor.max()
    else:  # mlx
        import mlx.core as mx

        return mx.max(tensor)


def _tensor_std(tensor: Any, framework: str) -> Any:
    """Get std of tensor.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        scalar tensor
    """
    if framework == "pytorch":
        import torch

        return tensor.std()
    else:  # mlx
        import mlx.core as mx

        return mx.std(tensor)


def _normalize_l2(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """L2 normalize array along axis.

    Args:
        x: numpy array
        axis: axis to normalize along

    Returns:
        L2-normalized array
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return x / norms


def _randperm(n: int, framework: str, seed: int | None = None) -> Any:
    """Generate random permutation.

    Args:
        n: number of elements
        framework: "pytorch" or "mlx"
        seed: optional random seed

    Returns:
        tensor/array of permuted indices
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(n)

    if framework == "pytorch":
        import torch

        return torch.from_numpy(indices).long()
    else:  # mlx
        import mlx.core as mx

        return mx.array(indices, dtype=mx.int32)


def _randint(low: int, high: int, shape: tuple[int, ...], framework: str, seed: int | None = None) -> Any:
    """Generate random integers.

    Args:
        low: lower bound (inclusive)
        high: upper bound (exclusive)
        shape: output shape
        framework: "pytorch" or "mlx"
        seed: optional random seed

    Returns:
        tensor/array of random integers
    """
    if seed is not None:
        np.random.seed(seed)
    values = np.random.randint(low, high, size=shape)

    if framework == "pytorch":
        import torch

        return torch.from_numpy(values).long()
    else:  # mlx
        import mlx.core as mx

        return mx.array(values, dtype=mx.int32)


def _effective_rank(tensor: Any, framework: str) -> float | None:
    """Compute effective rank via SVD entropy.

    Effective rank = exp(entropy of singular value distribution).
    Returns None for non-2D tensors or empty tensors.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        effective rank or None
    """
    if _get_tensor_ndim(tensor, framework) != 2:
        return None

    # Convert to numpy for SVD computation (works for both frameworks)
    x = _to_numpy(tensor, framework)
    if x.size == 0:
        return None

    # Ensure float32 for numerical stability
    x = x.astype(np.float32)

    # Compute SVD
    try:
        singular_values = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return None

    if singular_values.size == 0:
        return None

    total = singular_values.sum()
    if float(total) == 0.0:
        return 0.0

    probs = singular_values / total
    # Avoid log(0)
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    return float(np.exp(entropy))


def _tensor_kurtosis(tensor: Any, framework: str) -> float:
    """Compute excess kurtosis of tensor values.

    Kurtosis measures tail heaviness. High kurtosis indicates
    heavy-tailed distributions (potential outliers).

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        excess kurtosis
    """
    # Convert to numpy for computation
    x = _to_numpy(tensor, framework).astype(np.float32).reshape(-1)
    if x.size < 4:
        return 0.0

    mean = x.mean()
    centered = x - mean
    var = np.mean(centered**2)
    if float(var) == 0.0:
        return 0.0

    fourth = np.mean(centered**4)
    return float(fourth / (var * var))


def _condition_number(tensor: Any, framework: str) -> float | None:
    """Compute condition number s[0] / s[-1] via SVD.

    High condition number indicates numerical instability.
    Returns None for non-2D tensors.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        condition number or None
    """
    if _get_tensor_ndim(tensor, framework) != 2:
        return None

    # Convert to numpy for SVD computation
    x = _to_numpy(tensor, framework).astype(np.float32)
    if x.size == 0:
        return None

    try:
        singular_values = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return None

    if singular_values.size == 0:
        return None

    s_max = float(singular_values[0])
    s_min = float(singular_values[-1])
    if s_min == 0:
        return float("inf")
    return s_max / s_min


def _weight_symmetry(tensor: Any, framework: str) -> float | None:
    """Compute weight symmetry via mean pairwise cosine similarity.

    Samples rows if tensor is large (>1000 rows) for efficiency.
    High symmetry indicates redundant rows.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        mean pairwise cosine similarity or None
    """
    # Convert to numpy for computation
    x = _to_numpy(tensor, framework).astype(np.float32)

    if x.ndim < 2:
        return None

    # Handle various weight shapes: (out, in), (out, in, k, k), etc.
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    n_rows = x.shape[0]
    if n_rows < 2:
        return None

    # Sample rows if too large (per PRD: 25% sampling)
    if n_rows > 100:
        sample_size = max(2, n_rows // 4)
        indices = np.random.permutation(n_rows)[:sample_size]
        x = x[indices]
        n_rows = sample_size

    # Normalize rows (L2 normalization)
    x_norm = _normalize_l2(x, axis=1)

    # Compute pairwise cosine similarities
    sim_matrix = x_norm @ x_norm.T

    # Mean of upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
    if not mask.any():
        return None

    similarities = sim_matrix[mask]
    return float(similarities.mean())


def _compute_singular_value_percentiles(tensor: Any, framework: str) -> dict[str, float] | None:
    """Compute singular value percentiles (p5, p25, p50, p75, p95) for a 2D tensor.

    Returns None for non-2D tensors or empty tensors.

    Args:
        tensor: PyTorch Tensor or MLX array
        framework: "pytorch" or "mlx"

    Returns:
        dict of percentiles or None
    """
    if _get_tensor_ndim(tensor, framework) != 2:
        return None

    # Convert to numpy for SVD computation
    x = _to_numpy(tensor, framework).astype(np.float32)
    if x.size == 0:
        return None

    try:
        singular_values = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return None

    if singular_values.size == 0:
        return None

    # Compute percentiles
    percentiles = [5, 25, 50, 75, 95]
    values = np.percentile(singular_values, percentiles)

    return {
        "p5": float(values[0]),
        "p25": float(values[1]),
        "p50": float(values[2]),
        "p75": float(values[3]),
        "p95": float(values[4]),
    }


def _compute_init_symmetry_score(weight_symmetries: dict[str, float]) -> float:
    """Compute overall initialization symmetry score.

    Score of 0 means all weight matrices have identical rows (fully symmetric).
    Score of 1 means all weight matrices are fully diverse (orthogonal rows).
    """
    if not weight_symmetries:
        return 0.0

    # Average symmetry score across all weight matrices
    mean_symmetry = sum(weight_symmetries.values()) / len(weight_symmetries)

    # Convert to diversity score (1 - symmetry)
    # High symmetry (close to 1) means low diversity (close to 0)
    diversity_score = 1.0 - mean_symmetry

    return max(0.0, min(1.0, diversity_score))


def _compute_capacity_metrics(
    effective_ranks: dict[str, float],
    model_named_params: dict[str, Any],
    framework: str,
) -> tuple[float, float]:
    """Compute capacity utilization and rank deficiency ratio.

    Returns:
        capacity_utilization: mean(effective_rank) / mean(theoretical_max_rank)
        rank_deficiency_ratio: % of matrices with eff_rank < 0.9 * min_dim
    """
    if not effective_ranks:
        return 0.0, 0.0

    total_matrices = len(effective_ranks)
    rank_deficient_count = 0
    total_eff_rank = 0.0
    total_max_rank = 0.0

    for name, eff_rank in effective_ranks.items():
        tensor = model_named_params.get(name)
        if tensor is None or _get_tensor_ndim(tensor, framework) != 2:
            continue

        shape = _get_tensor_shape(tensor, framework)
        min_dim = min(shape)
        max_rank = min_dim  # theoretical max rank is min_dim

        total_eff_rank += eff_rank
        total_max_rank += max_rank

        # Check if rank deficient (effective rank < 90% of theoretical max)
        if eff_rank < 0.9 * min_dim:
            rank_deficient_count += 1

    capacity_utilization = (total_eff_rank / total_max_rank) if total_max_rank > 0 else 0.0
    rank_deficiency_ratio = rank_deficient_count / total_matrices if total_matrices > 0 else 0.0

    return capacity_utilization, rank_deficiency_ratio


def _aggregate_spectrum_percentiles(
    all_percentiles: list[dict[str, float]]
) -> dict[str, float]:
    """Aggregate singular value percentiles across all weight matrices.

    Returns the mean of each percentile across all matrices.
    """
    if not all_percentiles:
        return {}

    keys = ["p5", "p25", "p50", "p75", "p95"]
    aggregated = {}

    for key in keys:
        values = [p[key] for p in all_percentiles if key in p]
        if values:
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = 0.0

    return aggregated


def _forward_pass(model: Any, input_ids: Any, framework: str) -> Any:
    """Run forward pass to get logits.

    Args:
        model: PyTorch or MLX model
        input_ids: input tensor/array
        framework: "pytorch" or "mlx"

    Returns:
        logits tensor/array
    """
    if framework == "pytorch":
        import torch
        import torch.nn.functional as F

        # PyTorch forward with no_grad
        with torch.no_grad():
            # Try different calling conventions
            try:
                # Some models support return_logits
                logits = model(input_ids, return_logits=True)
            except TypeError:
                # Model doesn't support return_logits
                try:
                    # Try calling with just input_ids
                    logits = model(input_ids)
                except TypeError:
                    # Model requires both arguments
                    # We need to get logits without computing cross-entropy loss
                    # Manually extract logits from model
                    x = model.tok_emb(input_ids)
                    x = F.rms_norm(x, (x.size(-1),))
                    # Handle different block signatures
                    for block in model.blocks:
                        if hasattr(block, 'forward'):
                            # Try with different signatures
                            try:
                                x = block(x, x)  # For blocks that expect (x, x0)
                            except TypeError:
                                try:
                                    x = block(x)  # For blocks that expect just x
                                except TypeError:
                                    # Fallback: use the block's forward method directly
                                    x = block.forward(x)
                    x = model.final_norm(x)
                    # Get logits from tok_emb (tied embeddings) or lm_head
                    if hasattr(model, 'tie_embeddings') and model.tie_embeddings:
                        logits = F.linear(x, model.tok_emb.weight)
                    elif hasattr(model, 'lm_head') and model.lm_head is not None:
                        logits = model.lm_head(x)
                    else:
                        raise RuntimeError("Cannot extract logits from model")
        return logits
    else:  # mlx
        import mlx.core as mx

        # MLX forward - no need for no_grad equivalent
        # MLX models typically have a forward or __call__ method
        if hasattr(model, "__call__"):
            logits = model(input_ids)
        elif hasattr(model, "forward"):
            logits = model.forward(input_ids)
        else:
            raise RuntimeError("MLX model has no callable forward method")

        # Handle MLX output - ensure it's array type
        if not isinstance(logits, mx.array):
            raise RuntimeError(f"MLX model output is not an array: {type(logits)}")

        return logits


def _get_named_parameters(model: Any, framework: str) -> dict[str, Any]:
    """Get named parameters from model.

    Args:
        model: PyTorch or MLX model
        framework: "pytorch" or "mlx"

    Returns:
        dict of parameter name -> tensor/array
    """
    if framework == "pytorch":
        return dict(model.named_parameters())
    else:  # mlx
        # MLX models: use tree_flatten on the model directly
        # This handles both standard nn.Module and custom models
        try:
            from mlx.utils import tree_flatten
            flat = tree_flatten(model)
            params = {name: value for name, value in flat}
        except ImportError:
            params = {}
            # Fallback: try to access modules and their parameters
            if hasattr(model, "__dict__"):
                import mlx.core as mx
                for name, value in model.__dict__.items():
                    if isinstance(value, mx.array) or (hasattr(value, "__dict__") and 
                        any(isinstance(v, mx.array) for v in value.__dict__.values())):
                        params[name] = value
        return params


def _get_vocab_size(model: Any, framework: str) -> int:
    """Get vocabulary size from model.

    Args:
        model: PyTorch or MLX model
        framework: "pytorch" or "mlx"

    Returns:
        vocabulary size
    """
    if framework == "pytorch":
        # PyTorch: tok_emb.num_embeddings
        if hasattr(model, "tok_emb"):
            return int(model.tok_emb.num_embeddings)
        raise RuntimeError("Cannot find tok_emb for vocab size")
    else:  # mlx
        # MLX: tok_emb.weight.shape[0]
        if hasattr(model, "tok_emb"):
            return int(model.tok_emb.weight.shape[0])
        raise RuntimeError("Cannot find tok_emb for vocab size")


def run_t5(inp: FalsifierInput) -> T5Result:
    """Run T5: Initialization Diagnostics test.

    Builds minimal model, runs forward, and analyzes:
    - Logit max and std at output
    - Effective rank of weight matrices (25% sampled for large matrices)
    - Condition numbers
    - Weight symmetry (cosine similarity)
    - Kurtosis and effective rank aggregates

    Supports both PyTorch and MLX frameworks.

    Args:
        inp: FalsifierInput with calibration data and train_gpt path

    Returns:
        T5Result with status, init metrics, and tags
    """
    start_time = time.perf_counter()

    try:
        # Build model with minimal env
        # Use train_gpt_path (the file path) not proposed_train_gpt (the source string)
        source_path = inp.train_gpt_path if isinstance(inp.train_gpt_path, (str, Path)) else ""
        if not source_path and inp.proposed_train_gpt:
            # Write source to temp file
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(inp.proposed_train_gpt)
                source_path = f.name

        # Detect framework from file content
        def _detect_framework_from_file(path: str) -> str:
            if "mlx" in Path(path).name.lower():
                return "mlx"
            try:
                content = Path(path).read_text()
                if "import mlx" in content or "from mlx" in content:
                    return "mlx"
            except Exception:
                pass
            return "pytorch"

        # Check if MLX model
        detected_framework = _detect_framework_from_file(source_path)
        mlx_mode = detected_framework == "mlx" and mlx_available()
        
        if mlx_mode:
            from falsifier.adapters.mlx_adapter import instantiate_mlx_model
            module, model = instantiate_mlx_model(source_path)
        else:
            module, model = instantiate_minimal_model(source_path)

        # Detect framework
        framework = _detect_framework(model)

        # Set model to eval mode
        if framework == "pytorch":
            model.eval()
        else:  # mlx - no explicit eval mode needed
            if hasattr(model, "eval"):
                model.eval()

        vocab_size = _get_vocab_size(model, framework)
        seq_len = 8

        # Forward pass to get logits
        np.random.seed(42)
        input_ids = _randint(0, vocab_size, (1, seq_len), framework, seed=42)

        logits = _forward_pass(model, input_ids, framework)

        # Logit analysis
        logit_max = _tensor_item(_tensor_max(logits, framework), framework)
        logit_std = _tensor_item(_tensor_std(logits, framework), framework)

        # Get calibration baseline
        calibration = inp.calibration
        baseline_logit_max = 10.0
        if calibration:
            baseline_logit_max = getattr(calibration, "sota_init_logit_max", 10.0)

        # Analyze weight matrices (25% sampling for large matrices)
        effective_ranks: dict[str, float] = {}
        condition_numbers: dict[str, float] = {}
        weight_symmetries: dict[str, float] = {}
        kurtosis_values: list[float] = []
        rank_values: list[float] = []
        spectrum_percentiles: list[dict[str, float]] = []

        # Store named params for later lookup
        model_named_params = _get_named_parameters(model, framework)

        for name, tensor in model_named_params.items():
            if not _is_floating_point(tensor, framework):
                continue

            # Kurtosis for all float tensors
            k = _tensor_kurtosis(tensor, framework)
            kurtosis_values.append(k)

            # Effective rank and condition number for 2D matrices
            if _get_tensor_ndim(tensor, framework) == 2:
                er = _effective_rank(tensor, framework)
                cn = _condition_number(tensor, framework)

                if er is not None:
                    effective_ranks[name] = er
                    rank_values.append(er)

                if cn is not None:
                    condition_numbers[name] = cn

                # Singular value spectrum analysis
                sp = _compute_singular_value_percentiles(tensor, framework)
                if sp is not None:
                    spectrum_percentiles.append(sp)

            # Weight symmetry analysis for weight matrices
            if "weight" in name.lower():
                ws = _weight_symmetry(tensor, framework)
                if ws is not None:
                    weight_symmetries[name] = ws

        # Compute aggregates
        kurtosis_mean = sum(kurtosis_values) / len(kurtosis_values) if kurtosis_values else 0.0
        effective_rank_mean = sum(rank_values) / len(rank_values) if rank_values else 0.0

        # Compute new weight spectrum metrics
        init_symmetry_score = _compute_init_symmetry_score(weight_symmetries)
        capacity_utilization, rank_deficiency_ratio = _compute_capacity_metrics(
            effective_ranks, model_named_params, framework
        )
        weight_spectrum_percentiles = _aggregate_spectrum_percentiles(spectrum_percentiles)

        # Determine thresholds
        logit_fatal_thresh = max(10.0, baseline_logit_max * 2.0)
        rank_warning_thresh = 0.3  # Effective rank < 30% of min dimension
        condition_warning_thresh = 1000.0
        symmetry_warning_thresh = 0.9  # High symmetry indicates redundancy
        rank_deficiency_fatal_thresh = 0.5  # 50% of matrices rank deficient
        capacity_utilization_warning_thresh = 0.5  # Poor capacity utilization

        # Evaluate status and tags
        tags: list[Tag] = []
        status: TestStatus = "PASS"
        kill_reason: str | None = None

        # FATAL: extreme logits
        if logit_max > logit_fatal_thresh:
            status = "FAIL_FATAL"
            kill_reason = (
                f"Logit max {logit_max:.2f} exceeds fatal threshold "
                f"{logit_fatal_thresh:.2f} (baseline: {baseline_logit_max:.2f})"
            )

        # FATAL: severe rank deficiency (>50% of matrices)
        elif rank_deficiency_ratio > rank_deficiency_fatal_thresh:
            status = "FAIL_FATAL"
            kill_reason = (
                f"{rank_deficiency_ratio:.0%} of weight matrices are rank-deficient "
                f"(effective rank < 90% of theoretical)"
            )

        # TAG: extreme logits (warning)
        elif logit_max > baseline_logit_max * 1.5:
            tags.append(
                Tag(
                    tag_id="T5_extreme_logits",
                    test_id="T5",
                    category="scale_pathology",
                    description=(
                        f"Logit max {logit_max:.2f} exceeds 1.5x baseline "
                        f"({baseline_logit_max:.2f}), std={logit_std:.2f}"
                    ),
                )
            )

        # TAG: low effective rank
        for name, er in effective_ranks.items():
            tensor = model_named_params.get(name)
            if tensor is None:
                continue
            shape = _get_tensor_shape(tensor, framework)
            min_dim = min(shape)
            er_ratio = er / min_dim if min_dim > 0 else 0.0

            if er_ratio < rank_warning_thresh:
                tags.append(
                    Tag(
                        tag_id="T5_low_effective_rank",
                        test_id="T5",
                        category="capacity_pathology",
                        description=(
                            f"Effective rank {er:.1f}/{min_dim} ({er_ratio:.2%}) for {name}"
                        ),
                    )
                )
                break  # One tag is sufficient

        # TAG: high condition number
        for name, cn in condition_numbers.items():
            if cn > condition_warning_thresh:
                tags.append(
                    Tag(
                        tag_id="T5_high_condition_number",
                        test_id="T5",
                        category="scale_pathology",
                        description=(
                            f"Condition number {cn:.1f} for {name} exceeds {condition_warning_thresh}"
                        ),
                    )
                )
                break  # One tag is sufficient

        # TAG: weight symmetry
        for name, ws in weight_symmetries.items():
            if ws > symmetry_warning_thresh:
                tags.append(
                    Tag(
                        tag_id="T5_weight_symmetry",
                        test_id="T5",
                        category="capacity_pathology",
                        description=(
                            f"Weight symmetry {ws:.3f} for {name} indicates redundancy"
                        ),
                    )
                )
                break  # One tag is sufficient

        # TAG: rank deficiency (warning if >20% but not fatal)
        if 0.2 < rank_deficiency_ratio <= rank_deficiency_fatal_thresh:
            tags.append(
                Tag(
                    tag_id="T5_rank_deficient",
                    test_id="T5",
                    category="capacity_pathology",
                    description=(
                        f"{rank_deficiency_ratio:.1%} of weight matrices are rank-deficient "
                        f"(effective rank < 90% of theoretical max)"
                    ),
                )
            )

        # TAG: poor capacity utilization
        if capacity_utilization < capacity_utilization_warning_thresh:
            tags.append(
                Tag(
                    tag_id="T5_poor_capacity_utilization",
                    test_id="T5",
                    category="capacity_pathology",
                    description=(
                        f"Capacity utilization {capacity_utilization:.1%} below threshold "
                        f"{capacity_utilization_warning_thresh:.0%}"
                    ),
                )
            )

        wall_seconds = time.perf_counter() - start_time

        return T5Result(
            status=status,
            test_id="T5",
            logit_max=logit_max,
            logit_std=logit_std,
            effective_ranks=effective_ranks,
            condition_numbers=condition_numbers,
            weight_symmetry=weight_symmetries,
            kurtosis_mean=kurtosis_mean,
            effective_rank_mean=effective_rank_mean,
            weight_spectrum_percentiles=weight_spectrum_percentiles,
            init_symmetry_score=init_symmetry_score,
            capacity_utilization=capacity_utilization,
            rank_deficiency_ratio=rank_deficiency_ratio,
            kill_reason=kill_reason,
            tags=tags,
            wall_seconds=wall_seconds,
        )

    except Exception as e:
        wall_seconds = time.perf_counter() - start_time
        import traceback
        tb_str = traceback.format_exc()
        return T5Result(
            status="FAIL_FATAL",
            test_id="T5",
            logit_max=0.0,
            logit_std=0.0,
            effective_ranks={},
            condition_numbers={},
            weight_symmetry={},
            kurtosis_mean=0.0,
            effective_rank_mean=0.0,
            weight_spectrum_percentiles={},
            init_symmetry_score=0.0,
            capacity_utilization=0.0,
            rank_deficiency_ratio=0.0,
            kill_reason=f"T5 error: {e}\n{tb_str}",
            tags=[],
            wall_seconds=wall_seconds,
        )
