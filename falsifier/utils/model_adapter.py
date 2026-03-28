"""Model adapter for unified PyTorch and MLX model interface.

Provides a UnifiedModel class that wraps any model and provides a standardized
interface for forward passes, regardless of whether the model:
- Returns logits (standard forward(x))
- Returns loss (forward(x, y) like train_gpt.py)
- Is PyTorch or MLX based
"""

from __future__ import annotations

import inspect
from typing import Any, Callable
from enum import Enum

# Import from existing framework adapter
from falsifier.utils.framework_adapter import (
    detect_framework,
    TORCH_AVAILABLE,
    MLX_AVAILABLE,
    compute_loss,
    model_train,
    model_eval,
)

# Framework availability checks
try:
    import torch
    import torch.nn as nn_torch
    import torch.nn.functional as F_torch
except ImportError:
    torch = None  # type: ignore
    nn_torch = None  # type: ignore
    F_torch = None  # type: ignore

try:
    import mlx.core as mx
    import mlx.nn as nn_mlx
except ImportError:
    mx = None  # type: ignore
    nn_mlx = None  # type: ignore


class ModelSignatureType(Enum):
    """Enum for different model signature patterns."""

    LOGITS_ONLY = "logits_only"  # forward(x) returns logits
    LOSS_RETURNING = "loss_returning"  # forward(x, y) returns loss
    DUAL_MODE = "dual_mode"  # forward(x, y=None) returns logits or loss
    MLX_STANDARD = "mlx_standard"  # MLX model with separate loss method
    UNKNOWN = "unknown"


class UnifiedModel:
    """Unified wrapper for PyTorch and MLX models.

    Provides a consistent interface for:
    - forward_logits(input_ids) → returns logits
    - forward_loss(input_ids, targets) → returns loss
    - train() and eval() modes

    Automatically detects the model's calling convention and adapts accordingly.

    Examples:
        # Wrap any model
        model = UnifiedModel.wrap(my_pytorch_model)

        # Get logits
        logits = model.forward_logits(input_ids)

        # Get loss
        loss = model.forward_loss(input_ids, targets)

        # Set training mode
        model.train()
        loss = model.forward_loss(input_ids, targets)

        # Set eval mode
        model.eval()
        logits = model.forward_logits(input_ids)
    """

    def __init__(
        self,
        model: Any,
        framework: str | None = None,
        signature_type: ModelSignatureType | None = None,
    ):
        """Initialize the unified model wrapper.

        Args:
            model: The underlying model (PyTorch or MLX)
            framework: "pytorch" or "mlx" (auto-detected if None)
            signature_type: The model signature type (auto-detected if None)
        """
        self._model = model
        self._framework = framework or detect_framework(model)
        self._signature_type = signature_type or self._detect_signature()

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        return self._model

    @property
    def framework(self) -> str:
        """Get the framework type ("pytorch" or "mlx")."""
        return self._framework

    @property
    def signature_type(self) -> ModelSignatureType:
        """Get the detected signature type."""
        return self._signature_type

    def _get_forward_callable(self) -> Callable:
        """Get the forward/__call__ method from the model."""
        # Try __call__ first (standard for both PyTorch and MLX)
        if hasattr(self._model, "__call__") and callable(self._model.__call__):
            return self._model.__call__
        # Fall back to forward
        if hasattr(self._model, "forward") and callable(self._model.forward):
            return self._model.forward
        raise ValueError(f"Model has no callable __call__ or forward method: {type(self._model)}")

    def _detect_signature(self) -> ModelSignatureType:
        """Detect the model's calling signature by inspection and testing."""
        forward_fn = self._get_forward_callable()
        sig = inspect.signature(forward_fn)
        params = list(sig.parameters.items())

        # Check for MLX models with separate loss method
        if self._framework == "mlx" and hasattr(self._model, "loss"):
            return ModelSignatureType.MLX_STANDARD

        # Analyze parameter count
        # self is always present, so we look at additional parameters
        param_names = [p[0] for p in params if p[0] != "self"]

        # Check for return_logits parameter (dual mode)
        if "return_logits" in param_names:
            return ModelSignatureType.DUAL_MODE

        # Check if second parameter looks like targets/labels
        if len(param_names) >= 2:
            second_param_name = param_names[1].lower()
            if any(x in second_param_name for x in ["target", "label", "y"]):
                return ModelSignatureType.LOSS_RETURNING

        # Test with dummy input to confirm signature
        test_result = self._test_signature_with_dummy_input()
        if test_result != ModelSignatureType.UNKNOWN:
            return test_result

        # Default to logits-only (most common case)
        return ModelSignatureType.LOGITS_ONLY

    def _test_signature_with_dummy_input(self) -> ModelSignatureType:
        """Test the model signature by trying actual calls with dummy data.

        This is a fallback when inspection is ambiguous.
        """
        # Create minimal dummy input
        if self._framework == "pytorch" and torch is not None:
            dummy_input = torch.zeros((1, 4), dtype=torch.long)
            dummy_targets = torch.zeros((1, 4), dtype=torch.long)
        elif self._framework == "mlx" and mx is not None:
            dummy_input = mx.zeros((1, 4), dtype=mx.int32)
            dummy_targets = mx.zeros((1, 4), dtype=mx.int32)
        else:
            return ModelSignatureType.UNKNOWN

        # Test 1: Try calling with just input (logits-only check)
        try:
            output = self._model(dummy_input)
            # If we get here, model accepts single arg
            # Check if output looks like logits (multi-dimensional)
            if hasattr(output, "shape") and len(output.shape) >= 2:
                # Looks like logits
                return ModelSignatureType.LOGITS_ONLY
        except Exception:
            pass

        # Test 2: Try calling with input and targets (loss-returning check)
        try:
            output = self._model(dummy_input, dummy_targets)
            if hasattr(output, "shape"):
                # Check if output is scalar (loss) or multi-dim (logits)
                if len(output.shape) == 0 or (len(output.shape) == 1 and output.shape[0] == 1):
                    return ModelSignatureType.LOSS_RETURNING
        except Exception:
            pass

        # Default assumption
        return ModelSignatureType.LOGITS_ONLY

    def forward_logits(self, input_ids: Any) -> Any:
        """Run forward pass to get logits.

        Args:
            input_ids: Input token IDs (framework-specific tensor/array)

        Returns:
            Logits tensor/array
        """
        if self._signature_type == ModelSignatureType.LOGITS_ONLY:
            # Standard logits-only model
            return self._model(input_ids)

        elif self._signature_type == ModelSignatureType.LOSS_RETURNING:
            # Loss-returning model - need to call without targets
            # Some models might support return_logits=True
            try:
                return self._model(input_ids, return_logits=True)
            except TypeError:
                # Model doesn't support return_logits, can't get logits directly
                raise RuntimeError(
                    "Loss-returning model does not support return_logits=True. "
                    "Cannot extract logits from this model."
                )

        elif self._signature_type == ModelSignatureType.DUAL_MODE:
            # Dual mode model - use return_logits flag
            return self._model(input_ids, None, return_logits=True)

        elif self._signature_type == ModelSignatureType.MLX_STANDARD:
            # MLX standard model - just call directly
            return self._model(input_ids)

        else:
            # Unknown signature - try direct call
            return self._model(input_ids)

    def forward_loss(self, input_ids: Any, targets: Any) -> Any:
        """Run forward pass to get loss.

        Args:
            input_ids: Input token IDs (framework-specific tensor/array)
            targets: Target token IDs (framework-specific tensor/array)

        Returns:
            Loss scalar tensor/array
        """
        if self._signature_type == ModelSignatureType.LOSS_RETURNING:
            # Loss-returning model - call with both args
            return self._model(input_ids, targets)

        elif self._signature_type == ModelSignatureType.LOGITS_ONLY:
            # Logits-only model - need to compute loss ourselves
            logits = self._model(input_ids)
            return compute_loss(logits, targets)

        elif self._signature_type == ModelSignatureType.DUAL_MODE:
            # Dual mode - call with targets
            return self._model(input_ids, targets)

        elif self._signature_type == ModelSignatureType.MLX_STANDARD:
            # MLX standard - use loss method
            return self._model.loss(input_ids, targets)

        else:
            # Unknown signature - try direct call with both args, fallback to manual loss
            try:
                output = self._model(input_ids, targets)
                # Check if output looks like loss (scalar)
                if hasattr(output, "shape") and (len(output.shape) == 0 or output.shape[-1] == 1):
                    return output
            except Exception:
                pass

            # Fallback: compute loss from logits
            logits = self._model(input_ids)
            return compute_loss(logits, targets)

    def train(self) -> None:
        """Set model to training mode."""
        model_train(self._model)

    def eval(self) -> None:
        """Set model to evaluation mode."""
        model_eval(self._model)

    def __call__(self, input_ids: Any, targets: Any | None = None) -> Any:
        """Direct call - delegates to forward_logits or forward_loss based on targets.

        Args:
            input_ids: Input token IDs
            targets: Optional target token IDs (if provided, returns loss)

        Returns:
            Logits if targets is None, otherwise loss
        """
        if targets is not None:
            return self.forward_loss(input_ids, targets)
        return self.forward_logits(input_ids)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the wrapped model."""
        from falsifier.utils.framework_adapter import get_model_info, get_param_count

        info = get_model_info(self._model)
        info["signature_type"] = self._signature_type.value
        info["param_count"] = get_param_count(self._model)
        return info

    @classmethod
    def wrap(cls, model: Any, framework: str | None = None) -> "UnifiedModel":
        """Wrap a model in a UnifiedModel interface.

        This is a convenience class method that creates a UnifiedModel wrapper.

        Args:
            model: The model to wrap (PyTorch or MLX)
            framework: "pytorch" or "mlx" (auto-detected if None)

        Returns:
            UnifiedModel instance
        """
        return cls(model, framework=framework)


def detect_framework(model: Any) -> str:
    """Detect whether a model is PyTorch or MLX-based.

    This is a re-export from framework_adapter for convenience.

    Args:
        model: The model instance to check

    Returns:
        "pytorch" if PyTorch model, "mlx" if MLX model

    Raises:
        ValueError: If model type cannot be determined
    """
    # Call the imported function from framework_adapter, not self
    from falsifier.utils.framework_adapter import detect_framework as _detect_framework_impl
    return _detect_framework_impl(model)


def wrap_model(model: Any, framework: str | None = None) -> UnifiedModel:
    """Wrap a model in a UnifiedModel interface.

    Args:
        model: The model to wrap (PyTorch or MLX)
        framework: "pytorch" or "mlx" (auto-detected if None)

    Returns:
        UnifiedModel instance

    Examples:
        # Auto-detect framework
        unified = wrap_model(my_model)

        # Explicit framework
        unified = wrap_model(my_model, framework="pytorch")

        # Use unified interface
        logits = unified.forward_logits(input_ids)
        loss = unified.forward_loss(input_ids, targets)
    """
    return UnifiedModel(model, framework=framework)


def is_unified(model: Any) -> bool:
    """Check if a model is already a UnifiedModel instance.

    Args:
        model: The model to check

    Returns:
        True if model is a UnifiedModel, False otherwise
    """
    return isinstance(model, UnifiedModel)


def ensure_unified(model: Any, framework: str | None = None) -> UnifiedModel:
    """Ensure a model is wrapped as UnifiedModel.

    If already a UnifiedModel, returns as-is. Otherwise wraps it.

    Args:
        model: The model (UnifiedModel or raw model)
        framework: "pytorch" or "mlx" (auto-detected if None)

    Returns:
        UnifiedModel instance
    """
    if is_unified(model):
        return model
    return wrap_model(model, framework=framework)
