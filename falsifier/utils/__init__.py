"""Shared helpers for falsifier tests."""

from falsifier.utils.framework_adapter import (
    check_finite,
    compute_loss,
    create_random_input,
    create_rolled_targets,
    detect_framework,
    get_model_info,
    get_named_parameters,
    get_param_count,
    get_trainable_param_count,
    has_inf,
    has_nan,
    model_eval,
    model_forward,
    model_train,
    requires_grad_context,
    tensor_mean,
    tensor_std,
)

__all__ = [
    "check_finite",
    "compute_loss",
    "create_random_input",
    "create_rolled_targets",
    "detect_framework",
    "get_model_info",
    "get_named_parameters",
    "get_param_count",
    "get_trainable_param_count",
    "has_inf",
    "has_nan",
    "model_eval",
    "model_forward",
    "model_train",
    "requires_grad_context",
    "tensor_mean",
    "tensor_std",
]

