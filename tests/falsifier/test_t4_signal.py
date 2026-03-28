"""Tests for T4 Signal Propagation gate."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from falsifier.types import FalsifierInput, T4Result, Calibration, TestStatus, Tag
from falsifier.utils.framework_adapter import (
    TORCH_AVAILABLE,
    MLX_AVAILABLE,
)

# Skip tests if frameworks not available
pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
]


def test_t4_result_dataclass():
    """Test T4Result dataclass structure."""
    result = T4Result(
        status="PASS",
        test_id="T4",
        layer_activation_norms={"layer_0_attn": 1.5},
        layer_gradient_norms={"layer_0_attn": 0.5},
        gradient_norm_ratio=2.0,
        gradient_max_layer="layer_1_attn",
        gradient_min_layer="layer_0_attn",
        output_entropy=2.5,
        entropy_ratio=0.5,
        loss_at_init=3.0,
        gradient_flow_health=0.8,
        dead_neuron_ratio=0.05,
        signal_to_noise_ratio=0.7,
    )

    assert result.status == "PASS"
    assert result.test_id == "T4"
    assert result.gradient_norm_ratio == 2.0
    assert result.output_entropy == 2.5


def test_t4_framework_adapter_imports():
    """Test that T4-specific framework adapter functions are available."""
    from falsifier.utils.framework_adapter import (
        compute_gradient_norms_pytorch,
        compute_output_entropy_pytorch,
        compute_activation_norm_pytorch,
        compute_activation_stats_pytorch,
        create_activation_hook_pytorch,
        make_random_input_pytorch,
    )

    # Functions should be importable
    assert callable(compute_gradient_norms_pytorch)
    assert callable(compute_output_entropy_pytorch)
    assert callable(compute_activation_norm_pytorch)
    assert callable(compute_activation_stats_pytorch)
    assert callable(create_activation_hook_pytorch)
    assert callable(make_random_input_pytorch)


def test_make_random_input_pytorch():
    """Test PyTorch random input generation."""
    from falsifier.utils.framework_adapter import make_random_input_pytorch
    import torch

    vocab_size = 100
    seq_len = 8

    input_ids, target_ids = make_random_input_pytorch(vocab_size, seq_len)

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(target_ids, torch.Tensor)
    assert input_ids.shape == (1, seq_len)
    assert target_ids.shape == (1, seq_len)
    assert input_ids.dtype == torch.long

    # Targets should be rolled version of inputs
    expected_targets = torch.roll(input_ids, shifts=-1, dims=1)
    assert torch.equal(target_ids, expected_targets)


def test_compute_activation_stats_pytorch():
    """Test activation statistics computation for PyTorch."""
    from falsifier.utils.framework_adapter import compute_activation_stats_pytorch
    import torch

    # Create tensor with known values
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    stats = compute_activation_stats_pytorch(tensor)

    assert "mean" in stats
    assert "std" in stats
    assert "snr" in stats
    assert "dead_ratio" in stats

    # Mean should be 3.5
    assert abs(stats["mean"] - 3.5) < 0.01

    # All values are > 0.01, so dead_ratio should be 0
    assert stats["dead_ratio"] == 0.0


def test_compute_activation_norm_pytorch():
    """Test activation norm computation for PyTorch."""
    from falsifier.utils.framework_adapter import compute_activation_norm_pytorch
    import torch

    # Create tensor with known norm: sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    tensor = torch.tensor([1.0, 2.0, 3.0])

    norm = compute_activation_norm_pytorch(tensor)
    expected_norm = math.sqrt(14)

    assert abs(norm - expected_norm) < 0.001


def test_compute_output_entropy_pytorch():
    """Test output entropy computation for PyTorch."""
    from falsifier.utils.framework_adapter import compute_output_entropy_pytorch
    import torch

    # Create logits that should give uniform distribution
    vocab_size = 10
    logits = torch.zeros(2, vocab_size)  # All zeros = uniform

    entropy = compute_output_entropy_pytorch(logits)

    # Uniform distribution entropy = log(vocab_size)
    expected_entropy = math.log(vocab_size)
    assert abs(entropy - expected_entropy) < 0.01


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
def test_mlx_framework_adapter_imports():
    """Test that MLX-specific framework adapter functions are available."""
    from falsifier.utils.framework_adapter import (
        compute_gradient_norms_mlx,
        compute_output_entropy_mlx,
        compute_activation_norm_mlx,
        compute_activation_stats_mlx,
        make_random_input_mlx,
    )

    assert callable(compute_gradient_norms_mlx)
    assert callable(compute_output_entropy_mlx)
    assert callable(compute_activation_norm_mlx)
    assert callable(compute_activation_stats_mlx)
    assert callable(make_random_input_mlx)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
def test_make_random_input_mlx():
    """Test MLX random input generation."""
    from falsifier.utils.framework_adapter import make_random_input_mlx
    import mlx.core as mx

    vocab_size = 100
    seq_len = 8

    input_ids, target_ids = make_random_input_mlx(vocab_size, seq_len)

    assert isinstance(input_ids, mx.array)
    assert isinstance(target_ids, mx.array)
    assert input_ids.shape == (1, seq_len)
    assert target_ids.shape == (1, seq_len)


def test_build_t4_result_evaluations():
    """Test T4 result evaluation logic."""
    from falsifier.stage1.t4_signal import _build_t4_result

    # Test with extreme gradient ratio (should trigger FATAL)
    result = _build_t4_result(
        status="PASS",
        layer_grad_norms={
            "layer_0": 10000.0,
            "layer_1": 1.0,
        },
        gradient_norm_ratio=10000.0,
        max_layer="layer_0",
        min_layer="layer_1",
        max_grad=10000.0,
        min_grad=1.0,
        entropy=2.5,
        entropy_ratio=0.5,
        loss_at_init=3.0,
        activation_norms={"layer_0": 50.0},
        activation_stats={
            "layer_0": {
                "mean": 0.5,
                "std": 1.0,
                "snr": 0.5,
                "dead_ratio": 0.05,
                "tensor_shape": [1, 8, 64],
            }
        },
        calibration=None,
        wall_seconds=1.0,
    )

    # Should be FATAL due to high gradient ratio
    assert result.status == "FAIL_FATAL"
    assert result.kill_reason is not None
    assert "ratio" in result.kill_reason.lower()


def test_build_t4_result_tags():
    """Test T4 result tag generation."""
    from falsifier.stage1.t4_signal import _build_t4_result

    # Test with low entropy (should trigger tag)
    result = _build_t4_result(
        status="PASS",
        layer_grad_norms={"layer_0": 1.0, "layer_1": 2.0},
        gradient_norm_ratio=2.0,
        max_layer="layer_1",
        min_layer="layer_0",
        max_grad=2.0,
        min_grad=1.0,
        entropy=0.5,
        entropy_ratio=0.1,  # Very low
        loss_at_init=3.0,
        activation_norms={"layer_0": 10.0},
        activation_stats={
            "layer_0": {
                "mean": 0.5,
                "std": 1.0,
                "snr": 0.5,
                "dead_ratio": 0.05,
                "tensor_shape": [1, 8, 64],
            }
        },
        calibration=None,
        wall_seconds=1.0,
    )

    # Should have low entropy tag
    tag_ids = [t.tag_id for t in result.tags]
    assert "T4_low_output_entropy" in tag_ids


def test_t4_error_result():
    """Test T4 error result building."""
    from falsifier.stage1.t4_signal import _build_error_result

    result = _build_error_result("Test error", 1.5)

    assert result.status == "FAIL_FATAL"
    assert result.test_id == "T4"
    assert result.kill_reason == "Test error"
    assert result.wall_seconds == 1.5
    assert result.layer_gradient_norms == {}


def test_t4_run_function_routes_to_pytorch():
    """Test that run_t4 routes to PyTorch implementation for .py files."""
    from falsifier.stage1.t4_signal import run_t4

    # Mock input with .py extension (PyTorch)
    inp = MagicMock(spec=FalsifierInput)
    inp.proposed_train_gpt = "some/path/train_gpt.py"
    inp.train_gpt_path = ""
    inp.calibration = None

    # Should return a result (even if error due to no actual model)
    with patch("falsifier.stage1.t4_signal.run_t4_pytorch") as mock_pytorch:
        mock_pytorch.return_value = T4Result(status="PASS", test_id="T4")
        result = run_t4(inp)
        mock_pytorch.assert_called_once()


def test_t4_run_function_routes_to_mlx():
    """Test that run_t4 routes to MLX implementation for _mlx.py files."""
    from falsifier.stage1.t4_signal import run_t4

    # Mock input with _mlx.py extension
    inp = MagicMock(spec=FalsifierInput)
    inp.proposed_train_gpt = "some/path/train_gpt_mlx.py"
    inp.train_gpt_path = ""
    inp.calibration = None

    # Should route to MLX
    with patch("falsifier.stage1.t4_signal.run_t4_mlx") as mock_mlx:
        mock_mlx.return_value = T4Result(status="PASS", test_id="T4")
        result = run_t4(inp)
        mock_mlx.assert_called_once()
