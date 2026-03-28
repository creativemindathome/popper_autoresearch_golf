"""T6c: Sensitivity Probes for causal validation.

Builds probes to measure sensitivity of model components to perturbations.
Provides stubs for full implementation including:
- Sensitivity probe construction
- Perturbation application
- Sensitivity measurement
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def build_sensitivity_probe(
    model: Any,
    target_component: str,
    probe_type: str = "gradient",
) -> dict[str, Any]:
    """Build a sensitivity probe for a model component.

    STUB: This is a placeholder for the full implementation.
    The full implementation would construct hooks or probes
    that can measure component sensitivity to various perturbations.

    Args:
        model: The model to probe
        target_component: Path to component (e.g., "blocks.0.attn.q_proj")
        probe_type: Type of probe ("gradient", "activation", "attention")

    Returns:
        Probe configuration dict
    """
    # STUB implementation
    return {
        "probe_built": False,
        "target_component": target_component,
        "probe_type": probe_type,
        "hook_handles": [],
        "note": "build_sensitivity_probe is a stub - full implementation requires hook framework",
    }


def apply_perturbation(
    tensor: torch.Tensor,
    perturbation_type: str = "gaussian",
    magnitude: float = 0.1,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply a perturbation to a tensor for sensitivity testing.

    Args:
        tensor: Input tensor to perturb
        perturbation_type: Type of perturbation ("gaussian", "uniform", "dropout")
        magnitude: Magnitude of perturbation (relative to tensor norm)
        seed: Random seed for reproducibility

    Returns:
        Perturbed tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    if perturbation_type == "gaussian":
        noise = torch.randn_like(tensor) * magnitude * tensor.std()
        return tensor + noise
    elif perturbation_type == "uniform":
        noise = (torch.rand_like(tensor) - 0.5) * 2 * magnitude * tensor.abs().max()
        return tensor + noise
    elif perturbation_type == "dropout":
        mask = torch.rand_like(tensor) > magnitude
        return tensor * mask
    elif perturbation_type == "sparse":
        # Sparse perturbation: only perturb some elements
        k = max(1, int(tensor.numel() * magnitude))
        flat = tensor.reshape(-1)
        indices = torch.randperm(tensor.numel())[:k]
        perturbed = flat.clone()
        noise = torch.randn(k) * tensor.std()
        perturbed[indices] += noise
        return perturbed.reshape(tensor.shape)
    else:
        # Unknown perturbation type, return unchanged
        return tensor


def measure_sensitivity(
    model: Any,
    input_data: Any,
    probe: dict[str, Any],
    perturbation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    num_perturbations: int = 10,
) -> dict[str, Any]:
    """Measure component sensitivity via perturbation analysis.

    STUB: This is a placeholder for the full implementation.

    Args:
        model: The model to measure
        input_data: Input data for forward pass
        probe: Probe configuration from build_sensitivity_probe
        perturbation_fn: Function to apply perturbations
        num_perturbations: Number of perturbation trials

    Returns:
        Sensitivity measurement dict
    """
    # STUB implementation
    return {
        "measured": False,
        "sensitivity_score": None,
        "output_variance": None,
        "num_perturbations": num_perturbations,
        "note": "measure_sensitivity is a stub - full implementation requires probe integration",
    }


def compare_sensitivity(
    candidate_sensitivity: dict[str, float],
    baseline_sensitivity: dict[str, float],
    tolerance: float = 0.5,
) -> dict[str, Any]:
    """Compare candidate sensitivity to baseline.

    Args:
        candidate_sensitivity: Component -> sensitivity score for candidate
        baseline_sensitivity: Component -> sensitivity score for baseline
        tolerance: Acceptable ratio difference

    Returns:
        Comparison result with divergences flagged
    """
    divergences: list[dict[str, Any]] = []

    all_components = set(candidate_sensitivity.keys()) | set(baseline_sensitivity.keys())

    for component in all_components:
        candidate_score = candidate_sensitivity.get(component)
        baseline_score = baseline_sensitivity.get(component)

        if candidate_score is None or baseline_score is None:
            divergences.append({
                "component": component,
                "candidate": candidate_score,
                "baseline": baseline_score,
                "issue": "missing_measurement",
            })
            continue

        if baseline_score == 0:
            ratio = float("inf") if candidate_score > 0 else 1.0
        else:
            ratio = candidate_score / baseline_score

        if ratio > 1 + tolerance or ratio < 1 - tolerance:
            divergences.append({
                "component": component,
                "candidate": candidate_score,
                "baseline": baseline_score,
                "ratio": ratio,
                "issue": "sensitivity_divergence",
            })

    return {
        "divergences": divergences,
        "divergence_count": len(divergences),
        "within_tolerance": len(divergences) == 0,
    }


def gradient_sensitivity_probe(
    model: Any,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    """Measure gradient-based sensitivity per parameter.

    STUB: This is a placeholder for the full implementation.

    Args:
        model: The model
        input_ids: Input token IDs
        target_ids: Target token IDs

    Returns:
        Dict mapping parameter names to sensitivity scores
    """
    # STUB implementation
    return {
        "note": "gradient_sensitivity_probe is a stub",
    }
