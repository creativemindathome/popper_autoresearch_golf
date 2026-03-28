from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from falsifier.adapters.parameter_golf import instantiate_minimal_model


def _tensor_kurtosis(tensor: torch.Tensor) -> float:
    x = tensor.detach().float().reshape(-1)
    if x.numel() < 4:
        return 0.0
    mean = x.mean()
    centered = x - mean
    var = centered.pow(2).mean()
    if float(var) == 0.0:
        return 0.0
    fourth = centered.pow(4).mean()
    return float(fourth / (var * var))


def _effective_rank(tensor: torch.Tensor) -> float | None:
    if tensor.ndim != 2:
        return None
    x = tensor.detach().float()
    if x.numel() == 0:
        return None
    singular_values = torch.linalg.svdvals(x)
    if singular_values.numel() == 0:
        return None
    total = singular_values.sum()
    if float(total) == 0.0:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * (probs + 1e-12).log()).sum()
    return float(torch.exp(entropy))


def compute_minimal_init_aggregates(
    train_gpt_path: str | Path,
    *,
    env_overrides: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate kurtosis / effective-rank stats over float weights (minimal smoke model)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _module, model = instantiate_minimal_model(train_gpt_path, env_overrides=env_overrides)
    wk: list[float] = []
    er: list[float] = []
    for _name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        wk.append(_tensor_kurtosis(tensor))
        r = _effective_rank(tensor)
        if r is not None:
            er.append(r)
    return {
        "seed": seed,
        "weight_kurtosis_mean": sum(wk) / len(wk) if wk else 0.0,
        "weight_kurtosis_max": max(wk) if wk else 0.0,
        "effective_rank_mean": sum(er) / len(er) if er else 0.0,
        "effective_rank_max": max(er) if er else 0.0,
        "tensor_count": len(model.state_dict()),
    }


def within_band(candidate: float, baseline: float, *, log_span: float = 2.5) -> bool:
    """Pass if log10(candidate/baseline) is within [-log_span, log_span] (when baseline > 0)."""
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return False
    if baseline <= 0.0:
        return candidate <= 1e-6
    ratio = candidate / baseline
    if ratio <= 0.0:
        return False
    return abs(math.log10(ratio)) <= log_span
