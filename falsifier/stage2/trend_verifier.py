"""Verify trends from T7 (100-step) extrapolate correctly to 500-step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import Calibration, T7Result


@dataclass
class TrendResult:
    """Result of trend verification."""
    
    broken: bool
    detail: str
    loss_deviation: float = 0.0
    actual_loss_500: float = 0.0


def _get_attr(obj, attr, default=None):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def verify_trends(
    run: Any,  # RunResult or dict
    t7: Any,  # T7Result or dict
    cal: Calibration,
) -> TrendResult:
    """Verify that trends from T7 (100-step) extrapolate correctly to 500-step.
    
    Checks:
    1. Log-linear extrapolation from 100→500
    2. Loss not increasing after step 100
    3. Learning ratio at 500 vs baseline
    4. Gradient stability degradation
    """
    parts: list[str] = []
    broken = False
    
    # Get 100-step trajectory from T7 (handle both object and dict)
    loss_trajectory = _get_attr(t7, 'loss_trajectory', [])
    losses_100 = loss_trajectory[:100] if len(loss_trajectory) >= 100 else loss_trajectory
    
    # Get 500-step actual from run (handle both object and dict)
    run_losses = _get_attr(run, 'losses', [])
    actual_500 = run_losses[499] if len(run_losses) >= 500 else (run_losses[-1] if run_losses else 0.0)
    
    # Log-linear extrapolation from 100→500
    if len(losses_100) >= 10:
        try:
            x = np.log(np.arange(1, len(losses_100) + 1))
            y = np.array(losses_100)
            a, b = np.polyfit(x, y, 1)
            extrap = a * np.log(500) + b
            
            if extrap != 0:
                dev = (actual_500 - extrap) / abs(extrap)
                if dev > 0.30:
                    broken = True
                    parts.append(f"Loss at 500 ({actual_500:.4f}) is {dev:.0%} worse than extrapolated ({extrap:.4f})")
        except Exception:
            pass
    
    # Loss increased after step 100
    if losses_100 and actual_500 > losses_100[-1]:
        broken = True
        parts.append(f"Loss increased after step 100: {losses_100[-1]:.4f} → {actual_500:.4f}")
    
    # Learning ratio at 500 vs baseline
    if cal.baseline_100.loss_drop_500_mean and losses_100:
        loss_drop_500 = losses_100[0] - actual_500
        lr_500 = loss_drop_500 / cal.baseline_100.loss_drop_500_mean
        if lr_500 < 0.30:
            broken = True
            parts.append(f"Learning ratio collapsed to {lr_500:.2f} at step 500")
    
    # Gradient stability degradation
    run_grad_norms = _get_attr(run, 'grad_norms', [])
    if len(run_grad_norms) >= 500:
        early = run_grad_norms[100:200]
        late = run_grad_norms[400:500]
        if early and late:
            e_mean = sum(early) / len(early)
            l_mean = sum(late) / len(late)
            if e_mean > 0 and l_mean > 0:
                e_std = math.sqrt(sum((g - e_mean)**2 for g in early) / len(early))
                l_std = math.sqrt(sum((g - l_mean)**2 for g in late) / len(late))
                e_cv = e_std / e_mean
                l_cv = l_std / l_mean
                if l_cv > e_cv * 3:
                    broken = True
                    parts.append(f"Gradient stability degraded {l_cv/e_cv:.1f}x (CV: {e_cv:.3f} → {l_cv:.3f})")
    
    detail = " | ".join(parts) if parts else "Trends hold within expected bounds."
    
    # Calculate loss deviation
    deviation = 0.0
    if losses_100 and losses_100[0] != 0:
        expected_drop = losses_100[0] - losses_100[-1]
        if expected_drop != 0:
            actual_drop = losses_100[0] - actual_500
            deviation = (expected_drop - actual_drop) / expected_drop
    
    return TrendResult(
        broken=broken,
        detail=detail,
        loss_deviation=deviation,
        actual_loss_500=actual_500,
    )
