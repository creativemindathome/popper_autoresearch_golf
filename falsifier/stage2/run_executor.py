"""Execute training runs for Stage 2."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..adapters.parameter_golf import instantiate_minimal_model


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


def execute_training_run(
    spec: Any,
    inp: Any,
) -> RunResult:
    """Execute a training run.
    
    Uses MLX if available for speed, falls back to PyTorch minimal-env.
    """
    # Try MLX first if available and this is a real training run
    try:
        from ..adapters.mlx_adapter import mlx_available, run_mlx_training
        
        if mlx_available() and hasattr(inp, "val_data_path") and inp.val_data_path:
            return _execute_mlx_run(spec, inp)
    except Exception:
        pass
    
    # Fall back to PyTorch minimal-env
    return _execute_pytorch_run(spec, inp)


def _execute_pytorch_run(spec: Any, inp: Any) -> RunResult:
    """Execute training run with PyTorch minimal-env."""
    torch.manual_seed(spec.seed)
    
    # Get source
    if spec.source == "theory":
        source = inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else ""
    elif spec.source.startswith("ablation_"):
        from .ablation import build_ablation_source
        source = build_ablation_source(
            inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else "",
            inp.sota_train_gpt if hasattr(inp, "sota_train_gpt") else "",
            inp.config_delta if hasattr(inp, "config_delta") else {},
            spec.source.replace("ablation_", ""),
        )
    elif spec.source == "baseline":
        source = inp.sota_train_gpt if hasattr(inp, "sota_train_gpt") else ""
    else:
        source = inp.proposed_train_gpt if hasattr(inp, "proposed_train_gpt") else ""
    
    # Build model
    model, _ = instantiate_minimal_model(source)
    model.train()
    
    # Setup optimizer
    from ..utils.model_utils import setup_optimizer_from_source
    optimizer = setup_optimizer_from_source(model, source)
    
    # Training loop with random data
    losses, grad_norms, entropies, comp_metrics = [], [], [], {}
    t0 = time.time()
    
    for step in range(spec.steps):
        # Random batch (minimal env fallback)
        batch = torch.randint(0, 64, (1, 8))  # Minimal: batch=1, seq=8, vocab=64
        
        output = model(batch[:, :-1])
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            batch[:, 1:].reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        grad_norms.append(grad_norm)
        
        losses.append(loss.item())
        
        # Entropy every 10 steps
        if step % 10 == 0:
            with torch.no_grad():
                probs = F.softmax(output[:, -1, :], dim=-1)
                ent = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
                entropies.append(ent)
        
        # Dense logging
        if spec.dense_logging and step % 10 == 0:
            cm: dict[str, list[float]] = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    comp = _classify_component(name)
                    cm.setdefault(comp, []).append(p.grad.float().norm().item())
            comp_metrics[step] = {k: sum(v)/len(v) for k, v in cm.items()}
        
        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Check for divergence
        if loss.item() > 100 or math.isnan(loss.item()):
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
    
    elapsed = time.time() - t0
    tokens = spec.steps * 8  # Approximate
    tps = tokens / elapsed if elapsed > 0 else 0.0
    
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


def _execute_mlx_run(spec: Any, inp: Any) -> RunResult:
    """Execute training run with MLX."""
    from ..adapters.mlx_adapter import run_mlx_training, instantiate_mlx_model
    
    # Build model
    model, module = instantiate_mlx_model({})
    
    # Get data path
    data_path = Path(inp.val_data_path) if hasattr(inp, "val_data_path") and inp.val_data_path else None
    
    # Run training
    result = run_mlx_training(
        model=model,
        module=module,
        steps=spec.steps,
        data_path=data_path,
        seed=spec.seed,
    )
    
    # Convert to RunResult format
    return RunResult(
        name=spec.name,
        losses=result.get("loss_trajectory", []),
        grad_norms=result.get("gradient_norms", []),
        entropies=result.get("entropy_trajectory", []),
        component_metrics={},  # MLX adapter doesn't provide per-step component metrics yet
        tokens_per_second=result.get("throughput_steps_per_sec", 0.0),
        diverged=result.get("loss_first", 0) < result.get("loss_last", 0),
        wall_seconds=result.get("wall_seconds", 0.0),
    )


def _classify_component(param_name: str) -> str:
    """Classify parameter by component."""
    name = param_name.lower()
    if "attn" in name or "query" in name or "key" in name or "value" in name:
        return "attn"
    elif "mlp" in name or "fc" in name:
        return "mlp"
    elif "embed" in name or "tok" in name:
        return "embed"
    elif "norm" in name or "ln" in name:
        return "norm"
    else:
        return "other"
