"""Optimize training runs by sharing across experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .experiment import ExperimentSpec


@dataclass
class RunSpec:
    """A unique training run specification."""
    
    name: str
    source: str  # "theory", "ablation_{change}", "baseline"
    steps: int
    seed: int = 42
    dense_logging: bool = False
    component_hooks: list[str] = field(default_factory=list)


@dataclass
class RunPlan:
    """Optimized plan of unique runs."""
    
    unique_runs: list[RunSpec]
    experiment_to_run: dict[str, str]  # Maps experiment name to run name


def optimize_run_plan(experiments: list[ExperimentSpec], inp: Any) -> RunPlan:
    """Minimize training runs by sharing across experiments.
    
    The theory run is shared across all experiments. Ablation runs are per change.
    """
    runs: dict[str, RunSpec] = {}
    experiment_to_run: dict[str, str] = {}
    
    for exp in experiments:
        step = exp.steps
        
        # Theory run (shared across all experiments)
        if exp.source == "theory":
            if "theory_run" not in runs:
                runs["theory_run"] = RunSpec(
                    name="theory_run",
                    source="theory",
                    steps=step,
                    dense_logging=True,
                    component_hooks=exp.component_hooks or [],
                )
            else:
                # Extend steps if needed
                runs["theory_run"].steps = max(runs["theory_run"].steps, step)
            experiment_to_run[exp.name] = "theory_run"
        
        # Ablation runs (unique per ablated change)
        elif exp.source.startswith("ablation_"):
            change = exp.source.replace("ablation_", "")
            run_name = f"ablation_{change}"
            
            if run_name not in runs:
                runs[run_name] = RunSpec(
                    name=run_name,
                    source=exp.source,
                    steps=step,
                    dense_logging=True,
                    component_hooks=exp.component_hooks or [],
                )
            else:
                runs[run_name].steps = max(runs[run_name].steps, step)
            experiment_to_run[exp.name] = run_name
        
        # Baseline run
        elif exp.source == "baseline":
            if "baseline" not in runs:
                runs["baseline"] = RunSpec(
                    name="baseline",
                    source="baseline",
                    steps=step,
                    dense_logging=True,
                )
            experiment_to_run[exp.name] = "baseline"
    
    # Ensure at least 500 steps for theory run (trend verification)
    if "theory_run" in runs:
        runs["theory_run"].steps = max(runs["theory_run"].steps, 500)
    
    return RunPlan(
        unique_runs=list(runs.values()),
        experiment_to_run=experiment_to_run,
    )
