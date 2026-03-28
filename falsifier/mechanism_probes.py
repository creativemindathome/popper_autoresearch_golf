from __future__ import annotations

from pathlib import Path
from typing import Any

from research.probe_library import architecture_profile, load_probe_context


def run_readonly_mechanism_probes(
    repo_root: Path,
    *,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """
    Checkpoint-safe, read-only summaries for later T6-style mechanism claims.
    Does not mutate training state or candidate code.
    """
    _module, state_dict, ctx = load_probe_context(checkpoint_path=checkpoint_path, repo_root=repo_root)
    arch = architecture_profile(checkpoint_path=checkpoint_path, repo_root=repo_root)
    return {
        "schema_version": "1",
        "readonly": True,
        "checkpoint_used": ctx.used_checkpoint,
        "param_tensor_count": len(state_dict),
        "weight_kurtosis_keys": len(arch.get("weight_kurtosis") or {}),
        "effective_rank_keys": len(arch.get("effective_rank") or {}),
    }
