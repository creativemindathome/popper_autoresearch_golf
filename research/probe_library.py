from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ProbeContext:
    repo_root: Path
    checkpoint_path: Path | None
    backend: str
    used_checkpoint: bool


def repo_root_from_path(path: Path | None = None) -> Path:
    base = path or Path(__file__)
    return base.resolve().parents[1]


def load_train_module(repo_root: Path):
    train_path = repo_root / "train_gpt.py"
    if not train_path.exists():
        raise FileNotFoundError(f"train_gpt.py not found at {train_path}")
    spec = importlib.util.spec_from_file_location("parameter_golf_train_gpt", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _group_name(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embedding"
    if "skip_weights" in name:
        return "skip"
    if "attn" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


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


def load_probe_context(
    checkpoint_path: str | None = None,
    repo_root: Path | None = None,
) -> tuple[Any, dict[str, Any], ProbeContext]:
    root = repo_root or repo_root_from_path()
    module = load_train_module(root)
    args = module.Hyperparameters()
    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    used_checkpoint = False
    if resolved_checkpoint is not None and resolved_checkpoint.exists():
        payload = torch.load(resolved_checkpoint, map_location="cpu")
        if isinstance(payload, dict) and payload.get("__quant_format__"):
            state_dict = module.dequantize_state_dict_int8(payload)
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise TypeError(f"unsupported checkpoint payload type: {type(payload)!r}")
        model.load_state_dict(state_dict, strict=True)
        used_checkpoint = True

    context = ProbeContext(
        repo_root=root,
        checkpoint_path=resolved_checkpoint,
        backend="torch",
        used_checkpoint=used_checkpoint,
    )
    return module, model.state_dict(), context


def architecture_profile(
    checkpoint_path: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    module, state_dict, context = load_probe_context(checkpoint_path=checkpoint_path, repo_root=repo_root)
    args = module.Hyperparameters()

    total_params = sum(int(t.numel()) for t in state_dict.values())
    group_param_counts: dict[str, int] = {}
    tensor_stats: dict[str, dict[str, Any]] = {}
    kurtosis_by_tensor: dict[str, float] = {}
    effective_rank_by_tensor: dict[str, float] = {}

    for name, tensor in state_dict.items():
        group = _group_name(name)
        group_param_counts[group] = group_param_counts.get(group, 0) + int(tensor.numel())
        tensor_stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": int(tensor.numel()),
            "group": group,
        }
        if tensor.is_floating_point():
            kurtosis_by_tensor[name] = _tensor_kurtosis(tensor)
            rank = _effective_rank(tensor)
            if rank is not None:
                effective_rank_by_tensor[name] = rank

    return {
        "context": {
            "repo_root": str(context.repo_root),
            "checkpoint_path": str(context.checkpoint_path) if context.checkpoint_path else None,
            "backend": context.backend,
            "used_checkpoint": context.used_checkpoint,
        },
        "hyperparameters": {
            "vocab_size": args.vocab_size,
            "num_layers": args.num_layers,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_mult": args.mlp_mult,
            "tie_embeddings": args.tie_embeddings,
            "iterations": args.iterations,
            "train_batch_tokens": args.train_batch_tokens,
            "train_seq_len": args.train_seq_len,
        },
        "param_count_total": total_params,
        "param_count_by_group": group_param_counts,
        "tensor_stats": tensor_stats,
        "weight_kurtosis": kurtosis_by_tensor,
        "effective_rank": effective_rank_by_tensor,
    }


def quantization_profile(
    checkpoint_path: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    module, state_dict, context = load_probe_context(checkpoint_path=checkpoint_path, repo_root=repo_root)
    quantized, quant_stats = module.quantize_state_dict_int8(state_dict)
    restored = module.dequantize_state_dict_int8(quantized)

    mse_by_tensor: dict[str, float] = {}
    mse_by_group: dict[str, list[float]] = {}
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        diff = tensor.detach().float() - restored[name].detach().float()
        mse = float(diff.pow(2).mean())
        mse_by_tensor[name] = mse
        group = _group_name(name)
        mse_by_group.setdefault(group, []).append(mse)

    avg_mse_by_group = {
        group: sum(values) / len(values)
        for group, values in mse_by_group.items()
        if values
    }

    return {
        "context": asdict(context),
        "quant_stats": quant_stats,
        "quantization_mse": {
            "by_tensor": mse_by_tensor,
            "by_group": avg_mse_by_group,
        },
    }


SUPPORTED_EXPERIMENTS = {
    "architecture_profile": architecture_profile,
    "quantization_profile": quantization_profile,
}


def run_experiment(name: str, checkpoint_path: str | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    if name not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"unsupported experiment {name!r}; expected one of {sorted(SUPPORTED_EXPERIMENTS)}")
    return SUPPORTED_EXPERIMENTS[name](checkpoint_path=checkpoint_path, repo_root=repo_root)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
