from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


DEFAULT_CODE_BUDGET_BYTES = 200_000
DEFAULT_LIMIT_BYTES = 16_777_216
DEFAULT_TRAIN_TOKENS = 7000 * 8192


def extract_hyperparameters(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Hyperparameters":
            for stmt in node.body:
                # Handle regular assignments (x = ...)
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    literal = _literal_from_env_assign(stmt.value)
                    if literal is not None:
                        values[name] = literal
                # Handle annotated assignments (x: int = ...)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    literal = _literal_from_env_assign(stmt.value)
                    if literal is not None:
                        values[name] = literal
    return values


def _literal_from_env_assign(node: ast.AST) -> Any | None:
    current = node
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Name) and current.args:
        current = current.args[0]
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr == "get" and current.args:
            default_index = 1 if len(current.args) > 1 else None
            if default_index is not None:
                current = current.args[default_index]
    try:
        return ast.literal_eval(current)
    except Exception:
        return None


def extract_model_config(source: str) -> dict[str, Any]:
    values = extract_hyperparameters(source)
    return {
        "vocab_size": int(values.get("vocab_size", 1024)),
        "num_layers": int(values.get("num_layers", 9)),
        "model_dim": int(values.get("model_dim", 512)),
        "num_heads": int(values.get("num_heads", 8)),
        "num_kv_heads": int(values.get("num_kv_heads", 4)),
        "mlp_mult": int(values.get("mlp_mult", 2)),
        "tie_embeddings": bool(values.get("tie_embeddings", True)),
        "iterations": int(values.get("iterations", 20_000)),
        "train_batch_tokens": int(values.get("train_batch_tokens", 524_288)),
        "train_seq_len": int(values.get("train_seq_len", 1024)),
    }


def count_parameters(config: dict[str, Any]) -> dict[str, int]:
    d = int(config["model_dim"])
    v = int(config["vocab_size"])
    layers = int(config["num_layers"])
    kv_heads = int(config["num_kv_heads"])
    heads = int(config["num_heads"])
    mlp_mult = int(config["mlp_mult"])
    head_dim = d // heads
    kv_dim = kv_heads * head_dim
    embedding = v * d
    skip = min(layers // 2, layers - layers // 2) * d
    attn_per_layer = d * d + d * kv_dim + d * kv_dim + d * d
    mlp_hidden = d * mlp_mult
    mlp_per_layer = d * mlp_hidden + mlp_hidden * d
    norm_other = layers * d
    return {
        "embedding": embedding,
        "skip": skip,
        "attention": layers * attn_per_layer,
        "mlp": layers * mlp_per_layer,
        "other": norm_other,
    }


def estimate_compressed_size(param_counts: dict[str, int], bits_per_param: float = 8.0) -> int:
    total_params = sum(int(v) for v in param_counts.values())
    raw_bytes = total_params * bits_per_param / 8.0
    return int(raw_bytes * 0.60)


def estimate_flops(config: dict[str, Any]) -> float:
    d = int(config["model_dim"])
    layers = int(config["num_layers"])
    seq_len = int(config["train_seq_len"])
    batch_tokens = int(config["train_batch_tokens"])
    mlp_mult = int(config["mlp_mult"])
    return batch_tokens * layers * (4 * d * d + 2 * seq_len * d + 2 * d * d * mlp_mult)


def estimate_flops_per_component(config: dict[str, Any]) -> dict[str, float]:
    """Estimate per-component FLOPs for architectural balance analysis.

    Returns dict with:
        - attention: Attention mechanism FLOPs
        - mlp: MLP block FLOPs
        - embedding: Embedding layer FLOPs
        - total: Total FLOPs
        - attn_ratio: attention / total
        - mlp_ratio: mlp / total
        - embed_ratio: embedding / total
    """
    d = int(config["model_dim"])
    layers = int(config["num_layers"])
    seq_len = int(config["train_seq_len"])
    batch_tokens = int(config["train_batch_tokens"])
    mlp_mult = int(config["mlp_mult"])
    v = int(config.get("vocab_size", 1024))

    # Per-token forward pass FLOPs per layer
    # Attention: 4*d^2 (Q,K,V,O projections) + 2*seq_len*d (attention computation)
    attn_per_layer = 4 * d * d + 2 * seq_len * d

    # MLP: 2*d*d*mlp_mult (up and down projections)
    mlp_per_layer = 2 * d * d * mlp_mult

    # Embedding lookups (vocab_size * model_dim per token)
    embed_forward = v * d

    # Multiply by batch tokens and layers
    total_attn = batch_tokens * layers * attn_per_layer
    total_mlp = batch_tokens * layers * mlp_per_layer
    total_embed = batch_tokens * embed_forward

    total = total_attn + total_mlp + total_embed

    return {
        "attention": float(total_attn),
        "mlp": float(total_mlp),
        "embedding": float(total_embed),
        "total": float(total),
        "attn_ratio": total_attn / total if total > 0 else 0.0,
        "mlp_ratio": total_mlp / total if total > 0 else 0.0,
        "embed_ratio": total_embed / total if total > 0 else 0.0,
    }


def estimate_artifact_bytes(
    source: str,
    auxiliary_files: dict[str, str],
    bits_per_param: float = 8.0,
    limit_bytes: int = DEFAULT_LIMIT_BYTES,
) -> tuple[int, int]:
    config = extract_model_config(source)
    param_counts = count_parameters(config)
    compressed = estimate_compressed_size(param_counts, bits_per_param=bits_per_param)
    code_bytes = len(source.encode("utf-8"))
    aux_bytes = sum(len(content.encode("utf-8")) for content in auxiliary_files.values())
    total = compressed + code_bytes + aux_bytes
    return total, limit_bytes - total

