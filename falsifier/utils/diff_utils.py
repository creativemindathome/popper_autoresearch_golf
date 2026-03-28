from __future__ import annotations

import difflib

HYPERPARAMETER_NAMES = {
    "num_layers",
    "model_dim",
    "num_heads",
    "num_kv_heads",
    "mlp_mult",
    "vocab_size",
    "train_seq_len",
    "train_batch_tokens",
    "iterations",
    "tie_embeddings",
}


def compute_unified_diff(old: str, new: str) -> list[str]:
    return list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="sota_train_gpt.py",
            tofile="proposed_train_gpt.py",
            lineterm="",
        )
    )


def classify_diff_changes(diff_lines: list[str]) -> set[str]:
    change_types: set[str] = set()
    payload_lines = [
        line[1:]
        for line in diff_lines
        if line and line[0] in {"+", "-"} and not line.startswith(("+++", "---"))
    ]
    for line in payload_lines:
        lowered = line.lower()
        if "class " in lowered or "def " in lowered:
            change_types.add("architecture")
        if (
            "os.environ.get" in lowered
            or "hyperparameters" in lowered
            or any(f"{name} =" in lowered for name in HYPERPARAMETER_NAMES)
        ):
            change_types.add("hyperparameter")
        if any(
            token in lowered
            for token in ("num_layers", "model_dim", "num_heads", "num_kv_heads", "mlp_mult", "vocab_size")
        ):
            change_types.add("hyperparameter")
        if "lr" in lowered or "warmup" in lowered or "momentum" in lowered:
            change_types.add("schedule")
        if "quant" in lowered or "int8" in lowered or "int6" in lowered or "int5" in lowered:
            change_types.add("quantization")
        if "dropout" in lowered or "weight_decay" in lowered:
            change_types.add("training")
    if not change_types and payload_lines:
        change_types.add("other")
    return change_types
