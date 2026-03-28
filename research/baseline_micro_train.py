from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from falsifier.adapters.parameter_golf import instantiate_minimal_model


def run_micro_train_summary(
    repo_root: Path,
    *,
    train_gpt_path: Path | None = None,
    steps: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic CPU micro-train on the minimal smoke-sized model for calibration-lite."""
    train_gpt = train_gpt_path if train_gpt_path is not None else repo_root / "train_gpt.py"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    module, model = instantiate_minimal_model(train_gpt)
    device = torch.device("cpu")
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = 8
    vocab = int(model.tok_emb.num_embeddings)
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, vocab, (1, seq_len), device=device)
        y = torch.roll(x, shifts=-1, dims=1)
        loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - t0
    first, last = losses[0], losses[-1]
    return {
        "steps": steps,
        "seed": seed,
        "device": "cpu",
        "loss_first": first,
        "loss_last": last,
        "loss_drop": first - last,
        "seconds_total": round(elapsed, 6),
        "throughput_steps_per_sec": round(steps / elapsed, 4) if elapsed > 0 else 0.0,
    }
