from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .precedent import graph_aware_precedent
from .types import CandidatePackage, ValidationResult


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_words(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) > 2}


def novelty_score(candidate_text: str, reference_texts: Iterable[str]) -> tuple[float, str]:
    novelty, reason, _ = graph_aware_precedent(candidate_text, theory_history=(), reference_texts=reference_texts)
    return novelty, reason


def validate_candidate_package(candidate: CandidatePackage) -> ValidationResult:
    reasons: list[str] = []
    details: dict[str, object] = {}

    train_gpt_path = Path(candidate.train_gpt_path)
    details["train_gpt_path"] = str(train_gpt_path)

    if not candidate.theory_id.strip():
        reasons.append("theory_id is required")
    if not candidate.what_and_why.strip():
        reasons.append("what_and_why is required")
    elif len(candidate.what_and_why.split()) < 6:
        reasons.append("what_and_why is too short for a falsifiable theory")
    if not train_gpt_path.exists():
        reasons.append(f"train_gpt.py not found at {train_gpt_path}")
    elif train_gpt_path.name != "train_gpt.py":
        reasons.append("candidate must point at a train_gpt.py file")

    return ValidationResult(ok=not reasons, reasons=reasons, details=details)
