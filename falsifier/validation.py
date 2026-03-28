from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .types import CandidatePackage, ValidationResult


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_words(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) > 2}


def novelty_score(candidate_text: str, reference_texts: Iterable[str]) -> tuple[float, str]:
    candidate_words = _normalize_words(candidate_text)
    if not candidate_words:
        return 0.0, "candidate text is empty or too short"

    references = [reference for reference in reference_texts if reference.strip()]
    if not references:
        return 1.0, "no prior theories provided"

    max_overlap = 0.0
    closest_reference = ""

    for reference in references:
        reference_words = _normalize_words(reference)
        if not reference_words:
            continue
        overlap = len(candidate_words & reference_words) / len(candidate_words | reference_words)
        if overlap >= max_overlap:
            max_overlap = overlap
            closest_reference = reference

    novelty = max(0.0, 1.0 - max_overlap)
    if closest_reference:
        return novelty, f"max token overlap {max_overlap:.2f} against one prior theory"
    return novelty, "no lexical overlap against prior theories"


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
