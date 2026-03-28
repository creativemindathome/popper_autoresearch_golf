from __future__ import annotations

from pathlib import Path

from .types import CandidatePackage, ValidationResult


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
    elif not train_gpt_path.suffix == ".py":
        reasons.append("candidate must be a Python (.py) file")

    return ValidationResult(ok=not reasons, reasons=reasons, details=details)
