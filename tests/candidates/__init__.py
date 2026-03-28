"""Test candidate theories for Stage 1 gate validation.

This package contains intentionally modified training scripts that should
pass or fail specific Stage 1 gates (T2-T7) of the falsifier.
"""

from pathlib import Path


def get_candidate_path(name: str) -> Path:
    """Get the path to a candidate training script.

    Args:
        name: Candidate name (e.g., "good_student", "overconfident_logits")

    Returns:
        Path to the candidate Python file
    """
    return Path(__file__).parent / f"{name}.py"


__all__ = ["get_candidate_path"]
