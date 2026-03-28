"""Falsifier: Two-stage theory prosecution system."""

from .calibrate import calibrate, load_calibration
from .stage1.orchestrator import run_stage_1
from .stage2.orchestrator import run_stage_2
from .types import (
    Baseline100,
    Calibration,
    CandidatePackage,  # Alias for FalsifierInput
    FalsifierInput,
    FalsifierOutput,
    Feedback,
    KillHypothesis,
    KnowledgeGraph,
    ParentRef,
    S2Result,
    Stage1Result,
    Tag,
    T2Result,
    T3Result,
    T4Result,
    T5Result,
    T7Result,
    Verdict,
)
from .validation import validate_candidate_package

__all__ = [
    # Main entry points
    "run_stage_1",
    "run_stage_2",
    "calibrate",
    "load_calibration",
    "validate_candidate_package",
    # Core types
    "FalsifierInput",
    "FalsifierOutput",
    "CandidatePackage",
    "Calibration",
    "Baseline100",
    "Feedback",
    "KnowledgeGraph",
    "ParentRef",
    # Stage 2 types
    "S2Result",
    "KillHypothesis",
    # Stage 1 result types (Streamlined: T2→T3→{T4,T5}→T7)
    "Stage1Result",
    "T2Result",
    "T3Result",
    "T4Result",
    "T5Result",
    "T7Result",
    # Supporting types
    "Tag",
    "Verdict",
]

__version__ = "0.2.0"
