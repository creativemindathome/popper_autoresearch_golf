"""Checkpoint subpackage for T6 mechanism validation.

Provides modules for:
- T6a: Citation extraction and verification (citations.py)
- T6b: Mechanism claim extraction and measurement (mechanism.py)
- T6c: Sensitivity probe building and measurement (sensitivity.py)
- T6d: Trend interpolation and extrapolation (interpolation.py)
"""

from falsifier.checkpoint.citations import (
    extract_numerical_citations,
    lookup_calibration_value,
    verify_citations,
)
from falsifier.checkpoint.interpolation import (
    build_interpolation_context,
    extrapolate_trend,
    find_similar_configs,
    validate_proposal_against_trend,
)
from falsifier.checkpoint.mechanism import (
    ablate_component,
    extract_mechanism_claims,
    intervene_on_pathway,
    measure_claim,
)
from falsifier.checkpoint.sensitivity import (
    apply_perturbation,
    build_sensitivity_probe,
    compare_sensitivity,
    gradient_sensitivity_probe,
    measure_sensitivity,
)

__all__ = [
    # citations.py
    "extract_numerical_citations",
    "lookup_calibration_value",
    "verify_citations",
    # mechanism.py
    "extract_mechanism_claims",
    "measure_claim",
    "ablate_component",
    "intervene_on_pathway",
    # sensitivity.py
    "build_sensitivity_probe",
    "measure_sensitivity",
    "apply_perturbation",
    "compare_sensitivity",
    "gradient_sensitivity_probe",
    # interpolation.py
    "extrapolate_trend",
    "validate_proposal_against_trend",
    "find_similar_configs",
    "build_interpolation_context",
]
