"""T6a: Citation extraction and verification.

Extracts numerical citations from what_and_why text and verifies them
against calibration values. Handles patterns like:
- "MLP effective rank ~0.35"
- "gradient norm ratio is 42.5"
- "loss drop of 1.2 within 100 steps"
"""

from __future__ import annotations

import re
from typing import Any


def extract_numerical_citations(what_and_why: str) -> dict[str, float]:
    """Extract numerical citations from theory text.

    Parses what_and_why for patterns like:
    - "effective rank ~0.35" -> {"effective_rank": 0.35}
    - "gradient norm ratio is 42.5" -> {"gradient_norm_ratio": 42.5}
    - "logit max of 8.5" -> {"logit_max": 8.5}

    Args:
        what_and_why: The theory description text

    Returns:
        Dictionary mapping measurement keys to cited values
    """
    citations: dict[str, float] = {}

    if not what_and_why:
        return citations

    # Pattern: <key> <connector> <number>
    # Connectors: ~, is, of, at, =, :, approximately, around, about
    patterns = [
        # effective rank patterns
        (r"(?:mlp\s+)?effective\s+rank\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "mlp_effective_rank_mean"),
        (r"(?:attn|attention)\s+effective\s+rank\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "attn_effective_rank_mean"),

        # gradient norm patterns
        (r"gradient\s+norm\s+ratio\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "gradient_norm_ratio"),
        (r"(?:max\s+)?gradient\s+norm\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "gradient_norm_max"),

        # activation norm patterns
        (r"(?:max\s+)?activation\s+norm\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "activation_norm_max"),

        # logit patterns
        (r"(?:max\s+)?logit\s*(?:max|value)?\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "logit_max"),

        # output entropy patterns
        (r"output\s+entropy\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "output_entropy"),

        # loss patterns
        (r"loss\s+(?:at\s+)?(?:step\s+)?100\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "loss_at_100"),
        (r"loss\s+drop\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "loss_drop"),

        # kurtosis patterns
        (r"kurtosis\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "weight_kurtosis_mean"),

        # condition number patterns
        (r"condition\s+number\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "condition_number"),

        # weight symmetry patterns
        (r"weight\s+symmetry\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "weight_symmetry_mean"),

        # tokens per second patterns
        (r"tokens\s+per\s+second\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.]+)",
         "tokens_per_second"),

        # parameter count patterns
        (r"(?:param|parameter)\s+count\s*(?:~|is|of|at|=|:|approximately|around|about)?\s*([0-9.e+]+)",
         "param_count"),

        # generic catch-all for "key value" patterns with tilde
        (r"([a-z_][a-z_0-9]*)\s*~\s*([0-9.]+)", None),  # Will set key dynamically
    ]

    for pattern, key in patterns:
        matches = re.finditer(pattern, what_and_why, re.IGNORECASE)
        for match in matches:
            if key is None:
                # Dynamic key from pattern group 1
                dynamic_key = match.group(1).lower().replace(" ", "_")
                try:
                    value = float(match.group(2))
                    citations[dynamic_key] = value
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    value = float(match.group(1))
                    citations[key] = value
                except (ValueError, IndexError):
                    continue

    return citations


def lookup_calibration_value(
    calibration: Any,
    key: str,
    default: float | None = None,
) -> float | None:
    """Lookup a value from calibration object by key.

    Supports both attribute access and dictionary-style access.
    Handles nested keys with dot notation (e.g., "baseline_100.loss_drop_mean").

    Args:
        calibration: Calibration dataclass or dict
        key: Key to lookup (supports dot notation for nested access)
        default: Default value if key not found

    Returns:
        The calibration value or default if not found
    """
    if calibration is None:
        return default

    parts = key.split(".")
    current: Any = calibration

    for part in parts:
        if current is None:
            return default

        # Try attribute access first (dataclass)
        if hasattr(current, part):
            current = getattr(current, part)
        # Then try dictionary access
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    # Convert to float if possible
    if isinstance(current, (int, float)):
        return float(current)

    return default


def verify_citations(
    citations: dict[str, float],
    calibration: Any,
    error_threshold: float = 0.5,  # 50% error threshold
) -> tuple[list[str], list[str]]:
    """Verify extracted citations against calibration values.

    Args:
        citations: Dictionary of extracted citations (key -> claimed value)
        calibration: Calibration object to compare against
        error_threshold: Relative error threshold for "bad" citation

    Returns:
        Tuple of (good_citations, bad_citations) where each is a list of
        strings describing the citation and its verification result
    """
    good: list[str] = []
    bad: list[str] = []

    # Map citation keys to calibration lookup paths
    key_mapping: dict[str, list[str]] = {
        "mlp_effective_rank_mean": ["sota_mlp_effective_ranks", "effective_rank_mean"],
        "attn_effective_rank_mean": ["sota_attn_effective_ranks", "effective_rank_mean"],
        "effective_rank_mean": ["effective_rank_mean"],
        "gradient_norm_ratio": ["sota_gradient_norm_ratio"],
        "gradient_norm_max": ["sota_layer_gradient_norms"],
        "activation_norm_max": ["sota_layer_activation_norms"],
        "logit_max": ["sota_init_logit_max"],
        "output_entropy": ["sota_output_entropy"],
        "loss_at_100": ["baseline_100", "loss_at_100_mean"],
        "loss_drop": ["baseline_100", "loss_drop_mean"],
        "weight_kurtosis_mean": ["weight_kurtosis_mean"],
        "kurtosis_mean": ["weight_kurtosis_mean"],
        "condition_number": ["condition_numbers"],
        "weight_symmetry_mean": ["weight_symmetry"],
        "tokens_per_second": ["sota_tokens_per_second", "baseline_100", "tokens_per_second_mean"],
        "param_count": ["sota_param_count"],
    }

    for citation_key, claimed_value in citations.items():
        # Get possible lookup paths
        lookup_paths = key_mapping.get(citation_key, [citation_key])

        calib_value = None
        for path in lookup_paths:
            calib_value = lookup_calibration_value(calibration, path)
            if calib_value is not None:
                break

        if calib_value is None:
            # Cannot verify - skip this citation
            good.append(f"{citation_key}={claimed_value}: unable to verify (no calibration)")
            continue

        # Calculate relative error
        if calib_value == 0:
            relative_error = abs(claimed_value) if claimed_value != 0 else 0.0
        else:
            relative_error = abs(claimed_value - calib_value) / abs(calib_value)

        if relative_error > error_threshold:
            bad.append(
                f"{citation_key}: claimed {claimed_value}, calibration {calib_value}, "
                f"error {relative_error:.1%} > {error_threshold:.0%}"
            )
        else:
            good.append(
                f"{citation_key}={claimed_value}: matches calibration {calib_value} "
                f"(error {relative_error:.1%})"
            )

    return good, bad
