"""T6d: Interpolation for trend extrapolation.

Uses historical graph data to extrapolate trends and validate
that proposed changes follow expected patterns.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extrapolate_trend(
    history: list[tuple[float, float]],
    proposed_value: float,
    method: str = "linear",
) -> dict[str, Any]:
    """Extrapolate trend from historical data and evaluate proposal.

    Given historical (config_value, measured_result) pairs, extrapolates
    the expected result for the proposed value and compares to predictions.

    Args:
        history: List of (config_value, measured_result) tuples
        proposed_value: The proposed config value to evaluate
        method: Interpolation method ("linear", "polynomial", "log")

    Returns:
        Extrapolation result dict:
        - expected_result: extrapolated value
        - confidence: confidence in extrapolation (0-1)
        - within_expected: bool if prediction matches expectation
        - trend_direction: "increasing", "decreasing", "flat"
    """
    if not history:
        return {
            "expected_result": None,
            "confidence": 0.0,
            "within_expected": None,
            "trend_direction": "unknown",
            "note": "No history available for interpolation",
        }

    # Sort by config value
    history = sorted(history, key=lambda x: x[0])
    x_vals = np.array([h[0] for h in history])
    y_vals = np.array([h[1] for h in history])

    # Determine trend direction
    if len(x_vals) >= 2:
        # Simple linear regression for trend
        slope = np.polyfit(x_vals, y_vals, 1)[0]
        if abs(slope) < 1e-6:
            trend_direction = "flat"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
    else:
        trend_direction = "unknown"

    # Extrapolate based on method
    if method == "linear" and len(x_vals) >= 2:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        expected_result = float(np.polyval(coeffs, proposed_value))
    elif method == "polynomial" and len(x_vals) >= 3:
        degree = min(2, len(x_vals) - 1)
        coeffs = np.polyfit(x_vals, y_vals, degree)
        expected_result = float(np.polyval(coeffs, proposed_value))
    elif method == "log" and all(x > 0 for x in x_vals):
        log_x = np.log(x_vals)
        coeffs = np.polyfit(log_x, y_vals, 1)
        expected_result = float(np.polyval(coeffs, np.log(proposed_value)))
    else:
        # Fallback: nearest neighbor interpolation
        distances = np.abs(x_vals - proposed_value)
        nearest_idx = np.argmin(distances)
        expected_result = float(y_vals[nearest_idx])

    # Calculate confidence based on:
    # 1. Number of historical points
    # 2. Distance from historical range
    # 3. R^2 of fit (for linear/polynomial)
    n_points_factor = min(1.0, len(history) / 5.0)  # Max confidence at 5+ points

    # Distance penalty
    x_min, x_max = x_vals.min(), x_vals.max()
    if proposed_value < x_min:
        distance_penalty = 1.0 - min(1.0, (x_min - proposed_value) / (x_max - x_min + 1e-6))
    elif proposed_value > x_max:
        distance_penalty = 1.0 - min(1.0, (proposed_value - x_max) / (x_max - x_min + 1e-6))
    else:
        distance_penalty = 1.0  # Within range

    confidence = n_points_factor * distance_penalty

    return {
        "expected_result": expected_result,
        "confidence": confidence,
        "within_expected": None,  # To be determined by caller with predicted value
        "trend_direction": trend_direction,
        "history_range": (float(x_min), float(x_max)),
        "num_historical_points": len(history),
    }


def validate_proposal_against_trend(
    history: list[tuple[float, float]],
    proposed_config: float,
    predicted_result: float,
    tolerance: float = 0.2,
    method: str = "linear",
) -> dict[str, Any]:
    """Validate that a predicted result matches the historical trend.

    Args:
        history: Historical (config, result) pairs
        proposed_config: Proposed configuration value
        predicted_result: The predicted/claimed result
        tolerance: Acceptable relative error
        method: Interpolation method

    Returns:
        Validation result dict
    """
    extrapolation = extrapolate_trend(history, proposed_config, method)
    expected = extrapolation["expected_result"]

    if expected is None:
        return {
            "valid": None,
            "expected": None,
            "predicted": predicted_result,
            "error": None,
            "within_tolerance": False,
            "confidence": 0.0,
            "note": "Cannot validate: no historical data",
        }

    error = abs(predicted_result - expected) / (abs(expected) + 1e-9)
    within_tolerance = error <= tolerance

    return {
        "valid": within_tolerance,
        "expected": expected,
        "predicted": predicted_result,
        "error": error,
        "within_tolerance": within_tolerance,
        "confidence": extrapolation["confidence"],
        "trend_direction": extrapolation["trend_direction"],
    }


def find_similar_configs(
    graph: Any,
    theory_type: str,
    change_types: set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find similar configurations from graph history.

    STUB: This is a placeholder for the full implementation.

    Args:
        graph: KnowledgeGraph object
        theory_type: Type of theory being proposed
        change_types: Set of change categories
        limit: Maximum number of similar configs to return

    Returns:
        List of similar historical configurations
    """
    # STUB implementation
    return {
        "similar_configs": [],
        "note": "find_similar_configs is a stub - full implementation requires graph query",
    }


def build_interpolation_context(
    graph: Any,
    config_key: str,
) -> dict[str, Any]:
    """Build context for interpolation from graph history.

    STUB: This is a placeholder for the full implementation.

    Args:
        graph: KnowledgeGraph object
        config_key: Configuration key to get history for

    Returns:
        Context dict with history and metadata
    """
    # STUB implementation
    history = graph.get_measurement_history(config_key) if hasattr(graph, "get_measurement_history") else []

    return {
        "config_key": config_key,
        "history": history,
        "num_points": len(history),
        "note": "build_interpolation_context is a stub",
    }
