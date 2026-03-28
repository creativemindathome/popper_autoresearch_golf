# Calibration Threshold Consumption PRD

## Goal

Connect calibration-lite artifacts to the deterministic falsifier so Stage 1 gates use measured thresholds instead of hard-coded guesses.

## Scope

- define the minimum threshold surface
- read profile data from `research/profiles/latest_baseline_profile.json`
- use threshold values in `T3`/`T4` without requiring graph intelligence

## Required behavior

- missing calibration data must fail clearly, not silently
- threshold loading must be versioned and schema-checked
- Stage 1 should continue to run in reduced mode when optional fields are absent

## Validation

- schema validation for calibration-lite
- threshold lookup tests
- one end-to-end test showing Stage 1 consumes the profile artifact
