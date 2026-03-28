## Goal

Consume calibration-lite thresholds inside the deterministic falsifier instead of relying on hard-coded gate values.

## Context

The profile generator exists, but the falsifier core still needs a schema-checked threshold loading path before `T4` can be trusted.

## Scope

- define threshold schema
- load thresholds from the latest profile artifact
- validate missing/invalid threshold behavior

## Acceptance Criteria

- threshold schema is documented
- invalid calibration data fails clearly
- Stage 1 can read and use the threshold artifact

## Validation

- `uv run pytest tests -q`
- `.venv/bin/python infra/agents/scripts/run_baseline_profile.py`

## Delivery

- `falsifier/`
- `tests/`
- `docs/prd/CALIBRATION_THRESHOLD_CONSUMPTION.md`

## Dependencies

- `buildout-calibration-lite.md`
- `buildout-falsifier-core.md`

## Ownership / Non-Overlap

Owns threshold consumption logic and schema validation. Does not own the profile generation command itself.

## Constraints

- schema versioning must be explicit
- no interpolation or graph-derived thresholds

## References

- `docs/prd/CALIBRATION_THRESHOLD_CONSUMPTION.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: threshold schema or consumption path changed
- Split If: profile generation and threshold loading need separate ownership
- Block If: calibration-lite artifacts are still unstable
