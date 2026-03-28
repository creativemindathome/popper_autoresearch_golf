## Goal

Define and implement the minimum calibration artifacts needed for deterministic falsifier gating.

## Context

The original PRD calibrates too much too early. Build only the minimum baseline profile and 100-step comparison surface first.

## Scope

- baseline random-init measurements
- baseline budget stats
- baseline 100-step micro-train summary
- optional checkpoint weight profile contract

## Acceptance Criteria

- calibration-lite artifact format is documented
- the baseline profile can be regenerated
- Stage 1 thresholds can consume the artifact without manual edits

## Validation

- `uv run python infra/agents/scripts/run_baseline_profile.py`
- `uv run pytest tests/falsifier`

## Delivery

- `research/profiles/latest_baseline_profile.json`
- calibration-related code under `falsifier/` or `research/`

## Dependencies

- `buildout-falsifier-core.md`

## Ownership / Non-Overlap

Owns calibration-lite formats and their generation path. Does not own execution-project theory issues.

## Constraints

- no full graph interpolation
- no heavy multi-seed calibration as a gating dependency

## References

- `infra/agents/docs/FALSIFIER_V1_PRD.md`
- `research/probe_library.py`

## Loop Policy

- Max Attempts: 2
- Retry Only If: threshold consumers or artifact schema changed
- Split If: checkpoint probes must separate from micro-train baselines
- Block If: baseline artifacts are not reproducible
