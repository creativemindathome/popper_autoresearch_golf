## Goal

Implement the deterministic `T3` candidate smoke gate for import, model construction, forward pass, and backward pass viability.

## Status

- in_progress

## Context

The falsifier core exists, but the next reliable admission slice is a real candidate smoke path before any micro-train or execution promotion.

## Scope

- add `T3` to the deterministic core
- reuse the parameter-golf adapter
- keep CPU-compatible fallback behavior
- add fixture coverage for broken candidates

## Acceptance Criteria

- baseline candidate passes T3
- import-broken candidate fails deterministically
- construction-broken candidate fails deterministically

## Validation

- `uv run pytest tests -q`
- `uv run python -m falsifier.main --help`

## Delivery

- `falsifier/`
- `tests/`
- `docs/prd/FALSIFIER_T3_SMOKE.md`

## Dependencies

- `buildout-falsifier-core.md`

## Ownership / Non-Overlap

Owns T3 smoke logic and tests. Does not own calibration thresholds or execution-project promotion rules.

## Constraints

- deterministic failure reasons
- no Stage 2 coupling

## References

- `docs/prd/FALSIFIER_T3_SMOKE.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: smoke behavior or fixture coverage changed
- Split If: adapter changes and policy changes need separate write lanes
- Block If: candidate adapter contract changes underneath the test
