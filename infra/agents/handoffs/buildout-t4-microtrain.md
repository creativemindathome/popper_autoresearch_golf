## Goal

Implement the deterministic `T4` 100-step micro-train gate as the first learning-based falsifier admission test.

## Status

- done

## Context

Once the candidate smoke gate exists, the next safe step is a bounded micro-train check that determines whether a candidate learns enough to justify later execution work.

## Scope

- 100-step bounded training loop
- deterministic seed policy
- loss-drop and throughput metrics
- fixture-backed regression coverage

## Acceptance Criteria

- baseline-like candidate completes a bounded 100-step run
- result artifact contains loss and throughput summaries
- bad candidate can be refuted deterministically

## Validation

- `uv run pytest tests -q`

## Delivery

- `falsifier/`
- `tests/`
- `docs/prd/FALSIFIER_T4_MICROTRAIN.md`

## Dependencies

- `buildout-t3-smoke.md`
- `buildout-calibration-lite.md`

## Ownership / Non-Overlap

Owns micro-train gate logic and tests. Does not own future Stage 2 or graph intelligence.

## Constraints

- bounded runtime only
- no 500-step follow-up in this issue

## References

- `docs/prd/FALSIFIER_T4_MICROTRAIN.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: training gate logic or thresholds changed
- Split If: data loading and learning policy need separate lanes
- Block If: calibration-lite thresholds are still unstable
