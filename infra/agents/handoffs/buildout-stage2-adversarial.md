## Goal

Design the bounded adversarial Stage 2 on top of the deterministic core without undermining deterministic admission.

## Context

Stage 2 must stay artifact-driven, budget-bounded, and subordinate to Stage 1. It should only run after deterministic promotion and stable calibration artifacts exist.

## Scope

- Stage 2 hypothesis schema
- bounded follow-up run planning contract
- deterministic verdict policy from replayable results

## Acceptance Criteria

- Stage 2 prerequisites are explicit
- run-planning limits are explicit
- replay-based verdict rules are documented

## Validation

- `uv run pytest tests -q`

## Delivery

- `docs/prd/FALSIFIER_STAGE2_ADVERSARIAL.md`

## Dependencies

- `buildout-mechanism-probes.md`
- `buildout-execution-admission.md`

## Ownership / Non-Overlap

Owns Stage 2 design and later implementation boundary. Does not own deterministic Stage 1 gating.

## Constraints

- no freeform unbounded agent loops
- no direct Linear state transitions from Stage 2

## References

- `docs/prd/FALSIFIER_STAGE2_ADVERSARIAL.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: prerequisites or run-planning contract changed
- Split If: planner and LLM hypothesis generation need separate ownership
- Block If: deterministic core and mechanism probes are not stable
