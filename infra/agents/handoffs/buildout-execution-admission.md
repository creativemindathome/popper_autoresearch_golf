## Goal

Add the local admission gate that blocks execution-project theory work until a valid falsifier promote artifact exists.

## Status

- done

## Context

The buildout/execution project split is documented, but the repo still needs an enforceable local rule for execution promotion.

## Scope

- define the promote artifact path
- validate promote artifact schema
- define local check for execution eligibility

## Acceptance Criteria

- missing verdict blocks execution promotion
- invalid verdict blocks execution promotion
- valid promote artifact unlocks execution eligibility

## Validation

- `uv run pytest tests -q`

## Delivery

- `infra/agents/scripts/`
- `falsifier/`
- `docs/prd/EXECUTION_ADMISSION_GATE.md`

## Dependencies

- `buildout-falsifier-core.md`
- `buildout-linear-project-split.md`

## Ownership / Non-Overlap

Owns local admission rule enforcement. Does not own theory generation or training execution.

## Constraints

- local repo artifacts remain authoritative
- no direct execution-project promotion without artifact proof

## References

- `docs/prd/EXECUTION_ADMISSION_GATE.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: admission rule or artifact schema changed
- Split If: admission checks and Linear sync automation need separate ownership
- Block If: verdict artifact contract is not stable
