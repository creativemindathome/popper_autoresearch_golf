## Goal

Implement the deterministic falsifier core that can reject, refute, or promote a candidate without relying on LLM Stage 2.

## Status

- done

## Context

The original falsifier PRD is too broad for a first trustworthy implementation. This issue covers the core runtime only.

## Scope

- candidate input schema
- repo-specific `train_gpt.py` adapter
- Stage 1 orchestrator
- stable verdict artifact
- core tests

## Acceptance Criteria

- the falsifier can evaluate a candidate package and emit a JSON verdict
- baseline repo candidate passes the smoke path
- broken candidates fail deterministically

## Validation

- `uv run pytest tests/falsifier`

## Delivery

- `falsifier/`
- `tests/`

## Dependencies

- `buildout-uv-runtime.md`

## Ownership / Non-Overlap

Owns the falsifier package and tests. Does not own Linear sync scripts or execution-project admission state.

## Constraints

- no Stage 2 LLM dependency in v1
- prefer deterministic checks over clever orchestration

## References

- `infra/agents/docs/FALSIFIER_V1_PRD.md`
- `docs/prd/FALSIFIER_REVISED_PRD.md`
- `train_gpt.py`

## Loop Policy

- Max Attempts: 2
- Retry Only If: verdict contract or tests materially changed
- Split If: adapters and stage logic need separate write lanes
- Block If: candidate runtime surface is not stable enough to test
