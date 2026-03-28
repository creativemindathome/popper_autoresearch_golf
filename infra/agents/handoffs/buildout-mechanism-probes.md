## Goal

Evolve calibration-lite into checkpoint-safe mechanism probes that can support later T6-style falsification without destabilizing the deterministic core.

## Status

- done

## Context

Mechanism probes should only arrive after the deterministic core and threshold schema are stable. They must be read-only, checkpoint-safe, and cheap relative to training.

## Scope

- checkpoint-safe read-only probes
- supported mechanism-claim types
- profile schema stabilization for later mechanism use

## Acceptance Criteria

- probe outputs are versioned and documented
- probe execution does not mutate candidate code or checkpoints
- at least one mechanism-probe artifact can be regenerated deterministically

## Validation

- `uv run pytest tests -q`
- `.venv/bin/python infra/agents/scripts/run_baseline_profile.py`

## Delivery

- `research/`
- `docs/prd/CALIBRATION_TO_MECHANISM_PROBES.md`

## Dependencies

- `buildout-threshold-consumption.md`

## Ownership / Non-Overlap

Owns mechanism-probe design and artifact contracts. Does not own Stage 2 run planning.

## Constraints

- read-only checkpoint interaction only
- no graph interpolation hard gate

## References

- `docs/prd/CALIBRATION_TO_MECHANISM_PROBES.md`
- `infra/agents/docs/FALSIFIER_V1_PRD.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: probe contract or schema changed
- Split If: probe generation and threshold consumption need separate ownership
- Block If: calibration-lite artifacts remain unstable
