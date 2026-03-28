## Goal

Normalize the repo onto `uv` so contributors and agents can share one deterministic Python setup.

## Status

- done

## Context

The repo currently works with a local `.venv`, but the contributor story and reproducibility are better if `uv` is the default operator path.

## Scope

- add `pyproject.toml`
- make `uv sync` the primary setup path
- keep current scripts compatible with `.venv/bin/python`

## Acceptance Criteria

- `uv sync` succeeds
- tests run from the uv-managed environment
- README and env docs reflect the uv-first path

## Validation

- `uv sync`
- `uv run pytest`

## Delivery

- `pyproject.toml`
- `README.md`

## Dependencies

- none

## Ownership / Non-Overlap

Owns dependency metadata and onboarding docs. Does not change falsifier verdict policy.

## Constraints

- preserve current train-time dependencies
- keep repo-local interpreter behavior stable

## References

- `requirements.txt`
- `README.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: dependency metadata or validation commands changed
- Split If: train deps and dev tooling need separate lanes
- Block If: uv environment cannot resolve
