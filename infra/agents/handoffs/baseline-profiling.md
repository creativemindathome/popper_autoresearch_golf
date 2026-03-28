## Goal

Produce the first versioned baseline profile artifact for the current repo so later theory and falsification issues have a stable input surface.

## Context

The baseline profiler is currently a stub writer. This issue is the first execution lane for turning the repo scaffold into a falsifier-first workflow.

## Scope

- Run the current baseline profile command
- Verify output lands under `research/profiles/`
- Keep the issue focused on profile artifact generation, not training or model integration

## Acceptance Criteria

- a versioned baseline profile JSON file exists under `research/profiles/`
- `research/profiles/latest_baseline_profile.json` is updated
- the run is reproducible from a single command

## Validation

- `python3 infra/agents/scripts/run_baseline_profile.py`

## Delivery

- `research/profiles/latest_baseline_profile.json`
- `research/profiles/*.json`

## Dependencies

- `environment-bootstrap.md`

## Ownership / Non-Overlap

Owns only baseline profile artifacts and the baseline profiling command path.

## Constraints

- No training runs
- No record packaging
- No changes to handoff dependency semantics in this issue

## References

- `research/falsification/README.md`
- `infra/agents/scripts/run_baseline_profile.py`

## Loop Policy

- Max Attempts: 2
- Retry Only If: the profile command or output contract changed
- Split If: real model instrumentation needs a separate implementation issue
- Block If: env/bootstrap prerequisites are not satisfied
