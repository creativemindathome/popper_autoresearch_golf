## Goal

Separate repo buildout work from actual falsifier execution work across Linear projects and local contracts.

## Status

- done

## Context

Repo plumbing and deterministic falsifier implementation should not share a project surface with actual candidate-theory execution.

## Scope

- create and document separate buildout and execution Linear projects
- update repo docs and env examples
- define the promotion rule from buildout to execution

## Acceptance Criteria

- buildout and execution project names/slugs are documented
- active Symphony project is clearly buildout-only
- execution admission requires a falsifier verdict artifact

## Validation

- `uv run python infra/agents/scripts/check_symphony_readiness.py`
- manual verification of project URLs and slugs

## Delivery

- `infra/agents/docs/FALSIFIER_V1_PRD.md`
- `infra/agents/handoffs/README.md`
- `infra/agents/env/.env.symphony.example`

## Dependencies

- none

## Ownership / Non-Overlap

Owns project split docs and env metadata. Does not own falsifier measurement code.

## Constraints

- no execution issue should be routed through the buildout project
- keep the current active environment usable

## References

- `README.md`
- `infra/agents/symphony/WORKFLOW.hermes.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: project split docs or env examples changed
- Split If: Linear sync automation needs its own implementation lane
- Block If: project slugs are unresolved
