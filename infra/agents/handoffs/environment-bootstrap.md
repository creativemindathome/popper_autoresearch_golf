## Goal

Validate the local repo plumbing and secret configuration so Symphony can safely operate this repository.

## Context

The repo scaffold, env templates, and repo-local launch scripts exist, but execution should not proceed until the required secrets and readiness gates are satisfied.

## Scope

- Fill `infra/agents/env/.env.symphony`
- Fill `infra/agents/env/.env.hermes` as needed
- Run bootstrap and readiness checks
- Confirm the Linear project slug matches the live project

## Acceptance Criteria

- `LINEAR_API_KEY` is present in `infra/agents/env/.env.symphony`
- bootstrap and local env checks pass
- Symphony readiness passes without missing required env vars

## Validation

- `infra/agents/scripts/bootstrap_repo.sh`
- `infra/agents/scripts/check_local_env.sh`
- `python3 infra/agents/scripts/check_symphony_readiness.py`

## Delivery

- `infra/agents/env/.env.symphony`
- `infra/agents/env/.env.hermes`

## Dependencies

- none

## Ownership / Non-Overlap

Owns only env files and readiness validation outputs. Does not modify research artifacts or records.

## Constraints

- Never commit secrets
- Keep project slug aligned with Linear project `parameter-golf-symphony`

## References

- `ISSUE_TEMPLATE.md`
- `infra/agents/symphony/WORKFLOW.hermes.md`

## Loop Policy

- Max Attempts: 2
- Retry Only If: env values or readiness conditions changed
- Split If: external runtime installation is required beyond env setup
- Block If: secrets or external tool access are unavailable
