# Local Constraints

This document defines non-negotiable local execution constraints for agents operating in this repo.

## 1. TDD Requirement

- Every non-trivial change must start from a declared validation target.
- Prefer writing or naming the smallest failing regression first.
- Implementation is only valid if the declared validation passes after the change.
- Manual inspection is not a substitute for a command, artifact check, or machine-verifiable result.

## 2. Non-Breakage Requirement

- Do not break repo paths, schema names, issue-template headings, or artifact contracts without an explicit migration issue.
- Do not silently change script exit behavior or output locations used by orchestration.
- If a change can affect agent routing, falsifier verdict handling, or record packaging, add a regression check for that surface.

## 3. Linear/Handoff Integrity

- Local handoff files under `infra/agents/handoffs/` are the source of truth.
- Linear is a synchronized execution surface, not an independent planning surface.
- No agent may advance a Linear issue beyond what the local handoff file, validation results, and delivery artifacts support.
- If Linear state and local handoff state diverge, stop work and reconcile before continuing.

## 4. Completion Rule

An issue is only complete when all are true:

- the local handoff file still matches the intended scope
- declared validation passes
- declared delivery artifacts exist in the repo
- any required falsifier or orchestration artifacts are present
- Linear state reflects, rather than contradicts, the local repo state

## 5. Minimum Checks for Agent-Surface Changes

If an issue touches any of these surfaces, include a regression command or artifact check:

- `infra/agents/scripts/`
- `infra/agents/handoffs/`
- `infra/agents/symphony/`
- `research/falsification/`
- any logic that changes how Linear issues are released, blocked, promoted, or completed
