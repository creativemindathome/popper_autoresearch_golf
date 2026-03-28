## Goal

State the exact outcome for this issue in one sentence.

## Context

Include the current architecture, run, theory, or artifact that motivates this task.

## Scope

List the allowed changes and the boundaries of this issue.

## Acceptance Criteria

Define the conditions that make the issue complete.

## Validation

List the exact commands or checks that must pass.
Include the smallest failing or reproducible check first when adding or changing behavior.

## Delivery

List the exact repo paths that must exist or be updated when the issue is done.

## Dependencies

List upstream issues, artifacts, or env prerequisites.

## Ownership / Non-Overlap

Name the files or surfaces this issue owns. Note any forbidden overlap.

## Constraints

Call out parameter budget, runtime, artifact-size, or workflow constraints.
Include any non-breakage requirements for scripts, schemas, handoff contracts, or Linear synchronization.

## References

Link to relevant files, runs, docs, or external references.

## Loop Policy

- Max Attempts: 2
- Retry Only If: the issue text, validation, or dependencies materially changed
- Split If: more than one subsystem or validation surface must change
- Block If: missing environment, dependency, or ownership clearance

## TDD / Regression Requirement

- Name the primary regression check for this issue.
- State what would count as breakage if the change is wrong.
- If the issue touches orchestration or agent handoffs, include the local command or artifact check that proves Linear-safe behavior still holds.
