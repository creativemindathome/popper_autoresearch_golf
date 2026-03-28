# Handoff Graph

This directory is the checked-in source of truth for the execution graph.

## Active Graph

The current checked-in graph is for repo buildout, not for execution research.

Execution research should move to the separate execution Linear project only after:

1. the deterministic falsifier core exists
2. verdict artifacts are stable
3. handoff-to-Linear promotion rules are enforced
4. calibration-lite artifacts can be reproduced

## Buildout Root Issues

1. Environment bootstrap
2. UV and runtime normalization
3. Deterministic falsifier core
4. Calibration-lite and probe contracts
5. Buildout/execution project split
6. Execution admission rules

## Policy

- Each executable issue lives in its own markdown file.
- Every issue must follow `ISSUE_TEMPLATE.md`.
- Each handoff file should declare a `## Status` section with one of `todo`, `in_progress`, `backlog`, or `done`.
- Only dependency-free issues are eligible for release into `Todo`.
- Training-oriented issues must name the falsifier verdict artifact they depend on.
- Every issue must declare a concrete validation plan and a TDD/regression requirement.
- Local handoff files are authoritative for scope, dependencies, and completion; Linear must not drift from them.
- Agents must not close, advance, split, or unblock a Linear issue unless the corresponding local handoff file and local artifacts justify the change.
- If an agent detects disagreement between a Linear issue and its local handoff file, it must stop, reconcile, and only then continue execution.
- The active handoff directory is for buildout unless a future document explicitly marks an execution graph takeover.
