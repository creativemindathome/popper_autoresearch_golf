# Handoff Graph

This directory is the checked-in source of truth for the execution graph.

## Initial Root Issues

1. Environment bootstrap
2. Baseline profiling
3. Theory proposal
4. Falsification batch
5. Engineering change
6. Training/eval run
7. Distillation update

## Policy

- Each executable issue lives in its own markdown file.
- Every issue must follow `ISSUE_TEMPLATE.md`.
- Only dependency-free issues are eligible for release into `Todo`.
- Training-oriented issues must name the falsifier verdict artifact they depend on.
- Every issue must declare a concrete validation plan and a TDD/regression requirement.
- Local handoff files are authoritative for scope, dependencies, and completion; Linear must not drift from them.
- Agents must not close, advance, split, or unblock a Linear issue unless the corresponding local handoff file and local artifacts justify the change.
- If an agent detects disagreement between a Linear issue and its local handoff file, it must stop, reconcile, and only then continue execution.
