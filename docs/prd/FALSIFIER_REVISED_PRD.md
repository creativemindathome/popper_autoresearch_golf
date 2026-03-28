# Falsifier Revised PRD

This repo does not adopt the full user PRD literally. It adopts a reduced first implementation that keeps the research standard high while removing unsafe complexity from the initial landing.

## Why the original PRD was too broad for a first implementation

- It mixes deterministic admission logic, GPU-heavy probes, checkpoint reasoning, LLM hypothesis generation, and multi-run ablation planning into one first milestone.
- It assumes stable helper APIs such as `create_model()`, `setup_optimizer_from_source()`, and `measure_claim()` that do not exist in the imported `parameter-golf` baseline.
- It tries to make Stage 2 depend on a large number of not-yet-proven abstractions before the Stage 1 contract is stable.
- It does not separate repo buildout work from live execution work in Linear, which would create queue drift.

## Revised implementation policy

### Buildout project vs execution project

- `Parameter Golf Repo Buildout` is the only active Symphony target during implementation.
- `Parameter Golf Falsifier Execution` is reserved for future live theory queues once the falsifier package, tests, and calibration surfaces stabilize.

### Reduced falsifier scope for the first landing

The initial implemented falsifier package must provide:

1. typed input and output contracts
2. deterministic Stage 1 admission core
3. repo-local CLI entrypoint
4. test coverage for non-GPU logic
5. uv-managed environment

The initial deterministic core includes only:

- validation preflight
- simplified `T1` novelty and precedent checking
- `T2` parameter, artifact, and time budget estimation
- `T3` import and forward/backward smoke diagnostics

The next implementation slice should add:

- `T4` random-init signal diagnostics
- `T5` bounded micro-train
- `T6a` citation validation against calibration data

### Research quality constraints preserved

- No theory is allowed into live execution unless it passes deterministic admission.
- The falsifier remains a separate critic, not a merged ideator/falsifier role.
- Every future GPU-bound test must have an explicit cost class, kill threshold, and reproducibility surface.
- Every future Stage 2 LLM hypothesis must be grounded in measured numbers, not prompt-only intuition.

## PRD improvements over the original draft

### 1. Add a distinct validation layer before T3

Before any GPU construction test, validate:

- required source fields exist
- `Hyperparameters` can be parsed
- artifact budget can be estimated
- parent references are structurally well-formed

This reduces expensive failures that should have been simple rejections.

### 2. Split T6 into independent milestones

Do not build the full checkpoint mechanism layer in one issue.

Implement in this order:

- T6a: citation verification against calibration values
- T6b: mechanism-claim extraction
- T6c: checkpoint sensitivity probes
- T6d: graph interpolation

### 3. Remove Stage 2 from the first code milestone

Stage 2 should not ship until:

- Stage 1 deterministic results are trustworthy
- calibration artifacts are versioned
- run execution and ablation helpers are stable
- tests cover at least one end-to-end deterministic falsifier run

### 4. Tighten tests

The original PRD’s tests are too integration-heavy for the first landing. Add these first:

- diff classification unit tests
- hyperparameter extraction unit tests
- budget estimator unit tests
- deterministic falsifier CLI smoke test
- known-trivial change must fail T0
- known-budget overflow must fail T2
- known-safe baseline-like submission must pass the reduced Stage 1 core

## Implementation sequence in this repo

1. `uv` packaging and repo-local test runner
2. falsifier typed contracts
3. deterministic Stage 1 core (`T0`, `T2`)
4. handoff graph and Linear buildout queue
5. T3/T6a helpers
6. calibration versioning
7. GPU-bound Stage 1 extensions
8. only then Stage 2 adversarial agent
