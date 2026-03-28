# Falsifier V1 PRD

This document is the repo-integrated replacement for the broader falsifier PRD. It intentionally cuts scope to a deterministic v1 that this environment can execute reliably.

## Why This Revision Exists

The original PRD mixes three systems:

- falsifier runtime
- scientific measurement framework
- orchestration and graph intelligence

That is too much to trust in one implementation pass. In this repo, the falsifier should ship in layers.

## Approved Changes to the Original PRD

### 1. Split the product in two

Implement:

- `falsifier-core`: deterministic measurement and verdict engine
- `falsifier-orchestrator`: Linear/Symphony integration and later LLM prosecution

V1 must ship `falsifier-core` first.

### 2. Reduce Stage 1 to the reliable gate set

V1 Stage 1 currently consists of:

- validation preflight
- `T2` static budget and artifact-size gate
- `T3` import / instantiate / forward-backward smoke test
- bounded micro-train gate (`T4` in code labels)

Deferred from the original PRD:

- random-init signal diagnostics
- 100-step micro-train
- citation hard-fails
- graph interpolation
- compound-tag and correlated-tag kill logic
- open-ended LLM Stage 2
- multi-ablation run planner
- multi-seed heavy calibration

### 3. Normalize the candidate interface

The original `proposed_train_gpt` raw-source contract is too loose for this repo.

V1 candidate package:

- candidate `train_gpt.py` source
- metadata JSON
- optional config delta
- optional notes

The falsifier runtime is responsible for importing this candidate in isolation and instantiating the model through a repo-specific adapter.

### 4. Make calibration-lite the default

V1 calibration includes:

- baseline parameter/artifact budget
- baseline random-init profile
- baseline 100-step micro-train summary
- optional checkpoint weight profile

Full checkpoint mechanism probes stay optional until the measurement substrate is stable.

### 5. Separate buildout from execution

Linear projects:

- `parameter-golf-buildout`: repo plumbing, falsifier-core implementation, orchestration contracts
- `parameter-golf-falsifier-execution`: actual candidate-theory execution and research lanes

The active Symphony project should remain buildout-focused until the admission gate is trustworthy.

## V1 Deliverables

### Repo/runtime

- `uv`-managed Python project
- deterministic falsifier package
- repo-specific adapter for `train_gpt.py`
- CLI entrypoint for candidate evaluation
- stable JSON verdict artifact

### Tests

- golden candidate fixtures
- baseline candidate should pass the `T3` smoke path
- syntax-broken candidate should fail validation/import
- over-budget candidate should fail `T2`
- orchestration contract tests for promote/block behavior

### Orchestration

- checked-in handoff graph for buildout
- separate execution Linear project reserved for later candidate testing
- explicit promotion rule: no execution issue without a falsifier verdict artifact

## V1 File Structure

```text
falsifier/
  adapters/
  stage1/
  types.py
  validation.py
  main.py
tests/
  falsifier/
```

The richer structure from the original PRD can be added later after Stage 1 is trustworthy.

## Validation Standard

The falsifier is only valuable if the harness is more trustworthy than the candidate.

Required test tiers:

- unit tests for parsing, validation, and adapters
- deterministic stage tests with fixture candidates
- artifact contract tests
- one end-to-end admission test from candidate input to verdict output

## Out of Scope for V1

- freeform LLM adversarial prosecution
- automatic ablation synthesis
- graph-embedding precedent search as a hard gate
- interpolation-based rejection
- heavy checkpoint activation instrumentation across all layers

## Current Traceability

The live code currently implements:

- validation preflight
- `T2` static budget gate
- `T3` deterministic smoke diagnostics
- `T4` bounded micro-train gate versus calibration-lite

Further gates (random-init diagnostics, citation checks, mechanism probes as hard gates) remain separate implementation slices.
