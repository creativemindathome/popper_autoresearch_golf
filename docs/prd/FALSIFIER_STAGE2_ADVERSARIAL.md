# Falsifier Stage 2 Adversarial PRD

## Goal

Add the eventual adversarial Stage 2 on top of the deterministic core, not in place of it.

## Scope

- Stage 2 only runs after deterministic Stage 1 emits a promotable artifact
- hypotheses must be constrained, structured, and budget-bounded
- Stage 2 is allowed to deepen criticism, not replace deterministic gates

## Phases

### Phase 1: Structured hypothesis generation

- generate 1-3 kill hypotheses only
- each hypothesis must name the exact metric, threshold, and cost class
- unsupported experiment types are rejected before execution

### Phase 2: Constrained follow-up runs

- allow a bounded 500-step run or one ablation run
- no open-ended planning loops
- all Stage 2 runs must write replayable result artifacts

### Phase 3: Feedback artifacts

- produce structured failure analysis for the Ideator
- keep feedback grounded in measured numbers and result artifacts

## Interfaces

- `Stage2Hypothesis`
- `Stage2ExperimentSpec`
- `Stage2RunSpec`
- `Stage2Verdict`

## Risks

- unconstrained LLM output will create fake rigor
- ablation planning can explode compute cost
- Stage 2 can become a second ideator if prompt boundaries are weak

## Tests

- replay-based Stage 2 evaluation tests
- schema validation for hypothesis JSON
- contract test: Stage 2 cannot run without a valid Stage 1 promote artifact
- budget test: Stage 2 planner never schedules more than the allowed run set
