# Calibration to Mechanism Probes PRD

## Goal

Evolve calibration-lite into a checkpoint-safe mechanism-probe layer that can support later `T6`-style falsification without destabilizing the deterministic core.

## Scope

- keep all probes read-only against the checkpoint by default
- start with weight- and logits-level probes before activation-heavy instrumentation
- define a versioned threshold schema that later stages can consume safely

## Phases

### Phase 1: Calibration-lite stabilization

- finalize baseline profile schema
- freeze threshold field names
- add schema validation and version checks

### Phase 2: Checkpoint-safe mechanism probes

- quantization sensitivity at the weight/logit level
- parameter-group rank and kurtosis summaries
- optional loss-delta probes that do not mutate training state

### Phase 3: Trusted mechanism claims

- define a small set of supported mechanism-claim types
- map each claim type to one or two measurable checkpoint probes
- reject unsupported claim types early instead of inventing measurements

## Interfaces

- `BaselineProfile`
- `ThresholdProfile`
- `MechanismProbeSpec`
- `MechanismProbeResult`

## Risks

- probe output will look precise before it is reliable
- activation-level instrumentation can become fragile fast
- graph-derived thresholds will be too noisy early on

## Tests

- schema contract tests
- checkpoint probe fixture tests
- replay tests for mechanism results
- one end-to-end path from profile -> threshold lookup -> mechanism result artifact
