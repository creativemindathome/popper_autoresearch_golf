# Falsifier PRD Alignment

This document is the explicit mapping between the original falsifier PRD and the code that is active in this repo today.

## Active deterministic path

- validation preflight: [falsifier/validation.py](falsifier/validation.py)
- `T2` budget gate: [falsifier/stage1/orchestrator.py](falsifier/stage1/orchestrator.py) using thresholds from [falsifier/thresholds.py](falsifier/thresholds.py)
- `T3` smoke gate: [falsifier/adapters/parameter_golf.py](falsifier/adapters/parameter_golf.py)
- `T5` init-signal gate: [falsifier/stage1/t5_init_gate.py](falsifier/stage1/t5_init_gate.py), [falsifier/stage1/init_aggregates.py](falsifier/stage1/init_aggregates.py) — compares minimal-model weight kurtosis / effective-rank means to `calibration_lite.minimal_init_baseline` when present; otherwise skipped (pass).
- `T4` bounded micro-train gate: [falsifier/stage1/micro_train_gate.py](falsifier/stage1/micro_train_gate.py)
- `T6` citation gate (T6a): [falsifier/stage1/t6_citation_gate.py](falsifier/stage1/t6_citation_gate.py) — optional `calibration_claims` on the candidate (dotted paths under `calibration_lite`) must match numeric values in the profile.
- calibration-lite + baseline profile: [docs/prd/CALIBRATION_LITE.md](docs/prd/CALIBRATION_LITE.md), [falsifier/calibration_lite.py](falsifier/calibration_lite.py), [infra/agents/scripts/run_baseline_profile.py](infra/agents/scripts/run_baseline_profile.py)
- read-only mechanism probe bundle (for later T6b–T6d): [falsifier/mechanism_probes.py](falsifier/mechanism_probes.py)

## Implemented now

- `T2` uses artifact and FLOPs limits; baseline FLOPs and limits are loaded from `calibration_lite` in the latest baseline profile when present.
- `T3` is the import / forward / backward smoke path through the repo `train_gpt.py` adapter.
- `T5` compares minimal-env init statistics to `minimal_init_baseline` recorded during baseline profile generation (same `train_gpt.py` path and seed policy). Wide log-band check (see `init_aggregates.within_band`).
- `T4` runs a deterministic 100-step CPU micro-train on the candidate module and compares loss drop to calibration-lite (5% of baseline loss drop when a baseline micro-train summary exists).
- `T6` runs only when `calibration_claims` is non-empty; otherwise skipped (pass).

## Not implemented yet

- original-PRD `T0` boldness / diff analysis
- full-architecture random-init diagnostics separate from minimal-init gate (full `architecture` profile still has `random_init_baseline` aggregates for reference)
- T6b–T6d (mechanism-claim extraction, checkpoint sensitivity as hard gate, graph interpolation)
- Stage 2 adversarial prosecution

## Drift policy

- docs must not imply a gate exists unless there is executable code and regression coverage for it
- deferred gates stay explicitly labeled deferred
- local handoff state and Linear state must agree before a later gate is started
