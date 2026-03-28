# Falsifier PRD Alignment

This document is the explicit mapping between the original falsifier PRD and the code that is active in this repo today.

## Active deterministic path

- validation preflight: `/Users/curiousmind/Desktop/null_fellow_hackathon/falsifier/validation.py`
- `T1` precedent filter: `/Users/curiousmind/Desktop/null_fellow_hackathon/falsifier/stage1/orchestrator.py`
- `T2` budget gate: `/Users/curiousmind/Desktop/null_fellow_hackathon/falsifier/stage1/orchestrator.py`
- `T3` smoke gate: `/Users/curiousmind/Desktop/null_fellow_hackathon/falsifier/adapters/parameter_golf.py`

## Implemented now

- `T1` exists as a deterministic graph-aware precedent gate over theory-history records and can fall back to legacy supplied reference theories.
- `T2` exists as a deterministic artifact-budget and FLOPs-ratio gate relative to the imported parameter-golf baseline.
- `T3` exists as an import, construction, forward-loss, and backward smoke path through the repo-specific `train_gpt.py` adapter.

## T1 contract

- candidate inputs may include replayable `theory_history` records with:
  - prior `theory_id`
  - prior verdict (`refuted`, `surviving`, or legacy `reference`)
  - prior theory text
  - optional failure context
  - optional mechanism tags
  - optional related prior theory ids
- Stage 1 verdict artifacts emit `t1_mode = "graph_aware_precedent"` and structured `precedent_evidence` naming the closest prior record and any graph context used.

## Not implemented yet

- original-PRD `T0` boldness / diff analysis
- original-PRD `T4` random-init signal diagnostics
- original-PRD `T5` initialization diagnostics
- original-PRD `T6` checkpoint mechanism validation
- original-PRD `T7` micro-train learning gate
- Stage 2 adversarial prosecution

## Drift policy

- docs must not imply a gate exists unless there is executable code and regression coverage for it
- deferred gates stay explicitly labeled deferred
- local handoff state and Linear state must agree before a later gate is started
