# Falsifier T4 Micro-Train PRD

## Goal

Implement the first learning-based admission gate: a bounded 100-step micro-train that checks whether a candidate actually learns enough to justify later execution work.

## Scope

- deterministic seed policy
- tiny but real token loading path
- loss-drop and throughput metrics
- promote/refute threshold inputs from calibration-lite

## Required behavior

- use a bounded token budget and fixed step count
- fail fast on divergence or non-finite loss
- emit a stable result artifact with loss trajectory and throughput summary
- remain separate from later 500-step or Stage 2 work

## Validation

- baseline-like candidate completes 100 steps
- known bad candidate diverges or underperforms deterministically
- result artifact is consumable by the admission policy
