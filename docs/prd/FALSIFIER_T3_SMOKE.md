# Falsifier T3 Smoke PRD

## Goal

Add a deterministic `T3` candidate smoke gate that imports a candidate `train_gpt.py`, instantiates the model through the parameter-golf adapter, and proves forward/backward viability before any heavier test runs.

## Scope

- extend the current deterministic falsifier core
- use the repo-specific adapter, not arbitrary runtime assumptions
- keep the smoke test CPU-compatible by default, with optional GPU acceleration

## Required behavior

- candidate import failures become `implementation_fail`
- model instantiation failures become `implementation_fail`
- non-finite forward loss becomes `implementation_fail`
- disconnected or missing gradients surface as a structured failure reason

## Validation

- add fixture coverage for syntax-broken and import-broken candidates
- baseline candidate must pass T3
- candidate with deliberately broken model construction must fail T3 deterministically
