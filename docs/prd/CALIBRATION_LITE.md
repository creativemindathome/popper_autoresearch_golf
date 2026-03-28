# Calibration-lite artifact

Calibration-lite is the minimal structured surface embedded in baseline profile JSON so Stage 1 can load thresholds without ad hoc constants.

## Location

- Checked-in profile: `research/profiles/latest_baseline_profile.json`
- Regeneration: `uv run python infra/agents/scripts/run_baseline_profile.py`
- Fast CI smoke (no micro-train): add `--skip-micro-train`

## Top-level fields

The profile may include `calibration_lite`, an object with `schema_version: "1"` and:

| Section | Purpose |
|--------|---------|
| `budget_baseline` | Baseline hyperparameters, param counts, FLOPs estimate, artifact byte limit, and max FLOPs ratio used alongside the static budget gate. |
| `random_init_baseline` | Aggregates over architecture `weight_kurtosis` and `effective_rank` from the baseline model init. |
| `quantization_mse_baseline` | Group-level quantization MSE from `quantization_profile`. |
| `micro_train_100_step` | Deterministic CPU summary (loss first/last, drop, throughput) unless `--skip-micro-train`. |
| `minimal_init_baseline` | Aggregates (kurtosis / effective-rank means, etc.) from the **minimal smoke model** built from repo `train_gpt.py`, used by **T5**. |
| `checkpoint_weight_profile` | Optional stub indicating whether a checkpoint was loaded for profiling. |

## Schema validation

`falsifier.calibration_lite.validate_calibration_lite` checks required keys for schema `"1"`. Profile generation fails fast if validation does not pass.

## Consumers

Threshold loading (Stage 1) reads `research/profiles/latest_baseline_profile.json` and uses `calibration_lite.budget_baseline` for baseline FLOPs and artifact limits where wired in code.
