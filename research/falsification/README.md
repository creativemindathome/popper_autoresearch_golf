# Falsification Contract

Each falsification batch should produce:

- an experiment spec JSON file
- a result bundle JSON file
- a verdict JSON file
- a markdown summary

Current executable experiment names:

- `architecture_profile`
- `quantization_profile`

Use `example_experiment_spec.json` as the starting schema for a theory-specific batch.

Verdict outcomes:

- `refuted`
- `survived`
- `inconclusive`
