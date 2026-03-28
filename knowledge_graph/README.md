# Knowledge graph folder

The Ideator reads context from `./knowledge_graph/`.

MVP supported inputs:

- `*.json` (optionally `{ "nodes": [...], "edges": [...] }`)
- `*.jsonl` (one JSON record per line: ideas, reviews, experiments, results)
- `.md` notes

The Ideator will include a compact summary when it can parse JSON; otherwise it will pass raw snippets into the prompt.

## Visuals

Render the seed knowledge graph JSON to PNG/SVG (requires Graphviz: `dot`, `sfdp`, etc. on PATH):

```bash
python3 knowledge_graph/render_seed_kg_graphviz.py
```

Outputs to `knowledge_graph/visuals/`:

- Overview: `seed_parameter_golf_kg.{png,svg}`
- Per-root breakdown: `seed_parameter_golf_kg_{data_pipeline,neural_network,training_evaluation}.{png,svg}`

Options:

- `--engine dot|sfdp|neato|twopi` (layout engine)
- `--mode overview|split-roots|both`

### Force-directed (Gephi-like)

This looks closer to the “network hairball” style (colored clusters / starbursts) and keeps labels in SVG tooltips:

```bash
python3 knowledge_graph/render_seed_kg_force_graphviz.py --mode both
```

Outputs to `knowledge_graph/visuals/`:

- Overview: `seed_parameter_golf_kg_force.{png,svg}`
- Per-root breakdown: `seed_parameter_golf_kg_force_{data_pipeline,neural_network,training_evaluation}.{png,svg}`

Options:

- `--labels none|roots|roots-branches|all` (defaults to `none`)
- `--engine sfdp|neato|fdp` (recommend: `sfdp`)

## Outbox

Generated ideas and candidate implementations are saved to `knowledge_graph/outbox/ideator/`:

- `latest.json` (most recent idea JSON)
- `latest_train_gpt.py` (most recent generated `train_gpt.py`)
- `latest_train_gpt.patch` (diff vs the parent)
- `latest_review.json` (most recent novelty review, if enabled)
- `runs/<run_id>/idea.json` (per-run idea JSON)
- `runs/<run_id>/train_gpt.py` (per-run generated implementation)
- `runs/<run_id>/train_gpt.patch` (per-run diff vs parent)
- `runs/<run_id>/parent_train_gpt.py` (archived parent used for diff)
- `runs/<run_id>/review.json` (per-run novelty review, if enabled)
