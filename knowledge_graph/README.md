# Knowledge graph folder

The Ideator reads context from `./knowledge_graph/`.

MVP supported inputs:

- `*.json` (optionally `{ "nodes": [...], "edges": [...] }`)
- `*.jsonl` (one JSON record per line: ideas, reviews, experiments, results)
- `.md` notes

The Ideator will include a compact summary when it can parse JSON; otherwise it will pass raw snippets into the prompt.

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
