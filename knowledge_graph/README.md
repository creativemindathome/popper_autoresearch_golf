# Knowledge graph folder

The Ideator reads context from `./knowledge_graph/`.

MVP supported inputs:

- `*.json` (optionally `{ "nodes": [...], "edges": [...] }`)
- `*.jsonl` (one JSON record per line: ideas, reviews, experiments, results)
- `.md` notes

The Ideator will include a compact summary when it can parse JSON; otherwise it will pass raw snippets into the prompt.

## Outbox

Generated ideas are saved to `knowledge_graph/outbox/ideator/`:

- `latest.json` (most recent idea)
- `YYYYMMDDTHHMMSSZ_<idea_id>.json` (timestamped history)
