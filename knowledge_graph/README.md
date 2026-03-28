# Knowledge graph folder

The Ideator reads context from `./knowledge_graph/`.

MVP supported inputs:

- `*.json` (optionally `{ "nodes": [...], "edges": [...] }`)
- `*.jsonl` (one JSON record per line: ideas, reviews, experiments, results)
- `.md` notes

The Ideator will include a compact summary when it can parse JSON; otherwise it will pass raw snippets into the prompt.

