# AutoResearch Loop: Ideator + Falsifier

This repo is a **hackathon scaffold** for automating a small piece of the research loop around [OpenAI Parameter Golf](https://github.com/openai/parameter-golf): models that train inside a strict size budget.

### What it’s trying to do (intuition)

Research often looks like: **have an idea → sanity-check it → stress-test it → remember what happened.** Here, software plays parts of that loop:

1. **Ideation** — A model proposes a concrete change to the training code (a new trick, architecture tweak, or training move), grounded in what you already know.
2. **Review** — A second model pushes back on novelty and clarity so you don’t queue obviously weak duplicates.
3. **Falsification** — A **checker** runs the idea through automated gates: Does it fit the budget? Does the code compile? Do quick training signals look sane? Optionally, a second phase tries harder to break the story before you spend a full GPU day.
4. **Memory** — A **knowledge graph** (plain JSON on disk) accumulates ideas, outcomes, and notes so later runs aren’t starting from zero.

Nothing here replaces real experiments at scale; it **front-loads cheap failures** and **keeps a paper trail** of what you tried.

---

## Run the multi-hypothesis experiment (Anthropic)

The path most people want for a **batch run** is: **Claude proposes several hypotheses**, each one is **checked by the falsifier**, and artifacts land in a timestamped folder under `experiments/ten_hypothesis_run/`. That lives in **`experiments/ten_hypothesis_run/`** — this section is the short version; more detail is in [`experiments/ten_hypothesis_run/README.md`](experiments/ten_hypothesis_run/README.md).

### 1. Install (from the repo root)

```bash
uv sync --extra llm
```

`--extra llm` pulls in the Anthropic client used by the runner. Add `--extra dev` if you plan to run tests.

### 2. API key

You need an **Anthropic API key** in the environment (or in a `.env` at the repo root that you `source`):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the run

```bash
cd experiments/ten_hypothesis_run
set -a && source ../../.env && set +a   # optional, if you keep keys in .env

python3 run_full_live_experiment.py --num-hypotheses 10
```

Or use the shell wrapper (same pipeline, optional follow-up steps like frames/video):

```bash
bash run_full_experiment.sh
```

### 4. Knobs you’ll actually use

| Flag | What it does |
|------|----------------|
| `--num-hypotheses N` | How many ideas to generate and run through the pipeline (default 10). |
| `--output-dir PATH` | Write into a fixed folder instead of a new `live_run_YYYYMMDD_HHMMSS/`. |
| `--disable-reviewer` | Skip the novelty reviewer (faster, less filtering). |
| `--disable-stage2` | Run only the first, faster checking stage. |

For model names and the rest: `python3 run_full_live_experiment.py --help`.

### 5. What you get

Each run creates a directory like `live_run_YYYYMMDD_HHMMSS/` with logs, per-hypothesis outputs, a `summary.json`, and (when enabled) graph snapshots and data for timelines or clips. See the folder layout in [`experiments/ten_hypothesis_run/README.md`](experiments/ten_hypothesis_run/README.md).

**Remote / Cursor Cloud:** same pipeline, setup notes in [`experiments/ten_hypothesis_run/CURSOR_CLOUD_SETUP.md`](experiments/ten_hypothesis_run/CURSOR_CLOUD_SETUP.md).

---

## How the pieces connect (big picture)

```
  Propose ideas          Push back (novelty)        Try to break it
  ─────────────          ───────────────────        ───────────────
  Claude (batch)    or   OpenAI reviewer      →    Falsifier checks
  Gemini (single)                               →    Knowledge graph
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                    remembers wins, losses, and notes
```

- **Anthropic path:** many hypotheses in one go — `run_full_live_experiment.py` (above).
- **Gemini path:** one idea per CLI invocation — `python3 -m ideator` from the **repo root**, with `GEMINI_API_KEY` (or `GOOGLE_API_KEY` / `GOOGLE_AI_API_KEY`) and `OPENAI_API_KEY` for the reviewer. This is separate from the Anthropic batch runner.

---

## Optional: single-idea flow (Gemini + OpenAI reviewer)

From the repo root, with Parameter Golf’s `train_gpt.py` available (e.g. clone the [parameter-golf](https://github.com/openai/parameter-golf) repo next to this project):

```bash
git clone https://github.com/openai/parameter-golf.git parameter-golf
python3 -m ideator --parent-train-gpt parameter-golf/train_gpt.py
```

Outputs default to `knowledge_graph/outbox/ideator/` (`latest.json`, per-run folders, and stable `<idea_id>.*` files). To **check** an approved idea with the falsifier:

```bash
uv run python -m falsifier.main \
  --idea-id "<idea_id>" \
  --knowledge-dir knowledge_graph \
  --output-json knowledge_graph/outbox/falsifier/<idea_id>_result.json
```

(Queue the idea under `knowledge_graph/inbox/approved/<idea_id>.json`, or use `--candidate-json` pointing at your JSON.)

---

## Knowledge graph visuals

Scripts under `knowledge_graph/visuals/` can render the seed ontology plus experiment branches and build short evolution clips. They need **Graphviz** (`dot`), and MP4 export needs **ffmpeg**. Example:

```bash
python3 knowledge_graph/visuals/render_original_kg_with_branches.py \
  --source merged \
  --output knowledge_graph/visuals/original_kg_with_branches.png
```

Evolution clips: `knowledge_graph/visuals/generate_evolution_movie_v2.py` (see `--help`). Timelines across several runs: `experiments/ten_hypothesis_run/visualize_all_live_runs_timeline.py`.

---

## Repo layout (short)

| Path | Role |
|------|------|
| `experiments/ten_hypothesis_run/` | **Anthropic** batch experiments (`live_run_*`). |
| `ideator/` | Single-idea generation (Gemini + reviewer). |
| `falsifier/` | Automated checks + optional second-stage probing. |
| `knowledge_graph/` | `graph.json`, inbox/outbox, seeds, visuals. |
| `docs/prd/` | Deeper specs. |
| `records/` | Optional local Parameter Golf submission trees; can be large — keep out of git if you want a small clone. |

---

## Development

```bash
uv sync --extra dev
uv run pytest
```

Symphony check: `python infra/agents/scripts/check_symphony_readiness.py`

**Further reading:** [`docs/prd/FALSIFIER_REVISED_PRD.md`](docs/prd/FALSIFIER_REVISED_PRD.md), [`docs/COMPLEXITY_AUDIT.md`](docs/COMPLEXITY_AUDIT.md), [`infra/agents/docs/FALSIFIER_V1_PRD.md`](infra/agents/docs/FALSIFIER_V1_PRD.md).

## License

MIT
