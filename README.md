# AutoResearch Loop: Ideator + Falsifier

Hackathon scaffold for an AutoResearch loop around **OpenAI [Parameter Golf](https://github.com/openai/parameter-golf)**, combining:

- **Ideator** (Gemini + OpenAI): proposes a single falsifiable idea per run, with a pessimistic novelty reviewer and automatic revision rounds
- **Falsifier** (systematic testing): Stage 1 gates **T2–T7** (budget → compile → signal/init → micro-train; **T0/T1/T6 are not in the Stage 1 orchestrator**—see `falsifier/stage1/orchestrator.py`) plus Stage 2 adversarial prosecution when enabled
- **Knowledge graph**: file-backed store of ideas, outcomes, and learnings (`knowledge_graph/graph.json`)

**Related:** batch runs that generate hypotheses with **Anthropic** and drive the falsifier live in [`experiments/ten_hypothesis_run/`](experiments/ten_hypothesis_run/README.md) (separate from the Gemini-based CLI ideator).

## Quick start

### 1. Environment

Install dependencies (from repo root):

```bash
uv sync
# For pytest:   uv sync --extra dev
# For training: uv sync --extra train
# For Anthropic (multi-hypothesis + optional Stage 2 LLM paths): uv sync --extra llm
```

The published package metadata only auto-discovers **`falsifier`** (`pyproject.toml`). Run **`python -m ideator`** from the **repository root** so the local `ideator/` package resolves.

**Ideator (Gemini)** — any one of:

```bash
export GEMINI_API_KEY="..."
# or: GOOGLE_API_KEY / GOOGLE_AI_API_KEY
```

**Novelty reviewer (OpenAI):**

```bash
export OPENAI_API_KEY="..."
# Optional: custom API base (e.g. proxies)
export OPENAI_BASE_URL="https://api.openai.com"
```

**Optional tuning:**

```bash
export GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_TIMEOUT_S="180"
export GEMINI_MAX_RETRIES="2"
export OPENAI_REVIEWER_MODEL="gpt-4o-mini"
export IDEATOR_MAX_REVIEW_ROUNDS="4"
export IDEATOR_REVIEWER_MIN_SCORE="6"
```

**Parent `train_gpt.py` without a local clone** (ideator can fetch from GitHub):

```bash
export PARAM_GOLF_PARENT_REPO_URL="https://github.com/openai/parameter-golf"
export PARAM_GOLF_PARENT_GIT_REF="main"
export PARAM_GOLF_PARENT_FILE_PATH="train_gpt.py"
```

**Anthropic** is **not** a fallback for `python -m ideator` (that path is Gemini-only). Use `ANTHROPIC_API_KEY` for [`experiments/ten_hypothesis_run/`](experiments/ten_hypothesis_run/README.md) and for falsifier Stage 2 code paths that call Anthropic when installed.

### 2. Generate an idea (Gemini ideator)

```bash
# Option A: clone Parameter Golf and point at train_gpt.py
git clone https://github.com/openai/parameter-golf.git parameter-golf

python3 -m ideator --parent-train-gpt parameter-golf/train_gpt.py

# Option B: rely on GitHub fetch (see PARAM_GOLF_* above) or a discovered ./parameter-golf/train_gpt.py
```

Convenience: `python3 -m ideator idea ...` is equivalent; the default subcommand is `idea`.

**Typical outputs** under `knowledge_graph/outbox/ideator/` (default save dir):

| Artifact | Role |
|----------|------|
| `latest.json`, `latest_train_gpt.py`, `latest_train_gpt.patch`, `latest_review.json` | Convenience copies of the last successful run |
| `runs/<run_id>/idea.json`, `train_gpt.py`, … | Full run bundle |
| `<idea_id>.json`, `<idea_id>_train_gpt.py`, `<idea_id>_review.json`, … | Stable names for queueing |
| `review_failures/` | When the reviewer never approves within max rounds |

### 3. Falsify an approved idea

Put the approved candidate at `knowledge_graph/inbox/approved/<idea_id>.json` (copy or symlink from `outbox/ideator/<idea_id>.json`), **or** pass `--candidate-json` to any JSON the falsifier accepts.

```bash
# Recommended: load by idea id (defaults graph to knowledge_graph/graph.json when present)
uv run python -m falsifier.main \
  --idea-id "<idea_id>" \
  --knowledge-dir knowledge_graph \
  --output-json knowledge_graph/outbox/falsifier/<idea_id>_result.json

# Or: explicit candidate file + graph
uv run python -m falsifier.main \
  --candidate-json knowledge_graph/inbox/approved/<idea_id>.json \
  --graph-path knowledge_graph/graph.json \
  --output-json knowledge_graph/outbox/falsifier/<idea_id>_result.json
```

Lock files go under `knowledge_graph/work/in_falsification/`. Use `--calibrate` / `--train-gpt` for calibration mode (`python -m falsifier.main --help`).

**Verdicts** (see `falsifier/types.py`): include `REJECTED`, `REFUTED`, `IMPLEMENTATION_FAIL`, `STAGE_1_PASSED`, `STAGE_2_PASSED`.

## System architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ideator   │────→│   Reviewer   │     │  Falsifier  │
│  (Gemini)   │     │  (OpenAI)    │────→│ (T2–T7 + S2)│
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  Knowledge Graph    │
              │  (Source of Truth)  │
              └─────────────────────┘
```

Multi-hypothesis **Anthropic** runs (parallel path): `experiments/ten_hypothesis_run/run_full_live_experiment.py` → falsifier → graph snapshots / viz.

## Repo layout

- `ideator/` — Gemini ideation, OpenAI novelty reviewer, patch application to `train_gpt.py`
- `falsifier/` — Stage 1 (T2–T7), Stage 2 (adversarial / trends), graph updates
- `knowledge_graph/` — Unified store
  - `seed_parameter_golf_kg.json` — Base hierarchy (RootBox / Branch / Leaf)
  - `visuals/` — Graphviz / matplotlib / ffmpeg assets; see **Knowledge graph visuals** below
  - `outbox/ideator/` — Generated ideas
  - `outbox/falsifier/` — Falsification results
  - `inbox/approved/` — Queue for falsification
  - `work/in_falsification/` — Lock files
  - `graph.json` — Unified graph (nodes + edges)
- `infra/agents/` — Symphony orchestration
- `research/` — Baseline profiles, probe library
- `records/` — Optional Parameter Golf submission trees (`track_10min_16mb/`, `track_non_record_16mb/`); large — keep local copies out of git if you want a smaller clone (see per-track READMEs)
- `docs/prd/` — Specifications
- `experiments/ten_hypothesis_run/` — Anthropic multi-hypothesis runs with `live_run_*` artifacts; see `experiments/ten_hypothesis_run/README.md`

## Multi-hypothesis experiments (Anthropic)

```bash
cd experiments/ten_hypothesis_run
export ANTHROPIC_API_KEY="..."   # or source ../../.env
python3 run_full_live_experiment.py --num-hypotheses 10
```

Optional: `bash run_full_experiment.sh`. Cursor Cloud: `experiments/ten_hypothesis_run/CURSOR_CLOUD_SETUP.md`.

## Knowledge graph visuals

Static renders and movies live under `knowledge_graph/visuals/`. Requires **Graphviz** (`dot`) for PNG/SVG; **matplotlib** for timeline charts; **ffmpeg** for MP4.

### Full seed ontology + hypothesis branches (Graphviz)

Renders `seed_parameter_golf_kg.json` plus experiment ideas as dashed links (see `generate_evolution_movie_v2.find_best_parent_for_hypothesis`).

```bash
python3 knowledge_graph/visuals/render_original_kg_with_branches.py \
  --source merged \
  --output knowledge_graph/visuals/original_kg_with_branches.png

python3 knowledge_graph/visuals/render_original_kg_with_branches.py \
  --experiment-dir experiments/ten_hypothesis_run/live_run_20260328_184308 \
  --output knowledge_graph/visuals/original_kg_with_branches_one_run.png

python3 knowledge_graph/visuals/render_original_kg_with_branches.py --source merged --svg
```

Helpers: `knowledge_graph/visuals/hypothesis_sources.py`.

### Evolution movie (MP4)

```bash
python3 knowledge_graph/visuals/generate_evolution_movie_v2.py \
  --merged --duration-seconds 10 \
  --output knowledge_graph/visuals/evolution_merged_10s.mp4

python3 knowledge_graph/visuals/generate_evolution_movie_v2.py \
  --experiment-dir experiments/ten_hypothesis_run/live_run_20260328_184308 \
  --duration-seconds 10 \
  --output knowledge_graph/visuals/evolution_one_run_10s.mp4
```

Still frame: `knowledge_graph/visuals/evolution_summary_v2.png`.

### Other charts (optional)

- All live runs vs time: `python3 experiments/ten_hypothesis_run/visualize_all_live_runs_timeline.py` → default `knowledge_graph/visuals/all_live_experiments_timeline.png`
- Parent tier evolution: `python3 knowledge_graph/visuals/visualize_parent_child_evolution.py` → `knowledge_graph/visuals/parent_child_evolution.png`

## Component details

### Ideator (`ideator/`)

- Loads knowledge context from `knowledge_graph/`
- One idea per run; reviewer loop with revision on reject
- Emits schema `ideator.idea.v2` with artifacts and `falsifier_instructions`

### Falsifier (`falsifier/`)

- **Stage 1:** T2 → T3 → T4 & T5 (theory-type routing may skip T4/T5) → T7
- **Stage 2:** Adversarial hypotheses, experiments, trend checks (when run)
- Updates the knowledge graph when `--graph-path` / default graph is used with `--idea-id`

### Knowledge graph (`knowledge_graph/`)

- Nodes progress through statuses such as **GENERATED** → **APPROVED** (queue) → **IN_FALSIFICATION** → terminal outcomes aligned with falsifier verdicts (there is **no** `PENDING_REVIEW` status in code—review happens inside the ideator before save)
- Failures and tags feed back into later ideation

## Development

```bash
uv sync --extra dev
uv run pytest

python infra/agents/scripts/check_symphony_readiness.py
```

## Documentation

- `docs/prd/FALSIFIER_REVISED_PRD.md` — Falsifier specification
- `docs/prd/CALIBRATION_LITE.md` — Calibration
- `docs/prd/EXECUTION_ADMISSION_GATE.md` — Execution validation
- `docs/COMPLEXITY_AUDIT.md` — Pipeline / schema notes (ideator ↔ falsifier)
- `experiments/ten_hypothesis_run/README.md` — Multi-hypothesis runner
- `infra/agents/docs/FALSIFIER_V1_PRD.md` — Original falsifier PRD

## Integration principles

- Knowledge graph is the source of truth for queued work and outcomes
- File-based storage (debuggable, version-control friendly)
- Lock files reduce duplicate falsification
- Rich failure analysis feeds the ideator context
- Schema versioning for backward compatibility

## License

MIT
