# AutoResearch Loop: Ideator + Falsifier

Hackathon scaffold for an AutoResearch loop focused on OpenAI's Parameter Golf benchmark, combining:
- **Ideator** (Gemini + OpenAI): Generates novel, testable ideas with novelty review
- **Falsifier** (Systematic testing): Validates ideas through T2-T7 gates + Stage 2 adversarial prosecution
- **Knowledge Graph**: Unified store of ideas, failures, and learnings

## Quick Start

### 1. Environment Setup

```bash
# Gemini API key (for ideator)
export GEMINI_API_KEY="..."

# OpenAI API key (for novelty reviewer)
export OPENAI_API_KEY="..."

# Optional configuration
export GEMINI_MODEL="gemini-2.5-flash"
export OPENAI_REVIEWER_MODEL="gpt-4o-mini"
export IDEATOR_MAX_REVIEW_ROUNDS="4"
export IDEATOR_REVIEWER_MIN_SCORE="6"
```

### 2. Generate an Idea

```bash
# Clone parameter-golf parent code
git clone https://github.com/openai/parameter-golf.git parameter-golf

# Generate and review an idea
python3 -m ideator --parent-train-gpt parameter-golf/train_gpt.py

# Outputs:
# - knowledge_graph/outbox/ideator/latest.json
# - knowledge_graph/outbox/ideator/latest_train_gpt.py
# - knowledge_graph/outbox/ideator/latest_review.json
```

### 3. Falsify an Approved Idea

```bash
# First, symlink approved idea to inbox
ln -s knowledge_graph/outbox/ideator/<idea_id>.json knowledge_graph/inbox/approved/

# Run falsifier
python -m falsifier.main \
    --candidate-json knowledge_graph/inbox/approved/<idea_id>.json \
    --graph-path knowledge_graph/graph.json \
    --output-json knowledge_graph/outbox/falsifier/<idea_id>_result.json
```

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ideator   │────→│   Reviewer   │────→│  Falsifier  │
│  (Gemini)   │     │  (OpenAI)    │     │  (T2-T7+S2) │
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

## Repo Layout

- `ideator/` - LLM-powered idea generation with novelty reviewer loop
- `falsifier/` - Systematic validation (Stage 1: T2-T7 gates, Stage 2: adversarial prosecution)
- `knowledge_graph/` - Unified knowledge store
  - `seed_parameter_golf_kg.json` - Base knowledge hierarchy (RootBox/Branch/Leaf)
  - `outbox/ideator/` - Generated ideas
  - `outbox/falsifier/` - Falsification results
  - `inbox/approved/` - Queue for falsification
  - `work/in_falsification/` - Lock files for in-progress work
  - `graph.json` - Unified graph (all nodes + edges)
- `infra/agents/` - Symphony orchestration infrastructure
- `research/` - Baseline profiles, probe library
- `docs/prd/` - Detailed specifications

## Component Details

### Ideator (`ideator/`)
- Reads knowledge context from `knowledge_graph/`
- Generates one novel, falsifiable idea per run
- Runs pessimistic novelty reviewer (OpenAI)
- Auto-revises up to max rounds if rejected
- Emits only reviewer-approved ideas

### Falsifier (`falsifier/`)
- **Stage 1**: T2 (budget) → T3 (compilation) → T4 (signal) → T5 (init) → T7 (micro-train)
- **Stage 2**: Adversarial hypothesis generation → experiments → trend verification
- **Output**: REFUTED, STAGE_1_PASSED, or STAGE_2_PASSED
- Updates knowledge graph with full failure analysis

### Knowledge Graph (`knowledge_graph/`)
- Single source of truth for all ideas and results
- Tracks full lifecycle: GENERATED → PENDING_REVIEW → APPROVED → IN_FALSIFICATION → REFUTED/PASSED
- Enables learning from failures (pattern recognition)

## Development

```bash
# Setup
uv sync

# Tests
uv run pytest

# Symphony readiness check
python infra/agents/scripts/check_symphony_readiness.py
```

## Documentation

- `docs/prd/FALSIFIER_REVISED_PRD.md` - Falsifier specification
- `docs/prd/CALIBRATION_LITE.md` - Calibration procedures
- `docs/prd/EXECUTION_ADMISSION_GATE.md` - Execution validation
- `infra/agents/docs/FALSIFIER_V1_PRD.md` - Original falsifier PRD

## Integration Principles

- Knowledge graph is the source of truth
- File-based storage (debuggable, version-control friendly)
- Lock files prevent duplicate falsification
- Rich failure analysis feeds back to ideator
- Schema versioning for backward compatibility
