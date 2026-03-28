# Null Fellow Hackathon

Integration note: [Verifier and Reviewer Integration](/Users/curiousmind/Desktop/null_fellow_hackathon/infra/agents/docs/VERIFIER_REVIEWER_INTEGRATION.md)

Repo-local scaffold for a falsifier-first `parameter-golf` workflow with:

- upstream-compatible training/eval integration
- Symphony + Linear orchestration under `infra/agents/`
- research evidence under `research/`
- record packaging under `records/`
- upstream challenge code imported at repo root (`train_gpt.py`, `train_gpt_mlx.py`, `data/`, `requirements.txt`)
- `uv` as the preferred dependency and contributor environment manager

## Repo Layout

- `infra/agents/`: orchestration prompts, env templates, scripts, handoffs, runtime state
- `research/profiles/`: baseline probe snapshots
- `research/theories/`: theory specs
- `research/falsification/`: experiment specs, results, verdicts
- `research/knowledge_graph/`: distilled learnings and anti-patterns
- `records/track_10min_16mb/`: record-track candidate runs
- `records/track_non_record_16mb/`: non-record runs

## Quick Start

1. Fill `infra/agents/env/.env.symphony` from the example file.
2. Fill `infra/agents/env/.env.hermes` from the example file.
3. Run `uv sync`.
4. Run `infra/agents/scripts/bootstrap_repo.sh`.
5. Run `uv run pytest`.
6. Run `uv run python infra/agents/scripts/check_symphony_readiness.py`.
7. Launch Symphony with `infra/agents/scripts/run_symphony.sh`.

The actual agent society can evolve later. The execution substrate is fixed here so Codex, Cursor, Symphony, and Linear all have predictable paths and contracts.

## Real Integration Status

- The upstream OpenAI `parameter-golf` code surface is now present locally.
- `run_baseline_profile.py` performs actual architecture and quantization analysis against the imported PyTorch model surface.
- `run_falsification_batch.py` executes real probe batches against the same surface and can use threshold-based refutation rules from a JSON experiment spec.
- Deeper activation probes and no-training ablations are not wired yet; the current probe layer is weight/checkpoint based.
- A reduced falsifier package now exists under `falsifier/` with typed contracts, a deterministic Stage 1 core (`T0`, `T2`), and tests.

## Project Split

- Buildout/control-plane work runs in the active Symphony project named in `SYMPHONY_LINEAR_PROJECT_SLUG`.
- Actual falsifier/execution research should live in a separate execution project named by `SYMPHONY_EXECUTION_PROJECT_SLUG`.
- This keeps repo plumbing and agent/runtime work separate from theory-testing and candidate evaluation work.

## Local Constraints

### Test-Driven Development

- Every non-trivial code change must start with a validation target before implementation begins.
- Every issue must include at least one concrete validation command under `Validation`.
- If a test does not exist yet, the issue must define the smallest reproducible failing check or smoke command that captures the expected behavior.
- An issue is not complete until its declared validation commands pass after the change.
- Agents must prefer narrow, behavior-level tests over broad manual verification.
- If a change affects orchestration, handoff parsing, verdict generation, or Linear synchronization, the issue must include a regression check for that surface.

### Non-Breakage Policy

- Preserve existing repo paths, artifact contracts, and issue-template headings unless the issue explicitly owns a migration.
- Do not mark work complete if a required delivery artifact, validation output, or verdict file is missing.
- Changes to scripts under `infra/agents/scripts/` must preserve machine-readable exit behavior so Symphony can continue to route issues safely.

### Linear and Handoff Safety

- Checked-in handoffs under `infra/agents/handoffs/` remain the repo-local source of truth; Linear must reflect them, not diverge from them.
- Agents must not mutate Linear issue scope, dependencies, or completion state unless the corresponding local handoff file supports that change.
- A Linear issue may move forward only when its local validation and delivery contract still match the repo state.
- If Linear and local handoffs disagree, agents must stop and reconcile the mismatch before proceeding.
