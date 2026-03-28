# Verifier and Reviewer Integration

This note describes the smallest clean way to add a `verifier` stage and a `reviewer` stage to the current repo without breaking the existing handoff-first orchestration model.

## Goal

Keep the current source of truth unchanged:

- local handoffs under `infra/agents/handoffs/` stay authoritative
- Symphony remains the executor/orchestrator
- falsifier artifacts under `research/falsification/` remain the execution evidence
- the knowledge graph under `research/knowledge_graph/` remains the durable learning surface

The `verifier` and `reviewer` should be added as execution roles around the existing workflow, not as a second planning system.

## Recommended Placement

Use this repo split:

- `infra/agents/symphony/`: role prompts and workflow policy
- `infra/agents/scripts/`: machine-enforced gating checks
- `infra/agents/handoffs/`: issue-level declarations for when verifier/reviewer are required
- `research/falsification/`: verifier outputs and evidence summaries
- `research/knowledge_graph/`: promoted learnings, anti-patterns, and durable nodes derived from verifier/reviewer output
- `tests/`: regression coverage for any new gating logic

If you want concrete prompt files, add:

- `infra/agents/symphony/VERIFIER.hermes.md`
- `infra/agents/symphony/REVIEWER.hermes.md`

## Role Boundaries

`verifier`

- runs the issue's declared validation commands
- checks required delivery artifacts exist
- checks required falsifier verdicts or summaries exist when the issue depends on them
- checks that any verifier-specific training surface is present when the issue requires a separate verifier entrypoint
- produces a short machine-readable pass/fail summary
- does not expand scope or rewrite acceptance criteria

`reviewer`

- checks that the delivered change still matches the local handoff file
- checks that repo contracts were not silently changed
- checks that Linear state is justified by local artifacts
- checks that new verifier findings worth preserving are promoted into the knowledge graph
- blocks completion if validation passed but scope or contract integrity failed

In short: verifier answers "did it run and produce the required artifacts?" and reviewer answers "is this the right change, in the right scope, with the right repo contracts?"

## Where They Fit

Recommended issue flow:

1. implementation from the local handoff file
2. verifier runs declared checks and artifact assertions
3. reviewer compares delivery against handoff scope and repo constraints
4. only then can the issue be marked complete or promoted in Linear

This means completion should require both:

- command-level success
- contract-level review success

## Handoff Changes

Each executable handoff should stay in the current template shape, but add or standardize these expectations:

- `Validation`: commands the verifier must run
- `Delivery`: files or artifacts the verifier must assert exist
- `Dependencies`: any required falsifier verdicts, profiles, or summaries
- `Knowledge Graph Impact`: whether the issue creates or updates a node under `research/knowledge_graph/`
- `Verifier Surface`: whether the issue depends on the default repo-root `train_gpt.py` or a verifier-specific training entrypoint
- `Completion Notes`: optional reviewer-specific checks for scope or migration safety

The important constraint is that verifier and reviewer consume the handoff file; they should not infer acceptance criteria from Linear alone.

## Script Integration

The cleanest implementation is to extend existing gating scripts instead of creating a parallel toolchain.

Recommended touch points:

- `infra/agents/scripts/check_symphony_readiness.py`
  Add checks that verifier/reviewer prompt files and required env keys exist.
- `infra/agents/scripts/resolve_handoff_queue.py`
  Prevent release or completion if verifier/reviewer requirements are declared but unresolved.
- `infra/agents/scripts/reconcile_delivery_contracts.py`
  Use this as the base for reviewer-side delivery and contract checks.
- `research/probe_library.py`
  Keep import logic explicit if verifier work introduces a second train surface instead of only the repo-root `train_gpt.py`.

If a new script is needed, keep it narrow, for example:

- `infra/agents/scripts/run_verifier.py`
- `infra/agents/scripts/run_reviewer.py`

Those scripts should emit deterministic pass/fail output so Symphony can gate issue transitions safely.

## Artifact Contract

Do not invent a new artifact tree. Reuse the current evidence layout.

Suggested pattern:

- verifier writes summaries under the relevant execution directory in `research/falsification/`
- verifier-owned reusable conclusions are promoted into `research/knowledge_graph/` as durable nodes, not left only inside run summaries
- reviewer writes a short completion or rejection note under the same issue-specific artifact area, or emits machine-readable status consumed by orchestration

That keeps issue evidence next to the falsifier results that justified the work.

## Verifier Train Node

If verifier work needs its own training entrypoint, treat that as a first-class delivery node rather than an implicit variant.

Recommended pattern:

- keep the existing repo-root `train_gpt.py` as the primary challenge-facing entrypoint
- add a verifier-specific file only if the verifier genuinely needs a separate execution surface
- declare that file explicitly in the handoff `Delivery` section
- make the verifier gate assert that the new file exists and is the one being imported by the verifier path

Examples of acceptable shapes:

- a dedicated file such as `verifier/train_gpt_verifier.py`
- a dedicated package entrypoint such as `verifier/train_gpt.py`

Avoid silently overloading the repo-root `train_gpt.py` with verifier-only behavior unless the migration is explicitly owned by the issue.

## Minimal Rollout

Start with this order:

1. add `VERIFIER.hermes.md` and `REVIEWER.hermes.md`
2. add gating in `resolve_handoff_queue.py` so completion requires verifier/reviewer success
3. standardize handoff expectations for `Validation`, `Delivery`, `Knowledge Graph Impact`, and `Verifier Surface`
4. decide whether verifier uses the repo-root `train_gpt.py` or a new verifier-specific train node
5. add tests for failed validation, missing artifacts, scope mismatch, and missing knowledge-graph promotion when required

This is enough to integrate both roles without changing the repo's core model: local handoffs define scope, scripts enforce gates, Symphony executes, and research artifacts remain the evidence surface.
