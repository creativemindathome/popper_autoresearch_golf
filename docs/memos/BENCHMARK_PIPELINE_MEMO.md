# Memo: Baseline GPT, falsifier “benchmarks,” and what we learned

**Audience:** engineering + research leads  
**Date:** 2026-03-28  
**Context:** Staff-level review of the informational value of the current test and measurement surface after iterative pipeline work (including removal of the text-novelty gate and focus on code-centric admission).

---

## 1. What “run this GPT” means in this repo

There is no separate **named benchmark suite** (e.g. GLUE, MMLU, or a fixed eval harness) checked into this repository. What exists instead is:

1. **`train_gpt.py`** – the parameter-golf style training entrypoint and model definition.
2. **Deterministic falsifier Stage 1** – a gate sequence that evaluates a **candidate `train_gpt.py` path** plus metadata, not end-task accuracy on a dataset.
3. **`pytest`** – regression tests that stand in for **benchmarks**: each test is a controlled scenario (happy path, budget blow-up, broken import, etc.).
4. **`research/profiles/latest_baseline_profile.json`** – calibration-lite artifacts used for thresholds and micro-train floors.
5. **Execution admission** – a separate contract that says whether a **verdict JSON** (not the Stage 1 struct above) allows “execution project” work.

So “running the GPT” in the falsifier sense means: **point the pipeline at `train_gpt.py`, run validation → T2 → T3 → T4**, and read the verdict artifact.

---

## 2. Live run: baseline `train_gpt.py` through the falsifier (representative)

Command pattern:

```bash
uv run python -m falsifier.main --input candidate.json --output verdict.json
```

with `train_gpt_path` set to the repo root `train_gpt.py` and a `what_and_why` string satisfying validation (minimum word count).

**Observed verdict (memo-smoke run, 2026-03-28):**

| Stage | Label | What it did | Numbers observed (illustrative) |
|--------|--------|-------------|-----------------------------------|
| Validation | `validation` | Checks `theory_id`, `what_and_why` length, file exists, filename `train_gpt.py` | `ok: true` |
| T2 | `T2` | Static **artifact size** (source bytes + compressed estimate) and **FLOPs estimate** vs calibration baseline from profile | `within_budget: true`; `flops_estimate` ~1.48e13; `total_artifact_bytes` ~10.3M |
| T3 | `T3` | Import candidate module with **minimal env**, build small model, forward + backward on CPU | `param_count` 16680 (2 layers × tiny dims); `smoke_loss` ~4.15; `backward_ok: true` |
| T4 | `T4` | **100-step** AdamW micro-train on CPU, same minimal construction | `loss_drop` ~0.0227; `throughput_steps_per_sec` ~500; `ok: true` |
| Outcome | `promote` | All gates passed | `stage_reached: "promote"` |

**Important:** T3/T4 use a **patched minimal environment** (small layers, vocab, seq len) via `falsifier/adapters/parameter_golf.py`, not your full training hyperparameters from the default `Hyperparameters` class for a real run. So the pipeline answers: *“Does this file import, build a model, and learn a bit under stress?”* — not *“What accuracy does this get on FineWeb?”*

---

## 3. “Benchmark” map: each automated experiment and what it actually measures

The **24 pytest tests** are the closest thing to a benchmark matrix. Below, **informational quality** = how much this test tells you about real-world success.

### 3.1 Stage 1 / core (`tests/falsifier/test_stage1.py`, `tests/test_falsifier_core.py`)

| Test | Behavior under test | Signal quality | Gaps |
|------|----------------------|----------------|------|
| Reject invalid candidate | Missing path / bad fields | **High** for plumbing | Does not touch model quality |
| Promote baseline | Full gate chain on real `train_gpt.py` | **High** for regression (“nothing broke”) | Single env; no multi-seed |
| Refute over-budget | Tiny fake `train_gpt.py` with huge env-driven sizes | **High** for T2 logic | Toy source, not real refactors |
| Fail import at T3 | Missing package import | **High** for “obvious break” | Narrow failure mode |
| Fail construction at T3 | `GPT.__init__` raises | **High** for init bugs | Same |
| CLI writes verdict | End-to-end subprocess + JSON | **High** for integration | Same as promote |

### 3.2 Parsing / diff utilities (`test_config_parser`, `test_diff_utils`)

| Test | Measures | Signal quality |
|------|----------|----------------|
| `extract_model_config` | AST-ish extraction of hyperparameters | **Medium** — matches static analysis assumptions |
| Parameter counts / artifact bytes | Budget math | **Medium** — approximations, not measured training cost |
| Diff classification | Categorizing hyperparameter vs architecture edits | **Medium** — useful for tooling, not for scientific claims |

### 3.3 Calibration (`test_calibration_lite`)

| Test | Measures | Signal quality |
|------|----------|----------------|
| Schema validation | `calibration_lite` object shape | **High** for contract stability |
| Profile roundtrip | Extract from JSON | **Medium** — depends on profile freshness |
| Latest profile present | Optional skip if missing | **Low** as a gate — informational only |

### 3.4 Thresholds (`test_thresholds`)

| Test | Measures | Signal quality |
|------|----------|----------------|
| Defaults without profile | Fallback constants | **High** for CI in empty workspaces |
| Load from repo profile | Real `latest_baseline_profile.json` | **High** when profile is committed and valid |

### 3.5 Mechanism probes (`test_mechanism_probes`)

| Test | Measures | Signal quality |
|------|----------|----------------|
| Readonly bundle | Tensor counts, kurtosis key counts from baseline load | **Low–medium** — sanity that probes run, not a scientific benchmark |

### 3.6 Execution admission (`test_execution_admission`)

| Test | Measures | Signal quality |
|------|----------|----------------|
| Missing file | Script fails cleanly | **High** for ops |
| Non-promote verdict | Schema / outcome rules | **High** for policy |
| Accept promote | Eligibility JSON | **High** for handoff to execution |

**Note:** Execution admission tests use a **different verdict schema** (`outcome`, `decision`, `supporting_results`) than the **Stage 1 CLI output** (`verdict`, `stage_reached`). That is a **documentation and integration risk**: two artifacts with similar words (“promote”) but different shapes.

---

## 4. Staff engineer review: informational quality of the current “benchmark suite”

### Strengths

1. **Determinism and speed** – Tests run in seconds; good for CI and regression.
2. **Layered falsification** – T2 (cheap static) before T3/T4 (runtime) is the right economic shape.
3. **Calibration hook** – Baseline profile gives a path to versioned thresholds instead of magic numbers.
4. **Explicit failure modes** – Import, construction, backward, micro-train each have clear failure classes.

### Weaknesses

1. **No task metric** – Nothing measures perplexity, BPB, or downstream accuracy. The pipeline is **admission**, not **evaluation**.
2. **Minimal env mismatch** – T3/T4 use a **minimal** model shape; a candidate could pass gates and still fail at full scale (OOM, numerical issues, distributed assumptions).
3. **Single-seed micro-train** – One seed and 100 steps; easy to pass or fail for unrepresentative reasons.
4. **Micro-train vs smoke env** – `env_overrides` apply to smoke; **T4 micro-train** historically did not always mirror the same overrides (worth verifying in code when iterating).
5. **Two verdict formats** – Stage 1 JSON vs execution admission JSON reduces clarity for humans and agents.

---

## 5. Proposed iterations (prioritized)

1. **Rename or document “benchmark”** – Call the pytest set **Regression matrix** or **Stage 1 fixture suite** to avoid implying dataset benchmarks.
2. **Align env matrix** – If stress overrides are added, apply **the same** overrides to T3 and T4 (or document divergence explicitly).
3. **Optional task eval hook** – Behind a flag, run a **fixed** validation batch or token count from a checked-in shard (costly, but closes the “does it actually work?” gap).
4. **Unify or bridge verdict artifacts** – One schema with a `kind: stage1 | execution_eligibility` or a transformer script.
5. **Multi-seed micro-train** – e.g. min loss drop over 2 seeds vs calibration distribution.
6. **Version field** – `admission_schema_version` in every verdict for forward compatibility.

---

## 6. What we learned from “dummy” pipeline changes and experiments

### 6.1 Text novelty gate (removed)

We **removed the lexical / T1-style gate** and relied on **T2–T4 only** for Stage 1. **Lesson:** For code-first ideators, text similarity was **low signal** and **high friction**; falsification is better anchored in **measurable** behavior (budget, gradients, learning). **Tradeoff:** we no longer have an automated “near-duplicate theory” detector; that must live in **process** (review) or **future** structured claims.

### 6.2 Ideator path (`skip_t1_novelty`, later removed)

We briefly experimented with **skipping** text novelty for code-centric proposals. **Lesson:** The product wanted **two modes** (narrative vs code); consolidating on **no text gate** simplified the contract and tests.

### 6.3 Calibration and thresholds

**Lesson:** Regenerating `latest_baseline_profile.json` and validating `calibration_lite` caught **real gaps** (e.g. missing `train_batch_tokens` in architecture profile). **Benchmark value:** schema tests are **high leverage** for catching silent drift.

### 6.4 Execution vs buildout

**Lesson:** **Admission scripts** (`check_execution_admission.py`) test **policy**, not **model quality**. Keeping that split explicit avoids confusing “promote” in Stage 1 with “allowed to spend GPU hours on execution project.”

---

## 7. Conclusion

- **Running the baseline GPT** through the falsifier yields a **rich structural verdict** (budget, smoke, micro-train) but **not** a benchmark in the ML sense.
- The **pytest suite** is the operational benchmark; its **informational quality** is **strong for admission and regression**, **weak for scientific claims** about end-task performance.
- **Next iteration** should focus on **clarity** (one verdict story), **env consistency** (T3/T4), and **optional** costly evals that connect admission to **measured** quality.

---

## Appendix: Commands used for this memo

```bash
uv run pytest tests/ -v
uv run python -m falsifier.main --input candidate.json --output verdict.json
```

(Exact `train_gpt_path` and `what_and_why` as in the memo-smoke run.)
