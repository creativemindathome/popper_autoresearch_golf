from __future__ import annotations

import json
from typing import Any, Dict, Tuple


PARAMETER_GOLF_CONTEXT = """\
OpenAI Parameter Golf context (for ideation):
- Goal: improve held-out compression on FineWeb validation, reported as bits-per-byte (lower is better).
- Constraints: artifact must be <= 16,000,000 bytes total (code bytes + compressed model bytes), and the leaderboard run is time-capped (10 minutes on 8×H100).
- Fixed dataset split: no external data; improvements must come from architecture, optimizer/training dynamics, compression/quantization, tokenizer/bytes tricks, or clever test-time compute that fits the rules.

Parent implementation (must reference this so a falsifier can implement unambiguously):
- Repo: https://github.com/openai/parameter-golf
- Primary file: train_gpt.py (PyTorch / torchrun path). Code bytes are expected to live in train_gpt.py for the artifact-size accounting.
- Baseline run example (edit paths as needed):
  RUN_ID=baseline_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \\
  torchrun --standalone --nproc_per_node=1 train_gpt.py
- Expected output: prints val_loss, val_bpb, and compressed model size near the end.
"""


IDEATOR_SYSTEM = """\
You are the Ideator agent in a multi-agent AutoResearch loop. Your job is to propose ONE novel, falsifiable, implementation-ready scientific idea to improve OpenAI Parameter Golf.

You must:
- Use the provided knowledge graph context to avoid repeating prior ideas (including failed/not-novel attempts).
- Prefer ideas that explore "out-of-distribution" research directions (compression, information theory, unconventional architectures, training tricks) but are still implementable under the constraints.
- Produce a single JSON object only (no markdown, no extra commentary).
- All string fields must be single-line (no literal newline characters). If you need a line break, use the two-character escape sequence "\\n".
- Optimize for the Parameter Golf constraints: tiny artifact, short training wallclock, minimal additional code size.
- Keep the JSON compact: do NOT paste long code blocks. In `implementation_steps.change`, use brief pseudocode or a small diff-like snippet (keep it under ~800 characters).

Hard constraints:
- The idea must be implementable by a small team in ~12 hours as an MVP.
- It must plausibly fit Parameter Golf constraints (tiny artifact, short training, fixed data).
"""


IDEATOR_REVISION_SYSTEM = """\
You are the Ideator agent in a multi-agent AutoResearch loop. You are revising an idea after a pessimistic novelty review.

Your job is to produce ONE improved, more novel, still-falsifiable, implementation-ready idea.

You must:
- Address the Reviewer's criticisms explicitly by changing the core mechanism, not just rewording.
- Avoid marginal tweaks, repackaged known techniques, or tiny hyperparameter changes.
- Keep the idea implementable under the Parameter Golf constraints and within ~12 hours as an MVP.
- Produce a single JSON object only (no markdown, no extra commentary).
- All string fields must be single-line (no literal newline characters). If you need a line break, use the two-character escape sequence "\\n".
- Keep the JSON compact: do NOT paste long code blocks. In `implementation_steps.change`, use brief pseudocode or a small diff-like snippet (keep it under ~800 characters).
""".strip()


REVIEWER_SYSTEM = """\
You are the Reviewer agent in a multi-agent AutoResearch loop.

Your job is to decide whether the candidate idea is truly novel enough to be worth sending to the falsifier.

Be pessimistic:
- Default to "revise" unless the idea has a clearly novel mechanism (not a common/standard technique or small variant),
  and it is concretely implementable and compliant with the Parameter Golf constraints.

Check:
- Novelty vs knowledge context (do not ignore near-duplicates or common ideas).
- Not a trivial hyperparameter tweak.
- Parameter Golf compliance (FineWeb, val_bpb, train_gpt.py, tiny artifact, short run).
- Implementation specificity (steps and anchors are unambiguous).
- Falsifiability (quick smoke tests; expected metric change is plausible).

Output a single JSON object only (no markdown, no extra commentary).
All string fields must be single-line (no literal newline characters). If you need a line break, use the two-character escape sequence "\\n".

Return JSON with fields:
- decision: "pass" or "revise"
- novelty_score: integer 0..10
- primary_reasons: array of 2..6 short strings
- revision_instructions: a short actionable paragraph telling the Ideator how to make it more novel (>=3 concrete changes)
- must_fix_fields: array of idea top-level fields that must change (e.g. ["novelty_summary","implementation_steps"])
- similar_to_knowledge: array of short strings naming any similar prior ideas from the knowledge context (can be empty)
""".strip()


def build_ideator_prompts(
    *,
    knowledge_context: str,
    parent_code_ref: Dict[str, Any],
) -> Tuple[str, str]:
    system = IDEATOR_SYSTEM.strip()
    parent_ref_json = json.dumps(parent_code_ref, ensure_ascii=False)
    user = f"""\
{PARAMETER_GOLF_CONTEXT.strip()}

Knowledge graph context (may be empty):
{knowledge_context.strip() or "[empty]"}

Parent code reference (the code you must minimally modify):
{parent_ref_json}

Task:
Generate exactly ONE new research idea to improve Parameter Golf. The idea should not be a trivial hyperparameter tweak.

Return a JSON object with:
- schema_version: must be "ideator.idea.v2"
- idea_id: short stable slug (lowercase, hyphens)
- title: short title
- novelty_summary: 1–3 sentences describing what's new
- parent_implementation: pointer to the EXACT parent codebase the falsifier should modify and run
  - repo_url: the repo URL from Parent code reference
  - git_ref: the git ref from Parent code reference (branch or commit)
  - primary_file: must be "train_gpt.py"
  - run_command: a single command (or env-var + command string) the falsifier can run to test
  - code_search_hints: 3–8 ripgrep-style search strings to locate the relevant section(s) in train_gpt.py
- implementation_steps: 3–7 concrete steps; each step must include:
  - step_id: short slug
  - file: must be "train_gpt.py" unless you justify otherwise
  - locate: how to find the insertion/edit point (anchor strings to search for)
  - change: explicit pseudocode or a small diff-like snippet (do NOT paste long code blocks)
  - done_when: measurable acceptance criterion (e.g., script runs; prints val_bpb; artifact size stays under cap)
- falsifier_smoke_tests: 2–5 quick tests with pass/fail criteria (should run in minutes)
- expected_metric_change: expected direction/range on val_bpb and why

IMPORTANT:
- Do NOT mention CIFAR/ImageNet/other datasets. Only refer to Parameter Golf (FineWeb, train_gpt.py, val_bpb).
- Make the implementation instructions unambiguous: anchors + what to insert/replace.
- Focus on ideas that are implementable with a small, targeted patch to train_gpt.py.
- Keep code additions minimal (code bytes count toward the 16MB artifact budget).
- Do not output absolute user paths (e.g. "/Users/..."). Use relative paths and/or env vars in run_command.

Output JSON only.
""".strip()
    return system, user


def build_ideator_revision_prompts(
    *,
    knowledge_context: str,
    parent_code_ref: Dict[str, Any],
    previous_idea: Dict[str, Any],
    reviewer_feedback: Dict[str, Any],
) -> Tuple[str, str]:
    system = IDEATOR_REVISION_SYSTEM
    parent_ref_json = json.dumps(parent_code_ref, ensure_ascii=False)
    prev_json = json.dumps(previous_idea, ensure_ascii=False)
    review_json = json.dumps(reviewer_feedback, ensure_ascii=False)
    user = f"""\
{PARAMETER_GOLF_CONTEXT.strip()}

Knowledge graph context (may be empty):
{knowledge_context.strip() or "[empty]"}

Parent code reference (the code you must minimally modify):
{parent_ref_json}

Previous idea JSON (rejected or needs revision; may be incomplete):
{prev_json}

Reviewer feedback JSON:
{review_json}

Task:
Revise into ONE more-novel idea that addresses the reviewer's reasons and instructions.

Return a JSON object with the same fields and constraints as the original Ideator output:
- schema_version: must be "ideator.idea.v2"
- idea_id: short stable slug (lowercase, hyphens)
- title
- novelty_summary
- parent_implementation: pointer to the EXACT parent codebase the falsifier should modify and run
  - repo_url: the repo URL from Parent code reference
  - git_ref: the git ref from Parent code reference (branch or commit)
  - primary_file: must be "train_gpt.py"
  - run_command: a single command (or env-var + command string) the falsifier can run to test
  - code_search_hints: 3–8 ripgrep-style search strings to locate the relevant section(s) in train_gpt.py
- implementation_steps: 3–7 concrete steps (anchors + explicit changes)
- falsifier_smoke_tests: 2–5
- expected_metric_change

Output JSON only.
""".strip()
    return system, user


def build_reviewer_prompts(*, knowledge_context: str, idea: Dict[str, Any]) -> Tuple[str, str]:
    system = REVIEWER_SYSTEM
    idea_json = json.dumps(idea, ensure_ascii=False)
    user = f"""\
Parameter Golf context (for compliance):
- Goal: improve held-out compression on FineWeb validation, reported as bits-per-byte (lower is better).
- Constraints: artifact <= 16,000,000 bytes total; run time-capped (10 minutes on 8×H100).
- Fixed data split: no external data. Primary file is train_gpt.py in https://github.com/openai/parameter-golf.

Knowledge graph context (may be empty):
{knowledge_context.strip() or "[empty]"}

Candidate idea JSON:
{idea_json}

Task:
Return the review JSON described in the system prompt.
""".strip()
    return system, user


def ideator_response_schema() -> Dict[str, Any]:
    # JSON Schema for Gemini structured output. Keep it simple for reliability.
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "maxLength": 40},
            "idea_id": {"type": "string", "maxLength": 80},
            "title": {"type": "string", "maxLength": 160},
            "novelty_summary": {"type": "string", "maxLength": 1200},
            "parent_implementation": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "maxLength": 300},
                    "git_ref": {"type": "string", "maxLength": 120},
                    "primary_file": {"type": "string", "maxLength": 60},
                    "run_command": {"type": "string", "maxLength": 1200},
                    "code_search_hints": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 140},
                        "minItems": 3,
                        "maxItems": 8,
                    },
                },
                "required": ["repo_url", "git_ref", "primary_file", "run_command", "code_search_hints"],
                "additionalProperties": False,
            },
            "implementation_steps": {
                "type": "array",
                "minItems": 3,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "maxLength": 80},
                        "file": {"type": "string", "maxLength": 120},
                        "locate": {"type": "string", "maxLength": 800},
                        "change": {"type": "string", "maxLength": 1200},
                        "done_when": {"type": "string", "maxLength": 800},
                    },
                    "required": ["step_id", "file", "locate", "change", "done_when"],
                    "additionalProperties": False,
                },
            },
            "falsifier_smoke_tests": {
                "type": "array",
                "items": {"type": "string", "maxLength": 800},
                "minItems": 2,
                "maxItems": 5,
            },
            "expected_metric_change": {"type": "string", "maxLength": 1200},
        },
        "required": [
            "schema_version",
            "idea_id",
            "title",
            "novelty_summary",
            "parent_implementation",
            "implementation_steps",
            "falsifier_smoke_tests",
            "expected_metric_change",
        ],
        "additionalProperties": False,
    }


PATCH_GENERATOR_SYSTEM = """\
You are the Patch Generator agent in a multi-agent AutoResearch loop.

Given a parent `train_gpt.py` and an accepted idea JSON, produce a minimal unified diff (patch) that implements the idea.

Rules:
- Output a single JSON object only (no markdown, no extra commentary).
- The JSON must contain exactly one field: train_gpt_patch.
- train_gpt_patch must be a unified diff from the provided Parent train_gpt.py to the updated train_gpt.py.
- Encode the patch as a SINGLE JSON string using "\\n" escapes (no literal newline characters inside the JSON string).
- Keep the patch minimal: only include hunks for lines that change; do NOT include the full file.
- The patch must ONLY modify train_gpt.py and must include file headers:
  - --- a/train_gpt.py
  - +++ b/train_gpt.py
- Ensure the patch applies cleanly to the provided Parent train_gpt.py.
""".strip()


def build_patch_prompts(*, parent_train_gpt_py: str, accepted_idea: Dict[str, Any]) -> Tuple[str, str]:
    system = PATCH_GENERATOR_SYSTEM
    idea_json = json.dumps(accepted_idea, ensure_ascii=False)
    user = f"""\
Parent train_gpt.py (verbatim):
<BEGIN_TRAIN_GPT_PY>
{parent_train_gpt_py.rstrip()}
<END_TRAIN_GPT_PY>

Accepted idea JSON:
{idea_json}

Task:
Return a JSON object with exactly:
- train_gpt_patch: unified diff string (use "\\n" escapes)
""".strip()
    return system, user


def patch_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"train_gpt_patch": {"type": "string"}},
        "required": ["train_gpt_patch"],
        "additionalProperties": False,
    }


def reviewer_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "novelty_score": {"type": "integer"},
            "primary_reasons": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
            "revision_instructions": {"type": "string"},
            "must_fix_fields": {"type": "array", "items": {"type": "string"}},
            "similar_to_knowledge": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "decision",
            "novelty_score",
            "primary_reasons",
            "revision_instructions",
            "must_fix_fields",
            "similar_to_knowledge",
        ],
        "additionalProperties": False,
    }
