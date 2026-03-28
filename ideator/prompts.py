from __future__ import annotations

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

Hard constraints:
- The idea must be implementable by a small team in ~12 hours as an MVP.
- It must plausibly fit Parameter Golf constraints (tiny artifact, short training, fixed data).
"""


def build_ideator_prompts(*, knowledge_context: str) -> Tuple[str, str]:
    system = IDEATOR_SYSTEM.strip()
    user = f"""\
{PARAMETER_GOLF_CONTEXT.strip()}

Knowledge graph context (may be empty):
{knowledge_context.strip() or "[empty]"}

Task:
Generate exactly ONE new research idea to improve Parameter Golf. The idea should not be a trivial hyperparameter tweak.

Return a JSON object with:
- schema_version: must be "ideator.idea.v1"
- idea_id: short stable slug (lowercase, hyphens)
- title: short title
- novelty_summary: 1–3 sentences describing what's new
- parent_implementation: pointer to the EXACT parent codebase the falsifier should modify and run
  - repo_url: must be "https://github.com/openai/parameter-golf"
  - primary_file: must be "train_gpt.py"
  - run_command: a single command (or env-var + command string) the falsifier can run to test
  - code_search_hints: 3–8 ripgrep-style search strings to locate the relevant section(s) in train_gpt.py
- implementation_steps: 3–7 concrete steps; each step must include:
  - step_id: short slug
  - file: must be "train_gpt.py" unless you justify otherwise
  - locate: how to find the insertion/edit point (anchor strings to search for)
  - change: explicit pseudocode or a small diff-like snippet
  - done_when: measurable acceptance criterion (e.g., script runs; prints val_bpb; artifact size stays under cap)
- falsifier_smoke_tests: 2–5 quick tests with pass/fail criteria (should run in minutes)
- expected_metric_change: expected direction/range on val_bpb and why

IMPORTANT:
- Do NOT mention CIFAR/ImageNet/other datasets. Only refer to Parameter Golf (FineWeb, train_gpt.py, val_bpb).
- Make the implementation instructions unambiguous: anchors + what to insert/replace.

Output JSON only.
""".strip()
    return system, user


def ideator_response_schema() -> Dict[str, Any]:
    # JSON Schema for Gemini structured output. Keep it simple for reliability.
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "idea_id": {"type": "string"},
            "title": {"type": "string"},
            "novelty_summary": {"type": "string"},
            "parent_implementation": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "primary_file": {"type": "string"},
                    "run_command": {"type": "string"},
                    "code_search_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 8,
                    },
                },
                "required": ["repo_url", "primary_file", "run_command", "code_search_hints"],
                "additionalProperties": False,
            },
            "implementation_steps": {
                "type": "array",
                "minItems": 3,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string"},
                        "file": {"type": "string"},
                        "locate": {"type": "string"},
                        "change": {"type": "string"},
                        "done_when": {"type": "string"},
                    },
                    "required": ["step_id", "file", "locate", "change", "done_when"],
                    "additionalProperties": False,
                },
            },
            "falsifier_smoke_tests": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
            },
            "expected_metric_change": {"type": "string"},
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
