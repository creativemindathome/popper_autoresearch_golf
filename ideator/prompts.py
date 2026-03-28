from __future__ import annotations

from typing import Any, Dict, Tuple


PARAMETER_GOLF_CONTEXT = """\
OpenAI Parameter Golf context (for ideation):
- Goal: improve held-out compression on FineWeb validation, reported as bits-per-byte (lower is better).
- Constraints: artifact must be <= 16MB total (weights + *training code* bundled), and training run is time-capped (challenge spec uses 10 minutes on 8×H100).
- Fixed dataset split: you can't use external data; improvements must come from architecture, optimizer/training dynamics, compression/quantization, tokenizer/bytes tricks, or clever test-time compute that fits the rules.
"""


IDEATOR_SYSTEM = """\
You are the Ideator agent in a multi-agent AutoResearch loop. Your job is to propose ONE novel, falsifiable, implementation-ready scientific idea to improve OpenAI Parameter Golf.

You must:
- Use the provided knowledge graph context to avoid repeating prior ideas (including failed/not-novel attempts).
- Prefer ideas that explore "out-of-distribution" research directions (compression, information theory, unconventional architectures, training tricks) but are still implementable under the constraints.
- Produce a single JSON object only (no markdown, no extra commentary).
- All string fields must be single-line (no literal newline characters). If you need a line break, use the two-character escape sequence "\\n".

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
- idea_id: short stable slug (lowercase, hyphens)
- title: short title
- one_liner: 1–2 sentences
- core_hypothesis: what mechanism improves bits-per-byte
- novelty: why this differs from prior graph entries
- components: list of 3–8 components you modify (e.g., attention, loss, optimizer, tokenizer, quantization, schedule, architecture)
- minimal_change: the smallest change to test first (fast falsification)
- full_proposal: what to implement if minimal_change looks promising
- evaluation_plan: 3–6 steps, each with an explicit measurable check
- expected_outcome: what improvement you expect and why (can be a range)
- main_risks: 3–6 risks/failure modes

Output JSON only.
""".strip()
    return system, user


def ideator_response_schema() -> Dict[str, Any]:
    # JSON Schema for Gemini structured output. Keep it simple for reliability.
    return {
        "type": "object",
        "properties": {
            "idea_id": {"type": "string"},
            "title": {"type": "string"},
            "one_liner": {"type": "string"},
            "core_hypothesis": {"type": "string"},
            "novelty": {"type": "string"},
            "components": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 8,
            },
            "minimal_change": {"type": "string"},
            "full_proposal": {"type": "string"},
            "evaluation_plan": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
            "expected_outcome": {"type": "string"},
            "main_risks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
        },
        "required": [
            "idea_id",
            "title",
            "one_liner",
            "core_hypothesis",
            "novelty",
            "components",
            "minimal_change",
            "full_proposal",
            "evaluation_plan",
            "expected_outcome",
            "main_risks",
        ],
        "additionalProperties": False,
    }
