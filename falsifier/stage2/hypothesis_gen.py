"""
Generate kill hypotheses using LLM (Anthropic Claude).

Gracefully degrades to simple heuristics when ANTHROPIC_API_KEY is unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..types import Calibration, FalsifierInput, KillHypothesis, KnowledgeGraph, Confidence

FALSIFIER_SYSTEM_PROMPT = """You are the Falsifier. Your job is to KILL theories about neural network architecture modifications. You are a critic whose only goal is to find the specific, concrete way this idea will fail.

You have three information sources:

1. THE THEORY: A proposed modification with a paragraph explaining what & why.

2. THE CHECKPOINT PROFILE: Detailed measurements from the current trained SOTA model — gradient norms, activation norms, effective ranks, singular value spectra, attention entropy per head.

3. THE KNOWLEDGE GRAPH: Every previous theory that was tested, including what worked, what failed, and WHY it failed.

Generate 3-5 KILL HYPOTHESES. Each must include:
- WHAT will go wrong (the failure mode)
- WHY (grounded in specific numbers from the checkpoint or graph)
- HOW to test it (experiment type + metric + threshold + step)
- CONFIDENCE (high / medium / low)

OUTPUT: JSON array of kill hypotheses:
{
  "hypothesis_id": "H1",
  "confidence": "high",
  "failure_mode": "...",
  "mechanism": "...",
  "experiment_type": "isolation|temporal|component|interaction|absolute|relative",
  "experiment_spec": {
    "metric": "...",
    "threshold": ...,
    "comparator": ">",
    "step": 500,
    "needs_ablation": false,
    "ablation_target": null,
    "component_to_instrument": null,
    "temporal_pattern": null
  },
  "evidence": "... (specific numbers from checkpoint or graph)"
}
"""


def generate_kill_hypotheses(
    inp: FalsifierInput,
    stage1_results: dict[str, Any],
) -> list[KillHypothesis]:
    """Generate adversarial kill hypotheses.

    Uses Composer 2 if available, falls back to standard Anthropic,
    then to heuristics.
    """
    # Try Composer 2 first (advanced reasoning)
    try:
        from .composer_falsifier import Composer2Falsifier
        composer = Composer2Falsifier()
        print("[hypothesis_gen] Using Composer 2 for advanced falsification...")
        return composer.generate_kill_hypotheses(inp, stage1_results)
    except Exception as e:
        print(f"[hypothesis_gen] Composer 2 not available: {e}")

    # Fall back to standard Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Fallback: generate hypotheses from Stage 1 tags
        return _generate_fallback_hypotheses(inp, stage1_results)

    try:
        return _generate_llm_hypotheses(inp, stage1_results, api_key)
    except Exception as e:
        print(f"[hypothesis_gen] LLM failed, using fallback: {e}")
        return _generate_fallback_hypotheses(inp, stage1_results)


def _generate_llm_hypotheses(
    inp: FalsifierInput,
    stage1_results: dict[str, Any],
    api_key: str,
) -> list[KillHypothesis]:
    """Use Claude to generate kill hypotheses."""
    try:
        import anthropic
    except ImportError:
        print("[hypothesis_gen] anthropic not installed, using fallback")
        return _generate_fallback_hypotheses(inp, stage1_results)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    context = _build_falsifier_context(inp, stage1_results)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=FALSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    
    # Parse JSON from response
    content = response.content[0].text if response.content else "[]"
    
    try:
        # Try to extract JSON array
        if "[" in content and "]" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        
        data = json.loads(content)
        hypotheses = [_parse_hypothesis(h) for h in data if isinstance(h, dict)]
        
        # Sort by confidence, keep top 5
        hypotheses.sort(key=lambda h: {"high": 0, "medium": 1, "low": 2}[h.confidence])
        return hypotheses[:5]
        
    except json.JSONDecodeError as e:
        print(f"[hypothesis_gen] JSON parse failed: {e}")
        return _generate_fallback_hypotheses(inp, stage1_results)


def _generate_fallback_hypotheses(
    inp: FalsifierInput,
    stage1_results: dict[str, Any],
) -> list[KillHypothesis]:
    """Generate simple hypotheses from Stage 1 tags when LLM unavailable."""
    hypotheses: list[KillHypothesis] = []
    
    # Get tags from Stage 1 results
    tags: list[dict[str, Any]] = []
    for test_id, result in stage1_results.items():
        if hasattr(result, "tags"):
            for tag in result.tags:
                tags.append({
                    "tag_id": tag.tag_id,
                    "test_id": tag.test_id,
                    "category": tag.category,
                    "description": tag.description,
                })
    
    # Generate hypothesis for each concerning tag
    for i, tag in enumerate(tags[:3]):
        h = KillHypothesis(
            hypothesis_id=f"H{i+1}",
            confidence="medium",
            failure_mode=f"Escalation of {tag['tag_id']}: {tag['description']}",
            mechanism=f"{tag['category']} observed in Stage 1 persists or worsens at 500 steps",
            experiment_type="relative",
            experiment_spec={
                "metric": tag["tag_id"],
                "threshold": 1.5 if "ratio" in tag["tag_id"] else 0.5,
                "comparator": ">",
                "step": 500,
                "needs_ablation": False,
                "ablation_target": None,
            },
            evidence=f"Stage 1 {tag['test_id']}: {tag['description']}",
        )
        hypotheses.append(h)
    
    # Add a generic hypothesis if no tags
    if not hypotheses:
        hypotheses.append(KillHypothesis(
            hypothesis_id="H1",
            confidence="low",
            failure_mode="Theory diverges during extended training",
            mechanism="Short-term stability (100 steps) doesn't guarantee long-term convergence",
            experiment_type="absolute",
            experiment_spec={
                "metric": "loss",
                "threshold": 100.0,
                "comparator": ">",
                "step": 500,
                "needs_ablation": False,
            },
            evidence="No specific Stage 1 warnings, but extended training may reveal instability",
        ))
    
    return hypotheses


def _build_falsifier_context(
    inp: FalsifierInput,
    stage1_results: dict[str, Any],
) -> str:
    """Build context for the LLM."""
    lines = [
        f"## THE THEORY",
        f"ID: {inp.theory_id} | Type: {inp.theory_type}",
        f"",
        f"### What and Why",
        f"{inp.what_and_why}",
        f"",
        f"### Config Changes",
        json.dumps(inp.config_delta, indent=2) if inp.config_delta else "None",
        f"",
        f"### Parents",
        json.dumps([{"node_id": p.node_id, "relationship": p.relationship} for p in inp.parents]),
        f"",
        f"## STAGE 1 RESULTS",
    ]
    
    for test_id, result in stage1_results.items():
        if result is None:
            continue
        status = getattr(result, "status", "?")
        lines.append(f"- {test_id}: {status}")
        if hasattr(result, "kill_reason") and result.kill_reason:
            lines.append(f"  Kill reason: {result.kill_reason}")
        if hasattr(result, "tags") and result.tags:
            for tag in result.tags:
                lines.append(f"  Tag: {tag.tag_id} - {tag.description}")
    
    lines.append("")
    lines.append("## TRAINED CHECKPOINT PROFILE")
    
    cal = inp.calibration
    if cal:
        lines.append(f"Component Gradient Norms: {json.dumps(cal.sota_component_gradient_norms)}")
        lines.append(f"Gradient Norm Ratio: {cal.sota_gradient_norm_ratio:.2f}")
        lines.append(f"Output Entropy: {cal.sota_output_entropy:.4f}")
        lines.append(f"Init Logit Max: {cal.sota_init_logit_max:.2f}")
    else:
        lines.append("No calibration available")
    
    lines.append("")
    lines.append("## YOUR TASK")
    lines.append("Generate 3-5 kill hypotheses. Ground each in SPECIFIC NUMBERS from the checkpoint or graph. Specify the EXACT experiment.")
    
    return "\n".join(lines)


def _parse_hypothesis(data: dict[str, Any]) -> KillHypothesis:
    """Parse hypothesis from JSON dict."""
    confidence = data.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    
    return KillHypothesis(
        hypothesis_id=data.get("hypothesis_id", "H?"),
        confidence=confidence,  # type: ignore
        failure_mode=data.get("failure_mode", "Unknown failure"),
        mechanism=data.get("mechanism", "Unknown mechanism"),
        experiment_type=data.get("experiment_type", "absolute"),
        experiment_spec=data.get("experiment_spec", {}),
        evidence=data.get("evidence", "No evidence provided"),
    )


def validate_hypothesis(h: KillHypothesis, inp: FalsifierInput) -> bool:
    """Basic validation of a hypothesis."""
    # Check required fields
    if not h.hypothesis_id:
        return False
    if not h.failure_mode or not h.mechanism:
        return False
    
    # Check experiment spec
    spec = h.experiment_spec
    if not isinstance(spec, dict):
        return False
    
    return True
