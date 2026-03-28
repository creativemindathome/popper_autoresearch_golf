#!/usr/bin/env python3
"""Composer 2 Falsifier for Stage 2 adversarial evaluation.

Composer 2 is an advanced reasoning system that:
1. Breaks down theories into mechanistic components
2. Generates specific, testable kill hypotheses
3. Designs targeted experiments
4. Evaluates with strict scientific rigor
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..types import FalsifierInput, KillHypothesis, Confidence


class ComposerError(RuntimeError):
    """Error from Composer API."""
    pass


@dataclass
class ComposerConfig:
    """Configuration for Composer 2 falsifier."""
    model: str = "composer-2-reasoning"
    max_hypotheses: int = 5
    temperature: float = 0.3  # Lower temp for rigorous reasoning
    max_tokens: int = 6000


class Composer2Falsifier:
    """Composer 2 Stage 2 falsifier with advanced reasoning.

    Composer 2 uses structured reasoning to:
    - Decompose theories into falsifiable claims
    - Generate mechanistic kill hypotheses
    - Design targeted experiments
    - Evaluate with scientific rigor
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("COMPOSER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.config = ComposerConfig()

    def _call_composer(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call Composer 2 API.

        Uses Anthropic Claude as the base model for Composer 2 reasoning.
        """
        if not self.api_key:
            raise ComposerError("No API key available. Set COMPOSER_API_KEY or ANTHROPIC_API_KEY")

        # Composer 2 uses Anthropic's API with specialized prompting
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-sonnet-4-20250514",  # Base model for Composer 2
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        req = urllib.request.Request(
            url,
            method="POST",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                response = json.loads(resp.read().decode("utf-8"))
                content = response.get("content", [])
                if content:
                    return content[0].get("text", "")
                return ""
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise ComposerError(f"Composer API error {e.code}: {error_body}")
        except Exception as e:
            raise ComposerError(f"Composer request failed: {e}")

    def generate_kill_hypotheses(
        self,
        inp: FalsifierInput,
        stage1_results: Dict[str, Any],
    ) -> List[KillHypothesis]:
        """Generate kill hypotheses using Composer 2 reasoning.

        Composer 2 uses structured reasoning:
        1. Decompose theory into components
        2. Identify failure modes for each
        3. Design specific experiments
        4. Prioritize by confidence
        """

        system_prompt = """You are Composer 2, an advanced scientific reasoning system specialized in falsifying machine learning theories.

Your task is to generate KILL HYPOTHESES - specific, mechanistic predictions about how a theory will fail.

REASONING PROCESS (follow strictly):

1. **Theory Decomposition**
   Break the theory into 3-5 falsifiable claims:
   - Architecture claim (e.g., "attention modification X improves Y")
   - Mechanism claim (e.g., "gradient flow pattern Z emerges")
   - Empirical claim (e.g., "loss reduction of W% at step N")

2. **Failure Mode Analysis**
   For each claim, identify:
   - The specific mechanism that could fail
   - Early warning signals (by step 50, 100, 200)
   - Root cause (initialization, optimization dynamics, numerical stability)

3. **Experiment Design**
   Design minimal experiments to test each failure mode:
   - Metric to measure
   - Threshold for "failed"
   - Step to check
   - Ablation needed?

4. **Confidence Calibration**
   Assign confidence based on:
   - HIGH: Grounded in specific Stage 1 measurements + established theory
   - MEDIUM: Reasonable extrapolation from related work
   - LOW: Speculative but worth testing

OUTPUT FORMAT - JSON array of kill hypotheses:
[
  {
    "hypothesis_id": "H1",
    "confidence": "high|medium|low",
    "target_claim": "Which theory claim this targets",
    "failure_mode": "Precise description of what will go wrong",
    "mechanism": "Step-by-step causal chain leading to failure",
    "early_signal": "What to observe by step 50",
    "experiment_type": "isolation|temporal|component|interaction|absolute|relative",
    "experiment_spec": {
      "metric": "specific_metric_name",
      "threshold": numeric_value,
      "comparator": ">|<|=",
      "step": 100,
      "needs_ablation": false,
      "ablation_target": null
    },
    "evidence": "Specific numbers from Stage 1 or theory grounding"
  }
]

RULES:
- Generate 3-5 hypotheses maximum
- Each must target a different claim or failure mode
- Must include specific, numeric predictions
- Must cite evidence from Stage 1 results
- NO vague predictions like "it might not work"
- Focus on EARLY detection (first 100-500 steps)"""

        # Build context
        context = self._build_context(inp, stage1_results)

        user_prompt = f"""Generate kill hypotheses for the following theory.

{context}

Generate 3-5 specific, falsifiable kill hypotheses following the reasoning process above.

Return ONLY the JSON array."""

        try:
            response = self._call_composer(system_prompt, user_prompt)

            # Extract JSON
            if "[" in response and "]" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                json_str = response[start:end]
            else:
                json_str = response

            data = json.loads(json_str)

            # Parse into KillHypothesis objects
            hypotheses = []
            for i, h_data in enumerate(data[:self.config.max_hypotheses]):
                confidence = h_data.get("confidence", "medium")
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"

                h = KillHypothesis(
                    hypothesis_id=h_data.get("hypothesis_id", f"H{i+1}"),
                    confidence=confidence,  # type: ignore
                    failure_mode=h_data.get("failure_mode", "Unknown failure mode"),
                    mechanism=h_data.get("mechanism", h_data.get("target_claim", "Unknown mechanism")),
                    experiment_type=h_data.get("experiment_type", "absolute"),
                    experiment_spec=h_data.get("experiment_spec", {}),
                    evidence=h_data.get("evidence", h_data.get("early_signal", "No evidence provided")),
                )
                hypotheses.append(h)

            # Sort by confidence (high -> medium -> low)
            confidence_order = {"high": 0, "medium": 1, "low": 2}
            hypotheses.sort(key=lambda h: confidence_order.get(h.confidence, 3))

            return hypotheses[:self.config.max_hypotheses]

        except json.JSONDecodeError as e:
            print(f"[Composer2] JSON parse error: {e}")
            return self._generate_fallback_hypotheses(inp, stage1_results)
        except Exception as e:
            print(f"[Composer2] Error: {e}")
            return self._generate_fallback_hypotheses(inp, stage1_results)

    def _build_context(
        self,
        inp: FalsifierInput,
        stage1_results: Dict[str, Any],
    ) -> str:
        """Build detailed context for Composer 2."""

        lines = [
            "## THEORY TO FALSIFY",
            f"Theory ID: {inp.theory_id}",
            f"Theory Type: {inp.theory_type}",
            "",
            "### What and Why",
            inp.what_and_why[:2000] if len(inp.what_and_why) > 2000 else inp.what_and_why,
            "",
            "### Config Changes",
            json.dumps(inp.config_delta, indent=2) if inp.config_delta else "None provided",
            "",
            "## STAGE 1 RESULTS (Ground Truth)",
        ]

        # Add each Stage 1 test result
        for test_id, result in stage1_results.items():
            if result is None:
                continue

            status = getattr(result, "status", "unknown")
            lines.append(f"\n### {test_id}: {status}")

            if hasattr(result, "kill_reason") and result.kill_reason:
                lines.append(f"Kill Reason: {result.kill_reason}")

            if hasattr(result, "tags") and result.tags:
                lines.append("Tags:")
                for tag in result.tags:
                    tag_id = getattr(tag, "tag_id", "unknown")
                    desc = getattr(tag, "description", "")
                    lines.append(f"  - {tag_id}: {desc}")

            # Add specific measurements based on test type
            if hasattr(result, "estimated_params"):
                lines.append(f"Parameter Estimate: {result.estimated_params:,}")
            if hasattr(result, "initial_loss"):
                lines.append(f"Initial Loss: {result.initial_loss:.4f}")
            if hasattr(result, "loss_drop"):
                lines.append(f"Loss Drop: {result.loss_drop:.4f}")

        lines.extend([
            "",
            "## FALSIFICATION TASK",
            "Generate hypotheses that predict SPECIFIC failure modes.",
            "Ground every prediction in the Stage 1 measurements above.",
            "Focus on early detection (first 100-500 steps).",
        ])

        return "\n".join(lines)

    def _generate_fallback_hypotheses(
        self,
        inp: FalsifierInput,
        stage1_results: Dict[str, Any],
    ) -> List[KillHypothesis]:
        """Generate simple fallback hypotheses if Composer fails."""
        return [
            KillHypothesis(
                hypothesis_id="H1",
                confidence="medium",
                failure_mode="Theory fails to generalize beyond micro-training",
                mechanism="Short-term improvements (100 steps) don't persist at 500+ steps",
                experiment_type="temporal",
                experiment_spec={
                    "metric": "loss",
                    "threshold": 5.0,
                    "comparator": ">",
                    "step": 500,
                    "needs_ablation": False,
                },
                evidence="Stage 1 only tested 100 steps",
            ),
            KillHypothesis(
                hypothesis_id="H2",
                confidence="low",
                failure_mode="Numerical instability at larger batch sizes",
                mechanism="Gradient patterns change with increased compute",
                experiment_type="absolute",
                experiment_spec={
                    "metric": "grad_norm",
                    "threshold": 100.0,
                    "comparator": ">",
                    "step": 300,
                    "needs_ablation": True,
                    "ablation_target": "batch_size",
                },
                evidence="Limited gradient testing in Stage 1",
            ),
        ]


def get_composer_api_key() -> Optional[str]:
    """Get Composer API key from environment."""
    return os.environ.get("COMPOSER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def test_composer_connection() -> bool:
    """Test if Composer 2 API is accessible."""
    api_key = get_composer_api_key()
    if not api_key:
        print("COMPOSER_API_KEY or ANTHROPIC_API_KEY not set")
        return False

    try:
        composer = Composer2Falsifier(api_key)
        # Simple test
        response = composer._call_composer(
            system_prompt="You are Composer 2. Say 'Composer 2 is ready for falsification.'",
            user_prompt="Confirm you're ready.",
        )
        print(f"✓ Composer 2 connected: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Composer 2 error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Composer 2 falsifier...")
    if test_composer_connection():
        print("\n✓ Composer 2 ready for falsification")
    else:
        print("\n✗ Set ANTHROPIC_API_KEY environment variable")
