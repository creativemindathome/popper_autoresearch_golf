#!/usr/bin/env python3
"""Kimi 2.5 (Moonshot AI) client for ideator.

Supports Kimi 2.5 for idea generation with strong engineering reasoning.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class KimiError(RuntimeError):
    """Error from Kimi API."""
    pass


@dataclass(frozen=True)
class KimiHTTPError(KimiError):
    """HTTP error from Kimi API."""
    status_code: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        base = f"Kimi API HTTP {self.status_code}: {self.message}"
        if self.body:
            return f"{base}\n{self.body}"
        return base


class KimiClient:
    """Client for Moonshot AI Kimi API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.moonshot.cn",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self._api_key}",
        }

        body_bytes: Optional[bytes] = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            method=method,
            headers=headers,
            data=body_bytes,
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            msg = raw
            try:
                parsed = json.loads(raw)
                msg = parsed.get("error", {}).get("message") or parsed.get("message") or msg
            except Exception:
                pass
            raise KimiHTTPError(status_code=int(e.code), message=str(msg), body=raw) from e
        except urllib.error.URLError as e:
            raise KimiError(f"Kimi API network error: {e}") from e
        except json.JSONDecodeError as e:
            raise KimiError(f"Kimi API returned non-JSON response: {e}") from e

    def generate_idea(
        self,
        *,
        model: str = "kimi-k2.5",
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        """Generate idea using Kimi 2.5.

        Args:
            model: Kimi model name
            system_prompt: System context with engineering constraints
            user_prompt: User request
            temperature: Sampling temperature (0-1)
            max_tokens: Max output tokens

        Returns:
            Generated text (JSON string)
        """
        url = self._url("v1/chat/completions")

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = self._request_json("POST", url, payload)

        # Extract content from response
        choices = response.get("choices", [])
        if not choices:
            raise KimiError("No choices in response")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        return content

    def generate_with_constraints(
        self,
        *,
        parent_train_gpt_code: str,
        what_and_why_guidance: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate idea with strict engineering constraints.

        This uses a specialized prompt that emphasizes:
        - Parameter budget limits
        - Memory constraints
        - Training stability
        - Falsifiability
        """
        system_prompt = f"""You are an expert ML systems engineer using Kimi 2.5. Your task is to generate novel, BOLD, but PRACTICAL transformer architecture modifications.

CRITICAL ENGINEERING CONSTRAINTS - YOU MUST RESPECT THESE:

1. **Parameter Budget: STRICT 10M parameter limit**
   - Count every parameter: embeddings, attention, FFN, norms, biases
   - Current baseline: GPT-2 small (124M) is TOO BIG - we need 10M max
   - Example valid sizes: 6M (n_embd=384, n_layer=6, n_head=6), 8M, 10M
   - IF you propose >10M, it will be REJECTED immediately by T2 Budget gate

2. **Memory Constraints**
   - Target: <2GB peak memory during training
   - No massive activation caches
   - Efficient attention (no O(n²) for long sequences)

3. **Training Stability Requirements**
   - Must initialize with variance scaling (GPT-2 style)
   - No gradient explosions at start (check initial loss < 15)
   - Must compile and run without errors

4. **Falsifiability**
   - Must be testable in 100 training steps (T7 Microtrain gate)
   - Must show loss reduction or clear failure mode
   - Avoid "magic" components that can't be measured

5. **Novelty vs Practicality Balance**
   - BOLD ideas welcome, but MUST be grounded in known mechanisms
   - Prefer modifications to attention, normalization, or architecture patterns
   - Avoid: "add more layers" (violates budget), "use quantum computing" (untestable)

OUTPUT FORMAT - Return ONLY valid JSON:
{{
  "theory_id": "unique-lowercase-hyphenated-name",
  "what_and_why": "2-3 paragraphs. Paragraph 1: What specific change you propose (be precise). Paragraph 2: Why it should work, citing specific mechanisms (attention dynamics, gradient flow, etc.). Paragraph 3: Expected falsification signatures - what would prove this wrong?",
  "train_gpt_code": "Complete, runnable Python code implementing the architecture. Must use PyTorch. Include imports, model class, forward pass. CRITICAL: Count parameters and verify <10M.",
  "parent_architecture": "GPT-2 small (modified)",
  "novelty_claims": ["Specific claim 1", "Specific claim 2", "Specific claim 3"],
  "expected_behavior": "What training curves should look like. Initial loss? Loss after 100 steps? What metrics indicate success vs failure?",
  "parameter_estimate": "Explicit count: ~X.M parameters (must be <10M)",
  "risk_factors": ["What could go wrong", "How to detect failure early"]
}}

BASELINE ARCHITECTURE (from parent train_gpt.py):
{parent_train_gpt_code[:2000]}

Remember: A 9M parameter idea that runs is better than a 50M parameter idea that gets killed by T2 Budget gate."""

        user_prompt = f"""Generate a hypothesis for a novel efficient transformer architecture.

Guidance: {what_and_why_guidance}

Requirements:
1. Must respect 10M parameter limit
2. Must be falsifiable in 100 training steps
3. Must compile without errors
4. Must show novel insight, not just "add more parameters"

Return ONLY the JSON object."""

        return self.generate_idea(
            model="kimi-k2.5",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=8192,
        )


def get_kimi_api_key() -> Optional[str]:
    """Get Kimi API key from environment."""
    return os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")


def test_kimi_connection() -> bool:
    """Test if Kimi API is accessible."""
    api_key = get_kimi_api_key()
    if not api_key:
        print("KIMI_API_KEY or MOONSHOT_API_KEY not set")
        return False

    try:
        client = KimiClient(api_key=api_key)
        response = client.generate_idea(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Kimi 2.5 is ready' and nothing else.",
            max_tokens=50,
            temperature=0,
        )
        print(f"✓ Kimi 2.5 connected: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Kimi API error: {e}")
        return False


if __name__ == "__main__":
    # Test the client
    print("Testing Kimi 2.5 client...")
    if test_kimi_connection():
        print("\n✓ Client ready for use")
    else:
        print("\n✗ Set KIMI_API_KEY environment variable")
        print("  Get key from: https://platform.moonshot.cn/")
