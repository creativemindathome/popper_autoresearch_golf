"""Anthropic Claude client for ideator.

Supports Claude 3.5/4 Sonnet for idea generation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class AnthropicError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnthropicHTTPError(AnthropicError):
    status_code: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        base = f"Anthropic API HTTP {self.status_code}: {self.message}"
        if self.body:
            return f"{base}\n{self.body}"
        return base


class AnthropicClient:
    """Client for Anthropic Claude API."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.anthropic.com") -> None:
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
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        
        body_bytes: Optional[bytes] = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, method=method, headers=headers, data=body_bytes)
        
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
            raise AnthropicHTTPError(status_code=int(e.code), message=str(msg), body=raw) from e
        except urllib.error.URLError as e:
            raise AnthropicError(f"Anthropic API network error: {e}") from e
        except json.JSONDecodeError as e:
            raise AnthropicError(f"Anthropic API returned non-JSON response: {e}") from e

    def generate_idea(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 8192,
    ) -> str:
        """Generate idea using Claude.
        
        Args:
            model: Claude model name
            system_prompt: System context
            user_prompt: User request
            temperature: Sampling temperature (0-1)
            max_tokens: Max output tokens
            
        Returns:
            Generated text (JSON string)
        """
        url = self._url("v1/messages")

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        response = self._request_json("POST", url, payload)
        
        # Extract content from response
        content_blocks = response.get("content", [])
        if not content_blocks:
            raise AnthropicError("No content in response")
            
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
                
        return "".join(text_parts)


def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


def test_anthropic_connection() -> bool:
    """Test if Anthropic API is accessible."""
    api_key = get_anthropic_api_key()
    if not api_key:
        print("ANTHROPIC_API_KEY not set")
        return False
        
    try:
        client = AnthropicClient(api_key=api_key)
        # Simple test message
        response = client.generate_idea(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Anthropic API is working' and nothing else.",
            max_tokens=50,
            temperature=0,
        )
        print(f"✓ Anthropic API connected: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Anthropic API error: {e}")
        return False


if __name__ == "__main__":
    # Test the client
    print("Testing Anthropic client...")
    if test_anthropic_connection():
        print("\n✓ Client ready for use")
    else:
        print("\n✗ Set ANTHROPIC_API_KEY environment variable")
