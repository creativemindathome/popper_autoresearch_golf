from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class GeminiError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiHTTPError(GeminiError):
    status_code: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        base = f"Gemini API HTTP {self.status_code}: {self.message}"
        if self.body:
            return f"{base}\n{self.body}"
        return base


class GeminiClient:
    def __init__(self, *, api_key: str, base_url: str = "https://generativelanguage.googleapis.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _url(self, path: str, *, api_version: str = "v1beta") -> str:
        qs = urllib.parse.urlencode({"key": self._api_key})
        return f"{self._base_url}/{api_version}/{path.lstrip('/')}?{qs}"

    def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body_bytes: Optional[bytes]
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if payload is None:
            body_bytes = None
        else:
            body_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, method=method, headers=headers, data=body_bytes)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            msg = raw
            try:
                parsed = json.loads(raw)
                msg = parsed.get("error", {}).get("message") or msg
            except Exception:
                pass
            raise GeminiHTTPError(status_code=int(e.code), message=msg, body=raw) from e
        except urllib.error.URLError as e:
            raise GeminiError(f"Gemini API network error: {e}") from e
        except json.JSONDecodeError as e:
            raise GeminiError(f"Gemini API returned non-JSON response: {e}") from e

    def list_models(self) -> List[Dict[str, Any]]:
        url = self._url("models")
        data = self._request_json("GET", url)
        return list(data.get("models") or [])

    def generate_content(self, *, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = model.replace("models/", "")
        url = self._url(f"models/{model}:generateContent")
        return self._request_json("POST", url, payload)

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 1.2,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> Any:
        payload_camel = {
            "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": int(max_output_tokens),
                "responseMimeType": "application/json",
            },
        }
        if seed is not None:
            payload_camel["generationConfig"]["seed"] = int(seed)
        if response_schema is not None:
            payload_camel["generationConfig"]["responseSchema"] = response_schema

        try:
            raw = self.generate_content(model=model, payload=payload_camel)
        except GeminiHTTPError as e:
            # Fallback for older/newer REST JSON field naming variants.
            if e.status_code == 400 and _looks_like_unknown_field_error(e.body):
                payload_snake = _camel_to_snake_payload(payload_camel)
                raw = self.generate_content(model=model, payload=payload_snake)
            else:
                raise

        text = _extract_text(raw)
        if text is None:
            raise GeminiError(f"Gemini API response missing text: {raw!r}")

        parsed = _extract_json_from_text(text)
        if parsed is None:
            raise GeminiError(f"Model did not return valid JSON. Raw text:\n{text}")
        return parsed


def _extract_text(resp: Dict[str, Any]) -> Optional[str]:
    candidates = resp.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: List[str] = []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str):
            texts.append(t)
    if not texts:
        return None
    return "".join(texts).strip()


def _extract_json_from_text(text: str) -> Optional[Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to recover a JSON object or array embedded in extra text.
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(1)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _looks_like_unknown_field_error(body: Optional[str]) -> bool:
    if not body:
        return False
    return "Unknown name" in body or "Cannot find field" in body or "Invalid JSON payload" in body


def _camel_to_snake_payload(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_camel_to_snake_payload(x) for x in obj]
    if not isinstance(obj, dict):
        return obj

    out: Dict[str, Any] = {}
    for k, v in obj.items():
        snake = _camel_to_snake_key(k)
        out[snake] = _camel_to_snake_payload(v)
    return out


_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake_key(name: str) -> str:
    s1 = _CAMEL_RE_1.sub(r"\1_\2", name)
    return _CAMEL_RE_2.sub(r"\1_\2", s1).lower()

