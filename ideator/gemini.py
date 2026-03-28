from __future__ import annotations

import json
import copy
import re
import socket
import time
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
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_s: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._max_retries = max(0, int(max_retries))

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
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
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
                reason = getattr(e, "reason", None)
                is_timeout = isinstance(reason, socket.timeout) or "timed out" in str(reason).lower()
                if is_timeout and attempt + 1 < attempts:
                    time.sleep(0.8 * (2**attempt))
                    continue
                if is_timeout:
                    raise GeminiError(
                        f"Gemini API request timed out after {self._timeout_s:.0f}s. "
                        "Try setting GEMINI_TIMEOUT_S=300 or lowering --max-output-tokens."
                    ) from e
                raise GeminiError(f"Gemini API network error: {e}") from e
            except (TimeoutError, socket.timeout) as e:
                if attempt + 1 < attempts:
                    time.sleep(0.8 * (2**attempt))
                    continue
                raise GeminiError(
                    f"Gemini API request timed out after {self._timeout_s:.0f}s. "
                    "Try setting GEMINI_TIMEOUT_S=300 or lowering --max-output-tokens."
                ) from e
            except json.JSONDecodeError as e:
                raise GeminiError(f"Gemini API returned non-JSON response: {e}") from e
        raise GeminiError("Gemini API request failed unexpectedly.")

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
        base_payload_camel = {
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
            base_payload_camel["generationConfig"]["seed"] = int(seed)

        raw = self._generate_with_schema_fallbacks(
            model=model,
            base_payload_camel=base_payload_camel,
            response_schema=response_schema,
        )

        text = _extract_text(raw)
        if text is None:
            raise GeminiError(f"Gemini API response missing text: {raw!r}")

        parsed = _parse_json_relaxed(text)
        if parsed is not None:
            return parsed

        # One repair attempt using the model itself (low temperature).
        repaired = self._repair_json_with_model(
            model=model,
            raw_text=text,
            response_schema=response_schema,
        )
        if repaired is not None:
            return repaired

        raise GeminiError(f"Model did not return valid JSON. Raw text:\n{text}")

    def _generate_with_schema_fallbacks(
        self,
        *,
        model: str,
        base_payload_camel: Dict[str, Any],
        response_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        schema_attempts: List[Dict[str, Any]] = []

        if response_schema is not None:
            # Newer structured-output field.
            schema_attempts.append({"_responseJsonSchema": response_schema})
            # Some clients/APIs expose the same concept without the leading underscore.
            schema_attempts.append({"responseJsonSchema": response_schema})
            # Older structured-output field.
            schema_attempts.append({"responseSchema": _json_schema_to_openapi_schema(response_schema)})

        # Last resort: JSON mime-type only.
        schema_attempts.append({})

        last_err: Optional[Exception] = None
        for schema_fields in schema_attempts:
            payload_camel = copy.deepcopy(base_payload_camel)
            payload_camel["generationConfig"].update(schema_fields)
            try:
                return self.generate_content(model=model, payload=payload_camel)
            except GeminiHTTPError as e:
                last_err = e
                if e.status_code == 400 and _looks_like_unknown_field_error(e.body):
                    # Try snake_case as a compatibility fallback for this attempt.
                    try:
                        payload_snake = _camel_to_snake_payload(payload_camel)
                        return self.generate_content(model=model, payload=payload_snake)
                    except GeminiHTTPError as e2:
                        last_err = e2
                        if e2.status_code == 400 and _looks_like_unknown_field_error(e2.body):
                            continue
                        raise
                    except GeminiError as e2:
                        last_err = e2
                        continue
                else:
                    raise
            except GeminiError as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        raise GeminiError("Gemini API request failed unexpectedly.")

    def _repair_json_with_model(
        self,
        *,
        model: str,
        raw_text: str,
        response_schema: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        raw_text = raw_text.strip()
        if not raw_text:
            return None

        system_prompt = (
            "You are a strict JSON repair tool. Return a single valid JSON object only. "
            "Do not add commentary. Do not wrap in markdown."
        )
        schema_hint = ""
        if response_schema is not None:
            schema_hint = json.dumps(response_schema, ensure_ascii=False)
        user_prompt = (
            "Fix the following into valid JSON.\n\n"
            f"Schema (if present): {schema_hint}\n\n"
            "Text to fix:\n"
            f"{raw_text[:12000]}\n\n"
            "Return ONLY the repaired JSON object."
        )

        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 1.0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }

        try:
            raw = self._generate_with_schema_fallbacks(
                model=model,
                base_payload_camel=payload,
                response_schema=response_schema,
            )
        except GeminiError:
            return None

        text = _extract_text(raw)
        if not text:
            return None
        return _parse_json_relaxed(text)


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


def _parse_json_relaxed(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None

    # First try: direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second try: escape common control chars that models sometimes emit inside strings.
    try:
        fixed = _escape_control_chars_inside_strings(text)
        return json.loads(fixed)
    except Exception:
        pass

    # Third try: extract JSON substring, then parse (with same escaping fallback).
    extracted = _extract_json_from_text(text)
    if extracted is not None:
        return extracted

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        for attempt in (candidate, _escape_control_chars_inside_strings(candidate)):
            try:
                return json.loads(attempt)
            except Exception:
                continue

    return None


def _escape_control_chars_inside_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


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


_TYPE_MAP_JSON_TO_OPENAPI = {
    "object": "OBJECT",
    "string": "STRING",
    "array": "ARRAY",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
}


def _json_schema_to_openapi_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal conversion for GenerateContent's `responseSchema` (OpenAPI Schema subset).
    def conv(node: Any) -> Any:
        if isinstance(node, list):
            return [conv(x) for x in node]
        if not isinstance(node, dict):
            return node

        out: Dict[str, Any] = {}
        t = node.get("type")
        if isinstance(t, str):
            out["type"] = _TYPE_MAP_JSON_TO_OPENAPI.get(t.lower(), t.upper())

        props = node.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: conv(v) for k, v in props.items()}

        items = node.get("items")
        if isinstance(items, dict):
            out["items"] = conv(items)

        req = node.get("required")
        if isinstance(req, list):
            out["required"] = req

        return out

    return conv(schema)
