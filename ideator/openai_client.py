from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class OpenAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIHTTPError(OpenAIError):
    status_code: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        base = f"OpenAI API HTTP {self.status_code}: {self.message}"
        if self.body:
            return f"{base}\n{self.body}"
        return base


class OpenAIClient:
    def __init__(self, *, api_key: str, base_url: str = "https://api.openai.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body_bytes: Optional[bytes]
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self._api_key}",
        }
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
                msg = parsed.get("error", {}).get("message") or parsed.get("message") or msg
            except Exception:
                pass
            raise OpenAIHTTPError(status_code=int(e.code), message=str(msg), body=raw) from e
        except urllib.error.URLError as e:
            raise OpenAIError(f"OpenAI API network error: {e}") from e
        except json.JSONDecodeError as e:
            raise OpenAIError(f"OpenAI API returned non-JSON response: {e}") from e

    def generate_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        force_json_object: bool = False,
    ) -> str:
        url = self._url("v1/chat/completions")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if force_json_object:
            payload["response_format"] = {"type": "json_object"}

        try:
            raw = self._request_json("POST", url, payload)
        except OpenAIHTTPError as e:
            # Compatibility fallback for older endpoints/models that may not support response_format.
            if force_json_object and e.status_code == 400:
                payload.pop("response_format", None)
                raw = self._request_json("POST", url, payload)
            else:
                raise

        text = _extract_chat_completion_text(raw)
        if text is None:
            raise OpenAIError(f"OpenAI response missing text: {raw!r}")
        return text

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> Any:
        text = self.generate_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            force_json_object=True,
        )

        parsed = _parse_json_relaxed(text)
        if parsed is not None:
            return parsed

        repaired = self._repair_json_with_model(
            model=model,
            raw_text=text,
            response_schema=response_schema,
            max_tokens=max_tokens,
        )
        if repaired is not None:
            return repaired

        raise OpenAIError(f"Model did not return valid JSON. Raw text:\n{text}")

    def _repair_json_with_model(
        self,
        *,
        model: str,
        raw_text: str,
        response_schema: Optional[Dict[str, Any]],
        max_tokens: int,
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

        try:
            text = self.generate_text(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=min(1024, int(max_tokens)),
                force_json_object=True,
            )
        except OpenAIError:
            return None

        return _parse_json_relaxed(text or "")


def _extract_chat_completion_text(resp: Dict[str, Any]) -> Optional[str]:
    choices = resp.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    return None


def _extract_json_from_text(text: str) -> Optional[Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

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

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        fixed = _escape_control_chars_inside_strings(text)
        return json.loads(fixed)
    except Exception:
        pass

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

