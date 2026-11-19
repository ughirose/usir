"""Minimal stdlib HTTP client for local Ollama `/api/chat` with structured output."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaResponse:
    """Captured response, timing, token counts, and parsed JSON from Ollama."""

    model: str
    think: bool | None
    format_mode: str  # "json_schema" | "json_fallback"
    raw_content: str
    thinking_content: str | None
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    elapsed_sec: float
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    eval_duration_ns: int | None


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return stripped


def _post_chat(
    ollama_url: str,
    payload: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_json_content(raw_content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON object from model content, tolerating markdown code fences."""
    cleaned = _strip_markdown_fences(raw_content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    if not isinstance(parsed, dict):
        return None, f"Expected JSON object, got {type(parsed).__name__}"
    return parsed, None


def call_ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    format_schema: dict[str, Any],
    ollama_url: str = "http://127.0.0.1:11434",
    num_ctx: int = 8192,
    num_predict: int = 4096,
    seed: int = 42,
    temperature: float = 0.0,
    timeout_sec: float = 240.0,
    think: bool | None = None,
) -> OllamaResponse:
    """Invoke Ollama `/api/chat` with structured schema output and fallback to `format='json'`."""
    base_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": format_schema,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if think is not None:
        base_payload["think"] = think

    t0 = time.perf_counter()
    format_mode = "json_schema"
    try:
        data = _post_chat(ollama_url, base_payload, timeout_sec)
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        if err.code in (400, 422, 500):
            # Retry with generic `format="json"` if schema constraint failed on server side.
            fallback_payload = dict(base_payload)
            fallback_payload["format"] = "json"
            format_mode = f"json_fallback_after_http_{err.code}"
            try:
                data = _post_chat(ollama_url, fallback_payload, timeout_sec)
            except urllib.error.HTTPError as err2:
                err2_body = err2.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Ollama HTTP {err2.code} for model={model}: {err2_body} "
                    f"(initial HTTP {err.code}: {err_body})"
                ) from err2
        else:
            raise RuntimeError(f"Ollama HTTP {err.code} for model={model}: {err_body}") from err

    elapsed = time.perf_counter() - t0
    msg = data.get("message") or {}
    raw_content = str(msg.get("content") or "")
    thinking_content = msg.get("thinking")
    parsed, parse_err = parse_json_content(raw_content)

    return OllamaResponse(
        model=model,
        think=think,
        format_mode=format_mode,
        raw_content=raw_content,
        thinking_content=str(thinking_content) if thinking_content is not None else None,
        parsed_json=parsed,
        parse_error=parse_err,
        elapsed_sec=round(elapsed, 3),
        prompt_eval_count=data.get("prompt_eval_count"),
        eval_count=data.get("eval_count"),
        total_duration_ns=data.get("total_duration"),
        load_duration_ns=data.get("load_duration"),
        eval_duration_ns=data.get("eval_duration"),
    )
