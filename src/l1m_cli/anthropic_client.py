from __future__ import annotations

import random
import re
import time
from contextlib import nullcontext
from typing import Any

from l1m_cli.config import Settings, require_api_key
from l1m_cli.errors import ModelError


class AnthropicModelClient:
    def __init__(self, settings: Settings):
        require_api_key(settings)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ModelError("Missing dependency: anthropic. Run: python -m pip install -e .") from exc

        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = Anthropic(**kwargs)
        self._settings = settings

    def create_raw_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        retries: int = 3,
    ) -> Any:
        kwargs = _message_kwargs(self._settings, system, messages, tools)
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < retries and _is_transient_error(str(exc)):
                    time.sleep(_retry_delay_seconds(exc, attempt))
                    continue
                raise ModelError(f"Model call failed: {_sanitize_model_error(str(exc))}") from exc
        raise ModelError(
            f"Model call failed after {retries} retries: {_sanitize_model_error(str(last_exc))}"
        ) from last_exc

    def create_streaming_raw_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_output_token_delta: Any | None = None,
        retries: int = 3,
    ) -> Any:
        if not hasattr(self._client.messages, "stream"):
            return self.create_raw_message(system=system, messages=messages, tools=tools, retries=retries)

        kwargs = _message_kwargs(self._settings, system, messages, tools)
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                stream_obj = self._client.messages.stream(**kwargs)
                with _stream_context(stream_obj) as stream:
                    for event in stream:
                        text_delta = _stream_text_delta(event)
                        if text_delta and on_output_token_delta is not None:
                            on_output_token_delta(_estimate_text_tokens(text_delta))
                    return stream.get_final_message()
            except Exception as exc:
                last_exc = exc
                if attempt < retries and _is_transient_error(str(exc)):
                    time.sleep(_retry_delay_seconds(exc, attempt))
                    continue
                raise ModelError(f"Model call failed: {_sanitize_model_error(str(exc))}") from exc
        raise ModelError(
            f"Model call failed after {retries} retries: {_sanitize_model_error(str(last_exc))}"
        ) from last_exc

    def create_message(self, system: str, messages: list[dict[str, Any]]) -> str:
        response = self.create_raw_message(system=system, messages=messages)
        return extract_text(response)


def extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _message_kwargs(
    settings: Settings,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": system,
        "messages": messages,
    }
    thinking = _thinking_config(settings)
    if thinking:
        kwargs["thinking"] = thinking
    if tools:
        kwargs["tools"] = tools
    return kwargs


def _stream_context(stream_obj: Any) -> Any:
    if hasattr(stream_obj, "__enter__") and hasattr(stream_obj, "__exit__"):
        return stream_obj
    return nullcontext(stream_obj)


def _stream_text_delta(event: Any) -> str:
    delta = getattr(event, "delta", None)
    if delta is not None:
        text = getattr(delta, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str):
                return text

    text = getattr(event, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(event, dict):
        delta = event.get("delta")
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str):
                return text
        text = event.get("text")
        if isinstance(text, str):
            return text
    return ""


def _estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _thinking_config(settings: Settings) -> dict[str, Any] | None:
    if not settings.thinking_enabled:
        return None
    if settings.max_tokens <= 1024:
        return None
    budget_tokens = max(1024, settings.thinking_budget_tokens)
    budget_tokens = min(budget_tokens, settings.max_tokens - 1)
    return {"type": "enabled", "budget_tokens": budget_tokens}


def _is_transient_error(msg: str) -> bool:
    transient_markers = [
        "429",
        "rate_limit",
        "rate limit",
        "too many requests",
        "503",
        "504",
        "502",
        "529",
        "530",
        "service_unavailable",
        "service unavailable",
        "overloaded",
        "timeout",
        "timed out",
        "gateway time-out",
        "gateway timeout",
        "upstream returned empty",
        "empty non-stream",
    ]
    lower = msg.lower()
    return any(marker in lower for marker in transient_markers)


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after

    if _is_rate_limit_error(str(exc)):
        return min(60.0, 4.0 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 1.0)
    return min(30.0, float(2**attempt))


def _sanitize_model_error(message: str) -> str:
    text = str(message or "").strip()
    lower = text.lower()
    if "504" in lower or "gateway time-out" in lower or "gateway timeout" in lower:
        return "模型服务网关超时，请稍后重试。"
    if "<html" in lower:
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
    return text if len(text) <= 500 else text[:500] + "..."


def _is_rate_limit_error(msg: str) -> bool:
    lower = msg.lower()
    return any(
        marker in lower
        for marker in [
            "429",
            "rate_limit",
            "rate limit",
            "too many requests",
        ]
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = _response_headers(exc)
    if not headers:
        return None

    raw = _header_value(headers, "retry-after")
    if raw is None:
        raw = _header_value(headers, "retry_after")
    if raw is None:
        return None

    try:
        wait = float(str(raw).strip())
    except ValueError:
        return None
    return max(0.0, min(wait, 120.0))


def _response_headers(exc: Exception) -> Any | None:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            return headers
    return getattr(exc, "headers", None)


def _header_value(headers: Any, name: str) -> Any | None:
    if not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value is not None:
        return value
    return headers.get(name.title())
