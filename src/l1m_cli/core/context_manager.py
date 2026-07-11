from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from l1m_cli.core.compact import compact_messages
from l1m_cli.core.memory import MemoryStore


DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_COMPACT_TRIGGER_RATIO = 0.85


@dataclass(frozen=True)
class ContextBundle:
    user_text: str
    messages: list[dict[str, Any]]
    memory: str
    context_summary: str
    estimated_tokens: int


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def context_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def context_tokens(self) -> int:
        return self.context_input_tokens + self.output_tokens


class ContextManager:
    def __init__(
        self,
        memory: MemoryStore | None = None,
        compact_threshold: int | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    ):
        self.memory = memory or MemoryStore()
        self.context_window_tokens = context_window_tokens
        self.compact_threshold = compact_threshold or int(
            context_window_tokens * DEFAULT_COMPACT_TRIGGER_RATIO
        )
        self._history: list[dict[str, Any]] = []
        self._rolling_summary = ""
        self.last_usage: ModelUsage | None = None
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
        self._pending_output_estimate_tokens = 0
        self._pending_context_estimate_tokens = 0
        self._pending_request_base_tokens = 0
        self._request_token_scale = 1.0
        self._has_usage_calibration = False

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def begin_turn(self, user_text: str) -> ContextBundle:
        messages = self.history
        estimated_tokens = _estimate_message_tokens(
            [
                *messages,
                {"role": "user", "content": user_text},
            ]
        )
        self._pending_context_estimate_tokens = estimated_tokens
        return ContextBundle(
            user_text=user_text,
            messages=messages,
            memory=self.render_memory(),
            context_summary=self.render_context_summary(),
            estimated_tokens=estimated_tokens,
        )

    def commit_result(self, result: Any) -> None:
        self.replace_history(list(getattr(result, "messages", []) or []))
        self._pending_context_estimate_tokens = 0
        self._pending_request_base_tokens = 0

    def replace_history(self, messages: list[dict[str, Any]]) -> None:
        self._history = [dict(message) for message in messages]

    def clear(self) -> None:
        self._history.clear()
        self._rolling_summary = ""
        self.last_usage = None
        self.begin_token_tracking()
        self._pending_context_estimate_tokens = 0
        self._pending_request_base_tokens = 0
        self._request_token_scale = 1.0
        self._has_usage_calibration = False
        self.memory.clear()

    def render_memory(self) -> str:
        return self.memory.render_for_prompt()

    def render_context_summary(self) -> str:
        return self._rolling_summary

    def compact_if_needed(self) -> bool:
        return False

    def needs_compact(self, messages: list[dict[str, Any]] | None = None) -> bool:
        if self.current_context_tokens >= self.compact_threshold:
            return True
        if messages is None:
            messages = self._history
        return _estimate_message_tokens(messages) >= self.compact_threshold

    def prepare_request(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        base_tokens = _estimate_request_base_tokens(system, messages, tools)
        estimated_tokens = max(1, math.ceil(base_tokens * self._request_token_scale))
        self._pending_request_base_tokens = base_tokens
        self._pending_context_estimate_tokens = estimated_tokens
        return estimated_tokens

    def compact(self, client: Any, prompts: Any, keep_last: int = 6) -> bool:
        if not self._history:
            return False
        summary, kept = compact_messages(client, prompts, self._history, keep_last=keep_last)
        self._rolling_summary = summary.strip()
        safe_kept = _drop_leading_orphan_tool_results(kept)
        summary_message = {
            "role": "user",
            "content": f"[Context Summary]\n{self._rolling_summary}",
        }
        self._history = [summary_message, *safe_kept]
        self.last_usage = None
        self._pending_context_estimate_tokens = 0
        self._pending_request_base_tokens = 0
        return True

    @property
    def current_context_tokens(self) -> int:
        if self._pending_context_estimate_tokens:
            return self._pending_context_estimate_tokens + self._pending_output_estimate_tokens
        if self.last_usage is not None:
            return self.last_usage.context_tokens + self._pending_output_estimate_tokens
        return _estimate_message_tokens(self._history) + self._pending_output_estimate_tokens

    @property
    def request_token_scale(self) -> float:
        return self._request_token_scale

    @property
    def turn_total_tokens(self) -> int:
        return self.turn_input_tokens + self.turn_output_tokens

    def record_usage(self, payload: dict[str, Any]) -> None:
        usage = ModelUsage(
            input_tokens=_as_int(payload.get("input_tokens")),
            output_tokens=_as_int(payload.get("output_tokens")),
            cache_creation_input_tokens=_as_int(payload.get("cache_creation_input_tokens")),
            cache_read_input_tokens=_as_int(payload.get("cache_read_input_tokens")),
        )
        self.last_usage = usage
        if self._pending_request_base_tokens > 0 and usage.context_input_tokens > 0:
            measured_scale = usage.context_input_tokens / self._pending_request_base_tokens
            measured_scale = max(0.25, min(measured_scale, 8.0))
            if self._has_usage_calibration:
                self._request_token_scale = self._request_token_scale * 0.7 + measured_scale * 0.3
            else:
                self._request_token_scale = measured_scale
                self._has_usage_calibration = True
        self._pending_context_estimate_tokens = 0
        self._pending_request_base_tokens = 0
        self.turn_input_tokens += usage.context_input_tokens
        self.turn_output_tokens += usage.output_tokens
        self._pending_output_estimate_tokens = 0

    def begin_model_call(self) -> None:
        self._pending_output_estimate_tokens = 0

    def record_output_token_delta(self, delta_tokens: int) -> None:
        delta = max(0, _as_int(delta_tokens))
        self._pending_output_estimate_tokens += delta

    def begin_token_tracking(self) -> None:
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
        self._pending_output_estimate_tokens = 0


class TurnTokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self._pending_output_estimates: dict[str, int] = {}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_total_tokens(self) -> int:
        return self.total_tokens + sum(self._pending_output_estimates.values())

    def begin_model_call(self, agent: object = "lead", step: object = "") -> None:
        self._pending_output_estimates[self._call_key(agent, step)] = 0

    def record_output_token_delta(
        self,
        delta_tokens: int,
        agent: object = "lead",
        step: object = "",
    ) -> None:
        delta = max(0, _as_int(delta_tokens))
        key = self._call_key(agent, step)
        self._pending_output_estimates[key] = self._pending_output_estimates.get(key, 0) + delta

    def record_usage(
        self,
        payload: dict[str, Any],
        agent: object = "lead",
        step: object = "",
    ) -> None:
        input_tokens = _as_int(payload.get("input_tokens"))
        output_tokens = _as_int(payload.get("output_tokens"))
        key = self._call_key(agent, step)
        self._pending_output_estimates.pop(key, 0)

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def as_payload(self) -> dict[str, int]:
        return {
            "turn_input_tokens": self.input_tokens,
            "turn_output_tokens": self.output_tokens,
            "turn_total_tokens": self.total_tokens,
            "turn_estimated_total_tokens": self.estimated_total_tokens,
        }

    @staticmethod
    def _call_key(agent: object, step: object) -> str:
        return f"{_normalize_agent_name(agent)}:{step or ''}"


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return _estimate_json_tokens(messages)


def _estimate_request_base_tokens(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    request_context: dict[str, Any] = {
        "system": system,
        "messages": messages,
    }
    if tools:
        request_context["tools"] = tools
    return _estimate_json_tokens(request_context)


def _estimate_json_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _drop_leading_orphan_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(messages)
    while result and _is_tool_result_message(result[0]):
        result.pop(0)
    return result


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


class AgentContextRegistry:
    def __init__(self, compact_threshold: int | None = None):
        self.compact_threshold = compact_threshold
        self._contexts: dict[str, ContextManager] = {}

    def for_agent(self, agent_name: object) -> ContextManager:
        name = _normalize_agent_name(agent_name)
        if name not in self._contexts:
            self._contexts[name] = ContextManager(compact_threshold=self.compact_threshold)
        return self._contexts[name]

    def clear_agent(self, agent_name: object) -> None:
        name = _normalize_agent_name(agent_name)
        context = self._contexts.get(name)
        if context is not None:
            context.clear()

    def clear(self) -> None:
        for context in self._contexts.values():
            context.clear()
        self._contexts.clear()


def _normalize_agent_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = text.strip("_-")
    return text[:40] or "agent"
