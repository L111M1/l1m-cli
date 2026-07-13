from __future__ import annotations

import json
from typing import Any, Callable

from l1m_cli.anthropic_client import AnthropicModelClient, extract_text
from l1m_cli.core.prompt_manager import PromptManager


AgentMessage = dict[str, Any]


def estimate_tokens(messages: list[AgentMessage]) -> int:
    text = "\n".join(_message_text(message) for message in messages)
    return max(1, len(text) // 4)


def should_compact(messages: list[AgentMessage], threshold: int) -> bool:
    return estimate_tokens(messages) >= threshold


def compact_messages(
    client: AnthropicModelClient,
    prompts: PromptManager,
    messages: list[AgentMessage],
    keep_last: int = 6,
    on_usage: Callable[[dict[str, int]], None] | None = None,
) -> tuple[str, list[AgentMessage]]:
    conversation = "\n\n".join(
        f"{message.get('role', 'unknown')}: {_message_text(message)}"
        for message in messages
    )
    compact_prompt = prompts.render("compact", conversation=conversation)
    summary_request = [{"role": "user", "content": "现在请生成上下文压缩摘要。"}]
    create_raw_message = getattr(client, "create_raw_message", None)
    if callable(create_raw_message):
        response = create_raw_message(system=compact_prompt, messages=summary_request)
        summary = extract_text(response)
        usage = _usage_payload(response)
        if usage is not None and on_usage is not None:
            on_usage(usage)
    else:
        summary = client.create_message(system=compact_prompt, messages=summary_request)
    kept = messages[-keep_last:] if keep_last > 0 else []
    return summary, kept


def _message_text(message: AgentMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _usage_payload(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    payload: dict[str, int] = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            payload[name] = int(value)
    return payload or None
