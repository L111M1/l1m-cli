from __future__ import annotations

import json
from typing import Any

from l1m_cli.anthropic_client import AnthropicModelClient
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
) -> tuple[str, list[AgentMessage]]:
    conversation = "\n\n".join(
        f"{message.get('role', 'unknown')}: {_message_text(message)}"
        for message in messages
    )
    compact_prompt = prompts.render("compact", conversation=conversation)
    summary = client.create_message(
        system=compact_prompt,
        messages=[{"role": "user", "content": "现在请生成上下文压缩摘要。"}],
    )
    kept = messages[-keep_last:] if keep_last > 0 else []
    return summary, kept


def _message_text(message: AgentMessage) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)
