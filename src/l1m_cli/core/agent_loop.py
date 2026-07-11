from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from l1m_cli.anthropic_client import AnthropicModelClient, extract_text
from l1m_cli.core.hooks import (
    AgentHooks,
    AgentRuntimeState,
    ContinuationProvider,
    StopContext,
    ToolUseContext,
    build_default_hooks,
)
from l1m_cli.core.tools import ToolRegistry


AgentEventHandler = Callable[[str, dict[str, Any]], None]
SystemPromptProvider = str | Callable[[], str]
ExternalEventProvider = Callable[[], str]
BeforeModelCall = Callable[[list[dict[str, Any]], int], None]


@dataclass
class AgentStep:
    index: int
    assistant_text: str = ""
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    final_text: str
    steps: list[AgentStep]
    messages: list[dict[str, Any]]


class AgentLoop:
    def __init__(
        self,
        client: AnthropicModelClient,
        tools: ToolRegistry,
        system_prompt: SystemPromptProvider,
        on_event: AgentEventHandler | None = None,
        continuation_provider: ContinuationProvider | None = None,
        hooks: AgentHooks | None = None,
        external_event_provider: ExternalEventProvider | None = None,
        before_model_call: BeforeModelCall | None = None,
    ):
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.on_event = on_event
        self.hooks = hooks or build_default_hooks(continuation_provider)
        self.external_event_provider = external_event_provider
        self.before_model_call = before_model_call

    def run(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        messages = list(history or [])
        messages.append({"role": "user", "content": goal})
        steps: list[AgentStep] = []
        state = AgentRuntimeState(goal=goal, messages=messages, steps=steps)
        index = 1

        while True:
            self._inject_external_events(messages, index)
            self._run_before_model_call(messages, index)
            response = self._create_model_response(messages, index)
            stop_reason = getattr(response, "stop_reason", None)
            assistant_content, tool_uses, assistant_text = _split_response_content(response)
            final_text = assistant_text or extract_text(response)
            step = AgentStep(index=index, assistant_text=final_text)
            messages.append({"role": "assistant", "content": assistant_content})

            if not tool_uses:
                result = self._handle_stop_without_tools(
                    state=state,
                    step=step,
                    index=index,
                    stop_reason=stop_reason,
                    final_text=final_text,
                )
                if result is not None:
                    return result
                index += 1
                continue

            tool_results = self._run_tool_uses(tool_uses, state, step, index)
            if tool_results is None:
                return self._invalid_tool_result(
                    state,
                    step,
                    "模型返回了不完整的工具调用，缺少 tool_use id 或 name，已停止。",
                )

            steps.append(step)
            messages.append({"role": "user", "content": tool_results})
            self._emit("step_complete", {"step": index})
            state.waiting_for_tool_after_review = False
            index += 1

    def _create_model_response(self, messages: list[dict[str, Any]], index: int) -> Any:
        self._emit("model_start", {"step": index})
        system_prompt = self._current_system_prompt()
        tool_specs = self.tools.specs()
        stream_method = getattr(self.client, "create_streaming_raw_message", None)
        if callable(stream_method):
            response = stream_method(
                system=system_prompt,
                messages=messages,
                tools=tool_specs,
                on_output_token_delta=lambda delta: self._emit(
                    "model_output_token_delta",
                    {"step": index, "output_tokens_delta": delta},
                ),
            )
        else:
            response = self.client.create_raw_message(
                system=system_prompt,
                messages=messages,
                tools=tool_specs,
            )
        usage_payload = _usage_payload(response, index)
        if usage_payload is not None:
            self._emit("model_usage", usage_payload)
        self._emit("model_stop", {"step": index, "stop_reason": getattr(response, "stop_reason", None)})
        return response

    def _handle_stop_without_tools(
        self,
        state: AgentRuntimeState,
        step: AgentStep,
        index: int,
        stop_reason: str | None,
        final_text: str,
    ) -> AgentResult | None:
        if stop_reason == "tool_use":
            return self._invalid_tool_result(
                state,
                step,
                "模型返回了 tool_use 停止原因，但没有提供任何工具调用，已停止。",
            )

        state.steps.append(step)
        stop_context = StopContext(
            state=state,
            step=step,
            step_index=index,
            stop_reason=stop_reason,
            final_text=final_text,
        )
        decision = self.hooks.trigger_stop(stop_context)
        final_text = decision.final_text or stop_context.final_text
        step.assistant_text = final_text

        if decision.event:
            payload = decision.payload or {"step": index, "assistant_text": final_text}
            self._emit(decision.event, payload)

        if decision.action == "continue":
            if decision.message:
                state.messages.append({"role": "user", "content": decision.message})
            return None

        return AgentResult(final_text=final_text, steps=state.steps, messages=state.messages)

    def _run_tool_uses(
        self,
        tool_uses: list[Any],
        state: AgentRuntimeState,
        step: AgentStep,
        index: int,
    ) -> list[dict[str, Any]] | None:
        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            tool_context = self._tool_context(tool_use, state, step, index)
            if tool_context is None:
                return None

            blocked = self.hooks.trigger_pre_tool_use(tool_context)
            if blocked is not None:
                self._emit(
                    "tool_rejected",
                    {
                        "step": index,
                        "name": tool_context.tool_name,
                        "input": tool_context.tool_input,
                        "reason": blocked.content,
                    },
                )
                tool_results.append(_tool_result(tool_context.tool_id, blocked.content))
                continue

            self._emit(
                "tool_start",
                {"step": index, "name": tool_context.tool_name, "input": tool_context.tool_input},
            )
            result = self.tools.run(tool_context.tool_name, tool_context.tool_input)
            self._emit(
                "tool_result",
                {
                    "step": index,
                    "name": tool_context.tool_name,
                    "input": tool_context.tool_input,
                    "result": result,
                },
            )
            step.tool_calls.append(tool_context.tool_name)
            self.hooks.trigger_post_tool_use(tool_context, result)
            tool_results.append(_tool_result(tool_context.tool_id, result))
        return tool_results

    def _tool_context(
        self,
        tool_use: Any,
        state: AgentRuntimeState,
        step: AgentStep,
        index: int,
    ) -> ToolUseContext | None:
        tool_name = getattr(tool_use, "name", "")
        tool_input = getattr(tool_use, "input", {}) or {}
        tool_id = getattr(tool_use, "id", "")
        if not tool_id or not tool_name:
            return None
        return ToolUseContext(
            state=state,
            step=step,
            step_index=index,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_id=tool_id,
        )

    def _invalid_tool_result(
        self,
        state: AgentRuntimeState,
        step: AgentStep,
        final_text: str,
    ) -> AgentResult:
        step.assistant_text = final_text
        state.steps.append(step)
        self._emit("invalid_tool_use", {"step": step.index, "reason": final_text})
        return AgentResult(final_text=final_text, steps=state.steps, messages=state.messages)

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)

    def _current_system_prompt(self) -> str:
        if callable(self.system_prompt):
            return self.system_prompt()
        return self.system_prompt

    def _inject_external_events(self, messages: list[dict[str, Any]], index: int) -> None:
        if self.external_event_provider is None:
            return
        text = self.external_event_provider().strip()
        if not text:
            return
        messages.append({"role": "user", "content": f"[Team Inbox]\n{text}"})
        self._emit("external_event_injected", {"step": index, "chars": len(text)})

    def _run_before_model_call(self, messages: list[dict[str, Any]], index: int) -> None:
        if self.before_model_call is None:
            return
        self.before_model_call(messages, index)


def _split_response_content(response: Any) -> tuple[list[dict[str, Any]], list[Any], str]:
    assistant_content: list[dict[str, Any]] = []
    tool_uses: list[Any] = []
    text_parts: list[str] = []

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            raw_text = getattr(block, "text", "")
            text = raw_text if isinstance(raw_text, str) else ""
            if text:
                text_parts.append(text)
                assistant_content.append({"type": "text", "text": text})
        elif block_type == "tool_use":
            item = {
                "type": "tool_use",
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {},
            }
            assistant_content.append(item)
            tool_uses.append(block)

    if not assistant_content:
        assistant_content.append({"type": "text", "text": "(模型没有输出可见文本。)"})

    return assistant_content, tool_uses, "\n".join(part for part in text_parts if part).strip()


def _tool_result(tool_use_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


def _usage_payload(response: Any, index: int) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None and output_tokens is None:
        return None
    payload = {"step": index}
    if input_tokens is not None:
        payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        payload["output_tokens"] = output_tokens
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    if cache_creation is not None:
        payload["cache_creation_input_tokens"] = cache_creation
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read is not None:
        payload["cache_read_input_tokens"] = cache_read
    return payload
