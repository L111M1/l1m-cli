from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from l1m_cli.core.tools import ANNOUNCE_ACTION_TOOL, is_action_tool


MAX_TOKEN_CONTINUATIONS = 3

ContinuationProvider = Callable[["ContinuationContext"], "ContinuationRequest | str | None"]
PreToolUseHook = Callable[["ToolUseContext"], "ToolResultOverride | None"]
PostToolUseHook = Callable[["ToolUseContext", str], None]
StopHook = Callable[["StopContext"], "StopDecision | None"]


@dataclass
class AgentRuntimeState:
    goal: str
    messages: list[dict[str, Any]]
    steps: list[Any]
    completion_review_requested: bool = False
    waiting_for_tool_after_review: bool = False
    truncated_final_parts: list[str] = field(default_factory=list)
    max_token_continuations: int = 0
    action_announcement_ready: bool = False


@dataclass
class ContinuationContext:
    goal: str
    final_text: str
    steps: list[Any]
    messages: list[dict[str, Any]]
    review_requested: bool


@dataclass
class ContinuationRequest:
    message: str
    event: str = "continue_required"
    mark_review_requested: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUseContext:
    state: AgentRuntimeState
    step: Any
    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_id: str


@dataclass
class ToolResultOverride:
    content: str


@dataclass
class StopContext:
    state: AgentRuntimeState
    step: Any
    step_index: int
    stop_reason: str | None
    final_text: str


@dataclass
class StopDecision:
    action: Literal["continue", "finish"]
    final_text: str | None = None
    message: str | None = None
    event: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class AgentHooks:
    def __init__(self):
        self._pre_tool_use: list[PreToolUseHook] = []
        self._post_tool_use: list[PostToolUseHook] = []
        self._stop: list[StopHook] = []

    def register_pre_tool_use(self, hook: PreToolUseHook) -> None:
        self._pre_tool_use.append(hook)

    def register_post_tool_use(self, hook: PostToolUseHook) -> None:
        self._post_tool_use.append(hook)

    def register_stop(self, hook: StopHook) -> None:
        self._stop.append(hook)

    def trigger_pre_tool_use(self, context: ToolUseContext) -> ToolResultOverride | None:
        for hook in self._pre_tool_use:
            result = hook(context)
            if result is not None:
                return result
        return None

    def trigger_post_tool_use(self, context: ToolUseContext, result: str) -> None:
        for hook in self._post_tool_use:
            hook(context, result)

    def trigger_stop(self, context: StopContext) -> StopDecision:
        for hook in self._stop:
            decision = hook(context)
            if decision is not None:
                return decision
        return StopDecision(action="finish", final_text=context.final_text)


def build_default_hooks(
    continuation_provider: ContinuationProvider | None = None,
    permission_hook: PreToolUseHook | None = None,
) -> AgentHooks:
    hooks = AgentHooks()
    hooks.register_pre_tool_use(require_action_announcement)
    if permission_hook is not None:
        hooks.register_pre_tool_use(permission_hook)
    hooks.register_post_tool_use(track_action_announcement)
    hooks.register_stop(MaxTokenContinuationHook())
    hooks.register_stop(CompletionReviewHook(continuation_provider))
    return hooks


def require_action_announcement(context: ToolUseContext) -> ToolResultOverride | None:
    if is_action_tool(context.tool_name) or context.state.action_announcement_ready:
        return None
    return ToolResultOverride(content=_missing_action_announcement_message(context.tool_name))


def track_action_announcement(context: ToolUseContext, _result: str) -> None:
    context.state.action_announcement_ready = is_action_tool(context.tool_name)


class MaxTokenContinuationHook:
    def __call__(self, context: StopContext) -> StopDecision | None:
        state = context.state
        if context.stop_reason == "max_tokens":
            if context.final_text:
                state.truncated_final_parts.append(context.final_text)
            if state.max_token_continuations < MAX_TOKEN_CONTINUATIONS:
                state.max_token_continuations += 1
                return StopDecision(
                    action="continue",
                    message=(
                        "你的上一条回复因为输出长度限制被截断了。"
                        "请从中断处继续，不要重复已经输出过的内容。"
                    ),
                    event="max_tokens_continue",
                    payload={
                        "step": context.step_index,
                        "assistant_text": context.final_text,
                        "count": state.max_token_continuations,
                    },
                )

            context.final_text = _combine_final_text(
                state.truncated_final_parts,
                "\n\n[回复仍然可能因为 max_tokens 被截断，已停止自动续写。]",
            )
            return StopDecision(
                action="finish",
                final_text=context.final_text,
                event="max_tokens_aborted",
                payload={"step": context.step_index, "assistant_text": context.final_text},
            )

        if state.truncated_final_parts:
            context.final_text = _combine_final_text([*state.truncated_final_parts, context.final_text])
        return None


class CompletionReviewHook:
    def __init__(self, continuation_provider: ContinuationProvider | None):
        self.continuation_provider = continuation_provider

    def __call__(self, context: StopContext) -> StopDecision | None:
        state = context.state
        if state.waiting_for_tool_after_review:
            state.action_announcement_ready = False
            final_text = _loop_guard_message(context.final_text)
            context.final_text = final_text
            return StopDecision(
                action="finish",
                final_text=final_text,
                event="continue_aborted",
                payload={"step": context.step_index, "assistant_text": final_text},
            )

        if self.continuation_provider is None:
            return None

        continuation = self.continuation_provider(
            ContinuationContext(
                goal=state.goal,
                final_text=context.final_text,
                steps=state.steps,
                messages=state.messages,
                review_requested=state.completion_review_requested,
            )
        )
        if not continuation:
            return None

        request = _normalize_continuation_request(continuation)
        state.action_announcement_ready = False
        if request.mark_review_requested:
            state.completion_review_requested = True
        state.waiting_for_tool_after_review = True
        return StopDecision(
            action="continue",
            message=request.message,
            event=request.event,
            payload={
                "step": context.step_index,
                "reason": request.message,
                "assistant_text": context.final_text,
                **request.payload,
            },
        )


def _loop_guard_message(last_text: str) -> str:
    detail = f"\n\n模型最后输出:\n{last_text}" if last_text else ""
    return (
        "模型在任务未完成时连续结束且没有调用工具。为了避免 agent loop 无限循环，已停止。"
        "请重新输入任务，或把下一步要求说得更具体。"
        f"{detail}"
    )


def _missing_action_announcement_message(tool_name: str) -> str:
    return (
        f"工具调用顺序错误：调用 {tool_name} 前必须先调用 {ANNOUNCE_ACTION_TOOL}，"
        "向用户展示下一步要做什么、为什么、准备调用哪个工具。"
        "本次工具调用未执行，请先补充行动预告后再重新调用该工具。"
    )


def _normalize_continuation_request(continuation: ContinuationRequest | str) -> ContinuationRequest:
    if isinstance(continuation, ContinuationRequest):
        return continuation
    return ContinuationRequest(message=continuation)


def _combine_final_text(parts: list[str], suffix: str = "") -> str:
    text = "\n".join(part for part in parts if part).strip()
    return f"{text}{suffix}"
