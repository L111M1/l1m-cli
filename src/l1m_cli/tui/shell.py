from __future__ import annotations

from typing import Any

from l1m_cli.anthropic_client import AnthropicModelClient
from l1m_cli.config import Settings, load_settings
from l1m_cli.core.agent_loop import AgentLoop
from l1m_cli.core.agent_team import AgentTeam
from l1m_cli.core.context_manager import ContextManager, TurnTokenTracker
from l1m_cli.core.hooks import build_default_hooks
from l1m_cli.core.memory import MemoryStore
from l1m_cli.core.permissions import PermissionManager
from l1m_cli.core.prompt_manager import PromptManager
from l1m_cli.core.task_system import TaskStore
from l1m_cli.core.tools import (
    EDIT_FILE_TOOL,
    LEAD_AGENT_NAME,
    WRITE_FILE_TOOL,
    build_workspace_tools,
    format_action_announcement,
    format_tool_call,
    invalid_workspace_file_path_reason,
    is_action_tool,
    is_task_tool,
    should_display_action_announcement,
)
from l1m_cli.core.workspace import Workspace
from l1m_cli.errors import L1mError
from l1m_cli.tui.renderer import TuiRenderer


def run_tui_shell(show_steps: bool = False) -> None:
    settings = load_settings()
    workspace = Workspace(settings.workspace_path)
    workspace.ensure()
    memory = MemoryStore()
    tasks = TaskStore()
    prompts = PromptManager(settings.prompt_dir)
    team = AgentTeam(settings=settings, workspace=workspace, prompts=prompts)
    renderer = TuiRenderer()
    context = ContextManager(memory=memory, compact_threshold=settings.compact_threshold)
    permission_manager = PermissionManager()
    context_visible = False

    renderer.intro(settings, workspace)

    while True:
        try:
            if renderer.layout_width_changed():
                renderer.intro(settings, workspace)
                task_panel = tasks.render_panel()
                if task_panel:
                    renderer.task_panel(task_panel)
                if context_visible:
                    renderer.context_status(context.current_context_tokens)
                renderer.remember_layout_width()
            user_text = renderer.read_input().strip()
        except (EOFError, KeyboardInterrupt):
            renderer.status("bye")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            renderer.status("bye")
            break
        if user_text == "/help":
            renderer.help()
            continue
        if user_text == "/clear":
            context.clear()
            tasks.clear()
            team.clear()
            context_visible = False
            renderer.intro(settings, workspace)
            renderer.status("context cleared")
            continue

        loop = _build_tui_agent_loop(
            settings=settings,
            workspace=workspace,
            memory=memory,
            tasks=tasks,
            prompts=prompts,
            team=team,
            renderer=renderer,
            show_steps=show_steps,
            permission_manager=permission_manager,
            context=context,
        )

        try:
            renderer.reset_turn_tokens()
            context.begin_token_tracking()
            bundle = context.begin_turn(user_text)
            context_visible = True
            result = loop.run(bundle.user_text, history=bundle.messages)
        except KeyboardInterrupt:
            renderer.status("bye")
            break
        except L1mError as exc:
            renderer.error(str(exc))
            continue
        except Exception as exc:
            renderer.error(f"未预期的错误: {exc}")
            continue

        try:
            context.commit_result(result)
            renderer.assistant_message(result.final_text)
            context.replace_history(_inject_team_inbox(context.history, team, renderer))
            tasks.clear_if_complete()
        except Exception as exc:
            renderer.error(f"回合收尾失败，已保留本轮上下文: {exc}")


def _build_tui_agent_loop(
    settings: Settings,
    workspace: Workspace,
    memory: MemoryStore,
    tasks: TaskStore,
    prompts: PromptManager,
    team: AgentTeam,
    renderer: TuiRenderer,
    show_steps: bool,
    permission_manager: PermissionManager,
    context: ContextManager | None = None,
) -> AgentLoop:
    tools = build_workspace_tools(workspace, tasks, team=team)
    system_prompt = lambda: prompts.render_system(
        workspace=str(settings.workspace_path),
        memory=memory.render_for_prompt(),
        tasks=tasks.render_for_prompt(),
        enabled_tools=tools.names(),
    )
    task_reminder = _make_task_update_reminder(tasks)
    event_handler = _with_usage_recording(
        _make_tui_event_handler(
            tasks,
            renderer,
            show_steps=show_steps,
            task_reminder=task_reminder,
        ),
        context,
        TurnTokenTracker(),
    )
    team.set_event_handler(event_handler)
    client = AnthropicModelClient(settings)
    return AgentLoop(
        client=client,
        tools=tools,
        system_prompt=system_prompt,
        on_event=event_handler,
        hooks=build_default_hooks(
            permission_hook=permission_manager,
        ),
        external_event_provider=lambda: team.consume_inbox_for_history(),
        before_model_call=_chain_before_model_call(
            task_reminder,
            _make_context_compactor(
                context,
                client,
                prompts,
                event_handler,
                system_prompt,
                tools.specs,
            ),
        ),
    )


def _make_tui_event_handler(
    tasks: TaskStore,
    renderer: TuiRenderer,
    show_steps: bool = False,
    task_reminder=None,
):
    last_task_text = {"value": tasks.render_panel()}

    def show_task_panel(force: bool = False) -> None:
        current = tasks.render_panel()
        if not current:
            return
        if force or current != last_task_text["value"]:
            renderer.task_panel(current)
            last_task_text["value"] = current

    def mark_task_update() -> None:
        reset = getattr(task_reminder, "reset", None)
        if callable(reset):
            reset()

    def handle(event: str, payload: dict[str, Any]) -> None:
        if event == "task_update_reminder":
            renderer.status("提醒模型更新任务进度...")
            return
        if event == "model_start":
            renderer.start_thinking_tokens(_as_int(payload.get("turn_total_tokens")))
            return
        if event == "context_estimate":
            renderer.context_status(_as_int(payload.get("context_tokens")))
            return
        if event == "model_usage":
            renderer.context_status(_as_int(payload.get("context_tokens", payload.get("input_tokens"))))
            renderer.finish_thinking_tokens(_as_int(payload.get("turn_total_tokens")))
            return
        if event == "model_output_token_delta":
            if "context_tokens" in payload:
                renderer.context_status(_as_int(payload.get("context_tokens")))
            return
        if event == "model_stop":
            if "turn_total_tokens" in payload:
                renderer.finish_thinking_tokens(_as_int(payload.get("turn_total_tokens")))
            return
        if event == "context_compact_start":
            renderer.transient_status("正在压缩上下文...")
            return
        if event == "context_compact_finish":
            renderer.transient_status("上下文压缩完成")
            renderer.clear_transient_status()
            renderer.context_status(_as_int(payload.get("after_tokens")))
            return
        if event == "external_event_injected":
            renderer.status("team inbox: 收到新消息，已自动注入当前上下文。")
            return
        if event == "tool_start":
            name = str(payload["name"])
            tool_input = payload.get("input") or {}
            if is_action_tool(name):
                if should_display_action_announcement(tool_input):
                    renderer.action_panel(format_action_announcement(tool_input))
                return
            if is_task_tool(name):
                return
            detail = format_tool_call(name, tool_input, mode="tui")
            renderer.tool_call(detail)
            _show_file_change_preview(renderer, name, tool_input)
            return
        if event == "tool_rejected":
            return
        if event == "tool_result":
            name = str(payload["name"])
            result = str(payload.get("result") or "")
            if is_action_tool(name):
                return
            if is_task_tool(name):
                if name == "task_update":
                    mark_task_update()
                    status = str((payload.get("input") or {}).get("status") or "").lower()
                    if status in {"completed", "done"}:
                        show_task_panel(force=True)
            elif show_steps:
                renderer.result_panel(_clip_for_display(result))
            return
        if event == "step_complete":
            return
        if event == "final_review_required":
            renderer.status("进入最终验收: 重新检查项目实际完成情况...")
            return
        if event == "continue_required":
            renderer.status("继续检查当前任务状态...")
            return
        if event == "continue_aborted":
            renderer.status("模型未按要求调用工具验收，已停止避免无限循环。")
            return
        if event == "max_tokens_continue":
            renderer.status("模型回复触及长度上限，继续获取剩余内容...")
            return
        if event == "max_tokens_aborted":
            renderer.status("模型多次触及长度上限，已返回目前拿到的内容。")
            return
        if event == "invalid_tool_use":
            renderer.error(str(payload.get("reason") or "模型返回了无效工具调用，已停止。"))

    return handle


def _chain_before_model_call(*callbacks):
    active_callbacks = [callback for callback in callbacks if callback is not None]
    if not active_callbacks:
        return None

    def run(messages: list[dict[str, Any]], step: int) -> None:
        for callback in active_callbacks:
            callback(messages, step)

    return run


def _make_task_update_reminder(tasks: TaskStore, every_model_calls: int = 4):
    calls_since_update = {"value": 0}

    def reminder(messages: list[dict[str, Any]], step: int) -> None:
        if not tasks.has_open_tasks():
            calls_since_update["value"] = 0
            return
        calls_since_update["value"] += 1
        if calls_since_update["value"] < every_model_calls:
            return
        calls_since_update["value"] = 0
        messages.append(
            {
                "role": "user",
                "content": (
                    "[Task Reminder]\n"
                    "如果你刚刚完成了某个任务或任务状态已经变化，请调用 task_update 更新任务面板。"
                    "如果任务状态没有变化，请忽略这条提醒，继续当前工作。"
                ),
            }
        )

    def reset() -> None:
        calls_since_update["value"] = 0

    reminder.reset = reset  # type: ignore[attr-defined]
    return reminder


def _clip_for_display(text: str, limit: int = 1600) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... 输出已截断"


def _with_usage_recording(
    handler,
    context: ContextManager | None,
    turn_tracker: TurnTokenTracker | None = None,
):
    tracker = turn_tracker or TurnTokenTracker()

    def wrapped(event: str, payload: dict[str, Any]) -> None:
        event_agent = str(payload.get("agent") or LEAD_AGENT_NAME)
        step = payload.get("step", "")
        if event == "model_start":
            tracker.begin_model_call(event_agent, step)
            if context is not None and event_agent == LEAD_AGENT_NAME:
                context.begin_model_call()
            payload.update(tracker.as_payload())
        if event == "model_output_token_delta":
            tracker.record_output_token_delta(
                _as_int(payload.get("output_tokens_delta")),
                event_agent,
                step,
            )
            if context is not None and event_agent == LEAD_AGENT_NAME:
                context.record_output_token_delta(_as_int(payload.get("output_tokens_delta")))
            if context is not None:
                payload["context_tokens"] = context.current_context_tokens
            payload.update(tracker.as_payload())
        if event == "model_usage":
            tracker.record_usage(payload, event_agent, step)
            if context is not None and event_agent == LEAD_AGENT_NAME:
                context.record_usage(payload)
            if context is not None:
                payload["context_tokens"] = context.current_context_tokens
            payload.update(tracker.as_payload())
        handler(event, payload)

    return wrapped


def _make_context_compactor(
    context: ContextManager | None,
    client: AnthropicModelClient,
    prompts: PromptManager,
    event_handler,
    system_prompt_provider,
    tool_specs_provider,
):
    if context is None:
        return None

    def compact(messages: list[dict[str, Any]], step: int) -> None:
        estimated_tokens = context.prepare_request(
            system_prompt_provider(),
            messages,
            tool_specs_provider(),
        )
        event_handler("context_estimate", {"step": step, "context_tokens": estimated_tokens})
        if not context.needs_compact(messages):
            return
        before_tokens = context.current_context_tokens
        event_handler(
            "context_compact_start",
            {"step": step, "context_tokens": before_tokens, "threshold": context.compact_threshold},
        )
        context.replace_history(messages)
        context.compact(client, prompts)
        messages[:] = context.history
        context.prepare_request(
            system_prompt_provider(),
            messages,
            tool_specs_provider(),
        )
        event_handler(
            "context_compact_finish",
            {
                "step": step,
                "before_tokens": before_tokens,
                "after_tokens": context.current_context_tokens,
            },
        )

    return compact


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _show_file_change_preview(renderer: TuiRenderer, name: str, tool_input: dict[str, Any]) -> None:
    if name == WRITE_FILE_TOOL:
        path = str(tool_input.get("path") or tool_input.get("file_path") or "?")
        content = str(tool_input.get("content") or "")
        if invalid_workspace_file_path_reason(path):
            return
        if content == "" and not bool(tool_input.get("allow_empty", False)):
            return
        mode = "overwrite" if tool_input.get("overwrite") else "create"
        renderer.file_change(
            f"File Change: {path}",
            [
                f"mode: {mode}",
                f"chars: {len(content)}",
                "content:",
                *_preview_text(content),
            ],
        )
        return

    if name == EDIT_FILE_TOOL:
        batch = tool_input.get("edits")
        if isinstance(batch, list):
            edits = [item for item in batch if isinstance(item, dict)]
            paths = {str(item.get("path") or "?") for item in edits}
            lines: list[str] = []
            for index, item in enumerate(edits, start=1):
                path = str(item.get("path") or "?")
                old_str = str(item.get("old_str") or "")
                new_str = str(item.get("new_str") or "")
                lines.extend(
                    [
                        f"[{index}] {path}",
                        "old:",
                        *_preview_text(old_str, prefix="- ", max_lines=30),
                        "new:",
                        *_preview_text(new_str, prefix="+ ", max_lines=30),
                    ]
                )
            renderer.file_change(f"File Edits: {len(paths)} files", lines)
            return

        path = str(tool_input.get("path") or "?")
        old_str = str(tool_input.get("old_str") or "")
        new_str = str(tool_input.get("new_str") or "")
        renderer.file_change(
            f"File Edit: {path}",
            [
                "old:",
                *_preview_text(old_str, prefix="- "),
                "new:",
                *_preview_text(new_str, prefix="+ "),
            ],
        )


def _preview_text(text: str, prefix: str = "  ", max_lines: int = 80, max_chars: int = 4000) -> list[str]:
    if not text:
        return [prefix + "(empty)"]

    clipped = text[:max_chars]
    lines = clipped.splitlines() or [clipped]
    shown = [prefix + line for line in lines[:max_lines]]
    if len(text) > max_chars or len(lines) > max_lines:
        shown.append(prefix + "... 内容预览已截断")
    return shown


def _inject_team_inbox(
    history: list[dict[str, Any]],
    team: AgentTeam,
    renderer: TuiRenderer,
) -> list[dict[str, Any]]:
    inbox_text = team.consume_inbox_for_history()
    if not inbox_text:
        return history
    renderer.status("team inbox: 已收到 teammate 消息，已注入下一轮上下文。")
    return [*history, {"role": "user", "content": f"[Team Inbox]\n{inbox_text}"}]
