from __future__ import annotations

import argparse
import platform
import sys
from typing import Any

from l1m_cli import __version__
from l1m_cli.anthropic_client import AnthropicModelClient
from l1m_cli.config import Settings, load_settings
from l1m_cli.console import error, info
from l1m_cli.core.agent_loop import AgentLoop
from l1m_cli.core.agent_team import AgentTeam
from l1m_cli.core.context_manager import ContextManager, TurnTokenTracker
from l1m_cli.core.hooks import build_default_hooks
from l1m_cli.core.memory import MemoryStore
from l1m_cli.core.permissions import PermissionManager
from l1m_cli.core.prompt_manager import PromptManager
from l1m_cli.core.task_system import TaskStore
from l1m_cli.core.tools import (
    LEAD_AGENT_NAME,
    build_workspace_tools,
    format_action_announcement,
    format_tool_call,
    is_action_tool,
    is_task_tool,
    should_display_action_announcement,
)
from l1m_cli.core.workspace import Workspace
from l1m_cli.errors import L1mError


def _settings() -> Settings:
    return load_settings()


def _workspace(settings: Settings) -> Workspace:
    workspace = Workspace(settings.workspace_path)
    workspace.ensure()
    return workspace


def _prompts(settings: Settings) -> PromptManager:
    return PromptManager(settings.prompt_dir)


def _runtime(settings: Settings) -> tuple[Workspace, MemoryStore, TaskStore]:
    return _workspace(settings), MemoryStore(), TaskStore()


def _system_prompt(
    settings: Settings,
    memory: MemoryStore,
    tasks: TaskStore,
    prompts: PromptManager | None = None,
    enabled_tools: list[str] | tuple[str, ...] | None = None,
) -> str:
    prompt_manager = prompts or _prompts(settings)
    return prompt_manager.render_system(
        workspace=str(settings.workspace_path),
        memory=memory.render_for_prompt(),
        tasks=tasks.render_for_prompt(),
        enabled_tools=enabled_tools,
    )


def _build_agent_loop(
    settings: Settings,
    workspace: Workspace,
    memory: MemoryStore,
    tasks: TaskStore,
    show_steps: bool = False,
    permission_manager: PermissionManager | None = None,
    prompts: PromptManager | None = None,
    team: AgentTeam | None = None,
    context: ContextManager | None = None,
) -> AgentLoop:
    prompts = prompts or _prompts(settings)
    task_reminder = _make_task_update_reminder(tasks)
    event_handler = _with_usage_recording(
        _make_agent_event_handler(
            tasks,
            show_steps=show_steps,
            task_reminder=task_reminder,
        ),
        context,
        TurnTokenTracker(),
    )
    team = team or AgentTeam(
        settings=settings,
        workspace=workspace,
        prompts=prompts,
        event_handler=event_handler,
    )
    team.set_event_handler(event_handler)
    tools = build_workspace_tools(workspace, tasks, team=team)
    client = AnthropicModelClient(settings)
    return AgentLoop(
        client=client,
        tools=tools,
        system_prompt=lambda: _system_prompt(
            settings,
            memory,
            tasks,
            prompts=prompts,
            enabled_tools=tools.names(),
        ),
        on_event=event_handler,
        hooks=build_default_hooks(
            permission_hook=permission_manager,
        ),
        external_event_provider=lambda: team.consume_inbox_for_history(),
        before_model_call=_chain_before_model_call(
            _make_context_compactor(context, client, prompts, event_handler),
            task_reminder,
        ),
    )


def _make_agent_event_handler(
    tasks: TaskStore,
    show_steps: bool = False,
    task_reminder=None,
):
    last_task_text = {"value": tasks.render_panel()}

    def show_task_panel(force: bool = False) -> None:
        current = tasks.render_panel()
        if not current:
            return
        if force or current != last_task_text["value"]:
            info("任务进度:")
            info(current)
            last_task_text["value"] = current

    def handle(event: str, payload: dict[str, Any]) -> None:
        if event == "model_start":
            info("thinking...")
            return
        if event == "model_usage":
            info(_format_token_usage(payload))
            return
        if event == "model_output_token_delta":
            return
        if event == "context_compact_start":
            info("正在压缩上下文...")
            return
        if event == "context_compact_finish":
            info("上下文压缩完成")
            return
        if event == "external_event_injected":
            info("team inbox: 收到新消息，已自动注入当前上下文。")
            return
        if event == "tool_start":
            name = str(payload["name"])
            tool_input = payload.get("input") or {}
            if is_action_tool(name):
                if should_display_action_announcement(tool_input):
                    info(_format_action_panel(tool_input))
                return
            if is_task_tool(name):
                return
            detail = format_tool_call(name, tool_input, mode="console")
            info(detail)
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
                    reset = getattr(task_reminder, "reset", None)
                    if callable(reset):
                        reset()
                    status = str((payload.get("input") or {}).get("status") or "").lower()
                    if status in {"completed", "done"}:
                        show_task_panel(force=True)
            elif show_steps:
                info(_clip_for_display(result))
            return
        if event == "step_complete":
            return
        if event == "tasks_continue_required":
            info("任务还没完成，继续执行剩余任务...")
            return
        if event == "final_review_required":
            info("进入最终验收: 重新检查项目实际完成情况...")
            return
        if event == "continue_required":
            info("继续检查当前任务状态...")
            return
        if event == "continue_aborted":
            info("模型未按要求调用工具验收，已停止避免无限循环。")
            return
        if event == "max_tokens_continue":
            info("模型回复触及长度上限，继续获取剩余内容...")
            return
        if event == "max_tokens_aborted":
            info("模型多次触及长度上限，已返回目前拿到的内容。")
            return
        if event == "invalid_tool_use":
            info(str(payload.get("reason") or "模型返回了无效工具调用，已停止。"))

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


def _format_action_panel(tool_input: dict[str, Any]) -> str:
    lines = format_action_announcement(tool_input)
    return "\n".join(f"thinking · {line}" for line in lines)


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
):
    if context is None:
        return None

    def compact(messages: list[dict[str, Any]], step: int) -> None:
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
        event_handler(
            "context_compact_finish",
            {
                "step": step,
                "before_tokens": before_tokens,
                "after_tokens": context.current_context_tokens,
            },
        )

    return compact


def _format_token_usage(payload: dict[str, Any]) -> str:
    context_tokens = _as_int(payload.get("context_tokens", payload.get("input_tokens")))
    turn_tokens = _as_int(payload.get("turn_total_tokens"))
    if turn_tokens:
        return f"tokens: context {context_tokens:,} / turn {turn_tokens:,}"
    output_tokens = _as_int(payload.get("output_tokens"))
    return f"tokens: context {context_tokens:,} / output {output_tokens:,}"


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clip_for_display(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... 输出已截断"


def cmd_config(_args: argparse.Namespace) -> None:
    settings = _settings()
    workspace = _workspace(settings)
    info(f"cwd: {settings.cwd}")
    info(f"cli_root: {settings.cli_root}")
    info("env_files:")
    if settings.env_files:
        for env_file in settings.env_files:
            info(f"- {env_file}")
    else:
        info("- (none)")
    info(f"api_key: {settings.masked_api_key}")
    info(f"model: {settings.model}")
    info(f"base_url: {settings.base_url or '(default)'}")
    info(f"max_tokens: {settings.max_tokens}")
    info(f"thinking_enabled: {settings.thinking_enabled}")
    info(f"thinking_budget_tokens: {settings.thinking_budget_tokens}")
    info(f"workspace: {workspace.root}")
    info(f"compact_threshold: {settings.compact_threshold}")
    info(f"debug: {settings.debug}")
    info("memory: in-memory only")
    info("tasks: in-memory only")


def cmd_ask(args: argparse.Namespace) -> None:
    settings = _settings()
    _workspace_obj, memory, tasks = _runtime(settings)
    client = AnthropicModelClient(settings)
    answer = client.create_message(
        system=_system_prompt(settings, memory, tasks, enabled_tools=[]),
        messages=[{"role": "user", "content": args.prompt}],
    )
    info(answer)


def cmd_agent(args: argparse.Namespace) -> None:
    if not args.goal:
        _run_agent_tui(show_steps=args.show_steps)
        return

    settings = _settings()
    workspace, memory, tasks = _runtime(settings)
    prompts = _prompts(settings)
    team = AgentTeam(settings=settings, workspace=workspace, prompts=prompts)
    context = ContextManager(memory=memory, compact_threshold=settings.compact_threshold)
    loop = _build_agent_loop(
        settings,
        workspace,
        context.memory,
        tasks,
        show_steps=args.show_steps,
        permission_manager=PermissionManager(),
        prompts=prompts,
        team=team,
        context=context,
    )
    context.begin_token_tracking()
    bundle = context.begin_turn(args.goal)
    result = loop.run(bundle.user_text, history=bundle.messages)
    try:
        info(result.final_text)
        context.commit_result(result)
        context.replace_history(_inject_team_inbox_for_console(context.history, team))
        tasks.clear_if_complete()
    except Exception as exc:
        error(f"回合收尾失败: {exc}")


def cmd_tui(args: argparse.Namespace) -> None:
    _run_agent_tui(show_steps=args.show_steps)


def _run_agent_tui(show_steps: bool = False) -> None:
    from l1m_cli.tui import run_tui_shell

    run_tui_shell(show_steps=show_steps)


def run_agent_shell(show_steps: bool = False) -> None:
    settings = _settings()
    workspace, memory, tasks = _runtime(settings)
    prompts = _prompts(settings) if hasattr(settings, "prompt_dir") else None
    team = (
        AgentTeam(settings=settings, workspace=workspace, prompts=prompts)
        if prompts is not None
        else None
    )
    context = ContextManager(memory=memory, compact_threshold=getattr(settings, "compact_threshold", None))
    permission_manager = PermissionManager()

    info("L1m Agent")
    info(f"workspace: {workspace.root}")
    info("输入任务开始，输入 /exit 退出，/clear 清空本轮上下文。")

    while True:
        try:
            user_text = input("l1m > ").strip()
        except (EOFError, KeyboardInterrupt):
            info("\nbye")
            break

        if not user_text:
            continue
        if user_text == "/exit":
            info("bye")
            break
        if user_text == "/clear":
            context.clear()
            tasks.clear()
            info("context cleared")
            continue

        try:
            loop = _build_agent_loop(
                settings,
                workspace,
                context.memory,
                tasks,
                show_steps=show_steps,
                permission_manager=permission_manager,
                prompts=prompts,
                team=team,
                context=context,
            )
            context.begin_token_tracking()
            bundle = context.begin_turn(user_text)
            result = loop.run(bundle.user_text, history=bundle.messages)
        except KeyboardInterrupt:
            info("\nbye")
            break
        except L1mError as exc:
            error(str(exc))
            continue
        except Exception as exc:
            error(f"未预期的错误: {exc}")
            continue

        try:
            context.commit_result(result)
            info(result.final_text)
            if team is not None:
                context.replace_history(_inject_team_inbox_for_console(context.history, team))
            tasks.clear_if_complete()
        except Exception as exc:
            error(f"回合收尾失败，已保留本轮上下文: {exc}")


def _inject_team_inbox_for_console(
    history: list[dict[str, Any]],
    team: AgentTeam,
) -> list[dict[str, Any]]:
    inbox_text = team.consume_inbox_for_history()
    if not inbox_text:
        return history
    info("team inbox: 已收到 teammate 消息，已注入下一轮上下文。")
    return [*history, {"role": "user", "content": f"[Team Inbox]\n{inbox_text}"}]


def cmd_doctor(args: argparse.Namespace) -> None:
    settings = _settings()
    workspace = _workspace(settings)
    info(f"platform: {platform.platform()}")
    info(f"python: {sys.version.split()[0]}")
    info(f"cwd: {settings.cwd}")
    info(f"cli_root: {settings.cli_root}")
    info(f"env_files: {len(settings.env_files)}")
    info(f"api_key: {settings.masked_api_key}")
    info(f"model: {settings.model}")
    info(f"workspace_exists: {workspace.root.exists()}")
    info(f"workspace_writable: {workspace.root.exists() and workspace.root.is_dir()}")

    try:
        import anthropic  # noqa: F401

        info("anthropic: installed")
    except ImportError:
        warn("anthropic: missing")

    if args.api_check:
        client = AnthropicModelClient(settings)
        answer = client.create_message(
            system="Reply with only: ok",
            messages=[{"role": "user", "content": "health check"}],
        )
        info(f"api_check: {answer}")


def cmd_prompt_list(_args: argparse.Namespace) -> None:
    settings = _settings()
    names = _prompts(settings).list_names()
    if not names:
        info("(no prompts)")
        return
    for name in names:
        info(name)


def cmd_prompt_show(args: argparse.Namespace) -> None:
    settings = _settings()
    info(_prompts(settings).load(args.name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l1m", description="L1m CLI learning agent demo.")
    parser.add_argument("--version", action="store_true", help="show version and exit")
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="show current configuration")
    config_parser.set_defaults(func=cmd_config)

    ask_parser = subparsers.add_parser("ask", help="ask one question")
    ask_parser.add_argument("prompt")
    ask_parser.set_defaults(func=cmd_ask)

    agent_parser = subparsers.add_parser("agent", help="run agent mode; starts TUI when no goal is given")
    agent_parser.add_argument("goal", nargs="?")
    agent_parser.add_argument("--show-steps", action="store_true")
    agent_parser.set_defaults(func=cmd_agent)

    tui_parser = subparsers.add_parser("tui", help="start the L1m terminal UI explicitly")
    tui_parser.add_argument("--show-steps", action="store_true")
    tui_parser.set_defaults(func=cmd_tui)

    doctor_parser = subparsers.add_parser("doctor", help="check the local demo environment")
    doctor_parser.add_argument("--api-check", action="store_true", help="also try a small model call")
    doctor_parser.set_defaults(func=cmd_doctor)

    prompt_parser = subparsers.add_parser("prompt", help="manage prompt files")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")
    prompt_list_parser = prompt_subparsers.add_parser("list", help="list prompts")
    prompt_list_parser.set_defaults(func=cmd_prompt_list)
    prompt_show_parser = prompt_subparsers.add_parser("show", help="show a prompt")
    prompt_show_parser.add_argument("name")
    prompt_show_parser.set_defaults(func=cmd_prompt_show)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        info(f"l1m {__version__}")
        return
    if not hasattr(args, "func"):
        try:
            _run_agent_tui()
        except L1mError as exc:
            error(str(exc))
            raise SystemExit(1) from exc
        except Exception as exc:
            error(f"未预期的错误: {exc}")
            raise SystemExit(1) from exc
        return
    try:
        args.func(args)
    except L1mError as exc:
        error(str(exc))
        raise SystemExit(1) from exc
    except Exception as exc:
        error(f"未预期的错误: {exc}")
        raise SystemExit(1) from exc
