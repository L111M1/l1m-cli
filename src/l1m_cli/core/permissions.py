from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from l1m_cli.core.hooks import ToolResultOverride, ToolUseContext
from l1m_cli.core.tools import (
    EDIT_FILE_TOOL,
    RUN_COMMAND_TOOL,
    WRITE_FILE_TOOL,
    is_task_tool,
)


class PermissionDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    tool_input: dict[str, Any]
    reason: str
    details: list[str]


PermissionPrompt = Callable[[PermissionRequest], PermissionDecision]


class PermissionManager:
    def __init__(self, prompt: PermissionPrompt | None = None):
        self.prompt = prompt or TerminalPermissionPrompt()
        self.allow_all_for_session = False

    def __call__(self, context: ToolUseContext) -> ToolResultOverride | None:
        hard_deny = _hard_deny_reason(context.tool_name, context.tool_input)
        if hard_deny:
            return ToolResultOverride(f"权限系统拒绝执行: {hard_deny}")

        request = _permission_request(context.tool_name, context.tool_input)
        if request is None:
            return None
        if self.allow_all_for_session:
            return None

        decision = self.prompt(request)
        if decision == PermissionDecision.ALLOW_SESSION:
            self.allow_all_for_session = True
            return None
        if decision == PermissionDecision.ALLOW_ONCE:
            return None
        return ToolResultOverride(f"用户拒绝了本次工具调用: {request.reason}")


class TerminalPermissionPrompt:
    def __init__(self):
        self._rendered_lines = 0

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return PermissionDecision.DENY

        self._rendered_lines = 0
        print()
        self._rendered_lines += 1
        _print_colored("Permission required", "\033[38;5;209m")
        self._rendered_lines += 1
        _print_colored(request.reason, "\033[38;5;252m")
        self._rendered_lines += 1
        for detail in request.details:
            _print_colored(f"  {detail}", "\033[38;5;245m")
            self._rendered_lines += 1

        options = [
            ("允许本次", PermissionDecision.ALLOW_ONCE),
            ("本会话始终允许", PermissionDecision.ALLOW_SESSION),
            ("拒绝", PermissionDecision.DENY),
        ]
        try:
            selected = _select_option([label for label, _decision in options])
            return options[selected][1]
        finally:
            self._rendered_lines += len(options)
            self.clear_rendered_prompt()

    def clear_rendered_prompt(self) -> None:
        if self._rendered_lines <= 0:
            return
        print(f"\033[{self._rendered_lines}A", end="")
        for _index in range(self._rendered_lines):
            print("\r\033[K", end="")
            print("\033[1B", end="")
        print(f"\033[{self._rendered_lines}A", end="")
        self._rendered_lines = 0


def _permission_request(tool_name: str, tool_input: dict[str, Any]) -> PermissionRequest | None:
    if is_task_tool(tool_name):
        return None
    if tool_name == WRITE_FILE_TOOL:
        path = str(tool_input.get("path") or tool_input.get("file_path") or "?")
        content = str(tool_input.get("content") or "")
        mode = "覆盖" if tool_input.get("overwrite") else "写入"
        return PermissionRequest(
            tool_name=tool_name,
            tool_input=tool_input,
            reason=f"L1m 想{mode}文件",
            details=[f"tool: {tool_name}", f"path: {path}", f"chars: {len(content)}"],
        )
    if tool_name == EDIT_FILE_TOOL:
        edits = _batch_or_single(tool_input, "edits", "path")
        paths = [str(item.get("path") or "?") for item in edits]
        return PermissionRequest(
            tool_name=tool_name,
            tool_input=tool_input,
            reason=f"L1m 想编辑 {len(set(paths))} 个文件",
            details=[
                f"tool: {tool_name}",
                f"edits: {len(edits)}",
                *[f"path: {path}" for path in paths],
            ],
        )
    if tool_name == RUN_COMMAND_TOOL:
        commands = _batch_or_single(tool_input, "commands", "command")
        return PermissionRequest(
            tool_name=tool_name,
            tool_input=tool_input,
            reason=f"L1m 想运行 {len(commands)} 条命令",
            details=[f"tool: {tool_name}", *[f"command: {_command_text(item)}" for item in commands]],
        )
    return None


def _hard_deny_reason(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name != RUN_COMMAND_TOOL:
        return ""
    deny_patterns = [
        r"rm\s+-rf\s+/",
        r"\bsudo\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r">\s*/dev/sda",
        r"\bformat\b",
        r"\btaskkill\b",
        r"\breg\s+delete\b",
        r"\bdel\b",
        r"\berase\b",
        r"\brmdir\b",
        r"\bremove-item\b",
    ]
    for command_input in _batch_or_single(tool_input, "commands", "command"):
        text = _command_text(command_input).lower()
        for pattern in deny_patterns:
            if re.search(pattern, text):
                return f"命令匹配硬拒绝规则: {pattern}"
    return ""


def _command_text(tool_input: dict[str, Any]) -> str:
    command = str(tool_input.get("command") or "")
    args = [str(item) for item in tool_input.get("args") or []]
    return " ".join([command, *args]).strip()


def _batch_or_single(tool_input: dict[str, Any], batch_key: str, single_key: str) -> list[dict[str, Any]]:
    batch = tool_input.get(batch_key)
    if isinstance(batch, list):
        return [item for item in batch if isinstance(item, dict)]
    if tool_input.get(single_key) is not None:
        return [tool_input]
    return []


def _select_option(labels: list[str]) -> int:
    selected = 0
    _render_options(labels, selected)
    while True:
        key = _read_key()
        if key == "up":
            selected = (selected - 1) % len(labels)
        elif key == "down":
            selected = (selected + 1) % len(labels)
        elif key == "enter":
            return selected
        elif key == "ctrl_c":
            raise KeyboardInterrupt
        else:
            continue
        print(f"\033[{len(labels)}A", end="")
        _render_options(labels, selected)


def _render_options(labels: list[str], selected: int) -> None:
    blue = "\033[38;5;75m"
    normal = "\033[38;5;252m"
    reset = "\033[0m"
    for index, label in enumerate(labels):
        color = blue if index == selected else normal
        marker = ">" if index == selected else " "
        print(f"\r\033[K{color}{marker} {label}{reset}")


def _read_key() -> str:
    if os.name == "nt":
        import msvcrt

        char = msvcrt.getwch()
        if char == "\x03":
            return "ctrl_c"
        if char == "\r":
            return "enter"
        if char in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            if code == "H":
                return "up"
            if code == "P":
                return "down"
        return char

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x03":
            return "ctrl_c"
        if char in ("\r", "\n"):
            return "enter"
        if char == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _print_colored(text: str, color: str) -> None:
    print(f"{color}{text}\033[0m")
