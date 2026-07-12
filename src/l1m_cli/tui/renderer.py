from __future__ import annotations

import os
import random
import shutil
import sys
import threading
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from l1m_cli import __version__
from l1m_cli.config import Settings
from l1m_cli.core.workspace import Workspace
from l1m_cli.tui.input_box import read_input_box
from l1m_cli.tui.markdown import render_markdown_lines


L1M_BANNER = [
    " _      _   __  __",
    "| |    / | |  \\/  |",
    "| |    | | | |\\/| |",
    "| |___ | | | |  | |",
    "|_____||_| |_|  |_|",
]


@dataclass(frozen=True)
class TuiPalette:
    reset: str = "\033[0m"
    dim: str = "\033[2m"
    bold: str = "\033[1m"
    brand: str = "\033[38;5;147m"
    system: str = "\033[38;5;209m"
    text: str = "\033[38;5;252m"
    muted: str = "\033[38;5;245m"
    panel_rule: str = "\033[38;5;131m"
    accent: str = "\033[38;5;110m"
    user: str = "\033[38;5;120m"
    input_edit: str = "\033[38;5;244m"
    input_text: str = "\033[38;5;255m"
    assistant: str = "\033[38;5;222m"
    tool: str = "\033[38;5;180m"
    intent: str = "\033[38;5;252m"
    error: str = "\033[38;5;203m"


StyledLine = list[tuple[str, str]]


class TuiRenderer:
    def __init__(self, stream: TextIO | None = None, color: bool | None = None):
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty() if color is None else color
        self.palette = TuiPalette()
        self._layout_width = _terminal_width(self.stream)
        self._write_lock = threading.RLock()
        self._thinking_stop = threading.Event()
        self._thinking_thread: threading.Thread | None = None
        self._thinking_tokens = 0
        self._thinking_visible = False
        self._status_line_size: tuple[int, int] | None = None
        self._status_line_active = False
        _make_stream_tolerate_unicode(self.stream)
        if self.color and os.name == "nt":
            os.system("")

    def clear(self) -> None:
        if self.stream.isatty():
            if os.name == "nt" and self.stream is sys.stdout:
                os.system("cls")
            with self._write_lock:
                print("\033[r\033[3J\033[2J\033[H", end="", file=self.stream, flush=True)
            self._status_line_size = None
            self._status_line_active = False

    def intro(self, settings: Settings, workspace: Workspace) -> None:
        self.clear()
        self.system_panel(settings, workspace)
        self.line(
            "输入任务开始，/exit 退出，/clear 清空上下文，/compact 压缩上下文，/help 查看命令。",
            color=self.palette.dim,
        )

    def system_panel(self, settings: Settings, workspace: Workspace) -> None:
        left_lines = [
            [],
            [("Welcome back!", self.palette.text)],
            [],
            *[[(line, self.palette.system)] for line in L1M_BANNER],
            [],
            [("model      ", self.palette.system), (settings.model, self.palette.text)],
            [
                ("workspace  ", self.palette.system),
                (compact_path(workspace.root, max_length=42), self.palette.text),
            ],
        ]
        right_lines = [
            [],
            [],
            [],
            [("Recent activity", self.palette.system)],
            [("当前进程内保存上下文；退出后清空。", self.palette.muted)],
            [("─" * 56, self.palette.panel_rule)],
            [("Shortcuts", self.palette.system)],
            [
                ("/help", self.palette.text),
                (" 查看命令   ", self.palette.muted),
                ("/clear", self.palette.text),
                (" 清空上下文   ", self.palette.muted),
                ("/exit", self.palette.text),
                (" 退出", self.palette.muted),
            ],
            [
                ("/compact", self.palette.text),
                (" 压缩上下文", self.palette.muted),
            ],
        ]
        self.split_panel(left_lines, right_lines, color=self.palette.system)

    def help(self) -> None:
        self.panel(
            "Help",
            [
                "/exit        退出 L1m",
                "/clear       清空本轮上下文、任务和记忆",
                "/compact     立即压缩当前 Agent 上下文",
                "/help        显示这段帮助",
                "普通文本     作为 Agent 任务发送",
            ],
            color=self.palette.accent,
        )

    def user_message(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.panel("You", [text], color=self.palette.user)

    def assistant_message(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.panel("L1m", render_markdown_lines(text), color=self.palette.assistant)

    def status(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.line(f"- {text}", color=self.palette.dim)

    def transient_status(self, text: str) -> None:
        if not self.stream.isatty():
            return
        width = _terminal_width(self.stream)
        shown = _fit_to_width(text, width - 2)
        self.write("\r" + self._style(f"- {shown}", self.palette.dim), end="")

    def clear_transient_status(self) -> None:
        if not self.stream.isatty():
            return
        width = _terminal_width(self.stream)
        self.write("\r" + " " * width + "\r", end="")

    def token_status(self, context_tokens: int, turn_tokens: int) -> None:
        self.context_status(context_tokens)
        self.finish_thinking_tokens(turn_tokens)

    def reset_turn_tokens(self) -> None:
        self.stop_thinking_tokens()
        self._thinking_tokens = 0
        self._thinking_visible = False

    def context_status(self, context_tokens: int, window_tokens: int = 1_000_000) -> None:
        percent = 0.0 if window_tokens <= 0 else min(999.9, context_tokens / window_tokens * 100)
        text = f"context {percent:.1f}%"
        if self.stream.isatty():
            self._status_line_active = True
            self._ensure_status_line_reserved()
            self._write_bottom_right(text)
            return
        width = _terminal_width(self.stream)
        self.line(_fit_to_width(text, width).rjust(width), color=self.palette.muted)

    def start_thinking_tokens(self, initial_tokens: int = 0) -> None:
        self.stop_thinking_tokens()
        self._thinking_tokens = max(0, initial_tokens)
        self._render_thinking_tokens(self._thinking_tokens)

    def finish_thinking_tokens(self, total_tokens: int) -> None:
        self.stop_thinking_tokens()
        self._thinking_tokens = max(0, total_tokens)
        self._render_thinking_tokens(self._thinking_tokens, final=True)

    def update_thinking_tokens(self, total_tokens: int) -> None:
        self._thinking_tokens = max(0, total_tokens)
        self._render_thinking_tokens(self._thinking_tokens)

    def add_thinking_tokens(self, delta_tokens: int) -> None:
        self._thinking_tokens += max(0, delta_tokens)
        self._render_thinking_tokens(self._thinking_tokens)

    def clear_thinking_tokens(self) -> None:
        self.stop_thinking_tokens()
        if not self._thinking_visible:
            return
        self._thinking_visible = False
        if not self.stream.isatty():
            return
        width = _terminal_width(self.stream)
        with self._write_lock:
            print("\r" + " " * width + "\r", end="", file=self.stream, flush=True)

    def stop_thinking_tokens(self) -> None:
        self._thinking_stop.set()
        thread = self._thinking_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._thinking_thread = None

    def _animate_thinking_tokens(self) -> None:
        return

    def _next_thinking_token_delta(self) -> int:
        base = random.randint(90, 260)
        scale = min(420, self._thinking_tokens // 14)
        jitter = random.randint(-35, 95)
        return max(45, base + scale + jitter)

    def _render_thinking_tokens(self, tokens: int, final: bool = False) -> None:
        text = f"- thinking... {tokens:,} tokens"
        self._thinking_visible = True
        if self.stream.isatty():
            width = _terminal_width(self.stream)
            shown = _fit_to_width(text, width)
            with self._write_lock:
                print(
                    "\r" + self._style(shown.ljust(width), self.palette.dim),
                    end="",
                    file=self.stream,
                    flush=True,
                )
            return
        self.line(text, color=self.palette.dim)

    def tool_call(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.line(f"tool  {text}", color=self.palette.tool)

    def action_panel(self, lines: list[str], speaker: str = "L1m") -> None:
        self.clear_thinking_tokens()
        width = _terminal_width(self.stream)
        prefix = ""
        for raw_line in lines or ["下一步行动已展示。"]:
            wrapped = _wrap_line(raw_line, max(20, width - _display_width(prefix)))
            for index, line in enumerate(wrapped):
                shown_prefix = prefix if index == 0 else " " * _display_width(prefix)
                self.line(f"{shown_prefix}{line}", color=self.palette.intent)

    def file_change(self, title: str, lines: list[str]) -> None:
        self.clear_thinking_tokens()
        self.panel(title, lines, color=self.palette.tool)

    def task_panel(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.panel("Tasks", text.splitlines(), color=self.palette.accent)

    def result_panel(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.panel("Result", text.splitlines(), color=self.palette.dim)

    def error(self, text: str) -> None:
        self.clear_thinking_tokens()
        self.panel("Error", [text], color=self.palette.error)

    def read_input(self) -> str:
        width = _terminal_width(self.stream)
        if self.stream.isatty():
            text = read_input_box()
            self.finish_input(text)
            return text

        self.line("\n" + _input_rule(width), color=self.palette.input_edit)
        text = input(self._style("> ", self.palette.input_edit))
        self.finish_input(text)
        return text

    def prompt(self) -> str:
        return self._style("> ", self.palette.input_edit)

    def finish_input(self, text: str) -> None:
        width = _terminal_width(self.stream)
        self.line(_input_rule(width), color=self.palette.input_edit)
        for line in _input_display_lines(text, width):
            self.line(line, color=self.palette.input_text)
        self.line(_input_rule(width), color=self.palette.input_edit)

    def layout_width_changed(self) -> bool:
        return self.stream.isatty() and _terminal_width(self.stream) != self._layout_width

    def remember_layout_width(self) -> None:
        self._layout_width = _terminal_width(self.stream)

    def panel(self, title: str, lines: list[str], color: str = "") -> None:
        width = _terminal_width(self.stream)
        title_text = f" {title} " if title else ""
        self.line(_top_border(width, title_text), color=color)
        for raw_line in lines or [""]:
            wrapped = _wrap_line(raw_line, width - 4)
            for line in wrapped:
                self.write(self._style(_boxed_line(line, width), color))
        self.line(_bottom_border(width), color=color)

    def split_panel(
        self,
        left_lines: list[str | StyledLine],
        right_lines: list[str | StyledLine],
        color: str = "",
    ) -> None:
        width = _terminal_width(self.stream)
        left_width, right_width = _split_widths(width)
        self.write(self._top_border_system(width))
        row_count = max(len(left_lines), len(right_lines))
        for index in range(row_count):
            left = _normalize_styled_line(left_lines[index]) if index < len(left_lines) else []
            right = _normalize_styled_line(right_lines[index]) if index < len(right_lines) else []
            self.write(self._styled_split_line(left, right, left_width, right_width, color))
        self.line(_bottom_border(width), color=color)

    def line(self, text: str, color: str = "") -> None:
        self.write(self._style(text, color))

    def write(self, text: str, end: str = "\n") -> None:
        if self.stream.isatty() and self._status_line_active:
            self._ensure_status_line_reserved()
        with self._write_lock:
            print(text, end=end, file=self.stream, flush=True)

    def _style(self, text: str, color: str) -> str:
        if not self.color or not color:
            return text
        return f"{color}{text}{self.palette.reset}"

    def _styled_split_line(
        self,
        left: StyledLine,
        right: StyledLine,
        left_width: int,
        right_width: int,
        border_color: str,
    ) -> str:
        if not self.color:
            return _boxed_split_line(
                _plain_styled_line(left),
                _plain_styled_line(right),
                left_width,
                right_width,
            )

        return (
            self._style("│ ", border_color)
            + self._styled_cell(left, left_width)
            + self._style(" │ ", border_color)
            + self._styled_cell(right, right_width)
            + self._style(" │", border_color)
        )

    def _styled_cell(self, parts: StyledLine, width: int) -> str:
        clipped, used = _fit_styled_line(parts, width)
        text = "".join(self._style(part, color) for part, color in clipped)
        return text + " " * max(0, width - used)

    def _top_border_system(self, width: int) -> str:
        title = " L1m "
        version = f"v{_minor_version(__version__)} "
        plain_title = title + version
        border = "╭" + plain_title + "─" * max(0, width - len(plain_title) - 2) + "╮"
        if not self.color:
            return border
        return (
            self.palette.system
            + "╭"
            + title
            + self.palette.muted
            + version
            + self.palette.system
            + "─" * max(0, width - len(plain_title) - 2)
            + "╮"
            + self.palette.reset
        )

    def _write_bottom_right(self, text: str) -> None:
        size = shutil.get_terminal_size((100, 30))
        width = max(1, size.columns)
        row = max(1, size.lines)
        shown = _fit_to_width(text, width)
        col = max(1, width - _display_width(shown) + 1)
        with self._write_lock:
            print("\033[s", end="", file=self.stream, flush=True)
            print(f"\033[{row};{col}H", end="", file=self.stream, flush=True)
            print("\033[K", end="", file=self.stream, flush=True)
            print(self._style(shown, self.palette.muted), end="", file=self.stream, flush=True)
            print("\033[u", end="", file=self.stream, flush=True)

    def _ensure_status_line_reserved(self) -> None:
        if not self.stream.isatty() or not self._status_line_active:
            return
        size = shutil.get_terminal_size((100, 30))
        width = max(1, size.columns)
        rows = max(2, size.lines)
        current = (width, rows)
        if self._status_line_size == current:
            return
        self._status_line_size = current
        scroll_bottom = max(1, rows - 1)
        with self._write_lock:
            print("\033[s", end="", file=self.stream, flush=True)
            print(f"\033[1;{scroll_bottom}r", end="", file=self.stream, flush=True)
            print(f"\033[{rows};1H\033[K", end="", file=self.stream, flush=True)
            print("\033[u", end="", file=self.stream, flush=True)


def _terminal_width(stream: TextIO) -> int:
    if stream.isatty():
        return max(60, min(shutil.get_terminal_size((100, 30)).columns, 120))
    return 88


def _top_border(width: int, title: str) -> str:
    title = title[: max(0, width - 4)]
    return "╭" + title + "─" * max(0, width - len(title) - 2) + "╮"


def _bottom_border(width: int) -> str:
    return "╰" + "─" * max(0, width - 2) + "╯"


def _input_rule(width: int) -> str:
    return "─" * width


def _input_line(line: str, width: int) -> str:
    clipped = _fit_to_width(line, width)
    return clipped + " " * max(0, width - _display_width(clipped))


def _input_display_lines(text: str, width: int) -> list[str]:
    content_width = max(1, width - 2)
    rows: list[str] = []
    for raw_index, raw_line in enumerate(text.splitlines() or [""]):
        wrapped = _wrap_line(raw_line, content_width)
        for wrap_index, wrapped_line in enumerate(wrapped):
            prefix = "> " if raw_index == 0 and wrap_index == 0 else "  "
            rows.append(_input_line(prefix + wrapped_line, width))
    return rows or [_input_line("> ", width)]


def _thin_rule(width: int, title: str = "") -> str:
    if not title:
        return "─" * width
    title = title[: max(0, width - 4)]
    return "─" * 2 + title + "─" * max(0, width - len(title) - 2)


def _split_widths(width: int) -> tuple[int, int]:
    usable = max(40, width - 7)
    left_width = min(44, max(24, usable // 2 - 2))
    right_width = usable - left_width
    if right_width < 24:
        right_width = 24
        left_width = max(16, usable - right_width)
    return left_width, right_width


def _boxed_line(line: str, width: int) -> str:
    content_width = max(1, width - 4)
    clipped = _fit_to_width(line, content_width)
    return f"│ {clipped}{' ' * max(0, content_width - _display_width(clipped))} │"


def _boxed_split_line(left: str, right: str, left_width: int, right_width: int) -> str:
    left_clipped = _fit_to_width(left, left_width)
    right_clipped = _fit_to_width(right, right_width)
    left_pad = " " * max(0, left_width - _display_width(left_clipped))
    right_pad = " " * max(0, right_width - _display_width(right_clipped))
    return f"│ {left_clipped}{left_pad} │ {right_clipped}{right_pad} │"


def _normalize_styled_line(line: str | StyledLine) -> StyledLine:
    if isinstance(line, str):
        return [(line, "")]
    return line


def _plain_styled_line(parts: StyledLine) -> str:
    return "".join(text for text, _color in parts)


def _fit_styled_line(parts: StyledLine, width: int) -> tuple[StyledLine, int]:
    result: StyledLine = []
    used = 0
    for text, color in parts:
        remaining = max(0, width - used)
        if remaining <= 0:
            break
        clipped = _fit_to_width(text, remaining)
        if clipped:
            result.append((clipped, color))
            used += _display_width(clipped)
        if _display_width(clipped) < _display_width(text):
            break
    return result, used


def _wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    if _display_width(line) <= width:
        return [line]
    if _is_mostly_ascii(line):
        return textwrap.wrap(
            line,
            width=max(20, width),
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [line]
    return _wrap_wide(line, max(20, width))


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _fit_to_width(text: str, width: int) -> str:
    used = 0
    chars: list[str] = []
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        )
        if used + char_width > width:
            break
        chars.append(char)
        used += char_width
    return "".join(chars)


def _wrap_wide(text: str, width: int) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        )
        if current and used + char_width > width:
            rows.append("".join(current))
            current = []
            used = 0
        current.append(char)
        used += char_width
    if current:
        rows.append("".join(current))
    return rows or [""]


def _is_mostly_ascii(text: str) -> bool:
    if not text:
        return True
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return ascii_count / len(text) > 0.85


def compact_path(path: Path, max_length: int = 80) -> str:
    text = str(path)
    if len(text) <= max_length:
        return text
    return "..." + text[-(max_length - 3) :]


def _minor_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return version


def _make_stream_tolerate_unicode(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(errors="replace")
    except (TypeError, ValueError):
        return
