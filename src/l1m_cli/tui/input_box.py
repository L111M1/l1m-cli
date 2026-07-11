from __future__ import annotations

import math
import os
import shutil
import unicodedata

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style


INPUT_RULE_STYLE = "#808080"
INPUT_TEXT_STYLE = "#ffffff"


def read_input_box() -> str:
    buffer = Buffer(multiline=True)
    control = BufferControl(
        buffer=buffer,
        input_processors=[BeforeInput("> ", style="class:prompt")],
    )
    key_bindings = KeyBindings()

    @key_bindings.add("enter")
    def _(event) -> None:
        event.app.exit(result=buffer.text)

    @key_bindings.add("escape", "enter")
    def _(event) -> None:
        buffer.insert_text("\n")

    @key_bindings.add("c-c")
    def _(event) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    @key_bindings.add("c-d")
    def _(event) -> None:
        event.app.exit(exception=EOFError)

    root = HSplit(
        [
            Window(char="─", height=1, style="class:input-rule"),
            Window(
                content=control,
                height=lambda: Dimension.exact(_input_height(buffer.text)),
                dont_extend_height=True,
                wrap_lines=True,
                style="class:input-text",
            ),
            Window(char="─", height=1, style="class:input-rule"),
        ]
    )
    app: Application[str] = Application(
        layout=Layout(root, focused_element=control),
        key_bindings=key_bindings,
        style=Style.from_dict(
            {
                "input-rule": INPUT_RULE_STYLE,
                "input-text": INPUT_TEXT_STYLE,
                "prompt": INPUT_RULE_STYLE,
            }
        ),
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
        terminal_size_polling_interval=0.1,
        output=_stable_output(),
    )
    return app.run()


def _stable_output(is_windows: bool | None = None):
    output = create_output()
    windows = os.name == "nt" if is_windows is None else is_windows
    if windows:
        # prompt_toolkit 默认会把 Windows 控制台视口滚到输入位置，导致首屏面板被推入滚动区。
        output.scroll_buffer_to_prompt = lambda: None
    return output


def _input_height(text: str, max_height: int = 8) -> int:
    width = max(3, shutil.get_terminal_size((100, 30)).columns)
    height = 0
    for index, line in enumerate(text.split("\n") or [""]):
        prompt_width = 2 if index == 0 else 0
        line_width = prompt_width + _display_width(line)
        height += max(1, math.ceil(line_width / width))
    return max(1, min(max_height, height))


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width
