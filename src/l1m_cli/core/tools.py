from __future__ import annotations

import json
import locale
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal

from l1m_cli.core.task_system import TaskStore
from l1m_cli.core.workspace import Workspace


ToolHandler = Callable[[dict[str, Any]], str]
ToolDisplayMode = Literal["console", "tui"]
ToolProfile = Literal["default", "lead"]

ANNOUNCE_ACTION_TOOL = "announce_action"
READ_FILE_TOOL = "read_file"
WRITE_FILE_TOOL = "write_file"
EDIT_FILE_TOOL = "edit_file"
RUN_COMMAND_TOOL = "run_command"
FINAL_CHECK_TOOL = "final_check"
TASK_TOOL_PREFIX = "task_"
TASK_CREATE_TOOL = "task_create"
TASK_LIST_TOOL = "task_list"
TASK_GET_TOOL = "task_get"
TASK_UPDATE_TOOL = "task_update"
TEAM_TOOL_PREFIX = "team_"
TEAM_SPAWN_AGENT_TOOL = "team_spawn_agent"
TEAM_SPAWN_REVIEWED_AGENT_TOOL = "team_spawn_reviewed_agent"
TEAM_SEND_MESSAGE_TOOL = "team_send_message"
TEAM_CHECK_INBOX_TOOL = "team_check_inbox"
TEAM_WAIT_FOR_INBOX_TOOL = "team_wait_for_inbox"
TEAM_REQUEST_PLAN_TOOL = "team_request_plan"
TEAM_SUBMIT_PLAN_TOOL = "team_submit_plan"
TEAM_REVIEW_PLAN_TOOL = "team_review_plan"
LEAD_AGENT_NAME = "lead"
PROCESS_OUTPUT_BYTE_LIMIT = 128_000
PROCESS_OUTPUT_TEXT_LIMIT = 12_000
PROCESS_PIPE_CHUNK_SIZE = 8192
MAX_PARALLEL_WORKERS = 8


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def specs(self) -> list[dict[str, Any]]:
        return [tool.to_anthropic() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def run(self, name: str, tool_input: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具不存在: {name}"
        try:
            return tool.handler(tool_input)
        except Exception as exc:
            return f"工具执行失败: {type(exc).__name__}: {exc}"


def build_tool_registry(
    workspace: Workspace,
    tasks: TaskStore | None = None,
    team: Any | None = None,
    team_sender: str = LEAD_AGENT_NAME,
    allow_team_spawn: bool = True,
    tool_profile: ToolProfile | None = None,
) -> ToolRegistry:
    tools = [
        Tool(
            name=ANNOUNCE_ACTION_TOOL,
            description=(
                "在调用其他工具前，先向用户展示一条自然的工作想法。"
                "不要强制使用固定开头；按当前情况直接写你接下来准备做什么。"
                "后续步骤可简短回看上一步工具结果或当前状态，并说明下一步行动；"
                "只说明可公开的观察和行动，不输出隐藏推理过程；"
                "不要暴露 agent 身份、创建子 agent、回传 lead 等内部协作机制。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "previous": {
                        "type": "string",
                        "description": (
                            "可选。一句话回看上一步工具结果或当前状态，例如 刚才已经确认入口文件存在。"
                            "任务第一步没有上一步结果时不要填写。"
                        ),
                    },
                    "action": {
                        "type": "string",
                        "description": "一句自然短句说明接下来要做什么；不要套固定开头，不必以“我先”开头。",
                    },
                    "tool": {
                        "type": "string",
                        "description": "接下来准备调用的工具名，例如 read_file、write_file。这个字段用于流程控制，展示时默认不直接说出来。",
                    },
                    "target": {
                        "type": "string",
                        "description": "准备处理的文件、命令或任务对象，可选。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "简短说明为什么要做这一步，可选。",
                    },
                },
                "required": ["action", "tool"],
            },
            handler=_announce_action,
        ),
        Tool(
            name=READ_FILE_TOOL,
            description=(
                "并行读取 workspace 内的一个或多个文本文件。需要一次查看多个文件时，把所有读取请求放进 files 数组，"
                "一次调用即可并行读取；单个文件仍可使用 path。文件较大时用 start_line/end_line 或 "
                "offset/max_chars 分段读取，不要改用命令读取文件。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "description": "要并行读取的文件请求；需要读取多个文件时优先使用。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "相对 workspace 的文件路径。"},
                                "max_chars": {"type": "integer", "description": "最多读取字符数，默认 60000，最大 200000。"},
                                "offset": {"type": "integer", "description": "从第几个字符开始读取，默认 0。"},
                                "start_line": {"type": "integer", "description": "从第几行开始读取，1-based。"},
                                "end_line": {"type": "integer", "description": "读取到第几行结束，1-based，包含该行。"},
                            },
                            "required": ["path"],
                        },
                    },
                    "path": {"type": "string", "description": "兼容单文件读取的相对 workspace 路径。"},
                    "max_chars": {"type": "integer", "description": "最多读取字符数，默认 60000，最大 200000。"},
                    "offset": {"type": "integer", "description": "从第几个字符开始读取，默认 0。"},
                    "start_line": {"type": "integer", "description": "从第几行开始读取，1-based。"},
                    "end_line": {"type": "integer", "description": "读取到第几行结束，1-based，包含该行。"},
                },
            },
            handler=lambda data: _read_file(workspace, data),
        ),
        Tool(
            name=WRITE_FILE_TOOL,
            description=(
                "写入 workspace 内的文本文件。path 必须是明确的相对文件路径，例如 index.html 或 src/styles.css；"
                "不能使用 ?、todo、unknown、placeholder 等占位路径。content 必须是要写入的完整文本内容；"
                "不要先写空文件再稍后补内容。默认不覆盖已有文件，除非 overwrite 为 true。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "明确的相对 workspace 文件路径，例如 index.html、src/main.js；不要填写 ? 或占位名称。",
                        "minLength": 1,
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文本内容；默认不能为空，除非 allow_empty=true。",
                    },
                    "overwrite": {"type": "boolean", "description": "是否允许覆盖已有文件，默认 false。"},
                    "allow_empty": {
                        "type": "boolean",
                        "description": "只有确实需要创建空文件（例如 __init__.py）时才设为 true，默认 false。",
                    },
                },
                "required": ["path", "content"],
            },
            handler=lambda data: _write_file(workspace, data),
        ),
        Tool(
            name=EDIT_FILE_TOOL,
            description=(
                "编辑 workspace 内的一个或多个已有文件。需要修改多个文件时，把所有编辑请求放进 edits 数组，"
                "一次调用即可并行处理不同文件；同一文件内的多项编辑会按数组顺序执行。old_str 必须在对应文件中唯一匹配。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "description": "要执行的编辑请求；不同文件并行，同一文件按顺序执行。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "相对 workspace 的文件路径。"},
                                "old_str": {"type": "string", "description": "要替换的原始文本，必须精确且唯一匹配。"},
                                "new_str": {"type": "string", "description": "替换后的新文本。"},
                            },
                            "required": ["path", "old_str", "new_str"],
                        },
                    },
                    "path": {"type": "string", "description": "兼容单文件编辑的相对 workspace 路径。"},
                    "old_str": {"type": "string", "description": "要替换的原始文本，必须精确匹配。"},
                    "new_str": {"type": "string", "description": "替换后的新文本。"},
                },
            },
            handler=lambda data: _edit_file(workspace, data),
        ),
        Tool(
            name=RUN_COMMAND_TOOL,
            description=(
                "在 workspace 目录下并行运行一个或多个独立命令。需要运行多条命令时，把所有命令请求放进 commands 数组，"
                "一次调用即可并行执行；单条命令仍可使用 command/args。需要观察目录时可以用 cmd /c dir、cmd /c tree /f "
                "或 powershell -Command Get-ChildItem。读取文件内容请优先用 read_file。 "
                "危险的删除、移动、格式化等命令会被拒绝。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "minItems": 1,
                        "description": "要并行执行的独立命令请求。存在依赖关系的命令不要放在同一批。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "可执行程序，例如 cmd、powershell、python。"},
                                "args": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "参数数组。",
                                },
                                "timeout": {"type": "integer", "description": "超时时间秒数，默认 30，最大 120。"},
                            },
                            "required": ["command"],
                        },
                    },
                    "command": {"type": "string", "description": "兼容单条命令执行的可执行程序。"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参数数组，例如 ['/c', 'dir'] 或 ['-m', 'pytest', '-q']。",
                    },
                    "timeout": {"type": "integer", "description": "超时时间秒数，默认 30，最大 120。"},
                },
            },
            handler=lambda data: _run_command(workspace, data),
        ),
    ]
    _ = tool_profile
    if tasks is not None:
        tools.extend(_build_task_tools(tasks))
    if team is not None:
        tools.extend(_build_team_tools(team, sender=team_sender, allow_spawn=allow_team_spawn))
    return ToolRegistry(tools)


def build_workspace_tools(
    workspace: Workspace,
    tasks: TaskStore | None = None,
    team: Any | None = None,
    team_sender: str = LEAD_AGENT_NAME,
    allow_team_spawn: bool = True,
    tool_profile: ToolProfile | None = None,
) -> ToolRegistry:
    return build_tool_registry(
        workspace,
        tasks=tasks,
        team=team,
        team_sender=team_sender,
        allow_team_spawn=allow_team_spawn,
        tool_profile=tool_profile,
    )


def format_action_announcement(data: dict[str, Any]) -> list[str]:
    previous = _single_line(
        data.get("previous")
        or data.get("last_step")
        or data.get("observation")
        or data.get("status")
        or ""
    )
    action = _single_line(data.get("action") or data.get("summary") or "")
    tool_name = _single_line(data.get("tool") or data.get("tool_name") or "")

    if previous and action:
        return [_combine_previous_action(previous, action)]
    if action:
        return [_natural_action(action)]
    if not tool_name:
        return ["正在确认接下来该做什么。"]
    return ["继续处理当前步骤。"]


def should_display_action_announcement(data: dict[str, Any]) -> bool:
    tool_name = _single_line(data.get("tool") or data.get("tool_name") or "")
    return not is_task_tool(tool_name)


def _announce_action(data: dict[str, Any]) -> str:
    return "\n".join(format_action_announcement(data))


def is_action_tool(name: str) -> bool:
    return name == ANNOUNCE_ACTION_TOOL


def is_task_tool(name: str) -> bool:
    return name.startswith(TASK_TOOL_PREFIX)


def is_team_tool(name: str) -> bool:
    return name.startswith(TEAM_TOOL_PREFIX)


def format_tool_call(name: str, tool_input: dict[str, Any], mode: ToolDisplayMode = "tui") -> str:
    if mode == "console":
        return _format_console_tool_call(name, tool_input)
    return _format_tui_tool_call(name, tool_input)


def _format_console_tool_call(name: str, tool_input: dict[str, Any]) -> str:
    if name == ANNOUNCE_ACTION_TOOL:
        return "tool: announce_action"
    if name == RUN_COMMAND_TOOL:
        return f"tool: run_command -> {_tool_item_count(tool_input, 'commands', 'command')} commands"
    if name == FINAL_CHECK_TOOL:
        return "tool: final_check"
    if name == READ_FILE_TOOL:
        return f"tool: read_file -> {_tool_item_count(tool_input, 'files', 'path')} files"
    if name == WRITE_FILE_TOOL:
        return f"tool: write_file -> {_path_from_tool_input(tool_input)}"
    if name == EDIT_FILE_TOOL:
        return f"tool: edit_file -> {_tool_item_count(tool_input, 'edits', 'path')} files"
    if name == TASK_CREATE_TOOL:
        return f"tool: task_create -> {len(tool_input.get('items') or [])} tasks"
    if name == TASK_GET_TOOL:
        return f"tool: task_get -> {tool_input.get('task_id')}"
    if name == TASK_UPDATE_TOOL:
        return f"tool: task_update -> {tool_input.get('task_id')} {tool_input.get('status')}"
    if name == TASK_LIST_TOOL:
        return "tool: task_list"
    if name == TEAM_SPAWN_AGENT_TOOL:
        return f"tool: team_spawn_agent -> {tool_input.get('name') or '?'}"
    if name == TEAM_SPAWN_REVIEWED_AGENT_TOOL:
        return f"tool: team_spawn_reviewed_agent -> {tool_input.get('name') or '?'}"
    if name == TEAM_SEND_MESSAGE_TOOL:
        return f"tool: team_send_message -> {tool_input.get('to') or '?'}"
    if name == TEAM_CHECK_INBOX_TOOL:
        return "tool: team_check_inbox"
    if name == TEAM_WAIT_FOR_INBOX_TOOL:
        return "tool: team_wait_for_inbox"
    if name == TEAM_REQUEST_PLAN_TOOL:
        return f"tool: team_request_plan -> {tool_input.get('teammate') or tool_input.get('to') or '?'}"
    if name == TEAM_SUBMIT_PLAN_TOOL:
        return f"tool: team_submit_plan -> {tool_input.get('request_id') or '?'}"
    if name == TEAM_REVIEW_PLAN_TOOL:
        return f"tool: team_review_plan -> {tool_input.get('request_id') or '?'}"
    return f"tool: {name} -> {json.dumps(tool_input, ensure_ascii=False)}"


def _format_tui_tool_call(name: str, tool_input: dict[str, Any]) -> str:
    if name == ANNOUNCE_ACTION_TOOL:
        return "announce_action"
    if name == RUN_COMMAND_TOOL:
        return f"run_command  运行 {_tool_item_count(tool_input, 'commands', 'command')} 条命令"
    if name == FINAL_CHECK_TOOL:
        return "final_check"
    if name == READ_FILE_TOOL:
        return f"read_file    读取 {_tool_item_count(tool_input, 'files', 'path')} 个文件"
    if name == WRITE_FILE_TOOL:
        return f"write_file   {_path_from_tool_input(tool_input)}"
    if name == EDIT_FILE_TOOL:
        return f"edit_file    编辑 {_tool_item_count(tool_input, 'edits', 'path')} 个文件"
    if name == TASK_CREATE_TOOL:
        return f"task_create  {len(tool_input.get('items') or [])} tasks"
    if name == TASK_GET_TOOL:
        return f"task_get     {tool_input.get('task_id')}"
    if name == TASK_UPDATE_TOOL:
        return f"task_update  {tool_input.get('task_id')} {tool_input.get('status')}"
    if name == TASK_LIST_TOOL:
        return "task_list"
    if name == TEAM_SPAWN_AGENT_TOOL:
        return f"team_spawn   {tool_input.get('name') or '?'}"
    if name == TEAM_SPAWN_REVIEWED_AGENT_TOOL:
        return f"team_review  {tool_input.get('name') or '?'}"
    if name == TEAM_SEND_MESSAGE_TOOL:
        return f"team_message {tool_input.get('to') or '?'}"
    if name == TEAM_CHECK_INBOX_TOOL:
        return "team_inbox"
    if name == TEAM_WAIT_FOR_INBOX_TOOL:
        return "team_wait"
    if name == TEAM_REQUEST_PLAN_TOOL:
        return f"team_plan?   {tool_input.get('teammate') or tool_input.get('to') or '?'}"
    if name == TEAM_SUBMIT_PLAN_TOOL:
        return f"team_plan    {tool_input.get('request_id') or '?'}"
    if name == TEAM_REVIEW_PLAN_TOOL:
        return f"team_review  {tool_input.get('request_id') or '?'}"
    return f"{name}  {json.dumps(tool_input, ensure_ascii=False)}"


def _command_text_from_input(tool_input: dict[str, Any]) -> str:
    command = str(tool_input.get("command") or "")
    args = [str(item) for item in tool_input.get("args") or []]
    return " ".join([command, *args]).strip()


def _tool_item_count(tool_input: dict[str, Any], batch_key: str, single_key: str) -> int:
    batch = tool_input.get(batch_key)
    if isinstance(batch, list):
        return len(batch)
    return 1 if tool_input.get(single_key) is not None else 0


def _path_from_tool_input(tool_input: dict[str, Any]) -> str:
    raw_path = tool_input.get("path") or tool_input.get("file_path") or ""
    reason = _invalid_workspace_file_path_reason(raw_path)
    if reason:
        return "(invalid path)"
    return str(raw_path)


def invalid_workspace_file_path_reason(value: Any) -> str:
    return _invalid_workspace_file_path_reason(value)


def _invalid_workspace_file_path_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "缺少参数: path（请指定明确的相对文件路径，例如 index.html 或 src/styles.css）"
    normalized = text.replace("\\", "/").strip().lower()
    placeholder_names = {
        "?",
        "??",
        "...",
        "todo",
        "tbd",
        "unknown",
        "placeholder",
        "file",
        "filename",
        "path",
        "null",
        "none",
    }
    if normalized in placeholder_names:
        return f"无效 path: {text}（不能使用占位路径，请先确定真实文件名）"
    if any(part in placeholder_names for part in normalized.split("/")):
        return f"无效 path: {text}（路径片段不能是占位符）"
    return ""


def _single_line(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    text = " ".join(text.splitlines())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _natural_action(action: str) -> str:
    text = action.strip(" ：:，,。.")
    if not text:
        return "继续处理当前步骤。"
    return _sentence(text)


def _natural_previous(previous: str) -> str:
    text = previous.strip(" ：:，,。.")
    if not text:
        return ""
    return _sentence(text)


def _combine_previous_action(previous: str, action: str) -> str:
    previous_text = _strip_sentence_end(_natural_previous(previous))
    action_text = _strip_sentence_end(_natural_action(action))
    return _sentence(f"{previous_text}；{action_text}")


def _strip_sentence_end(text: str) -> str:
    return text.rstrip("。！？，,.!? ")


def _sentence(text: str) -> str:
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else text + "。"


def _build_task_tools(tasks: TaskStore) -> list[Tool]:
    return [
        Tool(
            name=FINAL_CHECK_TOOL,
            description=(
                "在准备最终回复前，由模型自行选择调用的最终检查工具。"
                "如果任务面板中已经包含并完成了明确的验证/测试/检查任务，不要调用它。"
                "如果任务面板没有验证任务，或你不确定实际项目是否已经符合用户目标，调用它获取验收清单。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "用户原始目标或当前任务目标。"},
                    "summary": {"type": "string", "description": "你准备最终回复前对当前完成情况的简短判断。"},
                },
            },
            handler=lambda data: _final_check(tasks, data),
        ),
        Tool(
            name=TASK_CREATE_TOOL,
            description=(
                "把复杂用户目标拆成 3-6 个实现/产出里程碑任务并注册。"
                "任务面板只用于展示进度，不是调度引擎；简单任务不需要创建任务。"
                "不要随手创建纯最终验收、整体验证、联调检查任务；收尾检查由模型按需要调用 final_check。"
                "可以用 reset=true 清空旧任务并创建新计划。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "任务标题。"},
                                "description": {"type": "string", "description": "任务说明。"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "任务初始状态。",
                                },
                                "active_form": {
                                    "type": "string",
                                    "description": "任务进行中显示的动词短语，可选。",
                                },
                                "blocked_by": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "依赖任务 ID 列表，仅用于记录关系，不会自动驱动流程。",
                                },
                            },
                            "required": ["title"],
                        },
                    },
                    "reset": {"type": "boolean", "description": "是否清空旧任务后创建新计划，默认 false。"},
                },
                "required": ["items"],
            },
            handler=lambda data: _task_create(tasks, data),
        ),
        Tool(
            name=TASK_LIST_TOOL,
            description="查看当前任务列表和状态。默认隐藏依赖等细节。",
            input_schema={"type": "object", "properties": {}},
            handler=lambda _data: tasks.render_panel() or "(no tasks)",
        ),
        Tool(
            name=TASK_GET_TOOL,
            description="查看某个任务的完整详情。",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID，例如 task_001。"},
                },
                "required": ["task_id"],
            },
            handler=lambda data: _task_get(tasks, data),
        ),
        Tool(
            name=TASK_UPDATE_TOOL,
            description="更新任务标题、说明、状态或进度说明。只有真实完成后才把状态改为 completed。",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID，例如 task_001。"},
                    "title": {"type": "string", "description": "新的任务标题，可选。"},
                    "description": {"type": "string", "description": "新的任务说明，可选。"},
                    "active_form": {"type": "string", "description": "任务进行中动词短语，可选。"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "新的任务状态。",
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "依赖任务 ID 列表，可选。",
                    },
                    "note": {"type": "string", "description": "进度说明，可选。"},
                },
                "required": ["task_id"],
            },
            handler=lambda data: _task_update(tasks, data),
        ),
    ]


def _build_team_tools(team: Any, sender: str = LEAD_AGENT_NAME, allow_spawn: bool = True) -> list[Tool]:
    sender_name = team.normalize_agent_name(sender)
    tools = [
        Tool(
            name=TEAM_SEND_MESSAGE_TOOL,
            description="向 lead 或某个已创建的 teammate 发送团队消息。",
            input_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "接收者名称，例如 lead 或 reviewer。"},
                    "content": {"type": "string", "description": "消息内容，简洁说明事实、结果或请求。"},
                },
                "required": ["to", "content"],
            },
            handler=lambda data: team.send_team_message(sender_name, data),
        ),
        Tool(
            name=TEAM_CHECK_INBOX_TOOL,
            description="读取当前 agent 的团队 inbox。lead 用它收集 teammate 的进展和结果。",
            input_schema={"type": "object", "properties": {}},
            handler=lambda _data: team.check_team_inbox(sender_name),
        ),
    ]
    if sender_name == LEAD_AGENT_NAME:
        tools.append(
            Tool(
                name=TEAM_WAIT_FOR_INBOX_TOOL,
                description=(
                    "让 lead 在本地暂停等待团队 inbox 的新消息。适合一次性创建/分配好 teammate 后调用；"
                    "调用期间 CLI 不会继续请求模型，直到收到未处理消息、所有 teammate 结束或超时。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "timeout_seconds": {
                            "type": "number",
                            "description": "最长等待秒数，默认 300，最大 1800。",
                        },
                        "poll_interval_seconds": {
                            "type": "number",
                            "description": "轮询间隔秒数，默认 1。",
                        },
                    },
                },
                handler=lambda data: team.wait_for_team_inbox(sender_name, data),
            )
        )
    if sender_name != LEAD_AGENT_NAME:
        tools.append(
            Tool(
                name=TEAM_SUBMIT_PLAN_TOOL,
                description="Submit an execution plan for a plan_request. Wait for lead approval before proceeding.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string", "description": "The request_id from the plan_request message."},
                        "plan": {"type": "string", "description": "A concise, executable, verifiable plan."},
                    },
                    "required": ["request_id", "plan"],
                },
                handler=lambda data: team.submit_team_plan(sender_name, data),
            )
        )
    if allow_spawn and sender_name == LEAD_AGENT_NAME:
        tools.insert(
            0,
            Tool(
                name=TEAM_REVIEW_PLAN_TOOL,
                description="Approve or reject a teammate plan by request_id.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string", "description": "The plan request_id to review."},
                        "approve": {"type": "boolean", "description": "True to approve, false to reject."},
                        "feedback": {"type": "string", "description": "Feedback for the teammate, especially when rejecting."},
                    },
                    "required": ["request_id", "approve"],
                },
                handler=lambda data: team.review_team_plan(sender_name, data),
            ),
        )
        tools.insert(
            0,
            Tool(
                name=TEAM_REQUEST_PLAN_TOOL,
                description="Ask a teammate to submit an execution plan before it starts work.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "teammate": {"type": "string", "description": "The teammate name."},
                        "task": {"type": "string", "description": "The task that needs a plan."},
                    },
                    "required": ["teammate", "task"],
                },
                handler=lambda data: team.request_team_plan(sender_name, data),
            ),
        )
        tools.insert(
            0,
            Tool(
                name=TEAM_SPAWN_AGENT_TOOL,
                description=(
                    "创建一个后台 teammate agent。只适合主 agent/lead 用来分派边界清晰、"
                    "可并行的子任务；teammate 不会获得继续创建 agent 的工具。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "teammate 名称，使用短英文或拼音，例如 reviewer、frontend。",
                        },
                        "role": {"type": "string", "description": "teammate 的职责角色。"},
                        "prompt": {
                            "type": "string",
                            "description": "清晰、可独立执行的任务说明，包含范围、交付物和回报方式。",
                        },
                    },
                    "required": ["name", "role", "prompt"],
                },
                handler=team.spawn_team_agent,
            ),
        )
        tools.insert(
            0,
            Tool(
                name=TEAM_SPAWN_REVIEWED_AGENT_TOOL,
                description=(
                    "创建一个带检查层的后台 teammate 协作单元。适合复杂、重要、需要审查或可能需要返工的子任务；"
                    "系统会创建 worker teammate 和 reviewer teammate，由 reviewer 审查 worker 结果，必要时要求 worker 修改，并同步 lead。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "worker teammate 名称，使用短英文或拼音，例如 frontend、builder。",
                        },
                        "role": {"type": "string", "description": "worker 的职责角色。"},
                        "prompt": {
                            "type": "string",
                            "description": "交给 worker 的清晰子任务说明，包含范围、交付物、限制条件和回报方式。",
                        },
                        "reviewer_name": {
                            "type": "string",
                            "description": "reviewer teammate 名称，可选；默认在 worker 名称后加 _reviewer。",
                        },
                        "reviewer_role": {
                            "type": "string",
                            "description": "reviewer 的职责角色，可选，例如 代码审查、UI 检查、验收检查。",
                        },
                        "review_prompt": {
                            "type": "string",
                            "description": "给 reviewer 的额外审查标准，可选；不填则按原子任务和 worker 结果审查。",
                        },
                        "max_revision_rounds": {
                            "type": "integer",
                            "description": "reviewer 要求返工时最多自动让 worker 修改几轮，默认 1，最大 2。",
                        },
                    },
                    "required": ["name", "role", "prompt"],
                },
                handler=team.spawn_reviewed_team_agent,
            ),
        )
    return tools


def _read_file(workspace: Workspace, data: dict[str, Any]) -> str:
    if "files" not in data:
        if not data.get("path"):
            return "缺少参数: path 或 files"
        return _read_file_one(workspace, data)

    items, error = _validated_batch_items(data.get("files"), "files")
    if error:
        return error
    results = _run_parallel(items, lambda item: _read_file_one(workspace, item))
    return _format_batch_results(
        f"批量读取完成: {len(items)} 个文件",
        items,
        results,
        lambda item: str(item.get("path") or "?"),
    )


def _read_file_one(workspace: Workspace, data: dict[str, Any]) -> str:
    path = workspace.resolve_inside(data["path"])
    if not path.exists():
        return f"文件不存在: {path.relative_to(workspace.root)}"
    if not path.is_file():
        return f"不是文件: {path.relative_to(workspace.root)}"
    max_chars = max(1, min(_as_int(data.get("max_chars"), 60000), 200000))
    text = path.read_text(encoding="utf-8", errors="replace")

    if data.get("start_line") is not None or data.get("end_line") is not None:
        return _read_file_lines(workspace, path, text, data, max_chars)

    offset = max(0, min(_as_int(data.get("offset"), 0), len(text)))
    end = min(len(text), offset + max_chars)
    chunk = text[offset:end]
    if offset == 0 and end == len(text):
        return chunk

    rel_path = path.relative_to(workspace.root)
    lines = [
        f"文件: {rel_path}",
        f"总字符数: {len(text)}",
        f"返回字符: {offset + 1}-{end}",
    ]
    if end < len(text):
        lines.append(
            f"提示: 文件未读完，可继续调用 read_file(path={rel_path}, offset={end}, max_chars={max_chars})。"
        )
    return "\n".join(lines) + "\n\n" + chunk + _truncated_suffix(end < len(text))


def _read_file_lines(
    workspace: Workspace,
    path,
    text: str,
    data: dict[str, Any],
    max_chars: int,
) -> str:
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return ""

    start_line = max(1, _as_int(data.get("start_line"), 1))
    end_line = _as_int(data.get("end_line"), total_lines)
    end_line = max(start_line, min(end_line, total_lines))
    if start_line > total_lines:
        return f"行号超出范围: {path.relative_to(workspace.root)} 只有 {total_lines} 行。"

    chunk = "".join(lines[start_line - 1 : end_line])
    truncated = len(chunk) > max_chars
    if truncated:
        chunk = chunk[:max_chars]

    rel_path = path.relative_to(workspace.root)
    header = [
        f"文件: {rel_path}",
        f"总行数: {total_lines}, 总字符数: {len(text)}",
        f"返回行: {start_line}-{end_line}",
    ]
    if truncated:
        header.append("提示: 当前行范围内容过长，请缩小 start_line/end_line 或提高 max_chars 后继续读取。")
    return "\n".join(header) + "\n\n" + chunk + _truncated_suffix(truncated)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncated_suffix(truncated: bool) -> str:
    return "\n\n... 文件内容已截断" if truncated else ""


def _task_create(tasks: TaskStore, data: dict[str, Any]) -> str:
    items = data.get("items") or []
    if not isinstance(items, list) or not items:
        return "没有创建任务，items 不能为空。"
    filtered_items, skipped_items = _filter_task_plan_items(items)
    if not filtered_items:
        return "没有创建任务：提交的任务都是纯最终验收/整体验证类任务；如需收尾检查，请按需要调用 final_check。"
    created = tasks.add_many(filtered_items, reset=bool(data.get("reset", False)))
    lines = ["已创建任务计划:"]
    for task in created:
        lines.append(f"- {task.id}: {task.title}")
    if skipped_items:
        lines.append("已跳过纯最终验收任务:")
        for item in skipped_items:
            lines.append(f"- {item.get('title') or '(untitled)'}")
    return "\n".join(lines)


def _final_check(tasks: TaskStore, data: dict[str, Any]) -> str:
    goal = str(data.get("goal") or "").strip() or "(未提供)"
    summary = str(data.get("summary") or "").strip() or "(未提供)"
    task_panel = tasks.render_panel() or "(no tasks)"
    validation_note = (
        "检测到任务面板里似乎已有验证/测试/检查类任务。"
        "如果这些任务已经真实完成，不需要重复检查；请直接基于证据总结。"
        if _has_validation_task(tasks)
        else "当前任务面板没有明显的验证/测试/检查类任务；请在最终回复前自行完成必要验收。"
    )
    return (
        "Final check guidance\n"
        f"- 用户目标: {goal}\n"
        f"- 当前判断: {summary}\n"
        f"- 任务面板:\n{task_panel}\n\n"
        f"{validation_note}\n\n"
        "请根据实际情况选择后续动作：\n"
        "- 如果没有足够证据证明完成，请继续调用 read_file/run_command/task_list 等工具检查 workspace。\n"
        "- 如果检查发现不符合用户目标，请继续修改文件或运行命令，并用 task_update 更新任务状态。\n"
        "- 如果已经符合用户目标，请给最终总结，说明改动、验证结果和剩余风险。\n"
        "- 不要只因为任务面板显示完成就直接结束；最终判断以 workspace 的实际状态为准。"
    )


def _has_validation_task(tasks: TaskStore) -> bool:
    markers = [
        "验证",
        "验收",
        "检查",
        "测试",
        "联调",
        "verify",
        "validation",
        "review",
        "check",
        "test",
    ]
    for task in tasks.list():
        text = f"{task.title} {task.description}".lower()
        if any(marker in text for marker in markers):
            return True
    return False


def _filter_task_plan_items(items: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if _is_pure_final_validation_task(item):
            skipped.append(item)
        else:
            kept.append(item)
    return kept, skipped


def _is_pure_final_validation_task(item: dict[str, Any]) -> bool:
    text = f"{item.get('title') or ''} {item.get('description') or ''}".lower()
    final_markers = [
        "最终验收",
        "整体验收",
        "整体检验",
        "整体验证",
        "验收测试",
        "联调验收",
        "最终检查",
        "最终验证",
        "收尾检查",
        "final validation",
        "final review",
        "acceptance test",
    ]
    implementation_markers = [
        "编写",
        "实现",
        "添加",
        "创建",
        "修复",
        "生成",
        "测试用例",
        "单元测试",
        "集成测试",
        "测试文件",
        "test file",
        "unit test",
        "integration test",
    ]
    return any(marker in text for marker in final_markers) and not any(
        marker in text for marker in implementation_markers
    )


def _task_get(tasks: TaskStore, data: dict[str, Any]) -> str:
    return tasks.render_detail(str(data["task_id"])) or "(no task)"


def _task_update(tasks: TaskStore, data: dict[str, Any]) -> str:
    task_id = str(data.get("task_id") or data.get("taskId") or "")
    if not task_id:
        return "缺少参数: task_id"
    blocked_by = _blocked_by_from_tool_input(data) if "blocked_by" in data or "blockedBy" in data else None
    task = tasks.update(
        task_id=task_id,
        title=_optional_str(data, "title", "subject"),
        description=_optional_str(data, "description"),
        active_form=_optional_str(data, "active_form", "activeForm"),
        status=_optional_str(data, "status"),
        blocked_by=blocked_by,
        note=str(data.get("note") or ""),
    )
    return f"已更新任务: {task.id}"


def _optional_str(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in data:
            return str(data.get(key) or "")
    return None


def _blocked_by_from_tool_input(data: dict[str, Any]) -> list[str]:
    raw = data.get("blocked_by", data.get("blockedBy", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("blocked_by must be a list of task IDs.")
    return [str(item) for item in raw]


def _write_file(workspace: Workspace, data: dict[str, Any]) -> str:
    raw_path = data.get("path") or data.get("file_path") or ""
    invalid_path = _invalid_workspace_file_path_reason(raw_path)
    if invalid_path:
        return invalid_path
    if "content" not in data:
        return "缺少参数: content（没有要写入的内容）"
    path = workspace.resolve_inside(raw_path)
    content = str(data.get("content") or "")
    if content == "" and not bool(data.get("allow_empty", False)):
        return (
            "拒绝写入空文件: content 为空。"
            "请在 content 中提供完整文件内容；只有确实需要空文件时才设置 allow_empty=true。"
        )
    overwrite = bool(data.get("overwrite", False))
    if path.exists() and not overwrite:
        return f"文件已存在，未覆盖: {path.relative_to(workspace.root)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入: {path.relative_to(workspace.root)} ({len(content)} 字符)"


def _edit_file(workspace: Workspace, data: dict[str, Any]) -> str:
    if "edits" not in data:
        if not all(key in data for key in ("path", "old_str", "new_str")):
            return "缺少参数: path/old_str/new_str 或 edits"
        return _edit_file_one(workspace, data)

    items, error = _validated_batch_items(data.get("edits"), "edits")
    if error:
        return error
    results = _run_grouped_edits(workspace, items)
    return _format_batch_results(
        f"批量编辑完成: {len(items)} 项，涉及 {_unique_path_count(items)} 个文件",
        items,
        results,
        lambda item: str(item.get("path") or "?"),
    )


def _edit_file_one(workspace: Workspace, data: dict[str, Any]) -> str:
    if not all(key in data for key in ("path", "old_str", "new_str")):
        return "缺少参数: path/old_str/new_str"
    path = workspace.resolve_inside(data["path"])
    if not path.exists():
        return f"文件不存在: {path.relative_to(workspace.root)}"
    if not path.is_file():
        return f"不是文件: {path.relative_to(workspace.root)}"
    old_str = str(data["old_str"])
    new_str = str(data["new_str"])
    if not old_str:
        return "old_str 不能为空。"
    text = path.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        return f"未找到匹配文本，替换失败: {path.relative_to(workspace.root)}"
    if count > 1:
        return f"匹配文本出现 {count} 次，请提供更精确的上下文让 old_str 唯一。"
    text = text.replace(old_str, new_str, 1)
    path.write_text(text, encoding="utf-8")
    return f"已编辑: {path.relative_to(workspace.root)} (替换 {len(old_str)} -> {len(new_str)} 字符)"


def _run_command(workspace: Workspace, data: dict[str, Any]) -> str:
    if "commands" not in data:
        if not data.get("command"):
            return "缺少参数: command 或 commands"
        return _run_command_one(workspace, data)

    items, error = _validated_batch_items(data.get("commands"), "commands")
    if error:
        return error
    results = _run_parallel(items, lambda item: _run_command_one(workspace, item))
    return _format_batch_results(
        f"批量命令完成: {len(items)} 条",
        items,
        results,
        _command_text_from_input,
    )


def _run_command_one(workspace: Workspace, data: dict[str, Any]) -> str:
    if not data.get("command"):
        return "缺少参数: command"
    command = str(data["command"])
    args = [str(item) for item in data.get("args") or []]
    blocked = _blocked_command_reason(command, args)
    if blocked:
        return blocked
    timeout = max(1, min(_as_int(data.get("timeout"), 30), 120))
    process = subprocess.Popen(
        [command, *args],
        cwd=workspace.root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )

    stdout_reader = _start_pipe_reader(process.stdout)
    stderr_reader = _start_pipe_reader(process.stderr)
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        return_code = process.wait()

    stdout_bytes, stdout_truncated, stdout_error = _finish_pipe_reader(stdout_reader)
    stderr_bytes, stderr_truncated, stderr_error = _finish_pipe_reader(stderr_reader)

    output = [f"exit_code: {return_code}"]
    if timed_out:
        output.append(f"timeout: command exceeded {timeout}s and was stopped")

    stdout = _decode_process_output(stdout_bytes)
    stderr = _decode_process_output(stderr_bytes)
    if stdout:
        output.append("stdout:")
        output.append(_clip(stdout, truncated=stdout_truncated))
    if stdout_error:
        output.append(f"stdout_reader_error: {stdout_error}")
    if stderr:
        output.append("stderr:")
        output.append(_clip(stderr, truncated=stderr_truncated))
    if stderr_error:
        output.append(f"stderr_reader_error: {stderr_error}")
    return "\n".join(output)


def _validated_batch_items(value: Any, field_name: str) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list) or not value:
        return [], f"参数 {field_name} 必须是非空数组。"
    if not all(isinstance(item, dict) for item in value):
        return [], f"参数 {field_name} 中的每一项都必须是对象。"
    return [dict(item) for item in value], ""


def _run_parallel(items: list[dict[str, Any]], handler: Callable[[dict[str, Any]], str]) -> list[str]:
    workers = min(len(items), MAX_PARALLEL_WORKERS)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="l1m-tool") as executor:
        futures = [executor.submit(handler, item) for item in items]
        results: list[str] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(f"工具执行失败: {type(exc).__name__}: {exc}")
        return results


def _run_grouped_edits(workspace: Workspace, items: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(items):
        path_key = str(item.get("path") or f"(missing-{index})").replace("/", "\\").casefold()
        groups.setdefault(path_key, []).append((index, item))

    def edit_group(group: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, str]]:
        group_results: list[tuple[int, str]] = []
        for index, item in group:
            try:
                result = _edit_file_one(workspace, item)
            except Exception as exc:
                result = f"工具执行失败: {type(exc).__name__}: {exc}"
            group_results.append((index, result))
        return group_results

    grouped_results = _run_parallel_groups(list(groups.values()), edit_group)
    results = [""] * len(items)
    for group_result in grouped_results:
        for index, result in group_result:
            results[index] = result
    return results


def _run_parallel_groups(
    groups: list[list[tuple[int, dict[str, Any]]]],
    handler: Callable[[list[tuple[int, dict[str, Any]]]], list[tuple[int, str]]],
) -> list[list[tuple[int, str]]]:
    workers = min(len(groups), MAX_PARALLEL_WORKERS)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="l1m-edit") as executor:
        futures = [executor.submit(handler, group) for group in groups]
        return [future.result() for future in futures]


def _unique_path_count(items: list[dict[str, Any]]) -> int:
    return len({str(item.get("path") or "").replace("/", "\\").casefold() for item in items})


def _format_batch_results(
    header: str,
    items: list[dict[str, Any]],
    results: list[str],
    labeler: Callable[[dict[str, Any]], str],
) -> str:
    sections = [header]
    total = len(items)
    for index, (item, result) in enumerate(zip(items, results), start=1):
        sections.append(f"\n--- [{index}/{total}] {labeler(item)} ---\n{result}")
    return "\n".join(sections)


def _start_pipe_reader(pipe: Any) -> tuple[threading.Thread | None, list[bytes], dict[str, Any]]:
    chunks: list[bytes] = []
    state: dict[str, Any] = {"size": 0, "truncated": False, "error": ""}
    if pipe is None:
        return None, chunks, state

    thread = threading.Thread(
        target=_drain_process_pipe,
        args=(pipe, chunks, state),
        daemon=True,
    )
    thread.start()
    return thread, chunks, state


def _drain_process_pipe(pipe: Any, chunks: list[bytes], state: dict[str, Any]) -> None:
    try:
        while True:
            chunk = pipe.read(PROCESS_PIPE_CHUNK_SIZE)
            if not chunk:
                break
            remaining = PROCESS_OUTPUT_BYTE_LIMIT - int(state["size"])
            if remaining > 0:
                kept = chunk[:remaining]
                chunks.append(kept)
                state["size"] = int(state["size"]) + len(kept)
            if len(chunk) > remaining:
                state["truncated"] = True
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _finish_pipe_reader(
    reader: tuple[threading.Thread | None, list[bytes], dict[str, Any]],
) -> tuple[bytes, bool, str]:
    thread, chunks, state = reader
    if thread is not None:
        thread.join(timeout=1)
        if thread.is_alive():
            state["truncated"] = True
    return b"".join(chunks), bool(state["truncated"]), str(state["error"] or "")


def _blocked_command_reason(command: str, args: list[str]) -> str:
    text = " ".join([command, *args]).lower()
    banned_patterns = [
        r"\bdel\b",
        r"\berase\b",
        r"\brd\b",
        r"\brmdir\b",
        r"\brm\b",
        r"\bremove-item\b",
        r"\bmove\b",
        r"\bmv\b",
        r"\bmove-item\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\btaskkill\b",
        r"\breg\s+delete\b",
    ]
    for pattern in banned_patterns:
        if re.search(pattern, text):
            return "拒绝执行危险命令。请改用 read_file/write_file，或使用只读命令观察项目。"
    return ""


def _clip(text: str, limit: int = PROCESS_OUTPUT_TEXT_LIMIT, truncated: bool = False) -> str:
    if len(text) > limit:
        return text[:limit] + "\n\n... output truncated"
    if truncated:
        return text + "\n\n... output truncated"
    return text


def _decode_process_output(data: bytes) -> str:
    if not data:
        return ""
    encodings = ["utf-8", locale.getpreferredencoding(False), "gbk"]
    seen = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
