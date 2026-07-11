from __future__ import annotations

from l1m_cli.core.hooks import ContinuationContext, ContinuationRequest
from l1m_cli.core.task_system import TaskStore
from l1m_cli.core.tools import is_action_tool, is_task_tool


def make_completion_review_provider(tasks: TaskStore):
    def provide(context: ContinuationContext) -> ContinuationRequest | None:
        if context.review_requested:
            return None
        if not needs_final_review(context, tasks):
            return None

        return _request_final_review(context, tasks)

    return provide


def needs_final_review(context: ContinuationContext, tasks: TaskStore) -> bool:
    return any(_is_workspace_work(tool_name) for step in context.steps for tool_name in step.tool_calls)


def _is_workspace_work(tool_name: str) -> bool:
    return bool(tool_name) and not is_task_tool(tool_name) and not is_action_tool(tool_name)


def _request_final_review(context: ContinuationContext, tasks: TaskStore) -> ContinuationRequest:
    task_panel = tasks.render_panel() or "(no tasks)"
    return ContinuationRequest(
        event="final_review_required",
        mark_review_requested=True,
        payload={"kind": "final_review"},
        message=(
            "这不是最终回复。现在请先重新验收整个项目的实际完成情况，而不是只看任务面板。\n"
            "你必须至少调用一次工具来检查当前 workspace，例如 task_list、read_file 或 run_command。\n"
            "请对照用户原始目标、当前文件内容和运行结果判断是否真的完成。\n"
            "未完成任务不会单独触发强制续跑；任务面板只是进度仪表盘，最终判断以实际项目状态为准。\n"
            "如果完成情况不如预期，请更新任务面板，然后继续调用 write_file/edit_file/run_command 等工具完成任务。\n"
            "如果确实已经完成，请按需要用 task_update 修正任务状态，然后再给最终总结。\n\n"
            f"用户原始目标:\n{context.goal}\n\n"
            f"当前任务面板:\n{task_panel}\n\n"
            f"你刚才准备输出的最终回复:\n{context.final_text}"
        ),
    )
