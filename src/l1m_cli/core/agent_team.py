from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from l1m_cli.anthropic_client import AnthropicModelClient
from l1m_cli.config import Settings
from l1m_cli.core.agent_loop import AgentLoop
from l1m_cli.core.context_manager import AgentContextRegistry
from l1m_cli.core.hooks import build_default_hooks
from l1m_cli.core.prompt_manager import PromptManager
from l1m_cli.core.tools import LEAD_AGENT_NAME, build_workspace_tools
from l1m_cli.core.workspace import Workspace


TeamEventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class TeamMessage:
    from_agent: str
    to_agent: str
    content: str
    msg_type: str = "message"
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "unread"
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class MessageBus:
    def __init__(self):
        self._mailboxes: dict[str, list[TeamMessage]] = {}
        self._lock = threading.Lock()

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        request_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TeamMessage:
        message = TeamMessage(
            from_agent=_agent_name(from_agent),
            to_agent=_agent_name(to_agent),
            content=str(content).strip(),
            msg_type=str(msg_type or "message").strip() or "message",
            request_id=str(request_id or "").strip(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._mailboxes.setdefault(message.to_agent, []).append(message)
        return message

    def read_inbox(
        self,
        agent: str,
        statuses: tuple[str, ...] = ("unread",),
        mark_status: str = "delivered",
    ) -> list[TeamMessage]:
        agent_name = _agent_name(agent)
        with self._lock:
            messages = [
                message
                for message in self._mailboxes.get(agent_name, [])
                if message.status in statuses
            ]
            for message in messages:
                message.status = mark_status
        return list(messages)

    def clear(self) -> None:
        with self._lock:
            self._mailboxes.clear()


class AgentTeam:
    def __init__(
        self,
        settings: Settings,
        workspace: Workspace,
        prompts: PromptManager,
        thread_starter: Callable[[Callable[[], None]], None] | None = None,
        event_handler: TeamEventHandler | None = None,
    ):
        self.settings = settings
        self.workspace = workspace
        self.prompts = prompts
        self.bus = MessageBus()
        self._active: dict[str, str] = {}
        self._pending_requests: dict[str, ProtocolState] = {}
        self._lock = threading.Lock()
        self._thread_starter = thread_starter or _start_daemon_thread
        self._event_handler = event_handler
        self.contexts = AgentContextRegistry(
            compact_threshold=getattr(settings, "compact_threshold", None)
        )

    def set_event_handler(self, event_handler: TeamEventHandler | None) -> None:
        self._event_handler = event_handler

    def clear(self) -> None:
        self.bus.clear()
        self.contexts.clear()
        with self._lock:
            self._pending_requests.clear()

    def consume_inbox_for_history(self, agent: str = LEAD_AGENT_NAME) -> str:
        messages = self.bus.read_inbox(agent)
        self._route_protocol_messages(agent, messages)
        return _format_messages(messages)

    def normalize_agent_name(self, value: object) -> str:
        return _agent_name(value)

    def protocol_status(self, request_id: str) -> str:
        with self._lock:
            state = self._pending_requests.get(request_id)
            return state.status if state else ""

    def spawn_team_agent(self, data: dict[str, Any]) -> str:
        return self._spawn_agent(data)

    def spawn_reviewed_team_agent(self, data: dict[str, Any]) -> str:
        return self._spawn_reviewed_agent(data)

    def send_team_message(self, sender: str, data: dict[str, Any]) -> str:
        return self._send_message(sender, data)

    def check_team_inbox(self, agent: str) -> str:
        return self._check_inbox(agent)

    def wait_for_team_inbox(self, agent: str, data: dict[str, Any]) -> str:
        return self._wait_for_inbox(agent, data)

    def request_team_plan(self, sender: str, data: dict[str, Any]) -> str:
        return self._request_plan(sender, data)

    def submit_team_plan(self, sender: str, data: dict[str, Any]) -> str:
        return self._submit_plan(sender, data)

    def review_team_plan(self, sender: str, data: dict[str, Any]) -> str:
        return self._review_plan(sender, data)

    def _spawn_agent(self, data: dict[str, Any]) -> str:
        name = _agent_name(data.get("name") or "")
        role = str(data.get("role") or "").strip() or "teammate"
        prompt = str(data.get("prompt") or "").strip()
        if not name or name == LEAD_AGENT_NAME:
            return "创建 teammate 失败：name 必须是非 lead 的短名称。"
        if not prompt:
            return "创建 teammate 失败：prompt 不能为空。"

        with self._lock:
            if name in self._active:
                return f"teammate 已存在且仍在运行: {name}"
            self._active[name] = role

        def run() -> None:
            try:
                self._run_teammate(name=name, role=role, prompt=prompt)
            except Exception as exc:
                self.bus.send(name, LEAD_AGENT_NAME, f"子 agent 异常退出: {type(exc).__name__}: {exc}", "error")
            finally:
                with self._lock:
                    self._active.pop(name, None)

        self._thread_starter(run)
        return f"已创建 teammate: {name} ({role})"

    def _spawn_reviewed_agent(self, data: dict[str, Any]) -> str:
        name = _agent_name(data.get("name") or "")
        role = str(data.get("role") or "").strip() or "worker"
        prompt = str(data.get("prompt") or "").strip()
        reviewer_name = _reviewer_name(name, data.get("reviewer_name"))
        reviewer_role = str(data.get("reviewer_role") or "").strip() or "reviewer"
        review_prompt = str(data.get("review_prompt") or "").strip()
        max_revision_rounds = _clamp_int(data.get("max_revision_rounds"), default=1, minimum=0, maximum=2)

        if not name or name == LEAD_AGENT_NAME:
            return "创建 reviewed teammate 失败：worker name 必须是非 lead 的短名称。"
        if not reviewer_name or reviewer_name == LEAD_AGENT_NAME:
            return "创建 reviewed teammate 失败：reviewer name 必须是非 lead 的短名称。"
        if name == reviewer_name:
            return "创建 reviewed teammate 失败：worker 和 reviewer 不能同名。"
        if not prompt:
            return "创建 reviewed teammate 失败：prompt 不能为空。"

        with self._lock:
            for agent_name in (name, reviewer_name):
                if agent_name in self._active:
                    return f"teammate 已存在且仍在运行: {agent_name}"
            self._active[name] = role
            self._active[reviewer_name] = reviewer_role

        def run() -> None:
            try:
                self._run_reviewed_teammate(
                    name=name,
                    role=role,
                    prompt=prompt,
                    reviewer_name=reviewer_name,
                    reviewer_role=reviewer_role,
                    review_prompt=review_prompt,
                    max_revision_rounds=max_revision_rounds,
                )
            except Exception as exc:
                self.bus.send(
                    reviewer_name,
                    LEAD_AGENT_NAME,
                    f"reviewed teammate 异常退出: {type(exc).__name__}: {exc}",
                    "error",
                )
            finally:
                with self._lock:
                    self._active.pop(name, None)
                    self._active.pop(reviewer_name, None)

        self._thread_starter(run)
        return f"已创建 reviewed teammate: worker={name} reviewer={reviewer_name}"

    def _send_message(self, sender: str, data: dict[str, Any]) -> str:
        to = _agent_name(data.get("to") or LEAD_AGENT_NAME)
        content = str(data.get("content") or "").strip()
        if not content:
            return "消息未发送：content 不能为空。"
        request_id = str(data.get("request_id") or "").strip()
        self.bus.send(sender, to, content, request_id=request_id)
        return f"已发送给 {to}"

    def _check_inbox(self, agent: str) -> str:
        messages = self.bus.read_inbox(agent)
        self._route_protocol_messages(agent, messages)
        if not messages:
            if agent == LEAD_AGENT_NAME:
                active = self._active_agents_text()
                return f"(inbox empty){active}"
            return "(inbox empty)"
        return _format_messages(messages)

    def _wait_for_inbox(self, agent: str, data: dict[str, Any]) -> str:
        agent_name = _agent_name(agent)
        if agent_name != LEAD_AGENT_NAME:
            return "只有 lead 可以等待团队 inbox。"

        timeout_seconds = _clamp_float(
            data.get("timeout_seconds"),
            default=300.0,
            minimum=0.1,
            maximum=1800.0,
        )
        poll_interval_seconds = _clamp_float(
            data.get("poll_interval_seconds"),
            default=1.0,
            minimum=0.01,
            maximum=5.0,
        )
        deadline = time.monotonic() + timeout_seconds

        while True:
            messages = self.bus.read_inbox(agent_name)
            self._route_protocol_messages(agent_name, messages)
            if messages:
                return _format_messages(messages)

            if not self._has_active_agents():
                return "(inbox empty; no active teammates)"

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "(wait timeout; inbox empty)"
            time.sleep(min(poll_interval_seconds, remaining))

    def _request_plan(self, sender: str, data: dict[str, Any]) -> str:
        sender_name = _agent_name(sender)
        if sender_name != LEAD_AGENT_NAME:
            return "只有 lead 可以请求 teammate 提交计划。"
        target = _agent_name(data.get("teammate") or data.get("to") or "")
        task = str(data.get("task") or "").strip()
        if not target or target == LEAD_AGENT_NAME:
            return "请求计划失败：teammate 必须是非 lead 的名称。"
        if not task:
            return "请求计划失败：task 不能为空。"
        request_id = _new_request_id()
        with self._lock:
            self._pending_requests[request_id] = ProtocolState(
                request_id=request_id,
                type="plan",
                sender=LEAD_AGENT_NAME,
                target=target,
                status="waiting_plan",
                payload=task,
            )
        self.bus.send(
            LEAD_AGENT_NAME,
            target,
            f"请先提交执行计划，等待 lead 审批后再动手。\n\n任务：\n{task}",
            msg_type="plan_request",
            request_id=request_id,
            metadata={"request_id": request_id, "task": task},
        )
        return f"已请求 {target} 提交计划: {request_id}"

    def _submit_plan(self, sender: str, data: dict[str, Any]) -> str:
        sender_name = _agent_name(sender)
        request_id = str(data.get("request_id") or "").strip()
        plan = str(data.get("plan") or "").strip()
        if not request_id:
            return "提交计划失败：request_id 不能为空。"
        if not plan:
            return "提交计划失败：plan 不能为空。"

        with self._lock:
            state = self._pending_requests.get(request_id)
            if state is None:
                return f"提交计划失败：未知 request_id {request_id}"
            if state.target != sender_name:
                return f"提交计划失败：{request_id} 属于 {state.target}，不是 {sender_name}"
            if state.status not in {"waiting_plan", "rejected"}:
                return f"提交计划失败：{request_id} 当前状态是 {state.status}"
            state.status = "waiting_review"
            state.payload = plan

        self.bus.send(
            sender_name,
            LEAD_AGENT_NAME,
            plan,
            msg_type="plan_response",
            request_id=request_id,
            metadata={"request_id": request_id},
        )
        return f"计划已提交给 lead，等待审批: {request_id}"

    def _review_plan(self, sender: str, data: dict[str, Any]) -> str:
        sender_name = _agent_name(sender)
        if sender_name != LEAD_AGENT_NAME:
            return "只有 lead 可以审批计划。"
        request_id = str(data.get("request_id") or "").strip()
        approve = bool(data.get("approve"))
        feedback = str(data.get("feedback") or "").strip()
        if not request_id:
            return "审批计划失败：request_id 不能为空。"

        with self._lock:
            state = self._pending_requests.get(request_id)
            if state is None:
                return f"审批计划失败：未知 request_id {request_id}"
            if state.status != "waiting_review":
                return f"审批计划失败：{request_id} 当前状态是 {state.status}"
            state.status = "approved" if approve else "rejected"
            target = state.target

        content = feedback or ("计划已通过，可以开始执行。" if approve else "计划未通过，请根据反馈修改后重新提交。")
        self.bus.send(
            LEAD_AGENT_NAME,
            target,
            content,
            msg_type="plan_review",
            request_id=request_id,
            metadata={"request_id": request_id, "approve": approve},
        )
        return f"计划已{'通过' if approve else '拒绝'}: {request_id}"

    def _route_protocol_messages(self, agent: str, messages: list[TeamMessage]) -> None:
        agent_name = _agent_name(agent)
        if agent_name != LEAD_AGENT_NAME:
            return
        with self._lock:
            for message in messages:
                if message.msg_type != "plan_response" or not message.request_id:
                    continue
                state = self._pending_requests.get(message.request_id)
                if state is not None and state.status == "waiting_plan":
                    state.status = "waiting_review"
                    state.payload = message.content

    def _run_teammate(self, name: str, role: str, prompt: str) -> None:
        final_text = self._run_agent_once(
            name=name,
            role=role,
            goal=prompt,
            task=prompt,
            template_name="sub_agent",
            include_team_tools=True,
        )
        self.bus.send(name, LEAD_AGENT_NAME, final_text, "result")

    def _run_reviewed_teammate(
        self,
        name: str,
        role: str,
        prompt: str,
        reviewer_name: str,
        reviewer_role: str,
        review_prompt: str,
        max_revision_rounds: int,
    ) -> None:
        preflight_task = _preflight_review_task(
            worker_name=name,
            worker_role=role,
            worker_task=prompt,
            review_prompt=review_prompt,
        )
        preflight_review = self._run_agent_once(
            name=reviewer_name,
            role=reviewer_role,
            goal=preflight_task,
            task=preflight_task,
            template_name="sub_agent_reviewer",
            include_team_tools=False,
            worker_name=name,
            worker_role=role,
            worker_task=prompt,
            worker_result="(worker 尚未执行；请先审查原始子需求是否清晰、可执行、可验收。)",
            review_prompt=review_prompt or "(无额外审查要求)",
        )
        preflight_decision = _review_decision(preflight_review)
        if preflight_decision == "needs_lead":
            self.bus.send(
                reviewer_name,
                LEAD_AGENT_NAME,
                _preflight_result_summary(name, reviewer_name, preflight_decision, preflight_review),
                "result",
            )
            return

        worker_prompt = prompt
        if preflight_decision == "revise":
            self.bus.send(
                reviewer_name,
                LEAD_AGENT_NAME,
                _preflight_result_summary(name, reviewer_name, preflight_decision, preflight_review),
                "message",
            )
            worker_prompt = f"{prompt}\n\n检查 agent 对子需求的预审补充:\n{preflight_review}"

        worker_task = _reviewed_worker_task(worker_prompt, reviewer_name)
        worker_result = self._run_agent_once(
            name=name,
            role=role,
            goal=worker_task,
            task=worker_task,
            template_name="sub_agent",
            include_team_tools=False,
        )

        review_text = ""
        decision = "needs_lead"
        for round_index in range(max_revision_rounds + 1):
            review_task = _review_task(
                worker_name=name,
                worker_role=role,
                worker_task=worker_prompt,
                worker_result=worker_result,
                review_prompt=review_prompt,
                round_index=round_index,
                max_revision_rounds=max_revision_rounds,
            )
            review_text = self._run_agent_once(
                name=reviewer_name,
                role=reviewer_role,
                goal=review_task,
                task=review_task,
                template_name="sub_agent_reviewer",
                include_team_tools=False,
                worker_name=name,
                worker_role=role,
                worker_task=worker_prompt,
                worker_result=worker_result,
                review_prompt=review_prompt or "(无额外审查要求)",
            )
            decision = _review_decision(review_text)
            if decision == "pass":
                self.bus.send(
                    reviewer_name,
                    LEAD_AGENT_NAME,
                    _reviewed_result_summary(name, reviewer_name, decision, worker_result, review_text),
                    "result",
                )
                return

            self.bus.send(
                reviewer_name,
                LEAD_AGENT_NAME,
                _review_progress_summary(name, reviewer_name, decision, round_index, review_text),
                "message",
            )

            if decision == "revise" and round_index < max_revision_rounds:
                revision_task = _revision_task(worker_prompt, worker_result, review_text, reviewer_name)
                worker_result = self._run_agent_once(
                    name=name,
                    role=role,
                    goal=revision_task,
                    task=revision_task,
                    template_name="sub_agent",
                    include_team_tools=False,
                )
                continue

            break

        self.bus.send(
            reviewer_name,
            LEAD_AGENT_NAME,
            _reviewed_result_summary(name, reviewer_name, decision, worker_result, review_text),
            "result",
        )

    def _run_agent_once(
        self,
        name: str,
        role: str,
        goal: str,
        task: str,
        template_name: str,
        include_team_tools: bool,
        **extra_values: object,
    ) -> str:
        tools = build_workspace_tools(
            self.workspace,
            tasks=None,
            team=self if include_team_tools else None,
            team_sender=name,
            allow_team_spawn=False,
        )
        system_prompt = self.prompts.render(
            template_name,
            name=name,
            role=role,
            task=task,
            workspace=str(self.workspace.root),
            tools="\n".join(f"- {tool}" for tool in tools.names()),
            **extra_values,
        )
        context = self.contexts.for_agent(name)
        context.begin_token_tracking()
        client = AnthropicModelClient(self.settings)
        loop = AgentLoop(
            client=client,
            tools=tools,
            system_prompt=system_prompt,
            on_event=lambda event, payload: self._record_and_emit_agent_event(
                context,
                name,
                event,
                payload,
            ),
            hooks=build_default_hooks(),
            external_event_provider=lambda: self.consume_inbox_for_history(name),
            before_model_call=lambda messages, step: self._compact_agent_context(
                context,
                client,
                messages,
                step,
                name,
                system_prompt,
                tools.specs(),
            ),
        )
        bundle = context.begin_turn(goal)
        result = loop.run(bundle.user_text, history=bundle.messages)
        context.commit_result(result)
        return result.final_text.strip() or f"{name} 已结束，但没有生成总结。"

    def _record_and_emit_agent_event(
        self,
        context,
        name: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if event == "model_start":
            context.begin_model_call()
            payload["turn_total_tokens"] = context.turn_total_tokens
        if event == "model_output_token_delta":
            context.record_output_token_delta(payload.get("output_tokens_delta"))
            payload["context_tokens"] = context.current_context_tokens
            payload["turn_input_tokens"] = context.turn_input_tokens
            payload["turn_output_tokens"] = context.turn_output_tokens
            payload["turn_total_tokens"] = context.turn_total_tokens
        if event == "model_usage":
            context.record_usage(payload)
            payload["context_tokens"] = context.current_context_tokens
            payload["turn_input_tokens"] = context.turn_input_tokens
            payload["turn_output_tokens"] = context.turn_output_tokens
            payload["turn_total_tokens"] = context.turn_total_tokens
        self._emit_agent_event(name, event, payload)

    def _compact_agent_context(
        self,
        context,
        client: AnthropicModelClient,
        messages: list[dict[str, Any]],
        step: int,
        name: str,
        system_prompt: str,
        tool_specs: list[dict[str, Any]],
    ) -> None:
        context.prepare_request(system_prompt, messages, tool_specs)
        if not context.needs_compact(messages):
            return
        before_tokens = context.current_context_tokens
        self._emit_agent_event(
            name,
            "context_compact_start",
            {"step": step, "context_tokens": before_tokens, "threshold": context.compact_threshold},
        )
        context.replace_history(messages)
        context.compact(client, self.prompts)
        messages[:] = context.history
        context.prepare_request(system_prompt, messages, tool_specs)
        self._emit_agent_event(
            name,
            "context_compact_finish",
            {
                "step": step,
                "before_tokens": before_tokens,
                "after_tokens": context.current_context_tokens,
            },
        )

    def _emit_agent_event(self, name: str, event: str, payload: dict[str, Any]) -> None:
        if self._event_handler is None:
            return
        enriched = dict(payload)
        enriched.setdefault("agent", name)
        self._event_handler(event, enriched)

    def _active_agents_text(self) -> str:
        with self._lock:
            if not self._active:
                return ""
            active = ", ".join(f"{name}({role})" for name, role in sorted(self._active.items()))
        return f"\n运行中的 teammate: {active}"

    def _has_active_agents(self) -> bool:
        with self._lock:
            return bool(self._active)


def _format_messages(messages: list[TeamMessage]) -> str:
    if not messages:
        return ""
    lines: list[str] = []
    for message in messages:
        content = message.content if len(message.content) <= 2000 else message.content[:2000] + "\n... 消息已截断"
        request = f" req:{message.request_id}" if message.request_id else ""
        lines.append(
            f"- from {message.from_agent} [{message.msg_type}{request}] {message.created_at}\n"
            f"  {content.replace(chr(10), chr(10) + '  ')}"
        )
    return "\n".join(lines)


def _agent_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = text.strip("_-")
    return text[:40]


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:10]}"


def _reviewer_name(worker_name: str, value: object) -> str:
    if value:
        return _agent_name(value)
    base = f"{worker_name}_reviewer"
    reviewer_name = _agent_name(base)
    if reviewer_name == worker_name:
        reviewer_name = _agent_name(f"{worker_name[:28]}_reviewer")
    return reviewer_name


def _reviewed_worker_task(prompt: str, reviewer_name: str) -> str:
    return (
        f"{prompt}\n\n"
        "协作说明：你的结果会先交给检查 agent / reviewer 审查。"
        f" reviewer 名称是 {reviewer_name}。请在最终回复中交付你的发现、改动、验证和剩余风险，"
        "不要主动向 lead 发送结果。"
    )


def _preflight_review_task(
    worker_name: str,
    worker_role: str,
    worker_task: str,
    review_prompt: str,
) -> str:
    return (
        "请先审查 lead 准备交给 worker teammate 的子需求是否清晰、可执行、可验收。\n"
        f"worker: {worker_name} ({worker_role})\n\n"
        f"拟分配的子需求:\n{worker_task}\n\n"
        f"额外审查要求:\n{review_prompt or '(无)'}\n\n"
        "如果子需求清晰，请输出 DECISION: pass。"
        "如果只需要给 worker 增加约束或注意事项，请输出 DECISION: revise 并列出补充要求。"
        "如果缺少 lead 决策或关键信息，请输出 DECISION: needs_lead。"
    )


def _review_task(
    worker_name: str,
    worker_role: str,
    worker_task: str,
    worker_result: str,
    review_prompt: str,
    round_index: int,
    max_revision_rounds: int,
) -> str:
    return (
        f"请审查 worker teammate 的结果。\n"
        f"worker: {worker_name} ({worker_role})\n"
        f"审查轮次: {round_index + 1}/{max_revision_rounds + 1}\n\n"
        f"原始子需求:\n{worker_task}\n\n"
        f"worker 返回结果:\n{worker_result}\n\n"
        f"额外审查要求:\n{review_prompt or '(无)'}\n\n"
        "请按 reviewer system prompt 要求输出 DECISION。"
    )


def _revision_task(prompt: str, worker_result: str, review_text: str, reviewer_name: str) -> str:
    return (
        f"{prompt}\n\n"
        f"{reviewer_name} 已审查你的上一版结果并要求返工。\n\n"
        f"你的上一版结果:\n{worker_result}\n\n"
        f"审查意见:\n{review_text}\n\n"
        "请只处理审查意见指出的问题，完成后输出新的结果总结。"
    )


def _review_progress_summary(
    worker_name: str,
    reviewer_name: str,
    decision: str,
    round_index: int,
    review_text: str,
) -> str:
    return (
        f"{reviewer_name} 已审查 {worker_name} 的第 {round_index + 1} 版结果，"
        f"decision={decision}。\n\n{review_text}"
    )


def _preflight_result_summary(
    worker_name: str,
    reviewer_name: str,
    decision: str,
    review_text: str,
) -> str:
    return (
        f"{reviewer_name} 已预审 {worker_name} 的子需求，decision={decision}。\n\n"
        f"{review_text}"
    )


def _reviewed_result_summary(
    worker_name: str,
    reviewer_name: str,
    decision: str,
    worker_result: str,
    review_text: str,
) -> str:
    return (
        f"reviewed teammate 完成: worker={worker_name}, reviewer={reviewer_name}, decision={decision}\n\n"
        f"worker result:\n{worker_result}\n\n"
        f"reviewer report:\n{review_text}"
    )


def _review_decision(text: str) -> str:
    first_lines = "\n".join(text.strip().splitlines()[:5]).lower()
    if re.search(r"decision\s*:\s*pass\b", first_lines):
        return "pass"
    if re.search(r"decision\s*:\s*revise\b", first_lines):
        return "revise"
    if re.search(r"decision\s*:\s*needs[_ -]?lead\b", first_lines):
        return "needs_lead"
    if "通过" in first_lines or "pass" in first_lines:
        return "pass"
    if "返工" in first_lines or "修改" in first_lines or "revise" in first_lines:
        return "revise"
    return "needs_lead"


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _start_daemon_thread(target: Callable[[], None]) -> None:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
