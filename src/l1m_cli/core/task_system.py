from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from l1m_cli.errors import TaskError


TASK_STATUS_ICONS = {
    "pending": "○",
    "in_progress": "●",
    "completed": "✓",
    "done": "✓",
}


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    status: str = "pending"
    active_form: str = ""
    blocked_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class TaskStore:
    allowed_statuses = {"pending", "in_progress", "completed"}
    status_aliases = {"done": "completed"}

    def __init__(self):
        self._tasks: list[Task] = []

    def list(self) -> list[Task]:
        return list(self._tasks)

    def save_all(self, tasks: list[Task]) -> None:
        self._tasks = list(tasks)

    def clear(self) -> None:
        self._tasks.clear()

    def clear_if_complete(self) -> bool:
        if not self._tasks or self.has_open_tasks():
            return False
        self.clear()
        return True

    def has_open_tasks(self) -> bool:
        return any(task.status in {"pending", "in_progress"} for task in self._tasks)

    def render_open_tasks(self) -> str:
        return self._render_tasks(
            [task for task in self._tasks if task.status in {"pending", "in_progress"}]
        )

    def add(
        self,
        title: str,
        description: str = "",
        status: str = "pending",
        blocked_by: list[str] | None = None,
        active_form: str = "",
    ) -> Task:
        status = self._normalize_status(status)
        tasks = self.list()
        task = Task(
            id=f"task_{len(tasks) + 1:03d}",
            title=title,
            description=description,
            status=status,
            active_form=active_form,
            blocked_by=list(blocked_by or []),
        )
        tasks.append(task)
        self.save_all(tasks)
        return task

    def add_many(self, items: list[dict[str, object]], reset: bool = False) -> list[Task]:
        tasks = [] if reset else self.list()
        created: list[Task] = []
        for item in items:
            title = str(item.get("title") or item.get("subject") or "").strip()
            if not title:
                raise TaskError("Task title is required.")
            status = self._normalize_status(str(item.get("status") or "pending"))
            blocked_by = _blocked_by_from_item(item)
            active_form = str(item.get("active_form") or item.get("activeForm") or "")
            task = Task(
                id=f"task_{len(tasks) + 1:03d}",
                title=title,
                description=str(item.get("description") or ""),
                status=status,
                active_form=active_form,
                blocked_by=blocked_by,
            )
            tasks.append(task)
            created.append(task)
        self.save_all(tasks)
        return created

    def get(self, task_id: str) -> Task:
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise TaskError(f"Task not found: {task_id}")

    def can_start(self, task_id: str) -> bool:
        return not self.blockers_for(task_id)

    def blockers_for(self, task_id: str) -> list[str]:
        task = self.get(task_id)
        blockers: list[str] = []
        for dep_id in task.blocked_by:
            try:
                dep = self.get(dep_id)
            except TaskError:
                blockers.append(dep_id)
                continue
            if dep.status != "completed":
                blockers.append(dep_id)
        return blockers

    def set_status(self, task_id: str, status: str, note: str = "") -> Task:
        return self.update(task_id, status=status, note=note)

    def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        status: str | None = None,
        blocked_by: list[str] | None = None,
        note: str = "",
    ) -> Task:
        tasks = self.list()
        for task in tasks:
            if task.id == task_id:
                if title is not None:
                    title = title.strip()
                    if not title:
                        raise TaskError("Task title cannot be empty.")
                    task.title = title
                if description is not None:
                    task.description = description
                if active_form is not None:
                    task.active_form = active_form
                if status is not None:
                    task.status = self._normalize_status(status)
                if blocked_by is not None:
                    task.blocked_by = list(blocked_by)
                if note:
                    task.notes.append(note)
                task.updated_at = _now()
                self.save_all(tasks)
                return task
        raise TaskError(f"Task not found: {task_id}")

    def render_for_prompt(self) -> str:
        return self.render_panel()

    def render_panel(self) -> str:
        return self._render_tasks(self._tasks, include_description=False, include_notes=False)

    def render_detail(self, task_id: str) -> str:
        return self._render_tasks(
            [self.get(task_id)],
            include_status=True,
            include_description=True,
            include_notes=True,
            include_dependencies=True,
            include_active_form=True,
        )

    def _render_tasks(
        self,
        tasks: list[Task],
        include_status: bool = False,
        include_description: bool = False,
        include_notes: bool = False,
        include_dependencies: bool = False,
        include_active_form: bool = False,
    ) -> str:
        if not tasks:
            return ""
        rows: list[str] = []
        for task in tasks:
            icon = TASK_STATUS_ICONS.get(task.status, "?")
            status = f" [{task.status}]" if include_status else ""
            detail = f" - {task.description}" if include_description and task.description else ""
            deps = f" blocked_by={task.blocked_by}" if include_dependencies and task.blocked_by else ""
            rows.append(f"- {icon} {task.id}: {task.title}{status}{deps}{detail}")
            if include_active_form and task.active_form:
                rows.append(f"  active: {task.active_form}")
            if include_notes and task.notes:
                for note in task.notes:
                    rows.append(f"  note: {note}")
        return "\n".join(rows)

    @classmethod
    def _normalize_status(cls, status: str) -> str:
        normalized = cls.status_aliases.get(status, status)
        if normalized not in cls.allowed_statuses:
            raise TaskError(f"Invalid task status: {status}")
        return normalized


def _blocked_by_from_item(item: dict[str, object]) -> list[str]:
    raw = item.get("blocked_by", item.get("blockedBy", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TaskError("blocked_by must be a list of task IDs.")
    return [str(dep) for dep in raw]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
