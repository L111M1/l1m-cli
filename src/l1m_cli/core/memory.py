from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryNote:
    content: str
    created_at: str


class MemoryStore:
    def __init__(self):
        self._notes: list[MemoryNote] = []
        self._summaries: list[MemoryNote] = []

    def add(self, content: str) -> MemoryNote:
        note = MemoryNote(content=content, created_at=_now())
        self._notes.append(note)
        return note

    def list_notes(self, limit: int | None = None) -> list[MemoryNote]:
        rows = list(self._notes)
        return rows[-limit:] if limit is not None else rows

    def list_summaries(self, limit: int | None = None) -> list[MemoryNote]:
        rows = list(self._summaries)
        return rows[-limit:] if limit is not None else rows

    def clear(self) -> None:
        self._notes.clear()
        self._summaries.clear()

    def render_for_prompt(self, limit: int = 20) -> str:
        notes = self.list_notes(limit=limit)
        summaries = self.list_summaries(limit=limit)
        if not notes and not summaries:
            return ""
        rows = [f"- 记忆: {note.content}" for note in notes]
        rows.extend(f"- 摘要: {summary.content}" for summary in summaries)
        return "\n".join(rows)

    def add_summary(self, content: str) -> None:
        self._summaries.append(MemoryNote(content=content, created_at=_now()))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
