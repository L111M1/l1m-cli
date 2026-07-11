from __future__ import annotations

from pathlib import Path

from l1m_cli.errors import WorkspaceError


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_inside(self, user_path: str | Path) -> Path:
        raw = Path(user_path)
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {user_path}") from exc
        return resolved
