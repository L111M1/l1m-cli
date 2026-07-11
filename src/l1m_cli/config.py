from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from l1m_cli.errors import ConfigError


@dataclass(frozen=True)
class Settings:
    cli_root: Path
    cwd: Path
    env_files: tuple[Path, ...]
    api_key: str
    model: str
    base_url: str | None
    max_tokens: int
    thinking_enabled: bool
    thinking_budget_tokens: int
    workspace_path: Path
    compact_threshold: int
    debug: bool

    @property
    def prompt_dir(self) -> Path:
        return self.cli_root / "prompts"

    @property
    def masked_api_key(self) -> str:
        if not self.api_key:
            return "<missing>"
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}****{self.api_key[-4:]}"


def find_cli_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "prompts").exists() and (path / "pyproject.toml").exists():
            return path
    return Path.cwd().resolve()


def find_nearest_env(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        env_file = path / ".env"
        if env_file.exists():
            return env_file
    return None


def _merged_env(env_files: list[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in env_files:
        if not env_file.exists():
            continue
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                values[key] = value
    for key, value in os.environ.items():
        values[key] = value
    return values


def _int_value(values: dict[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_value(values: dict[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path_from_value(raw: str, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_settings(cwd: Path | None = None, cli_root: Path | None = None) -> Settings:
    run_cwd = (cwd or Path.cwd()).resolve()
    root = find_cli_root(cli_root)
    cli_env = root / ".env"
    cwd_env = find_nearest_env(run_cwd)
    env_files = [cli_env]
    if cwd_env is not None and cwd_env.resolve() != cli_env.resolve():
        env_files.append(cwd_env)
    values = _merged_env(env_files)

    api_key = values.get("ANTHROPIC_API_KEY", "").strip()
    model = values.get("L1M_MODEL", "deepseek-v4-pro").strip()
    base_url = values.get("L1M_BASE_URL", "").strip() or None
    workspace_raw = values.get("L1M_WORKSPACE", "").strip()

    if not model:
        raise ConfigError("L1M_MODEL is empty.")

    workspace_path = _path_from_value(workspace_raw, run_cwd) if workspace_raw else run_cwd

    return Settings(
        cli_root=root,
        cwd=run_cwd,
        env_files=tuple(path for path in env_files if path.exists()),
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=_int_value(values, "L1M_MAX_TOKENS", 8192),
        thinking_enabled=_bool_value(values, "L1M_THINKING_ENABLED", True),
        thinking_budget_tokens=_int_value(values, "L1M_THINKING_BUDGET_TOKENS", 1024),
        workspace_path=workspace_path,
        compact_threshold=_int_value(values, "L1M_CONTEXT_COMPACT_THRESHOLD", 850000),
        debug=_bool_value(values, "L1M_DEBUG", False),
    )


def require_api_key(settings: Settings) -> None:
    if not settings.api_key:
        raise ConfigError("ANTHROPIC_API_KEY is empty. Fill it in .env first.")
