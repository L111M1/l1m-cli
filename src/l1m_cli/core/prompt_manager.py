from __future__ import annotations

import json
from pathlib import Path

from l1m_cli.errors import PromptError
from l1m_cli.core.tools import is_task_tool, is_team_tool


SYSTEM_SECTION_FILES = {
    "identity": "system_identity",
    "basic_rules": "system_basic_rules",
    "tools": "system_tools",
    "task_system": "task_system",
    "agent_teams": "system_agent_teams",
    "agent_loop": "system_agent_loop",
    "workspace": "system_workspace",
    "memory": "system_memory",
    "tasks": "system_tasks",
}


class PromptManager:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = prompt_dir
        self._last_system_context_key: str | None = None
        self._last_system_prompt: str | None = None
        self.last_loaded_sections: list[str] = []

    def list_names(self) -> list[str]:
        if not self.prompt_dir.exists():
            return []
        return sorted(path.stem for path in self.prompt_dir.glob("*.md"))

    def load(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.md"
        if not path.exists():
            raise PromptError(f"Prompt not found: {name}")
        return path.read_text(encoding="utf-8")

    def render(self, template_name: str, **values: object) -> str:
        text = self.load(template_name)
        return self.render_text(text, **values)

    def render_text(self, text: str, **values: object) -> str:
        for key, value in values.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    def render_system(
        self,
        workspace: str,
        memory: str = "",
        tasks: str = "",
        enabled_tools: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        context = {
            "workspace": workspace,
            "memory": memory.strip(),
            "tasks": tasks.strip(),
            "enabled_tools": sorted(enabled_tools or []),
        }
        return self.get_system_prompt(context)

    def get_system_prompt(self, context: dict[str, object]) -> str:
        key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        if key == self._last_system_context_key and self._last_system_prompt is not None:
            return self._last_system_prompt

        prompt = self.assemble_system_prompt(context)
        self._last_system_context_key = key
        self._last_system_prompt = prompt
        return prompt

    def assemble_system_prompt(self, context: dict[str, object]) -> str:
        if not self._has_section("identity"):
            self.last_loaded_sections = ["system"]
            return self.render(
                "system",
                workspace=str(context.get("workspace") or ""),
                memory=str(context.get("memory") or "(none)"),
                tasks=str(context.get("tasks") or "(none)"),
            )

        sections: list[str] = []
        loaded: list[str] = []
        enabled_tools = [str(tool) for tool in context.get("enabled_tools") or []]
        memory = str(context.get("memory") or "").strip()
        tasks = str(context.get("tasks") or "").strip()

        self._append_section(sections, loaded, "identity")
        self._append_section(sections, loaded, "basic_rules")
        if enabled_tools:
            self._append_section(
                sections,
                loaded,
                "tools",
                tools="\n".join(f"- {tool}" for tool in enabled_tools),
            )
            if any(is_task_tool(tool) for tool in enabled_tools):
                self._append_section(sections, loaded, "task_system")
            if any(is_team_tool(tool) for tool in enabled_tools) and self._has_section("agent_teams"):
                self._append_section(sections, loaded, "agent_teams")
            self._append_section(sections, loaded, "agent_loop")
        self._append_section(
            sections,
            loaded,
            "workspace",
            workspace=str(context.get("workspace") or ""),
        )
        if memory:
            self._append_section(sections, loaded, "memory", memory=memory)
        if tasks:
            self._append_section(sections, loaded, "tasks", tasks=tasks)

        self.last_loaded_sections = loaded
        return "\n\n".join(section for section in sections if section.strip())

    def _append_section(
        self,
        sections: list[str],
        loaded: list[str],
        key: str,
        **values: object,
    ) -> None:
        name = SYSTEM_SECTION_FILES[key]
        sections.append(self.render(name, **values))
        loaded.append(key)

    def _has_section(self, key: str) -> bool:
        return (self.prompt_dir / f"{SYSTEM_SECTION_FILES[key]}.md").exists()

    def clear_system_cache(self) -> None:
        self._last_system_context_key = None
        self._last_system_prompt = None
        self.last_loaded_sections = []
