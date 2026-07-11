# L1m CLI

一个学习用的 Python CLI Agent demo。

当前重点：
- 使用 Anthropic 官方 Python SDK 调用模型。
- 默认模型名预设为 `deepseekv4-pro`。
- 默认从 L1m 项目自己的 `.env` 读取配置；当前目录或上级目录的 `.env` 可以覆盖它。
- 在任意目录运行 `l1m` 时，默认把当前目录作为 workspace。
- 默认进入 L1m TUI Agent；界面层独立放在 `src/l1m_cli/tui/`。
- 任务、记忆和 Agent 上下文都只存在于当前进程中，退出后清空。

## 安装

```bash
python -m pip install -e .
```

如果缺少开发依赖：

```bash
python -m pip install -e ".[dev]"
```

## 配置

先从示例文件创建本地配置：

```powershell
Copy-Item .env.example .env
```

然后编辑项目根目录的 `.env`：

```env
ANTHROPIC_API_KEY=填你的 key
L1M_MODEL=deepseekv4-pro
L1M_BASE_URL=
L1M_MAX_TOKENS=8192
L1M_THINKING_ENABLED=true
L1M_THINKING_BUDGET_TOKENS=1024
L1M_WORKSPACE=
L1M_CONTEXT_COMPACT_THRESHOLD=850000
L1M_DEBUG=false
```

`.env` 只用于本机运行，已被 `.gitignore` 排除，不能提交到 Git。仓库中只保留不含真实密钥和真实服务地址的 `.env.example`。API Key 和自定义 Base URL 均只从 `.env` 或进程环境变量读取，源码中不保存这些值。

`L1M_WORKSPACE` 留空表示使用运行 `l1m` 时的当前目录。需要手动指定时，可以填绝对路径，或填相对当前目录的路径。

如果你接入的是 Anthropic 协议兼容网关，并且需要自定义地址，再填写：

```env
L1M_BASE_URL=https://your-compatible-endpoint.example
```

## 常用命令

```bash
l1m
l1m --help
l1m config
l1m prompt list
l1m prompt show system
l1m agent
l1m agent --show-steps
l1m tui          # 显式别名，效果等同交互式 Agent TUI
l1m tui --show-steps
l1m agent "查看当前目录并说明这个项目是什么"
l1m agent "创建一个 hello.txt 文件" --show-steps
```

说明：
- `l1m` 会直接进入 L1m TUI Agent。
- `l1m agent` 不带目标时同样进入 L1m TUI Agent；带目标时执行一次性 Agent Loop。
- `l1m tui` 保留为显式别名，方便确认自己启动的是 TUI。
- `l1m agent "..."` 是单次 Agent Loop，会让模型根据目标决定是否读取文件、写文件或运行命令。
- Agent 模式会把任务系统作为工具暴露给模型，模型可以创建任务计划、更新状态，并在终端显示 `thinking...`、行动预告、工具调用和任务进度。
- TUI 会在模型调用 `write_file` 或 `edit_file` 时展示文件变更预览，方便你确认它实际改了什么。
- TUI 会对模型最终回复做轻量 Markdown 渲染，包括标题、列表、引用、代码块、粗体、行内代码和链接。
- 任务状态使用图标展示：`○ pending`、`● in_progress`、`✓ done`。
- 模型准备结束前，Agent 会要求它基于当前 workspace 重新验收项目完成情况；如果不符合目标，需要继续更新任务并调用工具完成。
- 目前没有外部 `l1m memory` 和 `l1m task` 命令；任务和记忆都只服务当前进程。

## Agent Loop 停止规则

循环规则很简单：

1. 调用模型。
2. 如果响应里包含 `tool_use` block，先检查真实工具调用前是否已经调用 `announce_action`；未预告的工具不会执行，会把纠错结果追加回上下文。
3. 对通过检查的工具，执行并把工具结果追加回上下文。
4. 如果没有工具调用，且需要最终验收，要求模型重新检查 workspace 完成情况。
5. 如果 `stop_reason == "max_tokens"`，自动请求模型从中断处继续，并合并最终文本。
6. 如果没有工具调用、没有截断、也不需要继续验收，输出最后一次模型文本。

当前没有人为步数上限；停止主要由模型是否继续调用工具和最终验收结果决定。

## System Prompt 管理

System prompt 现在按 section 组合，而不是只依赖一个大文件：

- `prompts/system_identity.md`：身份设定
- `prompts/system_basic_rules.md`：基础规则
- `prompts/system_tools.md`：工具使用规则，只有启用工具时注入
- `prompts/task_system.md`：任务状态规则，只有启用 task 工具时注入
- `prompts/system_agent_loop.md`：Agent Loop 行为规则，只有启用工具时注入
- `prompts/system_workspace.md`：当前 workspace
- `prompts/system_memory.md`：有记忆内容时注入
- `prompts/system_tasks.md`：有任务面板时注入

`PromptManager` 会根据真实运行状态装配 system prompt，并用稳定的 context key 做进程内缓存。
