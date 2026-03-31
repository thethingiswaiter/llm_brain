# llm_brain

基于 LangGraph 的通用 Agent 原型项目，当前主线聚焦于“特征提取 -> 任务规划 -> 子任务执行 -> 反思推进”的执行闭环，并逐步接入记忆、技能与后续 MCP 扩展能力。

## 文档入口

- 项目总览：./docs/project_overview.md
- 完备化路线图：./docs/completeness_roadmap.md
- 能力概念说明：./docs/agent.md

如果你是第一次接手该仓库，建议按以下顺序阅读：

1. 先看 `docs/project_overview.md`，快速建立整体认知。
2. 再看 `agent_core.py`、`agent_runtime.py`、`agent_snapshots.py`、`agent_observability.py`、`agent_tools.py`、`cli.py`、`cli_commands.py`、`llm_manager.py`、`llm_logging.py`、`llm_runtime.py`、`llm_factory.py`，理解真实调用链。
3. 然后按子系统查看 `cognitive/`、`memory/`、`skills_md/`。
4. 最后再看 `docs/completeness_roadmap.md` 和 `docs/agent.md`，分别理解后续方向和能力概念。

说明：运行时自动加载的 Python 技能只从 `skills/` 目录读取；`examples/skills/` 和 `tests/fixtures/skills/` 中的示例/测试技能不会参与自动加载。

## 当前项目重点

- 主入口：`cli.py`
- 主编排器：`agent_core.py`
- 核心执行框架：LangGraph 状态图
- 认知子系统：`cognitive/`
- 记忆持久化：`memory/`
- Python 工具技能：`skills/`
- Markdown 技能系统：`skills_md/`

## 当前状态

项目仍处于原型阶段，但已经不只是最小骨架。当前已经打通 CLI 到 Agent 的完整主链，并具备任务拆解、反思推进、记忆落盘与召回、统一技能路由、请求级日志追踪、运行时快照恢复，以及配置式 MCP 工具和真实 MCP stdio server 接入能力。当前测试套件维持在 70 个 unittest 用例通过。

## 运行说明

1. 根据 `config.json` 准备模型配置，默认走本地 Ollama。
2. 安装项目依赖。
3. 运行 `python cli.py` 启动命令行交互。

### CLI 命令
- `/load_skill <skill_name.py|skill_name.md>` 动态加载本地技能文件。
- `/load_mcp <config_name.json|config_name.yaml|absolute_path|server.py|stdio:command ...>` 加载 MCP 工具源：既支持原有 JSON/YAML 配置，也支持真实 MCP stdio server。
- `/list_mcp` 查看已加载的 MCP server。
- `/refresh_mcp <server_name|source>` 刷新某个 MCP server 并重新枚举工具。
- `/unload_mcp <server_name|source>` 卸载某个 MCP server，并移除其工具。
- `/cancel_request <request_id>` 对活跃请求发起协作式取消。
- `/list_snapshots <request_id>` 查看某次请求的可用恢复点。
- `/resume_snapshot <request_id> [latest|index|stage|snapshot_file]` 从运行时快照恢复继续执行。
- `/request_summary <request_id>` 聚合查看该次请求的状态、快照、关键 checkpoint 和关联 memory。
- `/recent_requests [limit]` 查看最近若干次请求的状态和关键指标摘要。

### AGENT

当前 `agent_core.py` 已将请求生命周期、快照、请求观测与工具运行时包装分别委托给 `agent_runtime.py`、`agent_snapshots.py`、`agent_observability.py` 和 `agent_tools.py`；`cli.py` 的命令定义已外提到 `cli_commands.py`；`llm_manager.py` 也已将日志、运行时和 provider 构造分别委托给 helper 模块，主入口文件的职责边界比之前清晰。

## Python MCP Server

仓库新增了一个独立可运行的 Python MCP server：`mcp_servers/system_mcp_server.py`。

它提供三个工具：

- `execute_terminal_command`：执行终端命令，默认限制在工作区内，并默认拦截明显破坏性命令。
- `get_system_info`：查看主机、平台、Python 版本、磁盘空间等系统信息。
- `inspect_file_system_path`：查看文件或目录信息，并可选返回 UTF-8 文本预览。

当前还增加了一层基础安全控制：

- 命令白名单：默认只允许 `echo`、`python`、`dir`、`ls`、`type`、`cat`、`pwd`、`whoami`、`hostname`、`systeminfo` 等前缀。
- 路径白名单：默认只允许访问工作区根目录；可通过环境变量 `LLM_BRAIN_MCP_ALLOWED_ROOTS` 追加额外根目录。
- 审计日志：每次工具调用都会写入 `runtime_state/audit/system_mcp_audit.jsonl`。

如果需要调整命令白名单，可通过环境变量 `LLM_BRAIN_MCP_ALLOWED_COMMANDS` 传入逗号分隔前缀列表。

运行方式：

1. 安装 `requirements.txt` 中的新依赖 `mcp`。
2. 运行 `python mcp_servers/system_mcp_server.py`。

现在 agent 侧已经补了基础版真实 MCP stdio transport：

- `/load_mcp mcp_servers/system_mcp_server.py` 可直接把本仓库里的 Python MCP server 作为真实 server 接入。
- `/load_mcp stdio:python mcp_servers/system_mcp_server.py` 也可通过显式命令行方式接入。
- `/load_mcp some_config.json` 仍然兼容原有配置式 MCP 工具定义。
- 真实 stdio server 现在会在加载后复用连接，而不是每次工具调用都重新拉起进程。
- `/list_mcp` 和 `/unload_mcp` 可用于查看和释放已加载的 MCP server 生命周期。
- 连接失活后，下次工具调用会自动重建 stdio 连接；也可以通过 `/refresh_mcp` 主动重建并刷新工具列表。

说明：当前实现的是基础版可复用 stdio transport，已经具备连接复用、自动重连和显式刷新/卸载，但仍未补工具刷新订阅和更细粒度的连接健康检查。

## 补充说明

- `docs/project_overview.md` 以“当前源码事实”为主。
- `docs/completeness_roadmap.md` 只保留仍未完成的后续工程事项。
- `docs/agent.md` 主要提供能力概念说明，不应直接视为现状功能清单。
