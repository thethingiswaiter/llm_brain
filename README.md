# llm_brain

基于 LangGraph 的通用 Agent 原型项目，当前主线聚焦于“特征提取 -> 任务规划 -> 子任务执行 -> 反思推进”的执行闭环，并逐步接入记忆、技能与后续 MCP 扩展能力。

## 文档入口

- 项目总览：./docs/project_overview.md
- 完备化路线图：./docs/completeness_roadmap.md
- 能力概念说明：./docs/agent.md
- 项目定位与方向：./docs/project_positioning.md

如果你是第一次接手该仓库，建议按以下顺序阅读：

1. 先看 `docs/project_overview.md`，快速建立整体认知。
2. 再看 `app/agent/core.py`、`app/agent/runtime.py`、`app/agent/snapshots.py`、`app/agent/observability.py`、`app/agent/tools_runtime.py`、`app/cli/main.py`、`app/cli/commands.py`、`core/llm/manager.py`、`core/llm/logging.py`、`core/llm/runtime.py`、`core/llm/factory.py`，理解真实调用链。
3. 然后按子系统查看 `cognitive/`、`memory/`、`tools/`、`skills/`。
4. 最后再看 `docs/completeness_roadmap.md` 和 `docs/agent.md`，分别理解后续方向和能力概念。

目录分层（标准版）说明：

- `app/`：应用编排层（Agent 主链、CLI 入口）
- `core/`：基础设施层（配置、LLM 运行时与日志）
- `tools/`：Python 工具定义
- `skills/`：Markdown 技能定义

兼容入口说明：仓库根目录仍保留同名文件作为轻量转发入口，方便已有脚本和测试继续运行；新开发请优先使用 `app/` 与 `core/` 下路径。

说明：运行时自动加载的 Python 工具只从 `tools/` 目录读取；`examples/skills/` 和 `tests/fixtures/skills/` 中的示例/测试技能不会参与自动加载。

## 当前项目重点

- 主入口：`app/cli/main.py`
- 主编排器：`app/agent/core.py`
- 核心执行框架：LangGraph 状态图
- 认知子系统：`cognitive/`
- 记忆持久化：`memory/`
- Python 工具系统：`tools/`
- Markdown 技能系统：`skills/`

## 当前状态

项目仍处于原型阶段，但已经不只是最小骨架。当前已经打通 CLI 到 Agent 的完整主链，并具备任务拆解、反思推进、记忆落盘与召回、统一技能路由、请求级日志追踪、运行时快照恢复，以及配置式 MCP 工具和真实 MCP stdio server 接入能力。当前测试套件维持在 131 个 unittest 用例通过。

## 运行说明

1. 根据 `config.json` 准备模型配置，默认走本地 Ollama。
2. 安装项目依赖。
3. 运行 `python app/cli/main.py` 启动命令行交互。

补充说明：

- 当前依赖需要 Python 3.10+。
- 如果本机当前 Python 环境里已有旧版 LangChain 相关依赖，建议执行一次 `python -m pip install -U -r requirements.txt`，避免 `langchain-core`、`langchain-ollama` 等包版本漂移导致导入错误。
- 当前仓库已完成一次验证，结果为 `Ran 131 tests ... OK`。
- 现在内置 `tools/terminal_command.py`，会自动加载一个 `bash` 工具（安全白名单、工作区路径约束、危险命令拦截、输出截断、超时上限）。
- 现在还内置 `tools/langchain_common_tools.py`：自动加载常用工具 `get_current_time`、`calculator`、`list_directory`、`read_text_file`、`grep_text`、`write_text_file`、`json_query`，统一采用 LangChain `@tool` 风格。

### 测试运行建议

- 日常开发快速回归（默认，不含慢集成）：`python tests/run_tests.py`
- 全量回归（含 CLI + MCP transport 慢集成）：`python tests/run_tests.py --all`
- 保留原始 discover 方式：`python -m unittest discover -s tests -p "test_*.py"`

### CLI 命令
- `/load_tool <tool_name.py>` 动态加载本地 Python 工具文件。
- `/load_skill <skill_name.md>` 动态加载本地 Markdown 技能文件。
- `/load_mcp <config_name.json|config_name.yaml|absolute_path|server.py|stdio:command ...>` 加载 MCP 工具源：既支持原有 JSON/YAML 配置，也支持真实 MCP stdio server。
- `/list_mcp` 查看已加载的 MCP server。
- `/refresh_mcp <server_name|source>` 刷新某个 MCP server 并重新枚举工具。
- `/unload_mcp <server_name|source>` 卸载某个 MCP server，并移除其工具。
- `/cancel_request <request_id>` 对活跃请求发起协作式取消。
- `/list_snapshots <request_id>` 查看某次请求的可用恢复点。
- `/resume_snapshot <request_id> [latest|index|stage|snapshot_file] [reroute]` 从运行时快照恢复；带 `reroute` 时会丢弃旧 plan，基于恢复点上下文重新规划更安全的路径。
- `/request_summary <request_id>` 聚合查看该次请求的状态、快照、关键 checkpoint 和关联 memory。
- `/recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]` 查看最近若干次请求的状态、关键指标和 attention 原因摘要；当 failure details 可解析时会直接显示 tool、reason、error_type 或 action。
- `/failed_requests [limit] [status=failed,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m]` 作为失败导向的 recent 查询别名，默认聚焦失败、阻断、超时和取消请求。
- `/resumed_requests [limit] [status=failed,blocked,...] [attention] [since=30m]` 作为恢复请求的 recent 查询别名，默认聚焦 resumed 请求。
- `/request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]` 查看最近若干次请求的全局聚合统计，并支持筛选。
- `/failure_memories [limit] [keywords...]` 单独查看失败案例记忆，优先返回 blocked、retry、timeout、ask_user 等 failure_case 记录。
- `/retention_status` 查看 logs、snapshots、audit logs 和 memory backups 的 retention 覆盖范围、可回收体积、最近一次自动 prune 摘要，以及最近一次自动 prune 跳过原因。
- `/prune_runtime_data [apply]` 先 dry-run 预览、再按 retention 配置清理过期 logs、snapshots、audit logs 和 memory backups。

当前 request_rollup 还会补两类排障摘要：Top failures 会优先显示最常见的 failure stage、tool、reason、error_type 和 action；Failure combos 会补最常见的 stage+tool 与 tool+reason 组合热点。

当前这些失败聚合还新增了稳定的 source bucket：request_summary、recent_requests 和 request_rollup 现在不仅能看见 failure stage，还会把失败来源归一化成 agent_invoke、agent_resume、model、tool_runtime、reflection 等稳定分类；rollup 里也会汇总 stage+source 组合热点，并额外输出 source bucket 分布、占比摘要和 recent-vs-earlier 趋势，便于区分入口失败、模型失败和工具链路失败。

当前记忆层已开始把失败步骤单独沉淀为 failure_case 视图，便于把 blocked、ask_user、retry 和 timeout 经验与成功经验分开检索。

当前 reroute 也已开始使用同一请求内的跨子任务失败历史：当某个工具在前序步骤里重复失败到阈值后，后续子任务会优先避开它；在多个备选工具都可用时，也会优先选择历史失败更少的候选。

当前恢复执行还新增了显式 reroute 模式：对 blocked 或中途中断的快照，可以不沿用旧 plan，而是把恢复点上下文、失败工具摘要和近期反思重新交给 planner，改走新的更安全路径。

当前这层历史失败信号又进一步细化为结构化严重度：timeout、dependency_unavailable、invalid_arguments 等类型会累计成 severity 分数，既会影响 reroute 备选工具排序，也会直接影响当前子任务建议工具的优先级。

当某个工具的历史 failure severity 超过阈值时，系统现在会直接把它从当前候选和 reroute 备选里排除；如果剩余工具都不安全，则会明确降级到无工具路径，而不是继续在高风险工具之间反复切换。

当前 no-tool 降级提示也更明确了：对 invalid_arguments 这类缺参型失败，会明确引导 ask-user；对高风险历史或无安全工具可用的场景，则优先引导在现有上下文内给出安全的 direct answer，只有在确实缺外部信息或副作用执行能力时才追问用户。

这些策略信号现在也进入了 request_summary 和 recent_requests：除了 failure stage 外，还会直接暴露最近一次 reroute mode，便于快速识别 fallback_high_risk_history 这类 no-tool 降级是否发生。

观测层现在还会把 request_failed 快照 extra 里的 error_type / error 回填进 triage 和 request_rollup；像 dependency-unavailable 这类入口级失败，即使没有单独的 tool checkpoint details，也能出现在 recent_requests 的 attention 摘要和 rollup 的 failure signals 里。

当前运行时对模型依赖异常也补了更稳的入口分流：即使测试或运行过程中发生模块 reload，`LLMDependencyUnavailableError` 的类身份发生漂移，invoke 和 resume 入口仍会把这类错误稳定识别为 dependency-unavailable，并返回明确的环境/依赖检查提示，而不是退化成通用的 `Error invoking agent`。

当前 runtime retention 基线已覆盖 logs、state snapshots、system MCP audit logs 和 memory backups，默认同时支持 retention_days、max_files 或 max_request_dirs，以及 max_bytes 三类上限控制，并提供 dry-run 清理入口。
另外已补一层低频自动 prune：快照落盘和大输入备份写入后会按 retention_auto_prune_min_interval_seconds 做节流清理，避免每次写入都重复扫描。

### AGENT

当前主链已收口到分层目录：`app/agent/core.py` 负责编排，`app/agent/runtime.py`、`app/agent/snapshots.py`、`app/agent/observability.py`、`app/agent/tools_runtime.py` 分别承接请求生命周期、快照、观测与工具运行时；CLI 入口与命令分发位于 `app/cli/main.py`、`app/cli/commands.py`；LLM 基础设施集中在 `core/llm/`。

## Python MCP Server

仓库新增了一个独立可运行的 Python MCP server：`mcp_servers/system_mcp_server.py`。

它提供三个工具：

- `execute_terminal_command`：执行终端命令，默认限制在工作区内，并默认拦截明显破坏性命令。
- `get_system_info`：查看主机、平台、Python 版本、磁盘空间等系统信息。
- `inspect_file_system_path`：查看文件或目录信息，并可选返回 UTF-8 文本预览。

另外，agent 运行时也提供了无需额外 MCP 加载的内置 Python 工具：

- `bash`（来自 `tools/terminal_command.py`）：复用同一套安全策略执行终端命令，便于在主工作流中直接使用 opencode 风格的命令执行能力。

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
