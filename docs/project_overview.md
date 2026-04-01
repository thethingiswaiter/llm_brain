# llm_brain 项目总览

## 1. 文档定位

这份文档只做一件事：基于当前源码，总结 llm_brain 现在已经实现了什么、主执行链如何运转、主要模块分别负责什么，以及哪些地方仍然只是基础版。

阅读原则：

- 以源码现状为准。
- README 用于快速入口，路线图用于后续开发，不替代当前实现说明。

## 2. 一句话定义

llm_brain 是一个基于 LangGraph 的命令行 Agent 原型，主循环围绕“特征提取 -> 任务规划 -> 子任务执行 -> 工具调用 -> 反思推进”展开，已经接入记忆、技能路由、请求级观测、运行时快照恢复、工具运行时治理，以及基础版真实 MCP stdio transport。

## 3. 当前项目画像

### 3.1 项目定位

- 项目类型：命令行交互式 Agent 原型
- 主入口：cli.py 中的 start_cli
- 主协调器：agent_core.py 中的 AgentCore
- 状态编排：LangGraph StateGraph
- 模型提供方：Ollama / OpenAI
- 记忆存储：SQLite
- 工具来源：Python 技能 + MCP 工具
- Prompt 技能来源：Markdown Front-matter 技能
- MCP 现状：配置式本地工具 + 真实 stdio MCP server 双模接入
- 当前测试状态：131 个 unittest 用例通过

### 3.2 当前主线重点

当前项目已经不再只是最小可运行骨架，最近一轮增强主要集中在稳定性和可观测性：

- request 级快照恢复与显式 reroute 恢复
- 跨子任务失败累计后的工具降级、排除和替代排序
- 高严重度历史失败触发 no-tool 降级
- request_summary、recent_requests、request_rollup 三层请求观测
- failure source 归一化为稳定 source bucket，并输出分布、占比和 recent-vs-earlier 趋势
- detached tool run 跟踪与回收
- runtime retention 覆盖日志、快照、审计和 memory backup
- 模型依赖缺失在 invoke / resume 入口的稳定识别

## 4. 主执行链

### 4.1 主流程

1. CLI 接收普通文本输入或管理命令。
2. 普通文本输入进入 AgentCore.invoke(query, session_id)。
3. 每次调用都会生成新的 request_id，并进入 llm_manager.request_scope()。
4. planner 节点提取全局关键词、判断领域、召回摘要级记忆、写入会话主记忆并拆解子任务。
5. agent 节点执行当前子任务，统一选择 Prompt 技能与候选工具，并结合失败历史做 reroute 或 no-tool 降级判断。
6. 如果模型返回 tool_calls，则进入 ToolNode 执行工具，再回到 agent。
7. reflect_and_advance 节点根据 expected_outcome 与实际结果决定 continue、retry 或 ask_user。
8. retry 首次失败时，系统可触发一次针对当前失败子任务的更小粒度重规划。
9. 成功时返回最终结果；阻断、超时、取消或依赖故障时返回明确的可诊断消息，并同步写入快照和日志。

### 4.2 LangGraph 节点

AgentCore._build_graph() 当前主节点如下：

| 节点名 | 内部函数 | 作用 |
| --- | --- | --- |
| planner | initial_planning | 全局特征提取、领域判断、记忆召回、会话主记忆初始化、任务拆解 |
| agent | call_model_subtask | 执行当前子任务、选择技能与工具、调用模型 |
| reflect_and_advance | reflect_and_advance | 判断成功、重试、重规划或 ask_user |
| tools | ToolNode(self.tools) | 执行模型发起的工具调用 |

主链路可以概括为：

START -> planner -> agent -> tools 或 reflect_and_advance -> agent 或 END

## 5. 当前核心状态

AgentState 当前字段如下：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| messages | list[BaseMessage] | LangGraph 消息历史 |
| plan | List[Dict[str, Any]] | 当前子任务计划 |
| current_subtask_index | int | 当前执行到的子任务索引 |
| reflections | List[str] | 每轮反思文本 |
| global_keywords | List[str] | 全局任务关键词 |
| failed_tools | Dict[str, List[str]] | 按子任务记录已失败工具名，供 reroute 过滤 |
| failed_tool_signals | Dict[str, Dict[str, Dict[str, Any]]] | 按子任务记录失败工具的结构化信号与严重度 |
| request_id | str | 单次调用唯一追踪 ID |
| session_id | str | 会话级唯一标识 |
| session_memory_id | int | 当前会话主记录在记忆库中的 ID |
| domain_label | str | 当前任务的领域标签 |
| memory_summaries | List[Dict[str, Any]] | 摘要级记忆召回结果 |
| retry_counts | Dict[str, int] | 每个子任务的当前重试次数 |
| replan_counts | Dict[str, int] | 每个子任务已触发的重规划次数 |
| blocked | bool | 当前流程是否阻断 |
| final_response | str | 最终返回文本 |

## 6. 当前已经具备的能力

### 6.1 执行与恢复

- 基于 LangGraph 的可运行状态图执行骨架
- 基于 LLM 的特征提取、领域判断、任务拆解与反思判断
- request_id 级运行时状态快照持久化
- 从最新、索引、阶段名或显式快照文件恢复执行
- 对 blocked 或中途中断快照支持显式 reroute 恢复模式
- 恢复前语义一致性校验，拒绝明显不一致的快照
- retry 路径上的一次性失败子任务重规划
- LLM、工具、整次请求三级超时控制
- 按 request_id 的协作式主动取消
- 模型依赖不可用在 invoke / resume 入口的稳定识别，不再退化成笼统错误

### 6.2 工具与 reroute

- 工具异常统一分类、结构化失败返回与参数预校验
- 工具运行生命周期跟踪、detached 标记与后台完成回收
- 工具失败后的自动排除、替代工具搜索与无工具降级
- 跨子任务失败累计进入 reroute 策略
- 历史失败严重度进入当前工具优先级和 reroute 候选排序
- 超过 severity 阈值的工具会被直接排除，必要时降级到 no-tool 路径
- no-tool 提示已区分 ask-user 补参与 direct-answer 安全收敛

### 6.3 记忆与技能

- 会话级与步骤级记忆写入
- 摘要级记忆召回与按需详情加载
- failure_case 失败案例记忆视图
- replay 历史记忆重放
- memory -> Markdown skill 转换
- Python 技能自动扫描加载
- 自动跳过 sample_*.py 与 test_*.py 这类示例/测试技能文件
- Markdown Prompt 技能自动扫描、Front-matter 解析和统一匹配
- Prompt 技能、Python 工具、MCP 工具统一能力注册和路由

### 6.4 MCP 与运行时治理

- 配置式 MCP 工具注册
- 真实 MCP stdio server 的加载、列举工具、远端调用、连接复用、自动重连、刷新与卸载
- 仓库内置 Python MCP server，支持终端命令、系统信息和文件信息查看
- runtime retention 已覆盖 logs、snapshots、audit logs 和 memory backups
- retention 支持 retention_days、max_files 或 max_request_dirs、max_bytes 三类约束
- retention_status 与 prune_runtime_data 支持 dry-run / apply
- 自动 prune 具备低频节流与执行/跳过原因记录

### 6.5 观测与排障

- request_id 与 session_id 双层追踪
- 文件日志中的结构化 JSON 事件与控制台阶段日志
- request_summary 单请求聚合视图
- recent_requests 最近请求摘要视图
- request_rollup 跨请求聚合视图
- request_failed 快照 extra 中的 error_type / error 会回填到 triage 与 rollup
- 最新失败来源会归一化为稳定 source bucket，例如 agent_invoke、agent_resume、model、tool_runtime、reflection、retention
- rollup 会输出 top failure signals、failure combinations、source bucket 分布、占比和 recent-vs-earlier 趋势
- detached tool 计数、明细与运行态查询视图
- checkpoint 级阶段耗时分布聚合

## 7. 当前仍属于基础版的部分

- MCP 真实 transport 目前只覆盖 stdio 基础版，尚未补资源同步、订阅刷新和更细粒度健康检查
- 超时与取消目前仍以逻辑取消为主，不是底层线程、进程或网络调用的硬终止
- 技能与工具路由仍以关键词和阈值匹配为主，没有混合检索或更强语义检索
- 记忆治理已有质量标签和精确去重，但还没有归档、衰减、相似合并和长期压缩策略
- 可观测性已经具备请求聚合与跨请求 rollup，但还不是完整的全局指标系统

## 8. 项目结构与职责

| 路径 | 角色 |
| --- | --- |
| cli.py | CLI 入口与主循环 |
| cli_commands.py | CLI 命令元数据、帮助文本与 handler 注册 |
| agent_core.py | Agent 门面、图构建、子系统编排 |
| agent_runtime.py | 请求生命周期、invoke / resume 收口、取消与超时协调 |
| agent_snapshots.py | 快照序列化、落盘、恢复与校验 |
| agent_observability.py | request_summary、recent_requests、request_rollup 聚合 |
| agent_tools.py | 工具包装、参数预校验、失败解析、reroute 计划与运行态跟踪 |
| agent_retention.py | 运行时产物 retention 统计、清理和自动 prune |
| llm_manager.py | LLM 对外门面 |
| llm_logging.py | 结构化日志、控制台输出、request_id 事件反查 |
| llm_runtime.py | request_scope、超时等待与取消检查 |
| llm_factory.py | provider 对应 LLM 实例构造 |
| cognitive/ | 特征提取、规划、反思与结构化输出解析 |
| memory/ | SQLite 记忆管理与大输入备份 |
| skills/ | 真实运行时 Python 技能目录 |
| examples/skills/ | 示例 Python 技能，不参与自动加载 |
| skills_md/ | Markdown Prompt 技能解析与统一能力路由 |
| mcp_servers/mcp_manager.py | MCP 双模接入层 |
| mcp_servers/system_mcp_server.py | 内置独立 Python MCP server |
| logs/ | LLM 日志目录 |
| runtime_state/ | 快照、审计与运行时状态目录 |
| tests/ | 回归测试目录 |
| docs/ | 文档目录 |

## 9. 对外接口

### 9.1 CLI 主要命令

| 命令 | 作用 |
| --- | --- |
| /llm <provider> <model> [base_url] [api_key] | 切换当前模型配置 |
| /load_skill <skill_name.py\|skill_name.md> | 增量加载 Python 或 Markdown 技能 |
| /load_mcp <config\|server.py\|stdio:command ...> | 加载配置式 MCP 工具或真实 stdio MCP server |
| /list_mcp | 查看已加载 MCP server 与 transport |
| /refresh_mcp <server_name\|source> | 刷新 MCP server 并重新枚举工具 |
| /unload_mcp <server_name\|source> | 卸载 MCP server 并移除其工具 |
| /cancel_request <request_id> | 对活跃请求发起协作式取消 |
| /list_snapshots <request_id> | 列出可用恢复点 |
| /resume_snapshot <request_id> [latest\|index\|stage\|snapshot_file] [reroute] | 从快照恢复；可选 reroute 重规划 |
| /request_summary <request_id> | 查看单请求摘要、指标、checkpoint、快照和记忆 |
| /recent_requests [limit] [status=...] [resumed] [attention] [since=30m] | 查看最近请求摘要与 attention 原因 |
| /failed_requests [limit] [status=...] [resumed] [attention] [since=30m] | 失败导向的 recent 查询别名 |
| /resumed_requests [limit] [status=...] [attention] [since=30m] | 恢复请求导向的 recent 查询别名 |
| /request_rollup [limit] [status=...] [resumed] [attention] [since=30m] | 查看最近请求的聚合统计、failure 热点和 source 摘要 |
| /list_tool_runs [request_id] [running\|detached] | 查看运行中或 detached 的工具运行态 |
| /failure_memories [limit] [keywords...] | 查看 failure_case 记忆视图 |
| /retention_status | 查看 retention 状态、可回收体积、最近自动 prune 摘要 |
| /prune_runtime_data [apply] | 对 retention 范围内的运行时产物做 dry-run 或实际清理 |
| /replay <memory_id> [injected features...] | 重放历史记忆并重新执行 |
| /convert_skill <memory_id> | 将历史记忆转换成 Markdown 技能 |
| /new_session | 创建新的会话 ID |
| exit / quit | 退出 CLI |

### 9.2 AgentCore 主要方法

| 方法 | 作用 |
| --- | --- |
| invoke(query, session_id=None) | 执行一次完整 Agent 调用 |
| replay(memory_id, injected_features=None) | 重放历史记忆并重新执行 |
| resume_from_snapshot(request_id, snapshot_name=None, reroute=False) | 从指定快照恢复继续执行 |
| list_snapshots(request_id) | 列出指定请求的可用快照 |
| get_request_summary(request_id) | 聚合单次请求的快照、checkpoint、memory 与指标 |
| get_recent_request_summaries(limit=10, statuses=None, resumed_only=False, attention_only=False, since_seconds=None) | 聚合最近请求摘要视图 |
| get_request_rollup(limit=20, statuses=None, resumed_only=False, attention_only=False, since_seconds=None) | 聚合最近请求的全局状态、计数、阶段耗时与 failure 热点 |
| list_tool_runs(request_id="", status="") | 查看 tracked tool run 运行态 |
| get_failure_memories(match_keywords=None, limit=5, exclude_conv_id=None, exclude_ids=None) | 聚合失败案例记忆视图 |
| get_retention_status() | 汇总运行时产物的 retention 状态和最近自动 prune 信息 |
| prune_runtime_data(apply=False) | 预览或执行运行时产物清理 |
| load_skill(skill_name) | 手动加载 Python 或 Markdown 技能 |
| load_mcp_server(server_ref) | 加载配置式 MCP 工具或真实 stdio MCP server |
| list_mcp_servers() | 列出当前已加载的 MCP server |
| refresh_mcp_server(server_ref) | 刷新某个 MCP server 并重新枚举工具 |
| unload_mcp_server(server_ref) | 卸载某个 MCP server 并移除其工具 |
| add_tool(tool) | 运行时动态添加工具 |
| start_session(session_id=None) | 新建或切换 session_id |
| get_last_request_id() | 获取最近一次调用生成的 request_id |

## 10. 观测、日志与恢复细节

### 10.1 追踪标识

- session_id：标识同一会话
- request_id：标识一次具体 Agent 调用

一次 invoke 会生成新的 request_id，并把整个执行包在 llm_manager.request_scope() 中。因此同一次请求的 LLM 调用、checkpoint、快照、工具运行态和记忆记录，都可以回溯到同一个 request_id。

### 10.2 请求级聚合

当前主要聚合入口有三层：

- get_request_summary(request_id)：看单次请求的状态、进度、checkpoint、快照、记忆、triage 和指标
- get_recent_request_summaries(...)：看最近请求的摘要列表，可按状态、是否恢复、是否需要 attention、时间窗口筛选
- get_request_rollup(...)：看最近请求的跨请求聚合，包括状态分布、总量指标、failure 热点和 source bucket 摘要

request_rollup 当前除基础指标外，还额外聚合：

- top_failure_signals：stages、tools、reasons、error_types、actions、sources、reroute_modes、no_tool_fallbacks
- top_failure_combinations：stage_tool、tool_reason、stage_source、stage_reroute
- source_bucket_breakdown：失败来源分布与占比
- source_bucket_trends：recent-vs-earlier 的 bucket 趋势变化

### 10.3 快照恢复

主执行链会把关键阶段状态写入 runtime_state/snapshots。当前支持：

- 按 request_id 恢复最新快照
- 按索引、阶段名或显式文件名选择恢复点
- 对已完成或已阻断的终态快照直接返回终态结果
- 对 blocked 或中途中断的恢复点显式走 reroute 模式
- 恢复前做语义一致性校验，避免在不一致状态上继续执行

## 11. 测试现状

当前 tests/ 主要覆盖：

- 技能与工具路由阈值
- 结构化输出、快照、超时、取消、reroute 与观测性路径
- 记忆质量、失败案例与重复合并
- 工具参数预校验与运行期输入拦截
- CLI 命令与主链基础集成
- 真实 MCP stdio transport、连接复用、自动重连、刷新与卸载
- system MCP server 的安全控制与审计行为

当前最新验证结果为 131 个 unittest 用例通过。

## 12. 当前实现判断

### 12.1 已经比较稳定的部分

- CLI -> AgentCore -> LangGraph 的主链可运行
- request_id / session_id 双层追踪
- 请求级快照、恢复、reroute 恢复与基础校验
- 工具异常安全包装、历史失败降级与 no-tool fallback
- request_summary、recent_requests、request_rollup 三层观测
- runtime retention 基线与自动 prune
- Python 技能、Markdown Prompt 技能、配置式 MCP 工具和真实 MCP 工具统一路由

### 12.2 仍未闭环的部分

- 更强的底层硬中止能力
- 更细粒度的节点级恢复
- 更强的语义检索与能力路由
- 更长期的记忆治理策略
- 更完整的全局指标和 MCP 工程化能力

## 13. 建议阅读顺序

1. 先读本文件，建立整体认知。
2. 再读 agent_core.py，理解主状态和主执行链。
3. 接着读 agent_runtime.py、agent_snapshots.py、agent_observability.py、agent_tools.py、agent_retention.py，理解辅助职责拆分。
4. 然后读 cli.py、cli_commands.py、llm_manager.py、llm_logging.py、llm_runtime.py、llm_factory.py，理解入口、命令分发、日志和 provider 封装。
5. 再读 cognitive/、memory/、skills_md/、mcp_servers/。
6. 最后再对照 README.md、docs/agent.md、docs/completeness_roadmap.md，区分当前事实、概念说明和后续路线图。

## 14. 结论

llm_brain 当前仍是原型工程，但已经具备了可追踪的执行主链、基础记忆闭环、统一技能路由、请求级与跨请求观测、快照恢复、工具运行时治理，以及基础版真实 MCP stdio 接入。后续接手时，重点不该再放在“能不能跑起来”，而应放在恢复粒度、底层中止、路由质量、长期记忆治理和更完整的协议工程化上。