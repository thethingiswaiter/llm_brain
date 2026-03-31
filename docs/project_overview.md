# llm_brain 项目总览

## 1. 文档目的

本文档面向后续接手该仓库的人或大模型，目标是基于当前源码回答四个问题：

- 这个项目现在是什么。
- 主执行链如何运转。
- 主要模块分别负责什么。
- 哪些能力已经可用，哪些地方仍属于基础版。

阅读本文件时默认遵循一个原则：

- 以源码事实为准。
- README 和目标态文档用于辅助理解，但不能替代代码现状。

## 2. 一句话定义

llm_brain 是一个基于 LangGraph 的通用 Agent 原型，围绕“特征提取 -> 任务规划 -> 子任务执行 -> 工具调用 -> 反思推进”的主循环，已经接入记忆、技能、请求级日志追踪、运行时快照、真实 MCP stdio transport，以及一组经过多轮低风险拆分后的内部 helper 模块。

## 3. 当前项目画像

### 3.1 项目定位

- 项目类型：命令行交互式 Agent 原型
- 主入口：cli.py 中的 start_cli
- CLI 命令定义：cli_commands.py
- 主协调器：agent_core.py 中的 AgentCore
- 状态编排：LangGraph StateGraph
- 模型提供方：Ollama / OpenAI
- 记忆存储：SQLite
- 工具来源：Python 技能文件 + MCP 工具
- Prompt 技能来源：Markdown Front-matter 技能文件
- MCP 现状：同时支持配置式本地工具和真实 stdio MCP server
- 当前测试状态：125 个 unittest 用例通过

### 3.2 当前主流程

1. CLI 接收普通文本输入或管理命令。
2. 普通文本输入进入 AgentCore.invoke(query, session_id)。
3. 每次调用都会生成新的 request_id，并进入 llm_manager.request_scope()。
4. planner 节点提取全局关键词、判断领域、召回摘要级记忆、写入会话主记忆并拆解子任务。
5. agent 节点执行当前子任务，统一选择 Prompt 技能与候选工具，并在需要时参考最近失败记录做 reroute。
6. 如果模型返回 tool_calls，则进入 ToolNode 执行工具，再回到 agent。
7. reflect_and_advance 节点根据 expected_outcome 与实际输出决定 continue、retry 或 ask_user。
8. 如果 retry 路径第一次失败，系统可调用 planner 对失败子任务做一次更小粒度的重规划。
9. 成功完成后返回最后结果；若重试超限或明确阻断，则返回需要用户介入的消息。

### 3.3 当前核心状态字段

Agent 的状态结构定义在 AgentState 中，当前字段如下：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| messages | list[BaseMessage] | LangGraph 状态消息历史 |
| plan | List[Dict[str, Any]] | 子任务计划数组 |
| current_subtask_index | int | 当前执行到的子任务索引 |
| reflections | List[str] | 每轮反思说明文本 |
| global_keywords | List[str] | 全局任务关键词 |
| failed_tools | Dict[str, List[str]] | 按子任务索引记录已失败工具名，供再路由过滤 |
| failed_tool_signals | Dict[str, Dict[str, Dict[str, Any]]] | 按子任务索引记录失败工具的结构化信号摘要与严重度 |
| request_id | str | 一次 agent 调用的唯一追踪 ID |
| session_id | str | 会话级唯一标识 |
| session_memory_id | int | 当前会话主记录在记忆库中的 ID |
| domain_label | str | 当前任务的领域标签 |
| memory_summaries | List[Dict[str, Any]] | 摘要级记忆召回结果 |
| retry_counts | Dict[str, int] | 每个子任务当前累计重试次数 |
| replan_counts | Dict[str, int] | 每个子任务已触发重规划的次数 |
| blocked | bool | 当前流程是否被阻断 |
| final_response | str | 最终返回或落盘的文本 |

## 4. 当前已经具备的能力

- 基于 LangGraph 的状态图执行骨架
- 基于 LLM 的特征提取、领域判断、任务拆解与反思判断
- 会话级与步骤级记忆写入，以及摘要召回和按需详情加载
- request_id 与 session_id 双层追踪
- 文件日志中的结构化 JSON 事件，以及控制台阶段日志简化输出
- 统一结构化输出解析层，用于认知模块 JSON 结果校验
- request_id 级运行时状态快照持久化与快照恢复继续执行
- request_id 级运行时状态快照持久化，以及可选的恢复后 reroute 重规划
- request_summary 与 recent_requests 两类请求级聚合视图
- failure_case 失败案例记忆视图，可独立于成功经验检索
- 快照恢复前的语义一致性校验，补充 retry/replan/failed_tools 索引边界与 terminal 状态合法性检查
- blocked 或中途中断的快照可显式进入 reroute 恢复模式，把恢复点上下文重新交给 planner 改路
- 跨子任务失败累计已进入 reroute 策略，重复失败的工具会在后续步骤中被降级或排除，备选工具和当前候选工具也会按历史失败严重度与次数做优先级排序
- 高严重度历史失败现在也会触发策略级降级：超过 severity 阈值的工具会从当前和 reroute 候选中剔除，必要时直接转入 no-tool 路径
- no-tool 路径的提示已区分 ask-user 与 direct-answer：缺参型失败会明确追问用户，高风险或无工具可用时则优先尝试安全的无工具回答
- reroute fallback 信号已进入 request triage 与 recent request 摘要，可直接看到最近一次 reroute mode，而不必手工翻 checkpoint
- detached tool 计数、明细聚合与运行态查询视图
- checkpoint 级阶段耗时分布聚合视图
- LLM、工具、整次请求三级超时控制与逻辑取消返回
- 按 request_id 的协作式主动取消，覆盖主请求等待、LLM 调用、工具等待和 CLI Ctrl+C
- 工具异常统一分类、结构化失败返回与参数预校验
- 工具运行生命周期跟踪、detached 标记与后台完成后回收
- 工具失败后的自动排除、替代工具搜索与无工具降级再路由
- retry 路径上的一次性子任务重规划
- Python 工具自动扫描加载
- 自动加载会跳过 sample_*.py 与 test_*.py 这类示例/测试技能文件
- Markdown Prompt 技能自动扫描和统一匹配
- Prompt 技能与 Python/MCP 工具的统一能力注册和路由
- 配置式 MCP 工具注册能力
- 真实 MCP stdio server 的加载、列举工具、远端调用、连接复用、自动重连、刷新与卸载
- 独立 Python MCP server：支持终端命令、系统信息和文件信息查看
- replay 历史记忆重放
- memory -> Markdown skill 转换

## 5. 当前仍属于基础版的部分

- MCP 真实 transport 目前只覆盖 stdio 基础版，尚未补工具刷新订阅、资源同步和更细粒度健康检查
- 超时与取消目前仍属于逻辑取消，不是底层线程、进程或网络调用的硬终止
- 技能与工具路由仍以关键词和阈值匹配为主，没有混合检索或更强语义检索
- 记忆治理目前已有基础质量标签和精确去重，但还没有归档、衰减和相似合并体系
- 观测性已有 request 级聚合，但还没有跨 request 的全局指标视图

## 6. 项目结构概览

| 路径 | 角色 |
| --- | --- |
| cli.py | CLI 入口与主循环 |
| cli_commands.py | CLI 命令定义、帮助文本与 handler 注册表 |
| agent_core.py | Agent 门面与 LangGraph 图构建点 |
| agent_runtime.py | 请求生命周期、invoke、恢复执行、取消与超时协调 |
| agent_snapshots.py | request 快照序列化、落盘、恢复加载 |
| agent_observability.py | request summary、recent requests 与指标聚合 |
| agent_tools.py | 工具运行时包装、参数预校验、失败解析与 reroute 计划 |
| llm_manager.py | LLM 对外门面 |
| llm_logging.py | 结构化日志、控制台输出与 request_id 事件反查 |
| llm_runtime.py | LLM request_scope、超时执行与取消检查 |
| llm_factory.py | provider 对应的 LLM 实例构造 |
| config.py / config.json | 配置解析、默认模型与目录配置 |
| cognitive/ | 特征提取、规划、反思三个认知子系统 |
| memory/ | SQLite 记忆管理与大输入备份 |
| skills/ | 真实运行时 Python 工具技能目录 |
| examples/skills/ | 示例 Python 技能，不参与自动加载 |
| skills_md/ | Markdown Prompt 技能解析与统一能力路由 |
| mcp_servers/mcp_manager.py | MCP 双模接入层 |
| mcp_servers/system_mcp_server.py | 独立 Python MCP server |
| logs/ | LLM 日志输出目录 |
| runtime_state/ | request_id 级运行时状态快照与审计目录 |
| tests/ | 回归测试目录 |
| docs/ | 文档目录 |

## 7. 运行入口与对外接口

### 7.1 CLI

项目运行入口是 cli.py。start_cli() 会启动命令行循环，并在启动时创建一个新的 session_id。

当前 CLI 结构：

- cli.py 只保留会话启动、命令调度和异常兜底
- cli_commands.py 维护命令元数据、帮助文本和具体 handler

当前 CLI 支持的主要管理命令：

| 命令 | 作用 |
| --- | --- |
| /llm <provider> <model> [base_url] [api_key] | 切换当前模型配置 |
| /load_skill <skill_name.py\|skill_name.md> | 增量加载 Python 或 Markdown 技能 |
| /load_mcp <config_name.json\|config_name.yaml\|absolute_path\|server.py\|stdio:command ...> | 加载配置式 MCP 工具或真实 stdio MCP server |
| /list_mcp | 查看已加载的 MCP server 和 transport |
| /refresh_mcp <server_name\|source> | 刷新 MCP server 并重新枚举工具 |
| /unload_mcp <server_name\|source> | 卸载 MCP server 并移除其工具 |
| /cancel_request <request_id> | 对活跃请求发起协作式取消 |
| /list_snapshots <request_id> | 列出可用恢复点 |
| /resume_snapshot <request_id> [snapshot_file] | 从快照恢复继续执行 |
| /request_summary <request_id> | 展示请求摘要、指标、checkpoint、快照和记忆 |
| /recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m] | 展示最近请求摘要，并支持按状态、恢复请求、attention 请求和时间窗口筛选；当 failure details 可解析时会直接显示 tool、reason、error_type 或 action 摘要 |
| /failed_requests [limit] [status=failed,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m] | 失败导向的 recent 查询别名，默认聚焦失败、阻断、超时和取消请求，并支持时间窗口筛选 |
| /resumed_requests [limit] [status=failed,blocked,...] [attention] [since=30m] | 恢复请求的 recent 查询别名，默认聚焦 resumed 请求，并支持时间窗口筛选 |
| /request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m] | 展示最近若干请求的聚合统计，并支持按状态、恢复请求、attention 请求和时间窗口筛选，同时输出 top failure signals 与 failure combination 热点 |
| /list_tool_runs [request_id] [running\|detached] | 展示当前运行中或已 detached 的工具运行态 |
| /failure_memories [limit] [keywords...] | 单独查看 failure_case 记忆视图，聚焦 blocked、retry、timeout、ask_user 等失败经验 |
| /retention_status | 展示 logs、snapshots、audit logs 和 memory backups 的 retention 状态、可回收体积、最近一次自动 prune 摘要，以及最近一次自动 prune 跳过原因 |
| /prune_runtime_data [apply] | 对 retention 范围内的运行时产物执行 dry-run 或实际清理 |
| /replay <memory_id> [injected features...] | 重放历史记忆并重新执行 |
| /convert_skill <memory_id> | 将历史记忆输出转换为 Markdown 技能 |
| /new_session | 创建新的会话 ID |
| exit / quit | 退出 CLI |

### 7.2 AgentCore 对外接口

AgentCore 当前暴露的主要方法如下：

| 方法 | 作用 |
| --- | --- |
| invoke(query, session_id=None) | 执行一次完整 Agent 调用 |
| replay(memory_id, injected_features=None) | 重放历史记忆并重新执行 |
| resume_from_snapshot(request_id, snapshot_name=None) | 从指定快照恢复继续执行 |
| list_snapshots(request_id) | 列出指定请求的可用快照 |
| get_request_summary(request_id) | 聚合单次请求的快照、checkpoint、memory 与指标 |
| get_recent_request_summaries(limit=10) | 聚合最近请求摘要视图 |
| get_request_rollup(limit=20) | 聚合最近请求的全局状态、计数、阶段耗时总览和 failure 热点 |
| get_failure_memories(match_keywords=None, limit=5) | 聚合失败案例记忆视图，优先返回 failure_case 与失败质量标签记录 |
| get_retention_status() | 汇总运行时产物的 retention 覆盖范围、过期数量、可回收体积、最近一次自动 prune 摘要和最近一次自动 prune 决策 |
| prune_runtime_data(apply=False) | 预览或执行过期 logs、snapshots、audit logs 和 memory backups 的清理 |
| convert_memory_to_skill(memory_id) | 将历史记忆转换为 Markdown 技能 |
| load_skill(skill_name) | 手动加载 Python 或 Markdown 技能 |
| load_mcp_server(server_ref) | 加载配置式 MCP 工具或真实 stdio MCP server |
| list_mcp_servers() | 列出当前已加载的 MCP server |
| refresh_mcp_server(server_ref) | 刷新某个 MCP server 并重新枚举工具 |
| unload_mcp_server(server_ref) | 卸载某个 MCP server 并移除其工具 |
| add_tool(tool) | 运行时动态添加工具 |
| start_session(session_id=None) | 新建或切换 session_id |
| get_last_request_id() | 获取最近一次调用生成的 request_id |

## 8. 主执行链

### 8.1 LangGraph 节点

AgentCore._build_graph() 当前构建的主节点如下：

| 节点名 | 内部函数 | 作用 |
| --- | --- | --- |
| planner | initial_planning | 全局特征提取、领域判断、记忆召回、会话主记忆初始化、任务拆解 |
| agent | call_model_subtask | 执行当前子任务、选择技能与工具、调用模型 |
| reflect_and_advance | reflect_and_advance | 验证结果并决定推进、重试、重规划或阻断 |
| tools | ToolNode(self.tools) | 执行模型发起的工具调用 |

主链路可以概括为：

START -> planner -> agent -> tools 或 reflect_and_advance -> agent 或 END

### 8.2 初始规划阶段

initial_planning 当前会做这些事：

1. 读取用户输入。
2. 提取全局关键词与一句话摘要。
3. 判断任务领域。
4. 基于全局关键词做摘要级记忆召回。
5. 写入一条会话主记忆记录。
6. 调用 TaskPlanner 生成 JSON 子任务数组。
7. 写入 planning_started / planning_completed checkpoint。
8. 按 request_id 持久化规划阶段状态快照。

### 8.3 子任务执行阶段

call_model_subtask 当前逻辑如下：

1. 读取当前子任务描述。
2. 提取子任务关键词，最多保留 5 个。
3. 优先从已有摘要记忆中选相关项；不够时再次查询记忆库。
4. 调用 SkillManager.assign_capabilities_to_task() 统一选择 Prompt 技能和候选工具。
5. 把原始请求、当前子任务、记忆摘要、技能正文、候选工具说明拼成 prompt。
6. 如果存在工具，则先 bind_tools()，再通过 llm_manager.invoke() 调用模型。
7. 写入 subtask_started / subtask_llm_dispatch 等 checkpoint。
8. 在真正发起模型调用前落一份子任务准备态快照。

当前执行特点：

- 每个子任务最多挂一个 Markdown Prompt 技能
- 每个子任务可以绑定多个工具
- 工具包括 Python 技能工具、配置式 MCP 工具和真实 MCP 远端工具
- 如果最近一轮工具执行失败，失败工具会从当前子任务候选中排除
- 可重试失败会触发更大候选池的替代工具搜索
- 参数错误等不安全重试场景会降级为无工具路径，并引导模型直接回答或请求用户补参

### 8.4 反思推进阶段

reflect_and_advance 负责闭环控制，输入包括：

- 当前子任务描述
- 当前子任务的 expected_outcome
- 上一个 agent 节点输出的实际结果

Reflector 会返回：

- success
- reflection_note
- action: continue / retry / ask_user

当前推进规则：

- success 或 action=continue：进入下一个子任务
- action=retry：留在当前子任务重试
- action=ask_user：阻断并返回提示用户介入
- retry 超过上限：升级为阻断
- 第一次 retry 时，如果当前子任务仍明显失败，系统可触发一次 planner 驱动的子任务重规划，并记录 subtask_replanned checkpoint / snapshot

## 9. 日志、追踪与请求聚合

### 9.1 request_id 与 session_id

当前项目使用两层追踪标识：

- session_id：标识同一会话
- request_id：标识一次具体的 agent 调用

invoke() 每次都会生成新的 request_id，并把整个 graph 执行包在 llm_manager.request_scope() 中。因此同一次调用中产生的多次 LLM 请求、checkpoint 日志和记忆记录都可以通过同一个 request_id 追踪。

### 9.2 LLMManager 当前结构

llm_manager.py 当前保留对外门面，内部已拆分为以下职责面：

- llm_manager.py：公开的 invoke、set_model、request_scope、日志读取与事件记录入口
- llm_logging.py：结构化日志、控制台阶段日志、payload/response 序列化、request_id 事件反查
- llm_runtime.py：request_scope、ContextVar、超时等待和取消检查
- llm_factory.py：不同 provider 的 LLM 实例构造

当前日志与追踪能力包括：

- 统一的 LLM request / response / error 日志
- 结构化 JSON 事件写入文件日志
- request_id 上下文注入
- 可截断的长文本日志
- checkpoint 日志
- console_event 简化阶段输出
- 按 request_id 从日志中回放事件列表

### 9.3 请求摘要与最近请求视图

AgentCore 当前提供两类聚合视图：

- get_request_summary(request_id)
- get_recent_request_summaries(limit)
- get_request_rollup(limit)

当前汇总指标包括：

- total_duration_ms
- llm_call_count / llm_error_count / llm_total_duration_ms
- tool_call_count / tool_success_count / tool_failure_count / tool_rejection_count / tool_cancelled_count / tool_detached_count
- tool_hit_rate
- stage_duration_ms
- retry_count
- subtask_count
- reflection_failure_count
- blocked_rate

其中 request_rollup 当前额外提供：

- request_count / status_counts
- resumed_count / needs_attention_count / active_count
- average_total_duration_ms
- stage_duration_ms_total
- top_failure_signals
- top_failure_combinations
- totals.llm_call_count / tool_call_count / tool_detached_count / retry_count / reflection_failure_count

当前 retention 视图额外提供：

- logs / snapshots / audit_logs / memory_backups 四类目标的 item_count、expired_count、total_bytes 和 reclaimable_bytes
- logs、audit_logs、memory_backups 可同时按 retention_days、max_files 和 max_bytes 控制
- snapshots 可同时按 retention_days、max_request_dirs 和 max_bytes 控制
- prune_runtime_data 的 dry_run / apply 两种模式
- retention_auto_prune_enabled 与 retention_auto_prune_min_interval_seconds 可控制低频自动 prune
- last_auto_prune 会保留最近一次自动清理的 trigger、deleted_count、expired_count、reclaimable_bytes 和执行时间
- last_auto_prune_check 会保留最近一次自动 prune 决策，区分 executed、skipped_throttled 和 skipped_disabled
- snapshots 以 request 目录为粒度清理，避免只删单个快照文件导致目录残缺

### 9.4 运行时快照与恢复

主执行链会把关键阶段状态写入 runtime_state/snapshots。当前已支持：

- 按 request_id 读取最新快照继续执行
- 按 request_id + latest / index / stage / 显式文件名选择恢复点
- 对已完成或已阻断的终态快照直接返回终态结果

当前超时控制已接入三层：

- 单次 LLM 调用超时
- 单次工具执行超时
- 整次 Agent 请求超时

现阶段超时后的处理方式仍属于逻辑取消：主流程会返回超时结果并写日志或快照，但不会强制终止已经在底层启动的线程执行。

当前工具运行时额外补了一层生命周期治理：

- 工具调用会登记运行态并在超时或取消后尝试宽限清理
- 无法及时停止的工具会记录为 detached tool run
- detached 工具会进入 request_summary、recent_requests 和 /list_tool_runs 的观测视图
- 后台执行完成后，运行态记录会被回收

## 10. 核心模块说明

### 10.1 AgentCore 及其 helper

AgentCore 是总协调器，负责：

- 初始化认知、规划、反思、记忆、技能、MCP 子系统
- 自动扫描 Python 技能目录
- 自动扫描 MCP 配置目录
- 构建并持有 LangGraph 图
- 委托 agent_runtime.py 处理 request 生命周期、invoke、恢复执行、取消与超时协调
- 委托 agent_snapshots.py 处理快照序列化、持久化与恢复加载
- 委托 agent_observability.py 处理请求摘要、最近请求视图与指标聚合
- 委托 agent_tools.py 处理工具包装、参数预校验、失败聚合与 reroute 计划

helper 职责如下：

| 模块 | 当前职责 |
| --- | --- |
| agent_runtime.py | request 注册、清理、活跃状态和取消状态管理；graph.invoke 的线程池调度与超时等待；invoke / resume 生命周期收口 |
| agent_snapshots.py | 状态消息序列化与反序列化；request_id 级快照目录解析；快照落盘与恢复载入 |
| agent_observability.py | request status 推导；checkpoint / LLM 事件指标聚合；request_summary 与 recent_requests 视图 |
| agent_tools.py | 工具异常分类；参数预校验；工具运行超时、取消与结构化失败包装；失败解析与 reroute 计划构造 |

### 10.2 CognitiveSystem

cognitive/feature_extractor.py 中的 CognitiveSystem 当前提供：

- extract_features(text, domain_hint="")：提取关键词和一句话摘要，要求模型返回 JSON
- determine_domain(text)：从内置领域树中选择一个领域标签

当前特征：

- 两个方法都通过 llm_manager.invoke() 调模型
- JSON 输出解析通过统一结构化解析模块完成
- 失败时有降级逻辑
- 领域树仍为硬编码列表，不是动态领域图谱

### 10.3 TaskPlanner

TaskPlanner 负责把用户输入拆成 JSON 数组子任务，每个子任务至少包含：

- id
- description
- expected_outcome

当前实现特点：

- 输出解析通过统一结构化解析模块完成
- 会对 description 和 expected_outcome 做基础校验和规范化
- 如果模型输出无法正确解析，会退化为单子任务执行

### 10.4 Reflector

Reflector 负责比较预期结果和实际结果，并让模型返回 success、reflection、action 三项结构化结果。

当前实现特点：

- 反思输出经过统一结构化解析
- action 被限制为 continue、retry、ask_user 三种合法值
- 解析失败时退化为 ask_user，避免主链静默失控

### 10.5 MemoryManager

memory/memory_manager.py 中的 MemoryManager 基于 SQLite 实现持久化，当前 interactions 表包含：

- id
- conv_id
- request_id
- timestamp
- memory_type
- quality_tags
- domain_label
- keywords
- summary
- raw_input
- raw_output
- large_file_path
- weight

当前主要特点：

- 数据库和备份目录从 config.json 读取并解析为项目内绝对路径
- 老数据库会自动补 request_id、memory_type、quality_tags 列
- raw_input 超过 5000 字符时会写入外部备份文件
- retrieve_memory() 会把质量标签纳入排序，优先成功案例
- add_memory() 会对精确重复记录做合并，累加权重并合并关键词与质量标签

### 10.6 SkillManager

skills_md/skill_parser.py 中的 SkillManager 现在同时负责：

- 自动扫描 skills_md/ 下的 Markdown 技能
- 解析 YAML Front-matter
- 注册 Python 工具技能元数据
- 注册 MCP 工具元数据
- 卸载 MCP 工具时移除对应工具索引
- 用统一的关键词阈值规则选择 Prompt 技能与工具

当前仍以关键词和简单匹配比例为主，不是语义向量检索。

### 10.7 MCPManager

mcp_servers/mcp_manager.py 当前已经是双模接入层：

- 读取 JSON / YAML 配置并生成本地 StructuredTool
- 支持通过 Python server 文件、stdio 配置或 stdio:command ... 连接真实 MCP stdio server
- 在真实 server 模式下，初始化会列举远端工具，并把工具包装成可调用的 StructuredTool
- 对真实 stdio server 维持可复用连接，并支持显式卸载/关闭
- 当连接失效时，可在下一次访问时自动重连，也支持显式 refresh 重建连接并重新枚举工具

### 10.8 system_mcp_server

mcp_servers/system_mcp_server.py 是仓库内置的独立 Python MCP server，当前提供的工具包括：

- execute_terminal_command
- get_system_info
- get_mcp_security_policy
- inspect_file_system_path

当前还带有一层基础安全控制：

- 默认命令前缀白名单
- 默认工作区根目录路径白名单
- 环境变量扩展的 allowed roots / allowed commands
- runtime_state/audit/system_mcp_audit.jsonl 审计日志

## 11. 配置与默认值

配置由 config.py 从 config.json 读取。重要配置项包括：

| 配置项 | 说明 |
| --- | --- |
| default_model | 默认模型键名 |
| models | 模型配置集合 |
| mcp_dir | MCP 配置目录 |
| skills_dir | Python 技能目录 |
| skills_md_dir | Markdown 技能目录 |
| memory_db_path | SQLite 记忆库路径 |
| memory_backup_dir | 长输入备份目录 |
| log_dir | 日志目录 |
| llm_log_file | LLM 详细日志文件名 |
| llm_log_max_chars | 单条日志最大字符数 |
| state_snapshot_dir | 运行时状态快照目录 |
| llm_timeout_seconds | 单次模型调用超时秒数 |
| tool_timeout_seconds | 单次工具调用超时秒数 |
| request_timeout_seconds | 整次请求超时秒数 |
| prompt_skill_min_overlap | Prompt 技能最小关键词重叠数 |
| prompt_skill_min_match_ratio | Prompt 技能最小匹配比例 |
| tool_skill_min_overlap | 工具最小关键词重叠数 |
| tool_skill_min_match_ratio | 工具最小匹配比例 |

当前默认模型是 local_ollama，对应：

- provider = ollama
- model = lfm2:latest
- base_url = http://localhost:11434

## 12. 测试现状

当前 tests/ 下的主要回归测试文件包括：

- test_skill_routing.py：技能与工具路由阈值
- test_core_reliability.py：结构化输出、快照、超时、取消、再路由与观测性路径
- test_memory_quality.py：记忆类型、质量标签、重复合并与排序
- test_tool_argument_prevalidation.py：工具参数预校验与运行期输入拦截
- test_cli_and_integration.py：CLI 命令、help/new_session、主链成功/阻断集成、MCP CLI 命令
- test_failure_and_migration.py：记忆库迁移与 retry_limit 阻断路径
- test_skill_loading.py：运行时自动加载会跳过示例/测试技能文件
- test_mcp_manager_transport.py：真实 MCP stdio transport、连接复用、自动重连、refresh、卸载
- test_system_mcp_server.py：system_mcp_server 的安全控制与审计行为

当前测试覆盖重点包括：

- 技能与工具路由阈值
- 记忆质量与重复合并
- 请求级摘要与最近请求聚合
- 快照恢复
- 超时控制与逻辑取消
- 工具失败后的再路由与 retry_limit 阻断
- 子任务重规划
- CLI 命令模块化后的帮助渲染与 session 切换
- 真实 MCP stdio transport 与 system MCP server

## 13. 当前实现状态评估

### 13.1 已经比较稳定的部分

- CLI -> AgentCore -> LangGraph 的可运行主链
- request_id / session_id 双层追踪
- 结构化日志落文件、控制台阶段输出精简
- request_summary / recent_requests 聚合视图
- request_id 级状态快照与恢复执行
- 工具异常安全包装与工具失败后的基础自动再路由
- 三级超时控制与协作式主动取消
- 特征提取、规划、反思、重试、阻断闭环
- retry 路径上的一次性子任务重规划
- 会话级和步骤级记忆写入与召回
- Python 工具、Markdown Prompt 技能、配置式 MCP 工具和真实 MCP 工具统一路由
- 独立 Python MCP server 与基础版真实 stdio transport

### 13.2 仍未闭环的部分

- 真实 MCP 仍缺少更完整的长连接工程化管理与资源同步机制
- 超时和取消仍是逻辑层面，不是底层硬中止
- 能力选择仍偏启发式，复杂任务下稳定性有限
- 记忆治理还缺少更强的长期维护策略

## 14. 与其他文档的关系

当前文档的角色是“源码事实总览”。

其他文档的定位如下：

- README.md：项目入口说明与快速阅读指引
- docs/requirements.md：目标态需求文档
- docs/agent.md：能力概念补充
- docs/completeness_roadmap.md：后续仍未完成事项的路线图

因此，若这些文档与源码不一致，应优先相信源码，其次参考本文，再回头对照目标态文档。

## 15. 建议的阅读顺序

1. 先读本文，建立整体认知。
2. 再读 agent_core.py，理解主状态和主执行链。
3. 接着读 agent_runtime.py、agent_snapshots.py、agent_observability.py、agent_tools.py，理解 AgentCore 已拆出的辅助职责。
4. 然后读 cli.py、cli_commands.py、llm_manager.py、llm_logging.py、llm_runtime.py、llm_factory.py，理解入口、命令分发、输出和追踪方式。
5. 再读 cognitive/、memory/、skills_md/、mcp_servers/。
6. 最后再读 README.md、docs/requirements.md、docs/agent.md、docs/completeness_roadmap.md，对照现状与后续路线。

## 16. 结论

llm_brain 当前仍是原型工程，但已经不再只是最小骨架。它现在具备了可追踪的执行链、基础记忆闭环、统一技能路由、请求级可观测性、快照恢复，以及基础版真实 MCP stdio 接入能力。

对后续接手者来说，最重要的不是把目标文档当作现状，而是先理解当前已经可靠落地的这些链路，再决定下一步要继续补的是更强的恢复能力、路由质量、观测深度，还是协议工程化。