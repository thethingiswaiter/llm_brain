# llm_brain 项目总览

## 1. 文档目的

这份文档只描述当前代码库已经实现的事实，不讲路线图，不复述历史方案，也不替 README 做快速上手。

文档目标只有三个：

- 说明这个项目现在是什么
- 说明主执行链如何跑起来
- 说明主要模块、能力边界和当前短板分别在哪里

如果源码和本文不一致，以源码为准。

## 2. 一句话定义

llm_brain 是一个以 LangGraph 为执行骨架的 Agent 原型系统，当前以 Textual TUI 作为终端入口；系统围绕“意图重写/轻聊天判断 -> 任务规划 -> 子任务执行 -> 工具调用 -> 反思推进 -> 可恢复收尾”运行，并已经接入工作空间级文件访问、技能/工具统一路由、请求级快照恢复、跨请求观测、运行时保留治理，以及可按需启用的 MCP server。

## 3. 当前项目画像

### 3.1 当前定位

- 形态上，是一个带恢复、观测、安全边界和终端交互层的 Agent 原型。
- 交互上，根目录入口 [main.py](main.py) 直接启动 Textual TUI；legacy CLI [app/cli/main.py](app/cli/main.py) 仍作为兼容与测试入口保留，但不再是默认用户入口。
- 架构上，核心编排集中在 [app/agent/core.py](app/agent/core.py)，运行生命周期收口在 [app/agent/runtime.py](app/agent/runtime.py)，快照恢复、可观测性、工具治理、保留策略分别拆成独立子系统。
- 能力来源上，系统同时支持 Python 工具、Markdown Prompt 技能和 MCP 工具，并把它们统一注册到同一条任务执行链中。

### 3.2 当前最重要的实现特征

- 用户面对的终端形态已经收口到 Textual TUI；legacy CLI 主要保留给兼容与测试使用，不再作为产品层面的第二终端入口。
- 仓库根和工作空间根已经分离。工具读写、技能目录、MCP 允许根、UI 文件浏览都优先基于 `workspace_root`，而不是硬编码仓库目录。
- 请求执行不只是单轮 invoke，还具备 request 级快照、恢复、reroute、取消、超时、失败聚合和历史失败信号路由。
- 当前实现已经开始显式区分“普通完成”“等待用户补充”“阻塞”“失败”“超时”“恢复后继续”等不同请求状态。

## 4. 启动与交互入口

### 4.1 启动入口

- 根入口：[main.py](main.py)
- 默认 UI：Textual TUI，实现在 [app/cli/textual_app.py](app/cli/textual_app.py)

当前启动逻辑很直接：

1. `main.py` 直接启动 Textual TUI。
2. 如果 Textual 依赖缺失，应视为运行环境不满足，而不是回退到另一套用户终端。

### 4.2 两套终端交互层的关系

当前仓库里仍然有两套终端代码路径，但对外定位已经不同：

- Textual TUI：当前唯一默认用户入口，提供固定工作台、状态栏、日志窗口、输入框、自动补全、快捷键和工作空间文件浏览。
- legacy CLI：保留的兼容/测试入口，用于复用命令处理与集成测试覆盖。

两层共享大部分命令与输出逻辑，但产品层面不再强调“双终端”。现在更准确的说法是：

- 公共命令体系来自 [app/cli/commands.py](app/cli/commands.py)
- Textual TUI 与 legacy CLI 共用大部分命令与恢复策略
- “普通补充句自动续接最近 waiting_user/blocked 请求”的能力现在已经同时接到 Textual TUI 和 legacy CLI

也就是说，当前“用户补一句话后自动续接上下文”的能力已经是共享行为，不再只属于 legacy CLI。

## 5. 工作空间模型

### 5.1 仓库根与工作空间根

[core/config.py](core/config.py) 现在区分了两个重要概念：

- `base_dir`：应用仓库根，用来定位项目自身代码和默认配置
- `workspace_root`：当前任务实际操作的工作区根，用来约束文件浏览、读写和 MCP 允许根

这意味着项目本身可以驻留在一个目录里，但实际允许操作的文件空间由配置决定。

### 5.2 当前基于工作空间根的能力

以下能力已经改成基于 `workspace_root`：

- 本地读写工具路径解析
- 默认可写根与额外授权写根
- Markdown 技能目录定位
- system MCP server 的允许路径根
- CLI/TUI 的工作区文件浏览与选择

### 5.3 工作区文件选择能力

当前终端交互层已经支持在工作空间内浏览和选择文件：

- 命令入口：`/workspace_files` 与 `/files`
- 选择机制：`/pick`
- Textual 快捷键：`Ctrl+O`

选择后的文件路径会写入当前会话选择上下文；在 TUI 中会直接回填到输入框，便于继续提问或拼接命令。

## 6. 主执行链

### 6.1 高层流程

普通请求的主链路可以概括为：

1. UI 接收用户输入。
2. 请求进入 `AgentCore.invoke(query, session_id)`。
3. 系统生成新的 `request_id`，建立 request scope。
4. 规划阶段提取特征、确定领域、召回记忆、拆解子任务。
5. 执行阶段为当前子任务选择技能、工具和补充提示。
6. 模型如发起 `tool_calls`，则进入工具节点执行，再回到模型节点。
7. 反思阶段判断当前子任务应当 `continue`、`retry` 还是 `ask_user`。
8. 成功则推进到下一子任务或结束；失败则可能重试、重规划、阻塞、等待用户或转入恢复路径。

### 6.2 LangGraph 节点

[app/agent/core.py](app/agent/core.py) 当前构建的主状态图包含四个核心节点：

- `planner`：初始规划
- `agent`：当前子任务执行
- `tools`：工具调用
- `reflect_and_advance`：反思与推进

主流程仍然是标准的 Planner-Agent-Tools-Reflect 结构，但项目已经在每个阶段上叠加了更多运行时治理：

- request 级快照
- 工具失败历史
- reroute 与 no-tool 降级
- 请求取消/超时
- 失败来源归因

## 7. Agent 状态模型

当前 `AgentState` 定义在 [app/agent/core.py](app/agent/core.py)，不是极简消息列表，而是一个带执行状态和治理信号的运行态对象。关键字段包括：

- `messages`：LangGraph 消息历史
- `raw_query`：用户原始输入
- `normalized_query`：重写后的意图文本
- `plan`：当前子任务计划
- `current_subtask_index`：当前执行到哪一步
- `reflections`：反思结果历史
- `global_keywords`：全局关键词
- `failed_tools`：按子任务记录失败工具
- `failed_tool_signals`：失败工具的结构化信号与严重度
- `retry_counts` / `replan_counts`：重试与重规划计数
- `blocked` / `waiting_for_user`：是否阻塞或等待用户补充
- `final_response`：当前请求最终输出
- `lite_mode`：是否走轻聊天模式
- `subtask_feature_cache`：子任务级特征缓存
- `agent_action`：当前代理动作标签

这套状态设计决定了项目当前重点不是“单次回答”，而是“可追踪、可恢复、可诊断的一次请求”。

## 8. 核心子系统分工

### 8.1 核心编排

- [app/agent/core.py](app/agent/core.py)：Agent 门面、状态图构建、主执行策略、子系统协调
- [app/agent/runtime.py](app/agent/runtime.py)：invoke / resume / timeout / cancel 等请求级生命周期收口
- [app/agent/snapshots.py](app/agent/snapshots.py)：快照落盘、迁移、校验、恢复状态构造

### 8.2 认知与决策

- [cognitive/feature_extractor.py](cognitive/feature_extractor.py)：特征提取与领域判断
- [cognitive/planner.py](cognitive/planner.py)：任务拆解
- [cognitive/reflector.py](cognitive/reflector.py)：反思与行动决策
- [cognitive/structured_output.py](cognitive/structured_output.py)：结构化输出解析与约束

### 8.3 工具与技能

- [app/agent/tools_runtime.py](app/agent/tools_runtime.py)：工具包装、参数预校验、失败分类、运行态跟踪
- [app/agent/skill_parser.py](app/agent/skill_parser.py)：Markdown Prompt 技能加载与路由
- [tools/langchain_common_tools.py](tools/langchain_common_tools.py)：本地文件/目录/搜索等常用工具

### 8.4 记忆、观测与保留

- [memory/memory_manager.py](memory/memory_manager.py)：SQLite 记忆存储、召回与备份
- [app/agent/observability.py](app/agent/observability.py)：request_summary / recent_requests / rollup 聚合
- [app/agent/retention.py](app/agent/retention.py)：logs / snapshots / audit / memory backups 的保留与清理

### 8.5 模型与日志

- [core/llm/manager.py](core/llm/manager.py)：模型门面
- [core/llm/factory.py](core/llm/factory.py)：Provider 实例构造
- [core/llm/runtime.py](core/llm/runtime.py)：request scope、超时与取消基础设施
- [core/llm/logging.py](core/llm/logging.py)：结构化日志与阶段事件

## 9. 当前已经具备的主要能力

### 9.1 请求执行与恢复

- 基于 LangGraph 的多阶段请求执行
- 每次请求分配 `request_id`，与 `session_id` 组成双层追踪
- request 级状态快照持久化
- 从最新快照、索引、阶段名或显式文件名恢复
- blocked / waiting_user / 中断请求的 reroute 恢复
- 恢复前做快照一致性校验，避免在坏状态上继续跑
- Textual TUI 与 legacy CLI 都已支持“用户直接补一句话”自动续接最近 waiting_user/blocked 请求
- 恢复链路已经可以携带 follow-up 文本，把补充信息注入 reroute 上下文

### 9.2 工具治理

- 工具参数预校验
- 工具失败统一结构化返回
- detached tool run 跟踪
- 超时、取消、执行失败、阻塞等失败类型归一化
- 跨子任务累积失败历史，并进入后续工具选择排序
- 高严重度失败工具可直接排除
- 无安全工具路径时可降级为 no-tool 路径
- 写任务场景下优先写工具而不是终端 fallback
- 写任务成功需要可验证写入回执，不能仅凭“看起来像成功”判定完成

### 9.3 技能与能力路由

- Python 工具自动扫描加载
- Markdown Front-matter 技能自动扫描和解析
- Prompt 技能、Python 工具、MCP 工具统一注册到同一能力面
- 路由时会考虑任务关键词、匹配比例、只读/写入意图和终端命令偏置
- 规划阶段会看到当前可用能力，而不是脱离真实工具集做抽象拆解

### 9.4 观测与排障

- 单请求聚合：`request_summary`
- 最近请求聚合：`recent_requests`
- 跨请求聚合：`request_rollup`
- triage 中保留 latest failure stage / source / details / reroute mode
- rollup 中聚合 failure signals、failure combinations、source buckets 和趋势
- 日志目录、审计目录、快照目录分离
- request 失败、模型依赖不可用、工具超时、阻塞路径都能进入统一观测口径

### 9.5 运行时保留治理

- 覆盖 logs、snapshots、audit logs、memory backups
- 支持按保留天数、数量上限、容量上限裁剪
- 支持 dry-run 和 apply
- 支持低频自动 prune 和执行原因记录

### 9.6 工作空间与权限边界

- 默认读操作相对 `workspace_root`
- 默认写根为 `workspace_root`
- 可额外 grant / revoke 写根
- system MCP server 的允许根默认绑定 `workspace_root`
- shell 命令通过 allowlist 前缀与危险操作标记约束
- 禁止 `&&`、`||`、`;`、换行等多段拼接式 shell 操作

## 10. CLI / TUI 当前能力面

### 10.1 终端层已具备的核心交互

- 模型切换
- Python 工具/Markdown 技能/MCP server 动态加载与卸载
- 快照列举与恢复
- 请求摘要、最近请求、请求 rollup
- detached tools、failure memories、retention status
- 工作区文件浏览与选择
- 最近输入、输入候选、自然语言安全映射

### 10.2 当前比较重要的命令类别

代表性命令包括：

- `/llm`
- `/workspace_files` 和 `/files`
- `/grant_write`、`/revoke_write`、`/list_write_roots`
- `/list_snapshots`、`/resume_snapshot`、`/resume`
- `/request_summary`、`/summary`
- `/recent_requests`、`/recent`
- `/request_rollup`、`/rollup`
- `/latest_failure`
- `/resume_last_blocked`
- `/pick`、`/selection`
- `/failure_memories`
- `/retention_status`
- `/prune_runtime_data`

### 10.3 Textual TUI 当前特征

Textual TUI 不是简单终端包装，而是固定工作台：

- 顶部状态区
- 中央日志区
- 输入框与自动补全面板
- 工作区文件浏览快捷键 `Ctrl+O`

它已经具备较好的日常交互体验，当前用户应当把它视为唯一默认终端入口。

## 11. MCP 现状

### 11.1 当前支持什么

- 配置式 MCP server 接入
- 真实 stdio transport 的连接、刷新、卸载和工具枚举
- 仓库内置 system MCP server

### 11.2 system MCP server 当前边界

[mcp_servers/system_mcp_server.py](mcp_servers/system_mcp_server.py) 当前提供的是“受控系统观察能力”，不是无约束 shell：

- 默认允许根来自 `workspace_root`
- 命令前缀必须在 allowlist 中
- 危险命令片段会被拒绝
- 多段 shell 运算符被拒绝
- 所有调用进入审计日志

这意味着它更像一个受限系统检查器，而不是完全开放的远程终端。

## 12. 当前测试形态

### 12.1 测试入口

[tests/run_tests.py](tests/run_tests.py) 当前把测试分成两层：

- 快速回归：核心、技能、记忆、工具安全、MCP server 等模块
- 全量回归：在快速回归基础上再加入 CLI 与 MCP transport 的慢集成测试

### 12.2 当前已验证的事实

最近已明确验证的回归包括：

- `python -m unittest tests.test_cli_and_integration tests.test_core_reliability`
- 结果：`Ran 186 tests`，`OK`

这说明近期这几轮新增的关键能力至少在以下方向上已经被回归覆盖：

- 工作空间根与文件选择
- 请求恢复与 reroute
- 写任务完成判定
- Textual / CLI 续接与集成路径

## 13. 当前最值得注意的实现边界

项目现在已经可用，但仍然是原型系统，不应被误判为“全能力闭环平台”。目前主要边界有这些：

- Textual TUI 和 legacy CLI 共享大部分命令，但两者的角色已经不同；默认交互应以 Textual TUI 为准。
- 工具与技能路由仍以关键词、阈值和规则增强为主，不是更强的混合检索或长期学习路由。
- 超时和取消主要是逻辑层面的协作式中止，不是对所有底层外部调用的硬中断。
- MCP transport 当前以 stdio 基础能力为主，还不是完整协议工程化实现。
- 记忆治理已经有质量标签、失败视图和基础去重，但还没有长期压缩、衰减、归档和聚类合并体系。

## 14. 适合怎么理解这个项目

如果只用一句更准确的话概括现在的 llm_brain，可以这样说：

它已经不是单纯的“命令行聊天脚本”，而是一个带工作空间边界、状态恢复、请求观测、工具治理和 Textual 终端外壳的 Agent 原型内核；真正成熟的部分在于执行链可追踪、失败可诊断、状态可恢复，而不是大而全的能力覆盖。

理解这个项目时，重点应该放在下面几件事上：

- 它如何把一次请求变成一个有 `request_id` 的可恢复运行过程
- 它如何基于真实工作空间根限制工具和文件访问
- 它如何把工具失败、reroute、ask_user、blocked 和 resume 纳入同一条状态链
- 它如何通过 summary / recent / rollup 把一次请求和多次请求都变成可观察对象

## 15. 建议阅读顺序

如果要继续接手开发，建议按这个顺序看：

1. [main.py](main.py)
2. [app/cli/textual_app.py](app/cli/textual_app.py)
3. [app/cli/main.py](app/cli/main.py)
4. [app/cli/commands.py](app/cli/commands.py)
5. [app/agent/core.py](app/agent/core.py)
6. [app/agent/runtime.py](app/agent/runtime.py)
7. [app/agent/snapshots.py](app/agent/snapshots.py)
8. [app/agent/tools_runtime.py](app/agent/tools_runtime.py)
9. [app/agent/observability.py](app/agent/observability.py)
10. [core/config.py](core/config.py)
11. [tools/langchain_common_tools.py](tools/langchain_common_tools.py)
12. [mcp_servers/system_mcp_server.py](mcp_servers/system_mcp_server.py)

这样读，能先把外层交互、再把执行主链、最后把安全边界和周边治理串起来。