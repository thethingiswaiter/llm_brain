# llm_brain 项目总览

## 1. 文档目的

本文档面向后续接手该仓库的人或大模型，目标是基于当前源码说明项目已经实现了什么、主流程如何运转、各模块分别负责什么，以及哪些能力仍处于原型或占位阶段。

阅读本文件时应默认遵循一个原则：

- 以源码事实为准。
- README 与需求文档用于辅助理解，但不能替代代码现状。

## 2. 一句话定义

llm_brain 是一个基于 LangGraph 的通用 Agent 原型，围绕“特征提取 -> 任务规划 -> 子任务执行 -> 反思推进”这一主循环，逐步接入记忆、技能、日志追踪和配置式 MCP 工具注册能力。

## 3. 机器优先摘要

### 3.1 项目定位

- 项目类型：命令行交互式 Agent 原型
- 主入口：cli.py 中的 start_cli
- 主协调器：agent_core.py 中的 AgentCore
- 状态编排：LangGraph StateGraph
- 模型提供方：Ollama / OpenAI
- 记忆存储：SQLite
- 工具来源：Python 技能文件 + 配置式 MCP 工具
- Prompt 技能来源：Markdown Front-matter 技能文件

### 3.2 当前真实主流程

1. CLI 接收普通文本输入或管理命令。
2. 普通文本输入进入 AgentCore.invoke(query, session_id)。
3. 每次调用先生成 request_id，并进入统一日志作用域。
4. planner 节点提取全局关键词、判断领域、召回历史记忆、写入会话主记忆、拆解子任务。
5. agent 节点执行当前子任务，统一选择 Prompt 技能和候选工具，并调用 LLM。
6. 如果模型返回 tool_calls，则转入 ToolNode 执行工具后再回到 agent。
7. reflect_and_advance 节点根据 expected_outcome 与实际结果决定 continue、retry 或 ask_user。
8. 成功完成后返回最后结果；若重试超限或明确阻断，则返回提示用户介入的消息。

### 3.3 当前主要状态字段

Agent 的状态结构定义在 AgentState 中，当前字段如下：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| messages | list[BaseMessage] | LangGraph 状态消息历史 |
| plan | List[Dict[str, Any]] | 子任务计划数组 |
| current_subtask_index | int | 当前执行到的子任务索引 |
| reflections | List[str] | 每轮反思产生的说明文本 |
| global_keywords | List[str] | 全局任务关键词 |
| request_id | str | 一次 agent 调用的唯一追踪 ID |
| session_id | str | 会话级唯一标识 |
| session_memory_id | int | 当前会话主记录在记忆库中的 ID |
| domain_label | str | 当前任务的领域标签 |
| memory_summaries | List[Dict[str, Any]] | 摘要级记忆召回结果 |
| retry_counts | Dict[str, int] | 每个子任务当前累计重试次数 |
| blocked | bool | 当前流程是否被阻断 |
| final_response | str | 最终返回或落盘的文本 |

### 3.4 当前已经具备的能力

- 基于 LangGraph 的状态图执行骨架
- 基于 LLM 的特征提取、领域判断、任务拆解与反思判断
- 会话级与步骤级记忆写入，以及摘要召回和按需详情加载
- request_id 与 session_id 双层追踪
- LLM 详细日志写文件、控制台阶段日志简化输出
- Python 工具自动扫描加载
- Markdown Prompt 技能自动扫描和匹配
- Prompt 技能与 Python/MCP 工具的统一能力注册和路由
- 基于配置文件的最小 MCP 工具注册能力
- replay 历史记忆重放
- memory -> Markdown skill 转换

### 3.5 当前未闭环或仍属原型的部分

- MCP 还没有真实协议连接与远程交互，只是本地配置式适配层
- 技能路由仍以关键词匹配为主，没有向量召回或更强语义检索
- 可信度评分、来源分级、系统化领域建模等仍停留在设计目标层

## 4. 项目结构概览

| 路径 | 角色 |
| --- | --- |
| cli.py | 命令行入口与用户交互层 |
| agent_core.py | Agent 主编排器与 LangGraph 图构建点 |
| llm_manager.py | 模型实例管理、请求日志、控制台阶段输出 |
| config.py / config.json | 配置解析、默认模型与目录配置 |
| cognitive/ | 特征提取、规划、反思三个认知子系统 |
| memory/ | SQLite 记忆管理与大输入备份 |
| skills/ | Python 工具技能目录 |
| skills_md/ | Markdown Prompt 技能解析与统一能力路由 |
| mcp_servers/ | 配置式 MCP 工具描述与适配层 |
| logs/ | LLM 详细日志文件输出目录 |
| tests/ | 最小回归测试，目前覆盖技能路由阈值 |
| docs/ | 文档目录 |

## 5. 运行入口与对外接口

### 5.1 CLI 入口

项目直接运行入口是 cli.py。start_cli() 会启动命令行循环，并在启动时创建一个新的 session_id。

CLI 当前支持的管理命令：

| 命令 | 作用 |
| --- | --- |
| /llm <provider> <model> [base_url] [api_key] | 切换当前模型配置 |
| /load_skill <skill_name.py|skill_name.md> | 增量加载 Python 或 Markdown 技能 |
| /load_mcp <config_name.json|config_name.yaml|absolute_path> | 加载 MCP 配置文件并注册工具 |
| /replay <memory_id> [injected features...] | 读取历史记忆并重新执行 |
| /convert_skill <memory_id> | 将历史记忆输出转换为 Markdown 技能 |
| /new_session | 创建新的会话 ID |
| exit / quit | 退出 CLI |

当前 CLI 的输出约定：

- 普通调用和 replay 都会先打印 Request ID。
- 控制台只输出简化阶段信息，例如 stage=planning_started。
- 详细的 LLM 请求、响应和 checkpoint 内容写入 logs/llm_trace.log。

### 5.2 AgentCore 对外接口

AgentCore 当前暴露的主要方法如下：

| 方法 | 作用 |
| --- | --- |
| invoke(query, session_id=None) | 执行一次完整 Agent 调用 |
| replay(memory_id, injected_features=None) | 重放历史记忆并重新执行 |
| convert_memory_to_skill(memory_id) | 将历史记忆转换为 Markdown 技能 |
| load_skill(skill_name) | 手动加载 Python 或 Markdown 技能 |
| load_mcp_server(server_ref) | 加载 MCP 配置并注册工具 |
| add_tool(tool) | 运行时动态添加工具 |
| start_session(session_id=None) | 新建或切换 session_id |
| get_last_request_id() | 获取最近一次调用生成的 request_id |

## 6. 主执行链

### 6.1 LangGraph 节点

AgentCore._build_graph() 当前构建的主节点如下：

| 节点名 | 内部函数 | 作用 |
| --- | --- | --- |
| planner | initial_planning | 全局特征提取、领域判断、记忆召回、会话主记忆初始化、任务拆解 |
| agent | call_model_subtask | 执行当前子任务、选择技能与工具、调用模型 |
| reflect_and_advance | reflect_and_advance | 验证结果并决定推进、重试或阻断 |
| tools | ToolNode(self.tools) | 执行模型发起的工具调用 |

主链路可以概括为：

START -> planner -> agent -> tools 或 reflect_and_advance -> agent 或 END

### 6.2 初始规划阶段

initial_planning 当前会做这些事：

1. 读取用户输入。
2. 提取全局关键词与一句话摘要。
3. 判断任务领域。
4. 基于全局关键词做摘要级记忆召回。
5. 写入一条会话主记忆记录。
6. 调用 TaskPlanner 生成 JSON 子任务数组。
7. 写入 planning_started / planning_completed 日志 checkpoint。

当前约束：

- 全局关键词最多保留 30 个。
- 会话主记忆记录会保存 request_id，结束时回写最终输出。

### 6.3 子任务执行阶段

call_model_subtask 是主流程里最重要的节点，当前逻辑如下：

1. 读取当前子任务描述。
2. 再次提取子任务关键词，最多保留 5 个。
3. 优先从已有摘要记忆中选相关项；不够时再次查询记忆库。
4. 统一调用 SkillManager.assign_capabilities_to_task() 选择 Prompt 技能和候选工具。
5. 将原始请求、当前子任务、记忆摘要、可选完整记忆、Markdown 技能正文、候选工具说明拼成 Prompt。
6. 如果存在工具，则先 bind_tools()，再通过 llm_manager.invoke() 调用模型。
7. 写入 subtask_started / subtask_llm_dispatch 等 checkpoint。

当前实现特点：

- 每个子任务最多挂一个 Markdown Prompt 技能。
- 每个子任务可以绑定多个工具。
- 工具包括 Python 技能工具和配置式 MCP 工具。

### 6.4 反思推进阶段

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

每轮反思之后，还会：

- 写步骤级记忆
- 更新 retry_counts
- 写 reflection_completed、subtask_advanced、agent_blocked、agent_completed 等 checkpoint

## 7. 日志与追踪

### 7.1 request_id 与 session_id

当前项目使用两层追踪标识：

- session_id：标识同一会话
- request_id：标识一次具体的 agent 调用

invoke() 每次都会生成新的 request_id，并把整个 graph 执行包在 LLMManager.request_scope() 中。因此同一次调用中产生的多次 LLM 请求、checkpoint 日志和记忆记录都可以通过同一个 request_id 追踪。

### 7.2 LLMManager 当前职责

llm_manager.py 当前除了管理模型实例，还负责日志与控制台输出分流：

- 文件日志：logs/llm_trace.log
- 控制台日志：仅输出 request_id + stage

当前已实现的日志能力：

- 统一的 LLM request / response / error 日志
- request_id 上下文注入
- 可截断的长文本日志
- checkpoint 日志
- console_event 简化阶段输出

当前控制台不会输出完整 Prompt 或模型返回详情；要看明细，需要按 request_id 去日志文件中检索。

## 8. 核心模块说明

### 8.1 AgentCore

AgentCore 是总协调器，负责：

- 初始化认知、规划、反思、记忆、技能、MCP 子系统
- 自动扫描 Python 技能目录
- 自动扫描 MCP 配置目录
- 构建并持有 LangGraph 图
- 对外提供 invoke、replay、convert_memory_to_skill 等接口
- 维护当前 session_id 与最近一次 request_id

### 8.2 CognitiveSystem

cognitive/feature_extractor.py 中的 CognitiveSystem 当前提供：

- extract_features(text, domain_hint="")：提取关键词和一句话摘要，要求模型返回 JSON
- determine_domain(text)：从内置领域树中选择一个领域标签

当前特征：

- 两个方法都通过 llm_manager.invoke() 调模型
- 失败时有降级逻辑
- 领域树仍为硬编码列表，不是动态领域图谱

### 8.3 TaskPlanner

cognitive/planner.py 中的 TaskPlanner 负责把输入拆成 JSON 数组子任务，每项至少包含：

- id
- description
- expected_outcome

如果 JSON 解析失败，会退化为单子任务执行。

### 8.4 Reflector

cognitive/reflector.py 中的 Reflector 负责比较“预期结果”和“实际结果”，让 LLM 决定后续动作。它是当前项目里“预测 -> 验证 -> 纠偏”闭环的直接落点。

### 8.5 MemoryManager

memory/memory_manager.py 中的 MemoryManager 基于 SQLite 实现持久化，当前 interactions 表包含：

- id
- conv_id
- request_id
- timestamp
- domain_label
- keywords
- summary
- raw_input
- raw_output
- large_file_path
- weight

当前主要方法：

| 方法 | 作用 |
| --- | --- |
| add_memory(...) | 新增一条记忆记录 |
| update_memory(...) | 更新现有记忆记录 |
| retrieve_memory(...) | 读取摘要级记忆并按匹配与权重排序 |
| load_full_memory(memory_id) | 读取完整记忆详情 |

当前实现特点：

- 数据库和备份目录从 config.json 读取并解析为项目内绝对路径
- 老数据库会自动补 request_id 列
- raw_input 超过 5000 字符时会写入外部备份文件
- retrieve_memory() 会在关键词命中时提升权重
- load_full_memory() 现在会返回 conv_id 和 request_id

### 8.6 SkillManager

skills_md/skill_parser.py 中的 SkillManager 已不只是 Markdown 解析器，而是统一能力注册中心。它现在同时负责：

- 自动扫描 skills_md/ 下的 Markdown 技能
- 解析 YAML Front-matter
- 注册 Python 工具技能元数据
- 注册 MCP 工具元数据
- 用统一的关键词阈值规则选择 Prompt 技能与工具

当前选择逻辑：

- Prompt 技能通过 find_best_skill() 选择
- 工具通过 find_relevant_tools() 排序并截断
- assign_capabilities_to_task() 同时返回 prompt_skill、tool_skills、tools

当前仍以关键词和简单匹配比例为主，不是语义向量检索。

### 8.7 MCPManager

mcp_servers/mcp_manager.py 当前是一个配置式适配层，不是真正的 MCP 协议客户端。它会：

- 读取 JSON / YAML 配置
- 校验 tools 列表
- 为每个工具生成 StructuredTool
- 支持 static_response 或 response_template 两种返回方式

这意味着当前所谓 MCP，更准确的说法是“按 MCP 风格配置生成本地 LangChain 工具”。

## 9. 配置与默认值

### 9.1 配置来源

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
| prompt_skill_min_overlap | Prompt 技能最小关键词重叠数 |
| prompt_skill_min_match_ratio | Prompt 技能最小匹配比例 |
| tool_skill_min_overlap | 工具最小关键词重叠数 |
| tool_skill_min_match_ratio | 工具最小匹配比例 |

### 9.2 当前默认模型

当前 config.json 的默认模型键是 local_ollama，对应：

- provider = ollama
- model = lfm2:latest
- base_url = http://localhost:11434

同时也预留了 remote_openai 配置项。

## 10. 测试与示例

当前 tests/test_skill_routing.py 提供最小回归覆盖，主要验证：

- 工具匹配需要满足最小匹配比例
- 弱相关工具不会被错误选中
- Prompt 技能匹配遵守阈值

当前 skills/test_skill.py 提供的示例工具包括：

- get_current_time
- calculate_sum
- get_mock_weather

## 11. 当前实现状态评估

### 11.1 已经打通的部分

- CLI -> AgentCore -> LangGraph 的可运行主链
- request_id / session_id 双层追踪
- 详细日志落文件、控制台阶段输出精简
- 特征提取、规划、反思、重试、阻断闭环
- 会话级和步骤级记忆写入与召回
- Python 工具、Markdown Prompt 技能、配置式 MCP 工具统一路由
- replay 与 memory -> skill 的辅助能力

### 11.2 仍未闭环的部分

- 真实 MCP 协议接入未实现，当前阶段也明确不处理
- 能力选择仍偏启发式，复杂任务下稳定性有限

### 11.3 工程风险点

- 认知与规划强依赖模型按要求返回 JSON，模型漂移时会触发降级逻辑
- 技能召回仍主要依赖关键词，长尾任务可能命中不足
- 自动加载技能和配置式 MCP 时仍有部分 print 输出，尚未完全收敛进统一日志系统

## 12. 与其他文档的关系

当前文档的角色是“源码事实总览”。

其他文档的定位如下：

- README.md：项目入口说明与快速阅读指引
- docs/requirements.md：目标态需求文档
- docs/agent.md：能力概念补充

因此，若这些文档与源码不一致，应优先相信源码，其次参考本文，再回头对照目标态文档。

## 13. 建议的阅读顺序

1. 先读本文，建立整体认知。
2. 再读 agent_core.py，理解主状态和主执行链。
3. 接着读 cli.py 与 llm_manager.py，理解入口、输出和追踪方式。
4. 然后读 cognitive/、memory/、skills_md/、mcp_servers/。
5. 最后再读 README.md、docs/requirements.md、docs/agent.md，对照现状和目标态差距。

## 14. 结论

llm_brain 当前仍是原型工程，但已经不只是最初的最小骨架。它现在具备了可追踪的执行链、基础记忆闭环、统一技能路由、配置式工具扩展和较清晰的模块边界。对后续接手者来说，最重要的不是把目标文档当作现状，而是先理解当前已经可靠落地的这些链路，再决定下一步要继续补的是协议能力、检索能力，还是更强的执行稳定性。