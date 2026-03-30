# llm_brain 项目总览

## 1. 文档目的

本文档面向后续接手该仓库的大模型，目标是用尽量少的歧义说明项目的当前实现、核心调用链、模块职责、扩展点，以及设计目标与现状之间的差距。

这份文档优先描述源码中已经存在的事实，其次才引用现有设计文档中的目标态能力。阅读时应始终区分以下两类信息：

- 当前实现：已经在代码中出现，并且能在主流程或辅助流程中找到调用路径。
- 设计目标：已写入需求文档，但尚未完全落地，或者仅有占位实现。

## 2. 一句话定义

llm_brain 是一个以 LangGraph 为执行骨架的通用 Agent 原型项目，核心思路是把用户请求拆解为细粒度子任务，再通过“特征提取 -> 任务规划 -> 子任务执行 -> 结果反思”的循环推进任务完成，并在过程中逐步接入记忆系统、技能系统和未来的 MCP 扩展能力。

## 3. 机器优先摘要

### 3.1 项目定位

- 项目类型：通用型 Agent 原型
- 运行形态：命令行交互式 Agent
- 编排核心：LangGraph 状态图
- 模型接入：Ollama / OpenAI
- 当前主入口：`cli.py` 中的 `start_cli`
- 当前核心协调器：`agent_core.py` 中的 `AgentCore`

### 3.2 当前真实主流程

1. 用户在 CLI 输入文本或管理命令。
2. 常规文本输入会调用 `agent.invoke(query)`。
3. Agent 先做全局特征提取与领域判断，再生成子任务计划。
4. Agent 逐个执行子任务，由统一能力路由同时挑选 Prompt 技能和候选工具。
5. 每次子任务执行后，会将实际结果与预期结果送入反思模块判断是否继续、重试或中止请求用户介入。
6. 所有子任务完成后，返回最后一条模型消息作为结果。

### 3.3 当前主要状态字段

Agent 的状态结构定义在 `AgentState` 中，核心字段如下：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `messages` | `list[BaseMessage]` | LangGraph 状态流中的消息历史 |
| `plan` | `List[Dict[str, Any]]` | 当前任务拆解出的子任务列表 |
| `current_subtask_index` | `int` | 当前执行到第几个子任务 |
| `reflections` | `List[str]` | 每一步反思记录 |
| `global_keywords` | `List[str]` | 全局任务特征关键字 |

### 3.4 当前已经具备的能力

- 基于 LangGraph 的状态图执行框架
- 基于 LLM 的特征提取、任务拆解与反思判断
- 基于 SQLite 的基础记忆落盘能力
- 基于 Python 文件自动扫描的工具技能加载
- 基于 Markdown Front-matter 的技能自动扫描与匹配能力
- 基于统一 SkillManager 的 Prompt 技能与工具能力联合注册和路由
- 基于历史记忆的重放与“记忆转技能”辅助接口
- 基于 CLI 的运行时技能加载入口
- 基于配置文件的最小 MCP 工具注册能力

### 3.5 当前未闭环或仅占位的能力

- MCP 服务器集成仍未接入真实协议连接
- 可信度评分体系仅存在于设计目标中

## 4. 项目结构概览

当前仓库可按职责分为以下几层：

| 路径 | 角色 |
| --- | --- |
| `cli.py` | 命令行入口，负责接收用户输入与管理命令 |
| `agent_core.py` | Agent 主编排器，负责构建 LangGraph 状态图 |
| `llm_manager.py` | 模型实例的统一管理与切换 |
| `config.py` / `config.json` | 配置加载与默认模型定义 |
| `cognitive/` | 认知子系统：特征提取、任务规划、反思验证 |
| `memory/` | 记忆持久化与历史重放支持 |
| `skills/` | Python 工具技能目录，按约定自动扫描加载 |
| `skills_md/` | Markdown 技能解析与匹配逻辑 |
| `docs/` | 需求说明、概念设计与本总览文档 |
| `mcp_servers/` | MCP 配置与适配层目录，当前含本地配置式示例 |

## 5. 运行入口与调用链

### 5.1 CLI 入口

项目的直接运行入口是 `cli.py`。`start_cli()` 会启动一个循环读取用户输入的命令行界面，支持两类输入：

- 管理命令：如 `/llm`、`/replay`、`/convert_skill`、`/load_mcp`
- 普通自然语言输入：交给 `agent.invoke()`

CLI 当前支持的主要命令语义如下：

| 命令 | 作用 | 当前状态 |
| --- | --- | --- |
| `/llm <provider> <model> [base_url] [api_key]` | 切换模型提供方与模型名 | 已实现 |
| `/load_skill <skill_name.py|skill_name.md>` | 动态加载本地 Python 或 Markdown 技能 | 已实现 |
| `/replay <memory_id> [injected features...]` | 重放历史记忆 | 已实现 |
| `/convert_skill <memory_id>` | 将记忆转为 Markdown 技能 | 已实现 |
| `/load_mcp <config_name.json|config_name.yaml|absolute_path>` | 加载 MCP 配置并注册工具 | 已实现 |

### 5.2 Agent 主执行链

主执行链定义在 `agent_core.py` 的 `AgentCore._build_graph()` 中，核心节点如下：

| 节点名 | 内部函数 | 作用 |
| --- | --- | --- |
| `planner` | `initial_planning` | 提取全局特征、判断领域、写入初始记忆、生成子任务计划 |
| `agent` | `call_model_subtask` | 执行当前子任务、装配技能上下文、调用 LLM |
| `reflect_and_advance` | `reflect_and_advance` | 验证预期与实际结果，并决定是否推进索引 |
| `tools` | `ToolNode(self.tools)` | 当模型发起工具调用时执行 Python 工具 |

图的实际流向可以概括为：

`START -> planner -> agent -> (tools | reflect_and_advance | END)`

补充说明：

- 当 `agent` 节点产出的最后一条消息包含 `tool_calls` 时，会转入 `tools` 节点。
- `tools` 执行后回到 `agent`，由模型继续处理工具结果。
- 若当前子任务已全部完成，流程结束。
- 若反思结果要求用户介入，会通过写入一条提示消息结束流程。

## 6. 执行流程细解

### 6.1 初始规划阶段

`initial_planning` 做了四件事：

1. 从最新一条用户消息中读取原始输入。
2. 调用 `CognitiveSystem.extract_features()` 生成关键字与一句话摘要。
3. 调用 `CognitiveSystem.determine_domain()` 生成领域标签。
4. 调用 `TaskPlanner.split_task()` 将用户请求拆成多个子任务。

同时，这一步会完成会话级记忆初始化：

- 为当前请求生成或继承会话 ID
- 先按全局关键词做摘要级记忆召回
- 写入一条会话主记录，后续在结束时回写最终输出与反思摘要
- 全局关键词最多截断到 30 个

### 6.2 子任务执行阶段

`call_model_subtask` 是当前主流程中最重要的执行节点，逻辑如下：

1. 读取 `plan[current_subtask_index]` 得到当前子任务。
2. 对子任务描述再次做特征提取，并将关键词限制在 5 个以内。
3. 调用 `SkillManager.assign_capabilities_to_task()` 统一挑选 Prompt 技能和候选工具。
4. 将记忆摘要、可选完整记忆、Markdown 技能正文和候选工具说明拼接到 Prompt 中。
5. 获取当前 LLM 实例；如果有候选工具，就通过 `bind_tools()` 绑定后再调用模型。
6. 返回模型响应消息。

当前实现中的关键约束：

- 每个子任务最多挂载一个 Markdown Prompt 技能，但可同时绑定多个候选工具。
- Python 工具和 MCP 工具已统一注册到 `SkillManager`，不再完全独立于 Markdown 技能路由。
- 子任务 Prompt 已接入摘要记忆，并在候选足够聚焦时补充完整记忆内容。

### 6.3 反思推进阶段

`reflect_and_advance` 负责闭环控制，输入包括：

- 当前子任务描述
- 当前子任务的 `expected_outcome`
- `agent` 节点输出的最后一条消息内容

反思模块会返回三项结果：

- `success`：本步是否成功
- `reflection_note`：分析说明
- `action`：`continue` / `retry` / `ask_user`

当前的推进规则如下：

- 成功，或虽然失败但允许继续：子任务索引加 1
- 失败且要求重试：索引不变，下一轮继续跑当前子任务
- 失败且需要用户介入：生成一条阻断消息并结束流程

当前 `retry` 已有次数上限；超过阈值后会升级为阻断并提示用户介入。

## 7. 核心模块说明

### 7.1 AgentCore

`AgentCore` 是项目中的总协调器，负责：

- 初始化认知、规划、反思、记忆、技能等子系统
- 自动扫描并加载 Python 技能
- 扫描 MCP 配置目录并注册配置式工具
- 构建 LangGraph 状态图
- 提供 `invoke()`、`replay()`、`convert_memory_to_skill()` 等外部接口

其中几个重要辅助接口含义如下：

| 方法 | 作用 |
| --- | --- |
| `invoke(query)` | 执行一次完整任务流程 |
| `replay(memory_id, injected_features=None)` | 读取历史记忆并重新注入执行 |
| `convert_memory_to_skill(memory_id)` | 将历史记忆输出转换成 Markdown 技能文件 |
| `add_tool(tool)` | 动态添加 Python 工具并重建图 |
| `load_mcp_server(server_ref)` | 加载 MCP 配置并注册为 LangChain 工具 |

### 7.2 LLMManager

`llm_manager.py` 提供统一的模型管理器，当前支持：

- Ollama：使用 `ChatOllama`
- OpenAI：使用 `ChatOpenAI`

配置来源是 `config.llm_config`，初始化时会根据 `config.json` 中的 `default_model` 选择默认模型。当前仓库默认模型是：

- `provider = ollama`
- `model = lfm2:latest`
- `base_url = http://localhost:11434`

### 7.3 CognitiveSystem

`cognitive/feature_extractor.py` 中的 `CognitiveSystem` 负责两类认知动作：

- `extract_features(text, domain_hint="")`
  - 使用 LLM 提取关键词和一句话摘要
  - Prompt 要求输出 JSON
  - 默认设计目标是 3 到 5 个关键词
- `determine_domain(text)`
  - 从预设领域树中选取一个领域标签
  - 当前领域树是硬编码在类中的基础列表

当前实现要点：

- 特征提取和领域判断都强依赖当前 LLM
- 解析失败时会回退到简单摘要或 `Other`
- 领域树仅是固定枚举，不是动态知识网

### 7.4 TaskPlanner

`cognitive/planner.py` 中的 `TaskPlanner` 负责把用户输入拆解为一组 JSON 子任务，每个子任务包含：

- `id`
- `description`
- `expected_outcome`

该模块体现了项目的核心设计思想：

- 复杂任务不直接一次性求解
- 每个子任务尽量能映射到一个工具或一个技能
- 每个子任务都要携带可被后续验证的预测结果

如果 JSON 解析失败，当前会退化为单子任务执行。

### 7.5 Reflector

`cognitive/reflector.py` 中的 `Reflector` 负责将“子任务目标”和“模型实际输出”做比对，并让 LLM 决定是否：

- 继续推进
- 重试当前步骤
- 请求用户介入

这是当前工程中“预测 - 验证 - 纠偏”设计的最直接落点。

### 7.6 MemoryManager

`memory/memory_manager.py` 中的 `MemoryManager` 提供基础持久化能力，底层使用 SQLite。表结构包含：

- 对话 ID
- 时间戳
- 领域标签
- 关键词
- 摘要
- 原始输入
- 原始输出
- 大文件路径
- 权重

当前提供的主要方法：

| 方法 | 作用 |
| --- | --- |
| `add_memory(...)` | 写入一条交互记录 |
| `retrieve_memory(...)` | 读取并排序记忆摘要，可按关键词和会话过滤 |
| `update_memory(...)` | 更新已有记忆记录的输出或摘要 |
| `load_full_memory(memory_id)` | 读取一条完整记忆详情 |

当前实现的事实与限制：

- 数据库路径和备份目录已改为通过 `config.json` 配置，并按项目根目录解析
- 当 `raw_input` 超过 5000 字符时，会落到外部文本文件
- `retrieve_memory()` 的两段式读取已接入主 Agent 流程，用于摘要召回和按需补全
- `load_full_memory()` 在查询不到记录时会返回空值，避免直接解包异常

### 7.7 SkillManager

`skills_md/skill_parser.py` 中的 `SkillManager` 现在既负责解析 Markdown 技能文件，也负责统一注册 Python/MCP 工具能力。Markdown 技能格式要求以 YAML Front-matter 开头，包含至少以下字段：

- `name`
- `confidence`
- `keywords`
- `description`
- `entry_node`

解析后会得到一个标准字典对象；与此同时，Python 工具和 MCP 工具也会被转成统一的能力元数据，并通过同一套关键词路由参与选择。

当前实现特征：

- 初始化时会自动扫描并加载 `skills_md/` 目录下的 Markdown 技能
- 技能正文会被完整拼接进 Prompt
- 技能目录已改为通过 `config.json` 配置，并按项目根目录解析
- Python 工具与 MCP 工具会同步注册到统一能力索引
- 当前仍以关键词匹配为主，不支持更复杂的向量召回
- 支持通过 `/load_skill` 命令按文件名增量加载单个 Markdown 技能

### 7.8 Python 技能目录

`skills/` 与 `skills_md/` 仍是两种不同的技能载体，但运行时已经收敛到统一的能力注册与选择链路：

- `skills/`：放 Python 工具函数，供 LangChain ToolNode 调用
- `skills_md/`：放 Markdown 技能说明，作为 Prompt 上下文注入
- 二者都会进入 `SkillManager` 的统一能力索引，由同一套路由逻辑参与子任务匹配

自动加载 Python 技能时，`AgentCore` 采用的约定是：

- 扫描 `skills/` 目录下的 `.py` 文件
- 跳过 `__` 开头文件
- 模块中必须导出一个名为 `tools` 的可迭代对象

当前已有两个示例：

- `sample_skill.py`：最小 Hello 工具示例
- `test_skill.py`：时间、求和、模拟天气工具

## 8. 配置与依赖

### 8.1 配置来源

配置层分为两部分：

- `config.json`：实际配置文件
- `config.py`：将 JSON 读入 `AppConfig` 与 `LLMConfig`

当前配置项包括：

| 配置项 | 说明 |
| --- | --- |
| `default_model` | 默认模型键名 |
| `models` | 模型配置集合 |
| `mcp_dir` | MCP 配置目录 |
| `skills_dir` | Python 技能目录 |
| `skills_md_dir` | Markdown 技能目录 |
| `memory_db_path` | SQLite 记忆库路径 |
| `memory_backup_dir` | 超长输入备份目录 |

### 8.2 可识别依赖

从源码可直接识别的关键依赖包括：

- `langgraph`
- `langchain_core`
- `langchain_ollama`
- `langchain_openai`
- `pydantic`
- `pyyaml`

补充说明：

- 仓库中的 `requirements.txt` 现已恢复为标准 UTF-8 文本，可直接作为依赖清单维护。
- 对后续模型而言，更可靠的依赖信息来源是实际 import 语句与运行代码路径。

## 9. 现有文档与代码的关系

当前仓库中的文档分为三层：

| 文件 | 角色 |
| --- | --- |
| `README.md` | 项目入口说明，强调当前源码事实与主要运行方式 |
| `docs/requirements.md` | 项目目标态 PRD，描述长期设计能力 |
| `docs/agent.md` | 对能力模型的概念性补充 |

需要特别注意：

- `docs/requirements.md` 和 `docs/agent.md` 不能被直接视为“当前功能说明”
- 它们更适合作为“为什么会有这些模块”和“未来要补到什么程度”的依据
- 本项目的真实现状，必须以源码实现为准

## 10. 当前实现状态评估

### 10.1 已经打通的部分

- CLI -> AgentCore -> LangGraph 的最小可运行链路
- 基于 LLM 的特征提取、任务拆解、反思判断
- 记忆摘要召回、步骤级写回与最终会话回写
- Python 工具自动加载与 ToolNode 接入
- Markdown Prompt 技能与 Python/MCP 工具的统一能力注册和路由
- 配置式 MCP 工具自动扫描与注册
- 历史记忆回放接口
- 历史记忆转 Markdown 技能接口

### 10.2 尚未闭环的部分

- 真实 MCP 协议连接仍未落地，当前是配置式本地适配层

### 10.3 工程风险点

- 若 Markdown 技能目录中没有内容，技能匹配实际上不起作用
- 当前技能路由仍以关键词匹配为主，复杂任务下可能需要更强的召回策略

## 11. 设计目标与现状差距

结合 `docs/requirements.md` 与 `docs/agent.md`，当前代码已经体现出设计方向，但多数能力仍处于早期阶段。

### 11.1 已体现设计意图的部分

- 通过特征提取减少上下文冗余
- 通过子任务拆解降低单步复杂度
- 通过 expected outcome 建立预测与验证机制
- 通过记忆持久化和 replay 为历史重演做接口准备
- 通过统一能力注册，把 Markdown 技能与工具能力收敛到同一条路由链

### 11.2 尚未完成的目标态能力

- 领域空间的系统化建模与聚类
- 可信度评分与来源分级
- 记忆与技能的双向互转闭环
- 更稳定的技能召回与任务路由
- 真正可用的 MCP 集成

## 12. 给后续大模型的阅读建议

如果后续要继续开发或分析这个项目，推荐按以下顺序阅读：

1. 先读本文，建立对项目定位与边界的整体认识。
2. 再读 `agent_core.py`，把握真实的执行主链和状态结构。
3. 接着读 `cli.py`、`config.py`、`llm_manager.py`，了解入口、配置与模型层。
4. 然后按子系统阅读 `cognitive/`、`memory/`、`skills_md/`。
5. 最后再回看 `docs/requirements.md` 与 `docs/agent.md`，对照未来目标补齐理解。

## 13. 结论

llm_brain 当前不是一个功能完备的生产级 Agent 平台，而是一个围绕“认知、规划、反思、记忆、技能”这些核心概念搭出的可运行原型。它的最大价值不在于现阶段功能覆盖度，而在于已经明确了主执行骨架，并把未来要扩展的能力边界拆成了较清晰的模块。对后续接手者来说，最重要的不是把设计文档当作既成事实，而是基于当前源码判断：哪些链路已经打通，哪些能力只是接口、占位或目标态设想。