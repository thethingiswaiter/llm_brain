# llm_brain 代码结构评审与重构建议

## 1. 文档目的

本文档专门评估当前仓库的文件与目录布局是否合理、单个 Python 文件的职责是否清晰，以及如果要继续扩展，哪些地方应该优先拆分或保持不动。

它不是目标态设计文档，也不是路线图，而是基于当前源码事实给出的结构评审结论。

## 2. 总体结论

当前结构整体上是合理的，已经具备按子系统分目录的基本形态：

- CLI 入口独立
- Agent 主编排独立
- 认知、记忆、技能、MCP 各有单独目录
- 测试与文档目录已分开

这意味着项目不是“布局混乱不可维护”的状态，而是“核心骨架合理，但局部职责开始过载”的状态。

如果只看当前可运行性，现有布局还能继续承载一段时间；但如果后续还要继续扩展执行链、MCP、路由与观测能力，那么有几处边界应该尽早收敛，否则后面会进入“功能都能加，但改动风险越来越高”的阶段。

一句话判断：

- 目录结构总体合格
- 主编排文件过大
- 少量目录职责不够纯
- 下一步应做“局部结构化拆分”，而不是“大规模推倒重来”

## 3. 当前布局的优点

### 3.1 顶层目录职责基本成立

当前仓库已经自然形成了以下边界：

- `cognitive/`：特征提取、规划、反思
- `memory/`：SQLite 记忆管理
- `skills/`：Python 工具技能
- `skills_md/`：Markdown Prompt 技能解析与路由
- `mcp_servers/`：MCP 相关实现
- `tests/`：自动化回归测试
- `docs/`：项目文档

这种布局对后续维护是友好的，因为阅读者可以很快判断一个功能应该落在哪个目录。

### 3.2 认知模块拆分是清晰的

`cognitive/` 下的文件边界比较好：

- `feature_extractor.py` 负责关键词与摘要提取、领域判断
- `planner.py` 负责子任务拆解
- `reflector.py` 负责结果验证与推进动作判断
- `structured_output.py` 负责结构化解析与校验

这一层已经具备“一个文件围绕一个核心职责”的特征，后续没有必要再继续拆细。

### 3.3 测试目录当前已经按职责命名

测试文件已经从阶段性命名切回职责命名，例如：

- `test_core_reliability.py`
- `test_memory_quality.py`
- `test_cli_and_integration.py`
- `test_failure_and_migration.py`

这比依赖 roadmap 阶段命名更稳定，也更适合长期维护。

## 4. 当前最主要的结构问题

### 4.1 `agent_core.py` 明显过载

这是当前最需要处理的结构问题。

`agent_core.py` 现在已经同时承担了以下职责：

- AgentState 定义
- 子系统初始化
- 技能与 MCP 自动加载
- LangGraph 图构建
- invoke / replay / resume 对外入口
- 请求级取消与超时控制
- 快照持久化与恢复
- 请求摘要聚合与指标统计
- 工具包装、失败分类与再路由
- 运行时工具与 MCP 生命周期管理

这会带来几个直接问题：

- 文件阅读成本持续上升
- 改一个点容易影响不相干逻辑
- 测试很难只针对一个局部职责做最小验证
- 后续继续补功能时，所有东西都会自然堆回这个文件

当前它仍然能工作，但已经不再适合作为长期扩展中心。

### 4.2 `skills/` 目录职责不纯

当前 `skills/` 目录里同时存在：

- 运行时真实技能
- 示例技能
- 带测试性质的技能文件

而 `AgentCore` 当前会自动扫描整个 `skills/` 目录中的 Python 文件并加载工具。这意味着示例文件或测试性质文件一旦放在这个目录里，就会直接进入真实运行时能力集合。

这不是命名上的小问题，而是运行时边界不清的问题。

建议把以下两类文件移出运行时技能目录：

- 示例技能
- 测试/实验性质技能

更合适的位置是：

- `examples/skills/`
- `tests/fixtures/skills/`

### 4.3 `cli.py` 目前可用，但扩展点不稳定

`cli.py` 现在体量还不算失控，但已经是典型的命令长分支结构：

- 主循环里通过 `if / elif` 分发命令
- 每加一个命令都要继续修改主函数
- 参数解析、渲染和命令执行耦合在一起

这类结构短期内没问题，但命令继续增长时，维护体验会明显下降。

### 4.4 `llm_manager.py` 已有“第二个中心文件”的趋势

`llm_manager.py` 当前同时负责：

- 当前模型实例管理
- provider 切换
- request_scope 上下文管理
- 超时执行
- 结构化日志写入
- 控制台日志输出
- request_id 事件反查

这些职责彼此相关，但不完全是同一个层面。现在文件仍可接受，但如果后续继续补 provider、日志 sink、trace 上报、统计聚合，复杂度会继续上升。

### 4.5 `mcp_manager.py` 边界暂时成立，但已接近拆分点

`mcp_manager.py` 目前同时包含两类逻辑：

- 配置式 MCP 工具适配
- 真实 stdio transport 连接管理

当前因为 transport 只有 stdio，所以文件仍然勉强保持清晰；但如果后面继续补更完整的 transport 生命周期、更多 transport 类型或更细粒度健康检查，这里很容易膨胀成另一个大文件。

## 5. 哪些文件不必现在拆

### 5.1 `cognitive/` 下的模块不建议继续细拆

这层已经足够清晰，继续拆的收益不高。

### 5.2 `memory/memory_manager.py` 当前可以保留

虽然它已经不算小，但职责仍然集中在记忆存储与查询本身，没有明显跨层污染。现阶段更适合在原文件内继续保持收敛，而不是拆成多个文件。

### 5.3 `skills_md/skill_parser.py` 当前可以保留

这个文件承担“统一能力注册与关键词路由”职责，边界是自洽的。只要后续不把更重的检索逻辑继续堆进去，就不必急着拆。

## 6. 推荐的拆分顺序

### 6.1 第一优先级：拆 `agent_core.py`

建议采用“保留 AgentCore 外观、把内部职责拆出去”的方式，而不是直接重写。

推荐拆分方向：

- `agent_core.py`
  - 保留对外门面
  - 保留初始化与高层协调
- `agent_runtime.py`
  - `invoke`
  - request 生命周期管理
  - 超时等待与取消协调
- `agent_snapshots.py`
  - 状态序列化
  - 快照持久化
  - 快照解析与恢复
- `agent_observability.py`
  - request summary
  - recent requests
  - 指标聚合
- `agent_tools.py`
  - 工具安全包装
  - 失败分类
  - reroute / fallback 逻辑
- `agent_loading.py`
  - 自动加载 Python skills
  - 自动加载 MCP server
  - 手动加载入口

这样做的好处是：

- 外部 API 不用大改
- 现有测试大部分可以保留
- 可以逐步拆，而不是一次性高风险改造

### 6.2 第二优先级：清理 `skills/` 目录边界

建议把示例或测试技能迁出 `skills/`，避免自动加载逻辑把它们当成真实能力。

推荐目标布局：

- `skills/`：仅保留真实运行时技能
- `examples/skills/`：示例技能
- `tests/fixtures/skills/`：测试专用技能

### 6.3 第三优先级：把 `cli.py` 改成命令注册表

不一定要引入额外框架，但至少应该把：

- 命令定义
- 参数解析
- 命令处理函数

从主循环中拆开。

比较稳妥的方式是引入一个简单的命令表，例如：

- 命令名
- handler
- usage
- help 文本

这样新增命令时不必继续堆 `elif`。

### 6.4 第四优先级：轻量拆分 `llm_manager.py`

建议只做职责收敛，不做接口推翻。

可考虑拆成：

- `llm_manager.py`：对外门面
- `llm_logging.py`：结构化日志与事件读取
- `llm_runtime.py`：请求上下文、超时执行、取消检查
- `llm_factory.py`：provider 实例构造

### 6.5 第五优先级：等 MCP 再扩展时再拆 `mcp_manager.py`

当前不需要立刻处理，但如果未来继续推进 MCP 工程化，建议拆为：

- `mcp_manager.py`：统一门面
- `mcp_stdio_transport.py`：真实 stdio transport
- `mcp_config_adapter.py`：配置式工具适配

## 7. 不建议现在做的事

### 7.1 不建议立刻整体改造成复杂包结构

例如把整个仓库一次性重组为：

- `llm_brain/core/`
- `llm_brain/runtime/`
- `llm_brain/infra/`

这种重构理论上可以更整齐，但当前阶段收益不如成本高。现阶段更重要的是先把明显过载文件拆开，而不是做全面目录迁移。

### 7.2 不建议为了“看起来整齐”去合并已清晰的小模块

尤其不建议把 `cognitive/` 下的几个文件重新合回一个文件。它们当前的可读性是成立的。

## 8. 推荐的目标布局

在不大改公开接口的前提下，可以逐步演进为：

```text
.
├─ cli.py
├─ agent_core.py
├─ agent_runtime.py
├─ agent_snapshots.py
├─ agent_observability.py
├─ agent_tools.py
├─ agent_loading.py
├─ llm_manager.py
├─ llm_logging.py
├─ llm_runtime.py
├─ llm_factory.py
├─ cognitive/
├─ memory/
├─ skills/
├─ examples/
│  └─ skills/
├─ tests/
│  └─ fixtures/
│     └─ skills/
├─ skills_md/
├─ mcp_servers/
└─ docs/
```

这个目标布局的核心思想不是“目录越多越好”，而是：

- 让运行时边界更纯
- 让大文件职责更单一
- 让后续扩展时的落点更明确

## 9. 结论

当前项目结构总体上是可接受的，不需要推倒重来。

真正值得处理的重点只有两个：

1. `agent_core.py` 过载，应作为最高优先级拆分对象。
2. `skills/` 目录职责不纯，应尽快把示例/测试技能移出自动加载路径。

如果这两点先处理掉，项目的后续扩展成本会明显下降。之后再视需要处理 `cli.py`、`llm_manager.py` 和 `mcp_manager.py`，会比现在直接做全量重构更稳妥。