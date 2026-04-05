# llm_brain

基于 LangGraph 的命令行 Agent 原型。主线聚焦：

- 稳定的执行闭环：特征提取 -> 任务规划 -> 子任务执行 -> 反思推进
- 可观测与可恢复：request 级日志、聚合摘要、快照恢复
- 工程化迭代效率：默认快速回归，扩展能力按需启用

## 文档入口

- 项目总览：./docs/project_overview.md
- 项目定位（速览）：./docs/project_positioning.md
- 能力说明：./docs/agent.md
- 能力边界清单：./docs/capabilities_reference.md
- 后续路线：./docs/completeness_roadmap.md

建议阅读顺序：

1. docs/project_positioning.md
2. docs/project_overview.md
3. main.py + app/agent/core.py + app/cli/main.py
4. docs/completeness_roadmap.md

## 快速开始

1. 安装依赖：

```bash
python -m pip install -U -r requirements.txt
```

2. 按 config.json 配置模型（默认可用本地 Ollama）。

3. 启动终端界面：

```bash
python main.py
```

默认入口会优先启动 Textual TUI；如果当前环境没有安装 Textual，会自动回退到旧的行式 CLI。

## 测试策略

日常开发：

```bash
python tests/run_tests.py
```

全量回归（含慢集成）：

```bash
python tests/run_tests.py --all
```

兼容 unittest discover：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

当前基线（2026-04）：

- 快速回归：111 tests
- 全量回归：148 tests

## 项目结构

- app/: 主业务编排层（Agent 主链、CLI）
- core/: 基础设施层（配置、LLM 运行时与日志）
- cognitive/: 特征提取、规划、反思
- memory/: 记忆存储与召回
- tools/: Python 工具
- skills/: Markdown 技能
- mcp_servers/: MCP 接入与内置 MCP server
- tests/: 测试用例与分层测试入口

## MCP 定位（重要）

- MCP 维持可支持，但默认弱化存在感。
- AgentCore 默认不自动加载 MCP，仅在显式启用或专项场景使用。
- MCP 改动不应破坏主链稳定性，也不应拖慢默认回归。

## 常用 CLI 命令

- /load_tool <tool_name.py>
- /load_skill <skill_name.md>
- /load_mcp <config|server.py|stdio:command ...>
- /list_mcp
- /refresh_mcp <server_name|source>
- /unload_mcp <server_name|source>
- /list_snapshots <request_id>
- /resume_snapshot <request_id> [latest|index|stage|snapshot_file] [reroute]
- /request_summary <request_id>
- /recent_requests [limit] [status=...] [resumed] [attention] [since=30m]
- /request_rollup [limit] [status=...] [resumed] [attention] [since=30m]
- /retention_status
- /prune_runtime_data [apply]

## 说明

- docs/project_overview.md 以“当前源码事实”为准。
- docs/agent.md 主要讲能力概念，不等于当前全部实现。
- docs/completeness_roadmap.md 仅描述尚未完成事项。
