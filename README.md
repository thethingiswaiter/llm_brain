# llm_brain

基于 LangGraph 的通用 Agent 原型项目，当前主线聚焦于“特征提取 -> 任务规划 -> 子任务执行 -> 反思推进”的执行闭环，并逐步接入记忆、技能与后续 MCP 扩展能力。

## 文档入口

- 项目总览：./docs/project_overview.md
- 架构问题清单：./docs/architecture_issues.md
- 需求目标文档：./docs/requirements.md
- 能力概念说明：./docs/agent.md

如果你是第一次接手该仓库，建议按以下顺序阅读：

1. 先看 `docs/project_overview.md`，快速建立整体认知。
2. 再看 `agent_core.py`、`cli.py`、`llm_manager.py`，理解真实调用链。
3. 然后按子系统查看 `cognitive/`、`memory/`、`skills_md/`。
4. 最后对照 `docs/requirements.md` 与 `docs/agent.md`，理解目标态设计。

## 当前项目重点

- 主入口：`cli.py`
- 主编排器：`agent_core.py`
- 核心执行框架：LangGraph 状态图
- 认知子系统：`cognitive/`
- 记忆持久化：`memory/`
- Python 工具技能：`skills/`
- Markdown 技能系统：`skills_md/`

## 当前状态

项目已具备最小可运行骨架，但仍是原型阶段。已经打通 CLI 到 Agent 的执行主链，并具备基础的任务拆解、反思、记忆落盘、技能挂载以及最小 MCP 配置式接入能力。当前已经完成路径配置化、Markdown 技能自动扫描加载、记忆闭环、工具筛选、重试保护、UTF-8 依赖清单修复，以及基于统一 SkillManager 的 Markdown 技能与 Python/MCP 工具联合路由。

## 运行说明

1. 根据 `config.json` 准备模型配置，默认走本地 Ollama。
2. 安装项目依赖。
3. 运行 `python cli.py` 启动命令行交互。

CLI 中可使用 `/load_skill <skill_name.py|skill_name.md>` 动态加载本地技能文件。
CLI 中也可使用 `/load_mcp <config_name.json|config_name.yaml|absolute_path>` 加载 MCP 配置文件，例如仓库内置的 `demo_mcp.json`。

## 补充说明

- `docs/project_overview.md` 以“当前源码事实”为主。
- `docs/requirements.md` 和 `docs/agent.md` 主要描述目标态，不应直接视为现状功能清单。
