# 当前技能、Tools、MCP 能力边界说明

这份文档只回答一件事：当前仓库里到底有哪些能力，它们各自能做什么，边界在哪里，默认有没有生效。

目标读者：后续继续开发 agent 主链、skill 路由、tool 规划、MCP 接入的人。

## 1. 总体结论

当前能力来源分成三层：

1. Prompt 技能
   当前 skills 目录为空，运行时默认没有任何 Markdown prompt skill 生效。
2. Python tools
   当前默认会自动加载 tools 目录下的 Python 工具文件，因此这是现在主链里默认可用的能力来源。
3. MCP tools
   仓库内已经支持两类 MCP：配置式 MCP 和 stdio MCP server，但 AgentCore 默认不自动加载 MCP，需要显式 `/load_mcp` 才会生效。

## 2. 当前生效状态

### 2.1 Prompt 技能

- 当前状态：无
- 目录位置：skills/
- 默认加载：会自动扫描 skills/，但当前目录为空，所以没有实际生效的 prompt skill
- 开发影响：
  当前规划和执行阶段都不能依赖自定义 Markdown skill 来承载“问题特化流程”，这也是你之前感觉很多特定问题逻辑容易落回 core 的原因之一。

### 2.2 Python tools

- 当前状态：默认自动加载
- 目录位置：tools/
- 自动加载规则：
  - 自动扫描 tools/ 下的 `.py` 文件
  - 跳过 `__*.py`
  - 跳过 `test_*.py`
  - 跳过 `sample_*.py`
- 当前默认可用文件：
  - tools/langchain_common_tools.py
  - tools/terminal_command.py

### 2.3 MCP

- 当前状态：仓库内可用，但默认不自动加载
- 目录位置：mcp_servers/
- 默认加载：关闭
- 启用方式：
  - `/load_mcp demo_mcp.json`
  - `/load_mcp system_mcp_server.py`
  - `/load_mcp stdio:...`
- 当前仓库内可加载来源：
  - mcp_servers/demo_mcp.json
  - mcp_servers/system_mcp_server.py

## 3. 所有能力共享的运行时治理

无论是 Python tool 还是 MCP tool，只要被 AgentCore 注册进来，都会先经过统一运行时包装。这个包装层非常重要，因为它决定了“工具在 agent 里真正有什么权限和限制”。

统一运行时行为：

- 参数预校验
  - 基于 args_schema 做 schema 校验
  - 对部分字段名做启发式校验，例如 city/date/time 一类参数
- 超时控制
  - 单次 tool 调用受 `tool_invoke_seconds` 限制
- 取消控制
  - request 被取消时，tool 会协作式停止
- 运行态跟踪
  - 每个 tool run 有 toolrun_id，可出现在 detached tool 观测里
- 失败归一化
  - 失败会被标准化为 invalid_arguments / timeout / dependency_unavailable / execution_error / no_output / cancelled 等类型
- 空输出治理
  - 对 bash 这类命令型工具，如果 exit_code 成功但 stdout/stderr 都为空，会被归一成 retryable 的 no_output，而不是直接算成功
- 日志与审计
  - tool_started / tool_succeeded / tool_failed / tool_rejected / tool_cancelled / tool_detached 都会进入 checkpoint 与日志

开发上你应该默认认为：

- “tool 描述里写的能力”不等于“agent 中实际能稳定使用的能力”
- 真正的行为边界还受到 tool runtime 包装层约束

## 4. Prompt 技能现状

### 4.1 当前没有已生效的 Markdown 技能

- skills/ 当前为空
- 因此运行时没有任何 prompt skill 会参与规划或执行

### 4.2 仓库里有示例 skill，但不会自动生效

示例位置：examples/skills/sample_skill.py

能力：

- 提供一个 `sample_hello(name)` 示例工具，返回简单问候语

边界：

- 只是示例，不在 tools/ 目录下
- 不会被默认 auto-load
- 名称本身也带有 sample 语义，不属于生产能力集合

权限：

- 无特殊权限，仅字符串处理

开发建议：

- 如果你要把“针对某类问题的专门处理流程”下沉到技能层，应该新增到 skills/ 下的 Markdown skill 或 tools/ 下的正式工具，而不是继续复用 examples/ 下的占位文件

## 5. Python tools 详细说明

## 5.1 get_current_time

- 来源：tools/langchain_common_tools.py
- 能力：返回中国时区当前时间字符串
- 输入：无
- 输出：字符串时间，例如 `YYYY-MM-DD HH:MM:SS CST`
- 边界：
  - 只能返回当前时间
  - 不提供时区切换、日期解析、历史时间计算
- 权限：
  - 只读
  - 不访问文件系统
  - 不执行命令

适用场景：

- 用户问“现在几点”
- 给输出附时间戳

## 5.2 calculator

- 来源：tools/langchain_common_tools.py
- 能力：安全计算数学表达式
- 输入：
  - `expression: str`
- 支持运算：
  - `+ - * / // % **`
  - 括号
  - 一元正负号
- 边界：
  - 只允许 AST 白名单中的纯数学表达式
  - 不允许变量、函数调用、属性访问、导入、任意 Python 代码
- 权限：
  - 纯内存计算
  - 不访问文件系统
  - 不执行命令

适用场景：

- 简单数学计算
- 需要高安全性的表达式求值

## 5.3 list_directory

- 来源：tools/langchain_common_tools.py
- 能力：列出任意本地目录项
- 输入：
  - `path: str = "."`
- 输出：
  - 路径
  - entries 列表
  - entry_count
  - truncated 标记
- 边界：
  - 允许绝对路径直接访问本地目录
  - 相对路径默认从工作区根目录解析
  - path 不存在或不是目录会失败
  - 最多返回前 500 个目录项
- 权限：
  - 本地只读目录访问
  - 不读取文件正文

适用场景：

- 看当前目录有哪些文件
- 做目录级统计或浏览

## 5.4 read_text_file

- 来源：tools/langchain_common_tools.py
- 能力：读取任意本地 UTF-8 文本文件指定行范围
- 输入：
  - `path: str`
  - `start_line: int = 1`
  - `end_line: int = 200`
- 边界：
  - 允许绝对路径直接读取本地文件
  - 相对路径默认从工作区根目录解析
  - 只支持 UTF-8 文本文件
  - 非文本或编码错误会失败
  - 内容预览最多约 12000 字符
- 权限：
  - 本地只读文件访问

适用场景：

- 查看源码片段
- 读取配置文件、文档、测试文件内容

## 5.5 grep_text

- 来源：tools/langchain_common_tools.py
- 能力：在任意本地路径下递归搜索文本
- 输入：
  - `query: str`
  - `path: str = "."`
  - `max_results: int = 20`
- 边界：
  - query 不能为空
  - 允许绝对路径直接搜索本地路径
  - 相对路径默认从工作区根目录解析
  - 只按大小写不敏感子串匹配
  - 不是正则搜索
  - 最多返回 100 条结果
  - 遇到不可解码文件会跳过
- 权限：
  - 本地只读文本搜索

适用场景：

- 查关键字
- 代码定位
- 文本存在性检查

## 5.6 write_text_file

- 来源：tools/langchain_common_tools.py
- 能力：写入 UTF-8 文本文件
- 输入：
  - `path: str`
  - `content: str`
  - `overwrite: bool = False`
  - `append: bool = False`
- 边界：
  - 默认仅允许写入工作区根目录内路径
  - 可通过 CLI 临时授权额外可写目录
  - 也可通过配置 `tools.write.extra_roots` 预置额外可写目录
  - 目标不能是目录
  - 默认不覆盖已存在文件
  - 需要显式 `overwrite=True` 或 `append=True`
- 权限：
  - 工作区写权限
  - 临时或配置授予的额外目录写权限
  - 可创建父目录
  - 未授权目录不可写

风险级别：

- 当前 Python tools 里唯一明确的写工具

适用场景：

- 生成结果文件
- 追加日志或报告

开发提示：

- 如果后续还要扩展写类工具，最好单独在路由和规划里标出“写操作风险”而不是把它混在通用只读工具里

## 5.7 json_query

- 来源：tools/langchain_common_tools.py
- 能力：读取任意本地 JSON 文件并按 key_path 取值
- 输入：
  - `path: str`
  - `key_path: str = ""`
- 边界：
  - 允许绝对路径直接读取本地 JSON 文件
  - 相对路径默认从工作区根目录解析
  - 文件必须是合法 JSON
  - key_path 按 `a.b.0.c` 形式逐层解析
  - 不支持 JSONPath 标准语法
- 权限：
  - 本地只读文件访问

适用场景：

- 查 config.json 的某个字段
- 查看结构化测试夹具内容

## 5.8 bash

- 来源：tools/terminal_command.py
- 底层实现：mcp_servers/system_mcp_server.py 的 `execute_terminal_command`
- 能力：执行 allowlist 内的终端命令，是一个覆盖面最广的兜底工具
- 输入：
  - `command: str`
  - `cwd: str = "."`
  - `timeout_seconds: int`
  - `shell: str = "auto"`

### bash 的真实边界

- 允许的命令前缀必须在 allowlist 中
  当前默认包括：
  - bash
  - cat
  - cd
  - cmd
  - dir
  - echo
  - findstr
  - git
  - get-childitem
  - get-content
  - get-location
  - hostname
  - ls
  - more
  - npm
  - npx
  - pip
  - pnpm
  - poetry
  - pwsh
  - pytest
  - pwd
  - py
  - python
  - python3
  - uv
  - systeminfo
  - type
  - ver
  - where
  - whoami
  - yarn
- 禁止 shell operator：
  - `&&`
  - `||`
  - `;`
  - 换行
- 禁止破坏性标记：
  - `rm -rf`
  - `remove-item -recurse -force`
  - `rd /s /q`
  - `del /f /s /q`
  - `format`
  - `diskpart`
  - `shutdown`
  - `reboot`
  - `mkfs`
  - `git reset --hard`
- shell 会被规范到 `powershell/cmd/bash/sh/auto`
- 有超时限制
- 输出会截断
- 所有执行会写审计日志

### bash 的权限范围

这里要特别注意：

- 这个工具不是严格的“仅工作区内文件工具”
- 当前在 Python tool 包装里，`allow_outside_workspace=True`
- 这意味着 cwd 可以被设置到工作区外
- 而且命令本身只受 allowlist 前缀约束，不会逐个解析命令参数里的路径

也就是说，bash 的权限模型是：

- 不是完全自由 shell
- 但也不是严格工作区沙箱
- 它是“允许前缀 + 禁止危险操作 + 审计 + 超时”的受限命令执行工具

风险级别：

- 当前所有默认能力里，bash 是权限最大、覆盖面最广、也最需要谨慎规划的工具

适用场景：

- 文件搜索
- 路径定位
- git 状态检查
- Python / pytest / package manager 命令
- Windows PowerShell 环境下的诊断类命令

开发建议：

- 需要严格只读、严格工作区边界时，优先不要走 bash
- 只有专用工具不够时，再把 bash 作为兜底能力

## 6. MCP 详细说明

## 6.1 MCP 总体模型

当前 MCPManager 支持两类 MCP：

1. 配置式 MCP
   - 从 `.json/.yaml` 读取工具定义
   - 本质是把配置转换成 StructuredTool
2. stdio MCP
   - 通过外部进程启动 MCP server
   - 动态枚举远端工具
   - 用输入 schema 构造本地代理 tool

默认边界：

- AgentCore 默认不自动加载 MCP
- MCP 属于显式扩展能力，不是主链默认能力

## 6.2 demo_mcp.json

- 类型：配置式 MCP
- 默认加载：否
- 启用方式：`/load_mcp demo_mcp.json`
- 权限模型：
  - 无本地文件权限
  - 无命令执行权限
  - 无外部网络权限
  - 本质是静态模板返回

### demo_mcp.lookup_release_note

- 能力：按 component 返回一段 mock release note
- 输入：
  - `component: string` required
- 边界：
  - 不查询真实发布记录
  - 不访问文件、数据库、服务
  - 只是模板字符串拼接
- 权限：无副作用

### demo_mcp.summarize_incident_status

- 能力：按 service 和 priority 返回一段 mock incident 摘要
- 输入：
  - `service: string` required
  - `priority: string` optional，默认 `P3`
- 边界：
  - 不查询真实故障系统
  - 只是模板响应
- 权限：无副作用

开发定位：

- 主要用于验证 MCP 接入链路
- 不属于真实生产能力

## 6.3 system_mcp_server.py

- 类型：stdio MCP server
- 默认加载：否
- 启用方式：`/load_mcp system_mcp_server.py`
- 权限模型：
  - 比 demo_mcp 强很多
  - 具备本地系统信息读取、路径检查、受限终端执行能力
  - 所有调用有审计日志

它暴露的 MCP tools 如下。

### get_system_info

- 能力：读取本机系统信息
- 输出内容包括：
  - hostname
  - user
  - platform/system/release
  - python_version
  - cpu_count
  - workspace_root
  - 当前时间
  - 时区
  - 磁盘空间
  - audit_log_path
  - allowed_roots
  - allowed_command_prefixes
- 边界：
  - 只读系统信息
  - 不写文件
  - 不执行命令
- 权限：
  - 可读宿主机环境信息

### get_mcp_security_policy

- 能力：返回当前 MCP 安全策略
- 输出内容包括：
  - workspace_root
  - audit_log_path
  - allowed_roots
  - allowed_command_prefixes
  - destructive_command_markers
  - disallowed_shell_operators
  - timeout 配置
- 边界：
  - 只读策略暴露
- 权限：
  - 不执行命令
  - 不读文件正文

开发价值：

- 这是最适合做“能力自解释”的 MCP 工具之一，后续如果你要把 tool/MCP 限制显式展示给用户，可以直接复用这类模式

### inspect_file_system_path

- 能力：检查路径状态，可选文件预览
- 输入：
  - `path`
  - `include_preview`
  - `preview_lines`
  - `allow_outside_workspace`
- 边界：
  - 默认受 allowed_roots 约束
  - 可以检查文件和目录
  - 文件预览只读，且有预览长度限制
  - 目录项最多返回 200 个
  - 文件预览行数最多 80 行
- 权限：
  - 路径检查权限
  - 默认是受 allowed_roots 限制的只读文件系统访问

### execute_terminal_command

- 能力：执行受限命令，能力与 bash 底层基本一致
- 输入：
  - `command`
  - `cwd`
  - `timeout_seconds`
  - `allow_outside_workspace`
  - `allow_destructive`
  - `shell`
- 边界：
  - 受 allowlist 前缀约束
  - 默认禁止 destructive marker
  - 禁止 shell operators
  - 有审计、有超时、有输出截断
- 权限：
  - 是整个 system MCP 里权限最大的一项
  - 如果 `allow_outside_workspace=True`，cwd 可超出工作区
  - 但依旧不是 unrestricted shell

## 7. 当前最重要的边界区分

开发时最容易混淆的不是“有没有这个 tool”，而是“它到底处在哪个权限层”。

你可以按下面理解：

### A. 纯计算/纯模板层

- get_current_time
- calculator
- demo_mcp.*

特征：

- 基本无副作用
- 不碰文件系统
- 不碰命令执行

### B. 本地只读层

- list_directory
- read_text_file
- grep_text
- json_query
- inspect_file_system_path（默认模式）

特征：

- 主要围绕任意本地路径读取
- 相对路径仍默认从工作区解析
- 适合做规划时的安全优先选择

### C. 受控写层

- write_text_file

特征：

- 有明确副作用
- 默认工作区可写
- 可临时或静态扩展到额外目录
- 需要和只读工具区分看待

### D. 受限命令执行层

- bash
- execute_terminal_command

特征：

- 覆盖面最大
- 风险也最大
- 虽然不是 unrestricted shell，但明显比工作区只读工具更敏感

## 8. 当前真正让开发混乱的点

你现在会糊涂，根因主要有四个：

1. skills 当前是空的
   很多“问题特化流程”没有承载层，只能往 core 或 description 上挤。
2. tool 描述长期是非结构化文本
   模型要自己从 description 猜边界，稳定性差。
3. bash 和 execute_terminal_command 权限明显高于其他工具
   但如果只看名字，开发时很容易把它们当成普通工具。
4. MCP 默认不自动加载
   仓库里“存在的能力”和“当前会参与规划的能力”不是一回事。

## 9. 开发建议

如果你要继续沿着“能力清晰、便于开发”的方向推进，建议按这个顺序做：

1. 先把 skills/ 补起来
   把你已经反复遇到的具体问题流程沉淀成 Markdown skills，而不是继续往 planner/core 塞分支。
2. 给 tool 增加显式元数据字段
   比如：
   - permission_level: readonly / write / command / system
   - workspace_scoped: true/false
   - side_effects: none / file_write / command_exec
   - risk_level: low / medium / high
3. 在 CLI 里提供一个能力总览命令
   例如 `/list_capabilities`，把当前已加载的 skills/tools/mcp 和它们的边界直接打印出来。
4. 在规划阶段优先使用结构化限制，而不是 description 文本
   当前已经开始把 args/constraints 注入 planner，下一步应该继续把权限级别也结构化注入。

## 10. 一句话总结

当前主链默认依赖的是 Python tools，不是 skills，也不是 MCP。

其中：

- skills 目前为空
- MCP 默认未加载
- 真正最强的默认能力是 bash
- 真正最安全的默认能力是本地只读工具组

如果后面你想把系统继续做稳，核心不是再加更多特判，而是把“能力、边界、权限”继续从描述文本推进成结构化元数据。