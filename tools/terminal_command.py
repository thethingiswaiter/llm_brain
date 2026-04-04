from typing import Any

from langchain_core.tools import tool

from mcp_servers.system_mcp_server import DEFAULT_TIMEOUT_SECONDS, execute_terminal_command


@tool
def bash(command: str, cwd: str = ".", timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, shell: str = "auto") -> dict[str, Any]:
    """安全执行终端命令的通用工具。可用于文件搜索、路径定位、读取环境信息、查看 git 状态、运行 Python 或测试命令等常见开发任务。

    这是一个覆盖面较广的兜底工具：当专用工具不适用、无法覆盖当前任务，或需要用安全命令直接完成操作时，可以优先考虑它。
    因为在当前 Windows 环境中, shell 实际连接 PowerShell, 应优先生成 PowerShell 兼容命令, 而不是仅适用于 bash 的语法。
    默认限制在工作区内，拦截危险命令与破坏性操作，并限制允许执行的命令前缀。
    """
    return execute_terminal_command(
        command=command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        allow_outside_workspace=True,
        allow_destructive=False,
        shell=shell,
    )


tools = [bash]
