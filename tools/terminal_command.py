from typing import Any

from langchain_core.tools import tool

from mcp_servers.system_mcp_server import DEFAULT_TIMEOUT_SECONDS, execute_terminal_command


@tool
def bash(command: str, cwd: str = ".", timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, shell: str = "auto") -> dict[str, Any]:
    """安全执行终端命令（opencode 风格 bash 工具）。默认限制在工作区、拦截危险命令、限制命令前缀。

    适用场景：读取环境信息、运行安全开发命令（例如 git status、python --version、pytest）。
    """
    return execute_terminal_command(
        command=command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        allow_outside_workspace=False,
        allow_destructive=False,
        shell=shell,
    )


tools = [bash]
