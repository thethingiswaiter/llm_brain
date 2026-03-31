import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import config

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None


SERVER_NAME = "llm-brain-system-tools"
AUDIT_LOG_PATH = Path(config.resolve_path(config.audit_log_dir)) / "system_mcp_audit.jsonl"
DEFAULT_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 12000
MAX_DIRECTORY_ENTRIES = 200
MAX_PREVIEW_LINES = 80
DEFAULT_ALLOWED_COMMAND_PREFIXES = [
    "cat",
    "cd",
    "dir",
    "echo",
    "findstr",
    "get-childitem",
    "get-content",
    "get-location",
    "hostname",
    "ls",
    "more",
    "pwd",
    "py",
    "python",
    "systeminfo",
    "type",
    "ver",
    "where",
    "whoami",
]
DESTRUCTIVE_COMMAND_MARKERS = [
    "rm -rf",
    "remove-item -recurse -force",
    "rd /s /q",
    "del /f /s /q",
    "format ",
    "diskpart",
    "shutdown ",
    "reboot",
    "mkfs",
    "git reset --hard",
]


def _get_allowed_root_strings() -> list[str]:
    extra_roots = os.getenv("LLM_BRAIN_MCP_ALLOWED_ROOTS", "").strip()
    allowed_roots = [str(WORKSPACE_ROOT.resolve())]
    if extra_roots:
        for raw_root in extra_roots.split(os.pathsep):
            cleaned = raw_root.strip()
            if cleaned:
                allowed_roots.append(str(Path(cleaned).resolve()))
    return allowed_roots


def _get_allowed_root_paths() -> list[Path]:
    return [Path(item) for item in _get_allowed_root_strings()]


def _get_allowed_command_prefixes() -> list[str]:
    configured = os.getenv("LLM_BRAIN_MCP_ALLOWED_COMMANDS", "").strip()
    if not configured:
        return list(DEFAULT_ALLOWED_COMMAND_PREFIXES)
    return [item.strip().lower() for item in configured.split(",") if item.strip()]


def _extract_command_prefix(command: str) -> str:
    stripped = command.strip()
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].lower()


def _write_audit_event(tool_name: str, payload: dict[str, Any]) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        **payload,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(event, ensure_ascii=True) + "\n")


def _truncate_text(value: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    hidden_chars = len(value) - limit
    return f"{value[:limit]}\n...<truncated {hidden_chars} chars>", True


def _resolve_path(path_value: str | None, allow_outside_workspace: bool = False) -> Path:
    candidate = Path(path_value or ".")
    resolved = candidate if candidate.is_absolute() else (WORKSPACE_ROOT / candidate)
    resolved = resolved.resolve()
    if allow_outside_workspace:
        return resolved

    allowed_roots = _get_allowed_root_paths()
    for allowed_root in allowed_roots:
        if resolved == allowed_root or allowed_root in resolved.parents:
            return resolved
    allowed_root_strings = [str(path) for path in allowed_roots]
    raise ValueError(f"Path is outside allowed roots: {resolved}. Allowed roots: {allowed_root_strings}")


def _is_command_allowed(command: str) -> tuple[bool, str, list[str]]:
    command_prefix = _extract_command_prefix(command)
    allowed_prefixes = _get_allowed_command_prefixes()
    if command_prefix in allowed_prefixes:
        return True, command_prefix, allowed_prefixes
    return False, command_prefix, allowed_prefixes


def get_mcp_security_policy() -> dict[str, Any]:
    return {
        "ok": True,
        "workspace_root": str(WORKSPACE_ROOT),
        "audit_log_path": str(AUDIT_LOG_PATH),
        "allowed_roots": _get_allowed_root_strings(),
        "allowed_command_prefixes": _get_allowed_command_prefixes(),
        "destructive_command_markers": list(DESTRUCTIVE_COMMAND_MARKERS),
    }


def _looks_destructive(command: str) -> str:
    normalized = " ".join(command.strip().lower().split())
    for marker in DESTRUCTIVE_COMMAND_MARKERS:
        if marker in normalized:
            return marker
    return ""


def execute_terminal_command(
    command: str,
    cwd: str = ".",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allow_outside_workspace: bool = False,
    allow_destructive: bool = False,
) -> dict[str, Any]:
    if not isinstance(command, str) or not command.strip():
        result = {"ok": False, "error": "Command must be a non-empty string."}
        _write_audit_event("execute_terminal_command", result)
        return result

    command_allowed, command_prefix, allowed_prefixes = _is_command_allowed(command)
    if not command_allowed:
        result = {
            "ok": False,
            "blocked": True,
            "reason": f"Command prefix is not in allowlist: {command_prefix or '<empty>'}",
            "command": command,
            "command_prefix": command_prefix,
            "allowed_command_prefixes": allowed_prefixes,
        }
        _write_audit_event("execute_terminal_command", result)
        return result

    destructive_marker = _looks_destructive(command)
    if destructive_marker and not allow_destructive:
        result = {
            "ok": False,
            "blocked": True,
            "reason": f"Command matched destructive marker: {destructive_marker}",
            "command": command,
        }
        _write_audit_event("execute_terminal_command", result)
        return result

    try:
        resolved_cwd = _resolve_path(cwd, allow_outside_workspace=allow_outside_workspace)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "command": command, "cwd": cwd}
        _write_audit_event("execute_terminal_command", result)
        return result

    try:
        completed = subprocess.run(
            command,
            cwd=str(resolved_cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate_text(exc.stdout or "")
        stderr, stderr_truncated = _truncate_text(exc.stderr or "")
        result = {
            "ok": False,
            "timeout": True,
            "command": command,
            "cwd": str(resolved_cwd),
            "timeout_seconds": timeout_seconds,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        _write_audit_event("execute_terminal_command", result)
        return result
    except Exception as exc:
        result = {
            "ok": False,
            "command": command,
            "cwd": str(resolved_cwd),
            "error": str(exc),
        }
        _write_audit_event("execute_terminal_command", result)
        return result

    stdout, stdout_truncated = _truncate_text(completed.stdout or "")
    stderr, stderr_truncated = _truncate_text(completed.stderr or "")
    result = {
        "ok": completed.returncode == 0,
        "command": command,
        "command_prefix": command_prefix,
        "allowed_command_prefixes": allowed_prefixes,
        "cwd": str(resolved_cwd),
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    _write_audit_event("execute_terminal_command", result)
    return result


def collect_system_info() -> dict[str, Any]:
    disk_root = Path(WORKSPACE_ROOT.anchor or WORKSPACE_ROOT.drive or "/")
    disk_usage = shutil.disk_usage(disk_root)
    result = {
        "ok": True,
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "workspace_root": str(WORKSPACE_ROOT),
        "current_time_utc": datetime.now(timezone.utc).isoformat(),
        "disk_total_bytes": disk_usage.total,
        "disk_used_bytes": disk_usage.used,
        "disk_free_bytes": disk_usage.free,
        "audit_log_path": str(AUDIT_LOG_PATH),
        "allowed_roots": _get_allowed_root_strings(),
        "allowed_command_prefixes": _get_allowed_command_prefixes(),
    }
    _write_audit_event("get_system_info", {"ok": True})
    return result


def inspect_file_system_path(
    path: str,
    include_preview: bool = False,
    preview_lines: int = 20,
    allow_outside_workspace: bool = False,
) -> dict[str, Any]:
    try:
        resolved = _resolve_path(path, allow_outside_workspace=allow_outside_workspace)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "path": path}
        _write_audit_event("inspect_file_system_path", result)
        return result

    if not resolved.exists():
        result = {"ok": False, "error": f"Path not found: {resolved}", "path": path, "resolved_path": str(resolved)}
        _write_audit_event("inspect_file_system_path", result)
        return result

    stat = resolved.stat()
    result: dict[str, Any] = {
        "ok": True,
        "path": path,
        "resolved_path": str(resolved),
        "name": resolved.name,
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }

    if resolved.is_dir():
        entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
        result["entry_count"] = len(entries)
        result["entries"] = [
            {
                "name": item.name,
                "is_dir": item.is_dir(),
                "is_file": item.is_file(),
            }
            for item in entries[:MAX_DIRECTORY_ENTRIES]
        ]
        result["entries_truncated"] = len(entries) > MAX_DIRECTORY_ENTRIES
        result["allowed_roots"] = _get_allowed_root_strings()
        _write_audit_event("inspect_file_system_path", {"ok": True, "path": str(resolved), "is_dir": True})
        return result

    result["suffix"] = resolved.suffix
    if include_preview:
        try:
            content = resolved.read_text(encoding="utf-8")
            preview_limit = max(1, min(int(preview_lines), MAX_PREVIEW_LINES))
            preview = "\n".join(content.splitlines()[:preview_limit])
            preview, preview_truncated = _truncate_text(preview, limit=4000)
            result["preview"] = preview
            result["preview_truncated"] = preview_truncated or len(content.splitlines()) > preview_limit
        except UnicodeDecodeError:
            result["preview"] = "<binary or non-UTF8 file>"
            result["preview_truncated"] = False
    result["allowed_roots"] = _get_allowed_root_strings()
    _write_audit_event("inspect_file_system_path", {"ok": True, "path": str(resolved), "is_dir": False})
    return result


if FastMCP is not None:
    mcp = FastMCP(SERVER_NAME)

    @mcp.tool(name="execute_terminal_command")
    def execute_terminal_command_tool(
        command: str,
        cwd: str = ".",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        allow_outside_workspace: bool = False,
        allow_destructive: bool = False,
    ) -> dict[str, Any]:
        return execute_terminal_command(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            allow_outside_workspace=allow_outside_workspace,
            allow_destructive=allow_destructive,
        )

    @mcp.tool(name="get_system_info")
    def get_system_info_tool() -> dict[str, Any]:
        return collect_system_info()

    @mcp.tool(name="get_mcp_security_policy")
    def get_mcp_security_policy_tool() -> dict[str, Any]:
        return get_mcp_security_policy()

    @mcp.tool(name="inspect_file_system_path")
    def inspect_file_system_path_tool(
        path: str,
        include_preview: bool = False,
        preview_lines: int = 20,
        allow_outside_workspace: bool = False,
    ) -> dict[str, Any]:
        return inspect_file_system_path(
            path=path,
            include_preview=include_preview,
            preview_lines=preview_lines,
            allow_outside_workspace=allow_outside_workspace,
        )


if __name__ == "__main__":
    if FastMCP is None:
        raise SystemExit("The 'mcp' package is required to run this server. Install dependencies from requirements.txt first.")
    mcp.run()