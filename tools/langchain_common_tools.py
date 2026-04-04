import ast
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from core.config import config
from core.time_utils import CHINA_TIMEZONE

MAX_FILE_PREVIEW_CHARS = 12000
MAX_SEARCH_RESULTS = 100

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Constant,
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _workspace_root() -> Path:
    return Path(config.resolve_path(".")).resolve()


def _resolve_workspace_path(path_value: str | None) -> Path:
    candidate = Path(path_value or ".")
    root = _workspace_root()
    resolved = candidate if candidate.is_absolute() else (root / candidate)
    resolved = resolved.resolve()
    if resolved == root or root in resolved.parents:
        return resolved
    raise ValueError(f"Path is outside workspace root: {resolved}")


def _safe_eval_node(node: ast.AST) -> float:
    if type(node) not in _ALLOWED_AST_NODES:
        raise ValueError("Expression contains unsupported syntax.")

    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only int/float constants are allowed.")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPERATORS:
            raise ValueError("Binary operator is not allowed.")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return float(_BINARY_OPERATORS[op_type](left, right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPERATORS:
            raise ValueError("Unary operator is not allowed.")
        operand = _safe_eval_node(node.operand)
        return float(_UNARY_OPERATORS[op_type](operand))

    raise ValueError("Unsupported expression node.")


@tool
def get_current_time() -> str:
    """获取当前时间（中国时区）。当用户询问现在几点、当前时间时使用。"""
    now = datetime.now(CHINA_TIMEZONE)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def calculator(expression: str) -> dict[str, Any]:
    """安全计算数学表达式，只支持 + - * / // % ** 和括号。"""
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _safe_eval_node(parsed)
        return {"ok": True, "expression": expression, "result": value}
    except Exception as exc:
        return {"ok": False, "expression": expression, "error": str(exc)}


@tool
def list_directory(path: str = ".") -> dict[str, Any]:
    """列出工作区内目录项（安全边界：仅允许工作区内路径）。"""
    try:
        resolved = _resolve_workspace_path(path)
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    if not resolved.exists():
        return {"ok": False, "path": path, "error": f"Path not found: {resolved}"}
    if not resolved.is_dir():
        return {"ok": False, "path": path, "error": f"Path is not a directory: {resolved}"}

    entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
    return {
        "ok": True,
        "path": str(resolved),
        "entries": [
            {
                "name": item.name,
                "is_dir": item.is_dir(),
                "is_file": item.is_file(),
            }
            for item in entries[:500]
        ],
        "truncated": len(entries) > 500,
        "entry_count": len(entries),
    }


@tool
def read_text_file(path: str, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
    """读取工作区内 UTF-8 文本文件的指定行范围。"""
    try:
        resolved = _resolve_workspace_path(path)
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    if not resolved.exists() or not resolved.is_file():
        return {"ok": False, "path": path, "error": f"File not found: {resolved}"}

    start = max(1, int(start_line))
    end = max(start, int(end_line))

    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"ok": False, "path": path, "error": "File is not valid UTF-8 text."}
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    sliced = lines[start - 1 : end]
    content = "\n".join(sliced)
    truncated = False
    if len(content) > MAX_FILE_PREVIEW_CHARS:
        content = content[:MAX_FILE_PREVIEW_CHARS] + "\n...<truncated>"
        truncated = True

    return {
        "ok": True,
        "path": str(resolved),
        "start_line": start,
        "end_line": end,
        "line_count": len(lines),
        "content": content,
        "truncated": truncated,
    }


@tool
def grep_text(query: str, path: str = ".", max_results: int = 20) -> dict[str, Any]:
    """在工作区内递归搜索文本，返回匹配文件与行号。"""
    if not query:
        return {"ok": False, "error": "query must be non-empty"}

    try:
        resolved_root = _resolve_workspace_path(path)
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    if not resolved_root.exists():
        return {"ok": False, "path": path, "error": f"Path not found: {resolved_root}"}

    limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    matches: list[dict[str, Any]] = []

    files = [resolved_root] if resolved_root.is_file() else resolved_root.rglob("*")
    for item in files:
        if len(matches) >= limit:
            break
        if not item.is_file():
            continue
        try:
            text = item.read_text(encoding="utf-8")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if query.lower() in line.lower():
                matches.append({"path": str(item), "line": idx, "text": line[:400]})
                if len(matches) >= limit:
                    break

    return {
        "ok": True,
        "query": query,
        "path": str(resolved_root),
        "matches": matches,
        "truncated": len(matches) >= limit,
    }


@tool
def write_text_file(path: str, content: str, overwrite: bool = False, append: bool = False) -> dict[str, Any]:
    """在工作区内写入 UTF-8 文本文件。默认不覆盖已存在文件，可选 append。"""
    try:
        resolved = _resolve_workspace_path(path)
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    if resolved.exists() and resolved.is_dir():
        return {"ok": False, "path": str(resolved), "error": "Target path is a directory."}

    if resolved.exists() and not overwrite and not append:
        return {
            "ok": False,
            "path": str(resolved),
            "error": "File already exists. Set overwrite=True or append=True.",
        }

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with resolved.open(mode, encoding="utf-8") as file_handle:
            file_handle.write(content)
    except Exception as exc:
        return {"ok": False, "path": str(resolved), "error": str(exc)}

    return {
        "ok": True,
        "path": str(resolved),
        "mode": "append" if append else "overwrite" if overwrite else "create",
        "bytes_written": len(content.encode("utf-8")),
    }


@tool
def json_query(path: str, key_path: str = "") -> dict[str, Any]:
    """读取工作区内 JSON 文件并按 key_path 提取字段（例如: user.profile.name 或 items.0.id）。"""
    try:
        resolved = _resolve_workspace_path(path)
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}

    if not resolved.exists() or not resolved.is_file():
        return {"ok": False, "path": path, "error": f"File not found: {resolved}"}

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "path": str(resolved), "error": f"Invalid JSON: {exc}"}

    if not key_path:
        return {"ok": True, "path": str(resolved), "value": payload}

    current: Any = payload
    try:
        for token in [part for part in key_path.split(".") if part != ""]:
            if isinstance(current, list):
                current = current[int(token)]
            elif isinstance(current, dict):
                current = current[token]
            else:
                raise KeyError(token)
    except Exception as exc:
        return {
            "ok": False,
            "path": str(resolved),
            "key_path": key_path,
            "error": f"Failed to resolve key_path: {exc}",
        }

    return {
        "ok": True,
        "path": str(resolved),
        "key_path": key_path,
        "value": current,
    }


tools = [
    get_current_time,
    calculator,
    list_directory,
    read_text_file,
    grep_text,
    write_text_file,
    json_query,
]
