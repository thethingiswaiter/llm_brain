import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import config
from core.time_utils import CHINA_TIMEZONE


class ChinaTimezoneFormatter(logging.Formatter):
    @staticmethod
    def current_timestamp() -> str:
        return datetime.now(tz=CHINA_TIMEZONE).isoformat(sep=" ", timespec="seconds")

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=CHINA_TIMEZONE)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(sep=" ", timespec="seconds")


class LLMLogging:
    def __init__(self, runtime):
        self.runtime = runtime
        self._logger = self._build_file_logger()
        self._console_logger = self._build_console_logger()

    def _build_file_logger(self) -> logging.Logger:
        logger = logging.getLogger("llm_brain.llm")
        log_dir = Path(config.resolve_path(config.log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / config.llm_log_file

        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if str(log_path) not in existing_paths:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(message)s")

        if not logger.handlers:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _build_console_logger(self) -> logging.Logger:
        logger = logging.getLogger("llm_brain.console")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def truncate_text(self, text: str) -> str:
        if len(text) <= config.llm_log_max_chars:
            return text
        hidden = len(text) - config.llm_log_max_chars
        return f"{text[:config.llm_log_max_chars]}\n...<truncated {hidden} chars>"

    def stringify_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            lines = []
            for index, item in enumerate(payload, start=1):
                role = getattr(item, "type", item.__class__.__name__)
                content = getattr(item, "content", item)
                if isinstance(content, list):
                    content = str(content)
                tool_calls = getattr(item, "tool_calls", None)
                line = f"[{index}] {role}: {content}"
                if tool_calls:
                    line += f" | tool_calls={tool_calls}"
                lines.append(line)
            return "\n".join(lines)
        return str(payload)

    def stringify_response(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, list):
            content = str(content)
        if content is not None:
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                return f"{content}\ntool_calls={tool_calls}"
            return str(content)
        return str(response)

    def stringify_field(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, dict)):
            return value
        return str(value)

    def build_structured_payload(
        self,
        event_type: str,
        message: str = "",
        level: int | str = logging.INFO,
        request_id: str | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> dict[str, Any]:
        source = fields.pop("source", "-")
        outcome = fields.pop("outcome", "-")
        logged_at = fields.pop("logged_at", ChinaTimezoneFormatter.current_timestamp())
        payload = {
            "request_id": request_id or self.runtime.get_request_id() or "-",
            "level": self.stringify_field(logging.getLevelName(level) if isinstance(level, int) else level),
            "event_type": event_type,
            "message": message,
            "stage": stage,
            "outcome": self.stringify_field(outcome),
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        for key, value in fields.items():
            payload[key] = self.stringify_field(value)
        payload["logged_at"] = self.stringify_field(logged_at)
        payload["session_id"] = session_id or self.runtime.get_session_id() or "-"
        payload["source"] = self.stringify_field(source)
        return {key: value for key, value in payload.items() if value not in (None, "")}

    def log_structured_event(
        self,
        event_type: str,
        message: str = "",
        level: int = logging.INFO,
        request_id: str | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> None:
        payload = self.build_structured_payload(
            event_type,
            message=message,
            level=level,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )
        self._logger.log(level, json.dumps(payload, ensure_ascii=False))

    def parse_log_line(self, raw_line: str) -> dict[str, Any] | None:
        line = str(raw_line or "").strip()
        if not line:
            return None

        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
            return None
        except json.JSONDecodeError:
            return None

    def get_log_path(self) -> str:
        return str(Path(config.resolve_path(config.log_dir)) / config.llm_log_file)

    def get_request_events(self, request_id: str, limit: int = 200) -> list[dict[str, Any]]:
        log_path = Path(self.get_log_path())
        if not request_id or not log_path.exists():
            return []

        events = []
        with open(log_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                payload = self.parse_log_line(line)
                if not payload:
                    continue

                if payload.get("request_id") != request_id:
                    continue

                events.append(payload)

        if limit > 0:
            return events[-limit:]
        return events

    def _format_console_details(self, details: str, max_chars: int = 160) -> str:
        raw = str(details or "").strip()
        if not raw:
            return ""
        if "\n" in raw:
            lines = []
            for index, line in enumerate(raw.splitlines()):
                normalized_line = re.sub(r"\s+", " ", str(line or "").strip())
                if not normalized_line:
                    continue
                if len(normalized_line) > max_chars:
                    normalized_line = normalized_line[: max(1, max_chars - 3)].rstrip() + "..."
                lines.append(normalized_line if index == 0 else f"  {normalized_line}")
            return "\n".join(lines)
        normalized = re.sub(r"\s+", " ", raw)
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(1, max_chars - 3)].rstrip() + "..."

    def _parse_detail_fields(self, details: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for part in str(details or "").split("|"):
            item = str(part or "").strip()
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            normalized_key = str(key or "").strip()
            normalized_value = str(value or "").strip()
            if normalized_key:
                fields[normalized_key] = normalized_value
        return fields

    def _summarize_console_details(self, stage: str, details: str) -> str:
        normalized_stage = str(stage or "").strip().lower()
        fields = self._parse_detail_fields(details)

        if normalized_stage == "planning_started":
            session_id = fields.get("session_id", "")
            return f"会话={session_id}" if session_id else details

        if normalized_stage == "planning_completed":
            parts = []
            if fields.get("subtask_count"):
                parts.append(f"计划={fields['subtask_count']}步")
            if fields.get("domain"):
                parts.append(f"领域={fields['domain']}")
            if fields.get("mode"):
                parts.append(f"模式={fields['mode']}")
            summary_line = " | ".join(parts) or details
            plan_preview = fields.get("plan_preview", "")
            if plan_preview:
                plan_lines = [item.strip() for item in plan_preview.split("||") if item.strip()]
                if plan_lines:
                    return summary_line + "\n" + "\n".join(f"- {item}" for item in plan_lines)
            return summary_line

        if normalized_stage == "subtask_started":
            index = fields.get("index", "")
            description = fields.get("description", "")
            mode = fields.get("mode", "")
            mode = fields.get("mode", "")
            parts = []
            if index and description:
                parts.append(f"子任务{index}: {description}")
            elif index:
                parts.append(f"子任务{index}")
            elif description:
                parts.append(description)
            if mode:
                parts.append(f"模式={mode}")
            summary_line = " | ".join(parts) or details
            return summary_line

        if normalized_stage == "subtask_llm_dispatch":
            parts = []
            if fields.get("index"):
                parts.append(f"子任务{fields['index']}")
            if fields.get("tool_count"):
                parts.append(f"工具数={fields['tool_count']}")
            after_tool = fields.get("after_tool", "")
            if after_tool:
                parts.append("工具续轮" if after_tool.lower() == "true" else "初次调度")
            if fields.get("mode"):
                parts.append(f"模式={fields['mode']}")
            return " | ".join(parts) or details

        if normalized_stage in {"tool_started", "tool_succeeded", "tool_cancelled"}:
            tool_name = fields.get("tool", "")
            error_type = fields.get("error_type", "")
            summary = fields.get("summary", "")
            parts = [f"工具={tool_name}"] if tool_name else []
            if error_type:
                parts.append(f"错误={error_type}")
            summary_line = " | ".join(parts) or details
            if summary:
                return summary_line + f"\n- 结果: {summary}"
            return summary_line

        if normalized_stage in {"tool_failed", "tool_rejected"}:
            tool_name = fields.get("tool", "")
            error_type = fields.get("error_type", "")
            summary = fields.get("summary", "")
            parts = []
            if tool_name:
                parts.append(f"工具={tool_name}")
            if error_type:
                parts.append(f"错误={error_type}")
            summary_line = " | ".join(parts) or details
            if summary:
                return summary_line + f"\n- 详情: {summary}"
            return summary_line

        if normalized_stage == "tool_detached":
            parts = []
            if fields.get("tool"):
                parts.append(f"工具={fields['tool']}")
            if fields.get("reason"):
                parts.append(f"原因={fields['reason']}")
            if fields.get("tool_run_id"):
                parts.append(f"run_id={fields['tool_run_id']}")
            return " | ".join(parts) or details

        if normalized_stage == "tool_reroute_applied":
            parts = []
            if fields.get("index"):
                parts.append(f"子任务{fields['index']}")
            if fields.get("mode"):
                parts.append(f"重试策略={fields['mode']}")
            if fields.get("failed_tools"):
                parts.append(f"失败工具={fields['failed_tools']}")
            return " | ".join(parts) or details

        if normalized_stage == "reflection_completed":
            parts = []
            if fields.get("index"):
                parts.append(f"子任务{fields['index']}")
            if fields.get("success"):
                parts.append("校验通过" if fields["success"].lower() == "true" else "校验失败")
            if fields.get("action"):
                parts.append(f"动作={fields['action']}")
            summary_line = " | ".join(parts) or details
            reflection_note = fields.get("reflection", "")
            if reflection_note:
                return summary_line + f"\n- 分析: {reflection_note}"
            return summary_line

        if normalized_stage == "subtask_replanned":
            parts = []
            if fields.get("index"):
                parts.append(f"子任务{fields['index']}")
            if fields.get("new_subtask_count"):
                parts.append(f"重规划为{fields['new_subtask_count']}步")
            return " | ".join(parts) or details

        if normalized_stage == "subtask_advanced":
            next_index = fields.get("next_index", "")
            mode = fields.get("mode", "")
            parts = [f"进入子任务{next_index}"] if next_index else []
            if mode:
                parts.append(f"模式={mode}")
            return " | ".join(parts) or details

        if normalized_stage == "agent_completed":
            subtask_count = fields.get("subtask_count", "")
            mode = fields.get("mode", "")
            parts = [f"完成{subtask_count}个子任务"] if subtask_count else []
            if mode:
                parts.append(f"模式={mode}")
            return " | ".join(parts) or details

        if normalized_stage == "agent_blocked":
            parts = []
            if fields.get("index"):
                parts.append(f"阻塞于子任务{fields['index']}")
            if fields.get("action"):
                parts.append(f"动作={fields['action']}")
            return " | ".join(parts) or details

        if normalized_stage == "agent_waiting_user":
            parts = []
            if fields.get("index"):
                parts.append(f"等待用户于子任务{fields['index']}")
            if fields.get("action"):
                parts.append(f"动作={fields['action']}")
            return " | ".join(parts) or details

        return details

    def log_event(self, message: str, level: int = logging.INFO, request_id: str | None = None, **fields) -> None:
        self.log_structured_event(
            "text_event",
            message=message,
            level=level,
            request_id=request_id,
            source="log_event",
            outcome="info" if level < logging.ERROR else "error",
            **fields,
        )

    def console_event(
        self,
        stage: str,
        request_id: str | None = None,
        level: int = logging.INFO,
        details: str = "",
    ) -> None:
        message = f"stage={stage}"
        formatted_details = self._format_console_details(self._summarize_console_details(stage, details))
        if formatted_details:
            message += f" | {formatted_details}"
        self._console_logger.log(level, self.runtime.with_request_id(message, request_id=request_id))

    def log_checkpoint(
        self,
        stage: str,
        details: str = "",
        request_id: str | None = None,
        level: int = logging.INFO,
        console: bool = False,
        session_id: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> None:
        source = fields.pop("source", "checkpoint")
        outcome = fields.pop("outcome", "recorded")
        self.log_structured_event(
            "checkpoint",
            message=f"Checkpoint {stage}",
            level=level,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            source=source,
            outcome=outcome,
            details=details,
            **fields,
        )
        if console:
            self.console_event(stage, request_id=request_id, level=level, details=details)