import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import config
from core.time_utils import CHINA_TIMEZONE


class ChinaTimezoneFormatter(logging.Formatter):
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
        formatter = ChinaTimezoneFormatter("%(asctime)s | %(levelname)s | %(message)s")

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
        request_id: str | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> dict[str, Any]:
        source = fields.pop("source", "-")
        outcome = fields.pop("outcome", "-")
        payload = {
            "event_type": event_type,
            "message": message,
            "request_id": request_id or self.runtime.get_request_id() or "-",
            "session_id": session_id or self.runtime.get_session_id() or "-",
            "stage": stage,
            "source": self.stringify_field(source),
            "outcome": self.stringify_field(outcome),
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        for key, value in fields.items():
            payload[key] = self.stringify_field(value)
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
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )
        self._logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def parse_text_log_message(self, raw_message: str) -> dict[str, Any]:
        payload = {
            "event_type": "text_event",
            "message": raw_message,
        }
        request_match = re.search(r"request_id=([^|]+)", raw_message)
        if request_match:
            payload["request_id"] = request_match.group(1).strip()
        stage_match = re.search(r"stage=([^|]+)", raw_message)
        if stage_match:
            payload["stage"] = stage_match.group(1).strip()
        return payload

    def get_log_path(self) -> str:
        return str(Path(config.resolve_path(config.log_dir)) / config.llm_log_file)

    def get_request_events(self, request_id: str, limit: int = 200) -> list[dict[str, Any]]:
        log_path = Path(self.get_log_path())
        if not request_id or not log_path.exists():
            return []

        events = []
        with open(log_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                parts = line.rstrip("\n").split(" | ", 2)
                raw_message = parts[2] if len(parts) == 3 else line.strip()
                if request_id not in raw_message:
                    continue

                try:
                    payload = json.loads(raw_message)
                except json.JSONDecodeError:
                    payload = self.parse_text_log_message(raw_message)

                if payload.get("request_id") != request_id:
                    continue

                payload["logged_at"] = parts[0] if len(parts) >= 1 else ""
                payload["level"] = parts[1] if len(parts) >= 2 else ""
                events.append(payload)

        if limit > 0:
            return events[-limit:]
        return events

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

    def console_event(self, stage: str, request_id: str | None = None, level: int = logging.INFO) -> None:
        self._console_logger.log(level, self.runtime.with_request_id(f"stage={stage}", request_id=request_id))

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
            self.console_event(stage, request_id=request_id, level=level)