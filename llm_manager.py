from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from config import config

_request_id_var: ContextVar[str | None] = ContextVar("llm_request_id", default=None)
_request_cancel_checker_var: ContextVar[Any] = ContextVar("llm_request_cancel_checker", default=None)


class RequestCancelledError(RuntimeError):
    pass

class LLMManager:
    def __init__(self):
        self._current_llm = None
        self._logger = self._build_file_logger()
        self._console_logger = self._build_console_logger()
        self._update_llm()

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
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

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

    def _truncate_text(self, text: str) -> str:
        if len(text) <= config.llm_log_max_chars:
            return text
        hidden = len(text) - config.llm_log_max_chars
        return f"{text[:config.llm_log_max_chars]}\n...<truncated {hidden} chars>"

    def _stringify_payload(self, payload: Any) -> str:
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

    def _stringify_response(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if isinstance(content, list):
            content = str(content)
        if content is not None:
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                return f"{content}\ntool_calls={tool_calls}"
            return str(content)
        return str(response)

    def _run_with_timeout(self, func, timeout_seconds: int, timeout_message: str):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        start = time.monotonic()
        cancel_checker = _request_cancel_checker_var.get()
        try:
            while True:
                if cancel_checker and cancel_checker():
                    future.cancel()
                    raise RequestCancelledError("Request cancelled during model invocation.")

                elapsed = time.monotonic() - start
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    future.cancel()
                    raise TimeoutError(timeout_message)

                try:
                    return future.result(timeout=min(0.1, remaining))
                except FuturesTimeoutError:
                    continue
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def get_request_id(self) -> str | None:
        return _request_id_var.get()

    @contextmanager
    def request_scope(self, request_id: str, cancel_checker = None):
        token = _request_id_var.set(request_id)
        cancel_token = _request_cancel_checker_var.set(cancel_checker)
        try:
            yield request_id
        finally:
            _request_cancel_checker_var.reset(cancel_token)
            _request_id_var.reset(token)

    def _with_request_id(self, message: str, request_id: str | None = None) -> str:
        active_request_id = request_id or self.get_request_id()
        if not active_request_id:
            return message
        return f"request_id={active_request_id} | {message}"

    def _stringify_field(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, dict)):
            return value
        return str(value)

    def _build_structured_payload(
        self,
        event_type: str,
        message: str = "",
        request_id: str | None = None,
        session_id: str | None = None,
        stage: str | None = None,
        duration_ms: float | None = None,
        **fields,
    ) -> dict[str, Any]:
        payload = {
            "event_type": event_type,
            "message": message,
            "request_id": request_id or self.get_request_id(),
            "session_id": session_id,
            "stage": stage,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        for key, value in fields.items():
            payload[key] = self._stringify_field(value)
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
        payload = self._build_structured_payload(
            event_type,
            message=message,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )
        self._logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _parse_text_log_message(self, raw_message: str) -> dict[str, Any]:
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
                    payload = self._parse_text_log_message(raw_message)

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
            **fields,
        )

    def console_event(self, stage: str, request_id: str | None = None, level: int = logging.INFO) -> None:
        self._console_logger.log(level, self._with_request_id(f"stage={stage}", request_id=request_id))

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
        self.log_structured_event(
            "checkpoint",
            message=f"Checkpoint {stage}",
            level=level,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            details=details,
            **fields,
        )
        if console:
            self.console_event(stage, request_id=request_id, level=level)

    def set_model(self, provider: str, model: str, base_url: str = None, api_key: str = None):
        config.llm_config.provider = provider
        config.llm_config.model = model
        if base_url: config.llm_config.base_url = base_url
        if api_key: config.llm_config.api_key = api_key
        self._update_llm()
        self.log_event(f"LLM model switched | provider={provider} | model={model}")
        return f"Successfully switched to {provider}:{model}"

    def _update_llm(self):
        cfg = config.llm_config
        if cfg.provider == "ollama":
            self._current_llm = ChatOllama(model=cfg.model, base_url=cfg.base_url or "http://localhost:11434")
        elif cfg.provider == "openai":
            self._current_llm = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

    def get_llm(self) -> BaseChatModel:
        if not self._current_llm:
            self._update_llm()
        return self._current_llm

    def invoke(self, payload: Any, source: str = "unknown", llm = None):
        target_llm = llm or self.get_llm()
        self.log_structured_event(
            "llm_request",
            message="LLM request",
            source=source,
            provider=config.llm_config.provider,
            model=config.llm_config.model,
            payload=self._truncate_text(self._stringify_payload(payload)),
        )
        started_at = time.monotonic()
        try:
            response = self._run_with_timeout(
                lambda: target_llm.invoke(payload),
                config.llm_timeout_seconds,
                f"LLM invocation timed out after {config.llm_timeout_seconds} seconds.",
            )
        except Exception as exc:
            self.log_structured_event(
                "llm_error",
                message="LLM invocation failed",
                level=logging.ERROR,
                source=source,
                provider=config.llm_config.provider,
                model=config.llm_config.model,
                duration_ms=(time.monotonic() - started_at) * 1000,
                error=str(exc),
            )
            raise

        self.log_structured_event(
            "llm_response",
            message="LLM response",
            source=source,
            provider=config.llm_config.provider,
            model=config.llm_config.model,
            duration_ms=(time.monotonic() - started_at) * 1000,
            response=self._truncate_text(self._stringify_response(response)),
        )
        return response

llm_manager = LLMManager()
