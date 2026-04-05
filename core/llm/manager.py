import logging
import os
import sys
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from core.config import config
from core.llm.factory import build_llm
from core.llm.logging import LLMLogging
from core.llm.runtime import LLMRuntime, RequestCancelledError


class LLMDependencyUnavailableError(RuntimeError):
    pass


class OfflineTestLLM:
    def invoke(self, payload):
        return AIMessage(content="[offline-test-llm] live model call skipped during unittest")

class LLMManager:
    def __init__(self):
        self._current_llm = None
        self._llm_init_error = None
        self.runtime = LLMRuntime()
        self.logging = LLMLogging(self.runtime)
        self._logger = self.logging._logger
        self._console_logger = self.logging._console_logger
        try:
            self._update_llm()
        except Exception as exc:
            self._llm_init_error = exc

    def _build_file_logger(self) -> logging.Logger:
        return self.logging._build_file_logger()

    def _build_console_logger(self) -> logging.Logger:
        return self.logging._build_console_logger()

    def _truncate_text(self, text: str) -> str:
        return self.logging.truncate_text(text)

    def _stringify_payload(self, payload: Any) -> str:
        return self.logging.stringify_payload(payload)

    def _stringify_response(self, response: Any) -> str:
        return self.logging.stringify_response(response)

    def _run_with_timeout(self, func, timeout_seconds: int, timeout_message: str):
        return self.runtime.run_with_timeout(func, timeout_seconds, timeout_message)

    def get_request_id(self) -> str | None:
        return self.runtime.get_request_id()

    def get_session_id(self) -> str | None:
        return self.runtime.get_session_id()

    def request_scope(self, request_id: str, session_id: str | None = None, cancel_checker = None):
        return self.runtime.request_scope(request_id, session_id=session_id, cancel_checker=cancel_checker)

    def _with_request_id(self, message: str, request_id: str | None = None) -> str:
        return self.runtime.with_request_id(message, request_id=request_id)

    def _stringify_field(self, value: Any) -> Any:
        return self.logging.stringify_field(value)

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
        return self.logging.build_structured_payload(
            event_type,
            message=message,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )

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
        self.logging.log_structured_event(
            event_type,
            message=message,
            level=level,
            request_id=request_id,
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
            **fields,
        )

    def _parse_text_log_message(self, raw_message: str) -> dict[str, Any]:
        return self.logging.parse_text_log_message(raw_message)

    def get_log_path(self) -> str:
        return self.logging.get_log_path()

    def get_request_events(self, request_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.logging.get_request_events(request_id, limit=limit)

    def log_event(self, message: str, level: int = logging.INFO, request_id: str | None = None, **fields) -> None:
        self.logging.log_event(message, level=level, request_id=request_id, **fields)

    def console_event(
        self,
        stage: str,
        request_id: str | None = None,
        level: int = logging.INFO,
        details: str = "",
    ) -> None:
        self.logging.console_event(stage, request_id=request_id, level=level, details=details)

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
        self.logging.log_checkpoint(
            stage,
            details=details,
            request_id=request_id,
            level=level,
            console=console,
            session_id=session_id,
            duration_ms=duration_ms,
            **fields,
        )

    def set_model(self, provider: str, model: str, base_url: str = None, api_key: str = None):
        config.llm_config.provider = provider
        config.llm_config.model = model
        if base_url: config.llm_config.base_url = base_url
        if api_key: config.llm_config.api_key = api_key
        try:
            self._update_llm()
        except Exception as exc:
            raise LLMDependencyUnavailableError(
                f"Failed to initialize provider {provider}: {exc}"
            ) from exc
        self.log_event(f"LLM model switched | provider={provider} | model={model}")
        return f"Successfully switched to {provider}:{model}"

    def _current_model_matches(self, model_config: dict[str, Any]) -> bool:
        if not isinstance(model_config, dict):
            return False
        if str(model_config.get("provider") or "") != str(config.llm_config.provider or ""):
            return False
        if str(model_config.get("model") or "") != str(config.llm_config.model or ""):
            return False

        configured_base_url = model_config.get("base_url")
        if configured_base_url is not None and str(configured_base_url) != str(config.llm_config.base_url or ""):
            return False

        configured_api_key = model_config.get("api_key")
        if configured_api_key is not None and str(configured_api_key) != str(config.llm_config.api_key or ""):
            return False
        return True

    def get_current_model_key(self) -> str:
        for model_key, model_config in (config.models_config or {}).items():
            if self._current_model_matches(model_config):
                return str(model_key)
        return ""

    def list_available_models(self) -> list[dict[str, Any]]:
        current_key = self.get_current_model_key()
        models: list[dict[str, Any]] = []
        for model_key, model_config in (config.models_config or {}).items():
            provider = str(model_config.get("provider") or "")
            model_name = str(model_config.get("model") or "")
            base_url = str(model_config.get("base_url") or "")
            models.append(
                {
                    "key": str(model_key),
                    "provider": provider,
                    "model": model_name,
                    "base_url": base_url,
                    "is_default": str(model_key) == str(config.default_model_key or ""),
                    "is_current": str(model_key) == current_key,
                }
            )
        return models

    def set_model_by_key(self, model_key: str):
        normalized_key = str(model_key or "").strip()
        model_config = (config.models_config or {}).get(normalized_key)
        if not isinstance(model_config, dict):
            available = ", ".join(sorted(str(item) for item in (config.models_config or {}).keys())) or "none"
            raise KeyError(f"Unknown model key: {normalized_key}. Available model keys: {available}")

        provider = str(model_config.get("provider") or "").strip()
        model = str(model_config.get("model") or "").strip()
        base_url = model_config.get("base_url")
        api_key = model_config.get("api_key")
        result = self.set_model(provider, model, base_url=base_url, api_key=api_key)
        return f"{result} (config={normalized_key})"

    def _update_llm(self):
        disable_live_llm = os.getenv("LLM_BRAIN_DISABLE_LIVE_LLM", "").strip().lower()
        if disable_live_llm in {"1", "true", "yes", "on"}:
            self._current_llm = OfflineTestLLM()
            self._llm_init_error = None
            return
        if disable_live_llm == "" and "unittest" in sys.modules:
            self._current_llm = OfflineTestLLM()
            self._llm_init_error = None
            return
        self._current_llm = build_llm(config.llm_config)
        self._llm_init_error = None

    def get_llm(self) -> BaseChatModel:
        if not self._current_llm:
            try:
                self._update_llm()
            except Exception as exc:
                root_exc = self._llm_init_error or exc
                raise LLMDependencyUnavailableError(
                    f"Failed to initialize provider {config.llm_config.provider}: {root_exc}"
                ) from root_exc
        return self._current_llm

    def invoke(self, payload: Any, source: str = "unknown", llm = None):
        target_llm = llm or self.get_llm()
        self.log_structured_event(
            "llm_request",
            message="LLM request",
            source=source,
            outcome="started",
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
            error_type = "execution_error"
            outcome = "failed"
            retryable = False
            raised_exc = exc
            if isinstance(exc, RequestCancelledError):
                error_type = "cancelled"
                outcome = "cancelled"
                retryable = True
            elif isinstance(exc, TimeoutError):
                error_type = "timeout"
                outcome = "timed_out"
                retryable = True
            elif isinstance(exc, (ConnectionError, OSError, ImportError, ModuleNotFoundError)):
                error_type = "dependency_unavailable"
                retryable = True
                raised_exc = LLMDependencyUnavailableError(
                    f"LLM dependency unavailable: {exc}"
                )
            self.log_structured_event(
                "llm_error",
                message="LLM invocation failed",
                level=logging.ERROR,
                source=source,
                outcome=outcome,
                provider=config.llm_config.provider,
                model=config.llm_config.model,
                duration_ms=(time.monotonic() - started_at) * 1000,
                error_type=error_type,
                retryable=retryable,
                error=str(exc),
            )
            if raised_exc is not exc:
                raise raised_exc from exc
            raise

        self.log_structured_event(
            "llm_response",
            message="LLM response",
            source=source,
            outcome="completed",
            provider=config.llm_config.provider,
            model=config.llm_config.model,
            duration_ms=(time.monotonic() - started_at) * 1000,
            response=self._truncate_text(self._stringify_response(response)),
        )
        return response

llm_manager = LLMManager()
