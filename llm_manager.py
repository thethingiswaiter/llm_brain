from contextlib import contextmanager
from contextvars import ContextVar
import logging
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from config import config

_request_id_var: ContextVar[str | None] = ContextVar("llm_request_id", default=None)

class LLMManager:
    def __init__(self):
        self._current_llm = None
        self._logger = self._build_file_logger()
        self._console_logger = self._build_console_logger()
        self._update_llm()

    def _build_file_logger(self) -> logging.Logger:
        logger = logging.getLogger("llm_brain.llm")
        if logger.handlers:
            return logger

        log_dir = Path(config.resolve_path(config.log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / config.llm_log_file

        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

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

    def get_request_id(self) -> str | None:
        return _request_id_var.get()

    @contextmanager
    def request_scope(self, request_id: str):
        token = _request_id_var.set(request_id)
        try:
            yield request_id
        finally:
            _request_id_var.reset(token)

    def _with_request_id(self, message: str, request_id: str | None = None) -> str:
        active_request_id = request_id or self.get_request_id()
        if not active_request_id:
            return message
        return f"request_id={active_request_id} | {message}"

    def log_event(self, message: str, level: int = logging.INFO, request_id: str | None = None) -> None:
        self._logger.log(level, self._with_request_id(message, request_id=request_id))

    def console_event(self, stage: str, request_id: str | None = None, level: int = logging.INFO) -> None:
        self._console_logger.log(level, self._with_request_id(f"stage={stage}", request_id=request_id))

    def log_checkpoint(
        self,
        stage: str,
        details: str = "",
        request_id: str | None = None,
        level: int = logging.INFO,
        console: bool = False,
    ) -> None:
        message = f"Checkpoint | stage={stage}"
        if details:
            message = f"{message} | {details}"
        self.log_event(message, level=level, request_id=request_id)
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
        self.log_event(
            f"LLM request | source={source} | provider={config.llm_config.provider} | model={config.llm_config.model}\n"
            f"{self._truncate_text(self._stringify_payload(payload))}"
        )
        try:
            response = target_llm.invoke(payload)
        except Exception as exc:
            self.log_event(
                f"LLM error | source={source} | provider={config.llm_config.provider} | model={config.llm_config.model} | error={exc}",
                level=logging.ERROR,
            )
            raise

        self.log_event(
            f"LLM response | source={source} | provider={config.llm_config.provider} | model={config.llm_config.model}\n"
            f"{self._truncate_text(self._stringify_response(response))}"
        )
        return response

llm_manager = LLMManager()
