import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_request_id_var: ContextVar[str | None] = ContextVar("llm_request_id", default=None)
_session_id_var: ContextVar[str | None] = ContextVar("llm_session_id", default=None)
_request_cancel_checker_var: ContextVar[Any] = ContextVar("llm_request_cancel_checker", default=None)


class RequestCancelledError(RuntimeError):
    pass


class LLMRuntime:
    def get_request_id(self) -> str | None:
        return _request_id_var.get()

    def get_session_id(self) -> str | None:
        return _session_id_var.get()

    @contextmanager
    def request_scope(self, request_id: str, session_id: str | None = None, cancel_checker=None):
        token = _request_id_var.set(request_id)
        session_token = _session_id_var.set(session_id)
        cancel_token = _request_cancel_checker_var.set(cancel_checker)
        try:
            yield request_id
        finally:
            _request_cancel_checker_var.reset(cancel_token)
            _session_id_var.reset(session_token)
            _request_id_var.reset(token)

    def with_request_id(self, message: str, request_id: str | None = None) -> str:
        active_request_id = request_id or self.get_request_id()
        if not active_request_id:
            return message
        return f"request_id={active_request_id} | {message}"

    def run_with_timeout(self, func, timeout_seconds: int, timeout_message: str):
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