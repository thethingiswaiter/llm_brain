import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from typing import Any

from langchain_core.messages import HumanMessage

from core.config import config
from cognitive.feature_extractor import DEFAULT_DOMAIN_LABEL
from core.llm.manager import LLMDependencyUnavailableError, RequestCancelledError, llm_manager


class AgentRuntime:
    def __init__(self, agent: Any):
        self.agent = agent

    def register_request(self, request_id: str):
        with self.agent._request_lock:
            self.agent._cancelled_request_ids.discard(request_id)
            cancel_event = self.agent._request_event_factory()
            self.agent._request_cancellations[request_id] = cancel_event
            return cancel_event

    def clear_request(self, request_id: str) -> None:
        with self.agent._request_lock:
            self.agent._request_cancellations.pop(request_id, None)

    def is_request_cancelled(self, request_id: str) -> bool:
        with self.agent._request_lock:
            if request_id in self.agent._cancelled_request_ids:
                return True
            cancel_event = self.agent._request_cancellations.get(request_id)
            return bool(cancel_event and cancel_event.is_set())

    def is_request_active(self, request_id: str) -> bool:
        with self.agent._request_lock:
            return request_id in self.agent._request_cancellations

    def cancel_request(self, request_id: str) -> str:
        with self.agent._request_lock:
            cancel_event = self.agent._request_cancellations.get(request_id)
            if not cancel_event:
                return f"Request is not active: {request_id}"
            cancel_event.set()
            self.agent._cancelled_request_ids.add(request_id)
        llm_manager.console_event("agent_cancel_requested", request_id=request_id, level=40)
        llm_manager.log_event("Agent cancellation requested by user.", level=40, request_id=request_id)
        return f"Cancellation requested for {request_id}."

    def raise_if_request_cancelled(self, request_id: str) -> None:
        if request_id and self.is_request_cancelled(request_id):
            raise RequestCancelledError(f"Request cancelled: {request_id}")

    def wait_for_graph_result(self, future, request_id: str, timeout_seconds: int):
        start = time.monotonic()
        while True:
            self.raise_if_request_cancelled(request_id)
            elapsed = time.monotonic() - start
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                with self.agent._request_lock:
                    self.agent._cancelled_request_ids.add(request_id)
                    cancel_event = self.agent._request_cancellations.get(request_id)
                    if cancel_event:
                        cancel_event.set()
                future.cancel()
                raise TimeoutError(
                    f"Request timed out after {timeout_seconds} seconds. "
                    "Execution was cancelled logically; inspect snapshots and logs with the request ID."
                )
            try:
                return future.result(timeout=min(0.1, remaining))
            except FuturesTimeoutError:
                continue

    def _build_initial_inputs(self, query: str, request_id: str, session_id: str) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=query)],
            "raw_query": query,
            "plan": [],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": [],
            "failed_tools": {},
            "failed_tool_signals": {},
            "request_id": request_id,
            "session_id": session_id,
            "session_memory_id": 0,
            "domain_label": DEFAULT_DOMAIN_LABEL,
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "waiting_for_user": False,
            "final_response": "",
            "lite_mode": False,
            "normalized_query": query,
            "subtask_feature_cache": {},
            "agent_action": "",
        }

    def _execute_graph(self, inputs: dict[str, Any], request_id: str):
        executor = ThreadPoolExecutor(max_workers=1)
        execution_context = copy_context()
        future = executor.submit(lambda: execution_context.run(self.agent.graph.invoke, inputs))
        try:
            return self.wait_for_graph_result(future, request_id, config.request_timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _is_dependency_unavailable_error(self, exc: Exception) -> bool:
        return (
            isinstance(exc, LLMDependencyUnavailableError)
            or exc.__class__.__name__ == "LLMDependencyUnavailableError"
        )

    def invoke(self, query: str, session_id: str = None):
        if not self.agent.graph:
            return "Graph is not initialized."

        request_id = self.agent._generate_request_id()
        self.agent.last_request_id = request_id
        self.register_request(request_id)
        inputs = None
        request_started_at = time.monotonic()
        try:
            active_session_id = session_id or self.agent.session_id or self.agent.start_session()
            with llm_manager.request_scope(
                request_id,
                session_id=active_session_id,
                cancel_checker=lambda: self.is_request_cancelled(request_id),
            ):
                llm_manager.console_event("agent_started", request_id=request_id)
                llm_manager.log_event(f"Agent request | session_id={active_session_id}\n{query}")
                inputs = self._build_initial_inputs(query, request_id, active_session_id)
                self.agent._persist_state_snapshot(
                    request_id,
                    "request_received",
                    inputs,
                    extra={"query": query},
                )
                try:
                    result = self._execute_graph(inputs, request_id)
                except RequestCancelledError:
                    cancel_message = (
                        f"Request cancelled: {request_id}. "
                        "Execution stopped cooperatively; inspect snapshots and logs if partial work was produced."
                    )
                    llm_manager.console_event("agent_cancelled", request_id=request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request cancelled",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="request_cancelled",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="cancelled",
                    )
                    llm_manager.log_event(cancel_message, level=40, request_id=request_id)
                    self.agent._persist_state_snapshot(
                        request_id,
                        "request_cancelled",
                        inputs,
                        extra={"query": query},
                    )
                    return cancel_message
                except TimeoutError as exc:
                    timeout_message = str(exc)
                    llm_manager.console_event("agent_timeout", request_id=request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request timed out",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="request_timed_out",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="timed_out",
                    )
                    llm_manager.log_event(timeout_message, level=40, request_id=request_id)
                    self.agent._persist_state_snapshot(
                        request_id,
                        "request_timed_out",
                        inputs,
                        extra={"timeout_seconds": config.request_timeout_seconds, "query": query},
                    )
                    return timeout_message
                except LLMDependencyUnavailableError as exc:
                    dependency_message = (
                        f"Request failed because the model dependency is unavailable: {exc}. "
                        "Check the current Python environment, provider service, or installed LLM packages before retrying."
                    )
                    llm_manager.console_event("agent_error", request_id=request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request failed due to unavailable model dependency",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="request_failed",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="failed",
                        error_type="dependency_unavailable",
                        error=str(exc),
                    )
                    llm_manager.log_event(dependency_message, level=40, request_id=request_id)
                    self.agent._persist_state_snapshot(
                        request_id,
                        "request_failed",
                        inputs,
                        extra={"error": str(exc), "error_type": "dependency_unavailable"},
                    )
                    return dependency_message

                final_output = result.get("final_response") or result["messages"][-1].content
                latest_stage = "request_completed"
                if result.get("waiting_for_user"):
                    latest_stage = "agent_waiting_user"
                elif result.get("blocked"):
                    latest_stage = "agent_blocked"
                if latest_stage == "request_completed":
                    self.agent._persist_state_snapshot(
                        request_id,
                        "request_completed",
                        result,
                        extra={"final_output": final_output},
                    )
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request completed",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="request_completed",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="completed",
                    )
                elif latest_stage == "agent_blocked":
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request blocked",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="agent_blocked",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="blocked",
                    )
                else:
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request is waiting for user input",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="agent_waiting_user",
                        source="agent.invoke",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="waiting_user",
                    )
                llm_manager.console_event("agent_finished", request_id=request_id)
                llm_manager.log_event(f"Agent response | session_id={active_session_id}\n{final_output}")
                return final_output
        except KeyboardInterrupt:
            self.cancel_request(request_id)
            cancel_message = (
                f"Request cancelled: {request_id}. "
                "Execution stop was requested by keyboard interrupt."
            )
            llm_manager.console_event("agent_cancelled", request_id=request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Agent request cancelled by keyboard interrupt",
                request_id=request_id,
                session_id=session_id or self.agent.session_id,
                stage="request_cancelled",
                source="agent.invoke",
                duration_ms=(time.monotonic() - request_started_at) * 1000,
                outcome="cancelled",
            )
            llm_manager.log_event(cancel_message, level=40, request_id=request_id)
            if inputs is not None:
                self.agent._persist_state_snapshot(
                    request_id,
                    "request_cancelled",
                    inputs,
                    extra={"query": query, "source": "keyboard_interrupt"},
                )
            return cancel_message
        except LLMDependencyUnavailableError as exc:
            dependency_message = (
                f"Request failed because the model dependency is unavailable: {exc}. "
                "Check the current Python environment, provider service, or installed LLM packages before retrying."
            )
            llm_manager.console_event("agent_error", request_id=request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Agent request failed due to unavailable model dependency",
                request_id=request_id,
                session_id=session_id or self.agent.session_id,
                stage="request_failed",
                source="agent.invoke",
                duration_ms=(time.monotonic() - request_started_at) * 1000,
                outcome="failed",
                error_type="dependency_unavailable",
                error=str(exc),
            )
            llm_manager.log_event(
                dependency_message,
                level=40,
                request_id=request_id,
            )
            self.agent._persist_state_snapshot(
                request_id,
                "request_failed",
                inputs or {},
                extra={"query": query, "error": str(exc), "error_type": "dependency_unavailable"},
            )
            return dependency_message
        except Exception as exc:
            if self._is_dependency_unavailable_error(exc):
                dependency_message = (
                    f"Request failed because the model dependency is unavailable: {exc}. "
                    "Check the current Python environment, provider service, or installed LLM packages before retrying."
                )
                llm_manager.console_event("agent_error", request_id=request_id, level=40)
                llm_manager.log_structured_event(
                    "agent_request",
                    message="Agent request failed due to unavailable model dependency",
                    request_id=request_id,
                    session_id=session_id or self.agent.session_id,
                    stage="request_failed",
                    source="agent.invoke",
                    duration_ms=(time.monotonic() - request_started_at) * 1000,
                    outcome="failed",
                    error_type="dependency_unavailable",
                    error=str(exc),
                )
                llm_manager.log_event(
                    dependency_message,
                    level=40,
                    request_id=request_id,
                )
                self.agent._persist_state_snapshot(
                    request_id,
                    "request_failed",
                    inputs or {},
                    extra={"query": query, "error": str(exc), "error_type": "dependency_unavailable"},
                )
                return dependency_message
            traceback.print_exc()
            llm_manager.console_event("agent_error", request_id=request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Agent request failed",
                request_id=request_id,
                session_id=session_id or self.agent.session_id,
                stage="request_failed",
                source="agent.invoke",
                duration_ms=(time.monotonic() - request_started_at) * 1000,
                outcome="failed",
                error=str(exc),
            )
            llm_manager.log_event(
                f"Agent error | session_id={session_id or self.agent.session_id} | error={exc}",
                level=40,
                request_id=request_id,
            )
            self.agent._persist_state_snapshot(
                request_id,
                "request_failed",
                inputs or {},
                extra={"error": str(exc)},
            )
            return f"Error invoking agent: {exc}"
        finally:
            self.clear_request(request_id)

    def resume_from_snapshot(
        self,
        request_id: str,
        snapshot_name: str = None,
        reroute: bool = False,
        user_followup: str | None = None,
    ):
        payload = self.agent._load_snapshot_payload(request_id, snapshot_name=snapshot_name)
        if not payload:
            return (
                f"Snapshot not found for request_id={request_id}. "
                "Use /list_snapshots <request_id> to inspect available recovery points."
            )

        is_valid, validation_error = self.agent.snapshot_store.validate_snapshot_payload(payload)
        if not is_valid:
            snapshot_path = payload.get("snapshot_path", "") if isinstance(payload, dict) else ""
            llm_manager.log_event(
                f"Snapshot validation failed | source_request_id={request_id} | snapshot_path={snapshot_path} | error={validation_error}",
                level=40,
            )
            return (
                f"Snapshot validation failed for request_id={request_id}: {validation_error} "
                "Resume was aborted to avoid restoring an inconsistent state."
            )

        stored_state = payload.get("state", {})
        if (stored_state.get("blocked") or stored_state.get("waiting_for_user")) and not reroute:
            return stored_state.get("final_response") or "Snapshot is already in a blocked terminal state."
        if stored_state.get("final_response") and stored_state.get("current_subtask_index", 0) >= len(stored_state.get("plan", [])):
            return stored_state.get("final_response")

        new_request_id = self.agent._generate_request_id()
        self.agent.last_request_id = new_request_id
        self.register_request(new_request_id)
        resume_mode = "reroute" if reroute else "continue"
        restored_state = self.agent.snapshot_store.build_resume_state_from_snapshot(
            payload,
            request_id=new_request_id,
            reroute=reroute,
            user_followup=str(user_followup or "").strip(),
        )
        self.agent.session_id = restored_state.get("session_id", self.agent.session_id)

        try:
            with llm_manager.request_scope(
                new_request_id,
                session_id=restored_state.get("session_id", self.agent.session_id),
                cancel_checker=lambda: self.is_request_cancelled(new_request_id),
            ):
                llm_manager.console_event("agent_resumed", request_id=new_request_id)
                llm_manager.log_event(
                    f"Agent resumed from snapshot | source_request_id={request_id} | snapshot_path={payload.get('snapshot_path', '')} | mode={resume_mode}",
                    request_id=new_request_id,
                )
                self.agent._persist_state_snapshot(
                    new_request_id,
                    "resume_requested",
                    restored_state,
                    extra={
                        "source_request_id": request_id,
                        "source_snapshot_path": payload.get("snapshot_path", ""),
                        "resume_mode": resume_mode,
                    },
                )
                try:
                    result = self._execute_graph(restored_state, new_request_id)
                except RequestCancelledError:
                    cancel_message = (
                        f"Resumed request cancelled: {new_request_id}. "
                        "Execution stopped cooperatively; inspect snapshots and logs for partial state."
                    )
                    llm_manager.console_event("agent_cancelled", request_id=new_request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Resumed request cancelled",
                        request_id=new_request_id,
                        session_id=restored_state.get("session_id", self.agent.session_id),
                        stage="request_cancelled",
                        source="agent.resume",
                        duration_ms=None,
                        outcome="cancelled",
                        source_request_id=request_id,
                        resume_mode=resume_mode,
                    )
                    llm_manager.log_event(cancel_message, level=40, request_id=new_request_id)
                    self.agent._persist_state_snapshot(
                        new_request_id,
                        "request_cancelled",
                        restored_state,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                            "resume_mode": resume_mode,
                        },
                    )
                    return cancel_message
                except TimeoutError as exc:
                    timeout_message = str(exc)
                    llm_manager.console_event("agent_timeout", request_id=new_request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Resumed request timed out",
                        request_id=new_request_id,
                        session_id=restored_state.get("session_id", self.agent.session_id),
                        stage="request_timed_out",
                        source="agent.resume",
                        duration_ms=None,
                        outcome="timed_out",
                        source_request_id=request_id,
                        resume_mode=resume_mode,
                    )
                    llm_manager.log_event(timeout_message, level=40, request_id=new_request_id)
                    self.agent._persist_state_snapshot(
                        new_request_id,
                        "request_timed_out",
                        restored_state,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                            "timeout_seconds": config.request_timeout_seconds,
                            "resume_mode": resume_mode,
                        },
                    )
                    return timeout_message
                except LLMDependencyUnavailableError as exc:
                    dependency_message = (
                        f"Resumed request failed because the model dependency is unavailable: {exc}. "
                        "Check the current Python environment, provider service, or installed LLM packages before retrying."
                    )
                    llm_manager.console_event("agent_error", request_id=new_request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Resumed request failed due to unavailable model dependency",
                        request_id=new_request_id,
                        session_id=restored_state.get("session_id", self.agent.session_id),
                        stage="request_failed",
                        source="agent.resume",
                        duration_ms=None,
                        outcome="failed",
                        source_request_id=request_id,
                        resume_mode=resume_mode,
                        error_type="dependency_unavailable",
                        error=str(exc),
                    )
                    llm_manager.log_event(dependency_message, level=40, request_id=new_request_id)
                    self.agent._persist_state_snapshot(
                        new_request_id,
                        "request_failed",
                        restored_state,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                            "resume_mode": resume_mode,
                            "error": str(exc),
                            "error_type": "dependency_unavailable",
                        },
                    )
                    return dependency_message

                final_output = result.get("final_response") or result["messages"][-1].content
                latest_stage = "agent_blocked" if result.get("blocked") else "request_completed"
                if latest_stage == "request_completed":
                    self.agent._persist_state_snapshot(
                        new_request_id,
                        "request_completed",
                        result,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                            "final_output": final_output,
                            "resume_mode": resume_mode,
                        },
                    )
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Resumed request completed",
                        request_id=new_request_id,
                        session_id=restored_state.get("session_id", self.agent.session_id),
                        stage="request_completed",
                        source="agent.resume",
                        duration_ms=None,
                        outcome="completed",
                        source_request_id=request_id,
                        resume_mode=resume_mode,
                    )
                else:
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Resumed request blocked",
                        request_id=new_request_id,
                        session_id=restored_state.get("session_id", self.agent.session_id),
                        stage="agent_blocked",
                        source="agent.resume",
                        duration_ms=None,
                        outcome="blocked",
                        source_request_id=request_id,
                        resume_mode=resume_mode,
                    )
                llm_manager.console_event("agent_finished", request_id=new_request_id)
                llm_manager.log_event(
                    f"Agent resume response | source_request_id={request_id}\n{final_output}",
                    request_id=new_request_id,
                )
                return final_output
        except LLMDependencyUnavailableError as exc:
            dependency_message = (
                f"Resumed request failed because the model dependency is unavailable: {exc}. "
                "Check the current Python environment, provider service, or installed LLM packages before retrying."
            )
            llm_manager.console_event("agent_error", request_id=new_request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Resumed request failed due to unavailable model dependency",
                request_id=new_request_id,
                session_id=restored_state.get("session_id", self.agent.session_id),
                stage="request_failed",
                source="agent.resume",
                duration_ms=None,
                outcome="failed",
                error_type="dependency_unavailable",
                error=str(exc),
                source_request_id=request_id,
                resume_mode=resume_mode,
            )
            llm_manager.log_event(
                dependency_message,
                level=40,
                request_id=new_request_id,
            )
            self.agent._persist_state_snapshot(
                new_request_id,
                "request_failed",
                restored_state,
                extra={
                    "source_request_id": request_id,
                    "source_snapshot_path": payload.get("snapshot_path", ""),
                    "resume_mode": resume_mode,
                    "error": str(exc),
                    "error_type": "dependency_unavailable",
                },
            )
            return dependency_message
        except Exception as exc:
            if self._is_dependency_unavailable_error(exc):
                dependency_message = (
                    f"Resumed request failed because the model dependency is unavailable: {exc}. "
                    "Check the current Python environment, provider service, or installed LLM packages before retrying."
                )
                llm_manager.console_event("agent_error", request_id=new_request_id, level=40)
                llm_manager.log_structured_event(
                    "agent_request",
                    message="Resumed request failed due to unavailable model dependency",
                    request_id=new_request_id,
                    session_id=restored_state.get("session_id", self.agent.session_id),
                    stage="request_failed",
                    source="agent.resume",
                    duration_ms=None,
                    outcome="failed",
                    error_type="dependency_unavailable",
                    error=str(exc),
                    source_request_id=request_id,
                    resume_mode=resume_mode,
                )
                llm_manager.log_event(
                    dependency_message,
                    level=40,
                    request_id=new_request_id,
                )
                self.agent._persist_state_snapshot(
                    new_request_id,
                    "request_failed",
                    restored_state,
                    extra={
                        "source_request_id": request_id,
                        "source_snapshot_path": payload.get("snapshot_path", ""),
                        "resume_mode": resume_mode,
                        "error": str(exc),
                        "error_type": "dependency_unavailable",
                    },
                )
                return dependency_message
            llm_manager.console_event("agent_error", request_id=new_request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Resumed request failed",
                request_id=new_request_id,
                session_id=restored_state.get("session_id", self.agent.session_id),
                stage="request_failed",
                source="agent.resume",
                duration_ms=None,
                outcome="failed",
                error=str(exc),
                source_request_id=request_id,
                resume_mode=resume_mode,
            )
            llm_manager.log_event(
                f"Agent resume error | source_request_id={request_id} | error={exc}",
                level=40,
                request_id=new_request_id,
            )
            self.agent._persist_state_snapshot(
                new_request_id,
                "request_failed",
                restored_state,
                extra={
                    "source_request_id": request_id,
                    "source_snapshot_path": payload.get("snapshot_path", ""),
                    "error": str(exc),
                    "resume_mode": resume_mode,
                },
            )
            return f"Error resuming snapshot: {exc}"
        finally:
            self.clear_request(new_request_id)