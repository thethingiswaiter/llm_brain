import os
import json
import re
import uuid
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from pydantic import ValidationError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from llm_manager import llm_manager, RequestCancelledError
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from config import config

# New Cognitive Imports
from cognitive.feature_extractor import CognitiveSystem
from cognitive.planner import TaskPlanner
from cognitive.reflector import Reflector
from memory.memory_manager import MemoryManager
from mcp_servers.mcp_manager import MCPManager
from skills_md.skill_parser import SkillManager

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: List[Dict[str, Any]]
    current_subtask_index: int
    reflections: List[str]
    global_keywords: List[str]
    failed_tools: Dict[str, List[str]]
    request_id: str
    session_id: str
    session_memory_id: int
    domain_label: str
    memory_summaries: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    replan_counts: Dict[str, int]
    blocked: bool
    final_response: str

class AgentCore:
    def __init__(self):
        self.tools = []
        self.loaded_python_skill_files = set()
        self.loaded_tool_names = set()
        self.graph = None
        self.session_id = self._generate_session_id()
        self.last_request_id = ""
        self._request_cancellations: Dict[str, Event] = {}
        self._request_lock = Lock()
        
        # Instantiate sub-systems
        self.cognitive = CognitiveSystem()
        self.planner = TaskPlanner()
        self.reflector = Reflector()
        self.memory = MemoryManager()
        self.mcp = MCPManager()
        self.skills = SkillManager()
        
        self._auto_load_skills()
        self._auto_load_mcp_servers()
        self._build_graph()

    def _generate_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:12]}"

    def _generate_request_id(self) -> str:
        return f"req_{uuid.uuid4().hex[:12]}"

    def start_session(self, session_id: str = None) -> str:
        self.session_id = session_id or self._generate_session_id()
        return self.session_id

    def _register_request(self, request_id: str) -> Event:
        with self._request_lock:
            cancel_event = Event()
            self._request_cancellations[request_id] = cancel_event
            return cancel_event

    def _clear_request(self, request_id: str) -> None:
        with self._request_lock:
            self._request_cancellations.pop(request_id, None)

    def is_request_cancelled(self, request_id: str) -> bool:
        with self._request_lock:
            cancel_event = self._request_cancellations.get(request_id)
            return bool(cancel_event and cancel_event.is_set())

    def is_request_active(self, request_id: str) -> bool:
        with self._request_lock:
            return request_id in self._request_cancellations

    def cancel_request(self, request_id: str) -> str:
        with self._request_lock:
            cancel_event = self._request_cancellations.get(request_id)
            if not cancel_event:
                return f"Request is not active: {request_id}"
            cancel_event.set()
        llm_manager.console_event("agent_cancel_requested", request_id=request_id, level=40)
        llm_manager.log_event("Agent cancellation requested by user.", level=40, request_id=request_id)
        return f"Cancellation requested for {request_id}."

    def _raise_if_request_cancelled(self, request_id: str) -> None:
        if request_id and self.is_request_cancelled(request_id):
            raise RequestCancelledError(f"Request cancelled: {request_id}")

    def _wait_for_graph_result(self, future, request_id: str, timeout_seconds: int):
        start = time.monotonic()
        while True:
            self._raise_if_request_cancelled(request_id)
            elapsed = time.monotonic() - start
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                future.cancel()
                raise TimeoutError(
                    f"Request timed out after {timeout_seconds} seconds. "
                    "Execution was cancelled logically; inspect snapshots and logs with the request ID."
                )
            try:
                return future.result(timeout=min(0.1, remaining))
            except FuturesTimeoutError:
                continue

    def _snapshot_request_dir(self, request_id: str, create: bool = True) -> str:
        snapshot_dir = config.resolve_path(config.state_snapshot_dir)
        request_dir = os.path.join(snapshot_dir, request_id)
        if create:
            os.makedirs(request_dir, exist_ok=True)
        return request_dir

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = str(content)
        return {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": str(content)[:config.llm_log_max_chars],
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }

    def _deserialize_message(self, payload: Dict[str, Any]) -> BaseMessage:
        message_type = str(payload.get("type", "ai")).lower()
        content = str(payload.get("content", ""))
        tool_calls = payload.get("tool_calls") or []
        if message_type == "human":
            return HumanMessage(content=content)
        if message_type == "system":
            return SystemMessage(content=content)
        if message_type == "tool":
            return ToolMessage(content=content, tool_call_id=payload.get("tool_call_id") or "restored_tool_call")
        return AIMessage(content=content, tool_calls=tool_calls)

    def _serialize_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        return {
            "request_id": state.get("request_id", ""),
            "session_id": state.get("session_id", ""),
            "session_memory_id": state.get("session_memory_id", 0),
            "current_subtask_index": state.get("current_subtask_index", 0),
            "plan": state.get("plan", []),
            "global_keywords": state.get("global_keywords", []),
            "reflections": state.get("reflections", []),
            "failed_tools": state.get("failed_tools", {}),
            "domain_label": state.get("domain_label", ""),
            "memory_summaries": state.get("memory_summaries", []),
            "retry_counts": state.get("retry_counts", {}),
            "replan_counts": state.get("replan_counts", {}),
            "blocked": state.get("blocked", False),
            "final_response": state.get("final_response", ""),
            "messages": [self._serialize_message(message) for message in messages],
        }

    def _persist_state_snapshot(
        self,
        request_id: str,
        stage: str,
        state: Dict[str, Any] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        if not request_id:
            return ""

        request_dir = self._snapshot_request_dir(request_id)
        snapshot_index = len([name for name in os.listdir(request_dir) if name.endswith(".json")]) + 1
        snapshot_path = os.path.join(request_dir, f"{snapshot_index:03d}_{stage}.json")
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "request_id": request_id,
            "state": self._serialize_state_snapshot(state or {}),
            "extra": extra or {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        llm_manager.log_event(
            f"State snapshot persisted | stage={stage} | path={snapshot_path}",
            request_id=request_id,
        )
        return snapshot_path

    def _resolve_snapshot_path(self, request_id: str, snapshot_name: str | None = None) -> str:
        request_dir = self._snapshot_request_dir(request_id, create=False)
        if not os.path.exists(request_dir):
            return ""

        snapshot_files = sorted(
            [name for name in os.listdir(request_dir) if name.endswith(".json")]
        )
        if not snapshot_files:
            return ""

        if snapshot_name:
            selector = os.path.basename(snapshot_name).strip().lower()
            if selector == "latest":
                return os.path.join(request_dir, snapshot_files[-1])

            direct_match = os.path.join(request_dir, os.path.basename(snapshot_name))
            if os.path.exists(direct_match):
                return direct_match

            if selector.isdigit():
                numeric_prefix = f"{int(selector):03d}_"
                for filename in snapshot_files:
                    if filename.startswith(numeric_prefix):
                        return os.path.join(request_dir, filename)

            for filename in reversed(snapshot_files):
                lowered = filename.lower()
                if lowered.endswith(f"_{selector}.json"):
                    return os.path.join(request_dir, filename)

        return os.path.join(request_dir, snapshot_files[-1])

    def list_snapshots(self, request_id: str) -> List[Dict[str, Any]]:
        request_dir = self._snapshot_request_dir(request_id, create=False)
        if not os.path.exists(request_dir):
            return []

        snapshot_files = sorted(
            [name for name in os.listdir(request_dir) if name.endswith(".json")]
        )
        summaries: List[Dict[str, Any]] = []
        for index, filename in enumerate(snapshot_files, start=1):
            snapshot_path = os.path.join(request_dir, filename)
            try:
                with open(snapshot_path, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
            except (OSError, json.JSONDecodeError):
                continue

            state = payload.get("state", {})
            summaries.append(
                {
                    "index": index,
                    "file": filename,
                    "stage": payload.get("stage", ""),
                    "created_at": payload.get("created_at", ""),
                    "subtask_index": state.get("current_subtask_index", 0),
                    "blocked": state.get("blocked", False),
                    "completed": bool(state.get("final_response")) and state.get("current_subtask_index", 0) >= len(state.get("plan", [])),
                }
            )
        return summaries

    def _derive_request_status(self, latest_stage: str, latest_state: Dict[str, Any], active: bool) -> str:
        if active:
            return "active"
        if latest_state.get("blocked"):
            return "blocked"

        stage_to_status = {
            "request_completed": "completed",
            "agent_completed": "completed",
            "request_failed": "failed",
            "request_cancelled": "cancelled",
            "request_timed_out": "timed_out",
            "agent_blocked": "blocked",
        }
        if latest_stage in stage_to_status:
            return stage_to_status[latest_stage]
        if latest_stage:
            return "in_progress"
        return "not_found"

    def _extract_bool_from_event(self, event: Dict[str, Any], key: str) -> bool | None:
        value = event.get(key)
        if isinstance(value, bool):
            return value
        details = str(event.get("details", ""))
        match = re.search(rf"{re.escape(key)}=(True|False|true|false)", details)
        if not match:
            return None
        return match.group(1).lower() == "true"

    def _parse_logged_at(self, value: str) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _build_request_metrics(self, events: List[Dict[str, Any]], latest_state: Dict[str, Any], status: str) -> Dict[str, Any]:
        llm_events = [event for event in events if event.get("event_type") in {"llm_response", "llm_error"}]
        checkpoint_events = [event for event in events if event.get("event_type") == "checkpoint"]
        stage_counts: Dict[str, int] = {}
        for event in checkpoint_events:
            stage = str(event.get("stage", ""))
            if stage:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        logged_times = [self._parse_logged_at(str(event.get("logged_at", ""))) for event in events]
        logged_times = [item for item in logged_times if item is not None]
        total_duration_ms = None
        if len(logged_times) >= 2:
            total_duration_ms = round((max(logged_times) - min(logged_times)).total_seconds() * 1000, 2)

        llm_total_duration_ms = round(
            sum(float(event.get("duration_ms", 0) or 0) for event in llm_events),
            2,
        )
        reflection_failure_count = 0
        for event in checkpoint_events:
            if event.get("stage") != "reflection_completed":
                continue
            success = self._extract_bool_from_event(event, "success")
            if success is False:
                reflection_failure_count += 1

        retry_count = sum(int(value) for value in (latest_state.get("retry_counts", {}) or {}).values())
        tool_started = stage_counts.get("tool_started", 0)
        tool_success = stage_counts.get("tool_succeeded", 0)
        tool_failures = stage_counts.get("tool_failed", 0)
        tool_rejections = stage_counts.get("tool_rejected", 0)
        tool_cancelled = stage_counts.get("tool_cancelled", 0)
        subtask_count = stage_counts.get("subtask_started", 0) or len(latest_state.get("plan", []))

        return {
            "total_duration_ms": total_duration_ms,
            "llm_call_count": sum(1 for event in events if event.get("event_type") == "llm_request"),
            "llm_error_count": sum(1 for event in events if event.get("event_type") == "llm_error"),
            "llm_total_duration_ms": llm_total_duration_ms,
            "tool_call_count": tool_started,
            "tool_success_count": tool_success,
            "tool_failure_count": tool_failures,
            "tool_rejection_count": tool_rejections,
            "tool_cancelled_count": tool_cancelled,
            "tool_hit_rate": round((tool_success / tool_started), 4) if tool_started else 0.0,
            "retry_count": retry_count,
            "subtask_count": subtask_count,
            "reflection_failure_count": reflection_failure_count,
            "blocked_rate": 1.0 if status == "blocked" else 0.0,
        }

    def get_request_summary(self, request_id: str) -> Dict[str, Any] | None:
        snapshots = self.list_snapshots(request_id)
        latest_payload = self._load_snapshot_payload(request_id)
        latest_state = latest_payload.get("state", {}) if latest_payload else {}
        latest_extra = latest_payload.get("extra", {}) if latest_payload else {}
        events = llm_manager.get_request_events(request_id)
        memories = self.memory.list_memories_by_request_id(request_id)

        if not snapshots and not events and not memories and not self.is_request_active(request_id):
            return None

        checkpoints = []
        for event in events:
            stage = event.get("stage")
            if event.get("event_type") != "checkpoint" and not stage:
                continue
            checkpoints.append(
                {
                    "logged_at": event.get("logged_at", ""),
                    "level": event.get("level", ""),
                    "stage": stage or "",
                    "message": event.get("message", ""),
                    "details": event.get("details", ""),
                }
            )

        latest_stage = ""
        if latest_payload:
            latest_stage = latest_payload.get("stage", "")
        elif checkpoints:
            latest_stage = checkpoints[-1].get("stage", "")

        final_response = latest_state.get("final_response") or latest_extra.get("final_output", "")
        status = self._derive_request_status(latest_stage, latest_state, self.is_request_active(request_id))
        session_id = latest_state.get("session_id", "") or (memories[-1].get("conv_id", "") if memories else "")
        metrics = self._build_request_metrics(events, latest_state, status)

        return {
            "request_id": request_id,
            "status": status,
            "active": self.is_request_active(request_id),
            "session_id": session_id,
            "source_request_id": latest_extra.get("source_request_id", ""),
            "latest_stage": latest_stage,
            "created_at": snapshots[0].get("created_at", "") if snapshots else (events[0].get("logged_at", "") if events else ""),
            "updated_at": snapshots[-1].get("created_at", "") if snapshots else (events[-1].get("logged_at", "") if events else ""),
            "subtask_index": latest_state.get("current_subtask_index", 0),
            "plan_length": len(latest_state.get("plan", [])),
            "blocked": latest_state.get("blocked", False),
            "final_response": final_response,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "memory_count": len(memories),
            "memories": memories,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "metrics": metrics,
        }

    def get_recent_request_summaries(self, limit: int = 10) -> List[Dict[str, Any]]:
        snapshot_root = config.resolve_path(config.state_snapshot_dir)
        if not os.path.exists(snapshot_root):
            return []

        request_ids = []
        for name in os.listdir(snapshot_root):
            candidate = os.path.join(snapshot_root, name)
            if os.path.isdir(candidate):
                request_ids.append(name)

        summarized_requests = []
        for request_id in request_ids:
            summary = self.get_request_summary(request_id)
            if not summary:
                continue
            summarized_requests.append(summary)

        summarized_requests.sort(
            key=lambda item: self._parse_logged_at(item.get("updated_at", "")) or datetime.min,
            reverse=True,
        )
        return summarized_requests[: max(0, limit)]

    def _load_snapshot_payload(self, request_id: str, snapshot_name: str | None = None) -> Dict[str, Any] | None:
        snapshot_path = self._resolve_snapshot_path(request_id, snapshot_name=snapshot_name)
        if not snapshot_path:
            return None

        with open(snapshot_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        payload["snapshot_path"] = snapshot_path
        return payload

    def _restore_state_from_snapshot(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        stored_state = payload.get("state", {})
        restored_messages = [
            self._deserialize_message(message_payload)
            for message_payload in stored_state.get("messages", [])
        ]
        return {
            "messages": restored_messages,
            "plan": stored_state.get("plan", []),
            "current_subtask_index": stored_state.get("current_subtask_index", 0),
            "reflections": stored_state.get("reflections", []),
            "global_keywords": stored_state.get("global_keywords", []),
            "request_id": request_id,
            "session_id": stored_state.get("session_id", self.session_id),
            "session_memory_id": stored_state.get("session_memory_id", 0),
            "domain_label": stored_state.get("domain_label", "general"),
            "memory_summaries": stored_state.get("memory_summaries", []),
            "retry_counts": stored_state.get("retry_counts", {}),
            "replan_counts": stored_state.get("replan_counts", {}),
            "blocked": stored_state.get("blocked", False),
            "final_response": stored_state.get("final_response", ""),
        }

    def _plans_are_meaningfully_different(self, original_plan: List[Dict[str, Any]], candidate_plan: List[Dict[str, Any]]) -> bool:
        if not candidate_plan:
            return False
        if len(candidate_plan) != len(original_plan):
            return True

        def normalize(plan: List[Dict[str, Any]]) -> List[tuple[str, str]]:
            normalized = []
            for item in plan:
                normalized.append(
                    (
                        str(item.get("description", "")).strip().lower(),
                        str(item.get("expected_outcome", "")).strip().lower(),
                    )
                )
            return normalized

        return normalize(original_plan) != normalize(candidate_plan)

    def _replan_subtask_after_failure(
        self,
        original_request: str,
        current_subtask: Dict[str, Any],
        actual_output: str,
        reflection_note: str,
        recent_tool_failures: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        failure_lines = []
        for item in recent_tool_failures or []:
            failure_lines.append(
                f"- tool={item.get('tool')} | error_type={item.get('error_type')} | retryable={item.get('retryable')} | message={item.get('message')}"
            )

        replan_prompt_sections = [
            "Replan the failed subtask into a safer sequence of smaller subtasks.",
            f"Original user request: {original_request}",
            f"Failed subtask: {current_subtask.get('description', '')}",
            f"Expected outcome: {current_subtask.get('expected_outcome', '')}",
            f"Observed output: {actual_output}",
            f"Reflection note: {reflection_note}",
            "Planning goal: avoid repeating the exact failed approach; prefer collecting missing information, decomposing the task further, or switching to a safer sequence.",
        ]
        if failure_lines:
            replan_prompt_sections.append("Recent tool failures:\n" + "\n".join(failure_lines))

        replanned = self.planner.split_task("\n\n".join(replan_prompt_sections))
        if not self._plans_are_meaningfully_different([current_subtask], replanned):
            return []
        return replanned

    def _tokenize_text(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

    def _classify_tool_exception(self, exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, (TypeError, ValueError)):
            return "invalid_arguments", False
        if isinstance(exc, TimeoutError):
            return "timeout", True
        if isinstance(exc, (ConnectionError, OSError)):
            return "dependency_unavailable", True
        return "execution_error", True

    def _build_tool_error_payload(self, tool_name: str, error_type: str, retryable: bool, message: str) -> str:
        return json.dumps(
            {
                "ok": False,
                "tool": tool_name,
                "error_type": error_type,
                "retryable": retryable,
                "message": message,
            },
            ensure_ascii=False,
        )

    def _validate_named_argument_heuristics(self, field_name: str, value: Any) -> str:
        normalized_name = field_name.strip().lower()
        if value is None:
            return ""

        if "city" in normalized_name or "location" in normalized_name:
            if not isinstance(value, str) or not value.strip():
                return f"Argument {field_name} must be a non-empty city/location name."
            compact = value.strip()
            if compact.isdigit() or not re.search(r"[a-zA-Z\u4e00-\u9fff]", compact):
                return f"Argument {field_name} does not look like a valid city/location name: {value}"

        date_pattern = r"^\d{4}[-/]\d{2}[-/]\d{2}$"
        time_pattern = r"^\d{2}:\d{2}(:\d{2})?$"
        datetime_pattern = r"^\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(:\d{2})?$"

        if "date" in normalized_name and isinstance(value, str):
            if not re.match(date_pattern, value.strip()):
                return f"Argument {field_name} must use a recognizable date format like YYYY-MM-DD."

        if normalized_name in {"time", "timestamp", "datetime"} or normalized_name.endswith("_time"):
            if isinstance(value, str) and not (
                re.match(time_pattern, value.strip()) or re.match(datetime_pattern, value.strip())
            ):
                return f"Argument {field_name} must use a recognizable time format like HH:MM or YYYY-MM-DD HH:MM:SS."

        return ""

    def _prevalidate_tool_arguments(self, tool_name: str, args_schema, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any] | None, str | None]:
        if not args_schema:
            return kwargs, None

        try:
            if hasattr(args_schema, "model_validate"):
                validated = args_schema.model_validate(kwargs)
                normalized_kwargs = validated.model_dump()
            else:
                validated = args_schema(**kwargs)
                normalized_kwargs = validated.dict()
        except ValidationError as exc:
            return None, self._build_tool_error_payload(
                tool_name,
                "invalid_arguments",
                False,
                f"Argument schema validation failed: {exc}",
            )
        except Exception as exc:
            return None, self._build_tool_error_payload(
                tool_name,
                "invalid_arguments",
                False,
                f"Argument validation failed: {exc}",
            )

        for field_name, value in normalized_kwargs.items():
            heuristic_error = self._validate_named_argument_heuristics(field_name, value)
            if heuristic_error:
                return None, self._build_tool_error_payload(
                    tool_name,
                    "invalid_arguments",
                    False,
                    heuristic_error,
                )

        return normalized_kwargs, None

    def _wrap_tool_for_runtime(self, tool):
        if getattr(tool, "_llm_brain_safe_tool", False):
            return tool

        tool_name = getattr(tool, "name", "runtime_tool")
        description = getattr(tool, "description", "") or ""
        args_schema = getattr(tool, "args_schema", None)
        return_direct = getattr(tool, "return_direct", False)

        def safe_tool_runner(**kwargs):
            request_id = llm_manager.get_request_id()
            self._raise_if_request_cancelled(request_id or "")
            started_at = time.monotonic()
            llm_manager.log_checkpoint(
                "tool_started",
                details=f"tool={tool_name}",
                request_id=request_id,
                console=True,
                tool_name=tool_name,
            )
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                normalized_kwargs, validation_error = self._prevalidate_tool_arguments(tool_name, args_schema, kwargs)
                if validation_error:
                    llm_manager.log_checkpoint(
                        "tool_rejected",
                        details=f"tool={tool_name} | error_type=invalid_arguments",
                        request_id=request_id,
                        level=40,
                        console=True,
                        duration_ms=(time.monotonic() - started_at) * 1000,
                        tool_name=tool_name,
                        error_type="invalid_arguments",
                    )
                    llm_manager.log_event(
                        f"Tool prevalidation rejected call | tool={tool_name} | payload={validation_error}",
                        level=40,
                        request_id=request_id,
                    )
                    return validation_error

                future = executor.submit(lambda: tool.invoke(normalized_kwargs))
                start = time.monotonic()
                while True:
                    self._raise_if_request_cancelled(request_id or "")
                    elapsed = time.monotonic() - start
                    remaining = config.tool_timeout_seconds - elapsed
                    if remaining <= 0:
                        future.cancel()
                        raise TimeoutError(
                            f"Tool {tool_name} timed out after {config.tool_timeout_seconds} seconds."
                        )
                    try:
                        result = future.result(timeout=min(0.1, remaining))
                        break
                    except FuturesTimeoutError:
                        continue
                llm_manager.log_checkpoint(
                    "tool_succeeded",
                    details=f"tool={tool_name}",
                    request_id=request_id,
                    console=True,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    tool_name=tool_name,
                )
                return result
            except RequestCancelledError as exc:
                llm_manager.log_checkpoint(
                    "tool_cancelled",
                    details=f"tool={tool_name}",
                    request_id=request_id,
                    level=40,
                    console=True,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    tool_name=tool_name,
                    error_type="cancelled",
                )
                return json.dumps(
                    {
                        "ok": False,
                        "tool": tool_name,
                        "error_type": "cancelled",
                        "retryable": False,
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                error_type, retryable = self._classify_tool_exception(exc)
                llm_manager.log_checkpoint(
                    "tool_failed",
                    details=f"tool={tool_name} | error_type={error_type}",
                    request_id=request_id,
                    level=40,
                    console=True,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    tool_name=tool_name,
                    error_type=error_type,
                    retryable=retryable,
                )
                llm_manager.log_event(
                    f"Tool execution failed | tool={tool_name} | error_type={error_type} | retryable={retryable} | error={exc}",
                    level=40,
                    request_id=request_id,
                )
                return self._build_tool_error_payload(tool_name, error_type, retryable, str(exc))
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        wrapped_tool = StructuredTool.from_function(
            func=safe_tool_runner,
            name=tool_name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
        )
        setattr(wrapped_tool, "_llm_brain_safe_tool", True)
        setattr(wrapped_tool, "_llm_brain_original_tool", tool)
        return wrapped_tool

    def _select_relevant_memories(self, memories: List[Dict[str, Any]], keywords: List[str], limit: int = 3):
        keyword_set = {kw.strip().lower() for kw in keywords if isinstance(kw, str) and kw.strip()}
        if not memories:
            return []
        if not keyword_set:
            return memories[:limit]

        relevant = []
        for memory_item in memories:
            overlap = memory_item.get("overlap_count")
            if overlap is None:
                memory_keyword_set = {
                    kw.strip().lower() for kw in memory_item.get("keywords", [])
                    if isinstance(kw, str) and kw.strip()
                }
                overlap = len(keyword_set & memory_keyword_set)
            if overlap > 0:
                enriched_item = dict(memory_item)
                enriched_item["overlap_count"] = overlap
                relevant.append(enriched_item)

        relevant.sort(key=lambda item: (item.get("overlap_count", 0), item.get("weight", 0), item.get("id", 0)), reverse=True)
        return relevant[:limit]

    def _parse_tool_error_payload(self, content: str) -> Dict[str, Any] | None:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("ok", True):
            return None
        if not payload.get("tool"):
            return None
        return payload

    def _collect_recent_tool_failures(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            payload = self._parse_tool_error_payload(getattr(message, "content", ""))
            if payload:
                failures.append(payload)
        failures.reverse()
        return failures

    def _merge_failed_tools(
        self,
        failed_tools: Dict[str, List[str]],
        subtask_index: int,
        recent_failures: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        merged = {key: list(value) for key, value in (failed_tools or {}).items()}
        if not recent_failures:
            return merged

        key = str(subtask_index)
        existing = set(merged.get(key, []))
        for item in recent_failures:
            existing.add(str(item.get("tool", "")))
        merged[key] = sorted(name for name in existing if name)
        return merged

    def _filter_failed_tools_for_subtask(
        self,
        subtask_index: int,
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        failed_tools: Dict[str, List[str]],
    ) -> tuple[List[Any], List[Dict[str, Any]], List[str]]:
        failed_names = set((failed_tools or {}).get(str(subtask_index), []))
        if not failed_names:
            return selected_tools, tool_skills, []

        filtered_tools = [tool for tool in selected_tools if getattr(tool, "name", "") not in failed_names]
        filtered_tool_skills = [item for item in tool_skills if item.get("name") not in failed_names]
        return filtered_tools, filtered_tool_skills, sorted(failed_names)

    def _expand_tool_candidates(
        self,
        task_description: str,
        extracted_keywords: List[str],
        failed_tool_names: List[str],
        excluded_tool_names: List[str] | None = None,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        excluded = set(excluded_tool_names or []) | set(failed_tool_names or [])
        ranked_tool_skills = self.skills.find_relevant_tools(
            task_description,
            extracted_keywords,
            limit=max(limit, len(self.skills.loaded_tool_skills) or limit),
        )
        alternatives: List[Dict[str, Any]] = []
        for tool_skill in ranked_tool_skills:
            tool_name = tool_skill.get("name", "")
            if not tool_name or tool_name in excluded:
                continue
            alternatives.append(tool_skill)
            if len(alternatives) >= limit:
                break
        return alternatives

    def _build_tool_reroute_plan(
        self,
        subtask_description: str,
        extracted_keywords: List[str],
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        recent_failures: List[Dict[str, Any]],
        failed_tool_names: List[str],
    ) -> Dict[str, Any]:
        if not recent_failures:
            return {
                "mode": "normal",
                "selected_tools": selected_tools,
                "tool_skills": tool_skills,
                "alternatives": [],
                "reason": "",
            }

        error_types = {str(item.get("error_type", "")) for item in recent_failures}
        retryable_failures = [item for item in recent_failures if bool(item.get("retryable", False))]
        non_retryable_failures = [item for item in recent_failures if not bool(item.get("retryable", False))]

        if non_retryable_failures and "invalid_arguments" in error_types:
            return {
                "mode": "fallback_invalid_arguments",
                "selected_tools": [],
                "tool_skills": [],
                "alternatives": [],
                "reason": "latest tool failures indicate invalid arguments or missing required input",
            }

        alternative_tool_skills = self._expand_tool_candidates(
            subtask_description,
            extracted_keywords,
            failed_tool_names,
            excluded_tool_names=[getattr(tool, "name", "") for tool in selected_tools],
        )
        if alternative_tool_skills:
            return {
                "mode": "alternative_tools",
                "selected_tools": [item["tool"] for item in alternative_tool_skills],
                "tool_skills": alternative_tool_skills,
                "alternatives": [item.get("name", "") for item in alternative_tool_skills],
                "reason": "retryable tool failures triggered a broader alternative-tool search",
            }

        if retryable_failures:
            return {
                "mode": "fallback_retryable_no_alternative",
                "selected_tools": [],
                "tool_skills": [],
                "alternatives": [],
                "reason": "retryable tool failures occurred but no viable alternative tool matched the subtask",
            }

        return {
            "mode": "fallback_no_tools",
            "selected_tools": [],
            "tool_skills": [],
            "alternatives": [],
            "reason": "tool failures left no safe tool route for the current subtask",
        }

    def _format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""

        lines = ["Relevant memory summaries:"]
        for memory_item in memories:
            keywords = ", ".join(memory_item.get("keywords", [])[:5])
            lines.append(
                f"- Memory #{memory_item['id']}: {memory_item.get('summary', '')}"
                f" | keywords: {keywords} | weight: {memory_item.get('weight', 0)}"
            )
        return "\n".join(lines)

    def _load_detailed_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        if not memories or len(memories) > 1:
            return ""

        memory_data = self.memory.load_full_memory(memories[0]["id"])
        if not memory_data:
            return ""

        details = []
        raw_input = memory_data.get("input", "").strip()
        raw_output = memory_data.get("output", "").strip()
        if raw_input:
            details.append(f"Input: {raw_input[:800]}")
        if raw_output:
            details.append(f"Output: {raw_output[:800]}")
        if not details:
            return ""
        return "Relevant memory details:\n" + "\n".join(details)

    def _record_step_memory(self, session_id: str, request_id: str, domain: str, subtask_desc: str,
                            actual: str, reflection_note: str, quality_tags: List[str] | None = None):
        step_keywords, step_summary = self.cognitive.extract_features(subtask_desc)
        memory_output = actual.strip()
        if reflection_note:
            memory_output = f"{memory_output}\n\nReflection: {reflection_note}".strip()
        self.memory.add_memory(
            session_id,
            domain,
            list(step_keywords)[:10],
            f"Step: {step_summary}",
            subtask_desc,
            memory_output,
            "",
            request_id=request_id,
            memory_type="step",
            quality_tags=quality_tags or ["pending"],
        )

    def _finalize_session_memory(self, state: AgentState, final_output: str, quality_tags: List[str] | None = None):
        session_memory_id = state.get("session_memory_id")
        if not session_memory_id:
            return

        reflections = state.get("reflections", [])
        persisted_output = final_output.strip()
        if reflections:
            persisted_output = (
                f"{persisted_output}\n\nReflection summary:\n" + "\n".join(reflections)
            ).strip()
        self.memory.update_memory(
            session_memory_id,
            raw_output=persisted_output,
            request_id=state.get("request_id", ""),
            memory_type="session_main",
            quality_tags=quality_tags or ["success"],
        )

    def _auto_load_skills(self):
        """Automatically scan and load python skills from the skills directory"""
        skills_path = os.path.join(os.path.dirname(__file__), config.skills_dir)
        if not os.path.exists(skills_path):
            os.makedirs(skills_path)
            return

        for filename in os.listdir(skills_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                self._load_python_skill_file(os.path.join(skills_path, filename), rebuild_graph=False)

    def _auto_load_mcp_servers(self):
        """Automatically scan and load MCP servers from the mcp directory"""
        mcp_path = config.resolve_path(config.mcp_dir)
        if not os.path.exists(mcp_path):
            os.makedirs(mcp_path)
            return

        for filename in sorted(os.listdir(mcp_path)):
            if filename.endswith((".json", ".yaml", ".yml")):
                self.load_mcp_server(filename, rebuild_graph=False)

    def select_active_tools(self, query: str):
        """
        Future enhancement: Dynamically filter which tools to pass to the LLM 
        based on the user's query semantics, to prevent token overflow.
        Returns the subset of selected tools.
        """
        capability_bundle = self.skills.assign_capabilities_to_task(query, list(self._tokenize_text(query)))
        selected_tools = capability_bundle.get("tools", [])
        if selected_tools:
            return selected_tools
        return []

    def add_tool(self, tool):
        safe_tool = self._wrap_tool_for_runtime(tool)
        tool_name = getattr(safe_tool, "name", None)
        if tool_name and tool_name in self.loaded_tool_names:
            return False
        self.tools.append(safe_tool)
        if tool_name:
            self.loaded_tool_names.add(tool_name)
        self.skills.register_tool(safe_tool, source_type="runtime")
        self._build_graph()
        return True

    def _load_python_skill_file(self, file_path: str, rebuild_graph: bool = True):
        filename = os.path.basename(file_path)
        if filename in self.loaded_python_skill_files:
            return False, f"Python skill {filename} is already loaded."

        module_name = os.path.splitext(filename)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "tools"):
                return False, f"Skill file {filename} does not export a tools list."

            added_tools = []
            for tool in module.tools:
                safe_tool = self._wrap_tool_for_runtime(tool)
                tool_name = getattr(safe_tool, "name", None)
                if tool_name and tool_name in self.loaded_tool_names:
                    continue
                self.tools.append(safe_tool)
                if tool_name:
                    self.loaded_tool_names.add(tool_name)
                added_tools.append(safe_tool)
            self.skills.register_tools(added_tools, source_type="python", source_file=filename)
            self.loaded_python_skill_files.add(filename)
            if rebuild_graph:
                self._build_graph()
            llm_manager.log_event(
                f"Python skill loaded | file={filename} | tool_count={len(added_tools)} | rebuild_graph={rebuild_graph}"
            )
            return True, f"Loaded Python skill file: {filename}"
        except Exception as e:
            llm_manager.log_event(
                f"Python skill load failed | file={filename} | error={e}",
                level=40,
            )
            return False, f"Failed to load skill {filename}: {e}"

    def load_skill(self, skill_name: str):
        normalized_name = skill_name.strip()
        if not normalized_name:
            return "Usage: /load_skill <skill_name.py|skill_name.md>"

        if normalized_name.endswith(".md"):
            skill = self.skills.load_skill_md(normalized_name)
            if not skill:
                return f"Markdown skill not found: {normalized_name}"
            return f"Loaded Markdown skill: {skill['name']} ({normalized_name})"

        python_name = normalized_name if normalized_name.endswith(".py") else f"{normalized_name}.py"
        python_path = os.path.join(os.path.dirname(__file__), config.skills_dir, python_name)
        if os.path.exists(python_path):
            _, message = self._load_python_skill_file(python_path)
            return message

        markdown_name = normalized_name if normalized_name.endswith(".md") else f"{normalized_name}.md"
        skill = self.skills.load_skill_md(markdown_name)
        if skill:
            return f"Loaded Markdown skill: {skill['name']} ({markdown_name})"

        return f"Skill not found: {skill_name}"

    def load_mcp_server(self, server_ref: str, rebuild_graph: bool = True):
        normalized_ref = server_ref.strip()
        if not normalized_ref:
            return False, "Usage: /load_mcp <config_name.json|config_name.yaml|absolute_path|server.py|stdio:command ...>"

        if normalized_ref.startswith("stdio:"):
            resolved_ref = normalized_ref
        elif os.path.isabs(normalized_ref):
            resolved_ref = normalized_ref
        else:
            direct_path = config.resolve_path(normalized_ref)
            if os.path.exists(direct_path):
                resolved_ref = direct_path
            else:
                resolved_ref = config.resolve_path(os.path.join(config.mcp_dir, normalized_ref))
            if not os.path.exists(resolved_ref):
                mcp_dir = config.resolve_path(config.mcp_dir)
                for extension in (".json", ".yaml", ".yml"):
                    candidate = os.path.join(mcp_dir, normalized_ref + extension)
                    if os.path.exists(candidate):
                        resolved_ref = candidate
                        break

        success, message, _, tools = self.mcp.load_server(resolved_ref)
        if not success:
            llm_manager.log_event(
                f"MCP server load failed | source={resolved_ref} | message={message}",
                level=40,
            )
            return False, message

        added_tools = []
        for tool in tools:
            safe_tool = self._wrap_tool_for_runtime(tool)
            tool_name = getattr(safe_tool, "name", None)
            if tool_name and tool_name in self.loaded_tool_names:
                continue
            self.tools.append(safe_tool)
            if tool_name:
                self.loaded_tool_names.add(tool_name)
            added_tools.append(safe_tool)

        source_label = os.path.basename(resolved_ref) if not normalized_ref.startswith("stdio:") else normalized_ref
        self.skills.register_tools(added_tools, source_type="mcp", source_file=source_label)
        added = len(added_tools)

        if rebuild_graph and added:
            self._build_graph()
        elif rebuild_graph and not added:
            message = f"{message} All declared tools were already registered."

        llm_manager.log_event(
            f"MCP server loaded | source={resolved_ref} | added_tools={added} | rebuild_graph={rebuild_graph} | message={message}"
        )

        return True, message

    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        return self.mcp.list_servers()

    def unload_mcp_server(self, server_ref: str, rebuild_graph: bool = True):
        success, message, server_info = self.mcp.unload_server(server_ref)
        if not success:
            llm_manager.log_event(
                f"MCP server unload failed | source={server_ref} | message={message}",
                level=40,
            )
            return False, message

        removed_names = set(server_info.get("tool_names", []))
        if removed_names:
            self.tools = [tool for tool in self.tools if getattr(tool, "name", None) not in removed_names]
            for tool_name in removed_names:
                self.loaded_tool_names.discard(tool_name)
            self.skills.unregister_tools(list(removed_names))

        if rebuild_graph:
            self._build_graph()

        llm_manager.log_event(
            f"MCP server unloaded | source={server_info.get('source', server_ref)} | removed_tools={len(removed_names)} | rebuild_graph={rebuild_graph} | message={message}"
        )
        return True, message

    def refresh_mcp_server(self, server_ref: str, rebuild_graph: bool = True):
        success, message, server_info, tools = self.mcp.refresh_server(server_ref)
        if not success:
            llm_manager.log_event(
                f"MCP server refresh failed | source={server_ref} | message={message}",
                level=40,
            )
            return False, message

        added_tools = []
        for tool in tools:
            safe_tool = self._wrap_tool_for_runtime(tool)
            tool_name = getattr(safe_tool, "name", None)
            if tool_name and tool_name in self.loaded_tool_names:
                continue
            self.tools.append(safe_tool)
            if tool_name:
                self.loaded_tool_names.add(tool_name)
            added_tools.append(safe_tool)

        source = server_info.get("source", server_ref)
        source_label = os.path.basename(source) if not str(source).startswith("stdio:") else source
        self.skills.register_tools(added_tools, source_type="mcp", source_file=source_label)

        if rebuild_graph:
            self._build_graph()

        llm_manager.log_event(
            f"MCP server refreshed | source={source} | added_tools={len(added_tools)} | rebuild_graph={rebuild_graph} | message={message}"
        )
        return True, message

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)

        def route_from_start(state: AgentState) -> Literal["planner", "agent"]:
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            if plan and idx < len(plan):
                return "agent"
            return "planner"
        
        def initial_planning(state: AgentState):
            msgs = state["messages"]
            last_message_content = msgs[-1].content if msgs else ""
            session_id = state.get("session_id") or self.start_session()
            request_id = state.get("request_id") or self._generate_request_id()
            self._raise_if_request_cancelled(request_id)
            planning_started_at = time.monotonic()
            llm_manager.log_checkpoint(
                "planning_started",
                details=f"session_id={session_id}",
                request_id=request_id,
                console=True,
                session_id=session_id,
            )
            
            # 1. Feature Extraction (Global task constraints: max 30 keywords)
            keywords, summary = self.cognitive.extract_features(last_message_content)
            self._raise_if_request_cancelled(request_id)
            normalized_keywords = list(keywords)[:30]
            domain = self.cognitive.determine_domain(last_message_content)
            memory_summaries = self.memory.retrieve_memory(
                match_keywords=normalized_keywords,
                limit=5,
                exclude_conv_id=session_id,
            )

            session_memory_id = self.memory.add_memory(
                session_id,
                domain,
                normalized_keywords,
                summary,
                last_message_content,
                "",
                "",
                request_id=request_id,
                memory_type="session_main",
                quality_tags=["pending"],
            )
            
            # 2. Planning (Decompose complex tasks into granular subtasks)
            plan = self.planner.split_task(last_message_content)
            self._raise_if_request_cancelled(request_id)
            llm_manager.log_checkpoint(
                "planning_completed",
                details=f"subtask_count={len(plan)} | domain={domain}",
                request_id=request_id,
                console=True,
                session_id=session_id,
                duration_ms=(time.monotonic() - planning_started_at) * 1000,
                subtask_count=len(plan),
                domain=domain,
            )
            
            next_state = {
                "plan": plan,
                "current_subtask_index": 0,
                "global_keywords": normalized_keywords,
                "reflections": [],
                "failed_tools": {},
                "session_id": session_id,
                "request_id": request_id,
                "session_memory_id": session_memory_id,
                "domain_label": domain,
                "memory_summaries": memory_summaries,
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "final_response": "",
            }
            self._persist_state_snapshot(
                request_id,
                "planning_completed",
                next_state,
                extra={"subtask_count": len(plan), "domain": domain},
            )
            return next_state
        
        def call_model_subtask(state: AgentState):
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            self._raise_if_request_cancelled(request_id)
            failed_tools_state = dict(state.get("failed_tools", {}))
            
            if idx >= len(plan):
                final_response = state.get("final_response") or "All subtasks completed successfully."
                return {"messages": [AIMessage(content=final_response)]}
                
            current_subtask = plan[idx]
            subtask_desc = current_subtask.get("description", "")
            recent_tool_failures = self._collect_recent_tool_failures(state.get("messages", []))
            merged_failed_tools = self._merge_failed_tools(failed_tools_state, idx, recent_tool_failures)
            llm_manager.log_checkpoint(
                "subtask_started",
                details=f"index={idx + 1} | description={subtask_desc[:120]}",
                request_id=request_id,
                console=True,
                session_id=state.get("session_id", ""),
                subtask_index=idx + 1,
                subtask_description=subtask_desc[:120],
            )
            
            # Subtask feature extraction: 3-5 keywords
            sub_kws, _ = self.cognitive.extract_features(subtask_desc)
            sub_kws = sub_kws[:5]

            relevant_memories = self._select_relevant_memories(state.get("memory_summaries", []), sub_kws)
            if not relevant_memories:
                relevant_memories = self.memory.retrieve_memory(
                    match_keywords=sub_kws,
                    limit=3,
                    exclude_conv_id=state.get("session_id"),
                    exclude_ids=[state.get("session_memory_id", 0)],
                )

            memory_sections = []
            summary_context = self._format_memory_context(relevant_memories)
            if summary_context:
                memory_sections.append(summary_context)
            detailed_context = self._load_detailed_memory_context(relevant_memories)
            if detailed_context:
                memory_sections.append(detailed_context)
            
            # Assign Skill
            capability_bundle = self.skills.assign_capabilities_to_task(subtask_desc, sub_kws)
            assigned_skill = capability_bundle.get("prompt_skill")
            skill_reason = capability_bundle.get("prompt_skill_reason", "")
            skill_context = ""
            if assigned_skill:
                skill_context = f"\nUse skill: {assigned_skill['name']}\n{assigned_skill['body']}"
                if skill_reason:
                    skill_context += f"\nSkill selection reason: {skill_reason}"
            tool_skills = capability_bundle.get("tool_skills", [])
            selected_tools = capability_bundle.get("tools", [])
            if not selected_tools:
                selected_tools = self.select_active_tools(subtask_desc)
            selected_tools, tool_skills, failed_tool_names = self._filter_failed_tools_for_subtask(
                idx,
                selected_tools,
                tool_skills,
                merged_failed_tools,
            )
            reroute_mode = "normal"
            reroute_plan = {
                "mode": "normal",
                "selected_tools": selected_tools,
                "tool_skills": tool_skills,
                "alternatives": [],
                "reason": "",
            }
            if recent_tool_failures:
                reroute_plan = self._build_tool_reroute_plan(
                    subtask_desc,
                    sub_kws,
                    selected_tools,
                    tool_skills,
                    recent_tool_failures,
                    failed_tool_names,
                )
                reroute_mode = reroute_plan["mode"]
                selected_tools = reroute_plan["selected_tools"]
                tool_skills = reroute_plan["tool_skills"]
                llm_manager.log_checkpoint(
                    "tool_reroute_applied",
                    details=(
                        f"index={idx + 1} | mode={reroute_mode} | failed_tools={','.join(failed_tool_names)}"
                    ),
                    request_id=request_id,
                    console=True,
                )
            
            # Generate local prompt for LLM
            prompt_sections = [
                f"Original user request: {state['messages'][0].content}",
                f"Executing Subtask {idx+1}: {subtask_desc}",
            ]
            if memory_sections:
                prompt_sections.append("\n\n".join(memory_sections))
            if skill_context:
                prompt_sections.append(skill_context.strip())
            if tool_skills:
                tool_context = "Suggested tools:\n" + "\n".join(
                    f"- {item['name']}: {item.get('description', '')} | reason: {item.get('route_reason', '')}" for item in tool_skills
                )
                prompt_sections.append(tool_context)
            if recent_tool_failures:
                failure_context_lines = [
                    f"- {item.get('tool')}: {item.get('error_type')} | retryable={item.get('retryable')} | {item.get('message')}"
                    for item in recent_tool_failures
                ]
                prompt_sections.append(
                    "Recent tool failures:\n" + "\n".join(failure_context_lines)
                )
                if reroute_mode == "alternative_tools" and reroute_plan.get("alternatives"):
                    prompt_sections.append(
                        "Alternative tools selected after reroute:\n" +
                        "\n".join(f"- {name}" for name in reroute_plan["alternatives"])
                    )
                if reroute_plan.get("reason"):
                    prompt_sections.append(f"Reroute decision: {reroute_plan['reason']}")
                if not selected_tools:
                    prompt_sections.append(
                        "Tool-assisted execution is currently unavailable for this subtask. Continue without tools, ask the user for missing parameters when necessary, or produce the best direct answer if possible."
                    )
            prompt = "\n\n".join(prompt_sections)
            messages = state["messages"] + [HumanMessage(content=prompt)]

            llm = llm_manager.get_llm()
            if selected_tools:
                llm = llm.bind_tools(selected_tools)
            self._raise_if_request_cancelled(request_id)
            self._persist_state_snapshot(
                request_id,
                "subtask_prepared",
                state,
                extra={
                    "subtask_index": idx + 1,
                    "subtask_description": subtask_desc,
                    "selected_tools": [getattr(tool, "name", "") for tool in selected_tools],
                    "failed_tools": failed_tool_names,
                },
            )
            llm_manager.log_checkpoint(
                "subtask_llm_dispatch",
                details=f"index={idx + 1} | tool_count={len(selected_tools)}",
                request_id=request_id,
                session_id=state.get("session_id", ""),
                subtask_index=idx + 1,
                tool_count=len(selected_tools),
            )
                
            response = llm_manager.invoke(messages, source="agent.execute_subtask", llm=llm)
            return {"messages": [response], "failed_tools": merged_failed_tools}

        def reflect_and_advance(state: AgentState):
            messages = state["messages"]
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            self._raise_if_request_cancelled(request_id)
            
            if idx >= len(plan):
                return {}
                
            current_subtask = plan[idx]
            expected = current_subtask.get("expected_outcome", "")
            actual = messages[-1].content
            
            # 3. Verification & Reflection
            success, reflection_note, action = self.reflector.verify_and_reflect(
                current_subtask.get("description", ""), expected, actual
            )
            llm_manager.log_checkpoint(
                "reflection_completed",
                details=f"index={idx + 1} | success={success} | action={action}",
                request_id=request_id,
                console=True,
                session_id=state.get("session_id", ""),
                subtask_index=idx + 1,
                success=success,
                action=action,
            )
            
            reflections = list(state.get("reflections", []))
            reflections.append(f"Subtask {idx+1}: {reflection_note}")
            failed_tools_state = dict(state.get("failed_tools", {}))
            self._record_step_memory(
                state.get("session_id", self.session_id),
                request_id,
                state.get("domain_label", "general"),
                current_subtask.get("description", ""),
                actual,
                reflection_note,
                quality_tags=["success"] if (success or action == "continue") else (["ask_user", "blocked"] if action == "ask_user" else ["retry"]),
            )

            retry_counts = dict(state.get("retry_counts", {}))
            replan_counts = dict(state.get("replan_counts", {}))
            retry_key = str(idx)
            retry_count = retry_counts.get(retry_key, 0)
            
            if not success and action == "ask_user":
                blocked_message = (
                    f"Error blocked subtask. Reflection analysis: {reflection_note}\nNeed user intervention."
                )
                llm_manager.log_checkpoint(
                    "agent_blocked",
                    details=f"index={idx + 1} | action=ask_user",
                    request_id=request_id,
                    level=40,
                    console=True,
                )
                self._persist_state_snapshot(
                    request_id,
                    "agent_blocked",
                    state,
                    extra={"subtask_index": idx + 1, "action": action, "reflection_note": reflection_note},
                )
                self._finalize_session_memory(state, blocked_message, quality_tags=["ask_user", "blocked"])
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": True,
                    "final_response": blocked_message,
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                }
            
            # Advance to next subtask if successful or if action is 'continue'
            if success or action == "continue":
                next_index = idx + 1
                failed_tools_state.pop(str(idx), None)
                if next_index >= len(plan):
                    llm_manager.log_checkpoint(
                        "agent_completed",
                        details=f"subtask_count={len(plan)}",
                        request_id=request_id,
                        console=True,
                    )
                    self._finalize_session_memory(state, actual, quality_tags=["success"])
                    self._persist_state_snapshot(
                        request_id,
                        "agent_completed",
                        state,
                        extra={"subtask_count": len(plan), "final_output": actual},
                    )
                else:
                    llm_manager.log_checkpoint(
                        "subtask_advanced",
                        details=f"next_index={next_index + 1}",
                        request_id=request_id,
                    )
                    self._persist_state_snapshot(
                        request_id,
                        "subtask_advanced",
                        state,
                        extra={"next_index": next_index},
                    )
                return {
                    "current_subtask_index": next_index,
                    "reflections": reflections,
                    "failed_tools": failed_tools_state,
                    "blocked": False,
                    "final_response": actual if next_index >= len(plan) else state.get("final_response", ""),
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                }
            
            retry_count += 1
            retry_counts[retry_key] = retry_count
            replan_key = str(idx)
            replan_count = replan_counts.get(replan_key, 0)
            recent_tool_failures = self._collect_recent_tool_failures(messages)

            if retry_count == 1 and replan_count < 1:
                replanned_subtasks = self._replan_subtask_after_failure(
                    state["messages"][0].content,
                    current_subtask,
                    actual,
                    reflection_note,
                    recent_tool_failures=recent_tool_failures,
                )
                if replanned_subtasks:
                    updated_plan = list(plan[:idx]) + replanned_subtasks + list(plan[idx + 1:])
                    replan_counts[replan_key] = replan_count + 1
                    retry_counts.pop(retry_key, None)
                    failed_tools_state.pop(str(idx), None)
                    llm_manager.log_checkpoint(
                        "subtask_replanned",
                        details=f"index={idx + 1} | new_subtask_count={len(replanned_subtasks)}",
                        request_id=request_id,
                        console=True,
                        session_id=state.get("session_id", ""),
                        subtask_index=idx + 1,
                        new_subtask_count=len(replanned_subtasks),
                    )
                    self._persist_state_snapshot(
                        request_id,
                        "subtask_replanned",
                        {
                            **state,
                            "plan": updated_plan,
                            "reflections": reflections,
                            "retry_counts": retry_counts,
                            "replan_counts": replan_counts,
                            "failed_tools": failed_tools_state,
                        },
                        extra={
                            "subtask_index": idx + 1,
                            "replacement_count": len(replanned_subtasks),
                            "reflection_note": reflection_note,
                        },
                    )
                    return {
                        "plan": updated_plan,
                        "reflections": reflections,
                        "failed_tools": failed_tools_state,
                        "retry_counts": retry_counts,
                        "replan_counts": replan_counts,
                        "blocked": False,
                    }

            if retry_count >= 2:
                blocked_message = (
                    f"Subtask {idx+1} exceeded retry limit. Reflection analysis: {reflection_note}\n"
                    "Need user intervention."
                )
                llm_manager.log_checkpoint(
                    "agent_blocked",
                    details=f"index={idx + 1} | action=retry_limit",
                    request_id=request_id,
                    level=40,
                    console=True,
                )
                self._persist_state_snapshot(
                    request_id,
                    "agent_blocked",
                    state,
                    extra={"subtask_index": idx + 1, "action": "retry_limit", "reflection_note": reflection_note},
                )
                self._finalize_session_memory(state, blocked_message, quality_tags=["blocked", "retry"])
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": True,
                    "final_response": blocked_message,
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                }

            return {
                "reflections": reflections,
                "failed_tools": failed_tools_state,
                "retry_counts": retry_counts,
                "replan_counts": replan_counts,
                "blocked": False,
            }

        def should_continue(state: AgentState) -> Literal["tools", "reflect_and_advance", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            
            if last_message.tool_calls:
                return "tools"
            
            return "reflect_and_advance"

        def should_continue_after_reflection(state: AgentState) -> Literal["agent", "__end__"]:
            if state.get("blocked"):
                return "__end__"

            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            if idx >= len(plan):
                return "__end__"
            return "agent"
            
        # Add nodes
        graph_builder.add_node("planner", initial_planning)
        graph_builder.add_node("agent", call_model_subtask)
        graph_builder.add_node("reflect_and_advance", reflect_and_advance)
        if self.tools:
            tool_node = ToolNode(self.tools)
            graph_builder.add_node("tools", tool_node)
            
        # Add edges
        graph_builder.add_conditional_edges(START, route_from_start, {
            "planner": "planner",
            "agent": "agent",
        })
        graph_builder.add_edge("planner", "agent")
        graph_builder.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "reflect_and_advance": "reflect_and_advance",
            "__end__": END
        })
        if self.tools:
            graph_builder.add_edge("tools", "agent")

        graph_builder.add_conditional_edges("reflect_and_advance", should_continue_after_reflection, {
            "agent": "agent",
            "__end__": END
        })
        
        self.graph = graph_builder.compile()

    def invoke(self, query: str, session_id: str = None):
        if not self.graph:
            return "Graph is not initialized."
        request_id = self._generate_request_id()
        self.last_request_id = request_id
        self._register_request(request_id)
        inputs = None
        request_started_at = time.monotonic()
        try:
            active_session_id = session_id or self.session_id or self.start_session()
            with llm_manager.request_scope(request_id, cancel_checker=lambda: self.is_request_cancelled(request_id)):
                llm_manager.console_event("agent_started", request_id=request_id)
                llm_manager.log_event(
                    f"Agent request | session_id={active_session_id}\n{query}"
                )
                inputs = {
                    "messages": [HumanMessage(content=query)],
                    "plan": [],
                    "current_subtask_index": 0,
                    "reflections": [],
                    "global_keywords": [],
                    "failed_tools": {},
                    "request_id": request_id,
                    "session_id": active_session_id,
                    "session_memory_id": 0,
                    "domain_label": "general",
                    "memory_summaries": [],
                    "retry_counts": {},
                    "replan_counts": {},
                    "blocked": False,
                    "final_response": "",
                }
                self._persist_state_snapshot(
                    request_id,
                    "request_received",
                    inputs,
                    extra={"query": query},
                )
                executor = ThreadPoolExecutor(max_workers=1)
                execution_context = copy_context()
                future = executor.submit(lambda: execution_context.run(self.graph.invoke, inputs))
                try:
                    res = self._wait_for_graph_result(future, request_id, config.request_timeout_seconds)
                except RequestCancelledError:
                    future.cancel()
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
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="cancelled",
                    )
                    llm_manager.log_event(cancel_message, level=40, request_id=request_id)
                    self._persist_state_snapshot(
                        request_id,
                        "request_cancelled",
                        inputs,
                        extra={"query": query},
                    )
                    return cancel_message
                except TimeoutError as exc:
                    future.cancel()
                    timeout_message = str(exc)
                    llm_manager.console_event("agent_timeout", request_id=request_id, level=40)
                    llm_manager.log_structured_event(
                        "agent_request",
                        message="Agent request timed out",
                        request_id=request_id,
                        session_id=active_session_id,
                        stage="request_timed_out",
                        duration_ms=(time.monotonic() - request_started_at) * 1000,
                        outcome="timed_out",
                    )
                    llm_manager.log_event(
                        timeout_message,
                        level=40,
                        request_id=request_id,
                    )
                    self._persist_state_snapshot(
                        request_id,
                        "request_timed_out",
                        inputs,
                        extra={"timeout_seconds": config.request_timeout_seconds, "query": query},
                    )
                    return timeout_message
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
                final_output = res["messages"][-1].content
                self._persist_state_snapshot(
                    request_id,
                    "request_completed",
                    res,
                    extra={"final_output": final_output},
                )
                llm_manager.console_event("agent_finished", request_id=request_id)
                llm_manager.log_structured_event(
                    "agent_request",
                    message="Agent request completed",
                    request_id=request_id,
                    session_id=active_session_id,
                    stage="request_completed",
                    duration_ms=(time.monotonic() - request_started_at) * 1000,
                    outcome="completed",
                )
                llm_manager.log_event(
                    f"Agent response | session_id={active_session_id}\n{final_output}"
                )
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
                session_id=session_id or self.session_id,
                stage="request_cancelled",
                duration_ms=(time.monotonic() - request_started_at) * 1000,
                outcome="cancelled",
            )
            llm_manager.log_event(cancel_message, level=40, request_id=request_id)
            if inputs is not None:
                self._persist_state_snapshot(
                    request_id,
                    "request_cancelled",
                    inputs,
                    extra={"query": query, "source": "keyboard_interrupt"},
                )
            return cancel_message
        except Exception as e:
            import traceback
            traceback.print_exc()
            llm_manager.console_event("agent_error", request_id=request_id, level=40)
            llm_manager.log_structured_event(
                "agent_request",
                message="Agent request failed",
                request_id=request_id,
                session_id=session_id or self.session_id,
                stage="request_failed",
                duration_ms=(time.monotonic() - request_started_at) * 1000,
                outcome="failed",
                error=str(e),
            )
            llm_manager.log_event(
                f"Agent error | session_id={session_id or self.session_id} | error={e}",
                level=40,
                request_id=request_id,
            )
            self._persist_state_snapshot(
                request_id,
                "request_failed",
                inputs or {},
                extra={"error": str(e)},
            )
            return f"Error invoking agent: {e}"
        finally:
            self._clear_request(request_id)

    def get_last_request_id(self) -> str:
        return self.last_request_id

    def resume_from_snapshot(self, request_id: str, snapshot_name: str = None):
        payload = self._load_snapshot_payload(request_id, snapshot_name=snapshot_name)
        if not payload:
            return (
                f"Snapshot not found for request_id={request_id}. "
                "Use /list_snapshots <request_id> to inspect available recovery points."
            )

        stored_state = payload.get("state", {})
        if stored_state.get("blocked"):
            return stored_state.get("final_response") or "Snapshot is already in a blocked terminal state."
        if stored_state.get("final_response") and stored_state.get("current_subtask_index", 0) >= len(stored_state.get("plan", [])):
            return stored_state.get("final_response")

        new_request_id = self._generate_request_id()
        self.last_request_id = new_request_id
        self._register_request(new_request_id)
        restored_state = self._restore_state_from_snapshot(payload, request_id=new_request_id)
        self.session_id = restored_state.get("session_id", self.session_id)

        try:
            with llm_manager.request_scope(new_request_id, cancel_checker=lambda: self.is_request_cancelled(new_request_id)):
                llm_manager.console_event("agent_resumed", request_id=new_request_id)
                llm_manager.log_event(
                    f"Agent resumed from snapshot | source_request_id={request_id} | snapshot_path={payload.get('snapshot_path', '')}",
                    request_id=new_request_id,
                )
                self._persist_state_snapshot(
                    new_request_id,
                    "resume_requested",
                    restored_state,
                    extra={
                        "source_request_id": request_id,
                        "source_snapshot_path": payload.get("snapshot_path", ""),
                    },
                )

                executor = ThreadPoolExecutor(max_workers=1)
                execution_context = copy_context()
                future = executor.submit(lambda: execution_context.run(self.graph.invoke, restored_state))
                try:
                    result = self._wait_for_graph_result(future, new_request_id, config.request_timeout_seconds)
                except RequestCancelledError:
                    future.cancel()
                    cancel_message = (
                        f"Resumed request cancelled: {new_request_id}. "
                        "Execution stopped cooperatively; inspect snapshots and logs for partial state."
                    )
                    llm_manager.console_event("agent_cancelled", request_id=new_request_id, level=40)
                    llm_manager.log_event(cancel_message, level=40, request_id=new_request_id)
                    self._persist_state_snapshot(
                        new_request_id,
                        "request_cancelled",
                        restored_state,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                        },
                    )
                    return cancel_message
                except TimeoutError as exc:
                    future.cancel()
                    timeout_message = str(exc)
                    llm_manager.console_event("agent_timeout", request_id=new_request_id, level=40)
                    llm_manager.log_event(timeout_message, level=40, request_id=new_request_id)
                    self._persist_state_snapshot(
                        new_request_id,
                        "request_timed_out",
                        restored_state,
                        extra={
                            "source_request_id": request_id,
                            "source_snapshot_path": payload.get("snapshot_path", ""),
                            "timeout_seconds": config.request_timeout_seconds,
                        },
                    )
                    return timeout_message
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                final_output = result["messages"][-1].content
                self._persist_state_snapshot(
                    new_request_id,
                    "request_completed",
                    result,
                    extra={
                        "source_request_id": request_id,
                        "source_snapshot_path": payload.get("snapshot_path", ""),
                        "final_output": final_output,
                    },
                )
                llm_manager.console_event("agent_finished", request_id=new_request_id)
                llm_manager.log_event(
                    f"Agent resume response | source_request_id={request_id}\n{final_output}",
                    request_id=new_request_id,
                )
                return final_output
        except Exception as exc:
            llm_manager.console_event("agent_error", request_id=new_request_id, level=40)
            llm_manager.log_event(
                f"Agent resume error | source_request_id={request_id} | error={exc}",
                level=40,
                request_id=new_request_id,
            )
            self._persist_state_snapshot(
                new_request_id,
                "request_failed",
                restored_state,
                extra={
                    "source_request_id": request_id,
                    "source_snapshot_path": payload.get("snapshot_path", ""),
                    "error": str(exc),
                },
            )
            return f"Error resuming snapshot: {exc}"
        finally:
            self._clear_request(new_request_id)

    def replay(self, memory_id: int, injected_features: list[str] = None):
        """
        重现能力 (Replay)
        """
        memory_data = self.memory.load_full_memory(memory_id)
        if not memory_data:
            return "Memory not found."
        
        raw_input = memory_data.get("input", "")
        if injected_features:
            raw_input = f"[Injected features: {', '.join(injected_features)}]\n" + raw_input
        llm_manager.log_checkpoint(
            "replay_started",
            details=f"memory_id={memory_id} | injected_features={len(injected_features or [])}",
        )
        return self.invoke(raw_input)

    def convert_memory_to_skill(self, memory_id: int):
        """
        记忆-技能转化
        """
        import sqlite3
        import json
        import os
        conn = sqlite3.connect(self.memory.db_path)
        c = conn.cursor()
        c.execute("SELECT summary, keywords FROM interactions WHERE id = ?", (memory_id,))
        res = c.fetchone()
        conn.close()
        
        if not res:
            return "Memory not found."
            
        summary, keywords = res
        keywords_list = json.loads(keywords)
        memory_data = self.memory.load_full_memory(memory_id)
        if not memory_data:
            return "Memory details not found."

        logic = memory_data.get("output", "None")

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", summary).strip("_").lower()[:20]
        if not slug:
            slug = "memory_skill"

        name = f"{slug}_{memory_id}"
        md_content = f"""---
name: "{name}"
confidence: 40
keywords: {json.dumps(keywords_list)}
description: "{summary}"
entry_node: "main"
---
{logic}
"""
        filepath = os.path.join(self.skills.skills_md_dir, f"{name}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.skills.load_skill_md(f"{name}.md", force_reload=True)
        return f"Successfully converted memory {memory_id} to skill {name}.md"

agent = AgentCore()
