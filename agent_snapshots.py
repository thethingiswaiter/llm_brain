import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import config
from cognitive.feature_extractor import DEFAULT_DOMAIN_LABEL
from llm_manager import llm_manager


SNAPSHOT_SCHEMA_VERSION = 1
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = {0, SNAPSHOT_SCHEMA_VERSION}


class AgentSnapshotStore:
    def __init__(self, agent: Any):
        self.agent = agent

    def snapshot_request_dir(self, request_id: str, create: bool = True) -> str:
        snapshot_dir = config.resolve_path(config.state_snapshot_dir)
        request_dir = os.path.join(snapshot_dir, request_id)
        if create:
            os.makedirs(request_dir, exist_ok=True)
        return request_dir

    def serialize_message(self, message: BaseMessage) -> dict[str, Any]:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = str(content)
        return {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": str(content)[:config.llm_log_max_chars],
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }

    def deserialize_message(self, payload: dict[str, Any]) -> BaseMessage:
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

    def serialize_state_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
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
            "failed_tool_signals": state.get("failed_tool_signals", {}),
            "domain_label": state.get("domain_label", ""),
            "memory_summaries": state.get("memory_summaries", []),
            "retry_counts": state.get("retry_counts", {}),
            "replan_counts": state.get("replan_counts", {}),
            "blocked": state.get("blocked", False),
            "final_response": state.get("final_response", ""),
            "messages": [self.serialize_message(message) for message in messages],
        }

    def normalize_plan(self, plan: Any) -> list[dict[str, Any]]:
        if not isinstance(plan, list):
            return []

        normalized_plan: list[dict[str, Any]] = []
        for index, item in enumerate(plan, start=1):
            if not isinstance(item, dict):
                continue
            normalized_plan.append(
                {
                    "id": item.get("id", index),
                    "description": str(item.get("description", "")).strip(),
                    "expected_outcome": str(item.get("expected_outcome", "")).strip(),
                }
            )
        return normalized_plan

    def normalize_message_payloads(self, messages: Any, fallback_query: str = "") -> list[dict[str, Any]]:
        normalized_messages: list[dict[str, Any]] = []
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, dict):
                    message_type = str(item.get("type", "ai") or "ai").lower()
                    content = item.get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    normalized_messages.append(
                        {
                            "type": message_type,
                            "content": content,
                            "tool_calls": item.get("tool_calls"),
                            "tool_call_id": item.get("tool_call_id"),
                        }
                    )
                elif isinstance(item, str):
                    normalized_messages.append(
                        {
                            "type": "ai",
                            "content": item,
                            "tool_calls": None,
                            "tool_call_id": None,
                        }
                    )

        if not normalized_messages and fallback_query.strip():
            normalized_messages.append(
                {
                    "type": "human",
                    "content": fallback_query.strip(),
                    "tool_calls": None,
                    "tool_call_id": None,
                }
            )
        return normalized_messages

    def migrate_snapshot_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        migrated = deepcopy(payload)
        schema_version = migrated.get("schema_version", 0)
        if not isinstance(schema_version, int):
            return migrated
        if schema_version not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
            return migrated

        state = migrated.setdefault("state", {})
        extra = migrated.setdefault("extra", {})
        migration_notes: list[str] = []

        fallback_query = str(extra.get("query", "") or "")
        state["plan"] = self.normalize_plan(state.get("plan", []))
        state["messages"] = self.normalize_message_payloads(state.get("messages", []), fallback_query=fallback_query)
        if state["messages"] and fallback_query.strip() and schema_version == 0:
            migration_notes.append("backfilled_messages_from_query")

        default_lists = {
            "reflections": [],
            "global_keywords": [],
            "memory_summaries": [],
        }
        for field_name, default_value in default_lists.items():
            value = state.get(field_name, default_value)
            state[field_name] = value if isinstance(value, list) else list(default_value)

        default_dicts = {
            "failed_tools": {},
            "failed_tool_signals": {},
            "retry_counts": {},
            "replan_counts": {},
        }
        for field_name, default_value in default_dicts.items():
            value = state.get(field_name, default_value)
            if isinstance(value, dict):
                state[field_name] = {str(key): value[key] for key in value}
            else:
                state[field_name] = dict(default_value)

        integer_fields = {
            "session_memory_id": 0,
            "current_subtask_index": 0,
        }
        for field_name, default_value in integer_fields.items():
            value = state.get(field_name, default_value)
            try:
                state[field_name] = max(0, int(value))
            except (TypeError, ValueError):
                state[field_name] = default_value

        state["request_id"] = str(state.get("request_id", migrated.get("request_id", "")) or "")
        state["session_id"] = str(state.get("session_id", self.agent.session_id or "") or "")
        state["domain_label"] = str(state.get("domain_label", DEFAULT_DOMAIN_LABEL) or DEFAULT_DOMAIN_LABEL)
        state["final_response"] = str(state.get("final_response", "") or "")
        state["blocked"] = bool(state.get("blocked", False))

        if schema_version == 0:
            migrated["schema_version"] = SNAPSHOT_SCHEMA_VERSION
            migrated["migrated_from_version"] = 0
            if migration_notes:
                migrated["migration_notes"] = migration_notes

        return migrated

    def persist_state_snapshot(
        self,
        request_id: str,
        stage: str,
        state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if not request_id:
            return ""

        request_dir = self.snapshot_request_dir(request_id)
        snapshot_index = len([name for name in os.listdir(request_dir) if name.endswith(".json")]) + 1
        snapshot_path = os.path.join(request_dir, f"{snapshot_index:03d}_{stage}.json")
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "request_id": request_id,
            "state": self.serialize_state_snapshot(state or {}),
            "extra": extra or {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        llm_manager.log_event(
            f"State snapshot persisted | stage={stage} | path={snapshot_path}",
            request_id=request_id,
        )
        if hasattr(self.agent, "retention"):
            self.agent.retention.maybe_auto_prune(trigger="snapshot")
        return snapshot_path

    def resolve_snapshot_path(self, request_id: str, snapshot_name: str | None = None) -> str:
        request_dir = self.snapshot_request_dir(request_id, create=False)
        if not os.path.exists(request_dir):
            return ""

        snapshot_files = sorted([name for name in os.listdir(request_dir) if name.endswith(".json")])
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

    def list_snapshots(self, request_id: str) -> list[dict[str, Any]]:
        request_dir = self.snapshot_request_dir(request_id, create=False)
        if not os.path.exists(request_dir):
            return []

        snapshot_files = sorted([name for name in os.listdir(request_dir) if name.endswith(".json")])
        summaries: list[dict[str, Any]] = []
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
                    "completed": bool(state.get("final_response"))
                    and state.get("current_subtask_index", 0) >= len(state.get("plan", [])),
                }
            )
        return summaries

    def load_snapshot_payload(self, request_id: str, snapshot_name: str | None = None) -> dict[str, Any] | None:
        snapshot_path = self.resolve_snapshot_path(request_id, snapshot_name=snapshot_name)
        if not snapshot_path:
            return None

        with open(snapshot_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        payload.setdefault("schema_version", 0)
        migrated_payload = self.migrate_snapshot_payload(payload)
        migrated_payload["snapshot_path"] = snapshot_path
        return migrated_payload

    def validate_snapshot_payload(self, payload: dict[str, Any] | None) -> tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "Snapshot payload is not a JSON object."

        schema_version = payload.get("schema_version", 0)
        if not isinstance(schema_version, int):
            return False, "Snapshot schema_version must be an integer."
        if schema_version not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
            return False, f"Unsupported snapshot schema_version: {schema_version}."

        stage = payload.get("stage", "")
        if not isinstance(stage, str) or not stage.strip():
            return False, "Snapshot stage is missing or invalid."

        state = payload.get("state")
        if not isinstance(state, dict):
            return False, "Snapshot state is missing or invalid."

        if not isinstance(state.get("messages", []), list):
            return False, "Snapshot messages must be a list."
        if not isinstance(state.get("plan", []), list):
            return False, "Snapshot plan must be a list."

        current_subtask_index = state.get("current_subtask_index", 0)
        if not isinstance(current_subtask_index, int) or current_subtask_index < 0:
            return False, "Snapshot current_subtask_index must be a non-negative integer."
        if current_subtask_index > len(state.get("plan", [])):
            return False, "Snapshot current_subtask_index exceeds plan length."

        required_dict_fields = ("failed_tools", "failed_tool_signals", "retry_counts", "replan_counts")
        for field_name in required_dict_fields:
            field_value = state.get(field_name, {})
            if not isinstance(field_value, dict):
                return False, f"Snapshot field {field_name} must be a dictionary."

        scalar_expectations = {
            "request_id": str,
            "session_id": str,
            "session_memory_id": int,
            "domain_label": str,
            "blocked": bool,
            "final_response": str,
        }
        for field_name, expected_type in scalar_expectations.items():
            field_value = state.get(field_name, "" if expected_type is str else 0 if expected_type is int else False)
            if not isinstance(field_value, expected_type):
                return False, f"Snapshot field {field_name} must be of type {expected_type.__name__}."

        for message_payload in state.get("messages", []):
            if not isinstance(message_payload, dict):
                return False, "Snapshot message entries must be JSON objects."
            message_type = message_payload.get("type", "")
            if not isinstance(message_type, str) or not message_type:
                return False, "Snapshot message type is missing or invalid."
            content = message_payload.get("content", "")
            if not isinstance(content, str):
                return False, "Snapshot message content must be a string."

        for plan_item in state.get("plan", []):
            if not isinstance(plan_item, dict):
                return False, "Snapshot plan entries must be JSON objects."
            description = plan_item.get("description", "")
            expected_outcome = plan_item.get("expected_outcome", "")
            if not isinstance(description, str) or not description.strip():
                return False, "Snapshot plan entry description is missing or invalid."
            if not isinstance(expected_outcome, str):
                return False, "Snapshot plan entry expected_outcome must be a string."

        plan = state.get("plan", [])
        final_response = state.get("final_response", "")
        blocked = state.get("blocked", False)
        is_terminal_completed = bool(final_response) and current_subtask_index >= len(plan)
        is_resumable = not blocked and not is_terminal_completed
        if is_resumable and not state.get("messages", []):
            return False, "Snapshot does not contain resumable message history."
        if blocked and not final_response:
            return False, "Blocked snapshot must include final_response for terminal state."
        if is_resumable and state.get("messages"):
            first_message_type = str(state["messages"][0].get("type", "")).lower()
            if first_message_type not in {"human", "system"}:
                return False, "Resumable snapshot must begin with a human or system message."

        for field_name in ("retry_counts", "replan_counts", "failed_tools", "failed_tool_signals"):
            for raw_key in state.get(field_name, {}).keys():
                try:
                    key_index = int(str(raw_key))
                except (TypeError, ValueError):
                    return False, f"Snapshot field {field_name} contains a non-numeric subtask key: {raw_key}."
                if key_index < 0 or key_index >= len(plan):
                    return False, f"Snapshot field {field_name} references out-of-range subtask index: {raw_key}."

        for raw_index, tool_map in state.get("failed_tool_signals", {}).items():
            if not isinstance(tool_map, dict):
                return False, f"Snapshot field failed_tool_signals[{raw_index}] must be a dictionary."
            for tool_name, signal_payload in tool_map.items():
                if not isinstance(tool_name, str) or not tool_name.strip():
                    return False, f"Snapshot field failed_tool_signals[{raw_index}] contains an invalid tool key."
                if not isinstance(signal_payload, dict):
                    return False, f"Snapshot field failed_tool_signals[{raw_index}][{tool_name}] must be a dictionary."

        if current_subtask_index >= len(plan) and not blocked and not final_response:
            return False, "Snapshot exhausted the plan but does not contain a terminal response."
        if final_response and not blocked and current_subtask_index < len(plan):
            return False, "Snapshot contains final_response before all planned subtasks were completed."

        return True, ""

    def restore_state_from_snapshot(self, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        stored_state = payload.get("state", {})
        restored_messages = [
            self.deserialize_message(message_payload)
            for message_payload in stored_state.get("messages", [])
        ]
        return {
            "messages": restored_messages,
            "plan": stored_state.get("plan", []),
            "current_subtask_index": stored_state.get("current_subtask_index", 0),
            "reflections": stored_state.get("reflections", []),
            "global_keywords": stored_state.get("global_keywords", []),
            "request_id": request_id,
            "session_id": stored_state.get("session_id", self.agent.session_id),
            "session_memory_id": stored_state.get("session_memory_id", 0),
            "domain_label": stored_state.get("domain_label", DEFAULT_DOMAIN_LABEL),
            "memory_summaries": stored_state.get("memory_summaries", []),
            "failed_tool_signals": stored_state.get("failed_tool_signals", {}),
            "retry_counts": stored_state.get("retry_counts", {}),
            "replan_counts": stored_state.get("replan_counts", {}),
            "blocked": stored_state.get("blocked", False),
            "final_response": stored_state.get("final_response", ""),
        }

    def _extract_resume_query(self, payload: dict[str, Any]) -> str:
        extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
        query = str(extra.get("query", "") or "").strip()
        if query:
            return query

        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        for message_payload in state.get("messages", []):
            if not isinstance(message_payload, dict):
                continue
            message_type = str(message_payload.get("type", "") or "").lower()
            if message_type == "human":
                content = str(message_payload.get("content", "") or "").strip()
                if content:
                    return content

        for message_payload in state.get("messages", []):
            if not isinstance(message_payload, dict):
                continue
            message_type = str(message_payload.get("type", "") or "").lower()
            if message_type == "system":
                content = str(message_payload.get("content", "") or "").strip()
                if content:
                    return content
        return ""

    def _build_resume_reroute_prompt(self, payload: dict[str, Any]) -> str:
        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        plan = state.get("plan", []) if isinstance(state.get("plan", []), list) else []
        current_subtask_index = int(state.get("current_subtask_index", 0) or 0)
        reflections = [
            str(item).strip() for item in state.get("reflections", [])
            if isinstance(item, str) and str(item).strip()
        ]
        failed_tools = state.get("failed_tools", {}) if isinstance(state.get("failed_tools", {}), dict) else {}
        failed_tool_signals = state.get("failed_tool_signals", {}) if isinstance(state.get("failed_tool_signals", {}), dict) else {}
        failed_tool_counts: dict[str, int] = {}
        for tool_names in failed_tools.values():
            if not isinstance(tool_names, list):
                continue
            for tool_name in tool_names:
                normalized_name = str(tool_name or "").strip()
                if not normalized_name:
                    continue
                failed_tool_counts[normalized_name] = failed_tool_counts.get(normalized_name, 0) + 1

        query = self._extract_resume_query(payload) or "Continue the original task with a safer route."
        prompt_lines = [query, "", "Resume reroute context:"]
        prompt_lines.append(f"- source_stage={payload.get('stage', '')}")
        prompt_lines.append(f"- completed_subtasks={min(current_subtask_index, len(plan))}/{len(plan)}")
        if current_subtask_index < len(plan):
            current_subtask = plan[current_subtask_index]
            prompt_lines.append(
                f"- interrupted_subtask={str(current_subtask.get('description', '')).strip()}"
            )
        if failed_tool_counts:
            ordered_failed_tools = sorted(
                failed_tool_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            prompt_lines.append(
                "- historical_failed_tools=" + ", ".join(
                    f"{tool_name}x{count}" for tool_name, count in ordered_failed_tools[:5]
                )
            )
        if failed_tool_signals:
            severity_pairs = []
            for tool_map in failed_tool_signals.values():
                if not isinstance(tool_map, dict):
                    continue
                for tool_name, signal_payload in tool_map.items():
                    if not isinstance(signal_payload, dict):
                        continue
                    severity = int(signal_payload.get("severity_score", 0) or 0)
                    if severity > 0:
                        severity_pairs.append((str(tool_name), severity))
            if severity_pairs:
                aggregated: dict[str, int] = {}
                for tool_name, severity in severity_pairs:
                    aggregated[tool_name] = aggregated.get(tool_name, 0) + severity
                ordered_severity = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
                prompt_lines.append(
                    "- historical_failure_severity=" + ", ".join(
                        f"{tool_name}:{severity}" for tool_name, severity in ordered_severity[:5]
                    )
                )
        if reflections:
            prompt_lines.append("- recent_reflections=" + " | ".join(reflections[-3:]))
        prompt_lines.append(
            "Please replan from this recovery point, avoid repeating unstable tools or failed approaches, and choose a safer execution path."
        )
        return "\n".join(prompt_lines)

    def build_resume_state_from_snapshot(
        self,
        payload: dict[str, Any],
        request_id: str,
        reroute: bool = False,
    ) -> dict[str, Any]:
        restored_state = self.restore_state_from_snapshot(payload, request_id)
        if not reroute:
            return restored_state

        return {
            "messages": [HumanMessage(content=self._build_resume_reroute_prompt(payload))],
            "plan": [],
            "current_subtask_index": 0,
            "reflections": restored_state.get("reflections", []),
            "global_keywords": restored_state.get("global_keywords", []),
            "request_id": request_id,
            "session_id": restored_state.get("session_id", self.agent.session_id),
            "session_memory_id": restored_state.get("session_memory_id", 0),
            "domain_label": restored_state.get("domain_label", "general"),
            "memory_summaries": restored_state.get("memory_summaries", []),
            "failed_tools": {},
            "failed_tool_signals": {},
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }
