import os
import re
from datetime import datetime
from typing import Any

from config import config
from llm_manager import llm_manager


class AgentObservability:
    def __init__(self, agent: Any):
        self.agent = agent

    def _find_last_attention_event(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        for event in reversed(events):
            outcome = str(event.get("outcome", "")).lower()
            level = str(event.get("level", "")).upper()
            stage = str(event.get("stage", ""))
            if outcome in {"failed", "cancelled", "timed_out", "blocked", "error"}:
                return event
            if level == "ERROR":
                return event
            if stage in {"agent_blocked", "request_failed", "request_cancelled", "request_timed_out", "tool_failed", "tool_rejected", "tool_cancelled"}:
                return event
        return None

    def build_request_triage(
        self,
        events: list[dict[str, Any]],
        checkpoints: list[dict[str, Any]],
        latest_state: dict[str, Any],
        latest_extra: dict[str, Any],
        status: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        attention_event = self._find_last_attention_event(events)
        latest_failure_stage = ""
        latest_failure_details = ""
        for item in reversed(checkpoints):
            stage = str(item.get("stage", ""))
            if stage in {"agent_blocked", "request_failed", "request_cancelled", "request_timed_out", "tool_failed", "tool_rejected", "tool_cancelled"}:
                latest_failure_stage = stage
                latest_failure_details = str(item.get("details", ""))
                break

        if not latest_failure_stage and status in {"blocked", "failed", "cancelled", "timed_out"}:
            latest_failure_stage = str(latest_extra.get("stage", "") or "")

        source_request_id = str(latest_extra.get("source_request_id", "") or "")
        tool_attention_count = (
            int(metrics.get("tool_failure_count", 0) or 0)
            + int(metrics.get("tool_rejection_count", 0) or 0)
            + int(metrics.get("tool_cancelled_count", 0) or 0)
        )

        return {
            "needs_attention": status in {"blocked", "failed", "cancelled", "timed_out"},
            "is_resumed": bool(source_request_id),
            "source_request_id": source_request_id,
            "latest_failure_stage": latest_failure_stage,
            "latest_failure_details": latest_failure_details,
            "last_error_message": str(attention_event.get("message", "")) if attention_event else "",
            "last_error_event_type": str(attention_event.get("event_type", "")) if attention_event else "",
            "tool_attention_count": tool_attention_count,
            "retry_count": int(metrics.get("retry_count", 0) or 0),
            "blocked": bool(latest_state.get("blocked", False)),
        }

    def derive_request_status(self, latest_stage: str, latest_state: dict[str, Any], active: bool) -> str:
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

    def extract_bool_from_event(self, event: dict[str, Any], key: str) -> bool | None:
        value = event.get(key)
        if isinstance(value, bool):
            return value
        details = str(event.get("details", ""))
        match = re.search(rf"{re.escape(key)}=(True|False|true|false)", details)
        if not match:
            return None
        return match.group(1).lower() == "true"

    def parse_logged_at(self, value: str) -> datetime | None:
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

    def build_request_metrics(self, events: list[dict[str, Any]], latest_state: dict[str, Any], status: str) -> dict[str, Any]:
        llm_events = [event for event in events if event.get("event_type") in {"llm_response", "llm_error"}]
        checkpoint_events = [event for event in events if event.get("event_type") == "checkpoint"]
        stage_counts: dict[str, int] = {}
        for event in checkpoint_events:
            stage = str(event.get("stage", ""))
            if stage:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        logged_times = [self.parse_logged_at(str(event.get("logged_at", ""))) for event in events]
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
            success = self.extract_bool_from_event(event, "success")
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

    def get_request_summary(self, request_id: str) -> dict[str, Any] | None:
        snapshots = self.agent.snapshot_store.list_snapshots(request_id)
        latest_payload = self.agent.snapshot_store.load_snapshot_payload(request_id)
        latest_state = latest_payload.get("state", {}) if latest_payload else {}
        latest_extra = latest_payload.get("extra", {}) if latest_payload else {}
        events = llm_manager.get_request_events(request_id)
        memories = self.agent.memory.list_memories_by_request_id(request_id)

        if not snapshots and not events and not memories and not self.agent.is_request_active(request_id):
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
        active = self.agent.is_request_active(request_id)
        status = self.derive_request_status(latest_stage, latest_state, active)
        session_id = latest_state.get("session_id", "") or (memories[-1].get("conv_id", "") if memories else "")
        metrics = self.build_request_metrics(events, latest_state, status)
        triage = self.build_request_triage(events, checkpoints, latest_state, latest_extra, status, metrics)

        return {
            "request_id": request_id,
            "status": status,
            "active": active,
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
            "triage": triage,
        }

    def get_recent_request_summaries(
        self,
        limit: int = 10,
        statuses: list[str] | None = None,
        resumed_only: bool = False,
    ) -> list[dict[str, Any]]:
        snapshot_root = config.resolve_path(config.state_snapshot_dir)
        if not os.path.exists(snapshot_root):
            return []

        normalized_statuses = {item.strip().lower() for item in (statuses or []) if item and item.strip()}

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
            if normalized_statuses and str(summary.get("status", "")).lower() not in normalized_statuses:
                continue
            if resumed_only and not summary.get("source_request_id"):
                continue
            summarized_requests.append(summary)

        summarized_requests.sort(
            key=lambda item: self.parse_logged_at(item.get("updated_at", "")) or datetime.min,
            reverse=True,
        )
        return summarized_requests[: max(0, limit)]
