import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from core.config import config
from core.llm.manager import llm_manager


class AgentObservability:
    def __init__(self, agent: Any):
        self.agent = agent

    def normalize_datetime(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def ordered_stage_duration_items(self, stage_duration_ms: dict[str, Any]) -> list[tuple[str, Any]]:
        preferred_order = ["planning", "subtask", "tool", "reflection", "request", "resume", "other"]
        ordered: list[tuple[str, Any]] = []
        for key in preferred_order:
            if key in stage_duration_ms:
                ordered.append((key, stage_duration_ms[key]))
        for key, value in stage_duration_ms.items():
            if key not in preferred_order:
                ordered.append((key, value))
        return ordered

    def stage_bucket_for_stage(self, stage: str) -> str:
        normalized = str(stage or "").strip().lower()
        if normalized.startswith("planning"):
            return "planning"
        if normalized.startswith("subtask"):
            return "subtask"
        if normalized.startswith("tool"):
            return "tool"
        if normalized.startswith("reflection"):
            return "reflection"
        if normalized.startswith("agent") or normalized.startswith("request"):
            return "request"
        if normalized.startswith("resume"):
            return "resume"
        return "other"

    def parse_detached_tool_details(self, details: str) -> dict[str, str]:
        parsed = {
            "tool_name": "",
            "reason": "",
            "tool_run_id": "",
        }
        raw_details = str(details or "")
        for key, target in (("tool", "tool_name"), ("reason", "reason"), ("tool_run_id", "tool_run_id")):
            match = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+)", raw_details)
            if match:
                parsed[target] = match.group(1).strip()
        return parsed

    def parse_failure_detail_fields(self, details: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        raw_details = str(details or "")
        for key in ("tool", "reason", "error_type", "action", "mode", "error"):
            match = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+)", raw_details)
            if match:
                parsed[key] = match.group(1).strip()
        return parsed

    def parse_subtask_index_from_details(self, details: str) -> int | None:
        match = re.search(r"(?:^|\|)\s*index=(\d+)", str(details or ""))
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def enrich_failure_detail_fields(self, details: str, latest_extra: dict[str, Any]) -> str:
        parsed = self.parse_failure_detail_fields(details)
        merged = dict(parsed)
        for key in ("tool", "reason", "error_type", "action", "error"):
            if merged.get(key):
                continue
            value = str(latest_extra.get(key, "") or "").strip()
            if value:
                merged[key] = value
        if not merged:
            return str(details or "")
        ordered_parts = []
        for key in ("tool", "reason", "error_type", "action", "mode", "error"):
            value = merged.get(key, "")
            if value:
                ordered_parts.append(f"{key}={value}")
        return " | ".join(ordered_parts)

    def build_failure_combination_key(self, left: str, right: str) -> str:
        left_value = str(left or "").strip()
        right_value = str(right or "").strip()
        if not left_value or not right_value:
            return ""
        return f"{left_value}+{right_value}"

    def build_detached_tool_details(self, request_id: str, checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        detached_tools: dict[str, dict[str, Any]] = {}
        for item in checkpoints:
            if str(item.get("stage", "")) != "tool_detached":
                continue
            parsed = self.parse_detached_tool_details(str(item.get("details", "")))
            tool_run_id = parsed.get("tool_run_id", "") or f"logged::{len(detached_tools) + 1}"
            detached_tools[tool_run_id] = {
                "tool_run_id": tool_run_id,
                "tool_name": parsed.get("tool_name", ""),
                "reason": parsed.get("reason", ""),
                "source": "checkpoint",
                "logged_at": str(item.get("logged_at", "") or ""),
                "status": "detached",
            }

        tracked = self.agent.tool_runtime.list_tracked_tool_runs(request_id=request_id, status="detached")
        for item in tracked:
            tool_run_id = str(item.get("tool_run_id", "") or "")
            if not tool_run_id:
                continue
            existing = detached_tools.get(tool_run_id, {})
            detached_tools[tool_run_id] = {
                "tool_run_id": tool_run_id,
                "tool_name": str(item.get("tool_name", "") or existing.get("tool_name", "")),
                "reason": str(item.get("reason", "") or existing.get("reason", "")),
                "source": "runtime",
                "logged_at": str(existing.get("logged_at", "") or ""),
                "status": str(item.get("status", "detached") or "detached"),
                "runtime_ms": item.get("runtime_ms"),
                "detached_runtime_ms": item.get("detached_runtime_ms"),
            }

        return sorted(detached_tools.values(), key=lambda item: (item.get("logged_at", ""), item.get("tool_run_id", "")))

    def _find_last_attention_event(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        for event in reversed(events):
            outcome = str(event.get("outcome", "")).lower()
            level = str(event.get("level", "")).upper()
            stage = str(event.get("stage", ""))
            if outcome in {"failed", "cancelled", "timed_out", "blocked", "waiting_user", "error"}:
                return event
            if level == "ERROR":
                return event
            if stage in {"agent_blocked", "agent_waiting_user", "request_failed", "request_cancelled", "request_timed_out", "tool_failed", "tool_rejected", "tool_cancelled", "tool_detached"}:
                return event
        return None

    def normalize_failure_source(self, source: str, stage: str = "") -> str:
        normalized_source = str(source or "").strip().lower()
        if normalized_source in {"", "-", "log_event"}:
            return ""
        if normalized_source.startswith("agent.invoke"):
            return "agent_invoke"
        if normalized_source.startswith("agent.resume"):
            return "agent_resume"
        if normalized_source.startswith("agent.execute_subtask"):
            return "model"
        if normalized_source.startswith("agent.reflect"):
            return "reflection"
        if normalized_source.startswith("test."):
            return "test"
        if normalized_source.startswith("retention"):
            return "retention"
        if normalized_source == "checkpoint":
            stage_bucket = self.stage_bucket_for_stage(stage)
            if stage_bucket == "tool":
                return "tool_runtime"
            if stage_bucket == "reflection":
                return "reflection"
            if stage in {"agent_blocked", "agent_waiting_user"}:
                return "reflection"
            return "checkpoint"
        if normalized_source.startswith("agent."):
            return "agent_runtime"
        return normalized_source.replace(".", "_")

    def _find_last_failure_source(self, events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            outcome = str(event.get("outcome", "")).lower()
            level = str(event.get("level", "")).upper()
            stage = str(event.get("stage", ""))
            source = self.normalize_failure_source(str(event.get("source", "") or ""), stage)
            if not source:
                continue
            if outcome in {"failed", "cancelled", "timed_out", "blocked", "waiting_user", "error"}:
                return source
            if level == "ERROR":
                return source
            if stage in {"agent_blocked", "agent_waiting_user", "request_failed", "request_cancelled", "request_timed_out", "tool_failed", "tool_rejected", "tool_cancelled", "tool_detached"}:
                return source
        return ""

    def build_request_triage(
        self,
        events: list[dict[str, Any]],
        checkpoints: list[dict[str, Any]],
        latest_state: dict[str, Any],
        latest_extra: dict[str, Any],
        latest_stage: str,
        status: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        attention_event = self._find_last_attention_event(events)
        latest_failure_stage = ""
        latest_failure_details = ""
        latest_failure_source = ""
        latest_reroute_mode = ""
        latest_reroute_details = ""
        for item in reversed(checkpoints):
            stage = str(item.get("stage", ""))
            if stage in {"agent_blocked", "agent_waiting_user", "request_failed", "request_cancelled", "request_timed_out", "tool_failed", "tool_rejected", "tool_cancelled", "tool_detached"}:
                latest_failure_stage = stage
                latest_failure_details = self.enrich_failure_detail_fields(str(item.get("details", "")), latest_extra)
                latest_failure_source = self.normalize_failure_source(str(item.get("source", "") or ""), stage)
                break

        for item in reversed(checkpoints):
            stage = str(item.get("stage", ""))
            if stage != "tool_reroute_applied":
                continue
            latest_reroute_mode = str(item.get("mode", "") or "")
            if not latest_reroute_mode:
                latest_reroute_mode = self.parse_failure_detail_fields(str(item.get("details", "") or "")).get("mode", "")
            latest_reroute_details = str(item.get("details", "") or "")
            break

        if not latest_failure_stage and status in {"waiting_user", "blocked", "failed", "cancelled", "timed_out"}:
            latest_failure_stage = str(latest_stage or latest_extra.get("stage", "") or "")
        if not latest_failure_details:
            latest_failure_details = self.enrich_failure_detail_fields("", latest_extra)
        if not latest_failure_source:
            latest_failure_source = self._find_last_failure_source(events) or self.normalize_failure_source(
                str(latest_extra.get("source", "") or ""),
                latest_failure_stage,
            )

        source_request_id = str(latest_extra.get("source_request_id", "") or "")
        detached_tool_count = int(metrics.get("tool_detached_count", 0) or 0)
        tool_attention_count = (
            int(metrics.get("tool_failure_count", 0) or 0)
            + int(metrics.get("tool_rejection_count", 0) or 0)
            + int(metrics.get("tool_cancelled_count", 0) or 0)
        )

        return {
            "needs_attention": status in {"waiting_user", "blocked", "failed", "cancelled", "timed_out"} or detached_tool_count > 0,
            "is_resumed": bool(source_request_id),
            "source_request_id": source_request_id,
            "latest_failure_stage": latest_failure_stage,
            "latest_failure_details": latest_failure_details,
            "latest_failure_source": latest_failure_source,
            "latest_reroute_mode": latest_reroute_mode,
            "latest_reroute_details": latest_reroute_details,
            "used_no_tool_fallback": latest_reroute_mode.startswith("fallback_"),
            "last_error_message": str(attention_event.get("message", "")) if attention_event else "",
            "last_error_event_type": str(attention_event.get("event_type", "")) if attention_event else "",
            "tool_attention_count": tool_attention_count,
            "detached_tool_count": detached_tool_count,
            "has_detached_tools": detached_tool_count > 0,
            "retry_count": int(metrics.get("retry_count", 0) or 0),
            "blocked": bool(latest_state.get("blocked", False)),
            "waiting_for_user": bool(latest_state.get("waiting_for_user", False)),
        }

    def derive_request_status(self, latest_stage: str, latest_state: dict[str, Any], active: bool) -> str:
        if active:
            return "active"
        if latest_state.get("waiting_for_user"):
            return "waiting_user"
        if latest_state.get("blocked"):
            return "blocked"

        stage_to_status = {
            "request_completed": "completed",
            "agent_completed": "completed",
            "request_failed": "failed",
            "request_cancelled": "cancelled",
            "request_timed_out": "timed_out",
            "agent_waiting_user": "waiting_user",
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
        stage_duration_ms: dict[str, float] = {}
        for event in checkpoint_events:
            stage = str(event.get("stage", ""))
            if stage:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                duration_ms = event.get("duration_ms")
                if duration_ms not in (None, ""):
                    bucket = self.stage_bucket_for_stage(stage)
                    stage_duration_ms[bucket] = round(
                        stage_duration_ms.get(bucket, 0.0) + float(duration_ms or 0),
                        2,
                    )

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
        unique_subtask_indexes: set[int] = set()
        for event in checkpoint_events:
            if event.get("stage") == "subtask_started":
                subtask_index = self.parse_subtask_index_from_details(str(event.get("details", "")))
                if subtask_index is not None:
                    unique_subtask_indexes.add(subtask_index)
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
        tool_detached = stage_counts.get("tool_detached", 0)
        subtask_count = len(unique_subtask_indexes) or len(latest_state.get("plan", [])) or stage_counts.get("subtask_started", 0)

        return {
            "total_duration_ms": total_duration_ms,
            "llm_call_count": sum(1 for event in events if event.get("event_type") == "llm_request"),
            "llm_error_count": sum(1 for event in events if event.get("event_type") == "llm_error"),
            "llm_total_duration_ms": llm_total_duration_ms,
            "stage_duration_ms": stage_duration_ms,
            "tool_call_count": tool_started,
            "tool_success_count": tool_success,
            "tool_failure_count": tool_failures,
            "tool_rejection_count": tool_rejections,
            "tool_cancelled_count": tool_cancelled,
            "tool_detached_count": tool_detached,
            "tool_hit_rate": round((tool_success / tool_started), 4) if tool_started else 0.0,
            "retry_count": retry_count,
            "subtask_count": subtask_count,
            "reflection_failure_count": reflection_failure_count,
            "blocked_rate": 1.0 if status == "blocked" else 0.0,
            "waiting_user_rate": 1.0 if status == "waiting_user" else 0.0,
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
                    "source": event.get("source", ""),
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
        triage = self.build_request_triage(events, checkpoints, latest_state, latest_extra, latest_stage, status, metrics)
        detached_tools = self.build_detached_tool_details(request_id, checkpoints)

        return {
            "request_id": request_id,
            "status": status,
            "active": active,
            "session_id": session_id,
            "source_request_id": latest_extra.get("source_request_id", ""),
            "latest_stage": latest_stage,
            "lite_mode": bool(latest_state.get("lite_mode", False)),
            "raw_query": str(latest_state.get("raw_query", "") or latest_extra.get("query", "") or ""),
            "normalized_query": str(latest_state.get("normalized_query", "") or ""),
            "created_at": snapshots[0].get("created_at", "") if snapshots else (events[0].get("logged_at", "") if events else ""),
            "updated_at": snapshots[-1].get("created_at", "") if snapshots else (events[-1].get("logged_at", "") if events else ""),
            "subtask_index": latest_state.get("current_subtask_index", 0),
            "plan_length": len(latest_state.get("plan", [])),
            "plan": self.agent.snapshot_store.normalize_plan(latest_state.get("plan", [])),
            "blocked": latest_state.get("blocked", False),
            "waiting_for_user": latest_state.get("waiting_for_user", False),
            "final_response": final_response,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "memory_count": len(memories),
            "memories": memories,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "detached_tools": detached_tools,
            "metrics": metrics,
            "triage": triage,
        }

    def get_recent_request_summaries(
        self,
        limit: int = 10,
        statuses: list[str] | None = None,
        resumed_only: bool = False,
        attention_only: bool = False,
        since_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        snapshot_root = config.resolve_path(config.state_snapshot_dir)
        if not os.path.exists(snapshot_root):
            return []

        normalized_statuses = {item.strip().lower() for item in (statuses or []) if item and item.strip()}
        cutoff = None
        if isinstance(since_seconds, int) and since_seconds > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_seconds)

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
            if attention_only and not bool((summary.get("triage", {}) or {}).get("needs_attention")):
                continue
            if cutoff is not None:
                updated_at = self.normalize_datetime(self.parse_logged_at(str(summary.get("updated_at", "") or "")))
                if updated_at is None or updated_at < cutoff:
                    continue
            summarized_requests.append(summary)

        summarized_requests.sort(
            key=lambda item: self.normalize_datetime(self.parse_logged_at(item.get("updated_at", ""))) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return summarized_requests[: max(0, limit)]

    def get_request_rollup(
        self,
        limit: int = 20,
        statuses: list[str] | None = None,
        resumed_only: bool = False,
        attention_only: bool = False,
        since_seconds: int | None = None,
    ) -> dict[str, Any]:
        summaries = self.get_recent_request_summaries(
            limit=limit,
            statuses=statuses,
            resumed_only=resumed_only,
            attention_only=attention_only,
            since_seconds=since_seconds,
        )
        status_counts: dict[str, int] = {}
        stage_duration_ms_total: dict[str, float] = {}
        failure_stage_counts: Counter[str] = Counter()
        failure_tool_counts: Counter[str] = Counter()
        failure_reason_counts: Counter[str] = Counter()
        failure_error_type_counts: Counter[str] = Counter()
        failure_action_counts: Counter[str] = Counter()
        failure_source_counts: Counter[str] = Counter()
        failure_reroute_mode_counts: Counter[str] = Counter()
        failure_no_tool_fallback_counts: Counter[str] = Counter()
        failure_stage_tool_counts: Counter[str] = Counter()
        failure_tool_reason_counts: Counter[str] = Counter()
        failure_stage_source_counts: Counter[str] = Counter()
        failure_stage_reroute_counts: Counter[str] = Counter()
        totals = {
            "llm_call_count": 0,
            "tool_call_count": 0,
            "tool_detached_count": 0,
            "retry_count": 0,
            "reflection_failure_count": 0,
        }
        resumed_count = 0
        needs_attention_count = 0
        active_count = 0
        total_duration_sum = 0.0
        total_duration_samples = 0
        attention_failure_sources: list[str] = []

        for summary in summaries:
            status = str(summary.get("status", "") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            metrics = summary.get("metrics", {}) or {}
            triage = summary.get("triage", {}) or {}
            if summary.get("source_request_id"):
                resumed_count += 1
            if triage.get("needs_attention"):
                needs_attention_count += 1
                latest_failure_stage = str(triage.get("latest_failure_stage", "") or "")
                if latest_failure_stage:
                    failure_stage_counts[latest_failure_stage] += 1
                parsed_details = self.parse_failure_detail_fields(str(triage.get("latest_failure_details", "") or ""))
                tool_name = parsed_details.get("tool", "")
                reason = parsed_details.get("reason", "")
                if tool_name:
                    failure_tool_counts[tool_name] += 1
                if reason:
                    failure_reason_counts[reason] += 1
                if parsed_details.get("error_type"):
                    failure_error_type_counts[parsed_details["error_type"]] += 1
                if parsed_details.get("action"):
                    failure_action_counts[parsed_details["action"]] += 1
                failure_source = str(triage.get("latest_failure_source", "") or "")
                if failure_source:
                    failure_source_counts[failure_source] += 1
                    attention_failure_sources.append(failure_source)
                reroute_mode = str(triage.get("latest_reroute_mode", "") or "")
                if reroute_mode:
                    failure_reroute_mode_counts[reroute_mode] += 1
                if triage.get("used_no_tool_fallback"):
                    failure_no_tool_fallback_counts["used"] += 1
                stage_tool_key = self.build_failure_combination_key(latest_failure_stage, tool_name)
                if stage_tool_key:
                    failure_stage_tool_counts[stage_tool_key] += 1
                tool_reason_key = self.build_failure_combination_key(tool_name, reason)
                if tool_reason_key:
                    failure_tool_reason_counts[tool_reason_key] += 1
                stage_source_key = self.build_failure_combination_key(latest_failure_stage, failure_source)
                if stage_source_key:
                    failure_stage_source_counts[stage_source_key] += 1
                stage_reroute_key = self.build_failure_combination_key(latest_failure_stage, reroute_mode)
                if stage_reroute_key:
                    failure_stage_reroute_counts[stage_reroute_key] += 1
            if summary.get("active"):
                active_count += 1

            for key in totals:
                totals[key] += int(metrics.get(key, 0) or 0)

            total_duration_ms = metrics.get("total_duration_ms")
            if isinstance(total_duration_ms, (int, float)):
                total_duration_sum += float(total_duration_ms)
                total_duration_samples += 1

            for key, value in (metrics.get("stage_duration_ms", {}) or {}).items():
                stage_duration_ms_total[key] = round(stage_duration_ms_total.get(key, 0.0) + float(value or 0), 2)

        average_total_duration_ms = None
        if total_duration_samples:
            average_total_duration_ms = round(total_duration_sum / total_duration_samples, 2)

        latest_updated_at = summaries[0].get("updated_at", "") if summaries else ""
        source_bucket_breakdown = []
        denominator = max(0, needs_attention_count)
        if denominator:
            for source, count in sorted(
                failure_source_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]:
                source_bucket_breakdown.append(
                    {
                        "source": source,
                        "count": count,
                        "share": round(count / denominator, 4),
                    }
                )
        source_bucket_trends = []
        if len(attention_failure_sources) >= 2:
            recent_window_size = (len(attention_failure_sources) + 1) // 2
            recent_sources = attention_failure_sources[:recent_window_size]
            earlier_sources = attention_failure_sources[recent_window_size:]
            if earlier_sources:
                recent_counts = Counter(recent_sources)
                earlier_counts = Counter(earlier_sources)
                recent_total = len(recent_sources)
                earlier_total = len(earlier_sources)
                trend_items = []
                for source in set(recent_counts) | set(earlier_counts):
                    recent_count = recent_counts.get(source, 0)
                    earlier_count = earlier_counts.get(source, 0)
                    recent_share = recent_count / recent_total if recent_total else 0.0
                    earlier_share = earlier_count / earlier_total if earlier_total else 0.0
                    delta_share = recent_share - earlier_share
                    direction = "flat"
                    if delta_share > 0:
                        direction = "up"
                    elif delta_share < 0:
                        direction = "down"
                    trend_items.append(
                        {
                            "source": source,
                            "recent_count": recent_count,
                            "earlier_count": earlier_count,
                            "recent_share": round(recent_share, 4),
                            "earlier_share": round(earlier_share, 4),
                            "delta_share": round(delta_share, 4),
                            "direction": direction,
                        }
                    )
                source_bucket_trends = sorted(
                    trend_items,
                    key=lambda item: (-abs(float(item["delta_share"])), -float(item["delta_share"]), item["source"]),
                )[:5]

        return {
            "request_count": len(summaries),
            "latest_updated_at": latest_updated_at,
            "filters": {
                "statuses": list(statuses or []),
                "resumed_only": resumed_only,
                "attention_only": attention_only,
                "since_seconds": since_seconds,
            },
            "status_counts": status_counts,
            "resumed_count": resumed_count,
            "needs_attention_count": needs_attention_count,
            "active_count": active_count,
            "average_total_duration_ms": average_total_duration_ms,
            "stage_duration_ms_total": stage_duration_ms_total,
            "source_bucket_breakdown": source_bucket_breakdown,
            "source_bucket_trends": source_bucket_trends,
            "top_failure_signals": {
                "stages": failure_stage_counts.most_common(3),
                "tools": failure_tool_counts.most_common(3),
                "reasons": failure_reason_counts.most_common(3),
                "error_types": failure_error_type_counts.most_common(3),
                "actions": failure_action_counts.most_common(3),
                "sources": failure_source_counts.most_common(3),
                "reroute_modes": failure_reroute_mode_counts.most_common(3),
                "no_tool_fallbacks": failure_no_tool_fallback_counts.most_common(3),
            },
            "top_failure_combinations": {
                "stage_tool": failure_stage_tool_counts.most_common(3),
                "tool_reason": failure_tool_reason_counts.most_common(3),
                "stage_source": failure_stage_source_counts.most_common(3),
                "stage_reroute": failure_stage_reroute_counts.most_common(3),
            },
            "totals": totals,
        }
