import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from threading import Lock
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from config import config
from llm_manager import RequestCancelledError, llm_manager


class AgentToolRuntime:
    CITY_FIELD_HINTS = ("city", "location", "城市", "地点", "地区", "地址", "位置")
    DATE_FIELD_HINTS = ("date", "日期")
    TIME_FIELD_HINTS = ("time", "timestamp", "datetime", "时间", "时刻", "时间点", "日期时间")

    FAILURE_TYPE_WEIGHTS = {
        "invalid_arguments": 4,
        "dependency_unavailable": 3,
        "timeout": 3,
        "execution_error": 2,
        "cancelled": 1,
    }

    def __init__(self, agent: Any):
        self.agent = agent
        self._tool_run_lock = Lock()
        self._tool_run_counter = 0
        self._tracked_tool_runs: dict[str, dict[str, Any]] = {}

    def _next_tool_run_id(self) -> str:
        with self._tool_run_lock:
            self._tool_run_counter += 1
            return f"toolrun_{self._tool_run_counter:06d}"

    def _register_tool_run(self, tool_name: str, request_id: str, executor: ThreadPoolExecutor, future) -> str:
        run_id = self._next_tool_run_id()
        with self._tool_run_lock:
            self._tracked_tool_runs[run_id] = {
                "tool_name": tool_name,
                "request_id": request_id,
                "executor": executor,
                "future": future,
                "started_at": time.monotonic(),
                "status": "running",
                "detached_reason": "",
            }
        return run_id

    def _mark_tool_run_detached(self, run_id: str, reason: str) -> bool:
        with self._tool_run_lock:
            tracked = self._tracked_tool_runs.get(run_id)
            if not tracked:
                return False
            tracked["status"] = "detached"
            tracked["detached_reason"] = reason
            tracked["detached_at"] = time.monotonic()
            return True

    def _cleanup_tool_run(self, run_id: str) -> bool:
        with self._tool_run_lock:
            tracked = self._tracked_tool_runs.pop(run_id, None)
        if not tracked:
            return False
        executor = tracked.get("executor")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        return True

    def prune_finished_tool_runs(self) -> dict[str, int]:
        finished_run_ids: list[str] = []
        active = 0
        detached = 0
        with self._tool_run_lock:
            for run_id, tracked in self._tracked_tool_runs.items():
                future = tracked.get("future")
                if future is not None and future.done():
                    finished_run_ids.append(run_id)
                    continue
                if tracked.get("status") == "detached":
                    detached += 1
                else:
                    active += 1

        for run_id in finished_run_ids:
            self._cleanup_tool_run(run_id)

        return {
            "active": active,
            "detached": detached,
            "cleaned": len(finished_run_ids),
        }

    def get_tool_run_stats(self) -> dict[str, int]:
        stats = self.prune_finished_tool_runs()
        return {
            "active": stats["active"],
            "detached": stats["detached"],
            "tracked": stats["active"] + stats["detached"],
        }

    def list_tracked_tool_runs(self, request_id: str = "", status: str = "") -> list[dict[str, Any]]:
        self.prune_finished_tool_runs()
        normalized_request_id = str(request_id or "")
        normalized_status = str(status or "")
        now = time.monotonic()
        items: list[dict[str, Any]] = []

        with self._tool_run_lock:
            for run_id, tracked in self._tracked_tool_runs.items():
                tracked_request_id = str(tracked.get("request_id", "") or "")
                tracked_status = str(tracked.get("status", "") or "")
                if normalized_request_id and tracked_request_id != normalized_request_id:
                    continue
                if normalized_status and tracked_status != normalized_status:
                    continue

                started_at = tracked.get("started_at")
                detached_at = tracked.get("detached_at")
                items.append(
                    {
                        "tool_run_id": run_id,
                        "tool_name": str(tracked.get("tool_name", "") or ""),
                        "request_id": tracked_request_id,
                        "status": tracked_status,
                        "reason": str(tracked.get("detached_reason", "") or ""),
                        "runtime_ms": round(max(0.0, now - started_at) * 1000, 2) if isinstance(started_at, (int, float)) else None,
                        "detached_runtime_ms": round(max(0.0, now - detached_at) * 1000, 2) if isinstance(detached_at, (int, float)) else None,
                    }
                )

        items.sort(key=lambda item: (item.get("status") != "detached", item.get("tool_run_id", "")))
        return items

    def request_tool_run_stop(self, run_id: str, reason: str, grace_seconds: float) -> bool:
        with self._tool_run_lock:
            tracked = self._tracked_tool_runs.get(run_id)
            if not tracked:
                return True
            future = tracked.get("future")
        if future is None:
            return True

        future.cancel()
        deadline = time.monotonic() + max(grace_seconds, 0.0)
        while time.monotonic() < deadline:
            if future.done():
                self._cleanup_tool_run(run_id)
                return True
            time.sleep(0.01)

        self._mark_tool_run_detached(run_id, reason)
        return False

    def classify_tool_exception(self, exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, (TypeError, ValueError)):
            return "invalid_arguments", False
        if isinstance(exc, TimeoutError):
            return "timeout", True
        if isinstance(exc, (ConnectionError, OSError)):
            return "dependency_unavailable", True
        return "execution_error", True

    def build_tool_error_payload(self, tool_name: str, error_type: str, retryable: bool, message: str) -> str:
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

    def validate_named_argument_heuristics(self, field_name: str, value: Any) -> str:
        normalized_name = field_name.strip().lower()
        raw_name = field_name.strip()
        if value is None:
            return ""

        if any(hint in normalized_name or hint in raw_name for hint in self.CITY_FIELD_HINTS):
            if not isinstance(value, str) or not value.strip():
                return f"Argument {field_name} must be a non-empty city/location name."
            compact = value.strip()
            if compact.isdigit() or not re.search(r"[a-zA-Z\u4e00-\u9fff]", compact):
                return f"Argument {field_name} does not look like a valid city/location name: {value}"

        date_pattern = r"^\d{4}[-/]\d{2}[-/]\d{2}$"
        date_pattern_cn = r"^\d{4}年\d{1,2}月\d{1,2}日$"
        time_pattern = r"^\d{2}:\d{2}(:\d{2})?$"
        time_pattern_cn = r"^(上午|中午|下午|晚上)?\s*\d{1,2}点(半|\d{1,2}分)?$"
        datetime_pattern = r"^\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(:\d{2})?$"
        datetime_pattern_cn = r"^\d{4}年\d{1,2}月\d{1,2}日\s*(上午|中午|下午|晚上)?\s*\d{1,2}点(半|\d{1,2}分)?$"

        if any(hint in normalized_name or hint in raw_name for hint in self.DATE_FIELD_HINTS) and isinstance(value, str):
            if not (re.match(date_pattern, value.strip()) or re.match(date_pattern_cn, value.strip())):
                return f"Argument {field_name} must use a recognizable date format like YYYY-MM-DD."

        if (
            normalized_name in {"time", "timestamp", "datetime"}
            or normalized_name.endswith("_time")
            or any(hint in normalized_name or hint in raw_name for hint in self.TIME_FIELD_HINTS)
        ):
            if isinstance(value, str) and not (
                re.match(time_pattern, value.strip())
                or re.match(datetime_pattern, value.strip())
                or re.match(time_pattern_cn, value.strip())
                or re.match(datetime_pattern_cn, value.strip())
            ):
                return f"Argument {field_name} must use a recognizable time format like HH:MM or YYYY-MM-DD HH:MM:SS."

        return ""

    def prevalidate_tool_arguments(self, tool_name: str, args_schema, kwargs: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
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
            return None, self.build_tool_error_payload(
                tool_name,
                "invalid_arguments",
                False,
                f"Argument schema validation failed: {exc}",
            )
        except Exception as exc:
            return None, self.build_tool_error_payload(
                tool_name,
                "invalid_arguments",
                False,
                f"Argument validation failed: {exc}",
            )

        for field_name, value in normalized_kwargs.items():
            heuristic_error = self.validate_named_argument_heuristics(field_name, value)
            if heuristic_error:
                return None, self.build_tool_error_payload(
                    tool_name,
                    "invalid_arguments",
                    False,
                    heuristic_error,
                )

        return normalized_kwargs, None

    def wrap_tool_for_runtime(self, tool):
        if getattr(tool, "_llm_brain_safe_tool", False):
            return tool

        tool_name = getattr(tool, "name", "runtime_tool")
        description = getattr(tool, "description", "") or ""
        args_schema = getattr(tool, "args_schema", None)
        return_direct = getattr(tool, "return_direct", False)

        def safe_tool_runner(**kwargs):
            self.prune_finished_tool_runs()
            request_id = llm_manager.get_request_id()
            self.agent._raise_if_request_cancelled(request_id or "")
            started_at = time.monotonic()
            tool_run_id = ""
            should_cleanup_run = True
            executor = None
            llm_manager.log_checkpoint(
                "tool_started",
                details=f"tool={tool_name}",
                request_id=request_id,
                console=True,
                tool_name=tool_name,
            )
            try:
                normalized_kwargs, validation_error = self.prevalidate_tool_arguments(tool_name, args_schema, kwargs)
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

                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(lambda: tool.invoke(normalized_kwargs))
                tool_run_id = self._register_tool_run(tool_name, request_id or "", executor, future)
                start = time.monotonic()
                while True:
                    self.agent._raise_if_request_cancelled(request_id or "")
                    elapsed = time.monotonic() - start
                    remaining = config.tool_timeout_seconds - elapsed
                    if remaining <= 0:
                        stopped = self.request_tool_run_stop(
                            tool_run_id,
                            reason="timeout",
                            grace_seconds=getattr(config, "tool_cancellation_grace_seconds", 0.2),
                        )
                        if not stopped:
                            should_cleanup_run = False
                            llm_manager.log_checkpoint(
                                "tool_detached",
                                details=f"tool={tool_name} | reason=timeout | tool_run_id={tool_run_id}",
                                request_id=request_id,
                                level=40,
                                console=False,
                                duration_ms=(time.monotonic() - started_at) * 1000,
                                tool_name=tool_name,
                                tool_run_id=tool_run_id,
                                error_type="timeout",
                            )
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
                if tool_run_id:
                    stopped = self.request_tool_run_stop(
                        tool_run_id,
                        reason="cancelled",
                        grace_seconds=getattr(config, "tool_cancellation_grace_seconds", 0.2),
                    )
                    if not stopped:
                        should_cleanup_run = False
                        llm_manager.log_checkpoint(
                            "tool_detached",
                            details=f"tool={tool_name} | reason=cancelled | tool_run_id={tool_run_id}",
                            request_id=request_id,
                            level=40,
                            console=False,
                            duration_ms=(time.monotonic() - started_at) * 1000,
                            tool_name=tool_name,
                            tool_run_id=tool_run_id,
                            error_type="cancelled",
                        )
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
                error_type, retryable = self.classify_tool_exception(exc)
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
                return self.build_tool_error_payload(tool_name, error_type, retryable, str(exc))
            finally:
                if tool_run_id and should_cleanup_run:
                    self._cleanup_tool_run(tool_run_id)
                elif executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                self.prune_finished_tool_runs()

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

    def parse_tool_error_payload(self, content: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("ok", True):
            return None
        if not payload.get("tool"):
            return None
        return payload

    def collect_recent_tool_failures(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            payload = self.parse_tool_error_payload(getattr(message, "content", ""))
            if payload:
                failures.append(payload)
        failures.reverse()
        return failures

    def merge_failed_tools(
        self,
        failed_tools: dict[str, list[str]],
        subtask_index: int,
        recent_failures: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        merged = {key: list(value) for key, value in (failed_tools or {}).items()}
        if not recent_failures:
            return merged

        key = str(subtask_index)
        existing = set(merged.get(key, []))
        for item in recent_failures:
            existing.add(str(item.get("tool", "")))
        merged[key] = sorted(name for name in existing if name)
        return merged

    def merge_failed_tool_signals(
        self,
        failed_tool_signals: dict[str, dict[str, dict[str, Any]]],
        subtask_index: int,
        recent_failures: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        merged: dict[str, dict[str, dict[str, Any]]] = {}
        for raw_key, raw_value in (failed_tool_signals or {}).items():
            if not isinstance(raw_value, dict):
                continue
            merged[str(raw_key)] = {
                str(tool_name): dict(signal_payload)
                for tool_name, signal_payload in raw_value.items()
                if isinstance(signal_payload, dict)
            }

        if not recent_failures:
            return merged

        key = str(subtask_index)
        tool_map = merged.setdefault(key, {})
        for item in recent_failures:
            tool_name = str(item.get("tool", "") or "").strip()
            if not tool_name:
                continue
            error_type = str(item.get("error_type", "execution_error") or "execution_error")
            retryable = bool(item.get("retryable", False))
            signal_payload = dict(tool_map.get(tool_name, {}))
            signal_payload["count"] = int(signal_payload.get("count", 0) or 0) + 1
            signal_payload["retryable_count"] = int(signal_payload.get("retryable_count", 0) or 0) + (1 if retryable else 0)
            signal_payload["non_retryable_count"] = int(signal_payload.get("non_retryable_count", 0) or 0) + (0 if retryable else 1)
            error_counts = dict(signal_payload.get("error_type_counts", {}))
            error_counts[error_type] = int(error_counts.get(error_type, 0) or 0) + 1
            signal_payload["error_type_counts"] = error_counts
            signal_payload["severity_score"] = self.calculate_failure_severity(signal_payload)
            tool_map[tool_name] = signal_payload
        return merged

    def calculate_failure_severity(self, signal_payload: dict[str, Any]) -> int:
        error_counts = signal_payload.get("error_type_counts", {}) if isinstance(signal_payload.get("error_type_counts", {}), dict) else {}
        severity = 0
        for error_type, count in error_counts.items():
            severity += self.FAILURE_TYPE_WEIGHTS.get(str(error_type), 1) * int(count or 0)
        severity += int(signal_payload.get("non_retryable_count", 0) or 0)
        return severity

    def summarize_historical_failed_tools(
        self,
        failed_tools: dict[str, list[str]],
        current_subtask_index: int,
    ) -> dict[str, int]:
        historical_counts: dict[str, int] = {}
        for raw_index, tool_names in (failed_tools or {}).items():
            try:
                subtask_index = int(str(raw_index))
            except (TypeError, ValueError):
                continue
            if subtask_index >= current_subtask_index:
                continue
            for tool_name in tool_names or []:
                normalized_name = str(tool_name or "").strip()
                if not normalized_name:
                    continue
                historical_counts[normalized_name] = historical_counts.get(normalized_name, 0) + 1
        return historical_counts

    def summarize_historical_failed_tool_signals(
        self,
        failed_tool_signals: dict[str, dict[str, dict[str, Any]]],
        current_subtask_index: int,
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for raw_index, tool_map in (failed_tool_signals or {}).items():
            try:
                subtask_index = int(str(raw_index))
            except (TypeError, ValueError):
                continue
            if subtask_index >= current_subtask_index or not isinstance(tool_map, dict):
                continue
            for tool_name, signal_payload in tool_map.items():
                normalized_name = str(tool_name or "").strip()
                if not normalized_name or not isinstance(signal_payload, dict):
                    continue
                current_payload = dict(summary.get(normalized_name, {}))
                current_payload["count"] = int(current_payload.get("count", 0) or 0) + int(signal_payload.get("count", 0) or 0)
                current_payload["retryable_count"] = int(current_payload.get("retryable_count", 0) or 0) + int(signal_payload.get("retryable_count", 0) or 0)
                current_payload["non_retryable_count"] = int(current_payload.get("non_retryable_count", 0) or 0) + int(signal_payload.get("non_retryable_count", 0) or 0)
                merged_error_counts = dict(current_payload.get("error_type_counts", {}))
                for error_type, count in dict(signal_payload.get("error_type_counts", {})).items():
                    merged_error_counts[str(error_type)] = int(merged_error_counts.get(str(error_type), 0) or 0) + int(count or 0)
                current_payload["error_type_counts"] = merged_error_counts
                current_payload["severity_score"] = self.calculate_failure_severity(current_payload)
                summary[normalized_name] = current_payload
        return summary

    def reprioritize_tool_skills(
        self,
        tool_skills: list[dict[str, Any]],
        historical_failed_tool_signals: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        signals = historical_failed_tool_signals or {}
        prioritized: list[dict[str, Any]] = []
        for tool_skill in tool_skills or []:
            enriched = dict(tool_skill)
            tool_name = str(enriched.get("name", "") or "")
            signal_payload = dict(signals.get(tool_name, {}))
            severity_score = int(signal_payload.get("severity_score", 0) or 0)
            failure_count = int(signal_payload.get("count", 0) or 0)
            enriched["historical_failure_count"] = failure_count
            enriched["historical_failure_severity"] = severity_score
            if severity_score > 0:
                enriched["route_reason"] = (
                    f"{enriched.get('route_reason', '')} | historical_severity={severity_score} | historical_failures={failure_count}"
                ).strip(" |")
            prioritized.append(enriched)

        prioritized.sort(
            key=lambda item: (
                int(item.get("historical_failure_severity", 0) or 0),
                int(item.get("historical_failure_count", 0) or 0),
                -int(item.get("overlap_count", 0) or 0),
                -float(item.get("match_ratio", 0.0) or 0.0),
                item.get("name", ""),
            )
        )
        return prioritized

    def filter_failed_tools_for_subtask(
        self,
        subtask_index: int,
        selected_tools: list[Any],
        tool_skills: list[dict[str, Any]],
        failed_tools: dict[str, list[str]],
        historical_failed_tools: dict[str, int] | None = None,
        historical_failed_tool_signals: dict[str, dict[str, Any]] | None = None,
        historical_failure_threshold: int = 0,
        historical_failure_severity_threshold: int = 0,
    ) -> tuple[list[Any], list[dict[str, Any]], list[str]]:
        failed_names = set((failed_tools or {}).get(str(subtask_index), []))
        if historical_failed_tools and historical_failure_threshold > 0:
            for tool_name, failure_count in historical_failed_tools.items():
                if failure_count >= historical_failure_threshold:
                    failed_names.add(tool_name)
        if historical_failed_tool_signals and historical_failure_severity_threshold > 0:
            for tool_name, signal_payload in historical_failed_tool_signals.items():
                severity_score = int(dict(signal_payload).get("severity_score", 0) or 0)
                if severity_score >= historical_failure_severity_threshold:
                    failed_names.add(tool_name)
        if not failed_names:
            return selected_tools, tool_skills, []

        filtered_tools = [tool for tool in selected_tools if getattr(tool, "name", "") not in failed_names]
        filtered_tool_skills = [item for item in tool_skills if item.get("name") not in failed_names]
        return filtered_tools, filtered_tool_skills, sorted(failed_names)

    def expand_tool_candidates(
        self,
        task_description: str,
        extracted_keywords: list[str],
        failed_tool_names: list[str],
        historical_failed_tool_signals: dict[str, dict[str, Any]] | None = None,
        historical_failed_tool_counts: dict[str, int] | None = None,
        excluded_tool_names: list[str] | None = None,
        historical_failure_severity_threshold: int = 0,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        excluded = set(excluded_tool_names or []) | set(failed_tool_names or [])
        historical_counts = historical_failed_tool_counts or {}
        historical_signals = historical_failed_tool_signals or {}
        ranked_tool_skills = self.agent.skills.find_relevant_tools(
            task_description,
            extracted_keywords,
            limit=max(limit, len(self.agent.skills.loaded_tool_skills) or limit),
        )
        alternatives: list[dict[str, Any]] = []
        for tool_skill in ranked_tool_skills:
            tool_name = tool_skill.get("name", "")
            if not tool_name or tool_name in excluded:
                continue
            enriched = dict(tool_skill)
            enriched["historical_failure_count"] = int(historical_counts.get(tool_name, 0) or 0)
            signal_payload = dict(historical_signals.get(tool_name, {}))
            enriched["historical_failure_severity"] = int(signal_payload.get("severity_score", 0) or 0)
            if historical_failure_severity_threshold > 0 and enriched["historical_failure_severity"] >= historical_failure_severity_threshold:
                continue
            if enriched["historical_failure_count"] or enriched["historical_failure_severity"]:
                enriched["route_reason"] = (
                    f"{enriched.get('route_reason', '')} | historical_severity={enriched['historical_failure_severity']} | historical_failures={enriched['historical_failure_count']}"
                ).strip(" |")
            alternatives.append(enriched)
            if len(alternatives) >= limit:
                break
        alternatives.sort(
            key=lambda item: (
                int(item.get("historical_failure_severity", 0) or 0),
                int(item.get("historical_failure_count", 0) or 0),
                -int(item.get("overlap_count", 0) or 0),
                -float(item.get("match_ratio", 0.0) or 0.0),
                item.get("name", ""),
            )
        )
        return alternatives

    def build_tool_reroute_plan(
        self,
        subtask_description: str,
        extracted_keywords: list[str],
        selected_tools: list[Any],
        tool_skills: list[dict[str, Any]],
        recent_failures: list[dict[str, Any]],
        failed_tool_names: list[str],
        historical_failed_tool_names: list[str] | None = None,
        historical_failed_tool_counts: dict[str, int] | None = None,
        historical_failed_tool_signals: dict[str, dict[str, Any]] | None = None,
        historical_failure_severity_threshold: int = 0,
    ) -> dict[str, Any]:
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

        high_risk_historical_names = sorted(
            tool_name
            for tool_name, signal_payload in (historical_failed_tool_signals or {}).items()
            if int(dict(signal_payload).get("severity_score", 0) or 0) >= historical_failure_severity_threshold > 0
        )

        alternative_tool_skills = self.expand_tool_candidates(
            subtask_description,
            extracted_keywords,
            sorted(set(failed_tool_names or []) | set(historical_failed_tool_names or [])),
            historical_failed_tool_signals=historical_failed_tool_signals,
            historical_failed_tool_counts=historical_failed_tool_counts,
            excluded_tool_names=[getattr(tool, "name", "") for tool in selected_tools],
            historical_failure_severity_threshold=historical_failure_severity_threshold,
        )
        if alternative_tool_skills:
            return {
                "mode": "alternative_tools",
                "selected_tools": [item["tool"] for item in alternative_tool_skills],
                "tool_skills": alternative_tool_skills,
                "alternatives": [item.get("name", "") for item in alternative_tool_skills],
                "reason": "retryable tool failures triggered a broader alternative-tool search that deprioritizes historically unstable tools",
            }

        if retryable_failures and high_risk_historical_names:
            return {
                "mode": "fallback_high_risk_history",
                "selected_tools": [],
                "tool_skills": [],
                "alternatives": [],
                "reason": (
                    "historical failure severity marked the available tool route as unsafe; "
                    f"high-risk tools={','.join(high_risk_historical_names)}"
                ),
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

    def build_no_tool_guidance(
        self,
        reroute_mode: str,
        recent_failures: list[dict[str, Any]] | None = None,
        reroute_reason: str = "",
    ) -> str:
        normalized_mode = str(reroute_mode or "normal").strip().lower()
        recent_failures = list(recent_failures or [])
        error_types = {str(item.get("error_type", "") or "") for item in recent_failures}

        if normalized_mode == "fallback_invalid_arguments" or "invalid_arguments" in error_types:
            return (
                "当前子任务暂时无法安全使用工具执行。"
                "不要猜测缺失的工具参数或隐藏参数。"
                "请明确向用户询问继续执行所需的缺失输入。"
            )

        if normalized_mode == "fallback_high_risk_history":
            return (
                "当前子任务暂时无法安全使用工具执行，因为剩余工具路径风险较高。"
                "不要调用工具。只有在现有上下文足以安全完成任务时，才直接给出答案。"
                "如果仍需要外部验证、额外参数或实际副作用，请清楚说明限制并询问用户如何继续。"
            )

        if normalized_mode in {"fallback_retryable_no_alternative", "fallback_no_tools", "normal"}:
            guidance = (
                "当前子任务暂时无法使用工具执行。"
                "如果依靠现有上下文就能安全完成任务，请直接给出最佳答案。"
                "如果仍然缺少关键外部数据、参数或必须发生实际副作用，请向用户询问缺失信息或明确说明限制。"
            )
            if reroute_reason:
                guidance += f" 当前限制: {reroute_reason}。"
            return guidance

        return (
            "当前子任务暂时无法使用工具执行。"
            "如果可以安全直接回答，就直接回答；否则请向用户询问缺失信息或下一步决策。"
        )