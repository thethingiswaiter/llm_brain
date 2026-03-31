import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from config import config
from llm_manager import RequestCancelledError, llm_manager


class AgentToolRuntime:
    def __init__(self, agent: Any):
        self.agent = agent

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
            request_id = llm_manager.get_request_id()
            self.agent._raise_if_request_cancelled(request_id or "")
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

                future = executor.submit(lambda: tool.invoke(normalized_kwargs))
                start = time.monotonic()
                while True:
                    self.agent._raise_if_request_cancelled(request_id or "")
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

    def filter_failed_tools_for_subtask(
        self,
        subtask_index: int,
        selected_tools: list[Any],
        tool_skills: list[dict[str, Any]],
        failed_tools: dict[str, list[str]],
    ) -> tuple[list[Any], list[dict[str, Any]], list[str]]:
        failed_names = set((failed_tools or {}).get(str(subtask_index), []))
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
        excluded_tool_names: list[str] | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        excluded = set(excluded_tool_names or []) | set(failed_tool_names or [])
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
            alternatives.append(tool_skill)
            if len(alternatives) >= limit:
                break
        return alternatives

    def build_tool_reroute_plan(
        self,
        subtask_description: str,
        extracted_keywords: list[str],
        selected_tools: list[Any],
        tool_skills: list[dict[str, Any]],
        recent_failures: list[dict[str, Any]],
        failed_tool_names: list[str],
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

        alternative_tool_skills = self.expand_tool_candidates(
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