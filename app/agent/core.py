import os
import json
import re
import uuid
import importlib.util
import time
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from core.llm.manager import llm_manager
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from core.config import config

# New Cognitive Imports
from cognitive.feature_extractor import CognitiveSystem, DEFAULT_DOMAIN_LABEL
from cognitive.planner import TaskPlanner
from cognitive.reflector import Reflector
from app.agent.observability import AgentObservability
from app.agent.retention import AgentRetentionManager
from app.agent.runtime import AgentRuntime
from app.agent.snapshots import AgentSnapshotStore
from app.agent.skill_parser import SkillManager
from app.agent.tools_runtime import AgentToolRuntime
from memory.memory_manager import MemoryManager
from mcp_servers.mcp_manager import MCPManager

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_query: str
    plan: List[Dict[str, Any]]
    current_subtask_index: int
    reflections: List[str]
    global_keywords: List[str]
    failed_tools: Dict[str, List[str]]
    failed_tool_signals: Dict[str, Dict[str, Dict[str, Any]]]
    request_id: str
    session_id: str
    session_memory_id: int
    domain_label: str
    memory_summaries: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    replan_counts: Dict[str, int]
    blocked: bool
    waiting_for_user: bool
    final_response: str
    lite_mode: bool
    normalized_query: str
    subtask_feature_cache: Dict[str, Dict[str, Any]]
    agent_action: str

class AgentCore:
    def __init__(self, *, auto_load_tools: bool = True, auto_load_mcp: bool = False, build_graph: bool = True):
        self.tools = []
        self.loaded_python_tool_files = set()
        self.loaded_tool_names = set()
        self.graph = None
        self.auto_load_tools = auto_load_tools
        self.auto_load_mcp = auto_load_mcp
        self.auto_build_graph = build_graph
        self.session_id = self._generate_session_id()
        self.last_request_id = ""
        self._request_cancellations: Dict[str, Event] = {}
        self._cancelled_request_ids: set[str] = set()
        self._request_lock = Lock()
        self._request_event_factory = Event
        
        # Instantiate sub-systems
        self.cognitive = CognitiveSystem()
        self.planner = TaskPlanner()
        self.reflector = Reflector()
        self.retention = AgentRetentionManager(self)
        self.memory = MemoryManager(retention_callback=self.retention.maybe_auto_prune)
        self.mcp = MCPManager()
        self.skills = SkillManager()
        self.snapshot_store = AgentSnapshotStore(self)
        self.observability = AgentObservability(self)
        self.runtime = AgentRuntime(self)
        self.tool_runtime = AgentToolRuntime(self)
        
        if self.auto_load_tools:
            self._auto_load_tools()
        if self.auto_load_mcp:
            self._auto_load_mcp_servers()
        if self.auto_build_graph:
            self._build_graph()

    def _generate_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:12]}"

    def _generate_request_id(self) -> str:
        return f"req_{uuid.uuid4().hex[:12]}"

    def start_session(self, session_id: str = None) -> str:
        self.session_id = session_id or self._generate_session_id()
        return self.session_id

    def _register_request(self, request_id: str) -> Event:
        return self.runtime.register_request(request_id)

    def _clear_request(self, request_id: str) -> None:
        self.runtime.clear_request(request_id)

    def is_request_cancelled(self, request_id: str) -> bool:
        return self.runtime.is_request_cancelled(request_id)

    def is_request_active(self, request_id: str) -> bool:
        return self.runtime.is_request_active(request_id)

    def cancel_request(self, request_id: str) -> str:
        return self.runtime.cancel_request(request_id)

    def _raise_if_request_cancelled(self, request_id: str) -> None:
        self.runtime.raise_if_request_cancelled(request_id)

    def _wait_for_graph_result(self, future, request_id: str, timeout_seconds: int):
        return self.runtime.wait_for_graph_result(future, request_id, timeout_seconds)

    def _snapshot_request_dir(self, request_id: str, create: bool = True) -> str:
        return self.snapshot_store.snapshot_request_dir(request_id, create=create)

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        return self.snapshot_store.serialize_message(message)

    def _deserialize_message(self, payload: Dict[str, Any]) -> BaseMessage:
        return self.snapshot_store.deserialize_message(payload)

    def _serialize_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.snapshot_store.serialize_state_snapshot(state)

    def _persist_state_snapshot(
        self,
        request_id: str,
        stage: str,
        state: Dict[str, Any] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        return self.snapshot_store.persist_state_snapshot(request_id, stage, state=state, extra=extra)

    def _resolve_snapshot_path(self, request_id: str, snapshot_name: str | None = None) -> str:
        return self.snapshot_store.resolve_snapshot_path(request_id, snapshot_name=snapshot_name)

    def list_snapshots(self, request_id: str) -> List[Dict[str, Any]]:
        return self.snapshot_store.list_snapshots(request_id)

    def _derive_request_status(self, latest_stage: str, latest_state: Dict[str, Any], active: bool) -> str:
        return self.observability.derive_request_status(latest_stage, latest_state, active)

    def _extract_bool_from_event(self, event: Dict[str, Any], key: str) -> bool | None:
        return self.observability.extract_bool_from_event(event, key)

    def _parse_logged_at(self, value: str) -> datetime | None:
        return self.observability.parse_logged_at(value)

    def _build_request_metrics(self, events: List[Dict[str, Any]], latest_state: Dict[str, Any], status: str) -> Dict[str, Any]:
        return self.observability.build_request_metrics(events, latest_state, status)

    def get_request_summary(self, request_id: str) -> Dict[str, Any] | None:
        return self.observability.get_request_summary(request_id)

    def get_recent_request_summaries(
        self,
        limit: int = 10,
        statuses: List[str] | None = None,
        resumed_only: bool = False,
        attention_only: bool = False,
        since_seconds: int | None = None,
    ) -> List[Dict[str, Any]]:
        return self.observability.get_recent_request_summaries(
            limit=limit,
            statuses=statuses,
            resumed_only=resumed_only,
            attention_only=attention_only,
            since_seconds=since_seconds,
        )

    def get_request_rollup(
        self,
        limit: int = 20,
        statuses: List[str] | None = None,
        resumed_only: bool = False,
        attention_only: bool = False,
        since_seconds: int | None = None,
    ) -> Dict[str, Any]:
        return self.observability.get_request_rollup(
            limit=limit,
            statuses=statuses,
            resumed_only=resumed_only,
            attention_only=attention_only,
            since_seconds=since_seconds,
        )

    def list_tool_runs(self, request_id: str = "", status: str = "") -> List[Dict[str, Any]]:
        return self.tool_runtime.list_tracked_tool_runs(request_id=request_id, status=status)

    def get_retention_status(self) -> Dict[str, Any]:
        return self.retention.get_retention_status()

    def prune_runtime_data(self, apply: bool = False) -> Dict[str, Any]:
        return self.retention.prune_runtime_data(apply=apply)

    def get_failure_memories(
        self,
        match_keywords: List[str] | None = None,
        limit: int = 5,
        exclude_conv_id: str | None = None,
        exclude_ids: List[int] | None = None,
    ) -> List[Dict[str, Any]]:
        return self.memory.retrieve_failure_memories(
            match_keywords=match_keywords,
            limit=limit,
            exclude_conv_id=exclude_conv_id,
            exclude_ids=exclude_ids,
        )

    def _load_snapshot_payload(self, request_id: str, snapshot_name: str | None = None) -> Dict[str, Any] | None:
        return self.snapshot_store.load_snapshot_payload(request_id, snapshot_name=snapshot_name)

    def _restore_state_from_snapshot(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        return self.snapshot_store.restore_state_from_snapshot(payload, request_id)

    def _normalize_execution_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "decomposable":
            return "decomposable"
        return "leaf"

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
                        self._normalize_execution_mode(item.get("execution_mode", "leaf")),
                    )
                )
            return normalized

        return normalize(original_plan) != normalize(candidate_plan)

    def _promote_retry_to_decomposable_subtask(
        self,
        original_request: str,
        current_subtask: Dict[str, Any],
        actual_output: str,
        reflection_note: str,
        recent_tool_failures: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any] | None:
        failure_lines = []
        for item in recent_tool_failures or []:
            failure_lines.append(
                f"- tool={item.get('tool')} | error_type={item.get('error_type')} | retryable={item.get('retryable')} | message={item.get('message')}"
            )

        wrapper_sections = [
            "请重新规划并完成下面这个失败的子任务，优先补充缺失信息、调整执行顺序，避免重复失败路径。",
            f"原始用户请求: {original_request}",
            f"失败子任务: {current_subtask.get('description', '')}",
            f"当前执行模式: {self._normalize_execution_mode(current_subtask.get('execution_mode', 'leaf'))}",
            f"实际观察到的输出: {actual_output}",
            f"反思说明: {reflection_note}",
            "规划目标: 先分析失败原因，再拆成更安全、更小粒度的步骤继续执行。",
        ]
        if failure_lines:
            wrapper_sections.append("最近的工具失败记录:\n" + "\n".join(failure_lines))

        wrapper_description = "\n\n".join(wrapper_sections)
        promoted_subtask = {
            "id": current_subtask.get("id", 1),
            "description": wrapper_description,
            "execution_mode": "decomposable",
        }
        if not self._plans_are_meaningfully_different([current_subtask], [promoted_subtask]):
            return None
        return promoted_subtask

    def _tokenize_text(self, text: str) -> set[str]:
        normalized = text.replace("_", " ").replace("-", " ").lower()
        expanded_tokens = set()
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized):
            if len(token) <= 1:
                continue
            expanded_tokens.add(token)
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
                for size in (2, 3):
                    if len(token) <= size:
                        continue
                    for index in range(len(token) - size + 1):
                        expanded_tokens.add(token[index:index + size])
        return expanded_tokens

    def _derive_routing_keywords(self, text: str, limit: int = 5) -> List[str]:
        source_text = str(text or "").strip()
        if not source_text or limit <= 0:
            return []

        try:
            extracted_keywords = self.cognitive.extract_keywords(source_text, limit=limit)
        except TypeError:
            extracted_keywords = self.cognitive.extract_keywords(source_text)

        normalized_keywords = []
        seen = set()
        for item in extracted_keywords or []:
            value = str(item).strip()
            marker = value.lower()
            if not value or marker in seen:
                continue
            seen.add(marker)
            normalized_keywords.append(value)
            if len(normalized_keywords) >= limit:
                break

        if normalized_keywords:
            return normalized_keywords[:limit]
        return []

    def _build_compact_summary(self, text: str, max_chars: int = 80) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip())
        if not normalized:
            return ""
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(1, max_chars - 3)].rstrip() + "..."

    def _build_execution_environment_context(self, selected_tools: List[Any]) -> str:
        selected_tool_names = {
            str(getattr(tool, "name", "") or "").strip().lower()
            for tool in (selected_tools or [])
            if str(getattr(tool, "name", "") or "").strip()
        }
        if "bash" not in selected_tool_names:
            return ""
        if os.name == "nt":
            return (
                "当前运行环境:\n"
                "- OS: Windows\n"
                "- 终端: PowerShell\n"
                "- 生成 bash 工具命令时，优先使用 PowerShell 或 cmd 兼容语法，不要使用仅适用于 Linux/macOS/bash 的命令。\n"
                "- 不要生成 powershell -Command、pwsh -Command、cmd /c 这类再包一层 shell 的前缀；请直接输出 allowlist 中允许的命令本体。\n"
                "- 文件搜索优先使用 Windows 兼容命令，例如 Get-ChildItem、dir /s、where、findstr。\n"
                "- bash 命令必须直接输出可验证结果，不能只依赖 exit code。优先输出完整路径、匹配数量，或明确的未找到标记。\n"
                "- 例如查找文件时，优先让命令输出 FullName、Resolve-Path 结果，或输出 count=0 这类可验证信号。"
            )
        return (
            "当前运行环境:\n"
            f"- OS: {os.name}\n"
            "- 生成终端命令时，请与当前运行环境兼容，不要假设是其他操作系统。\n"
            "- 终端命令必须输出可验证结果，不要只依赖成功退出码。"
        )

    def _build_tool_retry_guidance(self, failure_payload: Dict[str, Any]) -> str:
        tool_name = str(failure_payload.get("tool", "工具") or "工具").strip()
        error_type = str(failure_payload.get("error_type", "execution_error") or "execution_error").strip()
        failure_message = str(failure_payload.get("message", "") or "").strip()
        failure_suffix = f" 失败信息: {failure_message}" if failure_message else ""

        if error_type == "no_output":
            return (
                f"上一个 {tool_name} 调用执行成功但没有产生可验证输出。"
                "不要根据退出码或上下文臆测结果。"
                "请先判断是命令范围过窄、命令本身没有打印结果，还是目标确实不存在。"
                "然后重新选择一个能直接输出证据的命令。"
                "如果继续使用 bash，命令本身必须打印完整路径、匹配数量，或明确的未找到标记。"
                + failure_suffix
            )

        if bool(failure_payload.get("retryable", False)):
            return (
                f"上一个 {tool_name} 调用失败，但该失败可重试。"
                "请根据失败原因排查问题，并重新选择更稳妥的命令或工具调用。"
                "不要重复同一个明显无效的调用。"
                + failure_suffix
            )

        return (
            f"上一个 {tool_name} 调用失败，且当前失败不适合盲目重试。"
            "请根据错误信息调整方案；如果仍缺少必要输入，就明确向用户询问。"
            + failure_suffix
        )

    def _build_tool_success_guidance(self) -> str:
        return (
            "请严格基于最新工具结果继续处理当前子任务。"
            "如果工具结果已经足够支持结论，请直接给出最终答复，并且必须使用中文。"
            "如果工具结果仍不足以支持结论，请继续调用合适的工具或明确说明缺失信息。"
            "如果上一个工具已经成功且返回了完整结果，不要重复调用同一个工具和相同目的的参数。"
            "不要无依据切换到英文，也不要输出与工具结果不一致的结论。"
        )

    def _summarize_console_text(self, text: str, max_chars: int = 120) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(1, max_chars - 3)].rstrip() + "..."

    def _build_plan_console_preview(self, plan: List[Dict[str, Any]], max_items: int = 3) -> str:
        lines: list[str] = []
        for item in list(plan or [])[:max_items]:
            description = self._summarize_console_text(str(item.get("description", "") or ""), max_chars=72)
            execution_mode = self._normalize_execution_mode(item.get("execution_mode", "leaf"))
            if description:
                lines.append(f"{item.get('id', len(lines) + 1)}. [{execution_mode}] {description}")
        if len(plan or []) > max_items:
            lines.append(f"... 其余 {len(plan) - max_items} 步省略")
        return " || ".join(lines)

    def _build_leaf_routing_keywords(self, state: AgentState, subtask_desc: str, limit: int = 5) -> List[str]:
        seen: set[str] = set()
        normalized_keywords: list[str] = []
        for item in list(state.get("global_keywords", []) or []) + sorted(self._tokenize_text(subtask_desc)):
            value = str(item).strip()
            marker = value.lower()
            if not value or marker in seen:
                continue
            seen.add(marker)
            normalized_keywords.append(value)
            if len(normalized_keywords) >= limit:
                break
        return normalized_keywords

    def _expand_decomposable_subtask(
        self,
        original_request: str,
        current_subtask: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        subtask_description = str(current_subtask.get("description", "") or "").strip()
        if not subtask_description:
            return []

        nested_query = (
            "请继续拆解下面这个复杂子任务，并保持与原始请求一致。\n\n"
            f"原始用户请求: {original_request}\n"
            f"当前复杂子任务: {subtask_description}"
        )
        candidate_plan = self._split_task_with_capabilities(nested_query)
        if not candidate_plan:
            return []

        normalized_candidate_plan: list[dict[str, Any]] = []
        for index, item in enumerate(candidate_plan, start=1):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "") or "").strip()
            if not description:
                continue
            normalized_candidate_plan.append(
                {
                    "id": item.get("id", index),
                    "description": description,
                    "execution_mode": self._normalize_execution_mode(item.get("execution_mode", "leaf")),
                }
            )

        if not normalized_candidate_plan:
            return []

        if len(normalized_candidate_plan) == 1:
            only_item = normalized_candidate_plan[0]
            if only_item["description"].strip().lower() == subtask_description.lower():
                only_item["execution_mode"] = "leaf"

        if not self._plans_are_meaningfully_different([current_subtask], normalized_candidate_plan):
            return []
        return normalized_candidate_plan

    def _parse_tool_message_payload(self, message: BaseMessage) -> Dict[str, Any] | None:
        if not isinstance(message, ToolMessage):
            return None
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            return None
        try:
            payload = json.loads(content)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _get_latest_successful_tool_name_to_pause(self, messages: List[BaseMessage]) -> str:
        if len(messages) < 2 or not isinstance(messages[-1], ToolMessage):
            return ""
        payload = self._parse_tool_message_payload(messages[-1])
        if not payload or not bool(payload.get("ok", False)):
            return ""
        if bool(payload.get("truncated", False)):
            return ""
        previous_ai = None
        for item in reversed(messages[:-1]):
            if isinstance(item, AIMessage) and getattr(item, "tool_calls", None):
                previous_ai = item
                break
        if previous_ai is None:
            return ""
        tool_calls = list(getattr(previous_ai, "tool_calls", None) or [])
        if len(tool_calls) != 1:
            return ""
        return str(tool_calls[0].get("name") or "").strip()

    def _should_downgrade_ask_user_to_continue(
        self,
        subtask_desc: str,
        actual: str,
        reflection_note: str,
        recent_failures: List[Dict[str, Any]] | None = None,
    ) -> bool:
        sanitized_actual = self._strip_internal_response_markup(actual)
        if not sanitized_actual:
            return False
        if recent_failures:
            return False
        if self._looks_like_missing_information_block(actual, reflection_note):
            return False
        if self._looks_like_tool_capability_block(actual, reflection_note):
            return False
        observation_markers = ["查看", "读取", "列出", "查询", "显示", "检查", "内容", "路径", "文件", "目录"]
        return any(marker in str(subtask_desc or "") for marker in observation_markers)

    def _enrich_successful_request_metadata(
        self,
        state: AgentState,
        final_output: str,
        request_id: str,
    ) -> Dict[str, Any]:
        if state.get("lite_mode", False):
            return {}

        normalized_query = str(state.get("normalized_query", "") or state.get("raw_query", "") or "").strip()
        if not normalized_query:
            normalized_query = self._build_compact_summary(final_output, max_chars=120)
        if not normalized_query:
            return {}

        domain = self.cognitive.determine_domain(normalized_query)
        self._raise_if_request_cancelled(request_id)
        try:
            extracted_keywords, summary = self.cognitive.extract_features(normalized_query, domain_hint=domain)
        except TypeError:
            extracted_keywords, summary = self.cognitive.extract_features(normalized_query)
        self._raise_if_request_cancelled(request_id)

        normalized_keywords = [str(item).strip() for item in extracted_keywords if str(item).strip()][:30]
        if not normalized_keywords:
            normalized_keywords = self._derive_routing_keywords(normalized_query, limit=10)
        normalized_summary = str(summary or "").strip() or self._build_compact_summary(normalized_query)

        session_memory_id = state.get("session_memory_id")
        if session_memory_id:
            self.memory.update_memory(
                session_memory_id,
                summary=normalized_summary,
                keywords=normalized_keywords,
                request_id=request_id,
                domain_label=domain,
            )

        return {
            "domain_label": domain,
            "global_keywords": normalized_keywords,
        }

    def _is_non_explicit_chat(self, text: str) -> bool:
        if not bool(getattr(config, "lite_chat_enabled", True)):
            return False
        normalized = re.sub(r"\s+", " ", str(text or "").strip()).lower()
        if not normalized:
            return False
        if len(normalized) > 24:
            return False
        if re.search(r"\d", normalized):
            return False
        if re.search(r"(请|帮我|怎么|如何|why|how|what|查询|执行|创建|删除|运行)", normalized):
            return False
        patterns = getattr(config, "lite_chat_patterns", [])
        if not isinstance(patterns, list) or not patterns:
            return False
        return any(re.fullmatch(str(pattern), normalized, flags=re.IGNORECASE) for pattern in patterns)

    def _classify_tool_exception(self, exc: Exception) -> tuple[str, bool]:
        return self.tool_runtime.classify_tool_exception(exc)

    def _build_tool_error_payload(self, tool_name: str, error_type: str, retryable: bool, message: str) -> str:
        return self.tool_runtime.build_tool_error_payload(tool_name, error_type, retryable, message)

    def _validate_named_argument_heuristics(self, field_name: str, value: Any) -> str:
        return self.tool_runtime.validate_named_argument_heuristics(field_name, value)

    def _prevalidate_tool_arguments(self, tool_name: str, args_schema, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any] | None, str | None]:
        return self.tool_runtime.prevalidate_tool_arguments(tool_name, args_schema, kwargs)

    def _wrap_tool_for_runtime(self, tool):
        return self.tool_runtime.wrap_tool_for_runtime(tool)

    def _get_planning_capability_context(self) -> Dict[str, Any]:
        return self.skills.get_planning_capability_context()

    def _split_task_with_capabilities(
        self,
        user_query: str,
        thinking_mode: bool = False,
        capability_context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        effective_context = capability_context if capability_context is not None else self._get_planning_capability_context()
        split_task = getattr(self.planner, "split_task")
        try:
            return split_task(
                user_query,
                thinking_mode=thinking_mode,
                capability_context=effective_context,
            )
        except TypeError as exc:
            if "capability_context" not in str(exc):
                raise
            return split_task(user_query, thinking_mode=thinking_mode)

    def _append_builtin_fallback_tools(self, selected_tools: List[Any] | None) -> List[Any]:
        merged_tools: List[Any] = []
        seen_names: set[str] = set()

        for tool in list(selected_tools or []):
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if tool_name and tool_name in seen_names:
                continue
            merged_tools.append(tool)
            if tool_name:
                seen_names.add(tool_name)

        fallback_names = getattr(self.skills, "ALWAYS_APPEND_TOOL_NAMES", {"bash"})
        for tool in self.tools:
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not tool_name or tool_name not in fallback_names or tool_name in seen_names:
                continue
            merged_tools.append(tool)
            seen_names.add(tool_name)

        return merged_tools

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
        return self.tool_runtime.parse_tool_error_payload(content)

    def _collect_recent_tool_failures(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        return self.tool_runtime.collect_recent_tool_failures(messages)

    def _response_needs_tool_fallback(self, response: BaseMessage) -> bool:
        if not isinstance(response, AIMessage):
            return False
        if getattr(response, "tool_calls", None):
            return False
        content = response.content
        if isinstance(content, str):
            return not content.strip()
        return not str(content or "").strip()

    def _normalize_empty_model_response(
        self,
        response: BaseMessage,
        request_id: str,
        subtask_desc: str,
    ) -> BaseMessage:
        if not self._response_needs_tool_fallback(response):
            return response
        llm_manager.log_checkpoint(
            "empty_model_response",
            details=f"subtask={subtask_desc[:120]}",
            request_id=request_id,
            console=False,
        )
        return AIMessage(
            content=(
                "模型本轮没有返回任何可执行内容、工具调用或最终答案。"
                "请重新选择合适的技能/工具，或直接给出可验证的答复。"
            )
        )

    def _merge_failed_tools(
        self,
        failed_tools: Dict[str, List[str]],
        subtask_index: int,
        recent_failures: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        return self.tool_runtime.merge_failed_tools(failed_tools, subtask_index, recent_failures)

    def _merge_failed_tool_signals(
        self,
        failed_tool_signals: Dict[str, Dict[str, Dict[str, Any]]],
        subtask_index: int,
        recent_failures: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self.tool_runtime.merge_failed_tool_signals(failed_tool_signals, subtask_index, recent_failures)

    def _filter_failed_tools_for_subtask(
        self,
        subtask_index: int,
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        failed_tools: Dict[str, List[str]],
        historical_failed_tools: Dict[str, int] | None = None,
        historical_failed_tool_signals: Dict[str, Dict[str, Any]] | None = None,
        historical_failure_threshold: int = 0,
        historical_failure_severity_threshold: int = 0,
    ) -> tuple[List[Any], List[Dict[str, Any]], List[str]]:
        return self.tool_runtime.filter_failed_tools_for_subtask(
            subtask_index,
            selected_tools,
            tool_skills,
            failed_tools,
            historical_failed_tools=historical_failed_tools,
            historical_failed_tool_signals=historical_failed_tool_signals,
            historical_failure_threshold=historical_failure_threshold,
            historical_failure_severity_threshold=historical_failure_severity_threshold,
        )

    def _expand_tool_candidates(
        self,
        task_description: str,
        extracted_keywords: List[str],
        failed_tool_names: List[str],
        historical_failed_tool_signals: Dict[str, Dict[str, Any]] | None = None,
        excluded_tool_names: List[str] | None = None,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        return self.tool_runtime.expand_tool_candidates(
            task_description,
            extracted_keywords,
            failed_tool_names,
            historical_failed_tool_signals=historical_failed_tool_signals,
            excluded_tool_names=excluded_tool_names,
            limit=limit,
        )

    def _reprioritize_tool_skills(
        self,
        tool_skills: List[Dict[str, Any]],
        historical_failed_tool_signals: Dict[str, Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        return self.tool_runtime.reprioritize_tool_skills(
            tool_skills,
            historical_failed_tool_signals=historical_failed_tool_signals,
        )

    def _build_tool_reroute_plan(
        self,
        subtask_description: str,
        extracted_keywords: List[str],
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        recent_failures: List[Dict[str, Any]],
        failed_tool_names: List[str],
        historical_failed_tool_names: List[str] | None = None,
        historical_failed_tool_counts: Dict[str, int] | None = None,
        historical_failed_tool_signals: Dict[str, Dict[str, Any]] | None = None,
        historical_failure_severity_threshold: int = 0,
    ) -> Dict[str, Any]:
        return self.tool_runtime.build_tool_reroute_plan(
            subtask_description,
            extracted_keywords,
            selected_tools,
            tool_skills,
            recent_failures,
            failed_tool_names,
            historical_failed_tool_names=historical_failed_tool_names,
            historical_failed_tool_counts=historical_failed_tool_counts,
            historical_failed_tool_signals=historical_failed_tool_signals,
            historical_failure_severity_threshold=historical_failure_severity_threshold,
        )

    def _build_no_tool_guidance(
        self,
        reroute_mode: str,
        recent_failures: List[Dict[str, Any]] | None = None,
        reroute_reason: str = "",
    ) -> str:
        return self.tool_runtime.build_no_tool_guidance(
            reroute_mode,
            recent_failures=recent_failures,
            reroute_reason=reroute_reason,
        )

    def _strip_internal_response_markup(self, text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"<function-call>.*?</function-call>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _looks_like_tool_capability_block(self, actual: str, reflection_note: str) -> bool:
        combined = "\n".join([
            self._strip_internal_response_markup(actual),
            self._strip_internal_response_markup(reflection_note),
        ]).lower()
        if "<function-call>" in str(actual or "").lower():
            return True
        capability_markers = [
            "当前工具",
            "工具调用限制",
            "无法使用工具",
            "工具受限",
            "工具能力",
            "tool limitations",
        ]
        return any(marker in combined for marker in capability_markers)

    def _looks_like_missing_information_block(self, actual: str, reflection_note: str) -> bool:
        combined = "\n".join([
            self._strip_internal_response_markup(actual),
            self._strip_internal_response_markup(reflection_note),
        ]).lower()
        missing_info_markers = [
            "缺少",
            "未提供",
            "请补充",
            "请确认",
            "参数不足",
            "missing parameter",
            "missing information",
            "need user",
            "need confirmation",
        ]
        return any(marker in combined for marker in missing_info_markers)

    def _build_blocked_user_response(
        self,
        subtask_desc: str,
        actual: str,
        reflection_note: str,
        recent_failures: List[Dict[str, Any]] | None = None,
        *,
        retry_limit: bool = False,
    ) -> str:
        sanitized_actual = self._strip_internal_response_markup(actual)
        sanitized_reflection = self._strip_internal_response_markup(reflection_note)
        recent_failures = list(recent_failures or [])
        error_types = {str(item.get("error_type", "") or "").lower() for item in recent_failures}

        if (
            sanitized_actual
            and not self._looks_like_tool_capability_block(actual, reflection_note)
            and not retry_limit
        ):
            return sanitized_actual

        if "invalid_arguments" in error_types or self._looks_like_missing_information_block(actual, reflection_note):
            prefix = f"当前还不能继续执行“{subtask_desc}”，因为缺少必要信息。"
        elif self._looks_like_tool_capability_block(actual, reflection_note):
            prefix = f"当前没能完成“{subtask_desc}”，因为现有工具能力不足，无法直接完成所需操作。"
        elif retry_limit:
            prefix = f"当前没能完成“{subtask_desc}”，因为这条执行路径已经重试到上限，仍然无法得到可靠结果。"
        else:
            prefix = f"当前暂时无法完成“{subtask_desc}”。"

        detail = sanitized_reflection or sanitized_actual
        if detail:
            return f"{prefix}{detail}"
        return prefix

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
                            actual: str, reflection_note: str, quality_tags: List[str] | None = None,
                            step_keywords: List[str] | None = None, step_summary: str = ""):
        normalized_keywords = [
            str(item).strip() for item in (step_keywords or []) if str(item).strip()
        ]
        if not normalized_keywords:
            normalized_keywords = self._derive_routing_keywords(subtask_desc, limit=10)
        normalized_summary = str(step_summary or "").strip() or self._build_compact_summary(subtask_desc)
        memory_output = actual.strip()
        if reflection_note:
            memory_output = f"{memory_output}\n\nReflection: {reflection_note}".strip()
        normalized_quality_tags = self.memory._normalize_quality_tags(quality_tags or ["pending"])
        memory_type = "failure_case" if self.memory._is_failure_memory("", normalized_quality_tags) else "step"
        self.memory.add_memory(
            session_id,
            domain,
            normalized_keywords[:10],
            f"Step: {normalized_summary}",
            subtask_desc,
            memory_output,
            "",
            request_id=request_id,
            memory_type=memory_type,
            quality_tags=normalized_quality_tags,
        )

    def _should_bypass_intent_rewrite(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return True
        lowered = normalized.lower()
        if normalized.startswith("/"):
            return True
        return lowered in {"exit", "quit", "help"}

    def _get_subtask_features(
        self,
        state: AgentState,
        idx: int,
        subtask_desc: str,
        request_id: str,
    ) -> tuple[List[str], str, Dict[str, Dict[str, Any]]]:
        cache = dict(state.get("subtask_feature_cache", {}))
        cache_key = str(idx)
        cached_entry = cache.get(cache_key, {})
        if cached_entry.get("description") == subtask_desc:
            cached_keywords = [str(item).strip() for item in cached_entry.get("keywords", []) if str(item).strip()]
            cached_summary = str(cached_entry.get("summary", "")).strip()
            return cached_keywords[:5], cached_summary, cache

        normalized_keywords = self._derive_routing_keywords(subtask_desc, limit=5)
        subtask_summary = self._build_compact_summary(subtask_desc)
        cache[cache_key] = {
            "description": subtask_desc,
            "keywords": normalized_keywords,
            "summary": str(subtask_summary or "").strip(),
        }
        return normalized_keywords, str(subtask_summary or "").strip(), cache

    def _build_execution_messages(self, messages: List[BaseMessage], normalized_query: str, reset_history: bool = False) -> List[BaseMessage]:
        if not messages:
            return []
        effective_query = str(normalized_query or "").strip()
        if not effective_query:
            return [messages[0]] if reset_history and messages else list(messages)

        updated_messages = [messages[0]] if reset_history else list(messages)
        first_message = updated_messages[0]
        if isinstance(first_message, HumanMessage) and str(first_message.content) != effective_query:
            updated_messages[0] = HumanMessage(content=effective_query)
        return updated_messages

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

    def _auto_load_tools(self):
        """Automatically scan and load python tools from the tool directory."""
        tools_path = config.resolve_path(config.tool_dir)
        if not os.path.exists(tools_path):
            os.makedirs(tools_path)
            return

        for filename in os.listdir(tools_path):
            if not self._is_auto_loadable_tool_file(filename):
                if filename.endswith(".py") and not filename.startswith("__"):
                    llm_manager.log_event(
                        f"Python tool auto-load skipped | file={filename} | reason=reserved_example_or_test_prefix"
                    )
                continue
            self._load_python_tool_file(os.path.join(tools_path, filename), rebuild_graph=False)

    def _is_auto_loadable_tool_file(self, filename: str) -> bool:
        normalized = os.path.basename(filename).strip().lower()
        if not normalized.endswith(".py"):
            return False
        if normalized.startswith("__"):
            return False
        return not (normalized.startswith("test_") or normalized.startswith("sample_"))

    def _auto_load_mcp_servers(self):
        """Automatically scan and load MCP servers from the mcp directory"""
        mcp_path = config.resolve_path(config.mcp_dir)
        if not os.path.exists(mcp_path):
            os.makedirs(mcp_path)
            return

        for filename in sorted(os.listdir(mcp_path)):
            if filename.endswith((".json", ".yaml", ".yml")) or filename.endswith("_mcp_server.py"):
                self.load_mcp_server(filename, rebuild_graph=False)

    def select_active_tools(self, query: str):
        """
        Future enhancement: Dynamically filter which tools to pass to the LLM 
        based on the user's query semantics, to prevent token overflow.
        Returns the subset of selected tools.
        """
        query_terms = list(self._tokenize_text(query))
        known_tool_terms = {
            keyword
            for tool_skill in self.skills.loaded_tool_skills.values()
            for keyword in tool_skill.get("keywords", [])
        }
        routing_terms = [term for term in query_terms if term in known_tool_terms] or query_terms
        capability_bundle = self.skills.assign_capabilities_to_task(query, routing_terms)
        selected_tools = capability_bundle.get("tools", [])
        return self._append_builtin_fallback_tools(selected_tools)

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

    def _load_python_tool_file(self, file_path: str, rebuild_graph: bool = True):
        filename = os.path.basename(file_path)
        if filename in self.loaded_python_tool_files:
            return False, f"Python tool {filename} is already loaded."

        module_name = os.path.splitext(filename)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "tools"):
                return False, f"Tool file {filename} does not export a tools list."

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
            self.loaded_python_tool_files.add(filename)
            if rebuild_graph:
                self._build_graph()
            llm_manager.log_event(
                f"Python tool loaded | file={filename} | tool_count={len(added_tools)} | rebuild_graph={rebuild_graph}"
            )
            return True, f"Loaded Python tool file: {filename}"
        except Exception as e:
            llm_manager.log_event(
                f"Python tool load failed | file={filename} | error={e}",
                level=40,
            )
            return False, f"Failed to load tool {filename}: {e}"

    def load_tool(self, tool_name: str):
        normalized_name = tool_name.strip()
        if not normalized_name:
            return "Usage: /load_tool <tool_name.py>"

        python_name = normalized_name if normalized_name.endswith(".py") else f"{normalized_name}.py"
        python_path = os.path.join(config.resolve_path(config.tool_dir), python_name)
        if not os.path.exists(python_path):
            return f"Tool not found: {tool_name}"
        _, message = self._load_python_tool_file(python_path)
        return message

    def load_skill(self, skill_name: str):
        normalized_name = skill_name.strip()
        if not normalized_name:
            return "Usage: /load_skill <skill_name.md>"

        if normalized_name.endswith(".md"):
            skill = self.skills.load_skill(normalized_name)
            if not skill:
                return f"Skill not found: {normalized_name}"
            return f"Loaded skill: {skill['name']} ({normalized_name})"

        markdown_name = normalized_name if normalized_name.endswith(".md") else f"{normalized_name}.md"
        skill = self.skills.load_skill(markdown_name)
        if skill:
            return f"Loaded skill: {skill['name']} ({markdown_name})"

        return f"Skill not found: {skill_name}"

    def grant_write_access(self, path: str) -> str:
        normalized = str(path or "").strip()
        if not normalized:
            return "Usage: /grant_write <folder_path>"
        granted_root = config.grant_write_root(normalized)
        return f"Granted write access: {granted_root}"

    def revoke_write_access(self, path: str) -> str:
        normalized = str(path or "").strip()
        if not normalized:
            return "Usage: /revoke_write <folder_path>"
        revoked = config.revoke_write_root(normalized)
        if revoked:
            return f"Revoked write access: {config.resolve_path(normalized)}"
        return f"Write access not found: {config.resolve_path(normalized)}"

    def list_write_access_roots(self) -> List[str]:
        return list(config.list_write_roots())

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

            if self._is_non_explicit_chat(last_message_content):
                lite_keywords = list(self._tokenize_text(last_message_content))[:5]
                domain = DEFAULT_DOMAIN_LABEL
                session_memory_id = 0
                if bool(getattr(config, "lite_chat_persist_memory", False)):
                    session_memory_id = self.memory.add_memory(
                        session_id,
                        domain,
                        lite_keywords,
                        "非明确任务交流，使用轻量流程。",
                        last_message_content,
                        "",
                        "",
                        request_id=request_id,
                        memory_type="session_main",
                        quality_tags=["pending", "lite_mode"],
                    )
                plan = [{
                    "id": 1,
                    "description": "对用户的日常交流进行简洁回复",
                    "execution_mode": "leaf",
                }]
                llm_manager.log_checkpoint(
                    "planning_completed",
                    details=(
                        f"subtask_count=1 | domain={domain} | mode=lite_chat | "
                        f"plan_preview={self._build_plan_console_preview(plan)}"
                    ),
                    request_id=request_id,
                    console=True,
                    session_id=session_id,
                    duration_ms=(time.monotonic() - planning_started_at) * 1000,
                    subtask_count=1,
                    domain=domain,
                    mode="lite_chat",
                )
                next_state = {
                    "plan": plan,
                    "raw_query": last_message_content,
                    "current_subtask_index": 0,
                    "global_keywords": lite_keywords,
                    "reflections": [],
                    "failed_tools": {},
                    "failed_tool_signals": {},
                    "session_id": session_id,
                    "request_id": request_id,
                    "session_memory_id": session_memory_id,
                    "domain_label": domain,
                    "memory_summaries": [],
                    "retry_counts": {},
                    "replan_counts": {},
                    "blocked": False,
                    "waiting_for_user": False,
                    "final_response": "",
                    "lite_mode": True,
                    "normalized_query": last_message_content,
                    "subtask_feature_cache": {},
                    "agent_action": "",
                }
                self._persist_state_snapshot(
                    request_id,
                    "planning_completed",
                    next_state,
                    extra={"subtask_count": 1, "domain": domain, "mode": "lite_chat"},
                )
                return next_state

            normalized_query = last_message_content
            if bool(getattr(config, "intent_rewrite_enabled", True)) and not self._should_bypass_intent_rewrite(last_message_content):
                normalized_query = self.cognitive.rewrite_intent(last_message_content)
                self._raise_if_request_cancelled(request_id)

            normalized_keywords = self._derive_routing_keywords(normalized_query, limit=10)
            domain = DEFAULT_DOMAIN_LABEL
            memory_summaries = self.memory.retrieve_memory(
                match_keywords=normalized_keywords,
                limit=5,
                exclude_conv_id=session_id,
            )

            session_memory_id = self.memory.add_memory(
                session_id,
                domain,
                normalized_keywords,
                "待处理请求，成功后补全摘要。",
                last_message_content,
                "",
                "",
                request_id=request_id,
                memory_type="session_main",
                quality_tags=["pending"],
            )
            
            # 2. Planning (Decompose complex tasks into granular subtasks)
            planning_capability_context = self._get_planning_capability_context()
            plan = self._split_task_with_capabilities(
                normalized_query,
                thinking_mode=False,
                capability_context=planning_capability_context,
            )
            self._raise_if_request_cancelled(request_id)
            llm_manager.log_checkpoint(
                "planning_completed",
                details=(
                    f"subtask_count={len(plan)} | domain={domain} | "
                    f"prompt_skill_count={len(planning_capability_context.get('prompt_skills', []))} | "
                    f"tool_count={len(planning_capability_context.get('tools', []))} | "
                    f"plan_preview={self._build_plan_console_preview(plan)}"
                ),
                request_id=request_id,
                console=True,
                session_id=session_id,
                duration_ms=(time.monotonic() - planning_started_at) * 1000,
                subtask_count=len(plan),
                domain=domain,
                prompt_skill_count=len(planning_capability_context.get("prompt_skills", [])),
                tool_count=len(planning_capability_context.get("tools", [])),
            )
            
            next_state = {
                "plan": plan,
                "raw_query": last_message_content,
                "current_subtask_index": 0,
                "global_keywords": normalized_keywords,
                "reflections": [],
                "failed_tools": {},
                "failed_tool_signals": {},
                "session_id": session_id,
                "request_id": request_id,
                "session_memory_id": session_memory_id,
                "domain_label": domain,
                "memory_summaries": memory_summaries,
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "waiting_for_user": False,
                "final_response": "",
                "lite_mode": False,
                "normalized_query": normalized_query,
                "subtask_feature_cache": {},
                "agent_action": "",
            }
            self._persist_state_snapshot(
                request_id,
                "planning_completed",
                next_state,
                extra={
                    "subtask_count": len(plan),
                    "domain": domain,
                    "prompt_skill_count": len(planning_capability_context.get("prompt_skills", [])),
                    "tool_count": len(planning_capability_context.get("tools", [])),
                },
            )
            return next_state
        
        def call_model_subtask(state: AgentState):
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            self._raise_if_request_cancelled(request_id)
            failed_tools_state = dict(state.get("failed_tools", {}))
            failed_tool_signals_state = dict(state.get("failed_tool_signals", {}))
            recent_tool_failures = self._collect_recent_tool_failures(state.get("messages", []))
            merged_failed_tools = self._merge_failed_tools(failed_tools_state, idx, recent_tool_failures)
            merged_failed_tool_signals = self._merge_failed_tool_signals(failed_tool_signals_state, idx, recent_tool_failures)
            
            if idx >= len(plan):
                final_response = state.get("final_response") or "All subtasks completed successfully."
                return {"messages": [AIMessage(content=final_response)]}

            if state.get("lite_mode", False):
                current_subtask = plan[idx]
                subtask_desc = current_subtask.get("description", "")
                llm_manager.log_checkpoint(
                    "subtask_started",
                    details=f"index={idx + 1} | description={subtask_desc[:120]} | mode=lite_chat",
                    request_id=request_id,
                    console=True,
                    session_id=state.get("session_id", ""),
                    subtask_index=idx + 1,
                    subtask_description=subtask_desc[:120],
                    mode="lite_chat",
                )
                llm = llm_manager.get_llm()
                self._persist_state_snapshot(
                    request_id,
                    "subtask_prepared",
                    state,
                    extra={
                        "subtask_index": idx + 1,
                        "subtask_description": subtask_desc,
                        "selected_tools": [],
                        "failed_tools": [],
                        "failed_tool_signal_tools": [],
                        "mode": "lite_chat",
                    },
                )
                llm_manager.log_checkpoint(
                    "subtask_llm_dispatch",
                    details=f"index={idx + 1} | tool_count=0 | after_tool=False | mode=lite_chat",
                    request_id=request_id,
                    session_id=state.get("session_id", ""),
                    subtask_index=idx + 1,
                    tool_count=0,
                    after_tool=False,
                    mode="lite_chat",
                )
                response = llm_manager.invoke(state["messages"], source="agent.execute_subtask", llm=llm)
                self._raise_if_request_cancelled(request_id)
                return {
                    "messages": [response],
                    "failed_tools": merged_failed_tools,
                    "failed_tool_signals": merged_failed_tool_signals,
                }
                
            current_subtask = plan[idx]
            subtask_desc = current_subtask.get("description", "")
            execution_mode = self._normalize_execution_mode(current_subtask.get("execution_mode", "leaf"))

            if execution_mode == "decomposable":
                expanded_subtasks = self._expand_decomposable_subtask(
                    state.get("raw_query", ""),
                    current_subtask,
                )
                if expanded_subtasks:
                    updated_plan = list(plan[:idx]) + expanded_subtasks + list(plan[idx + 1:])
                    llm_manager.log_checkpoint(
                        "subtask_replanned",
                        details=f"index={idx + 1} | new_subtask_count={len(expanded_subtasks)}",
                        request_id=request_id,
                        console=False,
                        session_id=state.get("session_id", ""),
                        subtask_index=idx + 1,
                        new_subtask_count=len(expanded_subtasks),
                    )
                    self._persist_state_snapshot(
                        request_id,
                        "subtask_replanned",
                        {
                            **state,
                            "plan": updated_plan,
                            "subtask_feature_cache": {},
                        },
                        extra={
                            "subtask_index": idx + 1,
                            "replacement_count": len(expanded_subtasks),
                            "reason": "decomposable_subtask_expanded",
                        },
                    )
                    return {
                        "plan": updated_plan,
                        "subtask_feature_cache": {},
                        "agent_action": "reenter",
                        "failed_tools": merged_failed_tools,
                        "failed_tool_signals": merged_failed_tool_signals,
                    }

            llm_manager.log_checkpoint(
                "subtask_started",
                details=(
                    f"index={idx + 1} | description={self._summarize_console_text(subtask_desc, max_chars=120)} | "
                    f"mode={execution_mode}"
                ),
                request_id=request_id,
                console=True,
                session_id=state.get("session_id", ""),
                subtask_index=idx + 1,
                subtask_description=subtask_desc[:120],
            )
            
            continuing_after_tool = isinstance(state["messages"][-1], ToolMessage)
            updated_subtask_feature_cache = dict(state.get("subtask_feature_cache", {}))
            if execution_mode == "leaf":
                sub_kws = self._build_leaf_routing_keywords(state, subtask_desc, limit=5)
            else:
                sub_kws, _, updated_subtask_feature_cache = self._get_subtask_features(
                    state,
                    idx,
                    subtask_desc,
                    request_id,
                )
            
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
            historical_failed_tool_counts = self.tool_runtime.summarize_historical_failed_tools(merged_failed_tools, idx)
            historical_failed_tool_signals = self.tool_runtime.summarize_historical_failed_tool_signals(
                merged_failed_tool_signals,
                idx,
            )
            if tool_skills:
                tool_skills = self._reprioritize_tool_skills(
                    tool_skills,
                    historical_failed_tool_signals=historical_failed_tool_signals,
                )
                selected_tools = [item["tool"] for item in tool_skills]
            selected_tools, tool_skills, failed_tool_names = self._filter_failed_tools_for_subtask(
                idx,
                selected_tools,
                tool_skills,
                merged_failed_tools,
                historical_failed_tools=historical_failed_tool_counts,
                historical_failed_tool_signals=historical_failed_tool_signals,
                historical_failure_threshold=max(0, int(getattr(config, "historical_tool_failure_reroute_threshold", 2) or 0)),
                historical_failure_severity_threshold=max(0, int(getattr(config, "historical_tool_failure_severity_threshold", 6) or 0)),
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
                    historical_failed_tool_names=sorted(historical_failed_tool_counts.keys()),
                    historical_failed_tool_counts=historical_failed_tool_counts,
                    historical_failed_tool_signals=historical_failed_tool_signals,
                    historical_failure_severity_threshold=max(0, int(getattr(config, "historical_tool_failure_severity_threshold", 6) or 0)),
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
            selected_tools = self._append_builtin_fallback_tools(selected_tools)
            
            normalized_query = str(state.get("normalized_query", "")).strip()
            execution_messages = self._build_execution_messages(
                state["messages"],
                normalized_query,
                reset_history=not continuing_after_tool,
            )
            if continuing_after_tool:
                trailing_tool_failure = self._parse_tool_error_payload(getattr(state["messages"][-1], "content", ""))
                if trailing_tool_failure:
                    messages = execution_messages + [HumanMessage(content=self._build_tool_retry_guidance(trailing_tool_failure))]
                else:
                    messages = execution_messages + [HumanMessage(content=self._build_tool_success_guidance())]
            else:
                # Generate local prompt for LLM only for the initial subtask dispatch.
                prompt_sections = [
                    f"正在执行子任务 {idx+1}: {subtask_desc}",
                ]
                if skill_context:
                    prompt_sections.append(skill_context.strip())
                if tool_skills:
                    tool_context = "建议使用的工具:\n" + "\n".join(
                        f"- {item['name']}: {item.get('description', '')} | reason: {item.get('route_reason', '')}" for item in tool_skills
                    )
                    prompt_sections.append(tool_context)
                environment_context = self._build_execution_environment_context(selected_tools)
                if environment_context:
                    prompt_sections.append(environment_context)
                if recent_tool_failures:
                    failure_context_lines = [
                        f"- {item.get('tool')}: {item.get('error_type')} | retryable={item.get('retryable')} | {item.get('message')}"
                        for item in recent_tool_failures
                    ]
                    prompt_sections.append(
                        "最近的工具失败记录:\n" + "\n".join(failure_context_lines)
                    )
                    if reroute_mode == "alternative_tools" and reroute_plan.get("alternatives"):
                        prompt_sections.append(
                            "重路由后选择的备选工具:\n" +
                            "\n".join(f"- {name}" for name in reroute_plan["alternatives"])
                        )
                    if reroute_plan.get("reason"):
                        prompt_sections.append(f"重路由决策: {reroute_plan['reason']}")
                if not selected_tools:
                    prompt_sections.append(
                        self._build_no_tool_guidance(
                            reroute_mode,
                            recent_failures=recent_tool_failures,
                            reroute_reason=reroute_plan.get("reason", ""),
                        )
                    )
                prompt = "\n\n".join(prompt_sections)
                messages = execution_messages + [HumanMessage(content=prompt)]

            paused_tool_name = self._get_latest_successful_tool_name_to_pause(state["messages"]) if continuing_after_tool else ""
            if paused_tool_name:
                selected_tools = [tool for tool in selected_tools if str(getattr(tool, "name", "") or "").strip() != paused_tool_name]
                tool_skills = [item for item in tool_skills if str(item.get("name", "") or "").strip() != paused_tool_name]

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
                    "failed_tool_signal_tools": sorted(merged_failed_tool_signals.get(str(idx), {}).keys()),
                },
            )
            llm_manager.log_checkpoint(
                "subtask_llm_dispatch",
                details=f"index={idx + 1} | tool_count={len(selected_tools)} | after_tool={continuing_after_tool}",
                request_id=request_id,
                session_id=state.get("session_id", ""),
                subtask_index=idx + 1,
                tool_count=len(selected_tools),
                after_tool=continuing_after_tool,
            )
                
            response = llm_manager.invoke(messages, source="agent.execute_subtask", llm=llm)
            self._raise_if_request_cancelled(request_id)
            response = self._normalize_empty_model_response(
                response,
                request_id,
                subtask_desc,
            )
            self._raise_if_request_cancelled(request_id)
            return {
                "messages": [response],
                "failed_tools": merged_failed_tools,
                "failed_tool_signals": merged_failed_tool_signals,
                "subtask_feature_cache": updated_subtask_feature_cache,
                "agent_action": "",
            }

        def reflect_and_advance(state: AgentState):
            messages = state["messages"]
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            self._raise_if_request_cancelled(request_id)
            
            if idx >= len(plan):
                return {}
                
            current_subtask = plan[idx]
            actual = messages[-1].content

            if state.get("lite_mode", False):
                reflections = list(state.get("reflections", []))
                reflections.append(f"Subtask {idx+1}: lite_mode auto-continue")
                next_index = idx + 1
                failed_tools_state = dict(state.get("failed_tools", {}))
                failed_tool_signals_state = dict(state.get("failed_tool_signals", {}))
                failed_tools_state.pop(str(idx), None)
                failed_tool_signals_state.pop(str(idx), None)
                if next_index >= len(plan):
                    llm_manager.log_checkpoint(
                        "agent_completed",
                        details=f"subtask_count={len(plan)} | mode=lite_chat",
                        request_id=request_id,
                        console=True,
                        mode="lite_chat",
                    )
                    self._finalize_session_memory(state, actual, quality_tags=["success", "lite_mode"])
                    self._persist_state_snapshot(
                        request_id,
                        "agent_completed",
                        state,
                        extra={"subtask_count": len(plan), "final_output": actual, "mode": "lite_chat"},
                    )
                else:
                    llm_manager.log_checkpoint(
                        "subtask_advanced",
                        details=f"next_index={next_index + 1} | mode=lite_chat",
                        request_id=request_id,
                        mode="lite_chat",
                    )
                    self._persist_state_snapshot(
                        request_id,
                        "subtask_advanced",
                        state,
                        extra={"next_index": next_index, "mode": "lite_chat"},
                    )
                return {
                    "current_subtask_index": next_index,
                    "reflections": reflections,
                    "failed_tools": failed_tools_state,
                    "failed_tool_signals": failed_tool_signals_state,
                    "blocked": False,
                    "waiting_for_user": False,
                    "final_response": actual if next_index >= len(plan) else state.get("final_response", ""),
                    "retry_counts": dict(state.get("retry_counts", {})),
                    "replan_counts": dict(state.get("replan_counts", {})),
                }
            
            # 3. Verification & Reflection
            success, reflection_note, action = self.reflector.verify_and_reflect(
                current_subtask.get("description", ""), "", actual
            )
            recent_tool_failures = self._collect_recent_tool_failures(messages)
            if not success and action == "ask_user" and self._should_downgrade_ask_user_to_continue(
                current_subtask.get("description", ""),
                actual if isinstance(actual, str) else str(actual or ""),
                reflection_note,
                recent_failures=recent_tool_failures,
            ):
                success = True
                action = "continue"
                reflection_note = (
                    (str(reflection_note or "").strip() + " ").strip() + "已经拿到可直接返回给用户的观测结果，因此不应阻塞并追问用户。"
                ).strip()
            self._raise_if_request_cancelled(request_id)
            llm_manager.log_checkpoint(
                "reflection_completed",
                details=(
                    f"index={idx + 1} | success={success} | action={action} | "
                    f"reflection={self._summarize_console_text(reflection_note, max_chars=160)}"
                ),
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
            failed_tool_signals_state = dict(state.get("failed_tool_signals", {}))
            subtask_feature_cache = dict(state.get("subtask_feature_cache", {}))
            cached_subtask_features = subtask_feature_cache.get(str(idx), {})
            execution_mode = self._normalize_execution_mode(current_subtask.get("execution_mode", "leaf"))
            if execution_mode != "leaf" or action in {"retry", "ask_user"}:
                self._record_step_memory(
                    state.get("session_id", self.session_id),
                    request_id,
                    state.get("domain_label", DEFAULT_DOMAIN_LABEL),
                    current_subtask.get("description", ""),
                    actual,
                    reflection_note,
                    quality_tags=["success"] if (success or action == "continue") else (["ask_user", "waiting_user"] if action == "ask_user" else ["retry"]),
                    step_keywords=list(cached_subtask_features.get("keywords", [])),
                    step_summary=str(cached_subtask_features.get("summary", "") or ""),
                )

            retry_counts = dict(state.get("retry_counts", {}))
            replan_counts = dict(state.get("replan_counts", {}))
            retry_key = str(idx)
            retry_count = retry_counts.get(retry_key, 0)
            
            if not success and action == "ask_user":
                blocked_message = self._build_blocked_user_response(
                    current_subtask.get("description", "当前子任务"),
                    actual if isinstance(actual, str) else str(actual or ""),
                    reflection_note,
                    recent_failures=recent_tool_failures,
                )
                llm_manager.log_checkpoint(
                    "agent_waiting_user",
                    details=f"index={idx + 1} | action=ask_user",
                    request_id=request_id,
                    level=40,
                    console=True,
                )
                self._persist_state_snapshot(
                    request_id,
                    "agent_waiting_user",
                    {
                        **state,
                        "reflections": reflections,
                        "blocked": False,
                        "waiting_for_user": True,
                        "final_response": blocked_message,
                        "retry_counts": retry_counts,
                        "replan_counts": replan_counts,
                        "failed_tool_signals": failed_tool_signals_state,
                    },
                    extra={"subtask_index": idx + 1, "action": action, "reflection_note": reflection_note},
                )
                self._finalize_session_memory(state, blocked_message, quality_tags=["ask_user", "waiting_user"])
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": False,
                    "waiting_for_user": True,
                    "final_response": blocked_message,
                    "failed_tool_signals": failed_tool_signals_state,
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                }
            
            # Advance to next subtask if successful or if action is 'continue'
            if success or action == "continue":
                next_index = idx + 1
                failed_tools_state.pop(str(idx), None)
                failed_tool_signals_state.pop(str(idx), None)
                completion_metadata: Dict[str, Any] = {}
                if next_index >= len(plan):
                    completion_metadata = self._enrich_successful_request_metadata(state, actual, request_id)
                    completed_state = {**state, **completion_metadata}
                    llm_manager.log_checkpoint(
                        "agent_completed",
                        details=f"subtask_count={len(plan)}",
                        request_id=request_id,
                        console=True,
                    )
                    self._finalize_session_memory(completed_state, actual, quality_tags=["success"])
                    self._persist_state_snapshot(
                        request_id,
                        "agent_completed",
                        completed_state,
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
                    "failed_tool_signals": failed_tool_signals_state,
                    "blocked": False,
                    "waiting_for_user": False,
                    "final_response": actual if next_index >= len(plan) else state.get("final_response", ""),
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                    **completion_metadata,
                }
            
            retry_count += 1
            retry_counts[retry_key] = retry_count
            replan_key = str(idx)
            replan_count = replan_counts.get(replan_key, 0)

            if retry_count == 1 and replan_count < 1:
                promoted_subtask = self._promote_retry_to_decomposable_subtask(
                    state["messages"][0].content,
                    current_subtask,
                    actual,
                    reflection_note,
                    recent_tool_failures=recent_tool_failures,
                )
                if promoted_subtask:
                    updated_plan = list(plan[:idx]) + [promoted_subtask] + list(plan[idx + 1:])
                    replan_counts[replan_key] = replan_count + 1
                    retry_counts.pop(retry_key, None)
                    failed_tools_state.pop(str(idx), None)
                    failed_tool_signals_state.pop(str(idx), None)
                    llm_manager.log_checkpoint(
                        "subtask_replanned",
                        details=f"index={idx + 1} | new_subtask_count=1 | mode=decomposable_reflection",
                        request_id=request_id,
                        console=True,
                        session_id=state.get("session_id", ""),
                        subtask_index=idx + 1,
                        new_subtask_count=1,
                        mode="decomposable_reflection",
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
                            "failed_tool_signals": failed_tool_signals_state,
                        },
                        extra={
                            "subtask_index": idx + 1,
                            "replacement_count": 1,
                            "reflection_note": reflection_note,
                        },
                    )
                    return {
                        "plan": updated_plan,
                        "reflections": reflections,
                        "failed_tools": failed_tools_state,
                        "failed_tool_signals": failed_tool_signals_state,
                        "retry_counts": retry_counts,
                        "replan_counts": replan_counts,
                        "subtask_feature_cache": {},
                        "agent_action": "",
                        "blocked": False,
                        "waiting_for_user": False,
                    }

            if retry_count >= 2:
                blocked_message = self._build_blocked_user_response(
                    current_subtask.get("description", f"子任务 {idx+1}"),
                    actual if isinstance(actual, str) else str(actual or ""),
                    reflection_note,
                    recent_failures=recent_tool_failures,
                    retry_limit=True,
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
                    {
                        **state,
                        "reflections": reflections,
                        "blocked": True,
                        "waiting_for_user": False,
                        "final_response": blocked_message,
                        "retry_counts": retry_counts,
                        "replan_counts": replan_counts,
                        "failed_tool_signals": failed_tool_signals_state,
                    },
                    extra={"subtask_index": idx + 1, "action": "retry_limit", "reflection_note": reflection_note},
                )
                self._finalize_session_memory(state, blocked_message, quality_tags=["blocked", "retry"])
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": True,
                    "waiting_for_user": False,
                    "final_response": blocked_message,
                    "failed_tool_signals": failed_tool_signals_state,
                    "retry_counts": retry_counts,
                    "replan_counts": replan_counts,
                }

            return {
                "reflections": reflections,
                "failed_tools": failed_tools_state,
                "failed_tool_signals": failed_tool_signals_state,
                "retry_counts": retry_counts,
                "replan_counts": replan_counts,
                "agent_action": "",
                "blocked": False,
                "waiting_for_user": False,
            }

        def should_continue(state: AgentState) -> Literal["agent", "tools", "reflect_and_advance", "__end__"]:
            if state.get("agent_action") == "reenter":
                return "agent"
            messages = state["messages"]
            last_message = messages[-1]
            
            if self.tools and last_message.tool_calls:
                return "tools"
            
            return "reflect_and_advance"

        def should_continue_after_reflection(state: AgentState) -> Literal["agent", "__end__"]:
            if state.get("blocked") or state.get("waiting_for_user"):
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
        agent_routes = {
            "agent": "agent",
            "reflect_and_advance": "reflect_and_advance",
            "__end__": END,
        }
        if self.tools:
            agent_routes["tools"] = "tools"
        graph_builder.add_conditional_edges("agent", should_continue, agent_routes)
        if self.tools:
            graph_builder.add_edge("tools", "agent")

        graph_builder.add_conditional_edges("reflect_and_advance", should_continue_after_reflection, {
            "agent": "agent",
            "__end__": END
        })
        
        self.graph = graph_builder.compile()

    def invoke(self, query: str, session_id: str = None):
        return self.runtime.invoke(query, session_id=session_id)

    def get_last_request_id(self) -> str:
        return self.last_request_id

    def resume_from_snapshot(self, request_id: str, snapshot_name: str = None, reroute: bool = False):
        return self.runtime.resume_from_snapshot(request_id, snapshot_name=snapshot_name, reroute=reroute)

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
        filepath = os.path.join(self.skills.skill_dir, f"{name}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.skills.load_skill(f"{name}.md", force_reload=True)
        return f"Successfully converted memory {memory_id} to skill {name}.md"

agent = AgentCore()
