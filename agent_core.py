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
from llm_manager import llm_manager
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from config import config

# New Cognitive Imports
from cognitive.feature_extractor import CognitiveSystem
from cognitive.planner import TaskPlanner
from cognitive.reflector import Reflector
from agent_observability import AgentObservability
from agent_runtime import AgentRuntime
from agent_snapshots import AgentSnapshotStore
from agent_tools import AgentToolRuntime
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
        self._request_event_factory = Event
        
        # Instantiate sub-systems
        self.cognitive = CognitiveSystem()
        self.planner = TaskPlanner()
        self.reflector = Reflector()
        self.memory = MemoryManager()
        self.mcp = MCPManager()
        self.skills = SkillManager()
        self.snapshot_store = AgentSnapshotStore(self)
        self.observability = AgentObservability(self)
        self.runtime = AgentRuntime(self)
        self.tool_runtime = AgentToolRuntime(self)
        
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
    ) -> List[Dict[str, Any]]:
        return self.observability.get_recent_request_summaries(
            limit=limit,
            statuses=statuses,
            resumed_only=resumed_only,
        )

    def _load_snapshot_payload(self, request_id: str, snapshot_name: str | None = None) -> Dict[str, Any] | None:
        return self.snapshot_store.load_snapshot_payload(request_id, snapshot_name=snapshot_name)

    def _restore_state_from_snapshot(self, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        return self.snapshot_store.restore_state_from_snapshot(payload, request_id)

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
        return self.tool_runtime.classify_tool_exception(exc)

    def _build_tool_error_payload(self, tool_name: str, error_type: str, retryable: bool, message: str) -> str:
        return self.tool_runtime.build_tool_error_payload(tool_name, error_type, retryable, message)

    def _validate_named_argument_heuristics(self, field_name: str, value: Any) -> str:
        return self.tool_runtime.validate_named_argument_heuristics(field_name, value)

    def _prevalidate_tool_arguments(self, tool_name: str, args_schema, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any] | None, str | None]:
        return self.tool_runtime.prevalidate_tool_arguments(tool_name, args_schema, kwargs)

    def _wrap_tool_for_runtime(self, tool):
        return self.tool_runtime.wrap_tool_for_runtime(tool)

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

    def _merge_failed_tools(
        self,
        failed_tools: Dict[str, List[str]],
        subtask_index: int,
        recent_failures: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        return self.tool_runtime.merge_failed_tools(failed_tools, subtask_index, recent_failures)

    def _filter_failed_tools_for_subtask(
        self,
        subtask_index: int,
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        failed_tools: Dict[str, List[str]],
    ) -> tuple[List[Any], List[Dict[str, Any]], List[str]]:
        return self.tool_runtime.filter_failed_tools_for_subtask(subtask_index, selected_tools, tool_skills, failed_tools)

    def _expand_tool_candidates(
        self,
        task_description: str,
        extracted_keywords: List[str],
        failed_tool_names: List[str],
        excluded_tool_names: List[str] | None = None,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        return self.tool_runtime.expand_tool_candidates(
            task_description,
            extracted_keywords,
            failed_tool_names,
            excluded_tool_names=excluded_tool_names,
            limit=limit,
        )

    def _build_tool_reroute_plan(
        self,
        subtask_description: str,
        extracted_keywords: List[str],
        selected_tools: List[Any],
        tool_skills: List[Dict[str, Any]],
        recent_failures: List[Dict[str, Any]],
        failed_tool_names: List[str],
    ) -> Dict[str, Any]:
        return self.tool_runtime.build_tool_reroute_plan(
            subtask_description,
            extracted_keywords,
            selected_tools,
            tool_skills,
            recent_failures,
            failed_tool_names,
        )

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
            if not self._is_auto_loadable_skill_file(filename):
                if filename.endswith(".py") and not filename.startswith("__"):
                    llm_manager.log_event(
                        f"Python skill auto-load skipped | file={filename} | reason=reserved_example_or_test_prefix"
                    )
                continue
            self._load_python_skill_file(os.path.join(skills_path, filename), rebuild_graph=False)

    def _is_auto_loadable_skill_file(self, filename: str) -> bool:
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
            
            if self.tools and last_message.tool_calls:
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
        agent_routes = {
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

    def resume_from_snapshot(self, request_id: str, snapshot_name: str = None):
        return self.runtime.resume_from_snapshot(request_id, snapshot_name=snapshot_name)

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
