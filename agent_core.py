import os
import re
import uuid
import importlib.util
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from llm_manager import llm_manager
from langgraph.prebuilt import ToolNode
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
    request_id: str
    session_id: str
    session_memory_id: int
    domain_label: str
    memory_summaries: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
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

    def _tokenize_text(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}

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
                            actual: str, reflection_note: str):
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
        )

    def _finalize_session_memory(self, state: AgentState, final_output: str):
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
                success, message = self.load_mcp_server(filename, rebuild_graph=False)
                print(message)

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
        tool_name = getattr(tool, "name", None)
        if tool_name and tool_name in self.loaded_tool_names:
            return False
        self.tools.append(tool)
        if tool_name:
            self.loaded_tool_names.add(tool_name)
        self.skills.register_tool(tool, source_type="runtime")
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
                tool_name = getattr(tool, "name", None)
                if tool_name and tool_name in self.loaded_tool_names:
                    continue
                self.tools.append(tool)
                if tool_name:
                    self.loaded_tool_names.add(tool_name)
                added_tools.append(tool)
            self.skills.register_tools(added_tools, source_type="python", source_file=filename)
            self.loaded_python_skill_files.add(filename)
            if rebuild_graph:
                self._build_graph()
            print(f"Loaded skills from {filename}")
            return True, f"Loaded Python skill file: {filename}"
        except Exception as e:
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
            return False, "Usage: /load_mcp <config_name.json|config_name.yaml|absolute_path>"

        if os.path.isabs(normalized_ref):
            config_path = normalized_ref
        else:
            config_path = config.resolve_path(os.path.join(config.mcp_dir, normalized_ref))
            if not os.path.exists(config_path):
                mcp_dir = config.resolve_path(config.mcp_dir)
                for extension in (".json", ".yaml", ".yml"):
                    candidate = os.path.join(mcp_dir, normalized_ref + extension)
                    if os.path.exists(candidate):
                        config_path = candidate
                        break

        success, message, _, tools = self.mcp.load_server(config_path)
        if not success:
            return False, message

        added_tools = []
        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if tool_name and tool_name in self.loaded_tool_names:
                continue
            self.tools.append(tool)
            if tool_name:
                self.loaded_tool_names.add(tool_name)
            added_tools.append(tool)

        self.skills.register_tools(added_tools, source_type="mcp", source_file=os.path.basename(config_path))
        added = len(added_tools)

        if rebuild_graph and added:
            self._build_graph()
        elif rebuild_graph and not added:
            message = f"{message} All declared tools were already registered."

        return True, message

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        
        def initial_planning(state: AgentState):
            msgs = state["messages"]
            last_message_content = msgs[-1].content if msgs else ""
            session_id = state.get("session_id") or self.start_session()
            request_id = state.get("request_id") or self._generate_request_id()
            llm_manager.log_checkpoint(
                "planning_started",
                details=f"session_id={session_id}",
                request_id=request_id,
                console=True,
            )
            
            # 1. Feature Extraction (Global task constraints: max 30 keywords)
            keywords, summary = self.cognitive.extract_features(last_message_content)
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
            )
            
            # 2. Planning (Decompose complex tasks into granular subtasks)
            plan = self.planner.split_task(last_message_content)
            llm_manager.log_checkpoint(
                "planning_completed",
                details=f"subtask_count={len(plan)} | domain={domain}",
                request_id=request_id,
                console=True,
            )
            
            return {
                "plan": plan,
                "current_subtask_index": 0,
                "global_keywords": normalized_keywords,
                "reflections": [],
                "session_id": session_id,
                "request_id": request_id,
                "session_memory_id": session_memory_id,
                "domain_label": domain,
                "memory_summaries": memory_summaries,
                "retry_counts": {},
                "blocked": False,
                "final_response": "",
            }
        
        def call_model_subtask(state: AgentState):
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            
            if idx >= len(plan):
                final_response = state.get("final_response") or "All subtasks completed successfully."
                return {"messages": [AIMessage(content=final_response)]}
                
            current_subtask = plan[idx]
            subtask_desc = current_subtask.get("description", "")
            llm_manager.log_checkpoint(
                "subtask_started",
                details=f"index={idx + 1} | description={subtask_desc[:120]}",
                request_id=request_id,
                console=True,
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
            skill_context = f"\nUse skill: {assigned_skill['name']}\n{assigned_skill['body']}" if assigned_skill else ""
            tool_skills = capability_bundle.get("tool_skills", [])
            selected_tools = capability_bundle.get("tools", [])
            if not selected_tools:
                selected_tools = self.select_active_tools(subtask_desc)
            
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
                    f"- {item['name']}: {item.get('description', '')}" for item in tool_skills
                )
                prompt_sections.append(tool_context)
            prompt = "\n\n".join(prompt_sections)
            messages = state["messages"] + [HumanMessage(content=prompt)]

            llm = llm_manager.get_llm()
            if selected_tools:
                llm = llm.bind_tools(selected_tools)
            llm_manager.log_checkpoint(
                "subtask_llm_dispatch",
                details=f"index={idx + 1} | tool_count={len(selected_tools)}",
                request_id=request_id,
            )
                
            response = llm_manager.invoke(messages, source="agent.execute_subtask", llm=llm)
            return {"messages": [response]}

        def reflect_and_advance(state: AgentState):
            messages = state["messages"]
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            request_id = state.get("request_id", "")
            
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
            )
            
            reflections = list(state.get("reflections", []))
            reflections.append(f"Subtask {idx+1}: {reflection_note}")
            self._record_step_memory(
                state.get("session_id", self.session_id),
                request_id,
                state.get("domain_label", "general"),
                current_subtask.get("description", ""),
                actual,
                reflection_note,
            )

            retry_counts = dict(state.get("retry_counts", {}))
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
                self._finalize_session_memory(state, blocked_message)
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": True,
                    "final_response": blocked_message,
                    "retry_counts": retry_counts,
                }
            
            # Advance to next subtask if successful or if action is 'continue'
            if success or action == "continue":
                next_index = idx + 1
                if next_index >= len(plan):
                    llm_manager.log_checkpoint(
                        "agent_completed",
                        details=f"subtask_count={len(plan)}",
                        request_id=request_id,
                        console=True,
                    )
                    self._finalize_session_memory(state, actual)
                else:
                    llm_manager.log_checkpoint(
                        "subtask_advanced",
                        details=f"next_index={next_index + 1}",
                        request_id=request_id,
                    )
                return {
                    "current_subtask_index": next_index,
                    "reflections": reflections,
                    "blocked": False,
                    "final_response": actual if next_index >= len(plan) else state.get("final_response", ""),
                    "retry_counts": retry_counts,
                }
            
            retry_count += 1
            retry_counts[retry_key] = retry_count
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
                self._finalize_session_memory(state, blocked_message)
                return {
                    "messages": [AIMessage(content=blocked_message)],
                    "reflections": reflections,
                    "blocked": True,
                    "final_response": blocked_message,
                    "retry_counts": retry_counts,
                }

            return {
                "reflections": reflections,
                "retry_counts": retry_counts,
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
        graph_builder.add_edge(START, "planner")
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
        try:
            active_session_id = session_id or self.session_id or self.start_session()
            with llm_manager.request_scope(request_id):
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
                    "request_id": request_id,
                    "session_id": active_session_id,
                    "session_memory_id": 0,
                    "domain_label": "general",
                    "memory_summaries": [],
                    "retry_counts": {},
                    "blocked": False,
                    "final_response": "",
                }
                res = self.graph.invoke(inputs)
                final_output = res["messages"][-1].content
                llm_manager.console_event("agent_finished", request_id=request_id)
                llm_manager.log_event(
                    f"Agent response | session_id={active_session_id}\n{final_output}"
                )
                return final_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            llm_manager.console_event("agent_error", request_id=request_id, level=40)
            llm_manager.log_event(
                f"Agent error | session_id={session_id or self.session_id} | error={e}",
                level=40,
                request_id=request_id,
            )
            return f"Error invoking agent: {e}"

    def get_last_request_id(self) -> str:
        return self.last_request_id

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
