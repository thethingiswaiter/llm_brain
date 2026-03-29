import os
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
from skills_md.skill_parser import SkillManager

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Advanced state variables for new architecture
    plan: List[Dict[str, Any]]  
    current_subtask_index: int
    reflections: List[str]
    global_keywords: List[str]

class AgentCore:
    def __init__(self):
        self.tools = []
        self.mcp_servers = []
        self.graph = None
        
        # Instantiate sub-systems
        self.cognitive = CognitiveSystem()
        self.planner = TaskPlanner()
        self.reflector = Reflector()
        self.memory = MemoryManager()
        self.skills = SkillManager()
        
        self._auto_load_skills()
        self._auto_load_mcp_servers()
        self._build_graph()

    def _auto_load_skills(self):
        """Automatically scan and load python skills from the skills directory"""
        skills_path = os.path.join(os.path.dirname(__file__), config.skills_dir)
        if not os.path.exists(skills_path):
            os.makedirs(skills_path)
            return

        for filename in os.listdir(skills_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                file_path = os.path.join(skills_path, filename)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Convention: look for 'tools' list in the module
                    if hasattr(module, "tools"):
                        self.tools.extend(module.tools)
                        print(f"Loaded skills from {filename}")
                except Exception as e:
                    print(f"Failed to load skill {filename}: {e}")

    def _auto_load_mcp_servers(self):
        """Automatically scan and load MCP servers from the mcp directory"""
        mcp_path = os.path.join(os.path.dirname(__file__), config.mcp_dir)
        if not os.path.exists(mcp_path):
            os.makedirs(mcp_path)
            return
            
        # Placeholder for scanning MCP config files (like JSON/YAML)
        for filename in os.listdir(mcp_path):
            if filename.endswith(".json") or filename.endswith(".yaml"):
                print(f"Found MCP server config: {filename} (Load logic to be implemented)")

    def select_active_tools(self, query: str):
        """
        Future enhancement: Dynamically filter which tools to pass to the LLM 
        based on the user's query semantics, to prevent token overflow.
        Returns the subset of selected tools.
        """
        # Right now we just return all tools, but you can add vector search or LLM routing here later.
        selected_tools = self.tools
        return selected_tools

    def add_tool(self, tool):
        self.tools.append(tool)
        self._build_graph()

    def load_mcp_server(self, server_url):
        # Placeholder for real MCP connection logic since it's v1 phase
        print(f"Loading MCP Server from {server_url}")

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        
        def initial_planning(state: AgentState):
            msgs = state["messages"]
            last_message_content = msgs[-1].content if msgs else ""
            
            # 1. Feature Extraction (Global task constraints: max 30 keywords)
            keywords, summary = self.cognitive.extract_features(last_message_content)
            domain = self.cognitive.determine_domain(last_message_content)
            
            # Save raw input to memory system
            self.memory.add_memory("session_1", domain, list(keywords)[:30], summary, last_message_content, "", "")
            
            # 2. Planning (Decompose complex tasks into granular subtasks)
            plan = self.planner.split_task(last_message_content)
            
            return {
                "plan": plan, 
                "current_subtask_index": 0, 
                "global_keywords": list(keywords)[:30], 
                "reflections": []
            }
        
        def call_model_subtask(state: AgentState):
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            
            if idx >= len(plan):
                # All subtasks complete
                return {"messages": [AIMessage(content="All subtasks completed successfully.")]}
                
            current_subtask = plan[idx]
            subtask_desc = current_subtask.get("description", "")
            
            # Subtask feature extraction: 3-5 keywords
            sub_kws, sub_sum = self.cognitive.extract_features(subtask_desc)
            sub_kws = sub_kws[:5]
            
            # Assign Skill
            assigned_skill = self.skills.assign_skill_to_task(subtask_desc, sub_kws)
            skill_context = f"\nUse skill: {assigned_skill['name']}\n{assigned_skill['body']}" if assigned_skill else ""
            
            # Generate local prompt for LLM
            prompt = f"Executing Subtask {idx+1}: {subtask_desc}\n{skill_context}"
            messages = state["messages"] + [HumanMessage(content=prompt)]
            
            selected_tools = self.select_active_tools(subtask_desc)
            llm = llm_manager.get_llm()
            if selected_tools:
                llm = llm.bind_tools(selected_tools)
                
            response = llm.invoke(messages)
            return {"messages": [response]}

        def reflect_and_advance(state: AgentState):
            messages = state["messages"]
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            
            if idx >= len(plan):
                return {}
                
            current_subtask = plan[idx]
            expected = current_subtask.get("expected_outcome", "")
            actual = messages[-1].content
            
            # 3. Verification & Reflection
            success, reflection_note, action = self.reflector.verify_and_reflect(
                current_subtask.get("description", ""), expected, actual
            )
            
            reflections = state.get("reflections", [])
            reflections.append(f"Subtask {idx+1}: {reflection_note}")
            
            if not success and action == "ask_user":
                return {
                    "messages": [AIMessage(content=f"Error blocked subtask. Reflection analysis: {reflection_note}\nNeed user intervention.")],
                    "reflections": reflections
                }
            
            # Advance to next subtask if successful or if action is 'continue'
            if success or action == "continue":
                return {
                    "current_subtask_index": idx + 1,
                    "reflections": reflections
                }
            
            # Handled 'retry' implicitly (index not incremented, will re-run subtask)
            return {"reflections": reflections}
            
        def should_continue(state: AgentState) -> Literal["tools", "reflect_and_advance", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            
            if last_message.tool_calls:
                return "tools"
            
            plan = state.get("plan", [])
            idx = state.get("current_subtask_index", 0)
            if idx >= len(plan):
                return "__end__"
                
            # Need user intervention check
            if "Need user intervention" in last_message.content:
                return "__end__"
                
            return "reflect_and_advance"
            
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
            
        graph_builder.add_edge("reflect_and_advance", "agent")
        
        self.graph = graph_builder.compile()

    def invoke(self, query: str):
        if not self.graph:
            return "Graph is not initialized."
        try:
            inputs = {"messages": [HumanMessage(content=query)], "plan": [], "current_subtask_index": 0, "reflections": [], "global_keywords": []}
            return res["messages"][-1].content
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error invoking agent: {e}"

agent = AgentCore()
