import os
import importlib.util
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from llm_manager import llm_manager
from langgraph.prebuilt import ToolNode
from config import config

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class AgentCore:
    def __init__(self):
        self.tools = []
        self.mcp_servers = []
        self.graph = None
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
        
        def call_model(state: AgentState):
            msgs = state["messages"]
            # Extract query for selection
            last_message_content = msgs[-1].content if msgs else ""
            selected_tools = self.select_active_tools(last_message_content)

            llm = llm_manager.get_llm()
            if selected_tools:
                llm = llm.bind_tools(selected_tools)
            response = llm.invoke(msgs)
            return {"messages": [response]}
            
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return "__end__"
            
        graph_builder.add_node("agent", call_model)
        
        if self.tools:
            tool_node = ToolNode(self.tools)
            graph_builder.add_node("tools", tool_node)
            graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
            graph_builder.add_edge("tools", "agent")
        else:
            graph_builder.add_edge("agent", END)
            
        graph_builder.add_edge(START, "agent")
        self.graph = graph_builder.compile()

    def invoke(self, query: str):
        if not self.graph:
            return "Graph is not initialized."
        try:
            inputs = {"messages": [HumanMessage(content=query)]}
            res = self.graph.invoke(inputs)
            return res["messages"][-1].content
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error invoking agent: {e}"

agent = AgentCore()
