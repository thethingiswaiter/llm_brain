import argparse
import sys
from langgraph.graph import StateGraph
from llm_manager import llm_manager
from agent_core import agent

def start_cli():
    print("Welcome to the General Agent CLI (LangGraph 1.x based)")
    print("Type 'help' to see special commands or just type your query.")
    while True:
        try:
            user_input = input("\nAgent> ")
            if not user_input.strip():
                continue
                
            cmd = user_input.split()[0].lower()
            if cmd == "exit" or cmd == "quit":
                break
            elif cmd == "help":
                print("Available commands:")
                print("  /llm <provider> <model> [base_url] [api_key] - Switch LLM provider (ollama/openai)")
                print("  /load_skill <skill_name> - Load a local python skill")
                print("  /load_mcp <server_url> - Load an MCP server")
                print("  <any other text> - Send message to Agent")
            elif cmd == "/llm":
                parts = user_input.split()
                if len(parts) >= 3:
                    provider = parts[1]
                    model = parts[2]
                    base_url = parts[3] if len(parts) > 3 else None
                    api_key = parts[4] if len(parts) > 4 else None
                    print(llm_manager.set_model(provider, model, base_url, api_key))
                else:
                    print("Usage: /llm <provider> <model> [base_url] [api_key]")
            elif cmd == "/load_skill":
                print("Skill loading feature placeholder.")
            elif cmd == "/load_mcp":
                parts = user_input.split()
                if len(parts) >= 2:
                    agent.load_mcp_server(parts[1])
                else:
                    print("Usage: /load_mcp <server_url>")
            else:
                response = agent.invoke(user_input)
                print(f"\nResponse: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_cli()
