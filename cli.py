from llm_manager import llm_manager
from agent_core import agent

def start_cli():
    session_id = agent.start_session()
    print("Welcome to the General Agent CLI (LangGraph 1.x based)")
    print(f"Session: {session_id}")
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
                print("  /new_session - Start a new memory session")
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
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    print(agent.load_skill(parts[1]))
                else:
                    print("Usage: /load_skill <skill_name.py|skill_name.md>")
            elif cmd == "/replay":
                parts = user_input.split()
                if len(parts) >= 2:
                    print(agent.replay(int(parts[1]), parts[2:]))
                else:
                    print("Usage: /replay <memory_id> [injected features...]")
            elif cmd == "/convert_skill":
                parts = user_input.split()
                if len(parts) >= 2:
                    print(agent.convert_memory_to_skill(int(parts[1])))
                else:
                    print("Usage: /convert_skill <memory_id>")
            elif cmd == "/load_mcp":
                parts = user_input.split()
                if len(parts) >= 2:
                    _, message = agent.load_mcp_server(parts[1])
                    print(message)
                else:
                    print("Usage: /load_mcp <config_name.json|config_name.yaml|absolute_path>")
            elif cmd == "/new_session":
                session_id = agent.start_session()
                print(f"Started new session: {session_id}")
            else:
                response = agent.invoke(user_input, session_id=session_id)
                print(f"\nResponse: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_cli()
