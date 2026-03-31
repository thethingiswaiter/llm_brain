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
                print("  /cancel_request <request_id> - Request cooperative cancellation for an active run")
                print("  /list_snapshots <request_id> - List available recovery points for a request")
                print("  /resume_snapshot <request_id> [snapshot_file] - Resume execution from a saved snapshot")
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
                    response = agent.replay(int(parts[1]), parts[2:])
                    request_id = agent.get_last_request_id()
                    if request_id:
                        print(f"Request ID: {request_id}")
                    print(f"\nResponse: {response}")
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
            elif cmd == "/resume_snapshot":
                parts = user_input.split()
                if len(parts) >= 2:
                    snapshot_name = parts[2] if len(parts) > 2 else None
                    response = agent.resume_from_snapshot(parts[1], snapshot_name=snapshot_name)
                    request_id = agent.get_last_request_id()
                    if request_id:
                        print(f"Request ID: {request_id}")
                    print(f"\nResponse: {response}")
                else:
                    print("Usage: /resume_snapshot <request_id> [snapshot_file]")
            elif cmd == "/list_snapshots":
                parts = user_input.split()
                if len(parts) >= 2:
                    snapshots = agent.list_snapshots(parts[1])
                    if not snapshots:
                        print("No snapshots found.")
                    else:
                        print("Available snapshots:")
                        for item in snapshots:
                            print(
                                f"  [{item['index']}] {item['file']} | stage={item['stage']} | "
                                f"subtask_index={item['subtask_index']} | blocked={item['blocked']} | completed={item['completed']}"
                            )
                else:
                    print("Usage: /list_snapshots <request_id>")
            elif cmd == "/cancel_request":
                parts = user_input.split()
                if len(parts) >= 2:
                    print(agent.cancel_request(parts[1]))
                else:
                    print("Usage: /cancel_request <request_id>")
            elif cmd == "/new_session":
                session_id = agent.start_session()
                print(f"Started new session: {session_id}")
            else:
                response = agent.invoke(user_input, session_id=session_id)
                request_id = agent.get_last_request_id()
                if request_id:
                    print(f"Request ID: {request_id}")
                print(f"\nResponse: {response}")
                
        except KeyboardInterrupt:
            active_request_id = agent.get_last_request_id()
            if active_request_id and agent.is_request_active(active_request_id):
                print(f"\n{agent.cancel_request(active_request_id)}")
                continue
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_cli()
