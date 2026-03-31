from llm_manager import llm_manager
from agent_core import agent

def start_cli(input_func=input, output_func=print, agent_instance=agent, llm_manager_instance=llm_manager):
    session_id = agent_instance.start_session()
    output_func("Welcome to the General Agent CLI (LangGraph 1.x based)")
    output_func(f"Session: {session_id}")
    output_func("Type 'help' to see special commands or just type your query.")
    while True:
        try:
            user_input = input_func("\nAgent> ")
            if not user_input.strip():
                continue
                
            cmd = user_input.split()[0].lower()
            if cmd == "exit" or cmd == "quit":
                break
            elif cmd == "help":
                output_func("Available commands:")
                output_func("  /llm <provider> <model> [base_url] [api_key] - Switch LLM provider (ollama/openai)")
                output_func("  /load_skill <skill_name> - Load a local python skill")
                output_func("  /load_mcp <config|server.py|stdio:command ...> - Load an MCP server")
                output_func("  /list_mcp - Show loaded MCP servers and transports")
                output_func("  /refresh_mcp <server_name|source> - Refresh an MCP server and re-enumerate tools")
                output_func("  /unload_mcp <server_name|source> - Unload an MCP server and remove its tools")
                output_func("  /cancel_request <request_id> - Request cooperative cancellation for an active run")
                output_func("  /list_snapshots <request_id> - List available recovery points for a request")
                output_func("  /resume_snapshot <request_id> [snapshot_file] - Resume execution from a saved snapshot")
                output_func("  /request_summary <request_id> - Show the aggregated status, checkpoints, snapshots, and memories for a request")
                output_func("  /recent_requests [limit] - Show recent request summaries and key metrics")
                output_func("  /new_session - Start a new memory session")
                output_func("  <any other text> - Send message to Agent")
            elif cmd == "/llm":
                parts = user_input.split()
                if len(parts) >= 3:
                    provider = parts[1]
                    model = parts[2]
                    base_url = parts[3] if len(parts) > 3 else None
                    api_key = parts[4] if len(parts) > 4 else None
                    output_func(llm_manager_instance.set_model(provider, model, base_url, api_key))
                else:
                    output_func("Usage: /llm <provider> <model> [base_url] [api_key]")
            elif cmd == "/load_skill":
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    output_func(agent_instance.load_skill(parts[1]))
                else:
                    output_func("Usage: /load_skill <skill_name.py|skill_name.md>")
            elif cmd == "/replay":
                parts = user_input.split()
                if len(parts) >= 2:
                    response = agent_instance.replay(int(parts[1]), parts[2:])
                    request_id = agent_instance.get_last_request_id()
                    if request_id:
                        output_func(f"Request ID: {request_id}")
                    output_func(f"\nResponse: {response}")
                else:
                    output_func("Usage: /replay <memory_id> [injected features...]")
            elif cmd == "/convert_skill":
                parts = user_input.split()
                if len(parts) >= 2:
                    output_func(agent_instance.convert_memory_to_skill(int(parts[1])))
                else:
                    output_func("Usage: /convert_skill <memory_id>")
            elif cmd == "/load_mcp":
                parts = user_input.split(maxsplit=1)
                if len(parts) >= 2:
                    _, message = agent_instance.load_mcp_server(parts[1])
                    output_func(message)
                else:
                    output_func("Usage: /load_mcp <config_name.json|config_name.yaml|absolute_path|server.py|stdio:command ...>")
            elif cmd == "/list_mcp":
                servers = agent_instance.list_mcp_servers()
                if not servers:
                    output_func("No MCP servers loaded.")
                else:
                    output_func("Loaded MCP servers:")
                    for item in servers:
                        output_func(
                            f"  {item['name']} | transport={item.get('transport', '-')} | tools={len(item.get('tool_names', []))} | source={item.get('source', '-')}"
                        )
            elif cmd == "/unload_mcp":
                parts = user_input.split(maxsplit=1)
                if len(parts) >= 2:
                    _, message = agent_instance.unload_mcp_server(parts[1])
                    output_func(message)
                else:
                    output_func("Usage: /unload_mcp <server_name|source>")
            elif cmd == "/refresh_mcp":
                parts = user_input.split(maxsplit=1)
                if len(parts) >= 2:
                    _, message = agent_instance.refresh_mcp_server(parts[1])
                    output_func(message)
                else:
                    output_func("Usage: /refresh_mcp <server_name|source>")
            elif cmd == "/resume_snapshot":
                parts = user_input.split()
                if len(parts) >= 2:
                    snapshot_name = parts[2] if len(parts) > 2 else None
                    response = agent_instance.resume_from_snapshot(parts[1], snapshot_name=snapshot_name)
                    request_id = agent_instance.get_last_request_id()
                    if request_id:
                        output_func(f"Request ID: {request_id}")
                    output_func(f"\nResponse: {response}")
                else:
                    output_func("Usage: /resume_snapshot <request_id> [snapshot_file]")
            elif cmd == "/list_snapshots":
                parts = user_input.split()
                if len(parts) >= 2:
                    snapshots = agent_instance.list_snapshots(parts[1])
                    if not snapshots:
                        output_func("No snapshots found.")
                    else:
                        output_func("Available snapshots:")
                        for item in snapshots:
                            output_func(
                                f"  [{item['index']}] {item['file']} | stage={item['stage']} | "
                                f"subtask_index={item['subtask_index']} | blocked={item['blocked']} | completed={item['completed']}"
                            )
                else:
                    output_func("Usage: /list_snapshots <request_id>")
            elif cmd == "/cancel_request":
                parts = user_input.split()
                if len(parts) >= 2:
                    output_func(agent_instance.cancel_request(parts[1]))
                else:
                    output_func("Usage: /cancel_request <request_id>")
            elif cmd == "/request_summary":
                parts = user_input.split()
                if len(parts) >= 2:
                    summary = agent_instance.get_request_summary(parts[1])
                    if not summary:
                        output_func("Request not found.")
                    else:
                        output_func(f"Request ID: {summary['request_id']}")
                        output_func(f"Status: {summary['status']}")
                        output_func(f"Session ID: {summary['session_id'] or '-'}")
                        output_func(f"Latest stage: {summary['latest_stage'] or '-'}")
                        output_func(f"Progress: {summary['subtask_index']}/{summary['plan_length']}")
                        output_func(f"Snapshots: {summary['snapshot_count']}")
                        output_func(f"Memories: {summary['memory_count']}")
                        output_func(f"Checkpoints: {summary['checkpoint_count']}")
                        metrics = summary.get("metrics", {})
                        if metrics:
                            output_func(
                                "Metrics: "
                                f"total_ms={metrics.get('total_duration_ms', '-')} | "
                                f"llm_calls={metrics.get('llm_call_count', 0)} | "
                                f"tool_calls={metrics.get('tool_call_count', 0)} | "
                                f"retries={metrics.get('retry_count', 0)} | "
                                f"reflection_failures={metrics.get('reflection_failure_count', 0)}"
                            )
                        if summary.get("source_request_id"):
                            output_func(f"Source request: {summary['source_request_id']}")
                        if summary.get("final_response"):
                            output_func(f"\nFinal response:\n{summary['final_response']}")
                        if summary["checkpoints"]:
                            output_func("\nRecent checkpoints:")
                            for item in summary["checkpoints"][-5:]:
                                detail_suffix = f" | {item['details']}" if item.get("details") else ""
                                output_func(f"  - {item['logged_at']} | {item['stage']}{detail_suffix}")
                        if summary["memories"]:
                            output_func("\nRelated memories:")
                            for item in summary["memories"][-5:]:
                                tags = ",".join(item.get("quality_tags", [])) or "-"
                                output_func(f"  - #{item['id']} | {item['memory_type']} | tags={tags} | {item['summary']}")
                else:
                    output_func("Usage: /request_summary <request_id>")
            elif cmd == "/recent_requests":
                parts = user_input.split()
                limit = 10
                if len(parts) >= 2:
                    try:
                        limit = max(1, int(parts[1]))
                    except ValueError:
                        output_func("Usage: /recent_requests [limit]")
                        continue
                summaries = agent_instance.get_recent_request_summaries(limit=limit)
                if not summaries:
                    output_func("No recent requests found.")
                else:
                    output_func("Recent requests:")
                    for item in summaries:
                        metrics = item.get("metrics", {})
                        output_func(
                            f"  - {item['request_id']} | status={item['status']} | session={item.get('session_id') or '-'} | "
                            f"updated_at={item.get('updated_at') or '-'} | total_ms={metrics.get('total_duration_ms', '-')} | "
                            f"llm_calls={metrics.get('llm_call_count', 0)} | tool_calls={metrics.get('tool_call_count', 0)}"
                        )
            elif cmd == "/new_session":
                session_id = agent_instance.start_session()
                output_func(f"Started new session: {session_id}")
            else:
                response = agent_instance.invoke(user_input, session_id=session_id)
                request_id = agent_instance.get_last_request_id()
                if request_id:
                    output_func(f"Request ID: {request_id}")
                output_func(f"\nResponse: {response}")
                
        except KeyboardInterrupt:
            active_request_id = agent_instance.get_last_request_id()
            if active_request_id and agent_instance.is_request_active(active_request_id):
                output_func(f"\n{agent_instance.cancel_request(active_request_id)}")
                continue
            output_func("\nExiting...")
            break
        except Exception as e:
            output_func(f"Error: {e}")

if __name__ == "__main__":
    start_cli()
