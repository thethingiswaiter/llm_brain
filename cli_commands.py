from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CLICommand:
    name: str
    help_text: str
    handler: Callable[[str, dict[str, Any]], bool | None]


def emit_response(output_func, response: str, request_id: str) -> None:
    if request_id:
        output_func(f"Request ID: {request_id}")
    output_func(f"\nResponse: {response}")


def handle_help(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    output_func("Available commands:")
    for item in context["command_order"]:
        if item.startswith("/"):
            output_func(f"  {context['commands'][item].help_text}")
    output_func("  <any other text> - Send message to Agent")
    return True


def handle_llm(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) >= 3:
        provider = parts[1]
        model = parts[2]
        base_url = parts[3] if len(parts) > 3 else None
        api_key = parts[4] if len(parts) > 4 else None
        output_func(context["llm_manager_instance"].set_model(provider, model, base_url, api_key))
    else:
        output_func("Usage: /llm <provider> <model> [base_url] [api_key]")
    return True


def handle_load_skill(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) == 2:
        output_func(context["agent_instance"].load_skill(parts[1]))
    else:
        output_func("Usage: /load_skill <skill_name.py|skill_name.md>")
    return True


def handle_replay(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    agent_instance = context["agent_instance"]
    parts = user_input.split()
    if len(parts) >= 2:
        response = agent_instance.replay(int(parts[1]), parts[2:])
        emit_response(output_func, response, agent_instance.get_last_request_id())
    else:
        output_func("Usage: /replay <memory_id> [injected features...]")
    return True


def handle_convert_skill(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) >= 2:
        output_func(context["agent_instance"].convert_memory_to_skill(int(parts[1])))
    else:
        output_func("Usage: /convert_skill <memory_id>")
    return True


def handle_load_mcp(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) >= 2:
        _, message = context["agent_instance"].load_mcp_server(parts[1])
        output_func(message)
    else:
        output_func("Usage: /load_mcp <config_name.json|config_name.yaml|absolute_path|server.py|stdio:command ...>")
    return True


def handle_list_mcp(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    servers = context["agent_instance"].list_mcp_servers()
    if not servers:
        output_func("No MCP servers loaded.")
    else:
        output_func("Loaded MCP servers:")
        for item in servers:
            output_func(
                f"  {item['name']} | transport={item.get('transport', '-')} | tools={len(item.get('tool_names', []))} | source={item.get('source', '-')}"
            )
    return True


def handle_unload_mcp(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) >= 2:
        _, message = context["agent_instance"].unload_mcp_server(parts[1])
        output_func(message)
    else:
        output_func("Usage: /unload_mcp <server_name|source>")
    return True


def handle_refresh_mcp(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) >= 2:
        _, message = context["agent_instance"].refresh_mcp_server(parts[1])
        output_func(message)
    else:
        output_func("Usage: /refresh_mcp <server_name|source>")
    return True


def handle_resume_snapshot(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    agent_instance = context["agent_instance"]
    parts = user_input.split()
    if len(parts) >= 2:
        snapshot_name = parts[2] if len(parts) > 2 else None
        response = agent_instance.resume_from_snapshot(parts[1], snapshot_name=snapshot_name)
        emit_response(output_func, response, agent_instance.get_last_request_id())
    else:
        output_func("Usage: /resume_snapshot <request_id> [snapshot_file]")
    return True


def handle_list_snapshots(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) >= 2:
        snapshots = context["agent_instance"].list_snapshots(parts[1])
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
    return True


def handle_cancel_request(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) >= 2:
        output_func(context["agent_instance"].cancel_request(parts[1]))
    else:
        output_func("Usage: /cancel_request <request_id>")
    return True


def handle_request_summary(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) < 2:
        output_func("Usage: /request_summary <request_id>")
        return True

    summary = context["agent_instance"].get_request_summary(parts[1])
    if not summary:
        output_func("Request not found.")
        return True

    output_func(f"Request ID: {summary['request_id']}")
    output_func(f"Status: {summary['status']}")
    output_func(f"Session ID: {summary['session_id'] or '-'}")
    output_func(f"Latest stage: {summary['latest_stage'] or '-'}")
    output_func(f"Progress: {summary['subtask_index']}/{summary['plan_length']}")
    output_func(f"Snapshots: {summary['snapshot_count']}")
    output_func(f"Memories: {summary['memory_count']}")
    output_func(f"Checkpoints: {summary['checkpoint_count']}")
    triage = summary.get("triage", {})
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
    if triage:
        output_func(
            "Triage: "
            f"needs_attention={triage.get('needs_attention', False)} | "
            f"resumed={triage.get('is_resumed', False)} | "
            f"failure_stage={triage.get('latest_failure_stage') or '-'} | "
            f"tool_attention={triage.get('tool_attention_count', 0)}"
        )
        if triage.get("last_error_message"):
            output_func(f"Last error: {triage['last_error_message']}")
        if triage.get("latest_failure_details"):
            output_func(f"Failure details: {triage['latest_failure_details']}")
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
    return True


def render_recent_request_rows(output_func, summaries: list[dict[str, Any]], heading: str) -> None:
    if not summaries:
        output_func("No matching requests found.")
        return

    output_func(heading)
    for item in summaries:
        metrics = item.get("metrics", {})
        resumed_suffix = f" | resumed_from={item.get('source_request_id')}" if item.get("source_request_id") else ""
        output_func(
            f"  - {item['request_id']} | status={item['status']} | session={item.get('session_id') or '-'} | "
            f"stage={item.get('latest_stage') or '-'} | updated_at={item.get('updated_at') or '-'} | "
            f"total_ms={metrics.get('total_duration_ms', '-')} | llm_calls={metrics.get('llm_call_count', 0)} | "
            f"tool_calls={metrics.get('tool_call_count', 0)}{resumed_suffix}"
        )


def handle_recent_requests(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    limit = 10
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output_func("Usage: /recent_requests [limit]")
            return True
    summaries = context["agent_instance"].get_recent_request_summaries(limit=limit)
    render_recent_request_rows(output_func, summaries, "Recent requests:")
    return True


def handle_failed_requests(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    limit = 10
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output_func("Usage: /failed_requests [limit]")
            return True
    summaries = context["agent_instance"].get_recent_request_summaries(
        limit=limit,
        statuses=["failed", "blocked", "timed_out", "cancelled"],
    )
    render_recent_request_rows(output_func, summaries, "Recent failed or blocked requests:")
    return True


def handle_resumed_requests(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    limit = 10
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output_func("Usage: /resumed_requests [limit]")
            return True
    summaries = context["agent_instance"].get_recent_request_summaries(limit=limit, resumed_only=True)
    render_recent_request_rows(output_func, summaries, "Recent resumed requests:")
    return True


def handle_new_session(_user_input: str, context: dict[str, Any]) -> bool:
    session_id = context["agent_instance"].start_session()
    context["session_state"]["session_id"] = session_id
    context["output_func"](f"Started new session: {session_id}")
    return True


def handle_plain_message(user_input: str, context: dict[str, Any]) -> bool:
    agent_instance = context["agent_instance"]
    output_func = context["output_func"]
    response = agent_instance.invoke(user_input, session_id=context["session_state"]["session_id"])
    emit_response(output_func, response, agent_instance.get_last_request_id())
    return True


def build_commands() -> tuple[dict[str, CLICommand], list[str]]:
    command_specs = [
        CLICommand("/llm", "/llm <provider> <model> [base_url] [api_key] - Switch LLM provider (ollama/openai)", handle_llm),
        CLICommand("/load_skill", "/load_skill <skill_name> - Load a local python skill", handle_load_skill),
        CLICommand("/replay", "/replay <memory_id> [injected features...] - Replay memory with optional injected features", handle_replay),
        CLICommand("/convert_skill", "/convert_skill <memory_id> - Convert memory into a markdown skill", handle_convert_skill),
        CLICommand("/load_mcp", "/load_mcp <config|server.py|stdio:command ...> - Load an MCP server", handle_load_mcp),
        CLICommand("/list_mcp", "/list_mcp - Show loaded MCP servers and transports", handle_list_mcp),
        CLICommand("/refresh_mcp", "/refresh_mcp <server_name|source> - Refresh an MCP server and re-enumerate tools", handle_refresh_mcp),
        CLICommand("/unload_mcp", "/unload_mcp <server_name|source> - Unload an MCP server and remove its tools", handle_unload_mcp),
        CLICommand("/cancel_request", "/cancel_request <request_id> - Request cooperative cancellation for an active run", handle_cancel_request),
        CLICommand("/list_snapshots", "/list_snapshots <request_id> - List available recovery points for a request", handle_list_snapshots),
        CLICommand("/resume_snapshot", "/resume_snapshot <request_id> [snapshot_file] - Resume execution from a saved snapshot", handle_resume_snapshot),
        CLICommand("/request_summary", "/request_summary <request_id> - Show the aggregated status, checkpoints, snapshots, and memories for a request", handle_request_summary),
        CLICommand("/recent_requests", "/recent_requests [limit] - Show recent request summaries and key metrics", handle_recent_requests),
        CLICommand("/failed_requests", "/failed_requests [limit] - Show recent failed, blocked, timed out, or cancelled requests", handle_failed_requests),
        CLICommand("/resumed_requests", "/resumed_requests [limit] - Show recent requests resumed from snapshots", handle_resumed_requests),
        CLICommand("/new_session", "/new_session - Start a new memory session", handle_new_session),
    ]
    command_map = {item.name: item for item in command_specs}
    command_map["help"] = CLICommand("help", "help - Show available commands", handle_help)
    return command_map, [item.name for item in command_specs]