import sys
from pathlib import Path
import logging

# Keep direct-file execution working, but prefer the root-level main.py entry.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from core.llm.manager import llm_manager
from app.agent.core import agent
from app.cli.commands import build_commands, build_natural_language_command, build_request_mode_label, handle_plain_message
from app.cli.terminal_ui import TerminalUI


def _status_text_for_input(user_input: str, is_command: bool) -> str:
    if not is_command:
        return "Running agent request"
    command_name = user_input.split()[0].lower()
    command_status_map = {
        "/request_summary": "Loading request summary",
        "/history": "Loading conversation history",
        "/recent_requests": "Loading recent requests",
        "/failed_requests": "Loading failed requests",
        "/latest_failure": "Loading latest failure",
        "/resumed_requests": "Loading resumed requests",
        "/request_rollup": "Building request rollup",
        "/list_tool_runs": "Loading tool run status",
        "/detached_tools": "Loading detached tools",
        "/list_snapshots": "Loading snapshots",
        "/resume_snapshot": "Resuming from snapshot",
        "/resume_last_blocked": "Resuming last blocked request",
        "/list_mcp": "Loading MCP servers",
        "/load_mcp": "Loading MCP server",
        "/refresh_mcp": "Refreshing MCP server",
        "/unload_mcp": "Unloading MCP server",
        "/failure_memories": "Loading failure memories",
        "/retention_status": "Loading retention status",
        "/prune_runtime_data": "Pruning runtime data",
        "/replay": "Replaying memory",
        "/convert_skill": "Converting memory to skill",
        "/llm": "Switching model",
        "/load_tool": "Loading tool",
        "/load_skill": "Loading markdown skill",
        "/new_session": "Starting new session",
        "/pick": "Selecting list item",
        "/selection": "Loading selection context",
        "/clear_selection": "Clearing selection context",
    }
    return command_status_map.get(command_name, "Running command")


def _build_prompt_overview(agent_instance) -> dict[str, object]:
    last_request_id = agent_instance.get_last_request_id() if hasattr(agent_instance, "get_last_request_id") else ""
    summary = None
    if last_request_id and hasattr(agent_instance, "get_request_summary"):
        summary = agent_instance.get_request_summary(last_request_id)

    active_tool_count = 0
    detached_tool_count = 0
    if hasattr(agent_instance, "list_tool_runs"):
        try:
            active_tool_count = len(agent_instance.list_tool_runs(request_id=last_request_id, status="running"))
            detached_tool_count = len(agent_instance.list_tool_runs(request_id=last_request_id, status="detached"))
        except Exception:
            active_tool_count = 0
            detached_tool_count = 0

    return {
        "request_id": last_request_id or "-",
        "mode": build_request_mode_label(summary) if summary else "-",
        "status": str(summary.get("status") or "idle") if summary else "idle",
        "stage": str(summary.get("latest_stage") or "-") if summary else "-",
        "active_tool_count": active_tool_count,
        "detached_tool_count": detached_tool_count,
    }


def _remember_cli_input(session_state: dict[str, object], user_input: str) -> None:
    history = list(session_state.get("recent_inputs") or [])
    normalized = str(user_input or "").strip()
    if not normalized:
        return
    history.append(normalized)
    session_state["recent_inputs"] = history[-20:]


def _install_terminal_stage_bridge(ui: TerminalUI, llm_manager_instance) -> None:
    logging_facade = getattr(llm_manager_instance, "logging", None)
    if logging_facade is None:
        return
    if getattr(logging_facade, "_terminal_cli_bridge_installed", False):
        return

    original_console_event = logging_facade.console_event

    def bridged_console_event(
        stage: str,
        request_id: str | None = None,
        level: int = logging.INFO,
        details: str = "",
    ) -> None:
        if str(stage or "").strip().lower() == "tool_started":
            return
        normalized_details = str(details or "")
        summarize = getattr(logging_facade, "_summarize_console_details", None)
        format_details = getattr(logging_facade, "_format_console_details", None)
        if callable(summarize):
            normalized_details = summarize(stage, normalized_details)
        if callable(format_details):
            normalized_details = format_details(normalized_details)
        ui.render_stage_event(stage, request_id=request_id or "", level=level, details=normalized_details)

    logging_facade._original_terminal_console_event = original_console_event
    logging_facade.console_event = bridged_console_event
    logging_facade._terminal_cli_bridge_installed = True


def start_cli(input_func=input, output_func=print, agent_instance=agent, llm_manager_instance=llm_manager):
    session_state = {"session_id": agent_instance.start_session(), "recent_inputs": []}
    commands, command_order = build_commands()
    ui = TerminalUI(output_func=output_func, input_func=input_func)
    context = {
        "agent_instance": agent_instance,
        "llm_manager_instance": llm_manager_instance,
        "output_func": output_func,
        "ui": ui,
        "session_state": session_state,
        "commands": commands,
        "command_order": command_order,
    }
    _install_terminal_stage_bridge(ui, llm_manager_instance)
    ui.render_welcome(session_state["session_id"])
    while True:
        try:
            user_input = ui.prompt()
            if not user_input.strip():
                continue

            cmd = user_input.split()[0].lower()
            if cmd in {"exit", "quit"}:
                break

            if cmd not in {"/recent_commands"}:
                _remember_cli_input(session_state, user_input)

            command = commands.get(cmd)
            if command:
                with ui.busy(_status_text_for_input(user_input, is_command=True)):
                    command.handler(user_input, context)
                continue

            mapped_command = build_natural_language_command(user_input)
            if mapped_command:
                ui.emit(f"Mapped to command: {mapped_command}")
                mapped_name = mapped_command.split()[0].lower()
                mapped_handler = commands.get(mapped_name)
                if mapped_handler:
                    with ui.busy(_status_text_for_input(mapped_command, is_command=True)):
                        mapped_handler.handler(mapped_command, context)
                    continue

            with ui.busy(_status_text_for_input(user_input, is_command=False)):
                handle_plain_message(user_input, context)
        except KeyboardInterrupt:
            active_request_id = agent_instance.get_last_request_id()
            if active_request_id and agent_instance.is_request_active(active_request_id):
                ui.emit(f"\n{agent_instance.cancel_request(active_request_id)}")
                continue
            ui.emit("\nExiting...")
            break
        except Exception as e:
            ui.emit(f"Error: {e}")

if __name__ == "__main__":
    start_cli()
