import logging

from core.llm.manager import llm_manager
from app.agent.core import agent
from app.cli.commands import build_commands, handle_plain_message
from app.cli.terminal_ui import TerminalUI
from core.config import config


def _status_text_for_input(user_input: str, is_command: bool) -> str:
    if not is_command:
        return "Running agent request"
    command_name = user_input.split()[0].lower()
    command_status_map = {
        "/request_summary": "Loading request summary",
        "/recent_requests": "Loading recent requests",
        "/failed_requests": "Loading failed requests",
        "/resumed_requests": "Loading resumed requests",
        "/request_rollup": "Building request rollup",
        "/list_tool_runs": "Loading tool run status",
        "/list_snapshots": "Loading snapshots",
        "/resume_snapshot": "Resuming from snapshot",
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
    }
    return command_status_map.get(command_name, "Running command")


def _install_rich_stage_bridge(ui: TerminalUI, llm_manager_instance) -> None:
    if not ui.rich_enabled:
        return
    logging_facade = getattr(llm_manager_instance, "logging", None)
    if logging_facade is None:
        return
    if getattr(logging_facade, "_rich_cli_bridge_installed", False):
        return

    original_console_event = logging_facade.console_event

    def bridged_console_event(stage: str, request_id: str | None = None, level: int = logging.INFO) -> None:
        ui.render_stage_event(stage, request_id=request_id or "", level=level)

    logging_facade._original_console_event = original_console_event
    logging_facade.console_event = bridged_console_event
    logging_facade._rich_cli_bridge_installed = True


def start_cli(input_func=input, output_func=print, agent_instance=agent, llm_manager_instance=llm_manager):
    session_state = {"session_id": agent_instance.start_session()}
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
    _install_rich_stage_bridge(ui, llm_manager_instance)
    ui.render_welcome(session_state["session_id"])
    while True:
        try:
            if ui.rich_enabled:
                mcp_count = len(agent_instance.list_mcp_servers()) if hasattr(agent_instance, "list_mcp_servers") else 0
                model_label = f"{config.llm_config.provider}:{config.llm_config.model}"
                ui.render_prompt_bar(session_state["session_id"], agent_instance.get_last_request_id(), mcp_count, model_label)
            user_input = ui.prompt()
            if not user_input.strip():
                continue

            cmd = user_input.split()[0].lower()
            if cmd in {"exit", "quit"}:
                break

            command = commands.get(cmd)
            if command:
                with ui.busy(_status_text_for_input(user_input, is_command=True)):
                    command.handler(user_input, context)
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
