from llm_manager import llm_manager
from agent_core import agent
from cli_commands import build_commands, handle_plain_message


def start_cli(input_func=input, output_func=print, agent_instance=agent, llm_manager_instance=llm_manager):
    session_state = {"session_id": agent_instance.start_session()}
    commands, command_order = build_commands()
    context = {
        "agent_instance": agent_instance,
        "llm_manager_instance": llm_manager_instance,
        "output_func": output_func,
        "session_state": session_state,
        "commands": commands,
        "command_order": command_order,
    }
    output_func("Welcome to the General Agent CLI (LangGraph 1.x based)")
    output_func(f"Session: {session_state['session_id']}")
    output_func("Type 'help' to see special commands or just type your query.")
    while True:
        try:
            user_input = input_func("\nAgent> ")
            if not user_input.strip():
                continue

            cmd = user_input.split()[0].lower()
            if cmd in {"exit", "quit"}:
                break

            command = commands.get(cmd)
            if command:
                command.handler(user_input, context)
                continue

            handle_plain_message(user_input, context)
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
