from __future__ import annotations

import logging
import threading
from typing import Any
from pathlib import Path

from rich.text import Text
from textual import events
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from app.agent.core import agent
from app.cli.commands import (
    RESPONSE_DIVIDER,
    build_commands,
    build_input_suggestions,
    get_selection_context_label,
    build_natural_language_command,
    handle_plain_message,
    resume_with_followup,
    should_auto_resume_from_followup,
)
from core.config import config
from core.llm.manager import llm_manager
from core.time_utils import now_china


class AutocompleteInput(Input):
    def on_key(self, event: events.Key) -> None:
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            autocomplete = getattr(self.app, "action_autocomplete", None)
            if callable(autocomplete):
                autocomplete()
            return
        if event.key in {"up", "down"}:
            move_selection = getattr(self.app, "move_autocomplete_selection", None)
            if callable(move_selection) and getattr(self.app, "has_autocomplete_suggestions", lambda: False)():
                event.stop()
                event.prevent_default()
                move_selection(-1 if event.key == "up" else 1)
                return
        if event.key == "escape":
            clear_autocomplete = getattr(self.app, "clear_autocomplete", None)
            if callable(clear_autocomplete) and getattr(self.app, "has_autocomplete_suggestions", lambda: False)():
                event.stop()
                event.prevent_default()
                clear_autocomplete()
                return
        if event.key == "enter":
            confirm_selection = getattr(self.app, "confirm_autocomplete_selection", None)
            if callable(confirm_selection) and getattr(self.app, "should_confirm_autocomplete_on_enter", lambda: False)():
                event.stop()
                event.prevent_default()
                confirm_selection()
                return


class AgentTextualApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #shell {
        height: 1fr;
    }

    #status {
        height: auto;
        padding: 0 1;
        color: #dbe4ff;
        background: #111827;
        border: round #3b82f6;
    }

    #workspace {
        height: 1fr;
        margin: 0;
    }

    #log {
        height: 1fr;
        border: round #2f5fd0;
        background: #0b1020;
    }

    #input {
        dock: bottom;
        margin: 0;
        border: round #3b82f6;
        background: #0f172a;
        color: #eff6ff;
    }

    #autocomplete_panel {
        height: auto;
        padding: 0 1;
        color: #cbd5e1;
        background: #172033;
        border: round #4b5563;
        margin: 0;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
        ("ctrl+r", "refresh_dashboard", "Refresh"),
        ("ctrl+o", "browse_workspace", "Browse Files"),
    ]

    def __init__(self, agent_instance=agent, llm_manager_instance=llm_manager):
        super().__init__()
        self.agent_instance = agent_instance
        self.llm_manager_instance = llm_manager_instance
        self.session_state: dict[str, Any] = {
            "session_id": agent_instance.start_session(),
            "recent_inputs": [],
        }
        self.structured_output_enabled = False
        self.commands, self.command_order = build_commands()
        self._busy = False
        self._autocomplete_seed = ""
        self._autocomplete_suggestions: list[str] = []
        self._autocomplete_index = -1
        self._response_block_active = False
        self._last_stage_request_id = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="shell"):
            yield Static(id="status")
            with Vertical(id="workspace"):
                yield RichLog(id="log", markup=False, wrap=True, highlight=False, auto_scroll=True)
            yield Static(id="autocomplete_panel", classes="hidden")
            yield AutocompleteInput(placeholder="输入消息或命令，例如 /summary、/history 5", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self._install_stage_bridge()
        self._write_welcome()
        self._refresh_dashboard()
        self.query_one("#input", Input).focus()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self._write_welcome()
        self._refresh_dashboard()

    def action_refresh_dashboard(self) -> None:
        self._refresh_dashboard()

    def refresh_dashboard(self) -> None:
        self._refresh_dashboard()

    def action_browse_workspace(self) -> None:
        if self._busy:
            self._write("System> 当前仍有请求在执行，请稍后。")
            return
        self._write("You> /workspace_files")
        self._process_input("/workspace_files")

    def set_input_value(self, value: str) -> None:
        input_widget = self.query_one("#input", Input)
        input_widget.value = str(value or "")
        if hasattr(input_widget, "cursor_position"):
            input_widget.cursor_position = len(input_widget.value)
        input_widget.focus()

    def action_autocomplete(self) -> None:
        input_widget = self.query_one("#input", Input)
        current_value = str(input_widget.value or "")
        if not self._supports_autocomplete(current_value):
            return

        if not self._autocomplete_suggestions:
            suggestions = self._build_autocomplete_suggestions(current_value)
            if not suggestions:
                return
            self._autocomplete_seed = current_value
            self._autocomplete_suggestions = suggestions
            self._autocomplete_index = 0
        elif current_value == self._autocomplete_suggestions[self._autocomplete_index]:
            self.move_autocomplete_selection(1)
            return

        self._apply_autocomplete_selection(self._autocomplete_index)

    def on_input_changed(self, event: Input.Changed) -> None:
        if not self._supports_autocomplete(event.value):
            self._clear_autocomplete_state()
            self._refresh_autocomplete_panel()
            return

        suggestions = self._build_autocomplete_suggestions(event.value)
        if not suggestions:
            self._clear_autocomplete_state()
            self._refresh_autocomplete_panel()
            return

        self._autocomplete_seed = event.value
        self._autocomplete_suggestions = suggestions
        if event.value in suggestions:
            self._autocomplete_index = suggestions.index(event.value)
        else:
            self._autocomplete_index = 0
        self._refresh_autocomplete_panel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = str(event.value or "").strip()
        event.input.value = ""
        self._clear_autocomplete_state()
        self._refresh_autocomplete_panel()
        if not user_input:
            return
        if user_input.lower() in {"exit", "quit"}:
            self.exit()
            return
        if self._busy:
            self._write("System> 当前仍有请求在执行，请稍后。")
            return
        self._write(f"You> {user_input}")
        self._process_input(user_input)

    @work(thread=True)
    def _process_input(self, user_input: str) -> None:
        self._set_busy(True)
        try:
            if user_input.split()[0].lower() not in {"/recent_commands"}:
                self._remember_input(user_input)

            context = {
                "agent_instance": self.agent_instance,
                "llm_manager_instance": self.llm_manager_instance,
                "output_func": self._threadsafe_emit,
                "ui": self,
                "session_state": self.session_state,
                "commands": self.commands,
                "command_order": self.command_order,
            }

            command_name = user_input.split()[0].lower()
            command = self.commands.get(command_name)
            if command is not None:
                command.handler(user_input, context)
                return

            mapped_command = build_natural_language_command(user_input)
            if mapped_command:
                self._threadsafe_emit(f"Mapped to command: {mapped_command}")
                mapped_handler = self.commands.get(mapped_command.split()[0].lower())
                if mapped_handler is not None:
                    mapped_handler.handler(mapped_command, context)
                    return

            should_auto_resume, resume_request_id = should_auto_resume_from_followup(self.agent_instance, user_input)
            if should_auto_resume:
                resume_with_followup(context, resume_request_id, user_input)
                return

            handle_plain_message(user_input, context)
        except Exception as exc:
            self._threadsafe_emit(f"Error: {exc}")
        finally:
            self._set_busy(False)

    def _remember_input(self, user_input: str) -> None:
        recent_inputs = list(self.session_state.get("recent_inputs") or [])
        recent_inputs.append(user_input)
        self.session_state["recent_inputs"] = recent_inputs[-20:]

    def _clear_autocomplete_state(self) -> None:
        self._autocomplete_seed = ""
        self._autocomplete_suggestions = []
        self._autocomplete_index = -1

    def clear_autocomplete(self) -> None:
        self._clear_autocomplete_state()
        self._refresh_autocomplete_panel()

    def has_autocomplete_suggestions(self) -> bool:
        return bool(self._autocomplete_suggestions)

    def should_confirm_autocomplete_on_enter(self) -> bool:
        if not self._autocomplete_suggestions or self._autocomplete_index < 0:
            return False
        input_widget = self.query_one("#input", Input)
        current_value = str(input_widget.value or "")
        selected_value = self._autocomplete_suggestions[self._autocomplete_index]
        return self._supports_autocomplete(current_value) and current_value != selected_value

    def confirm_autocomplete_selection(self) -> None:
        if not self._autocomplete_suggestions or self._autocomplete_index < 0:
            return
        self._apply_autocomplete_selection(self._autocomplete_index)

    def move_autocomplete_selection(self, delta: int) -> None:
        if not self._autocomplete_suggestions:
            return
        next_index = (self._autocomplete_index + delta) % len(self._autocomplete_suggestions)
        self._apply_autocomplete_selection(next_index)

    def _apply_autocomplete_selection(self, index: int) -> None:
        if not self._autocomplete_suggestions:
            return
        normalized_index = index % len(self._autocomplete_suggestions)
        self._autocomplete_index = normalized_index
        completed_value = self._autocomplete_suggestions[normalized_index]
        input_widget = self.query_one("#input", Input)
        input_widget.value = completed_value
        if hasattr(input_widget, "cursor_position"):
            input_widget.cursor_position = len(completed_value)
        self._refresh_autocomplete_panel()

    def _build_autocomplete_suggestions(self, prefix: str) -> list[str]:
        context = {
            "agent_instance": self.agent_instance,
            "llm_manager_instance": self.llm_manager_instance,
            "command_order": self.command_order,
            "session_state": self.session_state,
        }
        return build_input_suggestions(context, prefix)

    def _supports_autocomplete(self, value: str) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized.startswith("/") or normalized == "llm" or normalized.startswith("llm ")

    def _build_autocomplete_panel_text(self) -> str:
        if not self._autocomplete_suggestions:
            return ""
        lines = ["Candidates:"]
        for index, item in enumerate(self._autocomplete_suggestions[:6]):
            prefix = ">" if index == self._autocomplete_index else "-"
            lines.append(f"{prefix} {item}")
        return "\n".join(lines)

    def _refresh_autocomplete_panel(self) -> None:
        panel = self.query_one("#autocomplete_panel", Static)
        if not self._autocomplete_suggestions:
            panel.update("")
            panel.add_class("hidden")
            return
        panel.remove_class("hidden")
        panel.update(self._build_autocomplete_panel_renderable())

    def _threadsafe_emit(self, message: str = "") -> None:
        if threading.current_thread() is threading.main_thread():
            self._write(message)
            return
        self.call_from_thread(self._write, message)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        if threading.current_thread() is threading.main_thread():
            self._refresh_dashboard()
            return
        self.call_from_thread(self._refresh_dashboard)

    def _write(self, message: str = "") -> None:
        log = self.query_one("#log", RichLog)
        text = str(message or "")
        lines = text.splitlines() or [""]
        for line in lines:
            renderable, response_block_active = self._build_log_line_renderable(line, self._response_block_active)
            log.write(renderable)
            self._response_block_active = response_block_active

    def render_response(self, request_id: str, response: str, summary: dict[str, Any] | None = None) -> None:
        if request_id:
            self._threadsafe_emit(f"Request ID: {request_id}")
        if summary:
            metrics = summary.get("metrics", {}) or {}
            self._threadsafe_emit("Summary:")
            self._threadsafe_emit(
                "  "
                f"request={summary.get('request_id') or '-'} | "
                f"stage={summary.get('latest_stage') or '-'} | "
                f"mode={'lite_chat' if summary.get('lite_mode') else 'task'} | "
                f"progress={summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)} | "
                f"tool_calls={metrics.get('tool_call_count', 0)} | "
                f"total_ms={metrics.get('total_duration_ms', '-') }"
            )
            raw_query = str(summary.get("raw_query") or "").strip()
            normalized_query = str(summary.get("normalized_query") or "").strip()
            mode_text = "lite_chat" if summary.get("lite_mode") else "task"
            status_text = str(summary.get("status") or "-")
            self._threadsafe_emit("")
            self._threadsafe_emit("Conversation Context:")
            if raw_query:
                self._threadsafe_emit(f"  Input: {raw_query}")
            if normalized_query and normalized_query != raw_query:
                self._threadsafe_emit(f"  Intent: {normalized_query}")
            self._threadsafe_emit(f"  Mode: {mode_text} | Status: {status_text}")
            blocked_reason = str(summary.get("blocked_reason") or "").strip()
            if blocked_reason:
                self._threadsafe_emit("")
                self._threadsafe_emit("Risk:")
                self._threadsafe_emit(f"  Blocked Reason: {blocked_reason}")
            suggestions = summary.get("suggested_actions") or []
            if suggestions:
                self._threadsafe_emit("")
                self._threadsafe_emit("Next Actions:")
                self._threadsafe_emit("  Next: " + " | ".join(str(item) for item in suggestions))
        response_lines = str(response or "").splitlines() or [""]
        self._threadsafe_emit("Agent>")
        for line in response_lines:
            self._threadsafe_emit(line)
        self._threadsafe_emit(RESPONSE_DIVIDER)

    def _write_welcome(self) -> None:
        self._write("llm_brain Textual Terminal")
        self._write(f"Session: {self.session_state['session_id']}")
        self._write("输入普通文本发送给 Agent；输入 help 查看命令；输入 quit 退出。")
        self._write("输入 / 后会出现命令候选；继续输入会实时过滤，Tab 会在候选之间轮换并写回输入框。")
        self._write("输入 /workspace_files 或按 Ctrl+O 浏览当前配置工作区的文件，并可用 /pick 选择。")
        self._write("")

    def _get_latest_summary(self) -> dict[str, Any]:
        last_request_id = self.agent_instance.get_last_request_id() if hasattr(self.agent_instance, "get_last_request_id") else ""
        if not last_request_id or not hasattr(self.agent_instance, "get_request_summary"):
            return {}
        summary = self.agent_instance.get_request_summary(last_request_id) or {}
        if summary:
            summary = dict(summary)
        return summary

    def _build_status_line(self) -> str:
        summary = self._get_latest_summary()
        model_label = f"{config.llm_config.provider}:{config.llm_config.model}"
        status_text = str(summary.get("status") or ("running" if self._busy else "idle"))
        mcp_count = len(self.agent_instance.list_mcp_servers()) if hasattr(self.agent_instance, "list_mcp_servers") else 0
        request_id = str(summary.get("request_id") or "-")
        stage_text = str(summary.get("latest_stage") or "-")
        progress_text = f"{summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)}" if summary else "-"
        workspace_label = Path(config.get_workspace_root()).name or config.get_workspace_root()
        selection_label = get_selection_context_label(self.session_state.get("selection_context") or {}) or "-"
        return (
            f"status {status_text} | request {request_id} | stage {stage_text} | "
            f"progress {progress_text} | workspace {workspace_label} | selection {selection_label} | "
            f"model {model_label} | mcp {mcp_count}"
        )

    def _refresh_status(self) -> None:
        status_widget = self.query_one("#status", Static)
        status_widget.update(self._build_status_renderable())

    def _refresh_dashboard(self) -> None:
        self._refresh_status()

    def _build_status_renderable(self) -> Text:
        summary = self._get_latest_summary()
        status_text = str(summary.get("status") or ("running" if self._busy else "idle"))
        model_text = f"{config.llm_config.provider}:{config.llm_config.model}"
        mcp_count = len(self.agent_instance.list_mcp_servers()) if hasattr(self.agent_instance, "list_mcp_servers") else 0
        request_text = str(summary.get("request_id") or "-")
        stage_text = str(summary.get("latest_stage") or "-")
        progress_text = f"{summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)}" if summary else "-"

        text = Text()
        self._append_status_segment(text, "status", status_text, self._status_style(status_text))
        self._append_separator(text)
        self._append_status_segment(text, "request", request_text, "bold #fde68a")
        self._append_separator(text)
        self._append_status_segment(text, "stage", stage_text, self._stage_style(stage_text))
        self._append_separator(text)
        self._append_status_segment(text, "progress", progress_text, "bold #93c5fd")
        self._append_separator(text)
        self._append_status_segment(text, "model", model_text, "bold #c4b5fd")
        self._append_separator(text)
        self._append_status_segment(text, "mcp", str(mcp_count), "bold #fca5a5")
        return text

    def _format_stage_label(self, stage: str) -> str:
        normalized = str(stage or "").strip().lower()
        stage_map = {
            "agent_started": "agent start",
            "planning_started": "planning",
            "planning_completed": "plan ready",
            "subtask_started": "subtask",
            "reflection_completed": "reflect",
            "subtask_replanned": "replan",
            "tool_started": "tool start",
            "tool_succeeded": "tool ok",
            "tool_failed": "tool failed",
            "tool_reroute_applied": "tool reroute",
            "agent_completed": "agent done",
            "agent_finished": "finished",
            "agent_blocked": "blocked",
            "agent_waiting_user": "waiting",
            "agent_timeout": "timeout",
        }
        return stage_map.get(normalized, normalized.replace("_", " "))

    def _stage_marker(self, stage: str, level: int) -> str:
        normalized = str(stage or "").strip().lower()
        if level >= logging.ERROR or any(token in normalized for token in ("failed", "blocked", "timeout", "rejected", "detached")):
            return "!"
        return ">"

    def _format_stage_event_line(self, stage: str, request_id: str = "", level: int = logging.INFO, details: str = "") -> str:
        prefix = self._stage_marker(stage, level)
        header_parts = [f"{prefix} {self._format_stage_label(stage)}"]
        normalized_request_id = str(request_id or "").strip()
        normalized_details = str(details or "").strip()
        if normalized_request_id and normalized_request_id != self._last_stage_request_id:
            header_parts.append(f"request={normalized_request_id}")
            self._last_stage_request_id = normalized_request_id
        if not normalized_details:
            return " | ".join(header_parts)
        detail_lines = [line.strip() for line in normalized_details.splitlines() if line.strip()]
        if not detail_lines:
            return " | ".join(header_parts)
        first_line = " | ".join(header_parts + [detail_lines[0]])
        if len(detail_lines) == 1:
            return first_line
        return first_line + "\n" + "\n".join(f"  {line}" for line in detail_lines[1:])

    def _build_autocomplete_panel_renderable(self) -> Text:
        text = Text()
        text.append("Commands\n", style="bold #c4b5fd")
        for index, item in enumerate(self._autocomplete_suggestions[:6]):
            if index == self._autocomplete_index:
                text.append("  > ", style="bold #22c55e")
                text.append(item, style="bold #f8fafc on #1f2937")
            else:
                text.append("  - ", style="#64748b")
                text.append(item, style="#cbd5e1")
            text.append("\n")
        return text

    def _build_log_line_renderable(self, line: str, response_block_active: bool) -> tuple[Text, bool]:
        raw_line = str(line or "")
        if not raw_line:
            return Text(""), False

        timestamp = now_china().strftime("%H:%M:%S")

        if raw_line == RESPONSE_DIVIDER:
            text = self._build_log_prefix(timestamp)
            text.append(raw_line, style="#475569")
            return text, False

        if raw_line in {"Response:", "Final Response:"}:
            text = self._build_log_prefix(timestamp)
            text.append("Agent", style="bold #34d399")
            text.append(">", style="#6ee7b7")
            return text, True

        if response_block_active and not raw_line.endswith(":"):
            text = self._build_log_prefix(timestamp)
            text.append(raw_line, style="#e2e8f0")
            return text, True

        if raw_line == "Agent>":
            text = self._build_log_prefix(timestamp)
            text.append("Agent", style="bold #34d399")
            text.append("> ", style="#6ee7b7")
            return text, True

        if raw_line.startswith("Agent> "):
            text = self._build_log_prefix(timestamp)
            text.append("Agent", style="bold #34d399")
            text.append("> ", style="#6ee7b7")
            text.append(raw_line[7:], style="#e2e8f0")
            return text, False

        if raw_line.startswith("You> "):
            text = self._build_log_prefix(timestamp)
            text.append("You", style="bold #60a5fa")
            text.append("> ", style="#93c5fd")
            text.append(raw_line[5:], style="#eff6ff")
            return text, False

        if raw_line.startswith("System> "):
            text = self._build_log_prefix(timestamp)
            text.append("System", style="bold #f59e0b")
            text.append("> ", style="#fcd34d")
            text.append(raw_line[8:], style="#fef3c7")
            return text, False

        if raw_line.startswith("Error: "):
            text = self._build_log_prefix(timestamp)
            text.append("Error", style="bold #ef4444")
            text.append(": ", style="#f87171")
            text.append(raw_line[7:], style="#fee2e2")
            return text, False

        if raw_line.startswith("Mapped to command: "):
            text = self._build_log_prefix(timestamp)
            text.append("Mapped", style="bold #a78bfa")
            text.append(" to command: ", style="#c4b5fd")
            text.append(raw_line[len("Mapped to command: "):], style="bold #e9d5ff")
            return text, False

        if raw_line.startswith(("> ", "! ")):
            parts = raw_line.split(" ", 2)
            if len(parts) >= 2:
                prefix, stage = parts[0], parts[1]
                details = parts[2] if len(parts) >= 3 else ""
                text = self._build_log_prefix(timestamp)
                text.append(prefix, style="bold #ef4444" if prefix == "!" else "bold #22c55e")
                text.append(" ")
                text.append(stage, style=self._stage_style(stage))
                if details:
                    text.append(" ")
                    detail_style = "#fecaca" if prefix == "!" else "#cbd5e1"
                    text.append(details, style=detail_style)
                return text, False

        if raw_line.startswith("Session: "):
            text = self._build_log_prefix(timestamp)
            text.append("Session", style="bold #93c5fd")
            text.append(": ", style="#bfdbfe")
            text.append(raw_line[9:], style="#e0f2fe")
            return text, False

        if raw_line == "llm_brain Textual Terminal":
            text = self._build_log_prefix(timestamp)
            text.append(raw_line, style="bold #f8fafc")
            return text, False

        text = self._build_log_prefix(timestamp)
        text.append(raw_line, style="#dbe4ff")
        return text, False

    def _build_log_prefix(self, timestamp: str) -> Text:
        text = Text()
        text.append("[", style="#475569")
        text.append(timestamp, style="bold #94a3b8")
        text.append("] ", style="#475569")
        return text

    def _append_status_segment(self, text: Text, label: str, value: str, value_style: str) -> None:
        text.append(f"{label} ", style="bold #64748b")
        text.append(value, style=value_style)

    def _append_separator(self, text: Text) -> None:
        text.append("  |  ", style="#475569")

    def _status_style(self, status_text: str) -> str:
        normalized = status_text.lower()
        if normalized == "completed":
            return "bold #22c55e"
        if normalized in {"blocked", "failed", "cancelled", "timed_out"}:
            return "bold #ef4444"
        if normalized in {"running", "in_progress"}:
            return "bold #f59e0b"
        return "bold #e2e8f0"

    def _stage_style(self, stage_text: str) -> str:
        normalized = stage_text.lower()
        if "blocked" in normalized or "failed" in normalized:
            return "bold #f87171"
        if normalized.endswith("completed") or normalized.endswith("finished"):
            return "bold #4ade80"
        if normalized.endswith("started") or normalized == "processing":
            return "bold #fbbf24"
        return "bold #cbd5e1"

    def _install_stage_bridge(self) -> None:
        logging_facade = getattr(self.llm_manager_instance, "logging", None)
        if logging_facade is None:
            return
        if getattr(logging_facade, "_textual_cli_bridge_installed", False):
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
            self._threadsafe_emit(
                self._format_stage_event_line(stage, request_id=request_id or "", level=level, details=normalized_details)
            )
            self._refresh_dashboard_from_thread()

        logging_facade._original_console_event = original_console_event
        logging_facade.console_event = bridged_console_event
        logging_facade._textual_cli_bridge_installed = True

    def _refresh_dashboard_from_thread(self) -> None:
        if threading.current_thread() is threading.main_thread():
            self._refresh_dashboard()
            return
        self.call_from_thread(self._refresh_dashboard)


def run_textual_cli(agent_instance=agent, llm_manager_instance=llm_manager) -> None:
    app = AgentTextualApp(agent_instance=agent_instance, llm_manager_instance=llm_manager_instance)
    app.run()