from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Callable

try:
    from rich import box
    from rich.console import Console
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Columns = None
    Panel = None
    Table = None
    box = None
    RICH_AVAILABLE = False


class TerminalUI:
    def __init__(self, output_func: Callable[[str], None], input_func: Callable[[str], str]):
        self.output_func = output_func
        self.input_func = input_func
        self.rich_enabled = bool(RICH_AVAILABLE and output_func is print and input_func is input)
        self.console = Console() if self.rich_enabled else None

    def emit(self, message: str = "") -> None:
        if self.rich_enabled and self.console is not None:
            self.console.print(message, markup=False)
            return
        self.output_func(message)

    def _status_style(self, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"completed", "success", "ok"}:
            return "bold green"
        if normalized in {"failed", "error", "cancelled", "timed_out", "blocked"}:
            return "bold red"
        if normalized in {"in_progress", "active", "running", "detached"}:
            return "bold yellow"
        return "bold cyan"

    def _stage_style(self, stage: str, level: int = logging.INFO) -> tuple[str, str]:
        normalized = str(stage or "").strip().lower()
        if level >= logging.ERROR or "error" in normalized or "failed" in normalized:
            return "red", "x"
        if "cancel" in normalized or "timeout" in normalized or "blocked" in normalized:
            return "yellow", "!"
        if normalized.endswith("started") or normalized.endswith("requested") or normalized.endswith("dispatch"):
            return "cyan", ">"
        if normalized.endswith("completed") or normalized.endswith("finished") or normalized.endswith("resumed"):
            return "green", "*"
        return "magenta", "-"

    def _format_status_value(self, value: Any) -> str:
        text = str(value or "-")
        return f"[{self._status_style(text)}]{text}[/{self._status_style(text)}]"

    def _format_cell(self, column: str, value: Any) -> str:
        column_key = str(column or "").strip().lower()
        if column_key in {"status"}:
            return self._format_status_value(value)
        if column_key in {"stage", "latest stage"}:
            style, icon = self._stage_style(str(value or ""))
            return f"[{style}]{icon} {value}[/{style}]"
        if column_key in {"attention"} and str(value or "-") != "-":
            return f"[yellow]{value}[/yellow]"
        if column_key in {"resumed from"} and str(value or "-") != "-":
            return f"[cyan]{value}[/cyan]"
        return str(value)

    def prompt(self) -> str:
        if self.rich_enabled and self.console is not None:
            return self.console.input("[bold cyan]Agent[/bold cyan] > ")
        return self.input_func("\nAgent> ")

    def render_prompt_bar(self, session_id: str, request_id: str, mcp_count: int, model_label: str) -> None:
        if not self.rich_enabled or self.console is None:
            return
        request_text = request_id or "-"
        bar = (
            f"[dim]session[/dim] [cyan]{session_id}[/cyan]    "
            f"[dim]last_request[/dim] [white]{request_text}[/white]    "
            f"[dim]mcp[/dim] [magenta]{mcp_count}[/magenta]    "
            f"[dim]model[/dim] [green]{model_label}[/green]"
        )
        self.console.print(bar)

    def render_welcome(self, session_id: str) -> None:
        if not self.rich_enabled or self.console is None or Panel is None:
            self.emit("Welcome to the General Agent CLI (LangGraph 1.x based)")
            self.emit(f"Session: {session_id}")
            self.emit("Type 'help' to see special commands or just type your query.")
            return

        hero = Panel.fit(
            f"[bold white]General Agent CLI[/bold white]\n"
            f"[dim]LangGraph-based interactive runtime[/dim]\n\n"
            f"[cyan]Session[/cyan]  {session_id}\n"
            f"[green]Flow[/green]     plan -> act -> tool -> reflect\n"
            f"[yellow]Hint[/yellow]     use [bold]help[/bold] to browse commands",
            title="llm_brain",
            border_style="cyan",
            padding=(1, 2),
        )
        quick_tips = Panel.fit(
            "[bold]/request_summary[/bold] inspect one request\n"
            "[bold]/recent_requests[/bold] scan recent runs\n"
            "[bold]/request_rollup[/bold] inspect failure trends\n"
            "[bold]/list_tool_runs[/bold] watch detached tools",
            title="Quick Start",
            border_style="magenta",
            padding=(1, 2),
        )
        if Columns is not None:
            self.console.print(Columns([hero, quick_tips], expand=False, equal=True))
        else:
            self.console.print(hero)
            self.console.print(quick_tips)

    def _group_command_texts(self, command_texts: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
        groups = {
            "Runtime": [],
            "Requests": [],
            "MCP": [],
            "Memory": [],
            "Session": [],
            "Model": [],
            "Other": [],
        }
        for item in command_texts:
            command_name, description = item.split(" - ", 1)
            lowered = command_name.lower()
            if lowered.startswith("/llm"):
                groups["Model"].append((command_name, description))
            elif lowered.startswith(("/request_", "/recent_", "/failed_", "/resumed_", "/list_tool_runs", "/cancel_request", "/list_snapshots", "/resume_snapshot")):
                groups["Requests"].append((command_name, description))
            elif lowered.startswith(("/load_mcp", "/list_mcp", "/refresh_mcp", "/unload_mcp")):
                groups["MCP"].append((command_name, description))
            elif lowered.startswith(("/replay", "/convert_skill", "/failure_memories")):
                groups["Memory"].append((command_name, description))
            elif lowered.startswith(("/new_session", "/load_tool", "/load_skill")):
                groups["Session"].append((command_name, description))
            elif lowered.startswith(("/retention_status", "/prune_runtime_data")):
                groups["Runtime"].append((command_name, description))
            else:
                groups["Other"].append((command_name, description))
        ordered = []
        for name in ("Model", "Session", "Requests", "MCP", "Memory", "Runtime", "Other"):
            items = groups[name]
            if items:
                ordered.append((name, items))
        return ordered

    def render_help(self, command_texts: list[str]) -> None:
        if not self.rich_enabled or self.console is None or Table is None:
            self.emit("Available commands:")
            for item in command_texts:
                self.emit(f"  {item}")
            self.emit("  <any other text> - Send message to Agent")
            return

        panels = []
        for group_name, items in self._group_command_texts(command_texts):
            table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
            table.add_column("Command", style="bold white", no_wrap=True)
            table.add_column("Description", style="white")
            for command_name, description in items:
                table.add_row(command_name, description)
            panels.append(Panel(table, title=group_name, border_style="cyan"))
        panels.append(Panel.fit("<any other text>\nSend message to Agent", title="Default Input", border_style="green"))
        if Columns is not None:
            self.console.print(Columns(panels, equal=True, expand=True))
        else:
            for panel in panels:
                self.console.print(panel)

    def render_response(self, request_id: str, response: str) -> None:
        if not self.rich_enabled or self.console is None or Panel is None:
            if request_id:
                self.emit(f"Request ID: {request_id}")
            self.emit(f"\nResponse: {response}")
            return

        if request_id:
            self.console.print(f"[dim]Request ID:[/dim] [bold]{request_id}[/bold]")
        self.console.print(Panel.fit(response or "", title="Response", border_style="green"))

    def render_key_value_block(self, title: str, rows: list[tuple[str, Any]]) -> None:
        if not self.rich_enabled or self.console is None or Table is None:
            for key, value in rows:
                self.emit(f"{key}: {value}")
            return

        table = Table(title=title, box=box.SIMPLE, show_header=False)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for key, value in rows:
            table.add_row(str(key), str(value))
        self.console.print(table)

    def render_list_table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        if not self.rich_enabled or self.console is None or Table is None:
            self.emit(title)
            for row in rows:
                self.emit("  - " + " | ".join(str(item) for item in row))
            return

        table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
        for column in columns:
            table.add_column(column)
        for row in rows:
            formatted_row = [self._format_cell(column, item) for column, item in zip(columns, row)]
            table.add_row(*formatted_row)
        self.console.print(table)

    def render_request_summary(
        self,
        summary: dict[str, Any],
        metrics_line: str,
        stage_duration_line: str,
        triage_line: str,
    ) -> None:
        if not self.rich_enabled or self.console is None or Table is None or Panel is None:
            return

        self.render_key_value_block(
            "Request Summary",
            [
                ("Request ID", summary.get("request_id") or "-"),
                ("Status", self._format_status_value(summary.get("status") or "-")),
                ("Session ID", summary.get("session_id") or "-"),
                ("Latest stage", self._format_cell("stage", summary.get("latest_stage") or "-")),
                ("Progress", f"{summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)}"),
                ("Snapshots", summary.get("snapshot_count", 0)),
                ("Memories", summary.get("memory_count", 0)),
                ("Checkpoints", summary.get("checkpoint_count", 0)),
            ],
        )
        if metrics_line:
            self.console.print(Panel.fit(metrics_line, title="Metrics", border_style="blue"))
        if stage_duration_line:
            self.console.print(Panel.fit(stage_duration_line, title="Stage Durations", border_style="magenta"))
        if triage_line:
            self.console.print(Panel.fit(triage_line, title="Triage", border_style="yellow"))

        if summary.get("final_response"):
            self.console.print(Panel(summary["final_response"], title="Final Response", border_style="green"))

        checkpoints = summary.get("checkpoints") or []
        if checkpoints:
            checkpoint_rows = []
            for item in checkpoints[-5:]:
                checkpoint_rows.append([
                    item.get("logged_at") or "-",
                    item.get("stage") or "-",
                    item.get("details") or "-",
                ])
            self.render_list_table("Recent Checkpoints", ["Logged at", "Stage", "Details"], checkpoint_rows)

        memories = summary.get("memories") or []
        if memories:
            memory_rows = []
            for item in memories[-5:]:
                memory_rows.append([
                    f"#{item.get('id')}",
                    item.get("memory_type") or "-",
                    ",".join(item.get("quality_tags", [])) or "-",
                    item.get("summary") or "-",
                ])
            self.render_list_table("Related Memories", ["ID", "Type", "Tags", "Summary"], memory_rows)

        detached_tools = summary.get("detached_tools") or []
        if detached_tools:
            tool_rows = []
            for item in detached_tools[-5:]:
                runtime_value = "-"
                if item.get("detached_runtime_ms") is not None:
                    runtime_value = f"detached_ms={item['detached_runtime_ms']}"
                elif item.get("runtime_ms") is not None:
                    runtime_value = f"runtime_ms={item['runtime_ms']}"
                tool_rows.append([
                    item.get("tool_name") or "-",
                    item.get("tool_run_id") or "-",
                    item.get("reason") or "-",
                    item.get("source") or "-",
                    runtime_value,
                ])
            self.render_list_table("Detached Tools", ["Tool", "Run ID", "Reason", "Source", "Runtime"], tool_rows)

    def render_recent_requests(self, heading: str, rows: list[list[Any]]) -> None:
        if not self.rich_enabled:
            return
        self.render_list_table(
            heading,
            ["Request", "Status", "Stage", "Session", "Updated", "Total ms", "Detached", "Resumed from", "Attention"],
            rows,
        )

    def render_request_rollup(
        self,
        overview_rows: list[tuple[str, Any]],
        filters_line: str,
        status_line: str,
        totals_line: str,
        top_failures_line: str,
        combos_line: str,
        source_buckets_line: str,
        source_trends_line: str,
        stage_totals_line: str,
        latest_updated_at: str,
    ) -> None:
        if not self.rich_enabled or self.console is None or Panel is None:
            return
        self.render_key_value_block("Request Rollup", overview_rows)
        for title, line, style in (
            ("Filters", filters_line, "cyan"),
            ("Statuses", status_line, "blue"),
            ("Totals", totals_line, "green"),
            ("Top Failures", top_failures_line, "yellow"),
            ("Failure Combos", combos_line, "magenta"),
            ("Source Buckets", source_buckets_line, "cyan"),
            ("Source Trends", source_trends_line, "magenta"),
            ("Stage Totals", stage_totals_line, "blue"),
        ):
            if line:
                self.console.print(Panel.fit(line, title=title, border_style=style))
        if latest_updated_at:
            self.console.print(f"[dim]Latest updated_at:[/dim] {latest_updated_at}")

    def render_mcp_servers(self, servers: list[dict[str, Any]]) -> None:
        if not self.rich_enabled:
            return
        if not servers:
            self.emit("No MCP servers loaded.")
            return
        rows = []
        for item in servers:
            rows.append([
                item.get("name") or "-",
                item.get("transport") or "-",
                len(item.get("tool_names", [])),
                item.get("source") or "-",
            ])
        self.render_list_table("Loaded MCP Servers", ["Name", "Transport", "Tools", "Source"], rows)

    def render_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        if not self.rich_enabled:
            return
        if not snapshots:
            self.emit("No snapshots found.")
            return
        rows = []
        for item in snapshots:
            rows.append([
                item.get("index"),
                item.get("file") or "-",
                item.get("stage") or "-",
                item.get("subtask_index"),
                item.get("blocked"),
                item.get("completed"),
            ])
        self.render_list_table("Available Snapshots", ["Index", "File", "Stage", "Subtask", "Blocked", "Completed"], rows)

    def render_tool_runs(self, items: list[dict[str, Any]]) -> None:
        if not self.rich_enabled:
            return
        rows = []
        for item in items:
            runtime_value = "-"
            if item.get("detached_runtime_ms") is not None:
                runtime_value = f"detached_ms={item['detached_runtime_ms']}"
            elif item.get("runtime_ms") is not None:
                runtime_value = f"runtime_ms={item['runtime_ms']}"
            rows.append([
                item.get("tool_run_id") or "-",
                item.get("tool_name") or "-",
                item.get("request_id") or "-",
                item.get("status") or "-",
                item.get("reason") or "-",
                runtime_value,
            ])
        self.render_list_table("Tracked Tool Runs", ["Run ID", "Tool", "Request", "Status", "Reason", "Runtime"], rows)

    def render_retention_targets(self, rows: list[list[Any]]) -> None:
        if not self.rich_enabled:
            return
        self.render_list_table(
            "Retention Targets",
            ["Target", "Days", "Max", "Max bytes", "Items", "Expired", "Total", "Reclaimable"],
            rows,
        )

    def render_failure_memories(self, rows: list[list[Any]], keywords: list[str]) -> None:
        if not self.rich_enabled:
            return
        title = "Failure Memories"
        if keywords:
            title += f" ({','.join(keywords)})"
        self.render_list_table(title, ["ID", "Request", "Type", "Tags", "Keywords", "Summary"], rows)

    def render_stage_event(self, stage: str, request_id: str = "", level: int = logging.INFO) -> None:
        if not self.rich_enabled or self.console is None:
            return
        color, icon = self._stage_style(stage, level=level)
        request_suffix = f" [dim]{request_id}[/dim]" if request_id else ""
        self.console.print(f"[{color}]{icon}[/{color}] [{color}]{stage}[/{color}]{request_suffix}")

    @contextmanager
    def busy(self, status_text: str):
        if not self.rich_enabled or self.console is None:
            yield
            return
        with self.console.status(f"[bold cyan]{status_text}[/bold cyan]", spinner="dots"):
            yield