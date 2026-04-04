from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Callable


class TerminalUI:
    def __init__(self, output_func: Callable[[str], None], input_func: Callable[[str], str]):
        self.output_func = output_func
        self.input_func = input_func
        self.structured_output_enabled = False

    def emit(self, message: str = "") -> None:
        self.output_func(str(message or ""))

    def _emit_plain_section(self, title: str, lines: list[str], leading_blank_line: bool = False) -> None:
        section_lines = [str(item) for item in lines if str(item or "").strip()]
        if not section_lines:
            return
        if leading_blank_line:
            self.output_func("")
        self.output_func(f"{title}:")
        for item in section_lines:
            self.output_func(f"  {item}")

    def prompt(self) -> str:
        return self.input_func("\nAgent> ")

    def render_prompt_bar(
        self,
        session_id: str,
        request_id: str,
        mcp_count: int,
        model_label: str,
        selection_label: str = "",
        overview: dict[str, Any] | None = None,
    ) -> None:
        return

    def render_welcome(self, session_id: str) -> None:
        self.emit("Welcome to the General Agent CLI (LangGraph 1.x based)")
        self.emit(f"Session: {session_id}")
        self.emit("Type 'help' to see special commands or just type your query.")

    def render_help(self, command_texts: list[str]) -> None:
        self.emit("Available commands:")
        for item in command_texts:
            self.emit(f"  {item}")
        self.emit("  <any other text> - Send message to Agent")

    def render_response(self, request_id: str, response: str, summary: dict[str, Any] | None = None) -> None:
        if request_id:
            self.emit(f"Request ID: {request_id}")
        if summary:
            metrics = summary.get("metrics", {}) or {}
            self._emit_plain_section(
                "Summary",
                [
                    f"request={summary.get('request_id') or '-'} | "
                    f"stage={summary.get('latest_stage') or '-'} | "
                    f"mode={'lite_chat' if summary.get('lite_mode') else 'task'} | "
                    f"progress={summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)} | "
                    f"tool_calls={metrics.get('tool_call_count', 0)} | "
                    f"total_ms={metrics.get('total_duration_ms', '-')}"
                ],
            )
            raw_query = str(summary.get("raw_query") or "").strip()
            normalized_query = str(summary.get("normalized_query") or "").strip()
            mode_text = "lite_chat" if summary.get("lite_mode") else "task"
            status_text = str(summary.get("status") or "-")
            context_lines = []
            if raw_query:
                context_lines.append(f"Input: {raw_query}")
            if normalized_query and normalized_query != raw_query:
                context_lines.append(f"Intent: {normalized_query}")
            context_lines.append(f"Mode: {mode_text} | Status: {status_text}")
            self._emit_plain_section("Conversation Context", context_lines, leading_blank_line=True)
            blocked_reason = str(summary.get("blocked_reason") or "").strip()
            if blocked_reason:
                self._emit_plain_section("Risk", [f"Blocked Reason: {blocked_reason}"], leading_blank_line=True)
            suggestions = summary.get("suggested_actions") or []
            if suggestions:
                self._emit_plain_section("Next Actions", ["Next: " + " | ".join(str(item) for item in suggestions)], leading_blank_line=True)
        self._emit_plain_section("Response", [response], leading_blank_line=True)

    def render_key_value_block(self, title: str, rows: list[tuple[str, Any]]) -> None:
        lines = [f"{key}: {value}" for key, value in rows]
        self._emit_plain_section(title, lines, leading_blank_line=True)

    def render_list_table(self, title: str, columns: list[str], rows: list[list[Any]]) -> None:
        if not rows:
            return
        formatted_rows = []
        for row in rows:
            parts = [f"{column}={value}" for column, value in zip(columns, row)]
            formatted_rows.append(" | ".join(parts))
        self._emit_plain_section(title, formatted_rows, leading_blank_line=True)

    def render_request_summary(
        self,
        summary: dict[str, Any],
        metrics_line: str,
        stage_duration_line: str,
        triage_line: str,
    ) -> None:
        request_id = str(summary.get("request_id") or "")
        if request_id:
            self.emit(f"Request ID: {request_id}")
        self._emit_plain_section(
            "Summary",
            [
                f"request={summary.get('request_id') or '-'} | "
                f"stage={summary.get('latest_stage') or '-'} | "
                f"mode={'lite_chat' if summary.get('lite_mode') else 'task'} | "
                f"progress={summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)} | "
                f"tool_calls={(summary.get('metrics', {}) or {}).get('tool_call_count', 0)} | "
                f"total_ms={(summary.get('metrics', {}) or {}).get('total_duration_ms', '-')}"
            ],
        )
        context_lines = [
            f"Mode: {'lite_chat' if summary.get('lite_mode') else 'task'}",
            f"Status: {summary.get('status') or '-'}",
            f"Session ID: {summary.get('session_id') or '-'}",
            f"Latest stage: {summary.get('latest_stage') or '-'}",
            f"Progress: {summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)}",
        ]
        raw_query = str(summary.get("raw_query") or "").strip()
        normalized_query = str(summary.get("normalized_query") or "").strip()
        if raw_query:
            context_lines.insert(0, f"Input: {raw_query}")
        if normalized_query and normalized_query != raw_query:
            context_lines.insert(1, f"Intent: {normalized_query}")
        self._emit_plain_section("Conversation Context", context_lines, leading_blank_line=True)
        if metrics_line:
            self._emit_plain_section("Metrics", [metrics_line], leading_blank_line=True)
        if stage_duration_line:
            self._emit_plain_section("Stage Durations", [stage_duration_line], leading_blank_line=True)
        if triage_line:
            self._emit_plain_section("Triage", [triage_line], leading_blank_line=True)

    def render_recent_requests(self, heading: str, rows: list[list[Any]]) -> None:
        if not rows:
            self.emit("No matching requests found.")
            return
        self.emit(heading)
        for row in rows:
            self.emit("  - " + " | ".join(str(item) for item in row))

    def render_conversation_history(self, heading: str, rows: list[list[Any]]) -> None:
        if not rows:
            self.emit("No matching requests found.")
            return
        self.emit(heading)
        for row in rows:
            self.emit("  - " + " | ".join(str(item) for item in row))

    def render_tool_feedback(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._emit_plain_section(
            "Tool Feedback",
            [
                f"- [{item.get('status') or '-'}] {item.get('tool') or '-'} | summary={item.get('summary') or '-'} | source={item.get('source') or '-'} | run_id={item.get('run_id') or '-'}"
                for item in rows
            ],
            leading_blank_line=True,
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
        overview_line = "Request rollup: " + " | ".join(f"{key.lower().replace(' ', '_')}={value}" for key, value in overview_rows)
        self.emit(overview_line)
        for line in (filters_line, status_line, totals_line, top_failures_line, combos_line, source_buckets_line, source_trends_line, stage_totals_line):
            if line:
                self.emit(line)
        if latest_updated_at:
            self.emit(f"Latest updated_at: {latest_updated_at}")

    def render_mcp_servers(self, servers: list[dict[str, Any]]) -> None:
        if not servers:
            self.emit("No MCP servers loaded.")
            return
        self.emit("Loaded MCP servers:")
        for item in servers:
            self.emit(
                f"  - {item.get('name') or '-'} | transport={item.get('transport') or '-'} | tools={len(item.get('tool_names', []))} | source={item.get('source') or '-'}"
            )

    def render_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        if not snapshots:
            self.emit("No snapshots found.")
            return
        self.emit("Available snapshots:")
        for item in snapshots:
            self.emit(
                f"  - [{item.get('index')}] {item.get('file') or '-'} | stage={item.get('stage') or '-'} | subtask={item.get('subtask_index')} | blocked={item.get('blocked')} | completed={item.get('completed')}"
            )

    def render_tool_runs(self, items: list[dict[str, Any]]) -> None:
        self.emit("Tracked tool runs:")
        for item in items:
            runtime_value = "-"
            if item.get("detached_runtime_ms") is not None:
                runtime_value = f"detached_ms={item['detached_runtime_ms']}"
            elif item.get("runtime_ms") is not None:
                runtime_value = f"runtime_ms={item['runtime_ms']}"
            self.emit(
                f"  - {item.get('tool_run_id') or '-'} | tool={item.get('tool_name') or '-'} | request={item.get('request_id') or '-'} | status={item.get('status') or '-'} | reason={item.get('reason') or '-'} | {runtime_value}"
            )

    def render_retention_targets(self, rows: list[list[Any]]) -> None:
        for row in rows:
            self.emit(
                f"  - {row[0]} | days={row[1]} | max={row[2]} | max_bytes={row[3]} | items={row[4]} | expired={row[5]} | total={row[6]} | reclaimable={row[7]}"
            )

    def render_failure_memories(self, rows: list[list[Any]], keywords: list[str]) -> None:
        if keywords:
            self.emit("Failure memories: keywords=" + ",".join(keywords))
        else:
            self.emit("Failure memories:")
        for row in rows:
            self.emit("  - " + " | ".join(str(item) for item in row))

    def render_stage_event(self, stage: str, request_id: str = "", level: int = logging.INFO) -> None:
        return

    @contextmanager
    def busy(self, status_text: str):
        yield