from dataclasses import dataclass
import re
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


def emit_agent_response(context: dict[str, Any], response: str, request_id: str) -> None:
    ui = context.get("ui")
    if ui:
        ui.render_response(request_id, response)
        return
    emit_response(context["output_func"], response, request_id)


def build_attention_detail_summary(triage: dict[str, Any]) -> str:
    details = str(triage.get("latest_failure_details", "") or "")
    extracted: list[str] = []
    if details:
        for key in ("tool", "reason", "error_type", "action"):
            match = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+)", details)
            if match:
                extracted.append(f"{key}={match.group(1).strip()}")
    reroute_mode = str(triage.get("latest_reroute_mode", "") or "")
    if reroute_mode:
        extracted.append(f"reroute={reroute_mode}")
    failure_source = str(triage.get("latest_failure_source", "") or "")
    if failure_source:
        extracted.append(f"source={failure_source}")
    return ",".join(extracted)


def parse_since_window_seconds(value: str) -> int | None:
    match = re.fullmatch(r"(\d+)([smhd])", str(value or "").strip().lower())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    unit_scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return amount * unit_scale[unit]


def format_since_window(seconds: int | None) -> str:
    if not isinstance(seconds, int) or seconds <= 0:
        return ""
    for unit, scale in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds % scale == 0:
            return f"{seconds // scale}{unit}"
    return f"{seconds}s"


def handle_help(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    command_texts = [context["commands"][item].help_text for item in context["command_order"] if item.startswith("/")]
    if ui:
        ui.render_help(command_texts)
        return True
    output_func("Available commands:")
    for item in command_texts:
        output_func(f"  {item}")
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
        output_func("Usage: /load_skill <skill_name.md>")
    return True


def handle_load_tool(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) == 2:
        output_func(context["agent_instance"].load_tool(parts[1]))
    else:
        output_func("Usage: /load_tool <tool_name.py>")
    return True


def handle_replay(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    agent_instance = context["agent_instance"]
    parts = user_input.split()
    if len(parts) >= 2:
        response = agent_instance.replay(int(parts[1]), parts[2:])
        emit_agent_response(context, response, agent_instance.get_last_request_id())
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
    ui = context.get("ui")
    servers = context["agent_instance"].list_mcp_servers()
    if ui and ui.rich_enabled:
        ui.render_mcp_servers(servers)
        return True
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
        snapshot_name = None
        reroute = False
        for token in parts[2:]:
            normalized = token.strip().lower()
            if normalized in {"reroute", "mode=reroute", "--reroute"}:
                reroute = True
                continue
            if snapshot_name is None:
                snapshot_name = token
                continue
            output_func("Usage: /resume_snapshot <request_id> [latest|index|stage|snapshot_file] [reroute]")
            return True

        response = agent_instance.resume_from_snapshot(parts[1], snapshot_name=snapshot_name, reroute=reroute)
        emit_agent_response(context, response, agent_instance.get_last_request_id())
    else:
        output_func("Usage: /resume_snapshot <request_id> [latest|index|stage|snapshot_file] [reroute]")
    return True


def handle_list_snapshots(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    if len(parts) >= 2:
        snapshots = context["agent_instance"].list_snapshots(parts[1])
        if ui and ui.rich_enabled:
            ui.render_snapshots(snapshots)
            return True
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


def build_request_summary_lines(summary: dict[str, Any], observability) -> tuple[str, str, str]:
    triage = summary.get("triage", {}) or {}
    metrics = summary.get("metrics", {}) or {}
    metrics_line = ""
    if metrics:
        metrics_line = (
            f"total_ms={metrics.get('total_duration_ms', '-')} | "
            f"llm_calls={metrics.get('llm_call_count', 0)} | "
            f"tool_calls={metrics.get('tool_call_count', 0)} | "
            f"tool_detached={metrics.get('tool_detached_count', 0)} | "
            f"retries={metrics.get('retry_count', 0)} | "
            f"reflection_failures={metrics.get('reflection_failure_count', 0)}"
        )
    stage_duration_line = ""
    stage_duration_ms = metrics.get("stage_duration_ms", {}) or {}
    if stage_duration_ms:
        duration_parts = [f"{key}={value}" for key, value in observability.ordered_stage_duration_items(stage_duration_ms)]
        stage_duration_line = " | ".join(duration_parts)
    triage_line = ""
    if triage:
        triage_line = (
            f"needs_attention={triage.get('needs_attention', False)} | "
            f"resumed={triage.get('is_resumed', False)} | "
            f"failure_stage={triage.get('latest_failure_stage') or '-'} | "
            f"failure_source={triage.get('latest_failure_source') or '-'} | "
            f"reroute={triage.get('latest_reroute_mode') or '-'} | "
            f"tool_attention={triage.get('tool_attention_count', 0)} | "
            f"detached_tools={triage.get('detached_tool_count', 0)}"
        )
    return metrics_line, stage_duration_line, triage_line


def handle_request_summary(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    if len(parts) < 2:
        output_func("Usage: /request_summary <request_id>")
        return True

    summary = context["agent_instance"].get_request_summary(parts[1])
    if not summary:
        output_func("Request not found.")
        return True

    triage = summary.get("triage", {}) or {}
    metrics_line, stage_duration_line, triage_line = build_request_summary_lines(summary, context["agent_instance"].observability)

    if ui and ui.rich_enabled:
        ui.render_request_summary(summary, metrics_line, stage_duration_line, triage_line)
        if triage.get("last_error_message"):
            ui.emit(f"Last error: {triage['last_error_message']}")
        if triage.get("latest_failure_details"):
            ui.emit(f"Failure details: {triage['latest_failure_details']}")
        if triage.get("latest_reroute_details"):
            ui.emit(f"Reroute details: {triage['latest_reroute_details']}")
        if summary.get("source_request_id"):
            ui.emit(f"Source request: {summary['source_request_id']}")
        return True

    output_func(f"Request ID: {summary['request_id']}")
    output_func(f"Status: {summary['status']}")
    output_func(f"Session ID: {summary['session_id'] or '-'}")
    output_func(f"Latest stage: {summary['latest_stage'] or '-'}")
    output_func(f"Progress: {summary['subtask_index']}/{summary['plan_length']}")
    output_func(f"Snapshots: {summary['snapshot_count']}")
    output_func(f"Memories: {summary['memory_count']}")
    output_func(f"Checkpoints: {summary['checkpoint_count']}")
    if metrics_line:
        output_func("Metrics: " + metrics_line)
    if stage_duration_line:
        output_func("Stage durations: " + stage_duration_line)
    if triage_line:
        output_func("Triage: " + triage_line)
    if triage.get("last_error_message"):
        output_func(f"Last error: {triage['last_error_message']}")
    if triage.get("latest_failure_details"):
        output_func(f"Failure details: {triage['latest_failure_details']}")
    if triage.get("latest_reroute_details"):
        output_func(f"Reroute details: {triage['latest_reroute_details']}")
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
    if summary.get("detached_tools"):
        output_func("\nDetached tools:")
        for item in summary["detached_tools"][-5:]:
            runtime_suffix = ""
            if item.get("detached_runtime_ms") is not None:
                runtime_suffix = f" | detached_ms={item['detached_runtime_ms']}"
            elif item.get("runtime_ms") is not None:
                runtime_suffix = f" | runtime_ms={item['runtime_ms']}"
            output_func(
                f"  - {item.get('tool_name') or '-'} | run_id={item.get('tool_run_id') or '-'} | "
                f"reason={item.get('reason') or '-'} | source={item.get('source') or '-'}{runtime_suffix}"
            )
    return True


def render_recent_request_rows(output_func, summaries: list[dict[str, Any]], heading: str) -> None:
    if not summaries:
        output_func("No matching requests found.")
        return

    output_func(heading)
    for item in summaries:
        metrics = item.get("metrics", {})
        triage = item.get("triage", {}) or {}
        resumed_suffix = f" | resumed_from={item.get('source_request_id')}" if item.get("source_request_id") else ""
        attention_suffix = ""
        if triage.get("needs_attention"):
            attention_parts = []
            if triage.get("latest_failure_stage"):
                attention_parts.append(f"stage={triage['latest_failure_stage']}")
            if triage.get("detached_tool_count"):
                attention_parts.append(f"detached={triage['detached_tool_count']}")
            elif triage.get("tool_attention_count"):
                attention_parts.append(f"tool_attention={triage['tool_attention_count']}")
            detail_summary = build_attention_detail_summary(triage)
            if detail_summary:
                attention_parts.append(detail_summary)
            if attention_parts:
                attention_suffix = " | attention=" + ",".join(attention_parts)
        output_func(
            f"  - {item['request_id']} | status={item['status']} | session={item.get('session_id') or '-'} | "
            f"stage={item.get('latest_stage') or '-'} | updated_at={item.get('updated_at') or '-'} | "
            f"total_ms={metrics.get('total_duration_ms', '-')} | llm_calls={metrics.get('llm_call_count', 0)} | "
            f"tool_calls={metrics.get('tool_call_count', 0)} | detached_tools={metrics.get('tool_detached_count', 0)}{resumed_suffix}{attention_suffix}"
        )


def build_recent_request_row(item: dict[str, Any]) -> list[Any]:
    metrics = item.get("metrics", {})
    triage = item.get("triage", {}) or {}
    return [
        item["request_id"],
        item.get("status") or "-",
        item.get("latest_stage") or "-",
        item.get("session_id") or "-",
        item.get("updated_at") or "-",
        metrics.get("total_duration_ms", "-"),
        metrics.get("tool_detached_count", 0),
        item.get("source_request_id") or "-",
        build_attention_detail_summary(triage) or "-",
    ]


def parse_request_filter_args(parts: list[str], usage_text: str) -> tuple[int, list[str], bool, bool, int | None] | None:
    limit = 10
    statuses: list[str] = []
    resumed_only = False
    attention_only = False
    since_seconds = None
    next_index = 1

    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
            next_index = 2
        except ValueError:
            next_index = 1

    valid_statuses = {"active", "blocked", "cancelled", "completed", "failed", "in_progress", "not_found", "timed_out"}
    for token in parts[next_index:]:
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized == "resumed":
            resumed_only = True
            continue
        if normalized == "attention":
            attention_only = True
            continue
        if normalized.startswith("since="):
            parsed_seconds = parse_since_window_seconds(normalized.split("=", 1)[1])
            if parsed_seconds is None:
                return None
            since_seconds = parsed_seconds
            continue
        if normalized.startswith("status="):
            raw_statuses = normalized.split("=", 1)[1]
            requested_statuses = [item.strip() for item in raw_statuses.split(",") if item.strip()]
            if not requested_statuses or any(item not in valid_statuses for item in requested_statuses):
                return None
            statuses = requested_statuses
            continue
        return None

    return limit, statuses, resumed_only, attention_only, since_seconds


def handle_recent_request_query(
    context: dict[str, Any],
    parts: list[str],
    usage_text: str,
    heading: str,
    default_statuses: list[str] | None = None,
    default_resumed_only: bool = False,
) -> bool:
    output_func = context["output_func"]
    agent_instance = context["agent_instance"]
    ui = context.get("ui")
    parsed = parse_request_filter_args(parts, usage_text)
    if parsed is None:
        output_func(f"Usage: {usage_text}")
        return True

    limit, statuses, resumed_only, attention_only, since_seconds = parsed
    merged_statuses = statuses or list(default_statuses or [])
    merged_resumed_only = resumed_only or default_resumed_only
    summaries = agent_instance.get_recent_request_summaries(
        limit=limit,
        statuses=merged_statuses or None,
        resumed_only=merged_resumed_only,
        attention_only=attention_only,
        since_seconds=since_seconds,
    )
    filter_parts = []
    if merged_statuses:
        filter_parts.append("statuses=" + ",".join(merged_statuses))
    if merged_resumed_only:
        filter_parts.append("resumed_only=True")
    if attention_only:
        filter_parts.append("attention_only=True")
    if since_seconds:
        filter_parts.append("since=" + format_since_window(since_seconds))
    if filter_parts:
        output_func("Filters: " + " | ".join(filter_parts))
    if ui and ui.rich_enabled:
        if not summaries:
            ui.emit("No matching requests found.")
        else:
            ui.render_recent_requests(heading, [build_recent_request_row(item) for item in summaries])
        return True
    render_recent_request_rows(output_func, summaries, heading)
    return True


def handle_recent_requests(user_input: str, context: dict[str, Any]) -> bool:
    parts = user_input.split()
    usage_text = "/recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]"
    return handle_recent_request_query(
        context,
        parts,
        usage_text,
        "Recent requests:",
    )


def handle_request_rollup(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    usage_text = "/request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]"
    parsed = parse_request_filter_args(parts, usage_text)
    if parsed is None:
        output_func(f"Usage: {usage_text}")
        return True
    limit, statuses, resumed_only, attention_only, since_seconds = parsed
    limit = max(limit, 1)

    rollup = context["agent_instance"].get_request_rollup(
        limit=limit,
        statuses=statuses or None,
        resumed_only=resumed_only,
        attention_only=attention_only,
        since_seconds=since_seconds,
    )
    if not rollup.get("request_count"):
        output_func("No requests found for rollup.")
        return True

    status_counts = rollup.get("status_counts", {}) or {}
    status_parts = [f"{key}={value}" for key, value in status_counts.items()]
    overview_line = (
        "Request rollup: "
        f"count={rollup.get('request_count', 0)} | "
        f"attention={rollup.get('needs_attention_count', 0)} | "
        f"resumed={rollup.get('resumed_count', 0)} | "
        f"active={rollup.get('active_count', 0)} | "
        f"avg_total_ms={rollup.get('average_total_duration_ms', '-')}"
    )
    filters = rollup.get("filters", {}) or {}
    filter_parts = []
    if filters.get("statuses"):
        filter_parts.append("statuses=" + ",".join(filters["statuses"]))
    if filters.get("resumed_only"):
        filter_parts.append("resumed_only=True")
    if filters.get("attention_only"):
        filter_parts.append("attention_only=True")
    if filters.get("since_seconds"):
        filter_parts.append("since=" + format_since_window(filters["since_seconds"]))
    filters_line = "Filters: " + " | ".join(filter_parts) if filter_parts else ""
    status_line = "Statuses: " + " | ".join(status_parts) if status_parts else ""
    totals = rollup.get("totals", {}) or {}
    totals_line = (
        "Totals: "
        f"llm_calls={totals.get('llm_call_count', 0)} | "
        f"tool_calls={totals.get('tool_call_count', 0)} | "
        f"tool_detached={totals.get('tool_detached_count', 0)} | "
        f"retries={totals.get('retry_count', 0)} | "
        f"reflection_failures={totals.get('reflection_failure_count', 0)}"
    )
    top_failure_signals = rollup.get("top_failure_signals", {}) or {}
    signal_parts = []
    for label, key in (("stage", "stages"), ("tool", "tools"), ("reason", "reasons"), ("error_type", "error_types"), ("action", "actions"), ("source", "sources"), ("reroute", "reroute_modes"), ("no_tool_fallback", "no_tool_fallbacks")):
        items = top_failure_signals.get(key, []) or []
        if items:
            value, count = items[0]
            signal_parts.append(f"{label}={value}({count})")
    top_failures_line = "Top failures: " + " | ".join(signal_parts) if signal_parts else ""
    top_failure_combinations = rollup.get("top_failure_combinations", {}) or {}
    combination_parts = []
    for label, key in (("stage+tool", "stage_tool"), ("tool+reason", "tool_reason"), ("stage+source", "stage_source"), ("stage+reroute", "stage_reroute")):
        items = top_failure_combinations.get(key, []) or []
        if items:
            value, count = items[0]
            combination_parts.append(f"{label}={value}({count})")
    combos_line = "Failure combos: " + " | ".join(combination_parts) if combination_parts else ""
    source_bucket_breakdown = rollup.get("source_bucket_breakdown", []) or []
    source_parts = []
    for item in source_bucket_breakdown:
        source = item.get("source", "")
        count = item.get("count", 0)
        share = item.get("share")
        if not source:
            continue
        if isinstance(share, (int, float)):
            source_parts.append(f"{source}={count}({share * 100:.1f}%)")
        else:
            source_parts.append(f"{source}={count}")
    source_buckets_line = "Source buckets: " + " | ".join(source_parts) if source_parts else ""
    source_bucket_trends = rollup.get("source_bucket_trends", []) or []
    trend_parts = []
    for item in source_bucket_trends:
        source = item.get("source", "")
        direction = item.get("direction", "flat")
        delta_share = item.get("delta_share")
        recent_count = item.get("recent_count", 0)
        earlier_count = item.get("earlier_count", 0)
        if not source:
            continue
        if isinstance(delta_share, (int, float)):
            trend_parts.append(f"{source}={direction}({delta_share * 100:+.1f}pp,recent={recent_count},earlier={earlier_count})")
        else:
            trend_parts.append(f"{source}={direction}(recent={recent_count},earlier={earlier_count})")
    source_trends_line = "Source trends: " + " | ".join(trend_parts) if trend_parts else ""
    stage_duration_ms_total = rollup.get("stage_duration_ms_total", {}) or {}
    stage_totals_line = ""
    if stage_duration_ms_total:
        duration_parts = [
            f"{key}={value}"
            for key, value in context["agent_instance"].observability.ordered_stage_duration_items(stage_duration_ms_total)
        ]
        stage_totals_line = "Stage totals: " + " | ".join(duration_parts)

    if ui and ui.rich_enabled:
        ui.render_request_rollup(
            [
                ("Count", rollup.get("request_count", 0)),
                ("Attention", rollup.get("needs_attention_count", 0)),
                ("Resumed", rollup.get("resumed_count", 0)),
                ("Active", rollup.get("active_count", 0)),
                ("Avg total ms", rollup.get("average_total_duration_ms", "-")),
            ],
            filters_line,
            status_line,
            totals_line,
            top_failures_line,
            combos_line,
            source_buckets_line,
            source_trends_line,
            stage_totals_line,
            rollup.get("latest_updated_at", ""),
        )
        return True

    output_func(overview_line)
    if filters_line:
        output_func(filters_line)
    if status_line:
        output_func(status_line)
    output_func(totals_line)
    if top_failures_line:
        output_func(top_failures_line)
    if combos_line:
        output_func(combos_line)
    if source_buckets_line:
        output_func(source_buckets_line)
    if source_trends_line:
        output_func(source_trends_line)
    if stage_totals_line:
        output_func(stage_totals_line)
    if rollup.get("latest_updated_at"):
        output_func(f"Latest updated_at: {rollup['latest_updated_at']}")
    return True


def handle_failed_requests(user_input: str, context: dict[str, Any]) -> bool:
    parts = user_input.split()
    usage_text = "/failed_requests [limit] [status=failed,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m]"
    return handle_recent_request_query(
        context,
        parts,
        usage_text,
        "Recent failed or blocked requests:",
        default_statuses=["failed", "blocked", "timed_out", "cancelled"],
    )


def handle_resumed_requests(user_input: str, context: dict[str, Any]) -> bool:
    parts = user_input.split()
    usage_text = "/resumed_requests [limit] [status=failed,blocked,...] [attention] [since=30m]"
    return handle_recent_request_query(
        context,
        parts,
        usage_text,
        "Recent resumed requests:",
        default_resumed_only=True,
    )


def handle_list_tool_runs(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    request_id = parts[1] if len(parts) >= 2 else ""
    status = parts[2] if len(parts) >= 3 else ""
    if status and status not in {"running", "detached"}:
        output_func("Usage: /list_tool_runs [request_id] [running|detached]")
        return True

    items = context["agent_instance"].list_tool_runs(request_id=request_id, status=status)
    if not items:
        output_func("No tracked tool runs found.")
        return True

    if ui and ui.rich_enabled:
        ui.render_tool_runs(items)
        return True

    output_func("Tracked tool runs:")
    for item in items:
        runtime_suffix = ""
        if item.get("detached_runtime_ms") is not None:
            runtime_suffix = f" | detached_ms={item['detached_runtime_ms']}"
        elif item.get("runtime_ms") is not None:
            runtime_suffix = f" | runtime_ms={item['runtime_ms']}"
        output_func(
            f"  - {item.get('tool_run_id') or '-'} | tool={item.get('tool_name') or '-'} | "
            f"request={item.get('request_id') or '-'} | status={item.get('status') or '-'} | "
            f"reason={item.get('reason') or '-'}{runtime_suffix}"
        )
    return True


def _format_bytes(value: Any) -> str:
    try:
        size = int(value or 0)
    except (TypeError, ValueError):
        return "0 B"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{round(size / 1024, 2)} KB"
    return f"{round(size / (1024 * 1024), 2)} MB"


def handle_retention_status(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    payload = context["agent_instance"].get_retention_status()
    totals = payload.get("totals", {}) or {}
    last_auto_prune = payload.get("last_auto_prune", {}) or {}
    last_auto_prune_check = payload.get("last_auto_prune_check", {}) or {}
    target_rows = []
    for item in payload.get("targets", []) or []:
        target_rows.append(
            [
                item.get("key"),
                item.get("retention_days"),
                item.get("max_items", 0),
                _format_bytes(item.get("max_total_bytes", 0)),
                item.get("item_count", 0),
                item.get("expired_count", 0),
                _format_bytes(item.get("total_bytes", 0)),
                _format_bytes(item.get("reclaimable_bytes", 0)),
            ]
        )
    if ui and ui.rich_enabled:
        ui.render_key_value_block(
            "Retention Status",
            [
                ("Items", totals.get("item_count", 0)),
                ("Expired", totals.get("expired_count", 0)),
                ("Total", _format_bytes(totals.get("total_bytes", 0))),
                ("Reclaimable", _format_bytes(totals.get("reclaimable_bytes", 0))),
            ],
        )
        if last_auto_prune.get("executed_at"):
            ui.emit(
                "Last auto prune: "
                f"trigger={last_auto_prune.get('trigger', '-')} | "
                f"deleted={last_auto_prune.get('deleted_count', 0)} | "
                f"expired={last_auto_prune.get('expired_count', 0)} | "
                f"reclaimable={_format_bytes(last_auto_prune.get('reclaimable_bytes', 0))} | "
                f"at={last_auto_prune.get('executed_at')}"
            )
        if last_auto_prune_check.get("checked_at") and last_auto_prune_check.get("status") != "executed":
            ui.emit(
                "Last auto prune check: "
                f"status={last_auto_prune_check.get('status', '-')} | "
                f"reason={last_auto_prune_check.get('reason', '-')} | "
                f"trigger={last_auto_prune_check.get('trigger', '-')} | "
                f"checked_at={last_auto_prune_check.get('checked_at')}"
            )
        ui.render_retention_targets(target_rows)
        return True
    output_func(
        "Retention status: "
        f"items={totals.get('item_count', 0)} | "
        f"expired={totals.get('expired_count', 0)} | "
        f"total={_format_bytes(totals.get('total_bytes', 0))} | "
        f"reclaimable={_format_bytes(totals.get('reclaimable_bytes', 0))}"
    )
    if last_auto_prune.get("executed_at"):
        output_func(
            "Last auto prune: "
            f"trigger={last_auto_prune.get('trigger', '-')} | "
            f"deleted={last_auto_prune.get('deleted_count', 0)} | "
            f"expired={last_auto_prune.get('expired_count', 0)} | "
            f"reclaimable={_format_bytes(last_auto_prune.get('reclaimable_bytes', 0))} | "
            f"at={last_auto_prune.get('executed_at')}"
        )
    if last_auto_prune_check.get("checked_at") and last_auto_prune_check.get("status") != "executed":
        output_func(
            "Last auto prune check: "
            f"status={last_auto_prune_check.get('status', '-')} | "
            f"reason={last_auto_prune_check.get('reason', '-')} | "
            f"trigger={last_auto_prune_check.get('trigger', '-')} | "
            f"checked_at={last_auto_prune_check.get('checked_at')}"
        )
    for item in payload.get("targets", []) or []:
        output_func(
            f"  - {item.get('key')} | days={item.get('retention_days')} | max={item.get('max_items', 0)} | max_bytes={_format_bytes(item.get('max_total_bytes', 0))} | items={item.get('item_count', 0)} | "
            f"expired={item.get('expired_count', 0)} | total={_format_bytes(item.get('total_bytes', 0))} | "
            f"reclaimable={_format_bytes(item.get('reclaimable_bytes', 0))}"
        )
    return True


def handle_prune_runtime_data(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    apply = len(parts) >= 2 and parts[1].strip().lower() == "apply"
    if len(parts) >= 2 and parts[1].strip().lower() not in {"apply"}:
        output_func("Usage: /prune_runtime_data [apply]")
        return True

    payload = context["agent_instance"].prune_runtime_data(apply=apply)
    totals = payload.get("totals", {}) or {}
    output_func(
        "Prune runtime data: "
        f"mode={payload.get('mode', 'dry_run')} | "
        f"expired={totals.get('expired_count', 0)} | "
        f"reclaimable={_format_bytes(totals.get('reclaimable_bytes', 0))} | "
        f"deleted={totals.get('deleted_count', 0)}"
    )
    for item in payload.get("targets", []) or []:
        output_func(
            f"  - {item.get('key')} | expired={item.get('expired_count', 0)} | "
            f"reclaimable={_format_bytes(item.get('reclaimable_bytes', 0))} | "
            f"deleted={item.get('deleted_count', 0)}"
        )
    if not apply:
        output_func("Use /prune_runtime_data apply to remove expired artifacts.")
    return True


def handle_failure_memories(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    limit = 5
    keyword_start = 1
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
            keyword_start = 2
        except ValueError:
            keyword_start = 1

    keywords = parts[keyword_start:]
    items = context["agent_instance"].get_failure_memories(
        match_keywords=keywords,
        limit=limit,
        exclude_conv_id=context["session_state"].get("session_id"),
    )
    if not items:
        output_func("No failure memories found.")
        return True

    memory_rows = []
    for item in items:
        tags = ",".join(item.get("quality_tags", [])) or "-"
        keywords_text = ",".join(item.get("keywords", [])[:5]) or "-"
        memory_rows.append([
            f"#{item.get('id')}",
            item.get("request_id") or "-",
            item.get("memory_type") or "-",
            tags,
            keywords_text,
            item.get("summary") or "-",
        ])
    if ui and ui.rich_enabled:
        ui.render_failure_memories(memory_rows, keywords)
        return True
    if keywords:
        output_func("Failure memories: keywords=" + ",".join(keywords))
    else:
        output_func("Failure memories:")
    for item in items:
        tags = ",".join(item.get("quality_tags", [])) or "-"
        keywords_text = ",".join(item.get("keywords", [])[:5]) or "-"
        output_func(
            f"  - #{item.get('id')} | request={item.get('request_id') or '-'} | type={item.get('memory_type') or '-'} | "
            f"tags={tags} | keywords={keywords_text} | summary={item.get('summary') or '-'}"
        )
    return True


def handle_new_session(_user_input: str, context: dict[str, Any]) -> bool:
    session_id = context["agent_instance"].start_session()
    context["session_state"]["session_id"] = session_id
    context["output_func"](f"Started new session: {session_id}")
    return True


def handle_plain_message(user_input: str, context: dict[str, Any]) -> bool:
    agent_instance = context["agent_instance"]
    response = agent_instance.invoke(user_input, session_id=context["session_state"]["session_id"])
    emit_agent_response(context, response, agent_instance.get_last_request_id())
    return True


def build_commands() -> tuple[dict[str, CLICommand], list[str]]:
    command_specs = [
        CLICommand("/llm", "/llm <provider> <model> [base_url] [api_key] - Switch LLM provider (ollama/openai)", handle_llm),
        CLICommand("/load_tool", "/load_tool <tool_name.py> - Load a local python tool", handle_load_tool),
        CLICommand("/load_skill", "/load_skill <skill_name.md> - Load a local markdown skill", handle_load_skill),
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
        CLICommand("/recent_requests", "/recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m] - Show recent request summaries and key metrics", handle_recent_requests),
        CLICommand("/request_rollup", "/request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m] - Show aggregate metrics across recent requests", handle_request_rollup),
        CLICommand("/failed_requests", "/failed_requests [limit] [status=failed,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m] - Alias of recent request query with failure-oriented defaults", handle_failed_requests),
        CLICommand("/resumed_requests", "/resumed_requests [limit] [status=failed,blocked,...] [attention] [since=30m] - Alias of recent request query scoped to resumed requests", handle_resumed_requests),
        CLICommand("/list_tool_runs", "/list_tool_runs [request_id] [running|detached] - Show tracked in-flight or detached tool runs", handle_list_tool_runs),
        CLICommand("/failure_memories", "/failure_memories [limit] [keywords...] - Show failure-case memories ranked by relevance", handle_failure_memories),
        CLICommand("/retention_status", "/retention_status - Show retention coverage and reclaimable runtime artifacts", handle_retention_status),
        CLICommand("/prune_runtime_data", "/prune_runtime_data [apply] - Dry-run or apply retention cleanup for logs, snapshots, audit logs, and memory backups", handle_prune_runtime_data),
        CLICommand("/new_session", "/new_session - Start a new memory session", handle_new_session),
    ]
    command_map = {item.name: item for item in command_specs}
    command_map["help"] = CLICommand("help", "help - Show available commands", handle_help)
    return command_map, [item.name for item in command_specs]