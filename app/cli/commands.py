from dataclasses import dataclass
import re
from typing import Any, Callable

RESPONSE_DIVIDER = "----------------------------------------"


@dataclass(frozen=True)
class CLICommand:
    name: str
    help_text: str
    handler: Callable[[str, dict[str, Any]], bool | None]


def emit_response(output_func, response: str, request_id: str) -> None:
    if request_id:
        output_func(f"Request ID: {request_id}")
    output_func(f"\nResponse: {response}")
    output_func(RESPONSE_DIVIDER)


def build_request_mode_label(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "-"
    return "lite_chat" if summary.get("lite_mode") else "task"


def parse_detail_fields(details: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    raw_details = str(details or "")
    for key in ("tool", "reason", "error_type", "action", "mode", "error", "tool_run_id"):
        match = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+)", raw_details)
        if match:
            parsed[key] = match.group(1).strip()
    return parsed


def build_blocked_reason(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    normalized_status = str(summary.get("status") or "").strip().lower()
    if normalized_status not in {"blocked", "waiting_user"}:
        return ""
    triage = summary.get("triage", {}) or {}
    details = parse_detail_fields(str(triage.get("latest_failure_details", "") or ""))
    stage = str(triage.get("latest_failure_stage", "") or "").strip().lower()
    action = details.get("action", "")
    tool_name = details.get("tool", "")
    error_type = details.get("error_type", "")
    reason = details.get("reason", "")
    error = details.get("error", "")

    if action == "ask_user":
        if tool_name:
            return f"当前请求需要补充信息后才能继续，相关工具: {tool_name}。"
        return "当前请求需要你补充关键信息后才能继续。"
    if action == "retry_limit":
        if tool_name and error_type:
            return f"当前路径已达到重试上限，失败点集中在工具 {tool_name}，错误类型为 {error_type}。"
        return "当前路径已达到重试上限，继续沿用同一路径的收益较低。"
    if stage == "agent_blocked":
        return "当前请求已进入阻塞状态，需要调整输入或切换恢复路径。"
    if stage == "agent_waiting_user":
        return "当前请求正在等待你补充信息，补充后即可继续。"
    if reason:
        return f"当前请求被阻塞，原因: {reason}。"
    if error:
        return f"当前请求被阻塞，错误: {error}。"
    return "当前请求被阻塞，需要补充信息或切换处理路径。"


def build_next_action_suggestions(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return []
    request_id = str(summary.get("request_id") or "").strip()
    status = str(summary.get("status") or "").strip().lower()
    if status in {"blocked", "waiting_user"}:
        triage = summary.get("triage", {}) or {}
        details = parse_detail_fields(str(triage.get("latest_failure_details", "") or ""))
        suggestions = [
            "补充缺失参数后继续提问" if details.get("action") == "ask_user" else "尝试换一种更明确的输入重新提问",
            f"/summary {request_id}" if request_id else "/summary",
            f"/snapshots {request_id}" if request_id else "/snapshots",
        ]
        if request_id:
            suggestions.append(f"/resume {request_id} planning_completed reroute")
        return [item for item in suggestions if item]
    suggestions = ["继续直接输入下一条消息"]
    if request_id:
        suggestions.append(f"/summary {request_id}")
        suggestions.append(f"/snapshots {request_id}")
    return suggestions


def build_process_summary_line(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    metrics = summary.get("metrics", {}) or {}
    return (
        f"request={summary.get('request_id') or '-'} | "
        f"stage={summary.get('latest_stage') or '-'} | "
        f"mode={build_request_mode_label(summary)} | "
        f"progress={summary.get('subtask_index', 0)}/{summary.get('plan_length', 0)} | "
        f"tool_calls={metrics.get('tool_call_count', 0)} | "
        f"total_ms={metrics.get('total_duration_ms', '-')}"
    )


def build_input_suggestion_candidates(context: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    candidates.extend(context.get("command_order") or [])

    llm_manager_instance = context.get("llm_manager_instance")
    if llm_manager_instance is not None and hasattr(llm_manager_instance, "list_available_models"):
        candidates.extend(["/llm", "/llm list", "llm"])
        for item in llm_manager_instance.list_available_models() or []:
            model_key = str(item.get("key") or "").strip()
            if not model_key:
                continue
            candidates.append(f"/llm {model_key}")
            candidates.append(f"llm {model_key}")

    recent_inputs = list(context.get("session_state", {}).get("recent_inputs") or [])
    candidates.extend(reversed(recent_inputs))

    agent_instance = context.get("agent_instance")
    if agent_instance is not None and hasattr(agent_instance, "get_last_request_id"):
        last_request_id = str(agent_instance.get_last_request_id() or "").strip()
        if last_request_id:
            candidates.append(last_request_id)
            if hasattr(agent_instance, "list_snapshots"):
                for item in agent_instance.list_snapshots(last_request_id) or []:
                    snapshot_name = str(item.get("file") or "").strip()
                    if snapshot_name:
                        candidates.append(snapshot_name)
    if agent_instance is not None and hasattr(agent_instance, "get_recent_request_summaries"):
        for item in agent_instance.get_recent_request_summaries(limit=5) or []:
            request_id = str(item.get("request_id") or "").strip()
            if request_id:
                candidates.append(request_id)

    deduped: list[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def build_input_suggestions(context: dict[str, Any], prefix: str = "") -> list[str]:
    normalized_prefix = str(prefix or "").strip().lower()
    suggestions: list[str] = []
    for item in build_input_suggestion_candidates(context):
        candidate = str(item or "")
        if not candidate:
            continue
        if normalized_prefix and not candidate.lower().startswith(normalized_prefix):
            continue
        suggestions.append(candidate)
    return suggestions[:12]


def build_natural_language_command(user_input: str) -> str | None:
    normalized = str(user_input or "").strip()
    if not normalized or normalized.startswith("/"):
        return None
    collapsed = re.sub(r"\s+", "", normalized.lower())
    safe_mappings = [
        (("最近失败请求", "看最近失败请求", "查看最近失败请求", "看一下最近失败请求"), "/latest_failure"),
        (("恢复上一个阻塞任务", "恢复最近阻塞请求", "恢复最近阻塞任务", "恢复上一个阻塞请求"), "/resume_last_blocked"),
        (("查看detachedtools", "看detachedtools", "查看最近detachedtools", "看最近detachedtools"), "/detached_tools"),
        (("查看最近请求", "看最近请求", "看一下最近请求", "最近请求"), "/recent 5"),
        (("查看对话历史", "看对话历史", "看一下对话历史", "最近对话历史"), "/history 5"),
    ]
    for phrases, command in safe_mappings:
        for phrase in phrases:
            if phrase in collapsed:
                return command
    return None


def emit_plain_section(output_func, title: str, lines: list[str], leading_blank_line: bool = False) -> None:
    section_lines = [str(item) for item in lines if str(item or "").strip()]
    if not section_lines:
        return
    if leading_blank_line:
        output_func("")
    output_func(f"{title}:")
    for item in section_lines:
        output_func(f"  {item}")


def emit_conversation_context(output_func, summary: dict[str, Any]) -> None:
    summary_line = build_process_summary_line(summary)
    emit_plain_section(output_func, "Summary", [summary_line])
    raw_query = str(summary.get("raw_query") or "").strip()
    normalized_query = str(summary.get("normalized_query") or "").strip()
    mode_label = build_request_mode_label(summary)
    status = str(summary.get("status") or "-")
    context_lines = []
    if raw_query:
        context_lines.append(f"Input: {raw_query}")
    if normalized_query and normalized_query != raw_query:
        context_lines.append(f"Intent: {normalized_query}")
    context_lines.append(f"Mode: {mode_label} | Status: {status}")
    emit_plain_section(output_func, "Conversation Context", context_lines, leading_blank_line=True)
    blocked_reason = build_blocked_reason(summary)
    if blocked_reason:
        emit_plain_section(output_func, "Risk", [f"Blocked Reason: {blocked_reason}"], leading_blank_line=True)


def emit_agent_response(context: dict[str, Any], response: str, request_id: str, summary: dict[str, Any] | None = None) -> None:
    ui = context.get("ui")
    if ui:
        ui.render_response(request_id, response, summary=summary)
        return
    if summary:
        emit_conversation_context(context["output_func"], summary)
    emit_plain_section(context["output_func"], "Response", [response], leading_blank_line=True)
    context["output_func"](RESPONSE_DIVIDER)


def set_selection_context(context: dict[str, Any], payload: dict[str, Any] | None) -> None:
    context["session_state"]["selection_context"] = payload or {}


def get_selection_context_label(selection_context: dict[str, Any] | None) -> str:
    context_payload = selection_context or {}
    selection_type = str(context_payload.get("type") or "").strip()
    items = context_payload.get("items") or []
    if selection_type == "request_list":
        return f"request_list:{len(items)}"
    if selection_type == "snapshot_list":
        request_id = str(context_payload.get("request_id") or "").strip()
        return f"snapshot_list:{request_id or '-'}"
    if selection_type == "llm_model_list":
        return f"llm_model_list:{len(items)}"
    return ""


def build_selection_hint(selection_type: str) -> str:
    if selection_type == "request_list":
        return "Select: /pick <index> [summary|snapshots|resume] to continue from a listed request."
    if selection_type == "snapshot_list":
        return "Select: /pick <index> to resume from a listed snapshot."
    if selection_type == "llm_model_list":
        return "Select: /pick <index> to switch to a listed LLM model."
    return ""


def choose_resume_snapshot_name(agent_instance, request_id: str, reroute: bool = False) -> str | None:
    snapshots = agent_instance.list_snapshots(request_id)
    if not snapshots:
        return None

    def select_by_stage_priority(stage_priority: list[str]) -> str | None:
        for stage_name in stage_priority:
            for item in reversed(snapshots):
                if str(item.get("stage") or "") == stage_name:
                    return str(item.get("file") or "") or None
        return None

    if reroute:
        candidate = select_by_stage_priority([
            "agent_blocked",
            "subtask_replanned",
            "subtask_advanced",
            "subtask_prepared",
            "planning_completed",
            "request_received",
        ])
        if candidate:
            return candidate

    for item in reversed(snapshots):
        if not bool(item.get("completed", False)):
            file_name = str(item.get("file") or "").strip()
            if file_name:
                return file_name

    last_item = snapshots[-1]
    return str(last_item.get("file") or "").strip() or None


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


def handle_suggest(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    prefix = user_input.split(maxsplit=1)[1] if len(user_input.split(maxsplit=1)) == 2 else ""
    suggestions = build_input_suggestions(context, prefix)
    if not suggestions:
        output_func("Suggestions: none")
        return True
    output_func("Suggestions: " + " | ".join(suggestions))
    return True


def handle_recent_commands(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    limit = 5
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output_func("Usage: /recent_commands [limit]")
            return True
    recent_inputs = list(context.get("session_state", {}).get("recent_inputs") or [])
    if not recent_inputs:
        output_func("Recent commands: none")
        return True
    output_func("Recent commands:")
    for item in list(reversed(recent_inputs))[:limit]:
        output_func(f"  - {item}")
    return True


def handle_llm(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    llm_manager_instance = context["llm_manager_instance"]
    parts = user_input.split()

    def emit_available_models() -> None:
        available_models = list(llm_manager_instance.list_available_models())
        current_key = ""
        if hasattr(llm_manager_instance, "get_current_model_key"):
            current_key = str(llm_manager_instance.get_current_model_key() or "")
        set_selection_context(
            context,
            {
                "type": "llm_model_list",
                "items": [
                    {
                        "index": index,
                        "key": item.get("key"),
                        "provider": item.get("provider"),
                        "model": item.get("model"),
                        "base_url": item.get("base_url"),
                    }
                    for index, item in enumerate(available_models, start=1)
                ],
            } if available_models else None,
        )
        output_func("Configured LLM models:")
        for index, item in enumerate(available_models, start=1):
            markers = []
            if item.get("is_current"):
                markers.append("current")
            if item.get("is_default"):
                markers.append("default")
            marker_text = f" [{', '.join(markers)}]" if markers else ""
            output_func(
                f"  [{index}] {item.get('key')}{marker_text} | {item.get('provider')}:{item.get('model')}"
                + (f" | base_url={item.get('base_url')}" if item.get("base_url") else "")
            )
        if current_key:
            output_func(f"Current model key: {current_key}")
        selection_hint = build_selection_hint("llm_model_list")
        if selection_hint:
            output_func(selection_hint)
        output_func("Usage: /llm <model_key>")
        output_func("Legacy: /llm raw <provider> <model> [base_url] [api_key]")

    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() in {"list", "ls"}):
        emit_available_models()
        return True

    if len(parts) == 2:
        try:
            output_func(llm_manager_instance.set_model_by_key(parts[1]))
        except KeyError as exc:
            output_func(str(exc))
            emit_available_models()
        return True

    if len(parts) >= 4 and parts[1].lower() == "raw":
        provider = parts[2]
        model = parts[3]
        base_url = parts[4] if len(parts) > 4 else None
        api_key = parts[5] if len(parts) > 5 else None
        output_func(llm_manager_instance.set_model(provider, model, base_url, api_key))
        return True

    output_func("Usage: /llm <model_key>")
    output_func("Legacy: /llm raw <provider> <model> [base_url] [api_key]")
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


def handle_grant_write(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) == 2:
        output_func(context["agent_instance"].grant_write_access(parts[1]))
    else:
        output_func("Usage: /grant_write <folder_path>")
    return True


def handle_revoke_write(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split(maxsplit=1)
    if len(parts) == 2:
        output_func(context["agent_instance"].revoke_write_access(parts[1]))
    else:
        output_func("Usage: /revoke_write <folder_path>")
    return True


def handle_list_write_roots(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    roots = context["agent_instance"].list_write_access_roots()
    output_func("Writable roots:")
    for item in roots:
        output_func(f"  {item}")
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
    if ui and ui.structured_output_enabled:
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
        summary = agent_instance.get_request_summary(agent_instance.get_last_request_id()) if hasattr(agent_instance, "get_request_summary") else None
        emit_agent_response(context, response, agent_instance.get_last_request_id(), summary=summary)
    else:
        output_func("Usage: /resume_snapshot <request_id> [latest|index|stage|snapshot_file] [reroute]")
    return True


def handle_list_snapshots(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    target_request_id = parts[1] if len(parts) >= 2 else context["agent_instance"].get_last_request_id()
    if target_request_id:
        snapshots = context["agent_instance"].list_snapshots(target_request_id)
        set_selection_context(
            context,
            {
                "type": "snapshot_list",
                "request_id": target_request_id,
                "items": [dict(item) for item in snapshots],
            },
        )
        if ui and ui.structured_output_enabled:
            ui.render_snapshots(snapshots)
            selection_hint = build_selection_hint("snapshot_list")
            if snapshots and selection_hint:
                ui.emit(selection_hint)
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
            selection_hint = build_selection_hint("snapshot_list")
            if selection_hint:
                output_func(selection_hint)
    else:
        output_func("Usage: /list_snapshots [request_id]")
    return True


def handle_cancel_request(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    if len(parts) >= 2:
        output_func(context["agent_instance"].cancel_request(parts[1]))
    else:
        output_func("Usage: /cancel_request <request_id>")
    return True


def handle_pick(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    selection_context = context["session_state"].get("selection_context") or {}
    selection_type = str(selection_context.get("type") or "").strip()
    parts = user_input.split()
    if len(parts) < 2:
        output_func("Usage: /pick <index> [summary|snapshots|resume]")
        return True
    if not selection_type:
        output_func("No selectable list context. Run /recent or /snapshots first.")
        return True
    try:
        selected_index = int(parts[1])
    except ValueError:
        output_func("Usage: /pick <index> [summary|snapshots|resume]")
        return True
    action = str(parts[2]).strip().lower() if len(parts) >= 3 else ""

    items = selection_context.get("items") or []
    selected_item = None
    for item in items:
        try:
            item_index = int(item.get("index"))
        except Exception:
            continue
        if item_index == selected_index:
            selected_item = item
            break
    if selected_item is None:
        output_func("Selection index out of range.")
        return True

    if selection_type == "request_list":
        request_id = str(selected_item.get("request_id") or "").strip()
        if not request_id:
            output_func("Selected request is invalid.")
            return True
        if action in {"", "summary", "open"}:
            return handle_request_summary(f"/request_summary {request_id}", context)
        if action in {"snapshots", "snapshot"}:
            return handle_list_snapshots(f"/list_snapshots {request_id}", context)
        if action in {"resume", "reroute"}:
            snapshot_name = choose_resume_snapshot_name(context["agent_instance"], request_id, reroute=True)
            if snapshot_name:
                return handle_resume_snapshot(f"/resume_snapshot {request_id} {snapshot_name} reroute", context)
            return handle_resume_snapshot(f"/resume_snapshot {request_id} planning_completed reroute", context)
        output_func("Usage: /pick <index> [summary|snapshots|resume]")
        return True

    if selection_type == "snapshot_list":
        request_id = str(selection_context.get("request_id") or "").strip()
        snapshot_name = str(selected_item.get("file") or "").strip()
        if not request_id or not snapshot_name:
            output_func("Selected snapshot is invalid.")
            return True
        if action and action not in {"resume", "open"}:
            output_func("Usage: /pick <index> [summary|snapshots|resume]")
            return True
        return handle_resume_snapshot(f"/resume_snapshot {request_id} {snapshot_name}", context)

    if selection_type == "llm_model_list":
        model_key = str(selected_item.get("key") or "").strip()
        if not model_key:
            output_func("Selected model is invalid.")
            return True
        if action and action not in {"switch", "open"}:
            output_func("Usage: /pick <index>")
            return True
        try:
            output_func(context["llm_manager_instance"].set_model_by_key(model_key))
        except KeyError as exc:
            output_func(str(exc))
        return True

    output_func("Current selection type is not supported.")
    return True


def handle_selection(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    selection_context = context["session_state"].get("selection_context") or {}
    selection_type = str(selection_context.get("type") or "").strip()
    if not selection_type:
        output_func("Selection context: none")
        return True
    label = get_selection_context_label(selection_context)
    output_func(f"Selection context: {label}")
    hint = build_selection_hint(selection_type)
    if hint:
        output_func(hint)
    return True


def handle_clear_selection(_user_input: str, context: dict[str, Any]) -> bool:
    set_selection_context(context, None)
    context["output_func"]("Selection context cleared.")
    return True


def get_latest_request_summary(
    context: dict[str, Any],
    statuses: list[str] | None = None,
    resumed_only: bool = False,
    attention_only: bool = False,
) -> dict[str, Any] | None:
    items = context["agent_instance"].get_recent_request_summaries(
        limit=1,
        statuses=statuses or None,
        resumed_only=resumed_only,
        attention_only=attention_only,
    )
    if not items:
        return None
    request_id = str(items[0].get("request_id") or "").strip()
    if not request_id:
        return None
    summary = context["agent_instance"].get_request_summary(request_id)
    return dict(summary) if summary else dict(items[0])


def handle_latest_failure(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    parts = user_input.split()
    action = str(parts[1]).strip().lower() if len(parts) >= 2 else "summary"
    if action not in {"summary", "snapshots", "resume"}:
        output_func("Usage: /latest_failure [summary|snapshots|resume]")
        return True

    summary = get_latest_request_summary(context, statuses=["failed", "blocked", "waiting_user", "timed_out", "cancelled"], attention_only=True)
    if not summary:
        output_func("No recent failed, waiting, or blocked request found.")
        return True
    request_id = str(summary.get("request_id") or "").strip()
    if action == "summary":
        return handle_request_summary(f"/request_summary {request_id}", context)
    if action == "snapshots":
        return handle_list_snapshots(f"/list_snapshots {request_id}", context)
    snapshot_name = choose_resume_snapshot_name(context["agent_instance"], request_id, reroute=True)
    if snapshot_name:
        return handle_resume_snapshot(f"/resume_snapshot {request_id} {snapshot_name} reroute", context)
    return handle_resume_snapshot(f"/resume_snapshot {request_id} planning_completed reroute", context)


def handle_resume_last_blocked(_user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    summary = get_latest_request_summary(context, statuses=["waiting_user", "blocked"], attention_only=True)
    if not summary:
        output_func("No recent waiting-user or blocked request found.")
        return True
    request_id = str(summary.get("request_id") or "").strip()
    snapshot_name = choose_resume_snapshot_name(context["agent_instance"], request_id, reroute=True)
    if snapshot_name:
        return handle_resume_snapshot(f"/resume_snapshot {request_id} {snapshot_name} reroute", context)
    return handle_resume_snapshot(f"/resume_snapshot {request_id} planning_completed reroute", context)


def handle_detached_tools(user_input: str, context: dict[str, Any]) -> bool:
    parts = user_input.split()
    request_id = parts[1] if len(parts) >= 2 else context["agent_instance"].get_last_request_id()
    if request_id:
        return handle_list_tool_runs(f"/list_tool_runs {request_id} detached", context)
    return handle_list_tool_runs("/list_tool_runs detached", context)


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


def build_subtask_view_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    plan = summary.get("plan") or []
    if not isinstance(plan, list):
        plan = []
    if not plan:
        if summary.get("lite_mode"):
            return [{"step": "Direct exchange", "status": "active", "detail": "lite_chat 请求直接走单轮交流路径"}]
        return []

    current_index = int(summary.get("subtask_index", 0) or 0)
    status_text = str(summary.get("status") or "").strip().lower()
    rows: list[dict[str, str]] = []
    for index, item in enumerate(plan, start=1):
        description = str(item.get("description") or "").strip() or f"Step {index}"
        execution_mode = str(item.get("execution_mode") or "leaf").strip() or "leaf"
        if index < current_index:
            row_status = "completed"
        elif index == current_index:
            row_status = "active"
        else:
            row_status = "pending"
        if status_text == "completed" and index <= max(current_index, 1):
            row_status = "completed"
        if status_text in {"blocked", "waiting_user"} and index == max(current_index, 1):
            row_status = "blocked"
        rows.append({"step": description, "status": row_status, "detail": f"mode={execution_mode}"})
    return rows


def build_tool_feedback_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    checkpoints = summary.get("checkpoints") or []
    stage_map = {
        "tool_started": "started",
        "tool_succeeded": "succeeded",
        "tool_failed": "failed",
        "tool_cancelled": "cancelled",
        "tool_detached": "detached",
        "tool_rejected": "rejected",
    }

    for item in checkpoints:
        stage = str(item.get("stage") or "").strip().lower()
        if stage not in stage_map:
            continue
        parsed = parse_detail_fields(str(item.get("details") or ""))
        tool_name = parsed.get("tool", "") or "-"
        run_id = parsed.get("tool_run_id", "") or "-"
        summary_parts = []
        if parsed.get("reason"):
            summary_parts.append(f"reason={parsed['reason']}")
        if parsed.get("error_type"):
            summary_parts.append(f"error_type={parsed['error_type']}")
        if parsed.get("action"):
            summary_parts.append(f"action={parsed['action']}")
        row = {
            "status": stage_map[stage],
            "tool": tool_name,
            "summary": " | ".join(summary_parts) or "-",
            "source": str(item.get("source") or "checkpoint") or "checkpoint",
            "run_id": run_id,
        }
        key = (row["status"], row["tool"], row["run_id"])
        if key not in seen_keys:
            seen_keys.add(key)
            rows.append(row)

    for item in summary.get("detached_tools") or []:
        run_id = str(item.get("tool_run_id") or "-")
        tool_name = str(item.get("tool_name") or "-")
        summary_parts = []
        if item.get("reason"):
            summary_parts.append(f"reason={item['reason']}")
        if item.get("detached_runtime_ms") is not None:
            summary_parts.append(f"detached_ms={item['detached_runtime_ms']}")
        elif item.get("runtime_ms") is not None:
            summary_parts.append(f"runtime_ms={item['runtime_ms']}")
        row = {
            "status": str(item.get("status") or "detached"),
            "tool": tool_name,
            "summary": " | ".join(summary_parts) or "-",
            "source": str(item.get("source") or "runtime") or "runtime",
            "run_id": run_id,
        }
        key = (row["status"], row["tool"], row["run_id"])
        if key not in seen_keys:
            seen_keys.add(key)
            rows.append(row)

    triage = summary.get("triage", {}) or {}
    failure_details = parse_detail_fields(str(triage.get("latest_failure_details") or ""))
    if failure_details.get("tool"):
        normalized_status = str(summary.get("status") or "").strip().lower()
        row_status = "blocked" if normalized_status in {"blocked", "waiting_user"} else "failed"
        summary_parts = []
        if failure_details.get("reason"):
            summary_parts.append(f"reason={failure_details['reason']}")
        if failure_details.get("error_type"):
            summary_parts.append(f"error_type={failure_details['error_type']}")
        if failure_details.get("action"):
            summary_parts.append(f"action={failure_details['action']}")
        row = {
            "status": row_status,
            "tool": failure_details["tool"],
            "summary": " | ".join(summary_parts) or "-",
            "source": str(triage.get("latest_failure_source") or "triage") or "triage",
            "run_id": failure_details.get("tool_run_id", "") or "-",
        }
        key = (row["status"], row["tool"], row["run_id"])
        if key not in seen_keys:
            rows.append(row)

    return rows[-5:]


def handle_request_summary(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    parts = user_input.split()
    target_request_id = parts[1] if len(parts) >= 2 else context["agent_instance"].get_last_request_id()
    if not target_request_id:
        output_func("Usage: /request_summary [request_id]")
        return True

    summary = context["agent_instance"].get_request_summary(target_request_id)
    if not summary:
        output_func("Request not found.")
        return True

    summary = dict(summary)
    summary["blocked_reason"] = build_blocked_reason(summary)
    summary["suggested_actions"] = build_next_action_suggestions(summary)
    triage = summary.get("triage", {}) or {}
    metrics_line, stage_duration_line, triage_line = build_request_summary_lines(summary, context["agent_instance"].observability)

    if ui and ui.structured_output_enabled:
        ui.render_request_summary(summary, metrics_line, stage_duration_line, triage_line)
        tool_feedback_rows = build_tool_feedback_rows(summary)
        if tool_feedback_rows:
            ui.render_tool_feedback(tool_feedback_rows)
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
    process_summary_line = build_process_summary_line(summary)
    emit_plain_section(output_func, "Summary", [process_summary_line])
    raw_query = str(summary.get("raw_query") or "").strip()
    normalized_query = str(summary.get("normalized_query") or "").strip()
    context_lines = []
    if raw_query:
        context_lines.append(f"Input: {raw_query}")
    if normalized_query and normalized_query != raw_query:
        context_lines.append(f"Intent: {normalized_query}")
    context_lines.extend(
        [
            f"Mode: {build_request_mode_label(summary)}",
            f"Status: {summary['status']}",
            f"Session ID: {summary['session_id'] or '-'}",
            f"Latest stage: {summary['latest_stage'] or '-'}",
            f"Progress: {summary['subtask_index']}/{summary['plan_length']}",
        ]
    )
    emit_plain_section(output_func, "Conversation Context", context_lines, leading_blank_line=True)
    subtask_rows = build_subtask_view_rows(summary)
    if subtask_rows:
        emit_plain_section(
            output_func,
            "Subtasks",
            [f"- [{item['status']}] {item['step']} | outcome={item['detail']}" for item in subtask_rows],
            leading_blank_line=True,
        )
    tool_feedback_rows = build_tool_feedback_rows(summary)
    if tool_feedback_rows:
        emit_plain_section(
            output_func,
            "Tool Feedback",
            [
                f"- [{item['status']}] {item['tool']} | summary={item['summary']} | source={item['source']} | run_id={item['run_id']}"
                for item in tool_feedback_rows
            ],
            leading_blank_line=True,
        )
    output_func("")
    output_func("Request Assets:")
    output_func(f"  Snapshots: {summary['snapshot_count']}")
    output_func(f"  Memories: {summary['memory_count']}")
    output_func(f"  Checkpoints: {summary['checkpoint_count']}")
    if metrics_line:
        emit_plain_section(output_func, "Metrics", [metrics_line], leading_blank_line=True)
    if stage_duration_line:
        emit_plain_section(output_func, "Stage Durations", [stage_duration_line], leading_blank_line=True)
    if triage_line:
        emit_plain_section(output_func, "Triage", [triage_line], leading_blank_line=True)
    if triage.get("last_error_message"):
        emit_plain_section(output_func, "Risk", [f"Last error: {triage['last_error_message']}"], leading_blank_line=True)
    if triage.get("latest_failure_details"):
        emit_plain_section(output_func, "Risk", [f"Failure details: {triage['latest_failure_details']}"], leading_blank_line=True)
    if triage.get("latest_reroute_details"):
        emit_plain_section(output_func, "Risk", [f"Reroute details: {triage['latest_reroute_details']}"], leading_blank_line=True)
    if summary.get("source_request_id"):
        emit_plain_section(output_func, "Risk", [f"Source request: {summary['source_request_id']}"], leading_blank_line=True)
    if summary.get("final_response"):
        emit_plain_section(output_func, "Final Response", [summary['final_response']], leading_blank_line=True)
    if summary["checkpoints"]:
        emit_plain_section(
            output_func,
            "Recent Checkpoints",
            [
                f"- {item['logged_at']} | {item['stage']}{f' | {item['details']}' if item.get('details') else ''}"
                for item in summary["checkpoints"][-5:]
            ],
            leading_blank_line=True,
        )
    if summary["memories"]:
        emit_plain_section(
            output_func,
            "Related Memories",
            [
                f"- #{item['id']} | {item['memory_type']} | tags={','.join(item.get('quality_tags', [])) or '-'} | {item['summary']}"
                for item in summary["memories"][-5:]
            ],
            leading_blank_line=True,
        )
    if summary.get("detached_tools"):
        detached_lines = []
        for item in summary["detached_tools"][-5:]:
            runtime_suffix = ""
            if item.get("detached_runtime_ms") is not None:
                runtime_suffix = f" | detached_ms={item['detached_runtime_ms']}"
            elif item.get("runtime_ms") is not None:
                runtime_suffix = f" | runtime_ms={item['runtime_ms']}"
            detached_lines.append(
                f"- {item.get('tool_name') or '-'} | run_id={item.get('tool_run_id') or '-'} | reason={item.get('reason') or '-'} | source={item.get('source') or '-'}{runtime_suffix}"
            )
        emit_plain_section(output_func, "Detached Tools", detached_lines, leading_blank_line=True)
    return True


def render_recent_request_rows(output_func, summaries: list[dict[str, Any]], heading: str) -> None:
    if not summaries:
        output_func("No matching requests found.")
        return

    output_func(heading)
    for index, item in enumerate(summaries, start=1):
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
            f"  - [{index}] {item['request_id']} | status={item['status']} | session={item.get('session_id') or '-'} | "
            f"stage={item.get('latest_stage') or '-'} | updated_at={item.get('updated_at') or '-'} | "
            f"total_ms={metrics.get('total_duration_ms', '-')} | llm_calls={metrics.get('llm_call_count', 0)} | "
            f"tool_calls={metrics.get('tool_call_count', 0)} | detached_tools={metrics.get('tool_detached_count', 0)}{resumed_suffix}{attention_suffix}"
        )


def render_history_rows(output_func, summaries: list[dict[str, Any]], heading: str) -> None:
    if not summaries:
        output_func("No matching requests found.")
        return
    output_func(heading)
    for index, item in enumerate(summaries, start=1):
        raw_query = str(item.get("raw_query") or "-").strip() or "-"
        normalized_query = str(item.get("normalized_query") or "").strip()
        intent_text = normalized_query if normalized_query and normalized_query != raw_query else "-"
        final_response = str(item.get("final_response") or "-").strip() or "-"
        mode_label = build_request_mode_label(item)
        output_func(
            f"  - [{index}] {item.get('request_id') or '-'} | mode={mode_label} | status={item.get('status') or '-'} | "
            f"input={raw_query} | intent={intent_text} | reply={final_response}"
        )


def build_recent_request_row(index: int, item: dict[str, Any]) -> list[Any]:
    metrics = item.get("metrics", {})
    triage = item.get("triage", {}) or {}
    return [
        index,
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


def build_history_row(index: int, item: dict[str, Any]) -> list[Any]:
    raw_query = str(item.get("raw_query") or "-").strip() or "-"
    normalized_query = str(item.get("normalized_query") or "").strip()
    intent_text = normalized_query if normalized_query and normalized_query != raw_query else "-"
    final_response = str(item.get("final_response") or "-").strip() or "-"
    return [
        index,
        item.get("request_id") or "-",
        build_request_mode_label(item),
        item.get("status") or "-",
        raw_query,
        intent_text,
        final_response,
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

    valid_statuses = {"active", "blocked", "waiting_user", "cancelled", "completed", "failed", "in_progress", "not_found", "timed_out"}
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
    set_selection_context(
        context,
        {
            "type": "request_list",
            "items": [
                {"index": index, "request_id": item.get("request_id")}
                for index, item in enumerate(summaries, start=1)
            ],
        } if summaries else None,
    )
    if ui and ui.structured_output_enabled:
        if not summaries:
            ui.emit("No matching requests found.")
        else:
            ui.render_recent_requests(heading, [build_recent_request_row(index, item) for index, item in enumerate(summaries, start=1)])
            selection_hint = build_selection_hint("request_list")
            if selection_hint:
                ui.emit(selection_hint)
        return True
    render_recent_request_rows(output_func, summaries, heading)
    if summaries:
        selection_hint = build_selection_hint("request_list")
        if selection_hint:
            output_func(selection_hint)
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


def handle_history(user_input: str, context: dict[str, Any]) -> bool:
    output_func = context["output_func"]
    ui = context.get("ui")
    agent_instance = context["agent_instance"]
    parts = user_input.split()
    limit = 5
    if len(parts) >= 2:
        try:
            limit = max(1, int(parts[1]))
        except ValueError:
            output_func("Usage: /history [limit]")
            return True

    summaries = agent_instance.get_recent_request_summaries(limit=limit)
    set_selection_context(
        context,
        {
            "type": "request_list",
            "items": [{"index": index, "request_id": item.get("request_id")} for index, item in enumerate(summaries, start=1)],
        } if summaries else None,
    )
    if ui and ui.structured_output_enabled:
        if not summaries:
            ui.emit("No matching requests found.")
        else:
            ui.render_conversation_history("Conversation History:", [build_history_row(index, item) for index, item in enumerate(summaries, start=1)])
            selection_hint = build_selection_hint("request_list")
            if selection_hint:
                ui.emit(selection_hint)
        return True

    render_history_rows(output_func, summaries, "Conversation history:")
    if summaries:
        selection_hint = build_selection_hint("request_list")
        if selection_hint:
            output_func(selection_hint)
    return True


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

    if ui and ui.structured_output_enabled:
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
    usage_text = "/failed_requests [limit] [status=failed,waiting_user,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m]"
    return handle_recent_request_query(
        context,
        parts,
        usage_text,
        "Recent failed, waiting, or blocked requests:",
        default_statuses=["failed", "waiting_user", "blocked", "timed_out", "cancelled"],
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

    if ui and ui.structured_output_enabled:
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
    if ui and ui.structured_output_enabled:
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
    if ui and ui.structured_output_enabled:
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
    summary = agent_instance.get_request_summary(agent_instance.get_last_request_id()) if hasattr(agent_instance, "get_request_summary") else None
    if summary:
        summary = dict(summary)
        summary["blocked_reason"] = build_blocked_reason(summary)
        summary["suggested_actions"] = build_next_action_suggestions(summary)
    emit_agent_response(context, response, agent_instance.get_last_request_id(), summary=summary)
    return True


def build_commands() -> tuple[dict[str, CLICommand], list[str]]:
    command_specs = [
        CLICommand("/llm", "/llm [model_key|list] - List configured LLM models, select with /pick, or switch by configured model key; use /llm raw <provider> <model> ... for manual override", handle_llm),
        CLICommand("/load_tool", "/load_tool <tool_name.py> - Load a local python tool", handle_load_tool),
        CLICommand("/grant_write", "/grant_write <folder_path> - Temporarily allow write access to a specific folder", handle_grant_write),
        CLICommand("/revoke_write", "/revoke_write <folder_path> - Revoke a previously granted write folder", handle_revoke_write),
        CLICommand("/list_write_roots", "/list_write_roots - Show the current writable root directories", handle_list_write_roots),
        CLICommand("/load_skill", "/load_skill <skill_name.md> - Load a local markdown skill", handle_load_skill),
        CLICommand("/replay", "/replay <memory_id> [injected features...] - Replay memory with optional injected features", handle_replay),
        CLICommand("/convert_skill", "/convert_skill <memory_id> - Convert memory into a markdown skill", handle_convert_skill),
        CLICommand("/load_mcp", "/load_mcp <config|server.py|stdio:command ...> - Load an MCP server", handle_load_mcp),
        CLICommand("/list_mcp", "/list_mcp - Show loaded MCP servers and transports", handle_list_mcp),
        CLICommand("/refresh_mcp", "/refresh_mcp <server_name|source> - Refresh an MCP server and re-enumerate tools", handle_refresh_mcp),
        CLICommand("/unload_mcp", "/unload_mcp <server_name|source> - Unload an MCP server and remove its tools", handle_unload_mcp),
        CLICommand("/cancel_request", "/cancel_request <request_id> - Request cooperative cancellation for an active run", handle_cancel_request),
        CLICommand("/list_snapshots", "/list_snapshots [request_id] - List available recovery points for a request", handle_list_snapshots),
        CLICommand("/resume_snapshot", "/resume_snapshot <request_id> [snapshot_file] - Resume execution from a saved snapshot", handle_resume_snapshot),
        CLICommand("/request_summary", "/request_summary [request_id] - Show the aggregated status, checkpoints, snapshots, and memories for a request", handle_request_summary),
        CLICommand("/summary", "/summary [request_id] - Alias of /request_summary using the latest request by default", handle_request_summary),
        CLICommand("/history", "/history [limit] - Show a compact view of recent user inputs, rewritten intents, and replies", handle_history),
        CLICommand("/recent_requests", "/recent_requests [limit] [status=failed,waiting_user,blocked,...] [resumed] [attention] [since=30m] - Show recent request summaries and key metrics", handle_recent_requests),
        CLICommand("/recent", "/recent [limit] [status=failed,waiting_user,blocked,...] [resumed] [attention] [since=30m] - Alias of /recent_requests", handle_recent_requests),
        CLICommand("/request_rollup", "/request_rollup [limit] [status=failed,waiting_user,blocked,...] [resumed] [attention] [since=30m] - Show aggregate metrics across recent requests", handle_request_rollup),
        CLICommand("/rollup", "/rollup [limit] [status=failed,waiting_user,blocked,...] [resumed] [attention] [since=30m] - Alias of /request_rollup", handle_request_rollup),
        CLICommand("/failed_requests", "/failed_requests [limit] [status=failed,waiting_user,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m] - Alias of recent request query with failure-oriented defaults", handle_failed_requests),
        CLICommand("/latest_failure", "/latest_failure [summary|snapshots|resume] - Open the most recent failed, waiting, or blocked request with a preset action", handle_latest_failure),
        CLICommand("/resumed_requests", "/resumed_requests [limit] [status=failed,waiting_user,blocked,...] [attention] [since=30m] - Alias of recent request query scoped to resumed requests", handle_resumed_requests),
        CLICommand("/resume_last_blocked", "/resume_last_blocked - Reroute resume the most recent waiting-user or blocked request", handle_resume_last_blocked),
        CLICommand("/list_tool_runs", "/list_tool_runs [request_id] [running|detached] - Show tracked in-flight or detached tool runs", handle_list_tool_runs),
        CLICommand("/detached_tools", "/detached_tools [request_id] - Show detached tools for a request, defaulting to the latest request", handle_detached_tools),
        CLICommand("/suggest", "/suggest [prefix] - Suggest commands, request ids, snapshots, and recent inputs for the given prefix", handle_suggest),
        CLICommand("/recent_commands", "/recent_commands [limit] - Show the most recent CLI inputs for reuse", handle_recent_commands),
        CLICommand("/snapshots", "/snapshots [request_id] - Alias of /list_snapshots using the latest request by default", handle_list_snapshots),
        CLICommand("/resume", "/resume <request_id> [snapshot_file] - Alias of /resume_snapshot", handle_resume_snapshot),
        CLICommand("/failure_memories", "/failure_memories [limit] [keywords...] - Show failure-case memories ranked by relevance", handle_failure_memories),
        CLICommand("/retention_status", "/retention_status - Show retention coverage and reclaimable runtime artifacts", handle_retention_status),
        CLICommand("/prune_runtime_data", "/prune_runtime_data [apply] - Dry-run or apply retention cleanup for logs, snapshots, audit logs, and memory backups", handle_prune_runtime_data),
        CLICommand("/new_session", "/new_session - Start a new memory session", handle_new_session),
        CLICommand("/pick", "/pick <index> [summary|snapshots|resume] - Continue from the most recent selectable list result", handle_pick),
        CLICommand("/selection", "/selection - Show the current selectable list context for /pick", handle_selection),
        CLICommand("/clear_selection", "/clear_selection - Clear the current selectable list context", handle_clear_selection),
    ]
    command_map = {item.name: item for item in command_specs}
    command_map["llm"] = command_map["/llm"]
    command_map["help"] = CLICommand("help", "help - Show available commands", handle_help)
    return command_map, [item.name for item in command_specs]