import importlib
import logging
import os
import tempfile
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from core.config import config


class FakeCLIOutput:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(str(message))

    def joined(self):
        return "\n".join(self.lines)


class FakeCLIInput:
    def __init__(self, commands):
        self._commands = iter(commands)

    def __call__(self, prompt=""):
        return next(self._commands)


class FakeCLIManager:
    def set_model(self, provider, model, base_url=None, api_key=None):
        return f"switched:{provider}:{model}"


class FakeCLIAgent:
    def __init__(self):
        self._last_request_id = "req_cli"
        self.loaded_mcp_refs = []
        self.unloaded_mcp_refs = []
        self.refreshed_mcp_refs = []
        self._mcp_servers = [
            {
                "name": "system_mcp_server",
                "transport": "stdio",
                "tool_names": ["get_system_info", "inspect_file_system_path"],
                "source": "mcp_servers/system_mcp_server.py",
            }
        ]
        self._summary = {
            "request_id": "req_cli",
            "status": "completed",
            "session_id": "session_cli",
            "latest_stage": "request_completed",
            "subtask_index": 1,
            "plan_length": 1,
            "snapshot_count": 3,
            "memory_count": 2,
            "checkpoint_count": 2,
            "source_request_id": "",
            "final_response": "done",
            "metrics": {
                "total_duration_ms": 123.0,
                "llm_call_count": 1,
                "stage_duration_ms": {"planning": 12.5, "subtask": 7.25, "reflection": 3.0},
                "tool_call_count": 0,
                "tool_detached_count": 1,
                "retry_count": 0,
                "reflection_failure_count": 0,
            },
            "triage": {
                "needs_attention": True,
                "is_resumed": False,
                "latest_failure_stage": "tool_detached",
                "latest_failure_details": "tool=slow_tool | reason=timeout | tool_run_id=toolrun_000001",
                "latest_failure_source": "tool_runtime",
                "latest_reroute_mode": "fallback_high_risk_history",
                "latest_reroute_details": "index=1 | mode=fallback_high_risk_history | failed_tools=slow_tool",
                "used_no_tool_fallback": True,
                "last_error_message": "Checkpoint tool_detached",
                "last_error_event_type": "",
                "tool_attention_count": 0,
                "detached_tool_count": 1,
                "has_detached_tools": True,
            },
            "checkpoints": [
                {"logged_at": "2026-03-31T00:00:00", "stage": "planning_completed", "details": "subtask_count=1"},
                {"logged_at": "2026-03-31T00:00:01", "stage": "request_completed", "details": ""},
            ],
            "detached_tools": [
                {"tool_name": "slow_tool", "tool_run_id": "toolrun_000001", "reason": "timeout", "source": "checkpoint"},
            ],
            "memories": [
                {"id": 1, "memory_type": "session_main", "quality_tags": ["success"], "summary": "session summary"},
                {"id": 2, "memory_type": "step", "quality_tags": ["success"], "summary": "step summary"},
            ],
        }
        self._recent = [
            {
                "request_id": "req_recent_3",
                "status": "failed",
                "session_id": "session_c",
                "source_request_id": "req_original_resume",
                "latest_stage": "request_failed",
                "updated_at": "2026-03-31T00:00:03+00:00",
                "metrics": {"total_duration_ms": 330.0, "llm_call_count": 3, "tool_call_count": 1, "tool_detached_count": 1},
                "triage": {
                    "needs_attention": True,
                    "latest_failure_stage": "tool_detached",
                    "latest_failure_details": "tool=slow_tool | reason=timeout | tool_run_id=toolrun_000001",
                    "latest_failure_source": "tool_runtime",
                    "latest_reroute_mode": "fallback_high_risk_history",
                    "latest_reroute_details": "index=1 | mode=fallback_high_risk_history | failed_tools=slow_tool",
                    "used_no_tool_fallback": True,
                    "tool_attention_count": 0,
                    "detached_tool_count": 1,
                },
            },
            {
                "request_id": "req_recent_2",
                "status": "blocked",
                "session_id": "session_b",
                "source_request_id": "",
                "latest_stage": "agent_blocked",
                "updated_at": "2026-03-31T00:00:02+00:00",
                "metrics": {"total_duration_ms": 220.0, "llm_call_count": 2, "tool_call_count": 1, "tool_detached_count": 0},
                "triage": {
                    "needs_attention": True,
                    "latest_failure_stage": "agent_blocked",
                    "latest_failure_details": "index=1 | action=retry_limit | tool=weather_tool | error_type=timeout",
                    "latest_failure_source": "reflection",
                    "latest_reroute_mode": "",
                    "latest_reroute_details": "",
                    "used_no_tool_fallback": False,
                    "tool_attention_count": 1,
                    "detached_tool_count": 0,
                },
            },
            {
                "request_id": "req_recent_1",
                "status": "completed",
                "session_id": "session_a",
                "source_request_id": "",
                "latest_stage": "request_completed",
                "updated_at": "2026-03-31T00:00:01+00:00",
                "metrics": {"total_duration_ms": 120.0, "llm_call_count": 1, "tool_call_count": 0, "tool_detached_count": 0},
                "triage": {"needs_attention": False, "latest_failure_stage": "", "tool_attention_count": 0, "detached_tool_count": 0},
            },
        ]
        self._tool_runs = [
            {
                "tool_run_id": "toolrun_000001",
                "tool_name": "slow_tool",
                "request_id": "req_cli",
                "status": "detached",
                "reason": "timeout",
                "detached_runtime_ms": 120.0,
            },
            {
                "tool_run_id": "toolrun_000002",
                "tool_name": "weather_tool",
                "request_id": "req_other",
                "status": "running",
                "reason": "",
                "runtime_ms": 40.0,
            },
        ]
        self._failure_memories = [
            {
                "id": 9,
                "request_id": "req_blocked_memory",
                "memory_type": "failure_case",
                "quality_tags": ["blocked", "ask_user"],
                "keywords": ["booking", "time"],
                "summary": "Step: booking failure",
            }
        ]
        self._rollup = {
            "request_count": 3,
            "latest_updated_at": "2026-03-31T00:00:03+00:00",
            "filters": {"statuses": [], "resumed_only": False, "attention_only": False, "since_seconds": None},
            "status_counts": {"failed": 1, "blocked": 1, "completed": 1},
            "resumed_count": 1,
            "needs_attention_count": 2,
            "active_count": 0,
            "average_total_duration_ms": 223.33,
            "stage_duration_ms_total": {"planning": 20.0, "tool": 30.0, "reflection": 12.0},
            "source_bucket_breakdown": [
                {"source": "tool_runtime", "count": 1, "share": 0.5},
                {"source": "reflection", "count": 1, "share": 0.5},
            ],
            "source_bucket_trends": [
                {"source": "tool_runtime", "recent_count": 1, "earlier_count": 0, "recent_share": 1.0, "earlier_share": 0.0, "delta_share": 1.0, "direction": "up"},
                {"source": "reflection", "recent_count": 0, "earlier_count": 1, "recent_share": 0.0, "earlier_share": 1.0, "delta_share": -1.0, "direction": "down"},
            ],
            "top_failure_signals": {
                "stages": [("tool_detached", 1)],
                "tools": [("slow_tool", 1)],
                "reasons": [("timeout", 1)],
                "error_types": [("timeout", 1)],
                "actions": [("retry_limit", 1)],
                "sources": [("tool_runtime", 1)],
                "reroute_modes": [("fallback_high_risk_history", 1)],
                "no_tool_fallbacks": [("used", 1)],
            },
            "top_failure_combinations": {
                "stage_tool": [("tool_detached+slow_tool", 1)],
                "tool_reason": [("slow_tool+timeout", 1)],
                "stage_source": [("tool_detached+tool_runtime", 1)],
                "stage_reroute": [("tool_detached+fallback_high_risk_history", 1)],
            },
            "totals": {
                "llm_call_count": 6,
                "tool_call_count": 2,
                "tool_detached_count": 1,
                "retry_count": 2,
                "reflection_failure_count": 1,
            },
        }
        self._retention_status = {
            "generated_at": "2026-03-31T00:00:05+00:00",
            "last_auto_prune": {
                "trigger": "snapshot",
                "executed_at": "2026-03-31T00:00:04+00:00",
                "deleted_count": 2,
                "expired_count": 2,
                "reclaimable_bytes": 3072,
            },
            "last_auto_prune_check": {
                "status": "skipped_throttled",
                "reason": "throttled",
                "trigger": "snapshot",
                "checked_at": "2026-03-31T00:00:05+00:00",
            },
            "targets": [
                {"key": "logs", "retention_days": 7, "max_items": 20, "max_total_bytes": 5120, "item_count": 2, "expired_count": 1, "total_bytes": 4096, "reclaimable_bytes": 1024},
                {"key": "snapshots", "retention_days": 7, "max_items": 200, "max_total_bytes": 10240, "item_count": 3, "expired_count": 1, "total_bytes": 8192, "reclaimable_bytes": 2048},
            ],
            "totals": {"item_count": 5, "expired_count": 2, "total_bytes": 12288, "reclaimable_bytes": 3072},
        }
        self._prune_payload = {
            "mode": "dry_run",
            "targets": [
                {"key": "logs", "expired_count": 1, "reclaimable_bytes": 1024, "deleted_count": 0},
                {"key": "snapshots", "expired_count": 1, "reclaimable_bytes": 2048, "deleted_count": 0},
            ],
            "totals": {"expired_count": 2, "reclaimable_bytes": 3072, "deleted_count": 0},
        }

    def start_session(self, session_id=None):
        return session_id or "session_cli"

    def get_request_summary(self, request_id):
        if request_id == "req_cli":
            return self._summary
        return None

    def invoke(self, query, session_id=None):
        self._last_request_id = "req_message"
        return f"echo:{query}:{session_id}"

    def resume_from_snapshot(self, request_id, snapshot_name=None, reroute=False):
        self._last_request_id = "req_resume"
        mode = "reroute" if reroute else "continue"
        return f"resumed:{request_id}:{snapshot_name}:{mode}"

    def get_recent_request_summaries(self, limit=10, statuses=None, resumed_only=False, attention_only=False, since_seconds=None):
        items = list(self._recent)
        if statuses:
            allowed = {item.lower() for item in statuses}
            items = [item for item in items if item.get("status", "").lower() in allowed]
        if resumed_only:
            items = [item for item in items if item.get("source_request_id")]
        if attention_only:
            items = [item for item in items if item.get("status") in {"failed", "blocked", "timed_out", "cancelled"} or item.get("metrics", {}).get("tool_detached_count", 0)]
        return items[:limit]

    def get_last_request_id(self):
        return self._last_request_id

    @property
    def observability(self):
        class _Obs:
            @staticmethod
            def ordered_stage_duration_items(stage_duration_ms):
                preferred_order = ["planning", "subtask", "tool", "reflection", "request", "resume", "other"]
                ordered = []
                for key in preferred_order:
                    if key in stage_duration_ms:
                        ordered.append((key, stage_duration_ms[key]))
                for key, value in stage_duration_ms.items():
                    if key not in preferred_order:
                        ordered.append((key, value))
                return ordered

        return _Obs()

    def list_tool_runs(self, request_id="", status=""):
        items = list(self._tool_runs)
        if request_id:
            items = [item for item in items if item.get("request_id") == request_id]
        if status:
            items = [item for item in items if item.get("status") == status]
        return items

    def get_failure_memories(self, match_keywords=None, limit=5, exclude_conv_id=None, exclude_ids=None):
        items = list(self._failure_memories)
        keywords = [item.lower() for item in (match_keywords or []) if isinstance(item, str)]
        if keywords:
            items = [
                item for item in items
                if any(keyword in [kw.lower() for kw in item.get("keywords", [])] for keyword in keywords)
            ]
        return items[:limit]

    def get_request_rollup(self, limit=20, statuses=None, resumed_only=False, attention_only=False, since_seconds=None):
        payload = dict(self._rollup)
        payload["filters"] = {
            "statuses": list(statuses or []),
            "resumed_only": resumed_only,
            "attention_only": attention_only,
            "since_seconds": since_seconds,
        }
        return payload

    def get_retention_status(self):
        return dict(self._retention_status)

    def prune_runtime_data(self, apply=False):
        payload = dict(self._prune_payload)
        if apply:
            payload["mode"] = "apply"
            payload["targets"] = [
                {"key": "logs", "expired_count": 1, "reclaimable_bytes": 1024, "deleted_count": 1},
                {"key": "snapshots", "expired_count": 1, "reclaimable_bytes": 2048, "deleted_count": 1},
            ]
            payload["totals"] = {"expired_count": 2, "reclaimable_bytes": 3072, "deleted_count": 2}
        return payload

    def is_request_active(self, request_id):
        return False

    def load_mcp_server(self, server_ref):
        self.loaded_mcp_refs.append(server_ref)
        return True, f"loaded_mcp:{server_ref}"

    def list_mcp_servers(self):
        return list(self._mcp_servers)

    def unload_mcp_server(self, server_ref):
        self.unloaded_mcp_refs.append(server_ref)
        return True, f"unloaded_mcp:{server_ref}"

    def refresh_mcp_server(self, server_ref):
        self.refreshed_mcp_refs.append(server_ref)
        return True, f"refreshed_mcp:{server_ref}"


class CLITestCases(unittest.TestCase):
    def test_help_command_lists_registered_commands(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["help", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Available commands:", rendered)
        self.assertIn("/request_summary <request_id>", rendered)
        self.assertIn("/recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]", rendered)
        self.assertIn("/request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]", rendered)
        self.assertIn("/failed_requests [limit] [status=failed,blocked,timed_out,cancelled,...] [resumed] [attention] [since=30m]", rendered)
        self.assertIn("/resumed_requests [limit] [status=failed,blocked,...] [attention] [since=30m]", rendered)
        self.assertIn("/list_tool_runs [request_id] [running|detached]", rendered)
        self.assertIn("/failure_memories [limit] [keywords...] - Show failure-case memories ranked by relevance", rendered)
        self.assertIn("/retention_status - Show retention coverage and reclaimable runtime artifacts", rendered)
        self.assertIn("/prune_runtime_data [apply] - Dry-run or apply retention cleanup for logs, snapshots, audit logs, and memory backups", rendered)
        self.assertIn("/new_session - Start a new memory session", rendered)

    def test_request_summary_command_renders_aggregated_output(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/request_summary req_cli", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Request ID: req_cli", rendered)
        self.assertIn("Status: completed", rendered)
        self.assertIn("Metrics: total_ms=123.0 | llm_calls=1 | tool_calls=0 | tool_detached=1 | retries=0 | reflection_failures=0", rendered)
        self.assertIn("Stage durations: planning=12.5 | subtask=7.25 | reflection=3.0", rendered)
        self.assertIn("Triage: needs_attention=True | resumed=False | failure_stage=tool_detached | failure_source=tool_runtime | reroute=fallback_high_risk_history | tool_attention=0 | detached_tools=1", rendered)
        self.assertIn("Reroute details: index=1 | mode=fallback_high_risk_history | failed_tools=slow_tool", rendered)
        self.assertIn("Recent checkpoints:", rendered)
        self.assertIn("Related memories:", rendered)
        self.assertIn("Detached tools:", rendered)
        self.assertIn("slow_tool | run_id=toolrun_000001 | reason=timeout | source=checkpoint", rendered)

    def test_plain_message_command_prints_request_id_and_response(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["hello agent", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Request ID: req_message", rendered)
        self.assertIn("Response: echo:hello agent:session_cli", rendered)

    def test_recent_requests_command_renders_recent_request_rows(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/recent_requests 2", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Recent requests:", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertIn("detached_tools=1", rendered)
        self.assertIn("resumed_from=req_original_resume", rendered)
        self.assertIn("attention=stage=tool_detached,detached=1,tool=slow_tool,reason=timeout,reroute=fallback_high_risk_history,source=tool_runtime", rendered)
        self.assertIn("req_recent_2 | status=blocked", rendered)
        self.assertIn("attention=stage=agent_blocked,tool_attention=1,tool=weather_tool,error_type=timeout,action=retry_limit", rendered)

    def test_recent_requests_command_accepts_filters(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/recent_requests 5 status=failed resumed attention", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: statuses=failed | resumed_only=True | attention_only=True", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertNotIn("req_recent_2 | status=blocked", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertNotIn("req_recent_2 | status=blocked", rendered)

    def test_recent_requests_command_accepts_since_filter(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/recent_requests 5 since=30m", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: since=30m", rendered)

    def test_recent_requests_command_rejects_invalid_filters(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/recent_requests 5 status=bad_status", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Usage: /recent_requests [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]", rendered)

    def test_request_rollup_command_renders_global_aggregate(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/request_rollup 5", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Request rollup: count=3 | attention=2 | resumed=1 | active=0 | avg_total_ms=223.33", rendered)
        self.assertIn("Statuses: failed=1 | blocked=1 | completed=1", rendered)
        self.assertIn("Totals: llm_calls=6 | tool_calls=2 | tool_detached=1 | retries=2 | reflection_failures=1", rendered)
        self.assertIn("Top failures: stage=tool_detached(1) | tool=slow_tool(1) | reason=timeout(1) | error_type=timeout(1) | action=retry_limit(1) | source=tool_runtime(1) | reroute=fallback_high_risk_history(1) | no_tool_fallback=used(1)", rendered)
        self.assertIn("Failure combos: stage+tool=tool_detached+slow_tool(1) | tool+reason=slow_tool+timeout(1) | stage+source=tool_detached+tool_runtime(1) | stage+reroute=tool_detached+fallback_high_risk_history(1)", rendered)
        self.assertIn("Source buckets: tool_runtime=1(50.0%) | reflection=1(50.0%)", rendered)
        self.assertIn("Source trends: tool_runtime=up(+100.0pp,recent=1,earlier=0) | reflection=down(-100.0pp,recent=0,earlier=1)", rendered)
        self.assertIn("Stage totals: planning=20.0 | tool=30.0 | reflection=12.0", rendered)

    def test_request_rollup_command_accepts_filters(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/request_rollup 5 status=failed,blocked resumed attention", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: statuses=failed,blocked | resumed_only=True | attention_only=True", rendered)

    def test_request_rollup_command_accepts_since_filter(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/request_rollup 5 since=2h", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: since=2h", rendered)

    def test_request_rollup_command_rejects_invalid_filters(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/request_rollup 5 status=bad_status", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Usage: /request_rollup [limit] [status=failed,blocked,...] [resumed] [attention] [since=30m]", rendered)

    def test_retention_status_command_renders_summary(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/retention_status", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Retention status: items=5 | expired=2 | total=12.0 KB | reclaimable=3.0 KB", rendered)
        self.assertIn("Last auto prune: trigger=snapshot | deleted=2 | expired=2 | reclaimable=3.0 KB | at=2026-03-31T00:00:04+00:00", rendered)
        self.assertIn("Last auto prune check: status=skipped_throttled | reason=throttled | trigger=snapshot | checked_at=2026-03-31T00:00:05+00:00", rendered)
        self.assertIn("logs | days=7 | max=20 | max_bytes=5.0 KB | items=2 | expired=1 | total=4.0 KB | reclaimable=1.0 KB", rendered)
        self.assertIn("snapshots | days=7 | max=200 | max_bytes=10.0 KB | items=3 | expired=1 | total=8.0 KB | reclaimable=2.0 KB", rendered)

    def test_prune_runtime_data_command_supports_dry_run_and_apply(self):
        from app.cli import main as cli
        dry_run_output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/prune_runtime_data", "quit"]),
            output_func=dry_run_output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = dry_run_output.joined()
        self.assertIn("Prune runtime data: mode=dry_run | expired=2 | reclaimable=3.0 KB | deleted=0", rendered)
        self.assertIn("Use /prune_runtime_data apply to remove expired artifacts.", rendered)

        apply_output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/prune_runtime_data apply", "quit"]),
            output_func=apply_output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered_apply = apply_output.joined()
        self.assertIn("Prune runtime data: mode=apply | expired=2 | reclaimable=3.0 KB | deleted=2", rendered_apply)

    def test_failed_requests_command_filters_failed_statuses(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/failed_requests 5", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Recent failed or blocked requests:", rendered)
        self.assertIn("Filters: statuses=failed,blocked,timed_out,cancelled", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertIn("req_recent_2 | status=blocked", rendered)
        self.assertNotIn("req_recent_1 | status=completed", rendered)

    def test_resumed_requests_command_filters_resumed_runs(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/resumed_requests 5", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Recent resumed requests:", rendered)
        self.assertIn("Filters: resumed_only=True", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertIn("resumed_from=req_original_resume", rendered)
        self.assertNotIn("req_recent_2 | status=blocked", rendered)

    def test_failed_requests_command_allows_extra_filters(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/failed_requests 5 status=failed attention", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: statuses=failed | attention_only=True", rendered)
        self.assertIn("req_recent_3 | status=failed", rendered)
        self.assertNotIn("req_recent_2 | status=blocked", rendered)

    def test_resumed_requests_command_allows_status_filter(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/resumed_requests 5 status=failed attention", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Filters: statuses=failed | resumed_only=True | attention_only=True", rendered)

    def test_resume_snapshot_command_accepts_reroute_mode(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/resume_snapshot req_old planning_completed reroute", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Request ID: req_resume", rendered)
        self.assertIn("Response: resumed:req_old:planning_completed:reroute", rendered)

    def test_list_tool_runs_command_renders_filtered_rows(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/list_tool_runs req_cli detached", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Tracked tool runs:", rendered)
        self.assertIn("toolrun_000001 | tool=slow_tool | request=req_cli | status=detached | reason=timeout", rendered)
        self.assertNotIn("toolrun_000002", rendered)

    def test_failure_memories_command_renders_ranked_results(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/failure_memories 5 booking", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Failure memories: keywords=booking", rendered)
        self.assertIn("#9 | request=req_blocked_memory | type=failure_case | tags=blocked,ask_user | keywords=booking,time | summary=Step: booking failure", rendered)

    def test_load_mcp_command_passes_full_reference(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        fake_agent = FakeCLIAgent()
        cli.start_cli(
            input_func=FakeCLIInput(["/load_mcp stdio:python mcp_servers/system_mcp_server.py", "quit"]),
            output_func=output,
            agent_instance=fake_agent,
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("loaded_mcp:stdio:python mcp_servers/system_mcp_server.py", rendered)
        self.assertEqual(fake_agent.loaded_mcp_refs, ["stdio:python mcp_servers/system_mcp_server.py"])

    def test_list_mcp_command_renders_loaded_servers(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        fake_agent = FakeCLIAgent()
        cli.start_cli(
            input_func=FakeCLIInput(["/list_mcp", "quit"]),
            output_func=output,
            agent_instance=fake_agent,
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Loaded MCP servers:", rendered)
        self.assertIn("system_mcp_server | transport=stdio | tools=2", rendered)

    def test_unload_mcp_command_passes_reference(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        fake_agent = FakeCLIAgent()
        cli.start_cli(
            input_func=FakeCLIInput(["/unload_mcp system_mcp_server", "quit"]),
            output_func=output,
            agent_instance=fake_agent,
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("unloaded_mcp:system_mcp_server", rendered)
        self.assertEqual(fake_agent.unloaded_mcp_refs, ["system_mcp_server"])

    def test_refresh_mcp_command_passes_reference(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        fake_agent = FakeCLIAgent()
        cli.start_cli(
            input_func=FakeCLIInput(["/refresh_mcp system_mcp_server", "quit"]),
            output_func=output,
            agent_instance=fake_agent,
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("refreshed_mcp:system_mcp_server", rendered)
        self.assertEqual(fake_agent.refreshed_mcp_refs, ["system_mcp_server"])

    def test_new_session_command_updates_session_for_following_message(self):
        from app.cli import main as cli
        output = FakeCLIOutput()
        fake_agent = FakeCLIAgent()
        fake_agent.start_session = lambda session_id=None: session_id or "session_new_cli"
        cli.start_cli(
            input_func=FakeCLIInput(["/new_session", "hello agent", "quit"]),
            output_func=output,
            agent_instance=fake_agent,
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Started new session: session_new_cli", rendered)
        self.assertIn("Response: echo:hello agent:session_new_cli", rendered)


class AgentIntegrationFlowTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_log_dir = config.log_dir
        self.original_llm_log_file = config.llm_log_file
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.log_dir = os.path.join(self.tempdir.name, "logs")
        config.llm_log_file = "cli_integration.log"
        from core.llm import manager as llm_manager
        from app.agent import core as agent_core
        self.llm_manager_module = importlib.reload(llm_manager)
        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        logger = logging.getLogger("llm_brain.llm")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        config.log_dir = self.original_log_dir
        config.llm_log_file = self.original_llm_log_file
        self.tempdir.cleanup()

    def _configure_successful_flow(self):
        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                return AIMessage(content="weather is sunny")

        self.agent.cognitive.extract_features = lambda text: (["weather", "beijing"], "weather summary")
        self.agent.cognitive.determine_domain = lambda text: "general"
        self.agent.planner.split_task = lambda text, thinking_mode=False: [
            {"id": 1, "description": "check weather in beijing", "expected_outcome": "provide the weather"}
        ]
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (True, "verified", "continue")
        self.agent.skills.assign_capabilities_to_task = lambda desc, kws: {
            "prompt_skill": None,
            "prompt_skill_reason": "",
            "tool_skills": [],
            "tool_reasons": [],
            "tools": [],
        }
        self.llm_manager_module.llm_manager.get_llm = lambda: FakeLLM()

    def test_invoke_runs_full_graph_and_records_summary(self):
        self._configure_successful_flow()

        result = self.agent.invoke("what is the weather in beijing?")
        request_id = self.agent.get_last_request_id()
        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(result, "weather is sunny")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["latest_stage"], "request_completed")
        self.assertGreaterEqual(summary["snapshot_count"], 5)
        self.assertGreaterEqual(summary["memory_count"], 2)
        self.assertEqual(summary["final_response"], "weather is sunny")
        self.assertEqual(summary["metrics"]["llm_call_count"], 1)
        self.assertEqual(summary["metrics"]["tool_call_count"], 0)
        self.assertEqual(summary["metrics"]["subtask_count"], 1)
        self.assertEqual(summary["metrics"]["reflection_failure_count"], 0)

    def test_invoke_blocked_path_records_blocked_summary(self):
        class ClarifyingLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                return AIMessage(content="Please confirm whether to query local or remote host.")

        self._configure_successful_flow()
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (False, "missing parameter", "ask_user")
        self.llm_manager_module.llm_manager.get_llm = lambda: ClarifyingLLM()

        result = self.agent.invoke("book something")
        request_id = self.agent.get_last_request_id()
        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(result, "Please confirm whether to query local or remote host.")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "blocked")
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["final_response"], "Please confirm whether to query local or remote host.")
        self.assertEqual(summary["metrics"]["reflection_failure_count"], 1)
        self.assertEqual(summary["metrics"]["blocked_rate"], 1.0)

    def test_recent_request_summaries_returns_latest_requests_first(self):
        self._configure_successful_flow()

        first_result = self.agent.invoke("first request")
        self.assertEqual(first_result, "weather is sunny")

        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (False, "missing parameter", "ask_user")
        self.llm_manager_module.llm_manager.get_llm = lambda: type(
            "ClarifyingLLM",
            (),
            {
                "bind_tools": lambda self, tools: self,
                "invoke": lambda self, payload: AIMessage(content="Need more booking details."),
            },
        )()
        second_result = self.agent.invoke("second request")
        self.assertEqual(second_result, "Need more booking details.")

        recent = self.agent.get_recent_request_summaries(limit=2)

        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["status"], "blocked")
        self.assertEqual(recent[1]["status"], "completed")

    def test_retry_path_can_replan_subtask_and_complete(self):
        class SequencedLLM:
            def __init__(self):
                self.responses = iter([
                    "first attempt failed",
                    "collected missing input",
                    "final stable result",
                ])

            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                return AIMessage(content=next(self.responses))

        def fake_split_task(text, thinking_mode=False):
            lowered = text.lower()
            if "replan the failed subtask" in lowered or "replan" in lowered or "重新规划" in text:
                return [
                    {"id": 1, "description": "collect missing input", "expected_outcome": "required input collected"},
                    {"id": 2, "description": "execute safer action", "expected_outcome": "stable result produced"},
                ]
            return [
                {"id": 1, "description": "perform unstable action", "expected_outcome": "stable result produced"}
            ]

        def fake_reflect(desc, expected, actual):
            if desc == "perform unstable action":
                return False, "initial approach was unstable", "retry"
            return True, "verified", "continue"

        self.agent.cognitive.extract_features = lambda text: (["retry", "replan"], "retry summary")
        self.agent.cognitive.determine_domain = lambda text: "general"
        self.agent.planner.split_task = fake_split_task
        self.agent.reflector.verify_and_reflect = fake_reflect
        self.agent.skills.assign_capabilities_to_task = lambda desc, kws: {
            "prompt_skill": None,
            "prompt_skill_reason": "",
            "tool_skills": [],
            "tool_reasons": [],
            "tools": [],
        }
        llm = SequencedLLM()
        self.llm_manager_module.llm_manager.get_llm = lambda: llm

        result = self.agent.invoke("do the unstable thing safely")
        request_id = self.agent.get_last_request_id()
        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(result, "final stable result")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["metrics"]["reflection_failure_count"], 1)
        self.assertTrue(any(item.get("stage") == "subtask_replanned" for item in summary["checkpoints"]))

    def test_tool_call_continuation_does_not_append_human_message_after_tool_result(self):
        @tool
        def get_hostname() -> str:
            """Get the local hostname."""
            return "thething"

        class ToolCallingLLM:
            def __init__(self):
                self.calls = []

            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    self.test_case.assertIsInstance(payload[-1], HumanMessage)
                    return AIMessage(
                        content="",
                        tool_calls=[{"name": "get_hostname", "args": {}, "id": "call_1", "type": "tool_call"}],
                    )

                self.test_case.assertIsInstance(payload[-1], ToolMessage)
                self.test_case.assertNotIsInstance(payload[-1], HumanMessage)
                return AIMessage(content="hostname is thething")

        self.agent.add_tool(get_hostname)
        llm = ToolCallingLLM()
        llm.test_case = self
        self.agent.cognitive.extract_features = lambda text: (["hostname", "name"], "hostname summary")
        self.agent.cognitive.determine_domain = lambda text: "computer"
        self.agent.planner.split_task = lambda text, thinking_mode=False: [
            {"id": 1, "description": "query hostname", "expected_outcome": "return hostname"}
        ]
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (True, "verified", "continue")
        self.agent.skills.assign_capabilities_to_task = lambda desc, kws: {
            "prompt_skill": None,
            "prompt_skill_reason": "",
            "tool_skills": [{"name": "get_hostname", "tool": next(tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_hostname"), "description": "Get the local hostname.", "route_reason": "matched hostname", "overlap_count": 2, "match_ratio": 1.0}],
            "tool_reasons": ["matched hostname"],
            "tools": [next(tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_hostname")],
        }
        self.llm_manager_module.llm_manager.get_llm = lambda: llm

        result = self.agent.invoke("query hostname")

        self.assertEqual(result, "hostname is thething")
        self.assertEqual(len(llm.calls), 2)

    def test_empty_llm_response_with_single_selected_tool_synthesizes_tool_call(self):
        @tool
        def get_hostname() -> str:
            """Get the local hostname."""
            return "thething"

        class EmptyThenAnswerLLM:
            def __init__(self):
                self.calls = []

            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                self.calls.append(payload)
                if len(self.calls) == 1:
                    self.test_case.assertIsInstance(payload[-1], HumanMessage)
                    return AIMessage(content="")

                self.test_case.assertIsInstance(payload[-1], ToolMessage)
                return AIMessage(content="hostname is thething")

        self.agent.add_tool(get_hostname)
        llm = EmptyThenAnswerLLM()
        llm.test_case = self
        selected_tool = next(tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_hostname")
        self.agent.cognitive.extract_features = lambda text: (["hostname", "name"], "hostname summary")
        self.agent.cognitive.determine_domain = lambda text: "computer"
        self.agent.planner.split_task = lambda text, thinking_mode=False: [
            {"id": 1, "description": "query hostname", "expected_outcome": "return hostname"}
        ]
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (True, "verified", "continue")
        self.agent.skills.assign_capabilities_to_task = lambda desc, kws: {
            "prompt_skill": None,
            "prompt_skill_reason": "",
            "tool_skills": [{"name": "get_hostname", "tool": selected_tool, "description": "Get the local hostname.", "route_reason": "matched hostname", "overlap_count": 2, "match_ratio": 1.0}],
            "tool_reasons": ["matched hostname"],
            "tools": [selected_tool],
        }
        self.llm_manager_module.llm_manager.get_llm = lambda: llm

        result = self.agent.invoke("query hostname")

        self.assertEqual(result, "hostname is thething")
        self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()


