import importlib
import logging
import os
import tempfile
import unittest

from langchain_core.messages import AIMessage

from config import config


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
                "tool_call_count": 0,
                "retry_count": 0,
                "reflection_failure_count": 0,
            },
            "checkpoints": [
                {"logged_at": "2026-03-31T00:00:00", "stage": "planning_completed", "details": "subtask_count=1"},
                {"logged_at": "2026-03-31T00:00:01", "stage": "request_completed", "details": ""},
            ],
            "memories": [
                {"id": 1, "memory_type": "session_main", "quality_tags": ["success"], "summary": "session summary"},
                {"id": 2, "memory_type": "step", "quality_tags": ["success"], "summary": "step summary"},
            ],
        }
        self._recent = [
            {
                "request_id": "req_recent_2",
                "status": "blocked",
                "session_id": "session_b",
                "updated_at": "2026-03-31T00:00:02+00:00",
                "metrics": {"total_duration_ms": 220.0, "llm_call_count": 2, "tool_call_count": 1},
            },
            {
                "request_id": "req_recent_1",
                "status": "completed",
                "session_id": "session_a",
                "updated_at": "2026-03-31T00:00:01+00:00",
                "metrics": {"total_duration_ms": 120.0, "llm_call_count": 1, "tool_call_count": 0},
            },
        ]

    def start_session(self, session_id=None):
        return session_id or "session_cli"

    def get_request_summary(self, request_id):
        if request_id == "req_cli":
            return self._summary
        return None

    def invoke(self, query, session_id=None):
        self._last_request_id = "req_message"
        return f"echo:{query}:{session_id}"

    def get_recent_request_summaries(self, limit=10):
        return self._recent[:limit]

    def get_last_request_id(self):
        return self._last_request_id

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
    def test_request_summary_command_renders_aggregated_output(self):
        import cli

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
        self.assertIn("Metrics: total_ms=123.0 | llm_calls=1 | tool_calls=0 | retries=0 | reflection_failures=0", rendered)
        self.assertIn("Recent checkpoints:", rendered)
        self.assertIn("Related memories:", rendered)

    def test_plain_message_command_prints_request_id_and_response(self):
        import cli

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
        import cli

        output = FakeCLIOutput()
        cli.start_cli(
            input_func=FakeCLIInput(["/recent_requests 2", "quit"]),
            output_func=output,
            agent_instance=FakeCLIAgent(),
            llm_manager_instance=FakeCLIManager(),
        )

        rendered = output.joined()
        self.assertIn("Recent requests:", rendered)
        self.assertIn("req_recent_2 | status=blocked", rendered)
        self.assertIn("req_recent_1 | status=completed", rendered)

    def test_load_mcp_command_passes_full_reference(self):
        import cli

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
        import cli

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
        import cli

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
        import cli

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

        import llm_manager
        import agent_core

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
        self.agent.planner.split_task = lambda text: [
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
        self._configure_successful_flow()
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (False, "missing parameter", "ask_user")

        result = self.agent.invoke("book something")
        request_id = self.agent.get_last_request_id()
        summary = self.agent.get_request_summary(request_id)

        self.assertIn("Need user intervention.", result)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "blocked")
        self.assertTrue(summary["blocked"])
        self.assertIn("Need user intervention.", summary["final_response"])
        self.assertEqual(summary["metrics"]["reflection_failure_count"], 1)
        self.assertEqual(summary["metrics"]["blocked_rate"], 1.0)

    def test_recent_request_summaries_returns_latest_requests_first(self):
        self._configure_successful_flow()

        first_result = self.agent.invoke("first request")
        self.assertEqual(first_result, "weather is sunny")

        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (False, "missing parameter", "ask_user")
        second_result = self.agent.invoke("second request")
        self.assertIn("Need user intervention.", second_result)

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

        def fake_split_task(text):
            if "Replan the failed subtask" in text:
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


if __name__ == "__main__":
    unittest.main()