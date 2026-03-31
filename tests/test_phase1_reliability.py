import copy
import importlib
import json
import os
import tempfile
import time
import unittest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from config import config
from cognitive.structured_output import (
    StructuredOutputFormatError,
    StructuredOutputSchemaError,
    parse_json_array,
    parse_json_object,
)


class StructuredOutputParserTests(unittest.TestCase):
    def test_parse_json_object_from_code_fence(self):
        payload = """```json
        {
          "keywords": ["agent", "memory"],
          "summary": "A task summary"
        }
        ```"""

        parsed = parse_json_object(payload, required_fields={"keywords": list, "summary": str})

        self.assertEqual(parsed["keywords"], ["agent", "memory"])
        self.assertEqual(parsed["summary"], "A task summary")

    def test_parse_json_array_from_mixed_text(self):
        payload = 'Plan:\n[{"id": 1, "description": "step", "expected_outcome": "done"}]'

        parsed = parse_json_array(payload)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["description"], "step")

    def test_parse_json_object_rejects_missing_field(self):
        with self.assertRaises(StructuredOutputSchemaError):
            parse_json_object('{"keywords": ["only"]}', required_fields={"keywords": list, "summary": str})

    def test_parse_json_array_rejects_missing_json(self):
        with self.assertRaises(StructuredOutputFormatError):
            parse_json_array("not a json payload")


class AgentStateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_persist_state_snapshot_creates_json_file(self):
        state = {
            "messages": [HumanMessage(content="hello world")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "request_id": "req_unit_test",
            "session_id": "session_unit_test",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }

        snapshot_path = self.agent._persist_state_snapshot(
            "req_unit_test",
            "request_received",
            state,
            extra={"query": "hello world"},
        )

        self.assertTrue(os.path.exists(snapshot_path))
        with open(snapshot_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        self.assertEqual(payload["request_id"], "req_unit_test")
        self.assertEqual(payload["stage"], "request_received")
        self.assertEqual(payload["state"]["session_id"], "session_unit_test")
        self.assertEqual(payload["state"]["messages"][0]["content"], "hello world")
        self.assertEqual(payload["extra"]["query"], "hello world")


class ToolSafetyWrapperTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_tool_wrapper_returns_structured_error_for_invalid_arguments(self):
        @tool
        def invalid_tool(value: int) -> str:
            """Tool that fails with argument issues."""
            raise ValueError("bad argument")

        safe_tool = self.agent._wrap_tool_for_runtime(invalid_tool)

        result = safe_tool.invoke({"value": 1})
        payload = json.loads(result)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertFalse(payload["retryable"])

    def test_tool_wrapper_returns_structured_error_for_runtime_failure(self):
        @tool
        def unstable_tool() -> str:
            """Tool that fails at runtime."""
            raise RuntimeError("boom")

        safe_tool = self.agent._wrap_tool_for_runtime(unstable_tool)

        result = safe_tool.invoke({})
        payload = json.loads(result)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "execution_error")
        self.assertTrue(payload["retryable"])

    def test_tool_wrapper_returns_timeout_error(self):
        @tool
        def slow_tool() -> str:
            """Tool that exceeds timeout."""
            time.sleep(0.2)
            return "done"

        original_timeout = config.tool_timeout_seconds
        config.tool_timeout_seconds = 0.01
        try:
            safe_tool = self.agent._wrap_tool_for_runtime(slow_tool)
            result = safe_tool.invoke({})
            payload = json.loads(result)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "timeout")
            self.assertTrue(payload["retryable"])
        finally:
            config.tool_timeout_seconds = original_timeout


class TimeoutControlTests(unittest.TestCase):
    def test_llm_manager_timeout_raises_timeout_error(self):
        from llm_manager import llm_manager

        class SlowLLM:
            def invoke(self, payload):
                time.sleep(0.2)
                return payload

        original_timeout = config.llm_timeout_seconds
        config.llm_timeout_seconds = 0.01
        try:
            with self.assertRaises(TimeoutError):
                llm_manager.invoke("hello", source="test.timeout", llm=SlowLLM())
        finally:
            config.llm_timeout_seconds = original_timeout

    def test_llm_manager_cancellation_raises_request_cancelled(self):
        from threading import Event
        from llm_manager import llm_manager, RequestCancelledError

        class SlowLLM:
            def invoke(self, payload):
                time.sleep(0.2)
                return payload

        cancel_event = Event()
        cancel_event.set()

        with self.assertRaises(RequestCancelledError):
            with llm_manager.request_scope("req_cancelled", cancel_checker=cancel_event.is_set):
                llm_manager.invoke("hello", source="test.cancel", llm=SlowLLM())

    def test_agent_request_timeout_returns_timeout_message(self):
        original_request_timeout = config.request_timeout_seconds
        self.tempdir = tempfile.TemporaryDirectory()
        original_memory_db_path = config.memory_db_path
        original_memory_backup_dir = config.memory_backup_dir
        original_state_snapshot_dir = config.state_snapshot_dir
        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.request_timeout_seconds = 0.01

        import agent_core

        agent_core_module = importlib.reload(agent_core)
        agent = agent_core_module.AgentCore()
        original_graph = agent.graph

        class SlowGraph:
            def invoke(self, inputs):
                time.sleep(0.2)
                return {"messages": [HumanMessage(content="done")]}

        agent.graph = SlowGraph()
        try:
            result = agent.invoke("timeout test")
            self.assertIn("Request timed out", result)
        finally:
            agent.graph = original_graph
            config.request_timeout_seconds = original_request_timeout
            config.memory_db_path = original_memory_db_path
            config.memory_backup_dir = original_memory_backup_dir
            config.state_snapshot_dir = original_state_snapshot_dir
            self.tempdir.cleanup()


class SnapshotResumeTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_resume_from_latest_snapshot_uses_restored_state(self):
        state = {
            "messages": [HumanMessage(content="resume me")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["resume"],
            "request_id": "req_original",
            "session_id": "session_original",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }
        self.agent._persist_state_snapshot("req_original", "planning_completed", state, extra={"query": "resume me"})

        captured_inputs = {}

        class FakeGraph:
            def invoke(self, inputs):
                captured_inputs.update(copy.deepcopy(inputs))
                return {"messages": [AIMessage(content="resumed ok")]}

        original_graph = self.agent.graph
        self.agent.graph = FakeGraph()
        try:
            result = self.agent.resume_from_snapshot("req_original")
        finally:
            self.agent.graph = original_graph

        self.assertEqual(result, "resumed ok")
        self.assertEqual(captured_inputs["session_id"], "session_original")
        self.assertEqual(captured_inputs["plan"][0]["description"], "step one")
        self.assertNotEqual(captured_inputs["request_id"], "req_original")

    def test_resume_from_terminal_snapshot_returns_final_response(self):
        state = {
            "messages": [HumanMessage(content="done")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": [],
            "global_keywords": ["done"],
            "request_id": "req_finished",
            "session_id": "session_finished",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "already completed",
        }
        self.agent._persist_state_snapshot("req_finished", "agent_completed", state, extra={})

        result = self.agent.resume_from_snapshot("req_finished")

        self.assertEqual(result, "already completed")

    def test_list_snapshots_returns_stage_summaries(self):
        state = {
            "messages": [HumanMessage(content="resume me")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["resume"],
            "request_id": "req_listed",
            "session_id": "session_listed",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }
        self.agent._persist_state_snapshot("req_listed", "planning_completed", state, extra={})
        self.agent._persist_state_snapshot("req_listed", "subtask_prepared", state, extra={})

        snapshots = self.agent.list_snapshots("req_listed")

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["stage"], "planning_completed")
        self.assertEqual(snapshots[1]["stage"], "subtask_prepared")
        self.assertEqual(snapshots[1]["index"], 2)

    def test_resume_from_snapshot_accepts_stage_selector(self):
        base_state = {
            "messages": [HumanMessage(content="resume me")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "reflections": [],
            "global_keywords": ["resume"],
            "request_id": "req_stage_selector",
            "session_id": "session_stage_selector",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }
        planning_state = dict(base_state)
        planning_state["current_subtask_index"] = 0
        advanced_state = dict(base_state)
        advanced_state["current_subtask_index"] = 1

        self.agent._persist_state_snapshot("req_stage_selector", "planning_completed", planning_state, extra={})
        self.agent._persist_state_snapshot("req_stage_selector", "subtask_advanced", advanced_state, extra={})

        captured_inputs = {}

        class FakeGraph:
            def invoke(self, inputs):
                captured_inputs.update(copy.deepcopy(inputs))
                return {"messages": [AIMessage(content="resumed from stage")]}

        original_graph = self.agent.graph
        self.agent.graph = FakeGraph()
        try:
            result = self.agent.resume_from_snapshot("req_stage_selector", snapshot_name="planning_completed")
        finally:
            self.agent.graph = original_graph

        self.assertEqual(result, "resumed from stage")
        self.assertEqual(captured_inputs["current_subtask_index"], 0)

    def test_resume_from_snapshot_accepts_numeric_selector(self):
        base_state = {
            "messages": [HumanMessage(content="resume me")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "reflections": [],
            "global_keywords": ["resume"],
            "request_id": "req_numeric_selector",
            "session_id": "session_numeric_selector",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }
        first_state = dict(base_state)
        first_state["current_subtask_index"] = 0
        second_state = dict(base_state)
        second_state["current_subtask_index"] = 1

        self.agent._persist_state_snapshot("req_numeric_selector", "planning_completed", first_state, extra={})
        self.agent._persist_state_snapshot("req_numeric_selector", "subtask_advanced", second_state, extra={})

        captured_inputs = {}

        class FakeGraph:
            def invoke(self, inputs):
                captured_inputs.update(copy.deepcopy(inputs))
                return {"messages": [AIMessage(content="resumed from numeric selector")]}

        original_graph = self.agent.graph
        self.agent.graph = FakeGraph()
        try:
            result = self.agent.resume_from_snapshot("req_numeric_selector", snapshot_name="1")
        finally:
            self.agent.graph = original_graph

        self.assertEqual(result, "resumed from numeric selector")
        self.assertEqual(captured_inputs["current_subtask_index"], 0)


class ToolRerouteTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_collect_recent_tool_failures_reads_trailing_tool_messages(self):
        messages = [
            HumanMessage(content="question"),
            ToolMessage(
                content=json.dumps({
                    "ok": False,
                    "tool": "get_mock_weather",
                    "error_type": "timeout",
                    "retryable": True,
                    "message": "timeout",
                }, ensure_ascii=False),
                tool_call_id="tool-call-1",
            ),
        ]

        failures = self.agent._collect_recent_tool_failures(messages)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["tool"], "get_mock_weather")
        self.assertEqual(failures[0]["error_type"], "timeout")

    def test_filter_failed_tools_for_subtask_excludes_failed_candidates(self):
        @tool
        def first_tool() -> str:
            """First tool."""
            return "first"

        @tool
        def second_tool() -> str:
            """Second tool."""
            return "second"

        tools = [self.agent._wrap_tool_for_runtime(first_tool), self.agent._wrap_tool_for_runtime(second_tool)]
        tool_skills = [
            {"name": tools[0].name, "tool": tools[0], "description": "first"},
            {"name": tools[1].name, "tool": tools[1], "description": "second"},
        ]

        filtered_tools, filtered_skills, failed_names = self.agent._filter_failed_tools_for_subtask(
            0,
            tools,
            tool_skills,
            {"0": [tools[0].name]},
        )

        self.assertEqual([tool.name for tool in filtered_tools], [tools[1].name])
        self.assertEqual([item["name"] for item in filtered_skills], [tools[1].name])
        self.assertEqual(failed_names, [tools[0].name])

    def test_build_tool_reroute_plan_uses_alternative_tool_for_retryable_failure(self):
        @tool
        def primary_weather_tool() -> str:
            """Get weather for a city."""
            return "primary"

        @tool
        def backup_weather_tool() -> str:
            """Backup weather lookup for city forecast."""
            return "backup"

        primary = self.agent._wrap_tool_for_runtime(primary_weather_tool)
        backup = self.agent._wrap_tool_for_runtime(backup_weather_tool)
        self.agent.skills.register_tools([primary, backup], source_type="test")

        reroute_plan = self.agent._build_tool_reroute_plan(
            "check weather forecast for beijing",
            ["weather", "forecast", "beijing"],
            [primary],
            [{"name": primary.name, "tool": primary, "description": "primary weather"}],
            [{"tool": primary.name, "error_type": "timeout", "retryable": True, "message": "timeout"}],
            [primary.name],
        )

        self.assertEqual(reroute_plan["mode"], "alternative_tools")
        self.assertEqual([tool.name for tool in reroute_plan["selected_tools"]], [backup.name])
        self.assertEqual(reroute_plan["alternatives"], [backup.name])

    def test_build_tool_reroute_plan_falls_back_for_invalid_arguments(self):
        @tool
        def weather_tool(city: str) -> str:
            """Get weather for a city."""
            return city

        wrapped_tool = self.agent._wrap_tool_for_runtime(weather_tool)

        reroute_plan = self.agent._build_tool_reroute_plan(
            "check weather",
            ["weather"],
            [wrapped_tool],
            [{"name": wrapped_tool.name, "tool": wrapped_tool, "description": "weather"}],
            [{"tool": wrapped_tool.name, "error_type": "invalid_arguments", "retryable": False, "message": "missing city"}],
            [wrapped_tool.name],
        )

        self.assertEqual(reroute_plan["mode"], "fallback_invalid_arguments")
        self.assertEqual(reroute_plan["selected_tools"], [])
        self.assertIn("invalid arguments", reroute_plan["reason"])


class RequestCancellationTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_cancel_request_marks_active_request(self):
        self.agent._register_request("req_cancel_me")
        try:
            message = self.agent.cancel_request("req_cancel_me")
            self.assertIn("Cancellation requested", message)
            self.assertTrue(self.agent.is_request_cancelled("req_cancel_me"))
        finally:
            self.agent._clear_request("req_cancel_me")

    def test_invoke_returns_cancelled_message_when_request_cancelled(self):
        import threading

        original_timeout = config.request_timeout_seconds
        config.request_timeout_seconds = 5

        class CooperativeGraph:
            def __init__(self, agent):
                self.agent = agent

            def invoke(self, inputs):
                request_id = inputs["request_id"]
                while not self.agent.is_request_cancelled(request_id):
                    time.sleep(0.02)
                return {"messages": [AIMessage(content="should not complete normally")]}

        original_graph = self.agent.graph
        self.agent.graph = CooperativeGraph(self.agent)
        result_holder = {}

        def run_invoke():
            result_holder["result"] = self.agent.invoke("cancel me")

        worker = threading.Thread(target=run_invoke)
        worker.start()
        try:
            while not self.agent.get_last_request_id() or not self.agent.is_request_active(self.agent.get_last_request_id()):
                time.sleep(0.01)
            request_id = self.agent.get_last_request_id()
            self.agent.cancel_request(request_id)
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertIn("Request cancelled", result_holder["result"])
        finally:
            self.agent.graph = original_graph
            config.request_timeout_seconds = original_timeout


if __name__ == "__main__":
    unittest.main()