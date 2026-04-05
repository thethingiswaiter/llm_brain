import copy
import importlib
import json
import logging
import os
import pathlib
import io
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from core.config import config
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

    def test_parse_json_object_tolerates_bom_and_nul_prefix(self):
        payload = '\ufeff\x00{"keywords": ["agent"], "summary": "ok"}'

        parsed = parse_json_object(payload, required_fields={"keywords": list, "summary": str})

        self.assertEqual(parsed["keywords"], ["agent"])
        self.assertEqual(parsed["summary"], "ok")

    def test_parse_json_array_rejects_missing_json(self):
        with self.assertRaises(StructuredOutputFormatError):
            parse_json_array("not a json payload")

    def test_reflector_falls_back_to_stringified_response_when_raw_content_is_not_parseable(self):
        import cognitive.reflector as reflector_module

        class FakeResponse:
            def __init__(self):
                self.content = "not-json"

        original_invoke = reflector_module.llm_manager.invoke
        original_stringify = reflector_module.llm_manager._stringify_response
        reflector = reflector_module.Reflector()
        reflector_module.llm_manager.invoke = lambda prompt, source="": FakeResponse()
        reflector_module.llm_manager._stringify_response = lambda response: '{"success": true, "reflection": "ok", "action": "continue"}'
        try:
            success, reflection, action = reflector.verify_and_reflect("subtask", "expected", "actual")
        finally:
            reflector_module.llm_manager.invoke = original_invoke
            reflector_module.llm_manager._stringify_response = original_stringify

        self.assertTrue(success)
        self.assertEqual(reflection, "ok")
        self.assertEqual(action, "continue")

    def test_reflector_short_circuits_observation_task_with_concrete_result(self):
        import cognitive.reflector as reflector_module

        original_invoke = reflector_module.llm_manager.invoke
        reflector = reflector_module.Reflector()
        reflector_module.llm_manager.invoke = lambda prompt, source="": (_ for _ in ()).throw(AssertionError("LLM should not be called"))
        try:
            success, reflection, action = reflector.verify_and_reflect(
                "查看 nginx.conf 文件内容",
                "直接给出用户需要的结果。只有在关键信息确实缺失时，才提出简洁的澄清问题。",
                "文件内容如下：\n`hello world`",
            )
        finally:
            reflector_module.llm_manager.invoke = original_invoke

        self.assertTrue(success)
        self.assertIn("可直接返回", reflection)
        self.assertEqual(action, "continue")


class TaskPlannerTests(unittest.TestCase):
    def test_split_task_preserves_llm_output_without_direct_lookup_collapse(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke
        planner_module.llm_manager.invoke = lambda prompt, source="": FakeResponse(
            """```json
            [
              {"id": 1, "description": "Identify the system or environment where the hostname is to be queried.", "expected_outcome": "Target system identified."},
              {"id": 2, "description": "Use the appropriate command-line tool based on the OS.", "expected_outcome": "Command output collected."},
              {"id": 3, "description": "Parse the command output to extract the hostname.", "expected_outcome": "Hostname extracted."},
              {"id": 4, "description": "Verify the hostname matches expected patterns.", "expected_outcome": "Hostname validated."},
              {"id": 5, "description": "Return or display the hostname to the user.", "expected_outcome": "Hostname returned."}
            ]
            ```"""
        )
        try:
            result = planner.split_task("查询一下主机名称")
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["description"], "Identify the system or environment where the hostname is to be queried.")

    def test_split_task_includes_planning_capability_context_in_prompt(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        captured = {}
        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke

        def fake_invoke(prompt, source=""):
            captured["prompt"] = prompt
            captured["source"] = source
            return FakeResponse(
                """[
                  {"id": 1, "description": "使用 list_directory 工具列出目录项", "expected_outcome": "得到目录项列表"}
                ]"""
            )

        planner_module.llm_manager.invoke = fake_invoke
        try:
            result = planner.split_task(
                "找到当前目录下core.py文件的完整路径",
                capability_context={
                    "planning_policy": "Plan only against registered capabilities.",
                    "prompt_skills": [
                        {
                            "name": "filesystem_helper",
                            "description": "处理工作区文件定位与阅读。",
                            "keywords": ["文件", "路径", "目录"],
                            "source_file": "filesystem.md",
                        }
                    ],
                    "tools": [
                        {
                            "name": "list_directory",
                            "description": "列出工作区内目录项。",
                            "keywords": ["目录", "文件", "路径"],
                            "arguments": ["path<string> optional: 工作区内路径"],
                            "constraints": ["安全边界：仅允许工作区内路径"],
                            "source_type": "runtime",
                        },
                        {
                            "name": "bash",
                            "description": "安全执行终端命令。",
                            "keywords": ["终端", "命令", "搜索文件"],
                            "arguments": ["command<string> required: shell command"],
                            "constraints": ["仅允许 allowlist 前缀", "禁止 shell operator"],
                            "source_type": "runtime",
                        },
                    ],
                },
            )
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 1)
        self.assertEqual(captured["source"], "planner.split_task")
        self.assertIn("当前能力清单", captured["prompt"])
        self.assertIn("filesystem_helper", captured["prompt"])
        self.assertIn("list_directory", captured["prompt"])
        self.assertIn("bash", captured["prompt"])
        self.assertIn("args:", captured["prompt"])
        self.assertIn("constraints:", captured["prompt"])
        self.assertIn("仅允许工作区内路径", captured["prompt"])
        self.assertIn("禁止输出", captured["prompt"])
        self.assertIn("手动操作 / 人工检查 / 让用户自己做", captured["prompt"])
        self.assertIn("依赖人类或外部 GUI 手动完成", captured["prompt"])

    def test_split_task_preserves_multi_step_plan_in_default_mode(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke
        planner_module.llm_manager.invoke = lambda prompt, source="": FakeResponse(
            """[
              {"id": 1, "description": "使用Python的os模块查找当前工作目录", "expected_outcome": "得到当前目录"},
              {"id": 2, "description": "拼接core.py路径", "expected_outcome": "得到候选路径"},
              {"id": 3, "description": "验证文件是否存在", "expected_outcome": "确认路径有效"}
            ]"""
        )
        try:
            result = planner.split_task("找到当前目录下core.py文件的完整路径")
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["description"], "使用Python的os模块查找当前工作目录")

    def test_split_task_forces_decomposable_for_obviously_complex_step(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke
        planner_module.llm_manager.invoke = lambda prompt, source="": FakeResponse(
            """[
              {"id": 1, "description": "分析失败原因并规划后续执行路径", "execution_mode": "leaf"}
            ]"""
        )
        try:
            result = planner.split_task("分析失败原因并规划后续执行路径")
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["execution_mode"], "decomposable")

    def test_split_task_preserves_validation_style_subtasks_in_thinking_mode(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke
        planner_module.llm_manager.invoke = lambda prompt, source="": FakeResponse(
            """[
                            {"id": 1, "description": "列出当前项目根目录下的文件和文件夹", "expected_outcome": "得到目录项列表"},
              {"id": 2, "description": "列出当前目录下的所有文件和文件夹", "expected_outcome": "得到目录项列表"},
              {"id": 3, "description": "统计列表中文件和文件夹的总数", "expected_outcome": "得到总数"}
            ]"""
        )
        try:
            result = planner.split_task(r"D:\file\vscode\llm_brain 当前项目下有多少个文件", thinking_mode=True)
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["description"], "列出当前项目根目录下的文件和文件夹")

    def test_split_task_rejects_manual_editor_steps_and_falls_back_to_single_task(self):
        import cognitive.planner as planner_module
        from cognitive.planner import TaskPlanner

        class FakeResponse:
            def __init__(self, content):
                self.content = content

        planner = TaskPlanner()
        original_invoke = planner_module.llm_manager.invoke
        planner_module.llm_manager.invoke = lambda prompt, source="": FakeResponse(
            """[
              {"id": 1, "description": "打开 VS Code 项目目录", "expected_outcome": "定位到目录"},
              {"id": 2, "description": "使用文本编辑器查看 nginx.conf", "expected_outcome": "读到文件内容"}
            ]"""
        )
        try:
            result = planner.split_task("查看 nginx.conf 文件内容")
        finally:
            planner_module.llm_manager.invoke = original_invoke

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["description"], "查看 nginx.conf 文件内容")


class DomainClassificationTests(unittest.TestCase):
    def test_normalize_domain_label_uses_chinese_tree_only(self):
        from cognitive.feature_extractor import CognitiveSystem, DEFAULT_DOMAIN_LABEL

        system = CognitiveSystem()

        self.assertEqual(system.normalize_domain_label("计算机"), "计算机")
        self.assertEqual(system.normalize_domain_label("其他"), DEFAULT_DOMAIN_LABEL)
        self.assertEqual(system.normalize_domain_label("Technology/Hardware"), DEFAULT_DOMAIN_LABEL)


class ObservabilityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        try:
            self.agent.mcp.close_all()
        except Exception:
            pass
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_build_request_metrics_deduplicates_reentered_subtask_started_events(self):
        events = [
            {"event_type": "checkpoint", "stage": "subtask_started", "details": "index=1 | description=鏌ヨ涓绘満鍚嶇О", "logged_at": "2026-04-01 14:00:00,000"},
            {"event_type": "checkpoint", "stage": "tool_started", "details": "tool=get_system_info", "logged_at": "2026-04-01 14:00:01,000"},
            {"event_type": "checkpoint", "stage": "tool_succeeded", "details": "tool=get_system_info", "logged_at": "2026-04-01 14:00:02,000"},
            {"event_type": "checkpoint", "stage": "subtask_started", "details": "index=1 | description=鏌ヨ涓绘満鍚嶇О", "logged_at": "2026-04-01 14:00:03,000"},
            {"event_type": "checkpoint", "stage": "reflection_completed", "details": "index=1 | success=True | action=continue", "logged_at": "2026-04-01 14:00:04,000"},
        ]

        metrics = self.agent.observability.build_request_metrics(
            events,
            latest_state={"plan": [{"id": 1, "description": "鏌ヨ涓绘満鍚嶇О"}], "retry_counts": {}},
            status="completed",
        )

        self.assertEqual(metrics["subtask_count"], 1)
        self.assertEqual(metrics["tool_call_count"], 1)


class AgentStateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
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
            "failed_tools": {"0": ["weather_tool"]},
            "request_id": "req_unit_test",
            "session_id": "session_unit_test",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {"0": 1},
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

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["request_id"], "req_unit_test")
        self.assertEqual(payload["stage"], "request_received")
        self.assertEqual(payload["state"]["session_id"], "session_unit_test")
        self.assertEqual(payload["state"]["messages"][0]["content"], "hello world")
        self.assertEqual(payload["state"]["failed_tools"], {"0": ["weather_tool"]})
        self.assertEqual(payload["state"]["replan_counts"], {"0": 1})
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
        from app.agent import core as agent_core
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

    def test_tool_wrapper_normalizes_failure_payload_returned_by_tool(self):
        @tool
        def blocked_tool() -> str:
            """Tool that returns a blocked payload."""
            return json.dumps(
                {
                    "ok": False,
                    "blocked": True,
                    "reason": "Command contains disallowed shell operator: '&&'",
                },
                ensure_ascii=False,
            )

        safe_tool = self.agent._wrap_tool_for_runtime(blocked_tool)

        result = safe_tool.invoke({})
        payload = json.loads(result)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["tool"], "blocked_tool")
        self.assertEqual(payload["error_type"], "blocked")
        self.assertFalse(payload["retryable"])

    def test_timed_out_tool_run_is_tracked_until_background_completion(self):
        @tool
        def slow_tool() -> str:
            """Tool that keeps running after timeout."""
            time.sleep(0.12)
            return "done"

        original_timeout = config.tool_timeout_seconds
        original_grace = getattr(config, "tool_cancellation_grace_seconds", 0.2)
        config.tool_timeout_seconds = 0.01
        config.tool_cancellation_grace_seconds = 0.01
        try:
            safe_tool = self.agent._wrap_tool_for_runtime(slow_tool)
            result = safe_tool.invoke({})
            payload = json.loads(result)

            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error_type"], "timeout")

            stats_during_detach = self.agent.tool_runtime.get_tool_run_stats()
            self.assertEqual(stats_during_detach["detached"], 1)
            self.assertEqual(stats_during_detach["tracked"], 1)

            time.sleep(0.2)
            stats_after_completion = self.agent.tool_runtime.get_tool_run_stats()
            self.assertEqual(stats_after_completion["detached"], 0)
            self.assertEqual(stats_after_completion["tracked"], 0)
        finally:
            config.tool_timeout_seconds = original_timeout
            config.tool_cancellation_grace_seconds = original_grace


class TimeoutControlTests(unittest.TestCase):
    def test_build_structured_payload_includes_core_guardrail_fields(self):
        from core.llm.manager import llm_manager

        with llm_manager.request_scope("req_guardrail", session_id="session_guardrail"):
            payload = llm_manager._build_structured_payload(
                "llm_request",
                message="hello",
                source="test.core",
                outcome="started",
            )

        self.assertEqual(payload["request_id"], "req_guardrail")
        self.assertEqual(payload["session_id"], "session_guardrail")
        self.assertEqual(payload["source"], "test.core")
        self.assertEqual(payload["outcome"], "started")

    def test_llm_manager_timeout_raises_timeout_error(self):
        from core.llm.manager import llm_manager

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
        from core.llm.manager import llm_manager, RequestCancelledError

        class SlowLLM:
            def invoke(self, payload):
                time.sleep(0.2)
                return payload

        cancel_event = Event()
        cancel_event.set()

        with self.assertRaises(RequestCancelledError):
            with llm_manager.request_scope("req_cancelled", cancel_checker=cancel_event.is_set):
                llm_manager.invoke("hello", source="test.cancel", llm=SlowLLM())

    def test_llm_manager_dependency_failure_raises_specific_error(self):
        from core.llm.manager import LLMDependencyUnavailableError, llm_manager

        class BrokenLLM:
            def invoke(self, payload):
                raise ConnectionError("ollama connection refused")

        with self.assertRaises(LLMDependencyUnavailableError):
            llm_manager.invoke("hello", source="test.dependency", llm=BrokenLLM())

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
        from app.agent import core as agent_core
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

    def test_agent_request_dependency_failure_returns_dependency_message(self):
        self.tempdir = tempfile.TemporaryDirectory()
        original_memory_db_path = config.memory_db_path
        original_memory_backup_dir = config.memory_backup_dir
        original_state_snapshot_dir = config.state_snapshot_dir
        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
        from core.llm.manager import LLMDependencyUnavailableError

        agent_core_module = importlib.reload(agent_core)
        agent = agent_core_module.AgentCore()
        original_graph = agent.graph

        class BrokenGraph:
            def invoke(self, inputs):
                raise LLMDependencyUnavailableError("LLM dependency unavailable: ollama connection refused")

        agent.graph = BrokenGraph()
        try:
            result = agent.invoke("dependency test")
            self.assertIn("model dependency is unavailable", result)
            self.assertIn("ollama connection refused", result)
        finally:
            agent.graph = original_graph
            config.memory_db_path = original_memory_db_path
            config.memory_backup_dir = original_memory_backup_dir
            config.state_snapshot_dir = original_state_snapshot_dir
            self.tempdir.cleanup()


class RetentionManagerTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_log_dir = config.log_dir
        self.original_audit_log_dir = getattr(config, "audit_log_dir", os.path.join("runtime_state", "audit"))
        self.original_log_retention_days = getattr(config, "log_retention_days", 7)
        self.original_snapshot_retention_days = getattr(config, "snapshot_retention_days", 7)
        self.original_audit_log_retention_days = getattr(config, "audit_log_retention_days", 14)
        self.original_memory_backup_retention_days = getattr(config, "memory_backup_retention_days", 14)
        self.original_log_retention_max_files = getattr(config, "log_retention_max_files", 20)
        self.original_snapshot_retention_max_request_dirs = getattr(config, "snapshot_retention_max_request_dirs", 200)
        self.original_audit_log_retention_max_files = getattr(config, "audit_log_retention_max_files", 50)
        self.original_memory_backup_retention_max_files = getattr(config, "memory_backup_retention_max_files", 20)
        self.original_log_retention_max_bytes = getattr(config, "log_retention_max_bytes", 0)
        self.original_snapshot_retention_max_bytes = getattr(config, "snapshot_retention_max_bytes", 0)
        self.original_audit_log_retention_max_bytes = getattr(config, "audit_log_retention_max_bytes", 0)
        self.original_memory_backup_retention_max_bytes = getattr(config, "memory_backup_retention_max_bytes", 0)
        self.original_retention_auto_prune_enabled = getattr(config, "retention_auto_prune_enabled", True)
        self.original_retention_auto_prune_min_interval_seconds = getattr(config, "retention_auto_prune_min_interval_seconds", 300)
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.log_dir = os.path.join(self.tempdir.name, "logs")
        config.audit_log_dir = os.path.join(self.tempdir.name, "audit")
        config.log_retention_days = 7
        config.snapshot_retention_days = 7
        config.audit_log_retention_days = 7
        config.memory_backup_retention_days = 7
        config.log_retention_max_files = 20
        config.snapshot_retention_max_request_dirs = 200
        config.audit_log_retention_max_files = 50
        config.memory_backup_retention_max_files = 20
        config.log_retention_max_bytes = 0
        config.snapshot_retention_max_bytes = 0
        config.audit_log_retention_max_bytes = 0
        config.memory_backup_retention_max_bytes = 0
        config.retention_auto_prune_enabled = True
        config.retention_auto_prune_min_interval_seconds = 300
        from app.agent import core as agent_core
        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        config.log_dir = self.original_log_dir
        config.audit_log_dir = self.original_audit_log_dir
        config.log_retention_days = self.original_log_retention_days
        config.snapshot_retention_days = self.original_snapshot_retention_days
        config.audit_log_retention_days = self.original_audit_log_retention_days
        config.memory_backup_retention_days = self.original_memory_backup_retention_days
        config.log_retention_max_files = getattr(self, "original_log_retention_max_files", 20)
        config.snapshot_retention_max_request_dirs = getattr(self, "original_snapshot_retention_max_request_dirs", 200)
        config.audit_log_retention_max_files = getattr(self, "original_audit_log_retention_max_files", 50)
        config.memory_backup_retention_max_files = getattr(self, "original_memory_backup_retention_max_files", 20)
        config.log_retention_max_bytes = getattr(self, "original_log_retention_max_bytes", 0)
        config.snapshot_retention_max_bytes = getattr(self, "original_snapshot_retention_max_bytes", 0)
        config.audit_log_retention_max_bytes = getattr(self, "original_audit_log_retention_max_bytes", 0)
        config.memory_backup_retention_max_bytes = getattr(self, "original_memory_backup_retention_max_bytes", 0)
        config.retention_auto_prune_enabled = self.original_retention_auto_prune_enabled
        config.retention_auto_prune_min_interval_seconds = self.original_retention_auto_prune_min_interval_seconds
        self.tempdir.cleanup()

    def _write_file(self, path: str, content: str, days_old: int) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(content, encoding="utf-8")
        timestamp = (datetime.now(timezone.utc) - timedelta(days=days_old)).timestamp()
        os.utime(path, (timestamp, timestamp))

    def _write_snapshot(self, request_id: str, filename: str, days_old: int) -> None:
        request_dir = pathlib.Path(config.resolve_path(config.state_snapshot_dir)) / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        file_path = request_dir / filename
        file_path.write_text("{}", encoding="utf-8")
        timestamp = (datetime.now(timezone.utc) - timedelta(days=days_old)).timestamp()
        os.utime(file_path, (timestamp, timestamp))

    def test_retention_status_and_prune_runtime_data(self):
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "old.log"), "old", 10)
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "new.log"), "new", 1)
        self._write_file(os.path.join(config.resolve_path(config.audit_log_dir), "system_mcp_audit.jsonl"), "{}\n", 10)
        self._write_file(os.path.join(config.resolve_path(config.memory_backup_dir), "backup_old.txt"), "backup", 10)
        self._write_snapshot("req_old", "001_request_completed.json", 10)
        self._write_snapshot("req_new", "001_request_completed.json", 1)

        status = self.agent.get_retention_status()

        target_map = {item["key"]: item for item in status["targets"]}
        self.assertEqual(target_map["logs"]["expired_count"], 1)
        self.assertEqual(target_map["audit_logs"]["expired_count"], 1)
        self.assertEqual(target_map["memory_backups"]["expired_count"], 1)
        self.assertEqual(target_map["snapshots"]["expired_count"], 1)

        dry_run = self.agent.prune_runtime_data(apply=False)
        self.assertEqual(dry_run["mode"], "dry_run")
        self.assertEqual(dry_run["totals"]["expired_count"], 4)
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "old.log")))
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.state_snapshot_dir), "req_old")))

        applied = self.agent.prune_runtime_data(apply=True)
        self.assertEqual(applied["mode"], "apply")
        self.assertEqual(applied["totals"]["deleted_count"], 4)
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "old.log")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.audit_log_dir), "system_mcp_audit.jsonl")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.memory_backup_dir), "backup_old.txt")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.state_snapshot_dir), "req_old")))
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "new.log")))
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.state_snapshot_dir), "req_new")))

    def test_retention_marks_excess_recent_files_when_over_max_items(self):
        config.log_retention_days = 30
        config.log_retention_max_files = 2

        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_a.log"), "a", 1)
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_b.log"), "b", 2)
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_c.log"), "c", 3)

        status = self.agent.get_retention_status()
        target_map = {item["key"]: item for item in status["targets"]}

        self.assertEqual(target_map["logs"]["item_count"], 3)
        self.assertEqual(target_map["logs"]["expired_count"], 1)
        self.assertEqual(target_map["logs"]["max_items"], 2)

        applied = self.agent.prune_runtime_data(apply=True)
        log_target = {item["key"]: item for item in applied["targets"]}["logs"]
        self.assertEqual(log_target["deleted_count"], 1)
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_a.log")))
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_b.log")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_c.log")))

    def test_retention_marks_excess_files_when_over_max_total_bytes(self):
        config.log_retention_days = 30
        config.log_retention_max_files = 10
        config.log_retention_max_bytes = 6

        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_new.log"), "1234", 1)
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_mid.log"), "12", 2)
        self._write_file(os.path.join(config.resolve_path(config.log_dir), "log_old.log"), "123", 3)

        status = self.agent.get_retention_status()
        log_target = {item["key"]: item for item in status["targets"]}["logs"]

        self.assertEqual(log_target["item_count"], 3)
        self.assertEqual(log_target["max_total_bytes"], 6)
        self.assertEqual(log_target["expired_count"], 1)
        self.assertEqual(log_target["reclaimable_bytes"], 3)

        applied = self.agent.prune_runtime_data(apply=True)
        log_target = {item["key"]: item for item in applied["targets"]}["logs"]
        self.assertEqual(log_target["deleted_count"], 1)
        self.assertEqual(log_target["deleted_bytes"], 3)
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_new.log")))
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_mid.log")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.log_dir), "log_old.log")))

    def test_snapshot_retention_tracks_deleted_bytes_for_request_dirs(self):
        config.snapshot_retention_days = 30
        config.snapshot_retention_max_request_dirs = 10
        config.snapshot_retention_max_bytes = 3

        self._write_snapshot("req_new", "001.json", 1)
        self._write_snapshot("req_old", "001.json", 3)
        req_new_path = pathlib.Path(config.resolve_path(config.state_snapshot_dir), "req_new", "001.json")
        req_old_path = pathlib.Path(config.resolve_path(config.state_snapshot_dir), "req_old", "001.json")
        req_new_path.write_text("123", encoding="utf-8")
        req_old_path.write_text("1234", encoding="utf-8")
        new_timestamp = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
        os.utime(req_new_path, (new_timestamp, new_timestamp))
        os.utime(req_old_path, (old_timestamp, old_timestamp))

        applied = self.agent.prune_runtime_data(apply=True)
        snapshot_target = {item["key"]: item for item in applied["targets"]}["snapshots"]
        self.assertEqual(snapshot_target["deleted_count"], 1)
        self.assertEqual(snapshot_target["deleted_bytes"], 4)
        self.assertTrue(os.path.exists(os.path.join(config.resolve_path(config.state_snapshot_dir), "req_new")))
        self.assertFalse(os.path.exists(os.path.join(config.resolve_path(config.state_snapshot_dir), "req_old")))

    def test_snapshot_persist_triggers_auto_prune_with_interval_guard(self):
        config.retention_auto_prune_enabled = True
        config.retention_auto_prune_min_interval_seconds = 3600
        config.log_retention_days = 1

        old_log_path = os.path.join(config.resolve_path(config.log_dir), "old.log")
        self._write_file(old_log_path, "old", 10)

        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": [],
            "failed_tools": {},
            "request_id": "req_auto_prune",
            "session_id": "session_auto_prune",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }

        self.agent._persist_state_snapshot("req_auto_prune", "planning_completed", state, extra={})
        self.assertFalse(os.path.exists(old_log_path))
        retention_status = self.agent.get_retention_status()
        self.assertEqual(retention_status["last_auto_prune"]["trigger"], "snapshot")
        self.assertEqual(retention_status["last_auto_prune"]["deleted_count"], 1)
        self.assertEqual(retention_status["last_auto_prune_check"]["status"], "executed")

        next_old_log_path = os.path.join(config.resolve_path(config.log_dir), "next_old.log")
        self._write_file(next_old_log_path, "old again", 10)
        self.agent._persist_state_snapshot("req_auto_prune", "subtask_started", state, extra={})
        self.assertTrue(os.path.exists(next_old_log_path))
        retention_status = self.agent.get_retention_status()
        self.assertEqual(retention_status["last_auto_prune_check"]["status"], "skipped_throttled")
        self.assertEqual(retention_status["last_auto_prune_check"]["reason"], "throttled")

    def test_memory_backup_write_can_trigger_auto_prune(self):
        config.retention_auto_prune_enabled = True
        config.retention_auto_prune_min_interval_seconds = 0
        config.memory_backup_retention_days = 1

        old_backup_path = os.path.join(config.resolve_path(config.memory_backup_dir), "old_backup.txt")
        self._write_file(old_backup_path, "stale backup", 10)

        memory_id = self.agent.memory.add_memory(
            conv_id="session_auto_backup",
            domain_label="general",
            keywords=["large"],
            summary="large input",
            raw_input="x" * 6001,
            raw_output="done",
            request_id="req_large_backup",
        )

        self.assertGreater(memory_id, 0)
        self.assertFalse(os.path.exists(old_backup_path))
        retention_status = self.agent.get_retention_status()
        self.assertEqual(retention_status["last_auto_prune"]["trigger"], "memory_backup")

    def test_auto_prune_check_reports_disabled_state(self):
        config.retention_auto_prune_enabled = False

        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": [],
            "failed_tools": {},
            "request_id": "req_auto_prune_disabled",
            "session_id": "session_auto_prune_disabled",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }

        self.agent._persist_state_snapshot("req_auto_prune_disabled", "planning_completed", state, extra={})
        retention_status = self.agent.get_retention_status()
        self.assertEqual(retention_status["last_auto_prune_check"]["status"], "skipped_disabled")
        self.assertEqual(retention_status["last_auto_prune_check"]["reason"], "disabled")


class SnapshotResumeTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
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

    def test_resume_from_snapshot_can_reroute_from_blocked_state(self):
        state = {
            "messages": [HumanMessage(content="book a meeting for tomorrow")],
            "plan": [{"id": 1, "description": "schedule meeting", "expected_outcome": "meeting booked"}],
            "current_subtask_index": 0,
            "reflections": ["Original path was blocked by missing time details."],
            "global_keywords": ["meeting", "schedule"],
            "failed_tools": {"0": ["calendar_tool", "calendar_tool"]},
            "request_id": "req_blocked_resume",
            "session_id": "session_blocked_resume",
            "session_memory_id": 1,
            "domain_label": "Other",
            "memory_summaries": [],
            "retry_counts": {"0": 1},
            "replan_counts": {"0": 1},
            "blocked": True,
            "final_response": "Need user intervention.",
        }
        self.agent._persist_state_snapshot(
            "req_blocked_resume",
            "agent_blocked",
            state,
            extra={"query": "book a meeting for tomorrow"},
        )

        captured_inputs = {}

        class FakeGraph:
            def invoke(self, inputs):
                captured_inputs.update(copy.deepcopy(inputs))
                return {"messages": [AIMessage(content="rerouted ok")]}

        original_graph = self.agent.graph
        self.agent.graph = FakeGraph()
        try:
            result = self.agent.resume_from_snapshot("req_blocked_resume", reroute=True)
        finally:
            self.agent.graph = original_graph

        self.assertEqual(result, "rerouted ok")
        self.assertEqual(captured_inputs["plan"], [])
        self.assertEqual(captured_inputs["current_subtask_index"], 0)
        self.assertEqual(captured_inputs["failed_tools"], {})
        self.assertEqual(captured_inputs["retry_counts"], {})
        self.assertEqual(captured_inputs["replan_counts"], {})
        self.assertFalse(captured_inputs["blocked"])
        self.assertIn("Resume reroute context:", captured_inputs["messages"][0].content)
        self.assertIn("historical_failed_tools=calendar_toolx2", captured_inputs["messages"][0].content)

    def test_resume_from_snapshot_rejects_invalid_state_shape(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_invalid_snapshot")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        invalid_payload = {
            "schema_version": 1,
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_invalid_snapshot",
            "state": {
                "messages": [],
                "plan": [],
                "current_subtask_index": 3,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_invalid_snapshot",
                "session_id": "session_invalid",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "final_response": "",
            },
            "extra": {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(invalid_payload, file_handle, ensure_ascii=False, indent=2)

        result = self.agent.resume_from_snapshot("req_invalid_snapshot")

        self.assertIn("Snapshot validation failed", result)
        self.assertIn("current_subtask_index exceeds plan length", result)

    def test_resume_from_snapshot_rejects_unsupported_schema_version(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_future_snapshot")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        future_payload = {
            "schema_version": 99,
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_future_snapshot",
            "state": {
                "messages": [{"type": "human", "content": "resume me"}],
                "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
                "current_subtask_index": 0,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_future_snapshot",
                "session_id": "session_future",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "final_response": "",
            },
            "extra": {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(future_payload, file_handle, ensure_ascii=False, indent=2)

        result = self.agent.resume_from_snapshot("req_future_snapshot")

        self.assertIn("Snapshot validation failed", result)
        self.assertIn("Unsupported snapshot schema_version", result)

    def test_load_snapshot_payload_migrates_legacy_snapshot_and_backfills_messages(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_legacy_snapshot")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        legacy_payload = {
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_legacy_snapshot",
            "state": {
                "messages": [],
                "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
                "current_subtask_index": 0,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_legacy_snapshot",
                "session_id": "session_legacy",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {},
                "blocked": False,
                "final_response": "",
            },
            "extra": {"query": "resume this legacy request"},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(legacy_payload, file_handle, ensure_ascii=False, indent=2)

        payload = self.agent._load_snapshot_payload("req_legacy_snapshot")

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["migrated_from_version"], 0)
        self.assertEqual(payload["state"]["messages"][0]["type"], "human")
        self.assertEqual(payload["state"]["messages"][0]["content"], "resume this legacy request")
        self.assertEqual(payload["state"]["replan_counts"], {})

    def test_resume_from_snapshot_rejects_missing_resumable_message_history(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_missing_messages")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        invalid_payload = {
            "schema_version": 1,
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_missing_messages",
            "state": {
                "messages": [],
                "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
                "current_subtask_index": 0,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_missing_messages",
                "session_id": "session_missing_messages",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "final_response": "",
            },
            "extra": {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(invalid_payload, file_handle, ensure_ascii=False, indent=2)

        result = self.agent.resume_from_snapshot("req_missing_messages")

        self.assertIn("Snapshot validation failed", result)
        self.assertIn("does not contain resumable message history", result)

    def test_resume_from_snapshot_rejects_out_of_range_retry_state(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_bad_retry_state")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        invalid_payload = {
            "schema_version": 1,
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_bad_retry_state",
            "state": {
                "messages": [{"type": "human", "content": "resume me"}],
                "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
                "current_subtask_index": 0,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_bad_retry_state",
                "session_id": "session_bad_retry_state",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {"3": 1},
                "replan_counts": {},
                "blocked": False,
                "final_response": "",
            },
            "extra": {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(invalid_payload, file_handle, ensure_ascii=False, indent=2)

        result = self.agent.resume_from_snapshot("req_bad_retry_state")

        self.assertIn("Snapshot validation failed", result)
        self.assertIn("retry_counts references out-of-range subtask index", result)

    def test_resume_from_snapshot_rejects_premature_final_response(self):
        request_dir = self.agent.snapshot_store.snapshot_request_dir("req_premature_final")
        snapshot_path = os.path.join(request_dir, "001_planning_completed.json")
        invalid_payload = {
            "schema_version": 1,
            "created_at": "2026-03-31T00:00:00+00:00",
            "stage": "planning_completed",
            "request_id": "req_premature_final",
            "state": {
                "messages": [{"type": "human", "content": "resume me"}],
                "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
                "current_subtask_index": 0,
                "reflections": [],
                "global_keywords": [],
                "failed_tools": {},
                "request_id": "req_premature_final",
                "session_id": "session_premature_final",
                "session_memory_id": 0,
                "domain_label": "general",
                "memory_summaries": [],
                "retry_counts": {},
                "replan_counts": {},
                "blocked": False,
                "final_response": "already finished",
            },
            "extra": {},
        }
        with open(snapshot_path, "w", encoding="utf-8") as file_handle:
            json.dump(invalid_payload, file_handle, ensure_ascii=False, indent=2)

        result = self.agent.resume_from_snapshot("req_premature_final")

        self.assertIn("Snapshot validation failed", result)
        self.assertIn("contains final_response before all planned subtasks were completed", result)


class RecentRequestFilterTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def _persist_request(self, request_id: str, stage: str, blocked: bool = False, source_request_id: str = ""):
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "request_id": request_id,
            "session_id": f"session_{request_id}",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": blocked,
            "final_response": "" if stage != "request_completed" else "done",
        }
        self.agent._persist_state_snapshot(
            request_id,
            stage,
            state,
            extra={"source_request_id": source_request_id} if source_request_id else {},
        )

    def test_get_recent_request_summaries_filters_failed_and_resumed_requests(self):
        self._persist_request("req_completed", "request_completed")
        self._persist_request("req_blocked", "agent_blocked", blocked=True)
        self._persist_request("req_resumed_failed", "request_failed", source_request_id="req_original")

        failed = self.agent.get_recent_request_summaries(limit=10, statuses=["failed", "blocked"])
        failed_ids = [item["request_id"] for item in failed]
        self.assertIn("req_blocked", failed_ids)
        self.assertIn("req_resumed_failed", failed_ids)
        self.assertNotIn("req_completed", failed_ids)

        resumed = self.agent.get_recent_request_summaries(limit=10, resumed_only=True)
        self.assertEqual([item["request_id"] for item in resumed], ["req_resumed_failed"])


class ToolRerouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_memory_db_path = config.memory_db_path
        cls.original_memory_backup_dir = config.memory_backup_dir
        cls.original_state_snapshot_dir = config.state_snapshot_dir
        cls.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(cls.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(cls.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(cls.tempdir.name, "snapshots")

        from app.agent import core as agent_core

        cls.agent_core_module = importlib.reload(agent_core)
        cls.shared_agent = cls.agent_core_module.AgentCore(
            auto_load_tools=False,
            auto_load_mcp=False,
            build_graph=False,
        )

    @classmethod
    def tearDownClass(cls):
        config.memory_db_path = cls.original_memory_db_path
        config.memory_backup_dir = cls.original_memory_backup_dir
        config.state_snapshot_dir = cls.original_state_snapshot_dir
        cls.tempdir.cleanup()

    def setUp(self):
        self.agent = self.__class__.shared_agent
        self.agent.skills.loaded_tool_skills = {}

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

    def test_get_latest_successful_tool_name_to_pause_skips_non_truncated_success(self):
        messages = [
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"name": "read_text_file", "args": {"path": "demo.txt"}, "id": "call-1", "type": "tool_call"}]),
            ToolMessage(
                content=json.dumps({"ok": True, "path": "demo.txt", "content": "hello", "truncated": False}, ensure_ascii=False),
                tool_call_id="call-1",
            ),
        ]

        paused = self.agent._get_latest_successful_tool_name_to_pause(messages)

        self.assertEqual(paused, "read_text_file")

    def test_get_latest_successful_tool_name_to_pause_keeps_truncated_reads_available(self):
        messages = [
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"name": "read_text_file", "args": {"path": "demo.txt"}, "id": "call-1", "type": "tool_call"}]),
            ToolMessage(
                content=json.dumps({"ok": True, "path": "demo.txt", "content": "hello", "truncated": True}, ensure_ascii=False),
                tool_call_id="call-1",
            ),
        ]

        paused = self.agent._get_latest_successful_tool_name_to_pause(messages)

        self.assertEqual(paused, "")

    def test_should_downgrade_ask_user_to_continue_for_observation_answer(self):
        should_downgrade = self.agent._should_downgrade_ask_user_to_continue(
            "查看 nginx.conf 文件内容",
            "文件内容如下：\n`hello world`",
            "结果与预期不完全一致。",
            recent_failures=[],
        )

        self.assertTrue(should_downgrade)

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

    def test_merge_failed_tools_ignores_retryable_no_output_failures(self):
        merged = self.agent._merge_failed_tools(
            {},
            0,
            [{"tool": "bash", "error_type": "no_output", "retryable": True, "message": "empty output"}],
        )

        self.assertEqual(merged, {"0": []})

    def test_summarize_historical_failed_tools_counts_prior_subtasks(self):
        counts = self.agent.tool_runtime.summarize_historical_failed_tools(
            {
                "0": ["tool_a", "tool_b"],
                "1": ["tool_a"],
                "2": ["tool_c"],
            },
            current_subtask_index=2,
        )

        self.assertEqual(counts, {"tool_a": 2, "tool_b": 1})

    def test_summarize_historical_failed_tool_signals_aggregates_severity(self):
        signals = self.agent.tool_runtime.summarize_historical_failed_tool_signals(
            {
                "0": {
                    "tool_a": {
                        "count": 1,
                        "retryable_count": 1,
                        "non_retryable_count": 0,
                        "error_type_counts": {"timeout": 1},
                        "severity_score": 3,
                    },
                    "tool_b": {
                        "count": 1,
                        "retryable_count": 0,
                        "non_retryable_count": 1,
                        "error_type_counts": {"invalid_arguments": 1},
                        "severity_score": 5,
                    },
                },
                "1": {
                    "tool_a": {
                        "count": 1,
                        "retryable_count": 0,
                        "non_retryable_count": 1,
                        "error_type_counts": {"execution_error": 1},
                        "severity_score": 3,
                    }
                },
            },
            current_subtask_index=2,
        )

        self.assertEqual(signals["tool_a"]["count"], 2)
        self.assertEqual(signals["tool_a"]["severity_score"], 6)
        self.assertEqual(signals["tool_a"]["error_type_counts"], {"timeout": 1, "execution_error": 1})
        self.assertEqual(signals["tool_b"]["severity_score"], 5)

    def test_filter_failed_tools_for_subtask_excludes_historically_unstable_tools(self):
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
            2,
            tools,
            tool_skills,
            {"0": [tools[0].name], "1": [tools[0].name]},
            historical_failed_tools={tools[0].name: 2},
            historical_failure_threshold=2,
        )

        self.assertEqual([tool.name for tool in filtered_tools], [tools[1].name])
        self.assertEqual([item["name"] for item in filtered_skills], [tools[1].name])
        self.assertEqual(failed_names, [tools[0].name])

    def test_filter_failed_tools_for_subtask_excludes_historically_severe_tools(self):
        @tool
        def severe_tool() -> str:
            """Severe tool."""
            return "severe"

        @tool
        def safer_tool() -> str:
            """Safer tool."""
            return "safe"

        tools = [self.agent._wrap_tool_for_runtime(severe_tool), self.agent._wrap_tool_for_runtime(safer_tool)]
        tool_skills = [
            {"name": tools[0].name, "tool": tools[0], "description": "severe"},
            {"name": tools[1].name, "tool": tools[1], "description": "safe"},
        ]

        filtered_tools, filtered_skills, failed_names = self.agent._filter_failed_tools_for_subtask(
            2,
            tools,
            tool_skills,
            {},
            historical_failed_tool_signals={
                tools[0].name: {
                    "count": 2,
                    "retryable_count": 1,
                    "non_retryable_count": 1,
                    "error_type_counts": {"timeout": 1, "invalid_arguments": 1},
                    "severity_score": 8,
                }
            },
            historical_failure_severity_threshold=6,
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

    def test_build_tool_reroute_plan_keeps_same_tool_for_retryable_no_output(self):
        @tool
        def bash(command: str) -> str:
            """Run a shell command."""
            return command

        wrapped_bash = self.agent._wrap_tool_for_runtime(bash)

        reroute_plan = self.agent._build_tool_reroute_plan(
            "find core.py under current directory",
            ["core.py", "当前目录"],
            [wrapped_bash],
            [{"name": wrapped_bash.name, "tool": wrapped_bash, "description": "Run a shell command."}],
            [{"tool": wrapped_bash.name, "error_type": "no_output", "retryable": True, "message": "empty output"}],
            [],
        )

        self.assertEqual(reroute_plan["mode"], "retry_same_tool_with_diagnostics")
        self.assertEqual([tool.name for tool in reroute_plan["selected_tools"]], [wrapped_bash.name])
        self.assertEqual(reroute_plan["alternatives"], [wrapped_bash.name])

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

    def test_build_tool_reroute_plan_excludes_historically_failed_alternatives(self):
        @tool
        def primary_weather_tool() -> str:
            """Get weather for a city."""
            return "primary"

        @tool
        def unstable_backup_tool() -> str:
            """Backup weather forecast lookup for a city with unstable execution history."""
            return "unstable"

        @tool
        def stable_backup_tool() -> str:
            """Stable backup weather forecast lookup for a city."""
            return "stable"

        primary = self.agent._wrap_tool_for_runtime(primary_weather_tool)
        unstable_backup = self.agent._wrap_tool_for_runtime(unstable_backup_tool)
        stable_backup = self.agent._wrap_tool_for_runtime(stable_backup_tool)
        self.agent.skills.register_tools([primary, unstable_backup, stable_backup], source_type="test")

        reroute_plan = self.agent._build_tool_reroute_plan(
            "check weather forecast for beijing",
            ["weather", "forecast", "beijing"],
            [primary],
            [{"name": primary.name, "tool": primary, "description": "primary weather"}],
            [{"tool": primary.name, "error_type": "timeout", "retryable": True, "message": "timeout"}],
            [primary.name],
            historical_failed_tool_names=[unstable_backup.name],
        )

        self.assertEqual(reroute_plan["mode"], "alternative_tools")
        self.assertEqual([tool.name for tool in reroute_plan["selected_tools"]], [stable_backup.name])
        self.assertEqual(reroute_plan["alternatives"], [stable_backup.name])

    def test_reprioritize_tool_skills_deprioritizes_more_severe_history(self):
        @tool
        def timeout_prone_tool() -> str:
            """Backup weather lookup for a city forecast."""
            return "timeout"

        @tool
        def cleaner_tool() -> str:
            """Backup weather lookup for a city forecast."""
            return "clean"

        timeout_prone = self.agent._wrap_tool_for_runtime(timeout_prone_tool)
        cleaner = self.agent._wrap_tool_for_runtime(cleaner_tool)
        tool_skills = [
            {"name": timeout_prone.name, "tool": timeout_prone, "description": "forecast", "overlap_count": 3, "match_ratio": 0.8, "route_reason": "matched"},
            {"name": cleaner.name, "tool": cleaner, "description": "forecast", "overlap_count": 3, "match_ratio": 0.8, "route_reason": "matched"},
        ]

        prioritized = self.agent._reprioritize_tool_skills(
            tool_skills,
            historical_failed_tool_signals={
                timeout_prone.name: {
                    "count": 1,
                    "retryable_count": 1,
                    "non_retryable_count": 0,
                    "error_type_counts": {"timeout": 1},
                    "severity_score": 3,
                },
                cleaner.name: {
                    "count": 1,
                    "retryable_count": 1,
                    "non_retryable_count": 0,
                    "error_type_counts": {"cancelled": 1},
                    "severity_score": 1,
                },
            },
        )

        self.assertEqual([item["name"] for item in prioritized], [cleaner.name, timeout_prone.name])
        self.assertEqual(prioritized[0]["historical_failure_severity"], 1)
        self.assertEqual(prioritized[1]["historical_failure_severity"], 3)

    def test_expand_tool_candidates_deprioritizes_historically_failed_tools(self):
        @tool
        def lightly_failed_tool() -> str:
            """Backup weather forecast lookup for a city."""
            return "light"

        @tool
        def clean_tool() -> str:
            """Backup weather forecast lookup for a city."""
            return "clean"

        light = self.agent._wrap_tool_for_runtime(lightly_failed_tool)
        clean = self.agent._wrap_tool_for_runtime(clean_tool)
        self.agent.skills.register_tools([light, clean], source_type="test")

        alternatives = self.agent.tool_runtime.expand_tool_candidates(
            "check weather forecast for beijing",
            ["weather", "forecast", "beijing"],
            failed_tool_names=[],
            historical_failed_tool_counts={light.name: 1, clean.name: 0},
            limit=2,
        )

        self.assertEqual([item["name"] for item in alternatives], [clean.name, light.name])
        self.assertEqual(alternatives[0]["historical_failure_count"], 0)
        self.assertEqual(alternatives[1]["historical_failure_count"], 1)

    def test_expand_tool_candidates_prioritizes_lower_severity_before_lower_count(self):
        @tool
        def invalid_args_tool() -> str:
            """Backup weather forecast lookup for a city."""
            return "invalid"

        @tool
        def timeout_tool() -> str:
            """Backup weather forecast lookup for a city."""
            return "timeout"

        invalid_args = self.agent._wrap_tool_for_runtime(invalid_args_tool)
        timeout = self.agent._wrap_tool_for_runtime(timeout_tool)
        self.agent.skills.register_tools([invalid_args, timeout], source_type="test")

        alternatives = self.agent.tool_runtime.expand_tool_candidates(
            "check weather forecast for beijing",
            ["weather", "forecast", "beijing"],
            failed_tool_names=[],
            historical_failed_tool_counts={invalid_args.name: 1, timeout.name: 2},
            historical_failed_tool_signals={
                invalid_args.name: {
                    "count": 1,
                    "retryable_count": 0,
                    "non_retryable_count": 1,
                    "error_type_counts": {"invalid_arguments": 1},
                    "severity_score": 5,
                },
                timeout.name: {
                    "count": 2,
                    "retryable_count": 2,
                    "non_retryable_count": 0,
                    "error_type_counts": {"timeout": 2},
                    "severity_score": 6,
                },
            },
            limit=2,
        )

        self.assertEqual([item["name"] for item in alternatives], [invalid_args.name, timeout.name])
        self.assertEqual(alternatives[0]["historical_failure_severity"], 5)
        self.assertEqual(alternatives[1]["historical_failure_severity"], 6)

    def test_build_tool_reroute_plan_falls_back_when_only_high_risk_alternatives_remain(self):
        @tool
        def primary_weather_tool() -> str:
            """Get weather for a city."""
            return "primary"

        @tool
        def risky_backup_tool() -> str:
            """Backup weather forecast lookup for a city."""
            return "risky"

        primary = self.agent._wrap_tool_for_runtime(primary_weather_tool)
        risky_backup = self.agent._wrap_tool_for_runtime(risky_backup_tool)
        self.agent.skills.register_tools([primary, risky_backup], source_type="test")

        reroute_plan = self.agent._build_tool_reroute_plan(
            "check weather forecast for beijing",
            ["weather", "forecast", "beijing"],
            [primary],
            [{"name": primary.name, "tool": primary, "description": "primary weather"}],
            [{"tool": primary.name, "error_type": "timeout", "retryable": True, "message": "timeout"}],
            [primary.name],
            historical_failed_tool_names=[risky_backup.name],
            historical_failed_tool_counts={risky_backup.name: 2},
            historical_failed_tool_signals={
                risky_backup.name: {
                    "count": 2,
                    "retryable_count": 1,
                    "non_retryable_count": 1,
                    "error_type_counts": {"timeout": 1, "invalid_arguments": 1},
                    "severity_score": 8,
                }
            },
            historical_failure_severity_threshold=6,
        )

        self.assertEqual(reroute_plan["mode"], "fallback_high_risk_history")
        self.assertEqual(reroute_plan["selected_tools"], [])
        self.assertIn(risky_backup.name, reroute_plan["reason"])

    def test_build_no_tool_guidance_prefers_ask_user_for_invalid_arguments(self):
        guidance = self.agent._build_no_tool_guidance(
            "fallback_invalid_arguments",
            recent_failures=[{"error_type": "invalid_arguments", "retryable": False, "message": "missing city"}],
        )

        self.assertIn("不要猜测缺失的工具参数", guidance)
        self.assertIn("请明确向用户询问", guidance)

    def test_build_no_tool_guidance_prefers_safe_direct_answer_for_high_risk_history(self):
        guidance = self.agent._build_no_tool_guidance(
            "fallback_high_risk_history",
            recent_failures=[{"error_type": "timeout", "retryable": True, "message": "timeout"}],
            reroute_reason="historical failure severity marked the available tool route as unsafe",
        )

        self.assertIn("不要调用工具", guidance)
        self.assertIn("只有在现有上下文足以安全完成任务时", guidance)

    def test_build_no_tool_guidance_is_available_without_recent_failures(self):
        guidance = self.agent._build_no_tool_guidance("normal", recent_failures=[])

        self.assertIn("请直接给出最佳答案", guidance)
        self.assertIn("请向用户询问", guidance)


class RequestCancellationTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        from app.agent import core as agent_core
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


class ObservabilityTests(unittest.TestCase):
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
        config.llm_log_file = "observability.log"
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

    def test_request_summary_aggregates_snapshots_memories_and_checkpoints(self):
        request_id = "req_observability"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "request_id": request_id,
            "session_id": "session_observability",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "all done",
        }

        self.agent._persist_state_snapshot(request_id, "planning_completed", state, extra={"query": "hello"})
        self.agent._persist_state_snapshot(request_id, "request_completed", state, extra={"final_output": "all done"})
        self.agent.memory.add_memory(
            "session_observability",
            "general",
            ["hello"],
            "session summary",
            "hello",
            "all done",
            request_id=request_id,
            memory_type="session_main",
            quality_tags=["success"],
        )
        self.agent.memory.add_memory(
            "session_observability",
            "general",
            ["step", "hello"],
            "step summary",
            "step one",
            "done",
            request_id=request_id,
            memory_type="step",
            quality_tags=["success"],
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "planning_completed",
            details="subtask_count=1",
            request_id=request_id,
            duration_ms=12.5,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "subtask_started",
            details="index=1",
            request_id=request_id,
            duration_ms=7.25,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "reflection_completed",
            details="success=True",
            request_id=request_id,
            duration_ms=3.0,
        )

        summary = self.agent.get_request_summary(request_id)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["session_id"], "session_observability")
        self.assertEqual(summary["final_response"], "all done")
        self.assertEqual(summary["snapshot_count"], 2)
        self.assertEqual(summary["memory_count"], 2)
        self.assertEqual(summary["checkpoint_count"], 3)
        self.assertEqual(
            [item["stage"] for item in summary["checkpoints"]],
            ["planning_completed", "subtask_started", "reflection_completed"],
        )
        self.assertFalse(summary["triage"]["needs_attention"])
        self.assertFalse(summary["triage"]["is_resumed"])
        self.assertEqual(summary["metrics"]["tool_detached_count"], 0)
        self.assertFalse(summary["triage"]["has_detached_tools"])
        self.assertEqual(summary["metrics"]["stage_duration_ms"]["planning"], 12.5)
        self.assertEqual(summary["metrics"]["stage_duration_ms"]["subtask"], 7.25)
        self.assertEqual(summary["metrics"]["stage_duration_ms"]["reflection"], 3.0)

    def test_structured_logs_are_written_as_json_events(self):
        request_id = "req_json_log"
        self.llm_manager_module.llm_manager.log_checkpoint(
            "subtask_started",
            details="index=1",
            request_id=request_id,
        )

        log_path = pathlib.Path(self.llm_manager_module.llm_manager.get_log_path())
        self.assertTrue(log_path.exists())

        log_line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        payload = json.loads(log_line.split(" | ", 2)[2])

        self.assertEqual(payload["event_type"], "checkpoint")
        self.assertEqual(payload["request_id"], request_id)
        self.assertEqual(payload["stage"], "subtask_started")
        self.assertEqual(payload["details"], "index=1")

    def test_console_checkpoint_output_includes_details(self):
        request_id = "req_console_details"
        console_logger = logging.getLogger("llm_brain.console")
        original_handlers = list(console_logger.handlers)
        original_propagate = console_logger.propagate
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        console_logger.handlers = [handler]
        console_logger.propagate = False
        try:
            self.llm_manager_module.llm_manager.log_checkpoint(
                "subtask_started",
                details="index=1 | description=获取当前目录下core.py文件的完整路径",
                request_id=request_id,
                console=True,
            )
        finally:
            handler.flush()
            console_logger.handlers = original_handlers
            console_logger.propagate = original_propagate

        rendered = stream.getvalue().strip()
        self.assertIn(f"request_id={request_id}", rendered)
        self.assertIn("stage=subtask_started", rendered)
        self.assertIn("子任务1", rendered)
        self.assertIn("获取当前目录下core.py文件的完整路径", rendered)

    def test_retryable_no_output_tool_failure_is_logged_as_warning(self):
        @tool
        def bash(command: str) -> dict:
            """Run a shell command."""
            return {
                "ok": True,
                "command": command,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "shell": "powershell",
            }

        safe_tool = self.agent._wrap_tool_for_runtime(bash)
        payload = safe_tool.invoke({"command": "Get-ChildItem -Path . -Filter core.py"})

        self.assertEqual(payload["error_type"], "no_output")
        events = self.llm_manager_module.llm_manager.get_request_events("-")
        relevant_events = [
            event for event in events
            if event.get("stage") == "tool_failed" and event.get("tool_name") == "bash"
        ]

        self.assertTrue(relevant_events)
        self.assertEqual(relevant_events[-1]["level"], "WARNING")
        self.assertEqual(relevant_events[-1]["outcome"], "retryable")

    def test_request_summary_triage_surfaces_failure_context(self):
        request_id = "req_triage_failure"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "failed_tools": {"0": ["weather_tool"]},
            "request_id": request_id,
            "session_id": "session_triage",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {"0": 2},
            "replan_counts": {},
            "blocked": True,
            "final_response": "Need user intervention.",
        }

        self.agent._persist_state_snapshot(
            request_id,
            "agent_blocked",
            state,
            extra={"source_request_id": "req_original_triage", "action": "retry_limit"},
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_failed",
            details="tool=weather_tool | error_type=timeout",
            request_id=request_id,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "agent_blocked",
            details="index=1 | action=retry_limit",
            request_id=request_id,
            level=logging.ERROR,
            outcome="blocked",
        )

        summary = self.agent.get_request_summary(request_id)

        self.assertTrue(summary["triage"]["needs_attention"])
        self.assertTrue(summary["triage"]["is_resumed"])
        self.assertEqual(summary["triage"]["source_request_id"], "req_original_triage")
        self.assertEqual(summary["triage"]["latest_failure_stage"], "agent_blocked")
        self.assertIn("retry_limit", summary["triage"]["latest_failure_details"])
        self.assertEqual(summary["triage"]["tool_attention_count"], 1)

    def test_request_summary_triage_surfaces_reroute_fallback_mode(self):
        request_id = "req_triage_reroute"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "failed_tools": {"0": ["weather_tool"]},
            "request_id": request_id,
            "session_id": "session_triage_reroute",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {"0": 1},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }

        self.agent._persist_state_snapshot(request_id, "subtask_prepared", state, extra={})
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_reroute_applied",
            details="index=1 | mode=fallback_high_risk_history | failed_tools=weather_tool",
            request_id=request_id,
            level=logging.ERROR,
            mode="fallback_high_risk_history",
        )

        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(summary["triage"]["latest_reroute_mode"], "fallback_high_risk_history")
        self.assertTrue(summary["triage"]["used_no_tool_fallback"])
        self.assertIn("mode=fallback_high_risk_history", summary["triage"]["latest_reroute_details"])

    def test_request_summary_triage_backfills_failure_details_from_snapshot_extra(self):
        request_id = "req_triage_dependency_extra"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "request_id": request_id,
            "session_id": "session_triage_dependency",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }

        self.agent._persist_state_snapshot(
            request_id,
            "request_failed",
            state,
            extra={
                "error_type": "dependency_unavailable",
                "error": "LLM dependency unavailable: ollama connection refused",
            },
        )

        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["triage"]["latest_failure_stage"], "request_failed")
        self.assertIn("error_type=dependency_unavailable", summary["triage"]["latest_failure_details"])
        self.assertIn("ollama connection refused", summary["triage"]["latest_failure_details"])

    def test_request_summary_surfaces_detached_tool_attention(self):
        request_id = "req_detached_tool_attention"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "request_id": request_id,
            "session_id": "session_detached",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "completed with detached tool cleanup pending",
        }

        self.agent._persist_state_snapshot(request_id, "request_completed", state, extra={"final_output": state["final_response"]})
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_detached",
            details="tool=slow_tool | reason=timeout | tool_run_id=toolrun_000001",
            request_id=request_id,
            level=logging.ERROR,
        )

        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["metrics"]["tool_detached_count"], 1)
        self.assertTrue(summary["triage"]["needs_attention"])
        self.assertTrue(summary["triage"]["has_detached_tools"])
        self.assertEqual(summary["triage"]["detached_tool_count"], 1)
        self.assertEqual(summary["triage"]["latest_failure_stage"], "tool_detached")
        self.assertEqual(len(summary["detached_tools"]), 1)
        self.assertEqual(summary["detached_tools"][0]["tool_name"], "slow_tool")
        self.assertEqual(summary["detached_tools"][0]["reason"], "timeout")
        self.assertEqual(summary["detached_tools"][0]["tool_run_id"], "toolrun_000001")

    def test_request_summary_merges_runtime_detached_tool_details(self):
        request_id = "req_runtime_detached"
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 0,
            "reflections": [],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "request_id": request_id,
            "session_id": "session_runtime_detached",
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "",
        }

        self.agent._persist_state_snapshot(request_id, "subtask_started", state, extra={})
        self.agent.tool_runtime._tracked_tool_runs["toolrun_runtime_1"] = {
            "tool_name": "slow_tool",
            "request_id": request_id,
            "executor": None,
            "future": None,
            "started_at": time.monotonic() - 0.2,
            "status": "detached",
            "detached_reason": "cancelled",
            "detached_at": time.monotonic() - 0.1,
        }

        summary = self.agent.get_request_summary(request_id)

        self.assertEqual(len(summary["detached_tools"]), 1)
        self.assertEqual(summary["detached_tools"][0]["tool_run_id"], "toolrun_runtime_1")
        self.assertEqual(summary["detached_tools"][0]["source"], "runtime")
        self.assertEqual(summary["detached_tools"][0]["reason"], "cancelled")
        self.assertEqual(summary["triage"]["detached_tool_count"], 0)

    def test_request_rollup_aggregates_recent_request_metrics(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "done",
        }

        state_completed = dict(base_state, request_id="req_rollup_completed", session_id="session_rollup_a")
        state_failed = dict(base_state, request_id="req_rollup_failed", session_id="session_rollup_b", blocked=True, retry_counts={"0": 2})

        self.agent._persist_state_snapshot("req_rollup_completed", "request_completed", state_completed, extra={"final_output": "done"})
        self.agent._persist_state_snapshot(
            "req_rollup_failed",
            "agent_blocked",
            state_failed,
            extra={"source_request_id": "req_original_rollup"},
        )

        self.llm_manager_module.llm_manager.log_checkpoint(
            "planning_completed",
            details="subtask_count=1",
            request_id="req_rollup_completed",
            duration_ms=10.0,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "reflection_completed",
            details="success=True",
            request_id="req_rollup_completed",
            duration_ms=5.0,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_detached",
            details="tool=slow_tool | reason=timeout | tool_run_id=toolrun_rollup_1",
            request_id="req_rollup_failed",
            duration_ms=20.0,
            level=logging.ERROR,
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_reroute_applied",
            details="index=1 | mode=fallback_high_risk_history | failed_tools=slow_tool",
            request_id="req_rollup_failed",
            level=logging.ERROR,
            mode="fallback_high_risk_history",
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "reflection_completed",
            details="success=False",
            request_id="req_rollup_failed",
            duration_ms=4.0,
        )

        rollup = self.agent.get_request_rollup(limit=10)

        self.assertEqual(rollup["request_count"], 2)
        self.assertEqual(rollup["status_counts"]["completed"], 1)
        self.assertEqual(rollup["status_counts"]["blocked"], 1)
        self.assertEqual(rollup["resumed_count"], 1)
        self.assertEqual(rollup["needs_attention_count"], 1)
        self.assertEqual(rollup["totals"]["tool_detached_count"], 1)
        self.assertEqual(rollup["totals"]["retry_count"], 2)
        self.assertEqual(rollup["totals"]["reflection_failure_count"], 1)
        self.assertEqual(rollup["stage_duration_ms_total"]["planning"], 10.0)
        self.assertEqual(rollup["stage_duration_ms_total"]["reflection"], 9.0)
        self.assertEqual(rollup["stage_duration_ms_total"]["tool"], 20.0)
        self.assertEqual(rollup["top_failure_signals"]["stages"][0], ("tool_detached", 1))
        self.assertEqual(rollup["top_failure_signals"]["tools"][0], ("slow_tool", 1))
        self.assertEqual(rollup["top_failure_signals"]["reasons"][0], ("timeout", 1))
        self.assertEqual(rollup["top_failure_signals"]["sources"][0], ("tool_runtime", 1))
        self.assertEqual(rollup["top_failure_signals"]["reroute_modes"][0], ("fallback_high_risk_history", 1))
        self.assertEqual(rollup["top_failure_signals"]["no_tool_fallbacks"][0], ("used", 1))
        self.assertEqual(rollup["source_bucket_breakdown"][0]["source"], "tool_runtime")
        self.assertEqual(rollup["source_bucket_breakdown"][0]["count"], 1)
        self.assertEqual(rollup["source_bucket_breakdown"][0]["share"], 1.0)
        self.assertEqual(rollup["top_failure_combinations"]["stage_tool"][0], ("tool_detached+slow_tool", 1))
        self.assertEqual(rollup["top_failure_combinations"]["tool_reason"][0], ("slow_tool+timeout", 1))
        self.assertEqual(rollup["top_failure_combinations"]["stage_source"][0], ("tool_detached+tool_runtime", 1))
        self.assertEqual(rollup["top_failure_combinations"]["stage_reroute"][0], ("tool_detached+fallback_high_risk_history", 1))

    def test_request_rollup_aggregates_dependency_failure_from_snapshot_extra(self):
        completed_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "done",
            "request_id": "req_rollup_dep_completed",
            "session_id": "session_rollup_dep_a",
        }
        failed_state = dict(
            completed_state,
            request_id="req_rollup_dep_failed",
            session_id="session_rollup_dep_b",
            current_subtask_index=0,
            reflections=[],
            final_response="",
        )

        self.agent._persist_state_snapshot(
            "req_rollup_dep_completed",
            "request_completed",
            completed_state,
            extra={"final_output": "done"},
        )
        self.agent._persist_state_snapshot(
            "req_rollup_dep_failed",
            "request_failed",
            failed_state,
            extra={
                "error_type": "dependency_unavailable",
                "error": "LLM dependency unavailable: ollama connection refused",
            },
        )
        self.llm_manager_module.llm_manager.log_structured_event(
            "agent_request",
            message="Agent request failed due to unavailable model dependency",
            request_id="req_rollup_dep_failed",
            session_id="session_rollup_dep_b",
            stage="request_failed",
            source="agent.invoke",
            outcome="failed",
            error_type="dependency_unavailable",
            error="LLM dependency unavailable: ollama connection refused",
        )

        rollup = self.agent.get_request_rollup(limit=10, attention_only=True)

        self.assertEqual(rollup["request_count"], 1)
        self.assertEqual(rollup["top_failure_signals"]["stages"][0], ("request_failed", 1))
        self.assertEqual(rollup["top_failure_signals"]["error_types"][0], ("dependency_unavailable", 1))
        self.assertEqual(rollup["top_failure_signals"]["sources"][0], ("agent_invoke", 1))
        self.assertEqual(rollup["source_bucket_breakdown"][0]["source"], "agent_invoke")
        self.assertEqual(rollup["source_bucket_breakdown"][0]["share"], 1.0)
        self.assertEqual(rollup["top_failure_combinations"]["stage_source"][0], ("request_failed+agent_invoke", 1))

    def test_request_rollup_builds_source_bucket_distribution(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": True,
            "final_response": "",
        }

        self.agent._persist_state_snapshot(
            "req_source_bucket_a",
            "agent_blocked",
            dict(base_state, request_id="req_source_bucket_a", session_id="session_source_bucket_a"),
            extra={},
        )
        self.agent._persist_state_snapshot(
            "req_source_bucket_b",
            "request_failed",
            dict(base_state, request_id="req_source_bucket_b", session_id="session_source_bucket_b", blocked=False),
            extra={"error_type": "dependency_unavailable", "error": "missing provider"},
        )
        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_detached",
            details="tool=slow_tool | reason=timeout | tool_run_id=toolrun_source_bucket_a",
            request_id="req_source_bucket_a",
            level=logging.ERROR,
        )
        self.llm_manager_module.llm_manager.log_structured_event(
            "agent_request",
            message="Agent request failed due to unavailable model dependency",
            request_id="req_source_bucket_b",
            session_id="session_source_bucket_b",
            stage="request_failed",
            source="agent.invoke",
            outcome="failed",
            error_type="dependency_unavailable",
            error="missing provider",
        )

        rollup = self.agent.get_request_rollup(limit=10, attention_only=True)

        self.assertEqual(rollup["request_count"], 2)
        self.assertEqual(rollup["source_bucket_breakdown"][0], {"source": "agent_invoke", "count": 1, "share": 0.5})
        self.assertEqual(rollup["source_bucket_breakdown"][1], {"source": "tool_runtime", "count": 1, "share": 0.5})
        trend_map = {item["source"]: item for item in rollup["source_bucket_trends"]}
        self.assertEqual(trend_map["agent_invoke"]["delta_share"], 1.0 if trend_map["agent_invoke"]["direction"] == "up" else -1.0)
        self.assertEqual(trend_map["tool_runtime"]["delta_share"], 1.0 if trend_map["tool_runtime"]["direction"] == "up" else -1.0)
        self.assertNotEqual(trend_map["agent_invoke"]["direction"], trend_map["tool_runtime"]["direction"])

    def test_request_rollup_builds_source_bucket_trends(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": True,
            "final_response": "",
        }

        request_specs = [
            ("req_source_trend_earlier_1", "request_failed", "session_source_trend_earlier_1", None, None, "agent.invoke", {"error_type": "dependency_unavailable", "error": "missing provider b"}),
            ("req_source_trend_earlier_2", "request_failed", "session_source_trend_earlier_2", None, None, "agent.invoke", {"error_type": "dependency_unavailable", "error": "missing provider c"}),
            ("req_source_trend_recent_1", "agent_blocked", "session_source_trend_recent_1", "tool_detached", "tool=slow_tool | reason=timeout | tool_run_id=trend_recent_1", "checkpoint", None),
            ("req_source_trend_recent_2", "request_failed", "session_source_trend_recent_2", None, None, "agent.invoke", {"error_type": "dependency_unavailable", "error": "missing provider a"}),
        ]

        for request_id, stage, session_id, checkpoint_stage, checkpoint_details, source, extra in request_specs:
            self.agent._persist_state_snapshot(
                request_id,
                stage,
                dict(base_state, request_id=request_id, session_id=session_id, blocked=(stage == "agent_blocked"), final_response=""),
                extra=extra or {},
            )
            if checkpoint_stage:
                self.llm_manager_module.llm_manager.log_checkpoint(
                    checkpoint_stage,
                    details=checkpoint_details,
                    request_id=request_id,
                    level=logging.ERROR,
                )
            else:
                self.llm_manager_module.llm_manager.log_structured_event(
                    "agent_request",
                    message="Agent request failed due to unavailable model dependency",
                    request_id=request_id,
                    session_id=session_id,
                    stage=stage,
                    source=source,
                    outcome="failed",
                    error_type="dependency_unavailable",
                    error=(extra or {}).get("error", "missing provider"),
                )

        rollup = self.agent.get_request_rollup(limit=10, attention_only=True)

        self.assertEqual(rollup["request_count"], 4)
        trend_map = {item["source"]: item for item in rollup["source_bucket_trends"]}
        self.assertEqual(abs(trend_map["tool_runtime"]["delta_share"]), 0.5)
        self.assertEqual(abs(trend_map["agent_invoke"]["delta_share"]), 0.5)
        self.assertNotEqual(trend_map["tool_runtime"]["direction"], trend_map["agent_invoke"]["direction"])
        self.assertEqual(trend_map["tool_runtime"]["delta_share"], -trend_map["agent_invoke"]["delta_share"])

    def test_request_rollup_supports_status_and_attention_filters(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "done",
        }

        self.agent._persist_state_snapshot(
            "req_rollup_completed_filter",
            "request_completed",
            dict(base_state, request_id="req_rollup_completed_filter", session_id="session_filter_a"),
            extra={"final_output": "done"},
        )
        self.agent._persist_state_snapshot(
            "req_rollup_failed_filter",
            "agent_blocked",
            dict(base_state, request_id="req_rollup_failed_filter", session_id="session_filter_b", blocked=True),
            extra={"source_request_id": "req_original_filter"},
        )

        self.llm_manager_module.llm_manager.log_checkpoint(
            "tool_detached",
            details="tool=slow_tool | reason=timeout | tool_run_id=toolrun_rollup_filter_1",
            request_id="req_rollup_failed_filter",
            level=logging.ERROR,
            duration_ms=11.0,
        )

        rollup = self.agent.get_request_rollup(
            limit=10,
            statuses=["blocked"],
            resumed_only=True,
            attention_only=True,
        )

        self.assertEqual(rollup["request_count"], 1)
        self.assertEqual(rollup["status_counts"], {"blocked": 1})
        self.assertEqual(rollup["resumed_count"], 1)
        self.assertEqual(rollup["needs_attention_count"], 1)
        self.assertEqual(rollup["filters"]["statuses"], ["blocked"])
        self.assertTrue(rollup["filters"]["resumed_only"])
        self.assertTrue(rollup["filters"]["attention_only"])
        self.assertEqual(rollup["top_failure_signals"]["stages"][0], ("tool_detached", 1))
        self.assertEqual(rollup["top_failure_combinations"]["stage_tool"][0], ("tool_detached+slow_tool", 1))

    def test_request_rollup_aggregates_repeated_failure_combinations(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": True,
            "final_response": "",
        }

        for request_id in ("req_combo_1", "req_combo_2"):
            self.agent._persist_state_snapshot(
                request_id,
                "agent_blocked",
                dict(base_state, request_id=request_id, session_id=request_id),
                extra={},
            )
            self.llm_manager_module.llm_manager.log_checkpoint(
                "tool_detached",
                details="tool=slow_tool | reason=timeout | tool_run_id=" + request_id,
                request_id=request_id,
                level=logging.ERROR,
            )

        rollup = self.agent.get_request_rollup(limit=10, attention_only=True)

        self.assertEqual(rollup["request_count"], 2)
        self.assertEqual(rollup["top_failure_combinations"]["stage_tool"][0], ("tool_detached+slow_tool", 2))
        self.assertEqual(rollup["top_failure_combinations"]["tool_reason"][0], ("slow_tool+timeout", 2))

    def test_recent_request_summaries_support_since_seconds_filter(self):
        base_state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [{"id": 1, "description": "step one", "expected_outcome": "done"}],
            "current_subtask_index": 1,
            "reflections": ["complete"],
            "global_keywords": ["hello"],
            "failed_tools": {},
            "session_memory_id": 1,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "replan_counts": {},
            "blocked": False,
            "final_response": "done",
        }

        self.agent._persist_state_snapshot(
            "req_recent_old",
            "request_completed",
            dict(base_state, request_id="req_recent_old", session_id="session_old"),
            extra={"final_output": "done"},
        )
        self.agent._persist_state_snapshot(
            "req_recent_new",
            "request_completed",
            dict(base_state, request_id="req_recent_new", session_id="session_new"),
            extra={"final_output": "done"},
        )

        old_snapshot_path = pathlib.Path(self.agent._resolve_snapshot_path("req_recent_old"))
        old_payload = json.loads(old_snapshot_path.read_text(encoding="utf-8"))
        old_payload["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        old_snapshot_path.write_text(json.dumps(old_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        summaries = self.agent.get_recent_request_summaries(limit=10, since_seconds=1800)

        summary_ids = [item["request_id"] for item in summaries]
        self.assertIn("req_recent_new", summary_ids)
        self.assertNotIn("req_recent_old", summary_ids)


if __name__ == "__main__":
    unittest.main()

