import importlib
import os
import tempfile
import unittest

from langchain_core.messages import HumanMessage

from core.config import config
from memory.memory_manager import MemoryManager


class MemoryQualityTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")

    def tearDown(self):
        config.memory_db_path = self.original_memory_db_path
        config.memory_backup_dir = self.original_memory_backup_dir
        config.state_snapshot_dir = self.original_state_snapshot_dir
        self.tempdir.cleanup()

    def test_add_and_load_memory_preserves_type_and_quality_tags(self):
        manager = MemoryManager()
        memory_id = manager.add_memory(
            "session_1",
            "general",
            ["weather", "beijing"],
            "weather lookup",
            "user input",
            "tool output",
            request_id="req_1",
            memory_type="step",
            quality_tags=["success", "continue"],
        )

        memory_data = manager.load_full_memory(memory_id)

        self.assertEqual(memory_data["memory_type"], "step")
        self.assertEqual(memory_data["quality_tags"], ["success", "continue"])

    def test_retrieve_memory_prefers_successful_cases(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_a",
            "general",
            ["weather", "beijing"],
            "successful weather lookup",
            "input a",
            "output a",
            request_id="req_success",
            memory_type="step",
            quality_tags=["success"],
        )
        manager.add_memory(
            "session_b",
            "general",
            ["weather", "beijing"],
            "blocked weather lookup",
            "input b",
            "output b",
            request_id="req_blocked",
            memory_type="step",
            quality_tags=["blocked", "ask_user"],
        )

        results = manager.retrieve_memory(match_keywords=["weather", "beijing"], limit=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["request_id"], "req_success")
        self.assertEqual(results[0]["quality_tags"], ["success"])
        self.assertEqual(results[1]["request_id"], "req_blocked")

    def test_add_memory_merges_exact_duplicates_and_increases_weight(self):
        manager = MemoryManager()
        first_id = manager.add_memory(
            "session_dup",
            "general",
            ["weather", "beijing"],
            "weather lookup",
            "check weather in beijing",
            "sunny",
            request_id="req_first",
            memory_type="step",
            quality_tags=["success"],
        )
        second_id = manager.add_memory(
            "session_dup",
            "general",
            ["forecast"],
            "weather lookup",
            "check weather in beijing",
            "sunny",
            request_id="req_second",
            memory_type="step",
            quality_tags=["continue"],
        )

        results = manager.retrieve_memory(match_keywords=["weather", "beijing"], limit=5)
        merged_item = next(item for item in results if item["id"] == first_id)

        self.assertEqual(first_id, second_id)
        self.assertEqual(merged_item["weight"], 2)
        self.assertEqual(merged_item["request_id"], "req_second")
        self.assertEqual(merged_item["quality_tags"], ["success", "continue"])
        self.assertIn("forecast", merged_item["keywords"])

    def test_add_memory_keeps_distinct_outputs_separate(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_distinct",
            "general",
            ["weather", "beijing"],
            "weather lookup",
            "check weather in beijing",
            "sunny",
            request_id="req_sunny",
            memory_type="step",
            quality_tags=["success"],
        )
        manager.add_memory(
            "session_distinct",
            "general",
            ["weather", "beijing"],
            "weather lookup",
            "check weather in beijing",
            "rainy",
            request_id="req_rainy",
            memory_type="step",
            quality_tags=["retry"],
        )

        results = manager.retrieve_memory(match_keywords=["weather", "beijing"], limit=5)

        self.assertEqual(len([item for item in results if item["memory_type"] == "step"]), 2)

    def test_retrieve_failure_memories_filters_failure_cases(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_ok",
            "general",
            ["weather", "beijing"],
            "successful weather lookup",
            "input ok",
            "output ok",
            request_id="req_success",
            memory_type="step",
            quality_tags=["success"],
        )
        manager.add_memory(
            "session_fail",
            "general",
            ["weather", "beijing"],
            "blocked weather lookup",
            "input fail",
            "output fail",
            request_id="req_blocked",
            memory_type="failure_case",
            quality_tags=["blocked", "ask_user"],
        )

        results = manager.retrieve_failure_memories(match_keywords=["weather", "beijing"], limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["request_id"], "req_blocked")
        self.assertEqual(results[0]["memory_type"], "failure_case")

    def test_retrieve_memory_returns_empty_when_keywords_do_not_match(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_weather",
            "general",
            ["weather", "beijing"],
            "successful weather lookup",
            "input weather",
            "output weather",
            request_id="req_weather",
            memory_type="step",
            quality_tags=["success"],
        )

        results = manager.retrieve_memory(match_keywords=["nginx", "config"], limit=5)

        self.assertEqual(results, [])

    def test_retrieve_memory_requires_full_keyword_match_by_default(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_partial",
            "general",
            ["weather"],
            "weather only memory",
            "input weather",
            "output weather",
            request_id="req_partial",
            memory_type="step",
            quality_tags=["success"],
        )

        results = manager.retrieve_memory(match_keywords=["weather", "beijing"], limit=5)

        self.assertEqual(results, [])

    def test_retrieve_memory_can_relax_match_threshold_via_hook(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_partial",
            "general",
            ["weather"],
            "weather only memory",
            "input weather",
            "output weather",
            request_id="req_partial",
            memory_type="step",
            quality_tags=["success"],
        )

        results = manager.retrieve_memory(
            match_keywords=["weather", "beijing"],
            limit=5,
            min_overlap_ratio=0.5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["request_id"], "req_partial")

    def test_retrieve_memory_returns_empty_when_keywords_missing(self):
        manager = MemoryManager()
        manager.add_memory(
            "session_weather",
            "general",
            ["weather", "beijing"],
            "successful weather lookup",
            "input weather",
            "output weather",
            request_id="req_weather",
            memory_type="step",
            quality_tags=["success"],
        )

        self.assertEqual(manager.retrieve_memory(match_keywords=[], limit=5), [])
        self.assertEqual(manager.retrieve_memory(match_keywords=None, limit=5), [])

    def test_agent_core_writes_session_and_step_memory_tags(self):
        from app.agent import core as agent_core
        agent_core_module = importlib.reload(agent_core)
        agent = agent_core_module.AgentCore()
        agent.cognitive.extract_features = lambda text, domain_hint="": (["weather", "beijing"], "weather lookup")

        session_memory_id = agent.memory.add_memory(
            "session_case",
            "general",
            ["planning"],
            "session summary",
            "raw input",
            "",
            request_id="req_session",
            memory_type="session_main",
            quality_tags=["pending"],
        )
        state = {
            "messages": [HumanMessage(content="hello")],
            "plan": [],
            "current_subtask_index": 0,
            "reflections": ["all good"],
            "global_keywords": [],
            "failed_tools": {},
            "request_id": "req_session",
            "session_id": "session_case",
            "session_memory_id": session_memory_id,
            "domain_label": "general",
            "memory_summaries": [],
            "retry_counts": {},
            "blocked": False,
            "final_response": "",
        }

        agent._record_step_memory(
            "session_case",
            "req_step",
            "general",
            "check weather in beijing",
            "sunny",
            "done",
            quality_tags=["success"],
        )
        agent._finalize_session_memory(state, "completed", quality_tags=["success"])

        step_results = agent.memory.retrieve_memory(match_keywords=["weather", "beijing"], limit=5)
        session_data = agent.memory.load_full_memory(session_memory_id)

        self.assertTrue(any(item["memory_type"] == "step" and item["quality_tags"] == ["success"] for item in step_results))
        self.assertEqual(session_data["memory_type"], "session_main")
        self.assertEqual(session_data["quality_tags"], ["success"])

    def test_agent_core_writes_failure_steps_to_failure_case_memory(self):
        from app.agent import core as agent_core
        agent_core_module = importlib.reload(agent_core)
        agent = agent_core_module.AgentCore()
        agent.cognitive.extract_features = lambda text, domain_hint="": (["booking", "missing"], "booking failure")

        agent._record_step_memory(
            "session_failure",
            "req_failure",
            "general",
            "book a meeting without time",
            "tool rejected",
            "missing required parameter",
            quality_tags=["blocked", "ask_user"],
        )

        results = agent.get_failure_memories(match_keywords=["booking", "missing"], limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["memory_type"], "failure_case")
        self.assertEqual(results[0]["quality_tags"], ["blocked", "ask_user"])


if __name__ == "__main__":
    unittest.main()


