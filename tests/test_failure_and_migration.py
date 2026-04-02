import importlib
import logging
import os
import sqlite3
import tempfile
import unittest

from langchain_core.messages import AIMessage

from core.config import config
from memory.memory_manager import MemoryManager


class MemorySchemaMigrationTests(unittest.TestCase):
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

    def test_init_db_migrates_legacy_interactions_table(self):
        db_path = config.resolve_path(config.memory_db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT,
                    timestamp TEXT,
                    domain_label TEXT,
                    keywords TEXT,
                    summary TEXT,
                    raw_input TEXT,
                    raw_output TEXT,
                    large_file_path TEXT,
                    weight INTEGER DEFAULT 1
                )"""
        )
        cursor.execute(
            "INSERT INTO interactions (conv_id, timestamp, domain_label, keywords, summary, raw_input, raw_output, large_file_path, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("session_legacy", "2026-03-31T00:00:00", "general", '["legacy"]', "legacy summary", "legacy input", "legacy output", "", 1),
        )
        conn.commit()
        conn.close()

        manager = MemoryManager()
        conn = sqlite3.connect(manager.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(interactions)")
        columns = {row[1] for row in cursor.fetchall()}
        cursor.execute("SELECT request_id, memory_type, quality_tags, summary FROM interactions WHERE conv_id = ?", ("session_legacy",))
        row = cursor.fetchone()
        conn.close()

        self.assertIn("request_id", columns)
        self.assertIn("memory_type", columns)
        self.assertIn("quality_tags", columns)
        self.assertEqual(row[0], None)
        self.assertEqual(row[1], "general")
        self.assertEqual(row[2], "[]")
        self.assertEqual(row[3], "legacy summary")


class AgentRetryLimitFailureTests(unittest.TestCase):
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
        config.llm_log_file = "failure_cases.log"
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

    def _configure_retry_limit_flow(self):
        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, payload):
                return AIMessage(content="still wrong")

        self.agent.cognitive.extract_features = lambda text: (["retry", "planner"], "retry summary")
        self.agent.cognitive.determine_domain = lambda text: "general"
        self.agent.planner.split_task = lambda text: [
            {"id": 1, "description": "perform unstable action", "expected_outcome": "stable result"}
        ]
        self.agent.reflector.verify_and_reflect = lambda desc, expected, actual: (False, "still failing", "retry")
        self.agent.skills.assign_capabilities_to_task = lambda desc, kws: {
            "prompt_skill": None,
            "prompt_skill_reason": "",
            "tool_skills": [],
            "tool_reasons": [],
            "tools": [],
        }
        self.llm_manager_module.llm_manager.get_llm = lambda: FakeLLM()

    def test_retry_limit_path_blocks_request_and_updates_metrics(self):
        self._configure_retry_limit_flow()

        result = self.agent.invoke("trigger retry limit")
        request_id = self.agent.get_last_request_id()
        summary = self.agent.get_request_summary(request_id)
        snapshots = self.agent.list_snapshots(request_id)

        self.assertIn("exceeded retry limit", result)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "blocked")
        self.assertTrue(summary["blocked"])
        self.assertEqual(summary["metrics"]["retry_count"], 2)
        self.assertEqual(summary["metrics"]["reflection_failure_count"], 2)
        self.assertEqual(summary["metrics"]["blocked_rate"], 1.0)
        self.assertTrue(any(item["stage"] == "agent_blocked" for item in snapshots))
        self.assertTrue(any(item.get("stage") == "agent_blocked" and "retry_limit" in item.get("details", "") for item in summary["checkpoints"]))


if __name__ == "__main__":
    unittest.main()

