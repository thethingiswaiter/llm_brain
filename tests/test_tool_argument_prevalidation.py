import importlib
import json
import os
import tempfile
import unittest

from langchain_core.tools import tool

from config import config


class ToolArgumentPrevalidationTests(unittest.TestCase):
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

    def test_prevalidate_rejects_missing_required_argument(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return city

        normalized_kwargs, error_payload = self.agent._prevalidate_tool_arguments(
            get_weather.name,
            get_weather.args_schema,
            {},
        )

        self.assertIsNone(normalized_kwargs)
        payload = json.loads(error_payload)
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertIn("schema validation failed", payload["message"].lower())

    def test_wrapped_tool_rejects_invalid_city_argument(self):
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return city

        safe_tool = self.agent._wrap_tool_for_runtime(get_weather)
        payload = json.loads(safe_tool.invoke({"city": "12345"}))

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertIn("city", payload["message"].lower())

    def test_wrapped_tool_rejects_invalid_time_argument(self):
        @tool
        def schedule_meeting(time: str) -> str:
            """Schedule a meeting at a given time."""
            return time

        safe_tool = self.agent._wrap_tool_for_runtime(schedule_meeting)
        payload = json.loads(safe_tool.invoke({"time": "tomorrow afternoon"}))

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertIn("time format", payload["message"].lower())

    def test_wrapped_tool_accepts_chinese_field_names_and_datetime_formats(self):
        @tool
        def schedule_trip(出发城市: str, 出发时间: str, 日期: str) -> str:
            """Schedule a trip with Chinese argument names."""
            return f"{出发城市}-{出发时间}-{日期}"

        safe_tool = self.agent._wrap_tool_for_runtime(schedule_trip)
        result = safe_tool.invoke({"出发城市": "北京", "出发时间": "下午3点半", "日期": "2026年4月1日"})

        self.assertEqual(result, "北京-下午3点半-2026年4月1日")

    def test_wrapped_tool_rejects_invalid_chinese_city_argument(self):
        @tool
        def get_weather_cn(城市: str) -> str:
            """Get weather for a Chinese city field."""
            return 城市

        safe_tool = self.agent._wrap_tool_for_runtime(get_weather_cn)
        payload = json.loads(safe_tool.invoke({"城市": "12345"}))

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertIn("城市", payload["message"])

    def test_prevalidate_accepts_valid_arguments(self):
        @tool
        def calculate_sum(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        normalized_kwargs, error_payload = self.agent._prevalidate_tool_arguments(
            calculate_sum.name,
            calculate_sum.args_schema,
            {"a": "1.5", "b": 2},
        )

        self.assertIsNone(error_payload)
        self.assertEqual(normalized_kwargs, {"a": 1.5, "b": 2.0})


if __name__ == "__main__":
    unittest.main()
