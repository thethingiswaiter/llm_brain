import os
import tempfile
import unittest

from core.config import config
from app.agent.skill_parser import SkillManager


class FakeTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class SkillRoutingTests(unittest.TestCase):
    def setUp(self):
        self.original_prompt_overlap = config.prompt_skill_min_overlap
        self.original_prompt_ratio = config.prompt_skill_min_match_ratio
        self.original_tool_overlap = config.tool_skill_min_overlap
        self.original_tool_ratio = config.tool_skill_min_match_ratio

        config.prompt_skill_min_overlap = 1
        config.prompt_skill_min_match_ratio = 0.34
        config.tool_skill_min_overlap = 1
        config.tool_skill_min_match_ratio = 0.34

        self.tempdir = tempfile.TemporaryDirectory()
        self.manager = SkillManager(self.tempdir.name)

    def tearDown(self):
        config.prompt_skill_min_overlap = self.original_prompt_overlap
        config.prompt_skill_min_match_ratio = self.original_prompt_ratio
        config.tool_skill_min_overlap = self.original_tool_overlap
        config.tool_skill_min_match_ratio = self.original_tool_ratio
        self.tempdir.cleanup()

    def _write_skill(self, filename: str, name: str, keywords: list[str], description: str, body: str):
        path = os.path.join(self.tempdir.name, filename)
        content = (
            "---\n"
            f"name: {name}\n"
            f"keywords: {keywords}\n"
            f"description: {description}\n"
            "entry_node: main\n"
            "---\n"
            f"{body}"
        )
        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)
        return self.manager.load_skill(filename, force_reload=True)

    def test_tool_match_requires_minimum_ratio(self):
        weather_tool = FakeTool("get_mock_weather", "Get weather for a city")
        hello_tool = FakeTool("sample_hello", "Greets the user by name")
        self.manager.register_tools([weather_tool, hello_tool])

        matched = self.manager.find_relevant_tools("tell me the weather in beijing", ["weather", "beijing"])

        self.assertEqual([item["name"] for item in matched], ["get_mock_weather"])
        self.assertIn("matched", matched[0]["route_reason"])
        self.assertEqual(matched[0]["matched_terms"], ["weather"])

    def test_tool_match_rejects_weak_overlap(self):
        generic_tool = FakeTool("status_lookup", "Lookup system status and service metadata")
        self.manager.register_tool(generic_tool)

        matched = self.manager.find_relevant_tools("explain retry policy for planner", ["retry", "planner"])

        self.assertEqual(matched, [])

    def test_prompt_skill_uses_threshold(self):
        self._write_skill(
            "travel.md",
            "travel_helper",
            ["travel", "itinerary", "hotel"],
            "Travel planning helper",
            "travel body",
        )

        matched = self.manager.find_best_skill(["travel", "itinerary"])
        not_matched = self.manager.find_best_skill(["travel", "budget", "visa", "passport"])

        self.assertIsNotNone(matched)
        self.assertEqual(matched["name"], "travel_helper")
        self.assertIn("itinerary", matched["route_reason"])
        self.assertIsNone(not_matched)

    def test_assign_capabilities_includes_route_reasons(self):
        self._write_skill(
            "weather.md",
            "weather_helper",
            ["weather", "forecast", "city"],
            "Weather planning helper",
            "weather body",
        )
        weather_tool = FakeTool("get_mock_weather", "Get weather forecast for a city")
        self.manager.register_tool(weather_tool)

        capabilities = self.manager.assign_capabilities_to_task(
            "tell me the weather forecast for beijing",
            ["weather", "forecast", "beijing"],
        )

        self.assertEqual(capabilities["prompt_skill"]["name"], "weather_helper")
        self.assertTrue(capabilities["prompt_skill_reason"])
        self.assertEqual(capabilities["tool_skills"][0]["name"], "get_mock_weather")
        self.assertTrue(capabilities["tool_reasons"][0])

    def test_assign_capabilities_prefers_tools_for_execution_tasks(self):
        self._write_skill(
            "fs.md",
            "filesystem_helper",
            ["目录", "文件", "统计"],
            "Filesystem prompt helper",
            "fs body",
        )
        list_tool = FakeTool("list_directory", "列出工作区内目录项（安全边界：仅允许工作区内路径）。")
        self.manager.register_tool(list_tool)

        capabilities = self.manager.assign_capabilities_to_task(
            "查一下你现在目录下有多少文件",
            ["文件数量", "目录", "统计"],
        )

        self.assertIsNone(capabilities["prompt_skill"])
        self.assertEqual(capabilities["prompt_skill_reason"], "suppressed_in_tool_execution_mode")
        self.assertEqual([item["name"] for item in capabilities["tool_skills"]], ["list_directory"])

    def test_tool_match_supports_contiguous_chinese_query_terms(self):
        host_tool = FakeTool("get_system_info", "Get hostname and system info. 中文关键词: 主机名 主机 名称 系统 信息")
        self.manager.register_tool(host_tool)

        matched = self.manager.find_relevant_tools("查询一下主机名称", ["查询", "主机", "名称"])

        self.assertEqual([item["name"] for item in matched], ["get_system_info"])

    def test_directory_file_count_query_matches_list_directory_tool(self):
        list_tool = FakeTool("list_directory", "列出工作区内目录项（安全边界：仅允许工作区内路径）。")
        self.manager.register_tool(list_tool)

        matched = self.manager.find_relevant_tools("查一下你现在目录下有多少文件", ["文件数量", "目录", "统计"])

        self.assertEqual([item["name"] for item in matched], ["list_directory"])
        self.assertIn("目录", matched[0]["matched_terms"])

    def test_tool_match_uses_relaxed_fallback_for_execution_queries(self):
        borderline_tool = FakeTool("status_lookup", "Lookup system status and service metadata. 中文关键词: 系统")
        self.manager.register_tool(borderline_tool)

        matched = self.manager.find_relevant_tools("查一下并执行检查", ["系统", "状态", "检查", "服务", "元数据"])

        self.assertEqual([item["name"] for item in matched], ["status_lookup"])
        self.assertIn("relaxed_tool_threshold", matched[0]["route_reason"])

    def test_read_only_file_lookup_excludes_write_tools(self):
        list_tool = FakeTool("list_directory", "列出工作区内目录项（安全边界：仅允许工作区内路径）。")
        write_tool = FakeTool("write_text_file", "在工作区内写入 UTF-8 文本文件。默认不覆盖已存在文件，可选 append。")
        self.manager.register_tools([list_tool, write_tool])

        matched = self.manager.find_relevant_tools("查找名为 agent.md 的文件", ["查找", "agent.md", "文件"])

        self.assertEqual([item["name"] for item in matched], ["list_directory"])

    def test_file_search_does_not_route_builtin_terminal_tool(self):
        bash_tool = FakeTool("bash", "安全执行终端命令，可用于 rg、git、python、pytest 和工作区内文件搜索。")
        list_tool = FakeTool("list_directory", "列出工作区内目录项（安全边界：仅允许工作区内路径）。")
        self.manager.register_tools([list_tool, bash_tool])

        matched = self.manager.find_relevant_tools("找下 core.py 文件在哪个文件夹下", ["core.py", "文件", "路径", "查找"])

        self.assertEqual([item["name"] for item in matched], ["list_directory"])

    def test_terminal_command_query_does_not_route_builtin_terminal_tool(self):
        bash_tool = FakeTool("bash", "安全执行终端命令，可用于 git status、python --version 和 pytest。")
        grep_tool = FakeTool("grep_text", "在工作区文本文件中搜索关键词。")
        self.manager.register_tools([grep_tool, bash_tool])

        matched = self.manager.find_relevant_tools("帮我执行 git status 看下当前仓库状态", ["git", "status", "终端", "命令"])

        self.assertEqual(matched, [])


if __name__ == "__main__":
    unittest.main()

