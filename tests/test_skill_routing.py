import os
import tempfile
import unittest

from config import config
from skills_md.skill_parser import SkillManager


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
        return self.manager.load_skill_md(filename, force_reload=True)

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


if __name__ == "__main__":
    unittest.main()