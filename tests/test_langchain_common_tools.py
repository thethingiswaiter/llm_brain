import tempfile
import unittest
from pathlib import Path

from tools import langchain_common_tools as tools_mod


class LangchainCommonToolsTests(unittest.TestCase):
    def test_tools_list_contains_common_tools(self):
        names = {getattr(tool_obj, "name", "") for tool_obj in tools_mod.tools}
        self.assertIn("get_current_time", names)
        self.assertIn("calculator", names)
        self.assertIn("list_directory", names)
        self.assertIn("read_text_file", names)
        self.assertIn("grep_text", names)
        self.assertIn("write_text_file", names)
        self.assertIn("json_query", names)

    def test_calculator_supports_basic_expression(self):
        result = tools_mod.calculator.invoke({"expression": "(2 + 3) * 4"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], 20.0)

    def test_calculator_blocks_unsafe_expression(self):
        result = tools_mod.calculator.invoke({"expression": "__import__('os').system('whoami')"})

        self.assertFalse(result["ok"])

    def test_read_text_file_returns_line_slice(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            file_path = root / "demo.txt"
            file_path.write_text("a\nb\nc\n", encoding="utf-8")

            original = tools_mod._workspace_root
            tools_mod._workspace_root = lambda: root
            try:
                result = tools_mod.read_text_file.invoke(
                    {"path": "demo.txt", "start_line": 2, "end_line": 3}
                )
            finally:
                tools_mod._workspace_root = original

        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "b\nc")

    def test_grep_text_finds_matches(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
            (root / "b.txt").write_text("HELLO agent\n", encoding="utf-8")

            original = tools_mod._workspace_root
            tools_mod._workspace_root = lambda: root
            try:
                result = tools_mod.grep_text.invoke({"query": "hello", "path": ".", "max_results": 10})
            finally:
                tools_mod._workspace_root = original

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(result["matches"]), 2)

    def test_write_text_file_creates_and_blocks_second_write_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)

            original = tools_mod._workspace_root
            tools_mod._workspace_root = lambda: root
            try:
                created = tools_mod.write_text_file.invoke(
                    {"path": "out.txt", "content": "hello", "overwrite": False, "append": False}
                )
                blocked = tools_mod.write_text_file.invoke(
                    {"path": "out.txt", "content": "world", "overwrite": False, "append": False}
                )
            finally:
                tools_mod._workspace_root = original

        self.assertTrue(created["ok"])
        self.assertFalse(blocked["ok"])

    def test_json_query_reads_nested_key_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "payload.json").write_text(
                '{"user": {"profile": {"name": "alice"}}, "items": [{"id": 42}]}',
                encoding="utf-8",
            )

            original = tools_mod._workspace_root
            tools_mod._workspace_root = lambda: root
            try:
                result = tools_mod.json_query.invoke(
                    {"path": "payload.json", "key_path": "user.profile.name"}
                )
                list_result = tools_mod.json_query.invoke(
                    {"path": "payload.json", "key_path": "items.0.id"}
                )
            finally:
                tools_mod._workspace_root = original

        self.assertTrue(result["ok"])
        self.assertEqual(result["value"], "alice")
        self.assertTrue(list_result["ok"])
        self.assertEqual(list_result["value"], 42)


if __name__ == "__main__":
    unittest.main()


