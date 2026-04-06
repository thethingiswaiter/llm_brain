import os
import tempfile
import unittest
from pathlib import Path

from mcp_servers import system_mcp_server
from mcp_servers.system_mcp_server import collect_system_info, execute_terminal_command, get_mcp_security_policy, inspect_file_system_path
from core.config import config


class SystemMCPServerTests(unittest.TestCase):
    def setUp(self):
        self._original_audit_path = system_mcp_server.AUDIT_LOG_PATH
        self._original_workspace_root = config.get_workspace_root()
        self._audit_dir = tempfile.TemporaryDirectory()
        system_mcp_server.AUDIT_LOG_PATH = Path(self._audit_dir.name) / "system_mcp_audit.jsonl"

    def tearDown(self):
        system_mcp_server.AUDIT_LOG_PATH = self._original_audit_path
        config.set_workspace_root(self._original_workspace_root)
        self._audit_dir.cleanup()

    def test_execute_terminal_command_runs_simple_command(self):
        result = execute_terminal_command("echo hello", timeout_seconds=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello", result["stdout"].lower())
        self.assertEqual(result["command_prefix"], "echo")
        self.assertTrue(result["shell"])

    def test_execute_terminal_command_blocks_destructive_patterns(self):
        result = execute_terminal_command("Remove-Item -Recurse -Force *")

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])

    def test_execute_terminal_command_blocks_non_allowlisted_prefix(self):
        result = execute_terminal_command("curl https://example.com")

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertEqual(result["command_prefix"], "curl")

    def test_execute_terminal_command_blocks_shell_chaining_operator(self):
        result = execute_terminal_command("echo hello && whoami")

        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertIn("operator", result["reason"].lower())

    def test_collect_system_info_returns_expected_fields(self):
        result = collect_system_info()

        self.assertTrue(result["ok"])
        self.assertTrue(result["hostname"])
        self.assertTrue(result["workspace_root"])
        self.assertIn("python_version", result)
        self.assertIn("allowed_roots", result)
        self.assertIn("allowed_command_prefixes", result)

    def test_get_mcp_security_policy_returns_allowlists(self):
        result = get_mcp_security_policy()

        self.assertTrue(result["ok"])
        self.assertIn("allowed_roots", result)
        self.assertIn("allowed_command_prefixes", result)
        self.assertTrue(result["audit_log_path"])

    def test_inspect_file_system_path_reads_file_preview(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "sample.txt")
            with open(file_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("line1\nline2\nline3\n")

            result = inspect_file_system_path(file_path, include_preview=True, allow_outside_workspace=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["is_file"])
        self.assertIn("line1", result["preview"])

    def test_inspect_file_system_path_blocks_paths_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "sample.txt")
            with open(file_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("line1\n")

            result = inspect_file_system_path(file_path)

        self.assertFalse(result["ok"])
        self.assertIn("outside allowed roots", result["error"])

    def test_inspect_file_system_path_lists_directory_entries(self):
        with tempfile.TemporaryDirectory() as tempdir:
            open(os.path.join(tempdir, "a.txt"), "w", encoding="utf-8").close()
            os.mkdir(os.path.join(tempdir, "nested"))

            result = inspect_file_system_path(tempdir, allow_outside_workspace=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["is_dir"])
        self.assertEqual(result["entry_count"], 2)

    def test_calls_write_audit_log(self):
        execute_terminal_command("echo hello", timeout_seconds=5)
        collect_system_info()

        with open(system_mcp_server.AUDIT_LOG_PATH, "r", encoding="utf-8") as audit_file:
            lines = audit_file.readlines()

        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("execute_terminal_command", lines[0])

    def test_security_policy_uses_configured_workspace_root(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config.set_workspace_root(tempdir)

            result = get_mcp_security_policy()

        self.assertEqual(result["workspace_root"], str(Path(tempdir).resolve()))
        self.assertIn(str(Path(tempdir).resolve()), result["allowed_roots"])


if __name__ == "__main__":
    unittest.main()

