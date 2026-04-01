import importlib
import os
import tempfile
import unittest
from textwrap import dedent

from config import config


class SkillAutoLoadingTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_skills_dir = config.skills_dir
        self.original_mcp_dir = config.mcp_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.skills_dir = os.path.join(self.tempdir.name, "skills")
        config.mcp_dir = os.path.join(self.tempdir.name, "mcp")

        os.makedirs(config.skills_dir, exist_ok=True)
        os.makedirs(config.mcp_dir, exist_ok=True)

        self._write_skill_file(
            "runtime_skill.py",
            "runtime_echo",
            "Runtime echo tool.",
            "return value",
            args_signature="value: str",
        )
        self._write_skill_file(
            "sample_demo.py",
            "sample_only",
            "Sample tool that should not auto-load.",
            "return value",
            args_signature="value: str",
        )
        self._write_skill_file(
            "test_demo.py",
            "test_only",
            "Test tool that should not auto-load.",
            "return value",
            args_signature="value: str",
        )

        import agent_core

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
        config.skills_dir = self.original_skills_dir
        config.mcp_dir = self.original_mcp_dir
        self.tempdir.cleanup()

    def _write_skill_file(self, filename: str, tool_name: str, description: str, return_expr: str, args_signature: str = ""):
        path = os.path.join(config.skills_dir, filename)
        signature = f"({args_signature})" if args_signature else "()"
        content = (
            "from langchain_core.tools import tool\n\n\n"
            f"@tool\n"
            f"def {tool_name}{signature}:\n"
            f"    \"\"\"{description}\"\"\"\n"
            f"    {return_expr}\n\n\n"
            f"tools = [{tool_name}]\n"
        )
        with open(path, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)

    def test_auto_load_skips_sample_and_test_skill_files(self):
        self.assertIn("runtime_skill.py", self.agent.loaded_python_skill_files)
        self.assertNotIn("sample_demo.py", self.agent.loaded_python_skill_files)
        self.assertNotIn("test_demo.py", self.agent.loaded_python_skill_files)

        loaded_names = {getattr(tool, "name", "") for tool in self.agent.tools}
        self.assertIn("runtime_echo", loaded_names)
        self.assertNotIn("sample_only", loaded_names)
        self.assertNotIn("test_only", loaded_names)


class AgentInitializationWithoutRuntimeSkillsTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_skills_dir = config.skills_dir
        self.original_mcp_dir = config.mcp_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.skills_dir = os.path.join(self.tempdir.name, "skills")
        config.mcp_dir = os.path.join(self.tempdir.name, "mcp")

        os.makedirs(config.skills_dir, exist_ok=True)
        os.makedirs(config.mcp_dir, exist_ok=True)

        import agent_core

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
        config.skills_dir = self.original_skills_dir
        config.mcp_dir = self.original_mcp_dir
        self.tempdir.cleanup()

    def test_agent_core_initializes_without_runtime_skills(self):
        self.assertEqual(self.agent.loaded_python_skill_files, set())
        self.assertIsNotNone(self.agent.graph)


class MCPAutoLoadingTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_skills_dir = config.skills_dir
        self.original_mcp_dir = config.mcp_dir
        self.tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.skills_dir = os.path.join(self.tempdir.name, "skills")
        config.mcp_dir = os.path.join(self.tempdir.name, "mcp")

        os.makedirs(config.skills_dir, exist_ok=True)
        os.makedirs(config.mcp_dir, exist_ok=True)

        server_path = os.path.join(config.mcp_dir, "temp_mcp_server.py")
        with open(server_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(dedent(
                '''
                try:
                    from mcp.server.fastmcp import FastMCP
                except ImportError:
                    FastMCP = None

                if FastMCP is not None:
                    mcp = FastMCP("temp-test-server")

                    @mcp.tool(name="get_temp_info")
                    def get_temp_info() -> dict:
                        """Return temp host info including hostname. 中文关键词: 主机名 主机 名称。"""
                        return {"ok": True, "hostname": "temp-host"}


                if __name__ == "__main__":
                    if FastMCP is None:
                        raise SystemExit("mcp package is required")
                    mcp.run()
                '''
            ).strip() + "\n")

        import agent_core

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
        config.skills_dir = self.original_skills_dir
        config.mcp_dir = self.original_mcp_dir
        self.tempdir.cleanup()

    def test_auto_loads_python_mcp_server_files(self):
        self.assertIn("get_temp_info", self.agent.loaded_tool_names)
        self.assertEqual(len(self.agent.list_mcp_servers()), 1)

    def test_select_active_tools_matches_contiguous_chinese_query(self):
        selected_names = [getattr(tool, "name", "") for tool in self.agent.select_active_tools("查询一下主机名称")]

        self.assertIn("get_temp_info", selected_names)


if __name__ == "__main__":
    unittest.main()
