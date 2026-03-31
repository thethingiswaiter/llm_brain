import importlib
import json
import logging
import os
import sys
import tempfile
import unittest

from config import config
from mcp_servers.mcp_manager import MCPManager


class MCPManagerTransportTests(unittest.TestCase):
    def setUp(self):
        self.manager = MCPManager()
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.server_path = os.path.join(self.project_root, "mcp_servers", "system_mcp_server.py")

    def test_load_python_stdio_server_and_invoke_remote_tool(self):
        success, message, server_info, tools = self.manager.load_server(self.server_path)

        self.assertTrue(success, message)
        self.assertEqual(server_info["transport"], "stdio")
        self.assertEqual(len(self.manager.list_servers()), 1)
        tool_names = {tool.name for tool in tools}
        self.assertIn("get_system_info", tool_names)
        self.assertIn("inspect_file_system_path", tool_names)
        self.assertIn(server_info["connection_key"], self.manager._remote_connections)

        system_info_tool = next(tool for tool in tools if tool.name == "get_system_info")
        result = system_info_tool.invoke({})

        self.assertIn("workspace_root", result)
        self.assertIn("allowed_roots", result)

        success, unload_message, unloaded_info = self.manager.unload_server(self.server_path)

        self.assertTrue(success, unload_message)
        self.assertEqual(unloaded_info["name"], server_info["name"])
        self.assertEqual(self.manager.list_servers(), [])
        self.assertEqual(self.manager._remote_connections, {})

    def test_load_stdio_config_file_and_call_security_policy_tool(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = os.path.join(tempdir, "system_stdio.json")
            with open(config_path, "w", encoding="utf-8") as file_handle:
                json.dump(
                    {
                        "name": "system_stdio",
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [self.server_path],
                        "cwd": self.project_root,
                        "timeout_seconds": 20,
                    },
                    file_handle,
                )

            success, message, server_info, tools = self.manager.load_server(config_path)

        self.assertTrue(success, message)
        self.assertEqual(server_info["name"], "system_stdio")
        policy_tool = next(tool for tool in tools if tool.name == "get_mcp_security_policy")
        result = policy_tool.invoke({})

        self.assertIn("allowed_command_prefixes", result)

    def test_runtime_error_triggers_connection_recreation(self):
        success, message, server_info, tools = self.manager.load_server(self.server_path)

        self.assertTrue(success, message)
        connection_key = server_info["connection_key"]
        original_connection = self.manager._remote_connections[connection_key]
        original_connection._runtime_error = RuntimeError("simulated connection loss")

        system_info_tool = next(tool for tool in tools if tool.name == "get_system_info")
        result = system_info_tool.invoke({})
        replacement_connection = self.manager._remote_connections[connection_key]

        self.assertIn("workspace_root", result)
        self.assertIsNot(original_connection, replacement_connection)

    def test_refresh_server_reloads_tools(self):
        success, message, server_info, tools = self.manager.load_server(self.server_path)

        self.assertTrue(success, message)
        refresh_success, refresh_message, refreshed_info, refreshed_tools = self.manager.refresh_server("system_mcp_server")

        self.assertTrue(refresh_success, refresh_message)
        self.assertEqual(refreshed_info["name"], server_info["name"])
        self.assertEqual({tool.name for tool in refreshed_tools}, {tool.name for tool in tools})

    def tearDown(self):
        self.manager.close_all()


class AgentMCPManagementTests(unittest.TestCase):
    def setUp(self):
        self.original_memory_db_path = config.memory_db_path
        self.original_memory_backup_dir = config.memory_backup_dir
        self.original_state_snapshot_dir = config.state_snapshot_dir
        self.original_log_dir = config.log_dir
        self.original_llm_log_file = config.llm_log_file
        self.original_mcp_dir = config.mcp_dir
        self.tempdir = tempfile.TemporaryDirectory()

        config.memory_db_path = os.path.join(self.tempdir.name, "memory.db")
        config.memory_backup_dir = os.path.join(self.tempdir.name, "backups")
        config.state_snapshot_dir = os.path.join(self.tempdir.name, "snapshots")
        config.log_dir = os.path.join(self.tempdir.name, "logs")
        config.llm_log_file = "mcp_agent_management.log"
        config.mcp_dir = os.path.join(self.tempdir.name, "mcp_servers")
        os.makedirs(config.resolve_path(config.mcp_dir), exist_ok=True)

        import agent_core

        self.agent_core_module = importlib.reload(agent_core)
        self.agent = self.agent_core_module.AgentCore()
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.server_path = os.path.join(self.project_root, "mcp_servers", "system_mcp_server.py")

    def tearDown(self):
        try:
            self.agent.mcp.close_all()
        except Exception:
            pass
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
        config.mcp_dir = self.original_mcp_dir
        self.tempdir.cleanup()

    def test_agent_load_refresh_and_unload_mcp_server_updates_tool_registry(self):
        success, message = self.agent.load_mcp_server(self.server_path)

        self.assertTrue(success, message)
        self.assertEqual(len(self.agent.list_mcp_servers()), 1)
        self.assertIn("get_system_info", self.agent.loaded_tool_names)
        self.assertIn("get_system_info", self.agent.skills.loaded_tool_skills)
        initial_tool_count = len([tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_system_info"])
        self.assertEqual(initial_tool_count, 1)

        refresh_success, refresh_message = self.agent.refresh_mcp_server("system_mcp_server")

        self.assertTrue(refresh_success, refresh_message)
        refreshed_tool_count = len([tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_system_info"])
        self.assertEqual(refreshed_tool_count, 1)

        unload_success, unload_message = self.agent.unload_mcp_server("system_mcp_server")

        self.assertTrue(unload_success, unload_message)
        self.assertEqual(self.agent.list_mcp_servers(), [])
        self.assertNotIn("get_system_info", self.agent.loaded_tool_names)
        self.assertNotIn("get_system_info", self.agent.skills.loaded_tool_skills)
        self.assertEqual(len([tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_system_info"]), 0)

    def test_agent_loading_same_mcp_server_twice_does_not_duplicate_tools(self):
        first_success, first_message = self.agent.load_mcp_server(self.server_path)
        second_success, second_message = self.agent.load_mcp_server(self.server_path)

        self.assertTrue(first_success, first_message)
        self.assertTrue(second_success, second_message)
        self.assertIn("already registered", second_message)
        self.assertEqual(len([tool for tool in self.agent.tools if getattr(tool, "name", "") == "get_system_info"]), 1)

    def test_agent_unload_unknown_mcp_server_returns_error(self):
        success, message = self.agent.unload_mcp_server("missing_server")

        self.assertFalse(success)
        self.assertIn("not loaded", message)


if __name__ == "__main__":
    unittest.main()