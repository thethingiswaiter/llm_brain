import os
import json
import getpass
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict

class LLMConfig(BaseModel):
    provider: str = "ollama" # "ollama" | "openai"
    model: str = "llama3:latest"
    base_url: Optional[str] = None
    api_key: Optional[str] = None

class AppConfig:
    def __init__(self):
        # core/config.py lives under core/, but runtime paths should resolve from project root.
        self.base_dir = str(Path(__file__).resolve().parents[1])
        self.config_path = os.path.join(self.base_dir, "config.json")
        self.models_config = {}
        self.default_model_key = ""
        self.mcp_dir = "mcp_servers"
        self.tool_dir = "tools"
        self.skill_dir = "skills"
        self.memory_db_path = os.path.join("memory", "memory.db")
        self.memory_backup_dir = os.path.join("memory", "backups")
        self.log_dir = "logs"
        self.llm_log_file = "llm_trace.log"
        self.llm_log_max_chars = 4000
        self.state_snapshot_dir = os.path.join("runtime_state", "snapshots")
        self.audit_log_dir = os.path.join("runtime_state", "audit")
        self.llm_timeout_seconds = 45
        self.tool_timeout_seconds = 20
        self.request_timeout_seconds = 120
        self.tool_cancellation_grace_seconds = 0.2
        self.log_retention_days = 7
        self.snapshot_retention_days = 7
        self.audit_log_retention_days = 14
        self.memory_backup_retention_days = 14
        self.log_retention_max_files = 20
        self.snapshot_retention_max_request_dirs = 200
        self.audit_log_retention_max_files = 50
        self.memory_backup_retention_max_files = 20
        self.log_retention_max_bytes = 0
        self.snapshot_retention_max_bytes = 0
        self.audit_log_retention_max_bytes = 0
        self.memory_backup_retention_max_bytes = 0
        self.retention_auto_prune_enabled = True
        self.retention_auto_prune_min_interval_seconds = 300
        self.historical_tool_failure_reroute_threshold = 2
        self.historical_tool_failure_severity_threshold = 6
        self.prompt_skill_min_overlap = 1
        self.prompt_skill_min_match_ratio = 0.34
        self.tool_skill_min_overlap = 1
        self.tool_skill_min_match_ratio = 0.34
        self.llm_config = LLMConfig()
        self._load_config()

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base_dir, path))

    @staticmethod
    def _get_nested(data: Dict, path: str, default=None):
        node = data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _read_value(self, data: Dict, new_path: str, legacy_key: str, default=None):
        value = self._get_nested(data, new_path, None)
        if value is not None:
            return value
        return data.get(legacy_key, default)

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.models_config = self._read_value(data, "llm.models", "models", {})
            self.default_model_key = self._read_value(data, "llm.default_model_key", "default_model", "")
            self.mcp_dir = self._read_value(data, "paths.mcp_servers_dir", "mcp_dir", "mcp_servers")
            self.tool_dir = self._get_nested(data, "paths.tools_dir", "tools")
            self.skill_dir = self._get_nested(data, "paths.skill_dir", "skills")
            self.memory_db_path = self._read_value(
                data,
                "paths.memory.db_file",
                "memory_db_path",
                os.path.join("memory", "memory.db"),
            )
            self.memory_backup_dir = self._read_value(
                data,
                "paths.memory.backup_dir",
                "memory_backup_dir",
                os.path.join("memory", "backups"),
            )
            self.log_dir = self._read_value(data, "paths.logs_dir", "log_dir", "logs")
            self.llm_log_file = self._read_value(data, "llm.logging.file_name", "llm_log_file", "llm_trace.log")
            self.llm_log_max_chars = self._read_value(data, "llm.logging.max_chars", "llm_log_max_chars", 4000)
            self.state_snapshot_dir = self._read_value(
                data,
                "paths.runtime.snapshot_dir",
                "state_snapshot_dir",
                os.path.join("runtime_state", "snapshots"),
            )
            self.audit_log_dir = self._read_value(
                data,
                "paths.runtime.audit_dir",
                "audit_log_dir",
                os.path.join("runtime_state", "audit"),
            )
            self.llm_timeout_seconds = self._read_value(data, "llm.timeouts.invoke_seconds", "llm_timeout_seconds", 45)
            self.tool_timeout_seconds = self._read_value(data, "timeouts.tool_invoke_seconds", "tool_timeout_seconds", 20)
            self.request_timeout_seconds = self._read_value(
                data,
                "timeouts.request_total_seconds",
                "request_timeout_seconds",
                120,
            )
            self.tool_cancellation_grace_seconds = self._read_value(
                data,
                "timeouts.tool_cancellation_grace_seconds",
                "tool_cancellation_grace_seconds",
                0.2,
            )
            self.log_retention_days = self._read_value(data, "retention.logs.max_age_days", "log_retention_days", 7)
            self.snapshot_retention_days = self._read_value(
                data,
                "retention.snapshots.max_age_days",
                "snapshot_retention_days",
                7,
            )
            self.audit_log_retention_days = self._read_value(
                data,
                "retention.audit.max_age_days",
                "audit_log_retention_days",
                14,
            )
            self.memory_backup_retention_days = self._read_value(
                data,
                "retention.memory_backups.max_age_days",
                "memory_backup_retention_days",
                14,
            )
            self.log_retention_max_files = self._read_value(
                data,
                "retention.logs.max_files",
                "log_retention_max_files",
                20,
            )
            self.snapshot_retention_max_request_dirs = self._read_value(
                data,
                "retention.snapshots.max_request_dirs",
                "snapshot_retention_max_request_dirs",
                200,
            )
            self.audit_log_retention_max_files = self._read_value(
                data,
                "retention.audit.max_files",
                "audit_log_retention_max_files",
                50,
            )
            self.memory_backup_retention_max_files = self._read_value(
                data,
                "retention.memory_backups.max_files",
                "memory_backup_retention_max_files",
                20,
            )
            self.log_retention_max_bytes = self._read_value(
                data,
                "retention.logs.max_bytes",
                "log_retention_max_bytes",
                0,
            )
            self.snapshot_retention_max_bytes = self._read_value(
                data,
                "retention.snapshots.max_bytes",
                "snapshot_retention_max_bytes",
                0,
            )
            self.audit_log_retention_max_bytes = self._read_value(
                data,
                "retention.audit.max_bytes",
                "audit_log_retention_max_bytes",
                0,
            )
            self.memory_backup_retention_max_bytes = self._read_value(
                data,
                "retention.memory_backups.max_bytes",
                "memory_backup_retention_max_bytes",
                0,
            )
            self.retention_auto_prune_enabled = self._read_value(
                data,
                "retention.auto_prune.enabled",
                "retention_auto_prune_enabled",
                True,
            )
            self.retention_auto_prune_min_interval_seconds = self._read_value(
                data,
                "retention.auto_prune.min_interval_seconds",
                "retention_auto_prune_min_interval_seconds",
                300,
            )
            self.historical_tool_failure_reroute_threshold = self._read_value(
                data,
                "routing.historical_tool_failures.reroute_threshold",
                "historical_tool_failure_reroute_threshold",
                2,
            )
            self.historical_tool_failure_severity_threshold = self._read_value(
                data,
                "routing.historical_tool_failures.severity_threshold",
                "historical_tool_failure_severity_threshold",
                6,
            )
            self.prompt_skill_min_overlap = self._read_value(
                data,
                "routing.skill_match.prompt.min_overlap",
                "prompt_skill_min_overlap",
                1,
            )
            self.prompt_skill_min_match_ratio = self._read_value(
                data,
                "routing.skill_match.prompt.min_ratio",
                "prompt_skill_min_match_ratio",
                0.34,
            )
            self.tool_skill_min_overlap = self._read_value(
                data,
                "routing.skill_match.tool.min_overlap",
                "tool_skill_min_overlap",
                1,
            )
            self.tool_skill_min_match_ratio = self._read_value(
                data,
                "routing.skill_match.tool.min_ratio",
                "tool_skill_min_match_ratio",
                0.34,
            )
            
            # Load default model
            if self.default_model_key in self.models_config:
                cfg = self.models_config[self.default_model_key]
                self.llm_config = LLMConfig(**cfg)
        
config = AppConfig()
