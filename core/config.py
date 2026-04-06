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
        self.workspace_root = self.base_dir
        self.models_config = {}
        self.default_model_key = ""
        self.mcp_dir = "mcp_servers"
        self.tool_dir = "tools"
        self.skill_dir = "skills"
        self.memory_db_path = os.path.join("memory", "memory.db")
        self.memory_backup_dir = os.path.join("memory", "backups")
        self.log_dir = "logs"
        self.llm_log_file = "llm_trace.jsonl"
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
        self.lite_chat_enabled = True
        self.lite_chat_persist_memory = False
        self.lite_chat_patterns = [
            r"^(你好|您好|嗨|哈喽|hello|hi|hey)[!！。\\s]*$",
            r"^(早上好|中午好|下午好|晚上好|晚安)[!！。\\s]*$",
            r"^(在吗|有人吗|忙吗)[?？!！。\\s]*$",
            r"^(谢谢|感谢|辛苦了|thx|thanks)[!！。\\s]*$",
        ]
        self.intent_rewrite_enabled = True
        self.extra_write_roots: list[str] = []
        self.llm_config = LLMConfig()
        self._load_config()

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base_dir, path))

    def resolve_workspace_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.workspace_root, path))

    def get_workspace_root(self) -> str:
        return os.path.abspath(self.workspace_root)

    def set_workspace_root(self, path: str) -> str:
        normalized = str(path or "").strip()
        if not normalized:
            self.workspace_root = self.base_dir
        else:
            resolved = normalized if os.path.isabs(normalized) else os.path.join(self.base_dir, normalized)
            self.workspace_root = os.path.abspath(resolved)
        return self.get_workspace_root()

    def _normalize_root_path(self, path: str) -> str:
        resolved = self.resolve_workspace_path(path)
        return os.path.abspath(resolved)

    def list_write_roots(self) -> list[str]:
        roots = [self.get_workspace_root()]
        for item in self.extra_write_roots:
            normalized = self._normalize_root_path(str(item))
            if normalized not in roots:
                roots.append(normalized)
        return roots

    def grant_write_root(self, path: str) -> str:
        normalized = self._normalize_root_path(path)
        if normalized == self.get_workspace_root():
            return normalized
        if normalized not in self.extra_write_roots:
            self.extra_write_roots.append(normalized)
        return normalized

    def revoke_write_root(self, path: str) -> bool:
        normalized = self._normalize_root_path(path)
        if normalized in self.extra_write_roots:
            self.extra_write_roots.remove(normalized)
            return True
        return False

    def clear_write_roots(self) -> None:
        self.extra_write_roots = []

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

            self.workspace_root = self._read_value(data, "paths.workspace_root", "workspace_root", self.base_dir)
            self.set_workspace_root(self.workspace_root)
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
            self.llm_log_file = self._read_value(data, "llm.logging.file_name", "llm_log_file", "llm_trace.jsonl")
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
            self.lite_chat_enabled = bool(
                self._read_value(
                    data,
                    "conversation.lite_chat.enabled",
                    "lite_chat_enabled",
                    True,
                )
            )
            self.lite_chat_persist_memory = bool(
                self._read_value(
                    data,
                    "conversation.lite_chat.persist_memory",
                    "lite_chat_persist_memory",
                    False,
                )
            )
            configured_lite_chat_patterns = self._read_value(
                data,
                "conversation.lite_chat.patterns",
                "lite_chat_patterns",
                self.lite_chat_patterns,
            )
            if isinstance(configured_lite_chat_patterns, list):
                sanitized_patterns = [
                    str(item).strip() for item in configured_lite_chat_patterns if str(item).strip()
                ]
                if sanitized_patterns:
                    self.lite_chat_patterns = sanitized_patterns
            self.intent_rewrite_enabled = bool(
                self._read_value(
                    data,
                    "conversation.intent_rewrite.enabled",
                    "intent_rewrite_enabled",
                    True,
                )
            )
            configured_write_roots = self._read_value(
                data,
                "tools.write.extra_roots",
                "extra_write_roots",
                [],
            )
            if isinstance(configured_write_roots, list):
                self.extra_write_roots = []
                for item in configured_write_roots:
                    value = str(item or "").strip()
                    if not value:
                        continue
                    normalized = self._normalize_root_path(value)
                    if normalized not in self.extra_write_roots and normalized != self.get_workspace_root():
                        self.extra_write_roots.append(normalized)
            
            # Load default model
            if self.default_model_key in self.models_config:
                cfg = self.models_config[self.default_model_key]
                self.llm_config = LLMConfig(**cfg)
        
config = AppConfig()
