import os
import json
import getpass
from pydantic import BaseModel
from typing import Optional, Dict

class LLMConfig(BaseModel):
    provider: str = "ollama" # "ollama" | "openai"
    model: str = "llama3:latest"
    base_url: Optional[str] = None
    api_key: Optional[str] = None

class AppConfig:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.base_dir, "config.json")
        self.models_config = {}
        self.default_model_key = ""
        self.mcp_dir = "mcp_servers"
        self.skills_dir = "skills"
        self.skills_md_dir = "skills_md"
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

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.models_config = data.get("models", {})
            self.default_model_key = data.get("default_model", "")
            self.mcp_dir = data.get("mcp_dir", "mcp_servers")
            self.skills_dir = data.get("skills_dir", "skills")
            self.skills_md_dir = data.get("skills_md_dir", "skills_md")
            self.memory_db_path = data.get("memory_db_path", os.path.join("memory", "memory.db"))
            self.memory_backup_dir = data.get("memory_backup_dir", os.path.join("memory", "backups"))
            self.log_dir = data.get("log_dir", "logs")
            self.llm_log_file = data.get("llm_log_file", "llm_trace.log")
            self.llm_log_max_chars = data.get("llm_log_max_chars", 4000)
            self.state_snapshot_dir = data.get("state_snapshot_dir", os.path.join("runtime_state", "snapshots"))
            self.audit_log_dir = data.get("audit_log_dir", os.path.join("runtime_state", "audit"))
            self.llm_timeout_seconds = data.get("llm_timeout_seconds", 45)
            self.tool_timeout_seconds = data.get("tool_timeout_seconds", 20)
            self.request_timeout_seconds = data.get("request_timeout_seconds", 120)
            self.tool_cancellation_grace_seconds = data.get("tool_cancellation_grace_seconds", 0.2)
            self.log_retention_days = data.get("log_retention_days", 7)
            self.snapshot_retention_days = data.get("snapshot_retention_days", 7)
            self.audit_log_retention_days = data.get("audit_log_retention_days", 14)
            self.memory_backup_retention_days = data.get("memory_backup_retention_days", 14)
            self.log_retention_max_files = data.get("log_retention_max_files", 20)
            self.snapshot_retention_max_request_dirs = data.get("snapshot_retention_max_request_dirs", 200)
            self.audit_log_retention_max_files = data.get("audit_log_retention_max_files", 50)
            self.memory_backup_retention_max_files = data.get("memory_backup_retention_max_files", 20)
            self.log_retention_max_bytes = data.get("log_retention_max_bytes", 0)
            self.snapshot_retention_max_bytes = data.get("snapshot_retention_max_bytes", 0)
            self.audit_log_retention_max_bytes = data.get("audit_log_retention_max_bytes", 0)
            self.memory_backup_retention_max_bytes = data.get("memory_backup_retention_max_bytes", 0)
            self.retention_auto_prune_enabled = data.get("retention_auto_prune_enabled", True)
            self.retention_auto_prune_min_interval_seconds = data.get("retention_auto_prune_min_interval_seconds", 300)
            self.historical_tool_failure_reroute_threshold = data.get("historical_tool_failure_reroute_threshold", 2)
            self.historical_tool_failure_severity_threshold = data.get("historical_tool_failure_severity_threshold", 6)
            self.prompt_skill_min_overlap = data.get("prompt_skill_min_overlap", 1)
            self.prompt_skill_min_match_ratio = data.get("prompt_skill_min_match_ratio", 0.34)
            self.tool_skill_min_overlap = data.get("tool_skill_min_overlap", 1)
            self.tool_skill_min_match_ratio = data.get("tool_skill_min_match_ratio", 0.34)
            
            # Load default model
            if self.default_model_key in self.models_config:
                cfg = self.models_config[self.default_model_key]
                self.llm_config = LLMConfig(**cfg)
        
config = AppConfig()
