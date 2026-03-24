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
        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.models_config = {}
        self.default_model_key = ""
        self.mcp_dir = "mcp_servers"
        self.skills_dir = "skills"
        self.llm_config = LLMConfig()
        self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.models_config = data.get("models", {})
            self.default_model_key = data.get("default_model", "")
            self.mcp_dir = data.get("mcp_dir", "mcp_servers")
            self.skills_dir = data.get("skills_dir", "skills")
            
            # Load default model
            if self.default_model_key in self.models_config:
                cfg = self.models_config[self.default_model_key]
                self.llm_config = LLMConfig(**cfg)
        
config = AppConfig()
