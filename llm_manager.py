from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from config import config

class LLMManager:
    def __init__(self):
        self._current_llm = None
        self._update_llm()

    def set_model(self, provider: str, model: str, base_url: str = None, api_key: str = None):
        config.llm_config.provider = provider
        config.llm_config.model = model
        if base_url: config.llm_config.base_url = base_url
        if api_key: config.llm_config.api_key = api_key
        self._update_llm()
        return f"Successfully switched to {provider}:{model}"

    def _update_llm(self):
        cfg = config.llm_config
        if cfg.provider == "ollama":
            self._current_llm = ChatOllama(model=cfg.model, base_url=cfg.base_url or "http://localhost:11434")
        elif cfg.provider == "openai":
            self._current_llm = ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

    def get_llm(self) -> BaseChatModel:
        if not self._current_llm:
            self._update_llm()
        return self._current_llm

llm_manager = LLMManager()
