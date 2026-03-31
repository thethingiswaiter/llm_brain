from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def build_llm(cfg) -> BaseChatModel:
    if cfg.provider == "ollama":
        return ChatOllama(model=cfg.model, base_url=cfg.base_url or "http://localhost:11434")
    if cfg.provider == "openai":
        return ChatOpenAI(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key)
    raise ValueError(f"Unknown provider: {cfg.provider}")