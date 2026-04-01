from langchain_ollama import ChatOllama
from src.config.settings import settings


def get_llm() -> ChatOllama:
    """Получить LLM модель"""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        temperature=0.7,
    )
