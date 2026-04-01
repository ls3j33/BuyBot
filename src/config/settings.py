import os


class Settings:
    """Конфигурация приложения - читает переменные окружения напрямую"""

    @property
    def embedding_model(self) -> str:
        return os.environ.get('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large').strip()

    @property
    def chroma_db_path(self) -> str:
        return os.environ.get('CHROMA_DB_PATH', './chroma_db_e5')

    @property
    def ollama_base_url(self) -> str:
        return os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')

    @property
    def llm_model(self) -> str:
        return os.environ.get('LLM_MODEL', 'qwen3:4b')

    @property
    def host(self) -> str:
        return os.environ.get('HOST', '0.0.0.0')

    @property
    def port(self) -> int:
        return int(os.environ.get('PORT', '8000'))

    @property
    def embedding_8bit(self) -> bool:
        env_8bit = os.environ.get('EMBEDDING_8BIT', 'false').lower().strip()
        return env_8bit in ('true', '1', 'yes')


settings = Settings()
