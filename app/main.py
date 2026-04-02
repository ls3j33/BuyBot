from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import sys
import os
from pathlib import Path

# Загружаем .env файл явно
env_path = Path(__file__).parent.parent / ".env"
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Устанавливаем переменные ПОСЛЕ загрузки .env (override)
os.environ['EMBEDDING_MODEL'] = os.environ.get('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large').strip()
# Используем абсолютный путь для CHROMA_DB_PATH
chroma_db_path = os.environ.get('CHROMA_DB_PATH', './chroma_db_e5')
if not os.path.isabs(chroma_db_path):
    chroma_db_path = str(Path(__file__).parent.parent / chroma_db_path)
os.environ['CHROMA_DB_PATH'] = chroma_db_path

# Проверяем переменные окружения
print(f"EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL')}")
print(f"CHROMA_DB_PATH: {os.environ.get('CHROMA_DB_PATH')}")

# Добавляем src в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импортируем settings ПОСЛЕ установки переменных
from src.config.settings import settings
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.

    Инициализация при запуске и очистка при остановке.
    """
    # Startup
    print("=" * 50)
    print("BuyBot запускается...")
    print(f"LLM модель: {settings.llm_model}")
    print(f"URL Ollama: {settings.ollama_base_url}")
    print(f"Embedding модель: {settings.embedding_model}")
    print("=" * 50)

    # Инициализация векторного хранилища (импорт внутри функции)
    try:
        from src.tools.vector_search import get_vector_store
        store = get_vector_store()
        stats = store.get_stats()
        print(f"Векторная БД загружена: {stats['total_documents']} документов")
    except Exception as e:
        print(f"Предупреждение: Ошибка инициализации БД: {e}")

    yield

    # Shutdown
    print("BuyBot остановлен")


def create_app() -> FastAPI:
    """
    Создать и настроить FastAPI приложение.

    Returns:
        Настроенное FastAPI приложение
    """
    app = FastAPI(
        title="BuyBot API",
        description="Умный помощник покупок на базе LLM и RAG",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # CORS middleware - ДОЛЖЕН БЫТЬ ДО include_router
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )

    # Подключение роутов
    app.include_router(router, prefix="/api/v1")

    return app


# Создание приложения
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
