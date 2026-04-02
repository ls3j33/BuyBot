from typing import Optional
from src.ingestion.vector_store import VectorStore


# Глобальный экземпляр векторного хранилища - КЭШИРУЕМ для избежания повторной загрузки модели
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Получить экземпляр векторного хранилища (кэшируется для избежания повторной загрузки модели)"""
    global _vector_store

    # Кэшируем экземпляр чтобы не загружать embedding-модель повторно
    if _vector_store is None:
        _vector_store = VectorStore(collection_name="products")
    return _vector_store
