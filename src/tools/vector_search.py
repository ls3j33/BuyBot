from typing import Optional
from langchain_core.tools import tool
from src.ingestion.vector_store import VectorStore


# Глобальный экземпляр векторного хранилища - НЕ КЭШИРУЕМ
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Получить экземпляр векторного хранилища (всегда новый для правильной модели)"""
    global _vector_store
    
    # Всегда создаём новый экземпляр чтобы использовать актуальную переменную окружения
    return VectorStore(collection_name="products")


@tool
def vector_search(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None
) -> list[dict]:
    """
    Поиск релевантных товаров в базе данных Пятёрочки.
    
    Использует векторный поиск на основе эмбеддингов и косинусного расстояния.
    
    Args:
        query: Текст запроса для поиска
        n_results: Количество результатов для возврата (по умолчанию 5)
        category: Опциональный фильтр по категории товара
    
    Returns:
        Список найденных товаров с метаданными:
        - id: ID документа
        - document: Текст документа
        - metadata: Метаданные (name, category, price, brand, country)
        - distance: Расстояние до запроса (косинусное)
    """
    store = get_vector_store()
    results = store.search(
        query=query,
        n_results=n_results,
        filter_category=category
    )
    return results


@tool
def get_collection_stats() -> dict:
    """
    Получить статистику коллекции товаров.
    
    Returns:
        Словарь со статистикой:
        - collection_name: Имя коллекции
        - total_documents: Общее количество документов
    """
    store = get_vector_store()
    return store.get_stats()
