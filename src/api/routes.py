from fastapi import APIRouter, HTTPException
from src.api.schemas import ChatRequest, ChatResponse, ProductInfo, StatsResponse, HealthResponse
from src.agents import process_query
from src.tools.vector_search import get_vector_store

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Отправить запрос чат-боту")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Обработать запрос пользователя и вернуть ответ.

    **Запрос:**
    - `query`: Текст вопроса о товарах в супермаркете
    - `conversation_id`: Опциональный ID сессии

    **Ответ:**
    - `response`: Текст ответа от бота
    - `search_results`: Список найденных товаров
    - `confidence`: Уровень уверенности (high/medium/low)
    - `success`: Флаг успешной обработки
    - `error`: Текст ошибки если произошла

    **Примеры запросов:**
    - "Какой сыр лучше для пиццы?"
    - "Что нужно для борща?"
    - "Найти молоко до 100 рублей"
    """
    try:
        result = await process_query(
            query=request.query,
            conversation_history=None
        )
        
        # Преобразование результатов поиска в ProductInfo
        products = []
        search_results = result.get("search_results") or []
        for item in search_results:
            metadata = item.get("metadata", {})
            products.append(ProductInfo(
                id=item.get("id", ""),
                name=metadata.get("name", "N/A"),
                category=metadata.get("category", "N/A"),
                price=metadata.get("price", "N/A"),
                distance=item.get("distance")
            ))
        
        return ChatResponse(
            response=result.get("response", ""),
            search_results=products,
            confidence=result.get("confidence", "medium"),
            success=result.get("is_complete", False),
            error=result.get("error")
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )


@router.get("/stats", response_model=StatsResponse, summary="Статистика базы товаров")
async def get_stats() -> StatsResponse:
    """
    Получить статистику векторной базы данных.
    
    Возвращает количество документов в коллекции и статус готовности.
    """
    try:
        store = get_vector_store()
        stats = store.get_stats()
        
        return StatsResponse(
            collection_name=stats["collection_name"],
            total_documents=stats["total_documents"],
            is_ready=stats["total_documents"] > 0
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse, summary="Проверка здоровья сервиса")
async def health_check() -> HealthResponse:
    """
    Проверить работоспособность сервиса.
    
    Используется для health checks в Docker и мониторинга.
    """
    return HealthResponse(status="ok", service="buybot")
