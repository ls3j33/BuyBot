"""
FastAPI‑router для публичного API бота‑покупателя.
- /chat   – отправка пользовательского запроса.
- /stats  – статистика векторного хранилища.
- /health – простая проверка «живости» сервиса.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ProductInfo,
    StatsResponse,
    HealthResponse,
)
from src.agents import process_query_graph
from src.tools.vector_search import get_vector_store

# ----------------------------------------------------------------------
#  Конфигурация роутера
# ----------------------------------------------------------------------
router = APIRouter()   # префикс указывается в main.py
app_logger = logging.getLogger("buybot.api")   # общий логгер проекта


# ----------------------------------------------------------------------
#  /chat
# ----------------------------------------------------------------------
MAX_RETURNED_PRODUCTS = 10  # ограничиваем размер ответа


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Отправить запрос чат‑боту",
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Обрабатывает пользовательский запрос и формирует ответ бота.

    Parameters
    ----------
    request : ChatRequest
        *query* – текст запроса пользователя.  
        *conversation_id* – (опционально) ID текущей сессии, позволяющий вести диалог.

    Returns
    -------
    ChatResponse
        response – текстовое сообщение от бота.  
        search_results – список найденных товаров (макс = MAX_RETURNED_PRODUCTS).  
        confidence – уровень уверенности (high/medium/low).  
        success – флаг успешной обработки.  
        error – сообщение об ошибке (если есть).

    Raises
    ------
    HTTPException
        500 – при любой внутренней ошибке.
    """
    app_logger.info("[API] /chat – запрос: %s", request.query)

    try:
        # process_query_graph может быть sync → выполняем в пуле потоков
        result = await run_in_threadpool(
            process_query_graph,
            query=request.query,
        )
    except Exception as exc:  # pragma: no cover
        app_logger.exception("[API] Ошибка обработки запроса")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка обработки запроса.",
        ) from exc

    # ------------------------------------------------------------------
    #  Формируем список товаров
    # ------------------------------------------------------------------
    raw_results = result.get("search_results") or []
    product_list: List[ProductInfo] = []

    for item in raw_results[:MAX_RETURNED_PRODUCTS]:
        meta = item.get("metadata", {})
        # Приводим типы к тем, что ожидает pydantic‑модель
        price = str(meta.get("price", "0"))

        product_list.append(
            ProductInfo(
                id=item.get("id", ""),
                name=meta.get("name", "N/A"),
                category=meta.get("category", "N/A"),
                price=price,
                distance=item.get("distance"),
            )
        )

    # Извлекаем response из agent_output
    agent_out = result.get("agent_output", {})
    response_text = agent_out.get("response_text", "") if agent_out else result.get("response", "")

    response = ChatResponse(
        response=response_text,
        search_results=product_list,
        confidence=agent_out.get("confidence", result.get("confidence", "medium")),
        success=result.get("success", False),
        error=result.get("error"),
    )

    app_logger.debug("[API] /chat – готов ответ (confidence=%s)", response.confidence)
    return response


# ----------------------------------------------------------------------
#  /stats
# ----------------------------------------------------------------------
@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Статистика базы товаров",
    status_code=status.HTTP_200_OK,
)
async def get_stats() -> StatsResponse:
    """
    Возвращает статистику векторного хранилища:
    * имя коллекции,
    * общее количество документов,
    * готовность (есть ли документы).

    Raises
    ------
    HTTPException
        500 – если не удалось достучаться до хранилища.
    """
    app_logger.info("[API] /stats запрос")
    try:
        store = await run_in_threadpool(get_vector_store)   # в случае sync‑функции
        stats = store.get_stats()
    except Exception as exc:  # pragma: no cover
        app_logger.exception("[API] Ошибка получения статистики")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения статистики.",
        ) from exc

    return StatsResponse(
        collection_name=stats.get("collection_name", ""),
        total_documents=stats.get("total_documents", 0),
        is_ready=bool(stats.get("total_documents", 0) > 0),
    )


# ----------------------------------------------------------------------
#  /health
# ----------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка здоровья сервиса",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> HealthResponse:
    """
    Эндпоинт‑«heartbeat», используемый оркестратором
    (Docker‑healthcheck, Kubernetes Liveness/Readiness).

    Returns
    -------
    HealthResponse
        status – строка «ok», если процесс работает.
        service – имя сервиса (для удобства в мониторинге).
    """
    app_logger.debug("[API] /health запрос")
    return HealthResponse(status="ok", service="buybot")
