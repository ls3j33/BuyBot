"""TypedDict‑модель состояния графа LangGraph.

Содержит всю информацию, которую передают между агентами
(сообщения, запрос, решения роутера, результаты поиска,
данные рецепта, финальный вывод и статус выполнения).
"""

from __future__ import annotations

from typing import Literal, List, NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class RouterDecision(TypedDict):
    """Решение роутера о типе запроса."""
    query_type: Literal["recipe", "search"]
    confidence: float
    reason: str


class AgentState(TypedDict):
    """
    Состояние, которое передаётся между узлами LangGraph.

    Поля, помеченные `NotRequired`, могут отсутствовать в начальном
    словаре, но после первой итерации они обязательно будут присутствовать
    (обычно со значением `None`).
    """

    # История сообщений (может быть пустой)
    messages: List[BaseMessage]

    # Исходный запрос пользователя (может быть `None` если запрос пустой)
    user_query: NotRequired[str]

    # Решение роутера о типе запроса (recipe | search)
    router_decision: NotRequired[RouterDecision | None]

    # Результаты поиска из векторного хранилища
    search_results: NotRequired[List[dict] | None]

    # Данные рецепта (если запрос относится к кулинарии)
    recipe_data: NotRequired[dict | None]

    # Финальный вывод, который будет отправлен пользователю
    agent_output: NotRequired[dict | None]

    # Флаг, показывающий, что весь граф выполнен
    is_complete: bool

    # Текст ошибки, если что‑то пошло не так
    error: NotRequired[str | None]


# Публичный API модуля
__all__ = ["AgentState", "RouterDecision"]
