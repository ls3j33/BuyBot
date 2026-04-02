"""
Router Agent — определяет тип запроса и направляет к нужному агенту.
Использует LLM для классификации запросов.
"""

from __future__ import annotations

from typing import Literal, Tuple
from enum import Enum

from src.agents.registry import agent_registry
from src.agents.state import AgentState


# ----------------------------------------------------------------------
#  Типы
# ----------------------------------------------------------------------
class QueryType(str, Enum):
    """Типы запросов"""
    RECIPE = "recipe"
    SEARCH = "search"


def _llm_result_to_tuple(result: dict) -> Tuple[QueryType, float]:
    """
    Приводит результат LLM‑классификатора к кортежу
    (QueryType, confidence).
    """
    qt_str = result.get("query_type", "search")
    confidence_raw = result.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.5

    query_type = QueryType.RECIPE if qt_str == "recipe" else QueryType.SEARCH
    return query_type, confidence


def detect_query_type_by_llm(query: str) -> Tuple[QueryType, float]:
    """
    Классификация через LLM.
    Возвращает (тип, confidence). При любой ошибке – (SEARCH, 0.0).
    """
    try:
        # Импортируем внутри функции чтобы избежать сетевых запросов при импорте модуля
        from src.config.llm import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from src.utils.logger import get_app_logger
        
        # Создаем промпт внутри функции
        classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Ты — классификатор запросов для чат‑бота супермаркета.

Определи тип запроса:
- "recipe" — запрос о полном рецепте/приготовлении блюда (ингредиенты + шаги готовки)
- "search" — запрос о поиске товаров, цен, наличии, выборе конкретного продукта

**Важно:**
- "какой X лучше" → search (пользователь хочет выбрать товар)
- "что нужно для X" → recipe (пользователь хочет список ингредиентов)
- "как приготовить X" → recipe (пользователь хочет рецепт)

Верни JSON:
{{"query_type": "recipe" | "search", "confidence": 0.0‑1.0}}

Примеры:
- "рецепт борща" → {{"query_type":"recipe","confidence":0.95}}
- "как приготовить плов" → {{"query_type":"recipe","confidence":0.92}}
- "что нужно для лазаньи" → {{"query_type":"recipe","confidence":0.94}}
- "молоко" → {{"query_type":"search","confidence":0.97}}
- "самый дешевый сыр" → {{"query_type":"search","confidence":0.96}}
- "какой сыр лучше для пиццы" → {{"query_type":"search","confidence":0.98}}
- "какое вино выбрать к мясу" → {{"query_type":"search","confidence":0.97}}
- "где купить молоко" → {{"query_type":"search","confidence":0.98}}
""",
                ),
                ("human", "Запрос: {query}"),
            ]
        )
        
        chain = classifier_prompt | get_llm() | JsonOutputParser()
        result = chain.invoke({"query": query})
        query_type, confidence = _llm_result_to_tuple(result)

        get_app_logger().debug(
            "[Router‑LLM] %s → %s (conf=%.2f)",
            query,
            query_type.value,
            confidence,
        )
        return query_type, confidence

    except Exception as exc:  # pragma: no cover
        # Ошибки могут быть сетевыми, JSON‑парсингом и т.п.
        get_app_logger().exception("[Router‑LLM] Ошибка классификации: %s", exc)
        return QueryType.SEARCH, 0.0


# ----------------------------------------------------------------------
#  Основной агент‑router
# ----------------------------------------------------------------------
@agent_registry.register("router")
def router_agent(state: AgentState) -> AgentState:
    """
    Определяет тип пользовательского запроса через LLM.
    Возвращает в состоянии поле ``router_decision``:
    {
        "query_type": "recipe" | "search",
        "confidence": float,
        "reason": str
    }
    """
    from src.utils.logger import get_app_logger

    query = state["user_query"]
    get_app_logger().info("[Router] Обрабатываю запрос: %s", query)

    # LLM‑классификатор
    query_type, confidence = detect_query_type_by_llm(query)

    decision = {
        "query_type": query_type.value,
        "confidence": confidence,
        "reason": f"Определён тип запроса: {query_type.value}",
    }

    get_app_logger().info(
        "[Router] Итог: %s (conf=%.2f)",
        query_type.value,
        confidence,
    )

    return {
        **state,
        "router_decision": decision,
    }


# ----------------------------------------------------------------------
#  Функция роутинга в графе (используется в orchestrator)
# ----------------------------------------------------------------------
def route_to_agent(state: AgentState) -> Literal["recipe_agent", "search_agent"]:
    """
    Возвращает имя агента, к которому следует передать *state*.
    """
    decision = state.get("router_decision", {})
    qtype = decision.get("query_type", "search")
    return "recipe_agent" if qtype == "recipe" else "search_agent"


# ----------------------------------------------------------------------
#  Регистрация роутера в центральном реестре
# ----------------------------------------------------------------------
agent_registry.register_router("router")(route_to_agent)
