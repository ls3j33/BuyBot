"""
LangGraph‑граф для обработки пользовательских запросов.
Граф собирается из зарегистрированных в `agent_registry` агентов,
поддерживает lazy‑initialisation и защищён от гонок.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Literal

# ----------------------------------------------------------------------
#  Импорт из LangGraph (поддержка разных версий)
# ----------------------------------------------------------------------
try:  # >=0.0.74
    from langgraph.graph import START, END, StateGraph
except ImportError:  # pragma: no cover – старые версии
    from langgraph import START, END, StateGraph

# ----------------------------------------------------------------------
#  Импорт наших компонентов
# ----------------------------------------------------------------------
from src.agents.state import AgentState
from src.utils.logger import get_app_logger
from langchain_core.runnables import Runnable  # тип, возвращаемый compile()

# registry импортируется внутри get_agent_graph() ПОСЛЕ регистрации агентов

# ----------------------------------------------------------------------
#  Глобальные переменные (thread‑safe)
# ----------------------------------------------------------------------
_agent_graph: Runnable | None = None
_graph_lock = threading.Lock()
app_logger = get_app_logger()

# ----------------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------------
__all__ = ["process_query_graph"]


def get_agent_graph() -> Runnable:
    """
    Возвратить уже построенный граф или создать его один раз
    (lazy‑initialisation, thread‑safe).
    """
    global _agent_graph
    if _agent_graph is None:
        with _graph_lock:
            if _agent_graph is None:  # double‑checked lock
                # Импортируем registry ПОСЛЕ регистрации агентов
                from src.agents.registry import agent_registry
                app_logger.info("[Graph] Создаю граф агентов…")
                _agent_graph = _create_and_compile_graph(agent_registry)
                app_logger.info("[Graph] Граф успешно скомпилирован")
    return _agent_graph


def _create_and_compile_graph(agent_registry) -> Runnable:
    """
    Сборка графа:
        START → router → (recipe_agent | search_agent) → END
    
    Args:
        agent_registry: Реестр агентов для получения узлов и роутинга
    """
    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------
    #   Регистрация узлов
    # ------------------------------------------------------------------
    for node_name in ("router", "recipe_agent", "search_agent"):
        agent_func = agent_registry.get_agent(node_name)
        if not agent_func:
            raise ValueError(f"[Graph] Агент «{node_name}» не найден в реестре")
        workflow.add_node(node_name, agent_func)
        app_logger.debug("[Graph] Добавлен узел: %s", node_name)

    # ------------------------------------------------------------------
    #   Условные переходы от роутера
    # ------------------------------------------------------------------
    workflow.add_edge(START, "router")

    router_func = agent_registry.get_router("router")
    if router_func is None:
        raise RuntimeError("[Graph] Функция‑router не зарегистрирована")

    # router‑func должна возвращать Literal["recipe_agent", "search_agent"]
    workflow.add_conditional_edges(
        "router",
        router_func,
        {
            "recipe_agent": "recipe_agent",
            "search_agent": "search_agent",
        },
    )

    # ------------------------------------------------------------------
    #   Завершение
    # ------------------------------------------------------------------
    workflow.add_edge("recipe_agent", END)
    workflow.add_edge("search_agent", END)

    return workflow.compile()


def process_query_graph(query: str) -> Dict[str, Any]:
    """
    Обработать пользовательский запрос через построенный граф.

    Parameters
    ----------
    query : str
        Текст запроса пользователя.

    Returns
    -------
    dict
        {
            "response": str,
            "search_results": list[dict],
            "confidence": "high" | "medium" | "low",
            "success": bool,
            "error": str | None,
        }
    """
    app = get_agent_graph()
    app_logger.info("[Graph] Запуск обработки запроса: %s", query)

    # --------------------------------------------------------------
    #   Начальное состояние (сообщения‑история)
    # --------------------------------------------------------------
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
        "router_decision": None,
        "search_results": None,
        "recipe_data": None,
        "agent_output": None,
        "is_complete": False,
        "error": None,
    }

    try:
        # LangGraph может вернуть Snapshot/State; приводим к dict
        raw_result = app.invoke(initial_state)
        result = dict(raw_result) if not isinstance(raw_result, dict) else raw_result
        app_logger.info("[Graph] Результат работы графа: %s", result)
        app_logger.info("[Graph] agent_output = %s", result.get("agent_output"))
        app_logger.info("[Graph] is_complete = %s", result.get("is_complete"))

        agent_out = result.get("agent_output", {})

        return {
            "response": agent_out.get("response_text", ""),
            "search_results": result.get("search_results", []),
            "confidence": agent_out.get("confidence", "low"),
            "success": bool(result.get("is_complete")),
            "error": result.get("error"),
        }

    except Exception:  # pragma: no cover
        # Выводим полный стек в лог, но в ответе передаём только сообщение
        app_logger.exception("[Graph] Ошибка при обработке запроса")
        return {
            "response": "Произошла ошибка при обработке запроса.",
            "search_results": [],
            "confidence": "low",
            "success": False,
            "error": "Internal server error",
        }
