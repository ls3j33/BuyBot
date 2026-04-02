"""
Реестр агентов для динамической регистрации.

Пример использования:
    from src.agents.registry import agent_registry

    @agent_registry.register("my_agent")
    def my_agent(state):  # тип state не аннотируем, чтобы избежать циклических импортов
        ...

    @agent_registry.register_router("router")
    def router(state) -> Literal["recipe_agent", "search_agent"]:
        # логика роутинга → возвращаем имя следующего узла
        ...
"""

from __future__ import annotations  # позволяет использовать строковые типы

import logging
from typing import Callable, Dict, Literal, Optional

# ------------------------------------------------------------
#  Регистратор
# ------------------------------------------------------------
class AgentRegistry:
    """
    Хранилище для всех агентов и роутеров.
    Позволяет динамически регистрировать функции‑агенты и
    функции‑условные роутеры, после чего получать их по имени.
    """

    def __init__(self) -> None:
        # pylint: disable=unused-private-member
        #   _agents/_routing_rules используются только через публичные методы
        self._agents: Dict[
            str, Callable[[ "AgentState" ], "AgentState"]
        ] = {}
        self._routing_rules: Dict[
            str, Callable[[ "AgentState" ], Literal["recipe_agent", "search_agent", "router"]]
        ] = {}

    # -----------------------------------------------------------------
    #  Регистрация обычных агентов
    # -----------------------------------------------------------------
    def register(self, name: str) -> Callable[[Callable], Callable]:
        """
        Декоратор для регистрации обычного агента.

        Пример:
            @agent_registry.register("search_agent")
            def search_agent(state):
                ...

        Возвращает саму функцию (без изменений), чтобы её можно было
        использовать как обычный колл‑абл.
        """

        def decorator(func: Callable) -> Callable:
            if name in self._agents:
                raise ValueError(f"Agent '{name}' уже зарегистрирован")
            self._agents[name] = func
            return func

        return decorator

    def get_agent(self, name: str) -> Callable:
        """Получить агент по имени (выбрасывает ValueError, если не найден)."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"Agent '{name}' not found in registry") from exc

    # -----------------------------------------------------------------
    #  Регистрация роутеров (conditional edges)
    # -----------------------------------------------------------------
    def register_router(self, name: str) -> Callable[[Callable], Callable]:
        """
        Декоратор для регистрации функции‑роутера.

        Функция‑роутер принимает `state` и **возвращает**:
            Literal["recipe_agent", "search_agent"]

        Пример:
            @agent_registry.register_router("router")
            def router(state) -> Literal["recipe_agent", "search_agent"]:
                ...

        Возвращает декоратор, аналогичный `register`.
        """
        def decorator(func: Callable) -> Callable:
            if name in self._routing_rules:
                raise ValueError(f"Router '{name}' уже зарегистрирован")
            self._routing_rules[name] = func
            logging.getLogger(__name__).debug("Router registered: %s", name)
            return func

        return decorator

    def get_router(
        self, name: str
    ) -> Optional[Callable[[ "AgentState" ], Literal["recipe_agent", "search_agent"]]]:
        """Получить функцию‑router по имени (может вернуть None)."""
        return self._routing_rules.get(name)

    # -----------------------------------------------------------------
    #  Утилиты
    # -----------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AgentRegistry agents={list(self._agents.keys())} "
            f"routers={list(self._routing_rules.keys())}>"
        )


# -----------------------------------------------------------------
#  Глобальный реестр (singleton)
# -----------------------------------------------------------------
agent_registry = AgentRegistry()

# Публичный API модуля
__all__ = ["agent_registry"]
