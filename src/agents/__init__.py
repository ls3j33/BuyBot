"""
Агенты BuyBot.
Импортируем агентов для регистрации в реестре, затем граф.
"""
# Явные импорты для регистрации агентов (должны быть ПЕРВЫМИ!)
import src.agents.router_agent  # noqa: F401
import src.agents.recipe_agent  # noqa: F401
import src.agents.search_agent  # noqa: F401

# Импортируем граф после регистрации агентов
from src.agents.graph import process_query_graph

# Публичный API
__all__ = ["process_query_graph"]
