"""Фикстуры для тестов."""
import pytest


@pytest.fixture
def sample_search_query():
    """Пример запроса поиска."""
    return {"query": "молоко"}


@pytest.fixture
def sample_recipe_query():
    """Пример запроса рецепта."""
    return {"query": "рецепт пиццы"}


@pytest.fixture
def sample_recommendation_query():
    """Пример запроса рекомендации."""
    return {"query": "какой сыр лучше для пиццы"}


@pytest.fixture
def sample_chat_request():
    """Базовый запрос к чату."""
    return {
        "query": "сыр",
        "conversation_id": "test-123"
    }
