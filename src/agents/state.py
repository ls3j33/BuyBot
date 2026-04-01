from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator


class ModerationResult(TypedDict):
    """Результат модерации запроса"""
    is_relevant: bool
    reason: str
    category: Literal["shopping", "off_topic", "spam", "other"]


class RAGDecision(TypedDict):
    """Решение RAG агента о поиске"""
    should_search: bool
    search_query: str
    search_category: str | None
    n_results: int
    reason: str


class WriterOutput(TypedDict):
    """Вывод writer агента"""
    response_text: str
    products_mentioned: list[str]
    has_recommendations: bool
    confidence: Literal["high", "medium", "low"]


class AgentState(TypedDict):
    """
    Состояние графа LangGraph.
    
    Содержит всю информацию о текущем состоянии обработки запроса.
    """
    # История сообщений
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Исходный запрос пользователя
    user_query: str
    
    # Результаты модерации
    moderation_result: ModerationResult | None
    
    # Решение RAG агента
    rag_decision: RAGDecision | None
    
    # Результаты поиска из векторной БД
    search_results: list[dict] | None
    
    # Финальный ответ пользователю
    writer_output: WriterOutput | None
    
    # Флаг завершения обработки
    is_complete: bool
    
    # Ошибка если произошла
    error: str | None
