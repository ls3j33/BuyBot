from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Запрос к чат-боту"""
    
    query: str = Field(
        ...,
        description="Текст запроса пользователя",
        examples=["Какой сыр лучше для пиццы?", "Что нужно для борща?"]
    )
    
    conversation_id: Optional[str] = Field(
        None,
        description="ID сессии для хранения истории (опционально)"
    )


class ProductInfo(BaseModel):
    """Информация о товаре в ответе"""
    
    id: str = Field(..., description="ID товара")
    name: str = Field(..., description="Название товара")
    category: str = Field(..., description="Категория товара")
    price: str = Field(..., description="Цена товара")
    distance: Optional[float] = Field(None, description="Расстояние релевантности")


class ChatResponse(BaseModel):
    """Ответ чат-бота"""
    
    response: str = Field(
        ...,
        description="Текст ответа пользователю"
    )
    
    search_results: list[ProductInfo] = Field(
        default_factory=list,
        description="Найденные товары"
    )
    
    confidence: str = Field(
        default="medium",
        description="Уровень уверенности ответа",
        examples=["high", "medium", "low"]
    )
    
    success: bool = Field(
        default=True,
        description="Флаг успешной обработки"
    )
    
    error: Optional[str] = Field(
        None,
        description="Текст ошибки если произошла"
    )


class StatsResponse(BaseModel):
    """Ответ эндпоинта статистики"""
    
    collection_name: str
    total_documents: int
    is_ready: bool


class HealthResponse(BaseModel):
    """Ответ эндпоинта здоровья"""
    
    status: str = "ok"
    service: str = "buybot"
