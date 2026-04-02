"""
Pydantic модели для данных агентов.
"""
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class QueryType(str, Enum):
    """Типы запросов"""
    RECIPE = "recipe"
    SEARCH = "search"
    UNKNOWN = "unknown"


class SearchIntent(BaseModel):
    """Намерения для поиска товаров"""
    items_to_search: list[str] = Field(default_factory=list, description="Список товаров для поиска")
    sort: Literal["price_asc", "price_desc", None] = Field(default=None, description="Сортировка")
    max_price: float | None = Field(default=None, description="Максимальная цена")
    volume_range: tuple[float, float] | None = Field(default=None, description="Диапазон объема (мл)")
    weight_range: tuple[float, float] | None = Field(default=None, description="Диапазон веса (г)")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
