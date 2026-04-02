"""
Search Agent — обработка обычных запросов поиска товаров.
Использует LLM для извлечения параметров + keywords как fallback.
"""

import re
from typing import List, Dict, Tuple, Optional

from src.agents.state import AgentState
from src.tools.vector_search import get_vector_store
from src.agents.registry import agent_registry
from src.models.schemas import SearchIntent
from src.utils.logger import get_app_logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config.llm import get_llm


# ----------------------------------------------------------------------
#  Helper utilities (общие функции для всех fallback‑парсеров)
# ----------------------------------------------------------------------
def _extract_number_with_unit(
    text: str,
    unit_patterns: List[str],
    multiplier: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """
    Ищет число + любой из указанных *unit_patterns* (рег. выражения).
    Возвращает диапазон (value‑tolerance, value+tolerance) или None.
    """
    # билдим одно большое регулярное выражение, например:
    #   (\d+(?:[.,]\d*)?)\s*(литр|л|мл|кг|г|грамм)
    pattern = rf'(\d+(?:[.,]\d*)?)\s*(?:{"|".join(unit_patterns)})\b'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    # нормализуем десятичный разделитель (запятая → точка)
    raw_value = match.group(1).replace(',', '.')
    value = float(raw_value) * multiplier
    tolerance = value * 0.15  # 15 % погрешность (чуть уже, чем 20 %)
    low, high = max(0.0, value - tolerance), value + tolerance
    return low, high


# ----------------------------------------------------------------------
#  Fallback‑детекторы (по ключевым словам)
# ----------------------------------------------------------------------
def detect_sort_from_query_fallback(query: str) -> Optional[str]:
    """Определить сортировку по ключевым словам (fallback)."""
    q = query.lower()

    # ищем отдельные слова, а не подстроки внутри других слов
    cheap_words = {"дешев", "недорог", "дешевле", "эконом"}
    expensive_words = {"дорог", "дороже", "премиум"}

    if any(re.search(rf"\b{w}\b", q) for w in cheap_words):
        return "price_asc"
    if any(re.search(rf"\b{w}\b", q) for w in expensive_words):
        return "price_desc"
    return None


def detect_volume_from_query_fallback(query: str) -> Optional[Tuple[float, float]]:
    """Определить диапазон объёма в мл по запросу."""
    q = query.lower()

    # литры → мл
    if "литр" in q or re.search(r"\bл\b", q):
        res = _extract_number_with_unit(q, ["литр", "л"], multiplier=1000)
        if res:
            return res
        # "литр" без числа → считаем 1 л (1000 мл) ±15 %
        return _extract_number_with_unit("1 л", ["литр", "л"], multiplier=1000)

    # миллилитры
    if "мл" in q:
        return _extract_number_with_unit(q, ["мл"])

    return None


def detect_weight_from_query_fallback(query: str) -> Optional[Tuple[float, float]]:
    """Определить диапазон веса в граммах по запросу."""
    q = query.lower()

    # килограммы → граммы
    if "кг" in q or "килограмм" in q:
        return _extract_number_with_unit(q, ["кг", "килограмм"], multiplier=1000)

    # граммы
    if re.search(r"\bг\b", q) or "грамм" in q:
        return _extract_number_with_unit(q, ["г", "грамм"])

    return None


def detect_price_from_query_fallback(query: str) -> Optional[float]:
    """
    Очень простой детектор цены (используется только в fallback‑потоке).
    Ищет шаблоны типа «до 100 рублей», «≤200₽» и т.п.
    """
    q = query.lower()
    # ищем число перед словом «рубль», «₽», «руб.» и т.д.
    match = re.search(r'(\d+(?:[.,]\d*)?)\s*(?:рублей?|р|₽|руб\.)', q)
    if match:
        return float(match.group(1).replace(',', '.'))
    return None


# ----------------------------------------------------------------------
#  Prompt & LLM extraction
# ----------------------------------------------------------------------
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Ты — ассистент для извлечения параметров поиска товаров.

Извлеки параметры из запроса:
- items_to_search: список конкретных товаров/категорий для поиска
- sort: "price_asc", "price_desc" или null
- max_price: число или null
- volume_range: [min_ml, max_ml] или null
- weight_range: [min_g, max_g] или null

**Важно:** 
- Если запрос о конкретном продукте (сыр, молоко, хлеб), указывай именно категорию продукта
- Если запрос о блюде (пицца, борщ, салат), определи какие ингредиенты/товары подходят
- Избегай общих слов — указывай конкретные категории

Примеры:
- "молоко" → {{"items_to_search": ["молоко"], ...}}
- "сыр" → {{"items_to_search": ["сыр"], ...}}
- "самое дешевое молоко" → {{"items_to_search": ["молоко"], "sort":"price_asc", ...}}
- "литр молока" → {{"items_to_search": ["молоко"], "volume_range": [900,1100]}}
- "молоко до 100 рублей" → {{"items_to_search": ["молоко"], "max_price":100}}
- "сыр 500г" → {{"items_to_search": ["сыр"], "weight_range": [425,575]}}
- "какой сыр лучше для пиццы?" → {{"items_to_search": ["моцарелла", "пармезан", "чеддер", "сыр"], ...}}
- "что нужно для борща?" → {{"items_to_search": ["свекла", "капуста", "морковь", "лук", "говядина"], ...}}
- "какие продукты для салата цезарь?" → {{"items_to_search": ["салат ромэн", "курица", "пармезан", "сухарики"], ...}}

Верни JSON:
{{
    "items_to_search": ["товар1"],
    "sort": "price_asc" | "price_desc" | null,
    "max_price": 123 | null,
    "volume_range": [900,1100] | null,
    "weight_range": [400,600] | null
}}
""",
        ),
        ("human", "Запрос: {query}"),
    ]
)


def extract_search_intent(query: str) -> SearchIntent:
    """Извлечь намерения поиска через LLM."""
    from src.utils.logger import get_app_logger
    
    try:
        chain = EXTRACTION_PROMPT | get_llm() | JsonOutputParser()
        result = chain.invoke({"query": query})

        # Приводим диапазоны к Tuple[float, float] если они есть
        def _as_range(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return float(val[0]), float(val[1])
            return None

        volume_range = _as_range(result.get("volume_range"))
        weight_range = _as_range(result.get("weight_range"))

        intent = SearchIntent(
            items_to_search=result.get("items_to_search", []),
            sort=result.get("sort"),
            max_price=result.get("max_price"),
            volume_range=volume_range,
            weight_range=weight_range,
            confidence="high",
        )

        get_app_logger().debug("[Search-LLM] %s → %s", query, intent)
        return intent

    except Exception as exc:  # pragma: no cover – логируем полностью
        get_app_logger().exception("[Search-LLM] Ошибка извлечения: %s", exc)
        # Возврат «пустого» интента, но с пометкой low‑confidence
        return SearchIntent(items_to_search=[], confidence="low")


def get_fallback_intent(query: str) -> SearchIntent:
    """Создаёт intent на основе ключевых слов (fallback)."""
    return SearchIntent(
        items_to_search=[query],
        sort=detect_sort_from_query_fallback(query),
        max_price=detect_price_from_query_fallback(query),
        volume_range=detect_volume_from_query_fallback(query),
        weight_range=detect_weight_from_query_fallback(query),
        confidence="low",
    )


# ----------------------------------------------------------------------
#  Парсинг названий (для post‑filter)
# ----------------------------------------------------------------------
def parse_volume_from_name(name: str) -> Optional[float]:
    """Извлечь объём из названия товара в миллилитрах."""
    n = name.lower()

    # литры → мл
    match = re.search(r'(\d+(?:[.,]\d*)?)\s*л\b', n)
    if match:
        return float(match.group(1).replace(',', '.')) * 1000

    # миллилитры
    match = re.search(r'(\d+(?:[.,]\d*)?)\s*мл\b', n)
    if match:
        return float(match.group(1).replace(',', '.'))

    return None  # вес в названиях игнорируем


def parse_weight_from_name(name: str) -> Optional[float]:
    """Извлечь вес из названия товара в граммах."""
    n = name.lower()
    match = re.search(r'(\d+(?:[.,]\d*)?)\s*(г|грамм|кг|килограмм)\b', n)
    if match:
        value = float(match.group(1).replace(',', '.'))
        unit = match.group(2)
        if unit in ("кг", "килограмм"):
            return value * 1000
        return value
    return None


# ----------------------------------------------------------------------
#  Фильтры по диапазонам
# ----------------------------------------------------------------------
def filter_by_relevance(results: List[Dict], items_to_search: List[str]) -> List[Dict]:
    """
    Отфильтровать товары по релевантности названия.
    Оставляем только товары где название содержит искомое слово или категорию.
    """
    if not items_to_search:
        return results
    
    # Создаем набор ключевых слов для поиска
    keywords = set()
    for item in items_to_search:
        keywords.add(item.lower())
    
    filtered = []
    for r in results:
        name = r.get("metadata", {}).get("name", "").lower()
        category = r.get("metadata", {}).get("category", "").lower()
        
        # Проверяем совпадение с категорией (точное совпадение)
        for kw in keywords:
            # Точное совпадение категории
            if category == kw or category.startswith(kw + " ") or category.endswith(" " + kw):
                filtered.append(r)
                break
            # Название начинается с ключевого слова
            # "Сыр ..." ✓, "Сырок ..." ✗
            if name.startswith(kw + " ") or name == kw:
                filtered.append(r)
                break
            # Название содержит ключевое слово как отдельное слово (через пробел)
            elif f" {kw} " in f" {name} ":
                filtered.append(r)
                break
    
    return filtered


def filter_by_volume(
    results: List[Dict], volume_range: Tuple[float, float]
) -> List[Dict]:
    """Отфильтровать товары по объёму (мл)."""
    min_vol, max_vol = volume_range
    filtered = [
        r
        for r in results
        if (vol := parse_volume_from_name(r.get("metadata", {}).get("name", "")))
        and min_vol <= vol <= max_vol
    ]
    return filtered  # [] –> дальше будет обработано как “нет результатов”


def filter_by_weight(
    results: List[Dict], weight_range: Tuple[float, float]
) -> List[Dict]:
    """Отфильтровать товары по весу (г)."""
    min_w, max_w = weight_range
    filtered = [
        r
        for r in results
        if (wt := parse_weight_from_name(r.get("metadata", {}).get("name", "")))
        and min_w <= wt <= max_w
    ]
    return filtered


def filter_by_price(
    results: List[Dict], max_price: float
) -> List[Dict]:
    """Отфильтровать товары по цене (руб)."""
    filtered = [
        r
        for r in results
        if (price := r.get("metadata", {}).get("price"))
        and isinstance(price, (int, float))
        and price <= max_price
    ]
    return filtered


# ----------------------------------------------------------------------
#  Формирование ответа пользователю
# ----------------------------------------------------------------------
def format_search_response(
    query: str,
    results: List[Dict],
    sort_order: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Сформировать человеко‑читаемый ответ."""
    if not results:
        return (
            "К сожалению, по вашему запросу ничего не найдено. Попробуйте уточнить запрос.",
            [],
        )

    # группируем по категории
    by_category: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r.get("metadata", {}).get("category", "Другое")
        by_category.setdefault(cat, []).append(r)

    # небольшая инфа о объёме, если запрос содержит объём‑слово
    volume_info = ""
    vol_range = detect_volume_from_query_fallback(query)
    if vol_range:
        low, high = vol_range
        if low >= 1000:
            volume_info = f" (объём ≈ {int(low/1000)} л)"
        else:
            volume_info = f" (объём {int(low)}‑{int(high)} мл)"

    # заголовок
    header = f'По запросу "{query}"{volume_info}'
    if sort_order == "price_asc":
        header += " (сортировка: от дешевых к дорогим)"
    elif sort_order == "price_desc":
        header += " (сортировка: от дорогих к дешевым)"
    header += f' найдено {len(results)} товаров:'
    lines = [header]

    # список товаров (по 5 в категории)
    for cat, items in by_category.items():
        lines.append(f"\n**{cat}:**")
        for item in items[:5]:
            md = item.get("metadata", {})
            name = md.get("name", "N/A")
            price = md.get("price", "N/A")
            lines.append(f"- {name} — {price} руб.")
    # если сильно много, покажем «ещё…»
    if len(results) > 10:
        lines.append(f"\n... и ещё {len(results) - 10} товаров.")

    # подсказки «самый дорогой/дешевый» – берём уже отсортированный список
    if sort_order == "price_asc" and results:
        cheapest = results[0]
        lines.append(
            f"\n💰 Самый дешёвый: {cheapest['metadata'].get('name')} — {cheapest['metadata'].get('price')} руб."
        )
    elif sort_order == "price_desc" and results:
        expensive = results[0]
        lines.append(
            f"\n💎 Самый дорогой: {expensive['metadata'].get('name')} — {expensive['metadata'].get('price')} руб."
        )

    # собираем список названий (для дальнейшей обработки)
    mentioned = [
        r.get("metadata", {}).get("name", "")
        for r in results[:10]
        if r.get("metadata", {}).get("name")
    ]

    return "\n".join(lines), mentioned


# ----------------------------------------------------------------------
#  Основной агент
# ----------------------------------------------------------------------
@agent_registry.register("search_agent")
def search_agent(state: AgentState) -> AgentState:
    """Обработка обычного поискового запроса."""
    query = state["user_query"]
    get_app_logger().info("[Search] Обрабатываю запрос: %s", query)

    # ------------------------------------------------------------------
    #  1️⃣ Получаем параметры (LLM → fallback)
    # ------------------------------------------------------------------
    intent = extract_search_intent(query)

    # Если LLM не извлек товары, переходим сразу к fallback‑интенту
    if not intent.items_to_search:
        get_app_logger().debug("[Search] LLM не смог выделить товары – использую fallback")
        intent = get_fallback_intent(query)

    # Заполняем недостающие параметры fallback‑детекторами
    if not intent.sort:
        intent.sort = detect_sort_from_query_fallback(query)
    if not intent.volume_range:
        intent.volume_range = detect_volume_from_query_fallback(query)
    if not intent.weight_range:
        intent.weight_range = detect_weight_from_query_fallback(query)
    if not intent.max_price:
        intent.max_price = detect_price_from_query_fallback(query)

    get_app_logger().debug(
        "[Search] Intent → items=%s, sort=%s, volume=%s, weight=%s, price=%s",
        intent.items_to_search,
        intent.sort,
        intent.volume_range,
        intent.weight_range,
        intent.max_price,
    )

    # ------------------------------------------------------------------
    #  2️⃣ Поиск в векторном хранилище
    # ------------------------------------------------------------------
    store = get_vector_store()
    all_results: List[Dict] = []
    for item in intent.items_to_search:
        # ищем 20‑й релевантных записей на каждый товар‑ключ
        all_results.extend(store.search(query=item, n_results=20))

    get_app_logger().debug("[Search] Найдено %d записей до фильтрации", len(all_results))

    # ------------------------------------------------------------------
    #  3️⃣ Фильтрация по релевантности названия
    # ------------------------------------------------------------------
    all_results = filter_by_relevance(all_results, intent.items_to_search)
    get_app_logger().debug(
        "[Search] После фильтра релевантности осталось %d записей", len(all_results)
    )

    # ------------------------------------------------------------------
    #  4️⃣ Фильтрация по диапазонам и цене
    # ------------------------------------------------------------------
    if intent.volume_range:
        all_results = filter_by_volume(all_results, intent.volume_range)
        get_app_logger().debug(
            "[Search] После фильтра объёма осталось %d записей", len(all_results)
        )
    if intent.weight_range:
        all_results = filter_by_weight(all_results, intent.weight_range)
        get_app_logger().debug(
            "[Search] После фильтра веса осталось %d записей", len(all_results)
        )
    if intent.max_price:
        all_results = filter_by_price(all_results, intent.max_price)
        get_app_logger().debug(
            "[Search] После фильтра цены осталось %d записей", len(all_results)
        )

    # ------------------------------------------------------------------
    #  4️⃣ Оставляем только релевантные (по расстоянию)
    # ------------------------------------------------------------------
    relevant = [r for r in all_results if r.get("distance", 1.0) < 0.95]

    # ------------------------------------------------------------------
    #  5️⃣ Сортировка по цене (если требуется)
    # ------------------------------------------------------------------
    if intent.sort == "price_asc":
        relevant.sort(
            key=lambda x: float(x.get("metadata", {}).get("price", float("inf")))
        )
    elif intent.sort == "price_desc":
        relevant.sort(
            key=lambda x: float(x.get("metadata", {}).get("price", 0.0)), reverse=True
        )

    # ------------------------------------------------------------------
    #  6️⃣ Формируем пользовательский ответ
    # ------------------------------------------------------------------
    response_text, products_mentioned = format_search_response(
        query, relevant[:10], intent.sort
    )
    get_app_logger().info("[Search] Ответ готов, упомянуто %d товаров", len(products_mentioned))

    # ------------------------------------------------------------------
    #  7️⃣ Возврат нового состояния
    # ------------------------------------------------------------------
    return {
        **state,
        "search_results": relevant[:10],
        "agent_output": {
            "response_text": response_text,
            "products_mentioned": products_mentioned,
            "has_recommendations": bool(relevant),
            "confidence": "high" if relevant else "low",
        },
        "is_complete": True,
    }
