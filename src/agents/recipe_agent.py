"""
Recipe Agent — обработка запросов рецептов.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.agents.registry import agent_registry
from src.agents.state import AgentState
from src.config.llm import get_llm
from src.tools.vector_search import get_vector_store
from src.utils.logger import get_app_logger
# from src.utils.logger import app_logger  # REMOVED
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------------
#  Константы
# ----------------------------------------------------------------------
COSINE_THRESHOLD = 0.60          # порог расстояния векторного поиска
WORD_OVERLAP_EXACT = 0.80        # ≥ 80 % совпадений → «точное» совпадение
WORD_OVERLAP_PARTIAL = 0.50      # ≥ 50 % → «частичное»

# ----------------------------------------------------------------------
#  Вспомогательные функции
# ----------------------------------------------------------------------
def extract_base_words(text: str) -> Set[str]:
    """
    Простейшее выделение «корней» слов.
    Удаляем пунктуацию, разбиваем на слова и обрезаем
    типичные русские суффиксы, оставляя минимум 3 символа.
    """
    # Убираем пунктуацию и цифры
    clean = re.sub(r"[^\w\s-]", " ", text.lower())
    words = clean.split()
    stems: Set[str] = set()
    for w in words:
        # Удаляем типичные суффиксы (очень упрощённый стемминг)
        w = re.sub(r"(ов|ий|ый|ая|ей|ие|их|ую|ой|ым|ем|ой|я)$", "", w)
        if len(w) > 2:
            stems.add(w)
    return stems


def has_word_overlap(ingredient: str, product_name: str) -> Tuple[bool, bool]:
    """
    Проверить перекрытие слов между ингредиентом и товаром.

    Returns
    -------
    (is_exact, is_partial)
        is_exact   –  overlap_ratio >= WORD_OVERLAP_EXACT
        is_partial –  WORD_OVERLAP_PARTIAL ≤ overlap_ratio < WORD_OVERLAP_EXACT
    """
    ing_words = extract_base_words(ingredient)
    prod_words = extract_base_words(product_name)

    if not ing_words or not prod_words:
        return False, False

    overlap = ing_words & prod_words
    overlap_ratio = len(overlap) / len(ing_words)

    if overlap_ratio >= WORD_OVERLAP_EXACT:
        return True, False
    if overlap_ratio >= WORD_OVERLAP_PARTIAL:
        return False, True
    return False, False


def clean_ingredient_name(ingredient: str) -> str:
    """
    Очистить название ингредиента от количества, размеров и справочных слов.
    """
    # Убираем количество и единицы измерения
    cleaned = re.sub(
        r"\s*—?\s*\d+[–\-]?\d*\s*(г|кг|шт|мл|л|ст\.?\s*л\.?|ч\.?\s*л\.?|для подачи|по вкусу).*",
        "",
        ingredient,
    )
    cleaned = re.sub(r"\s+среднего размера", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(.*\)", "", cleaned)  # убираем скобки
    return cleaned.strip()


def match_ingredient_to_product(
    ingredient: str,
    product_name: str,
    distance: float,
    threshold: float = COSINE_THRESHOLD,
) -> Tuple[bool, bool]:
    """
    Сопоставление ингредиента с товаром.

    - Сначала проверяем векторное расстояние (`distance`).
    - Затем пытаемся сопоставить по строке (начало названия, первое слово).
    - Если выше не сработало – используем слово‑overlap.
    """
    if distance >= threshold:
        return False, False

    name_lower = product_name.lower()
    ing_clean = clean_ingredient_name(ingredient).lower().strip()

    # 1️⃣ Начало названия (самый надёжный критерий)
    if name_lower.startswith(ing_clean):
        return True, False

    # 2️⃣ Первое слово ингредиента встречается в названии товара
    ing_first = ing_clean.split()[0] if ing_clean.split() else ""
    prod_first = name_lower.split()[0] if name_lower.split() else ""

    if ing_first and prod_first and len(ing_first) > 3 and ing_first in name_lower:
        return True, False

    # 3️⃣ Словесный overlap
    return has_word_overlap(ingredient, product_name)


def load_recipe_prompt() -> str:
    """Считать файл‑промпт для генерации рецептов."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "recipe_writer.txt"
    with prompt_path.open(encoding="utf-8") as f:
        return f.read()


def generate_recipe_with_llm(query: str) -> Dict[str, Any]:
    """Сгенерировать рецепт через LLM, вернуть структурированный dict."""
    get_app_logger().info("[Recipe] Генерация рецепта для запроса: %s", query)
    prompt_text = load_recipe_prompt()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "Запрос: {query}\n\nВерни JSON с рецептом."),
        ]
    )
    chain = prompt | get_llm() | JsonOutputParser()

    try:
        result = chain.invoke({"query": query})
        if isinstance(result, dict) and result.get("ingredients"):
            get_app_logger().info(
                "[Recipe] Сгенерировано %d ингредиентов", len(result["ingredients"])
            )
            return {
                "recipe_name": result.get("recipe_name", "Рецепт"),
                "intro": result.get("intro", ""),
                "ingredients": result.get("ingredients", []),
                "steps": result.get("steps", []),
                "tips": result.get("tips", []),
            }
    except Exception:  # pragma: no cover
        get_app_logger().exception("[Recipe] Ошибка генерации рецепта")
    # fallback‑ответ, если LLM не смог выдать структуру
    return {"recipe_name": "Рецепт", "ingredients": [], "steps": [], "tips": []}


def find_ingredients_in_results(
    ingredients: List[str],
    search_results: List[Dict],
    threshold: float = COSINE_THRESHOLD,
) -> Dict[str, List[Dict]]:
    """
    Поиск товаров, соответствующих каждому ингредиенту.

    Возврат: { "ingredient": [список найденных товаров] }
    """
    found: Dict[str, List[Dict]] = {}

    for ingredient in ingredients:
        cleaned = clean_ingredient_name(ingredient).lower().strip()
        found[ingredient] = []
        exact: List[Dict] = []
        partial: List[Dict] = []

        for result in search_results:
            meta = result.get("metadata", {})
            name = meta.get("name", "").lower()
            distance = float(result.get("distance", 1.0))

            is_exact, is_partial = match_ingredient_to_product(
                ingredient, name, distance, threshold
            )
            entry = {"id": result.get("id", ""), "metadata": meta, "distance": distance}
            if is_exact:
                exact.append(entry)
            elif is_partial:
                partial.append(entry)

        # Сортируем по возрастанию расстояния (чем меньше – тем ближе)
        exact.sort(key=lambda x: x["distance"])
        partial.sort(key=lambda x: x["distance"])
        found[ingredient] = exact + partial
    return found


def format_recipe_availability(
    found_ingredients: Dict[str, List[Dict]],
) -> Tuple[str, int, int]:
    """
    Сформировать текстовое представление наличия ингредиентов.

    Returns
    -------
    (text, count_available, count_missing)
    """
    lines: List[str] = []
    available = 0
    missing = 0

    for ing, products in found_ingredients.items():
        if products:
            available += 1
            best = products[0]
            meta = best.get("metadata", {})
            name = meta.get("name", "N/A")
            price = meta.get("price", "N/A")
            lines.append(f"[ЕСТЬ] {ing}: {name} — {price} руб.")
        else:
            missing += 1
            lines.append(f"[НЕТ] {ing}: нет в наличии")

    return "\n".join(lines), available, missing


# ----------------------------------------------------------------------
#  Основной агент
# ----------------------------------------------------------------------
@agent_registry.register("recipe_agent")
def recipe_agent(state: AgentState) -> AgentState:
    """
    Обрабатывает запрос пользователя, генерирует рецепт через LLM,
    ищет все требуемые ингредиенты в векторном хранилище и формирует
    человеко‑читаемый ответ.
    """
    from src.utils.logger import get_app_logger
    
    query = state["user_query"]
    get_app_logger().info("[Recipe] Начато обработка запроса: %s", query)

    # 1️⃣ Генерация рецепта
    recipe = generate_recipe_with_llm(query)

    if not recipe.get("ingredients"):
        get_app_logger().warning("[Recipe] Не удалось сгенерировать рецепт для: %s", query)
        return {
            **state,
            "agent_output": {
                "response_text": (
                    f'К сожалению, у меня пока нет рецепта для блюда "{query}". '
                    "Попробуйте спросить про борщ, плов или другое популярное блюдо!"
                ),
                "products_mentioned": [],
                "has_recommendations": False,
                "confidence": "medium",
            },
            "is_complete": True,
        }

    # 2️⃣ Поиск всех ингредиентов в векторном хранилище
    store = get_vector_store()
    all_search_results: List[Dict] = []

    for ingredient in recipe["ingredients"]:
        search_query = clean_ingredient_name(ingredient) or ingredient
        try:
            results = store.search(query=search_query, n_results=20)
        except Exception:  # pragma: no cover
            get_app_logger().exception("[Recipe] Ошибка поиска для ингредиента %s", ingredient)
            results = []
        all_search_results.extend(results or [])

    get_app_logger().debug("[Recipe] Найдено %d результатов по всем ингредиентам", len(all_search_results))

    # 3️⃣ Сопоставление ингредиентов с найденными товарами
    found_ingredients = find_ingredients_in_results(
        recipe["ingredients"], all_search_results, threshold=COSINE_THRESHOLD
    )

    # 4️⃣ Формирование human‑readable ответа
    lines: List[str] = [f"# {recipe['recipe_name']}\n"]
    if recipe.get("intro"):
        lines.append(f"{recipe['intro']}\n")

    lines.append("## Ингредиенты и наличие в магазине\n")
    availability_text, available_cnt, missing_cnt = format_recipe_availability(found_ingredients)
    lines.append(availability_text)
    lines.append("")
    lines.append(f"**Найдено {available_cnt} из {len(recipe['ingredients'])} ингредиентов**\n")

    if recipe.get("steps"):
        lines.append("## Приготовление\n")
        for idx, step in enumerate(recipe["steps"], 1):
            lines.append(f"{idx}. {step}")
        lines.append("")

    if recipe.get("tips"):
        lines.append("## Советы\n")
        for tip in recipe["tips"]:
            lines.append(f"- {tip}")
        lines.append("")

    response_text = "\n".join(lines)

    # 5️⃣ Список всех упомянутых продуктов (для UI‑подсказок)
    products_mentioned: List[str] = [
        p.get("metadata", {}).get("name", "")
        for prod_list in found_ingredients.values()
        for p in prod_list
    ]

    # 6️⃣ Список товаров, которые будем передавать дальше (по одному лучшему варианту на ингредиент)
    found_products: List[Dict] = [
        prod_list[0] for prod_list in found_ingredients.values() if prod_list
    ]

    get_app_logger().info(
        "[Recipe] Ответ готов: %d/%d ингредиентов найдено",
        available_cnt,
        len(recipe["ingredients"]),
    )

    return {
        **state,
        "recipe_data": recipe,
        "search_results": found_products,
        "agent_output": {
            "response_text": response_text,
            "products_mentioned": products_mentioned,
            "has_recommendations": available_cnt > 0,
            "confidence": "high"
            if available_cnt > len(recipe["ingredients"]) / 2
            else "medium",
        },
        "is_complete": True,
    }
