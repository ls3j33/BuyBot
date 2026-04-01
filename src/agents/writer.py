from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config.llm import get_llm
from src.agents.state import AgentState, WriterOutput


PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "writer.txt"
RECIPE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "recipe_writer.txt"

# Ключевые слова для определения запроса рецепта
RECIPE_KEYWORDS = [
    "рецепт", "борщ", "суп", "каша", "плов", "котлеты", "пельмени",
    "запеканка", "омлет", "салат", "пюре", "бульон", "рагу", "гуляш",
    "как приготовить", "как сварить", "как сделать", "как пожарить",
    "приготовление", "готовить", "сварить", "пожарить", "запечь"
]


def is_recipe_query(query: str) -> bool:
    """Проверить, является ли запрос запросом рецепта"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in RECIPE_KEYWORDS)


def generate_recipe_with_llm(query: str) -> dict:
    """Сгенерировать рецепт через LLM для неизвестных блюд"""
    try:
        print(f"[LLM Recipe] Генерирую рецепт для: {query}")
        chain = create_recipe_writer_chain()
        result = chain.invoke({"query": query})
        print(f"[LLM Recipe] Результат: {result}")
        
        # Валидация результата
        if result and isinstance(result, dict):
            ingredients = result.get("ingredients", [])
            if ingredients and len(ingredients) > 0:
                return {
                    "recipe_name": result.get("recipe_name", "Рецепт"),
                    "intro": result.get("intro", ""),
                    "ingredients": ingredients,
                    "steps": result.get("steps", []),
                    "tips": result.get("tips", [])
                }
            else:
                print(f"[LLM Recipe] Пустые ингредиенты в результате")
    except Exception as e:
        print(f"[LLM Recipe] Ошибка генерации: {e}")
    
    # Фолбэк — пустой рецепт
    return {
        "recipe_name": "Рецепт",
        "intro": "Универсальный рецепт.",
        "ingredients": [],
        "steps": [],
        "tips": []
    }


def clean_ingredient_name(ingredient: str) -> str:
    """Очистить название ингредиента от количества (г, кг, шт, ст. л. и т.д.)"""
    import re
    # Удаляем количества: "500 г", "2 шт.", "1 ст. л.", "по вкусу", "— 500–700 г"
    cleaned = re.sub(r'\s*—?\s*\d+[–\-]?\d*\s*(г|кг|шт|мл|л|ст\.?\s*л\.?|ч\.?\s*л\.?|для подачи|по вкусу).*', '', ingredient)
    # Удаляем размеры: "2 шт. среднего размера"
    cleaned = re.sub(r'\s+среднего размера', '', cleaned)
    # Удаляем тип зелени в скобках
    cleaned = re.sub(r'\s*\(.*\)', '', cleaned)
    return cleaned.strip()


def find_ingredients_in_results(ingredients: list[str], search_results: list[dict], threshold: float = 0.95) -> dict[str, list[dict]]:
    """
    Найти каждый ингредиент в результатах поиска.

    Args:
        ingredients: Список ингредиентов для поиска
        search_results: Результаты векторного поиска
        threshold: Порог расстояния для релевантности

    Returns:
        Словарь {ингредиент: [найденные товары]}
    """
    found = {}

    for ingredient in ingredients:
        # Очищаем название от количества для поиска
        ingredient_clean = clean_ingredient_name(ingredient)
        ingredient_lower = ingredient_clean.lower().strip()
        found[ingredient] = []
        
        # Сначала ищем точное совпадение начала названия
        exact_matches = []
        partial_matches = []
        
        for result in search_results:
            metadata = result.get("metadata", {})
            name = metadata.get("name", "").lower()
            distance = result.get("distance", 1.0)
            
            # Проверяем релевантность по расстоянию
            if distance < threshold:
                # Проверяем, соответствует ли товар ингредиенту
                is_exact = False
                is_partial = False
                
                # Для многокомпонентных ингредиентов (например "говядина на косточке")
                # проверяем наличие ключевых слов в названии
                ingredient_words = ingredient_lower.split()
                
                if len(ingredient_words) > 1:
                    # Многокомпонентный ингредиент - проверяем ключевые слова
                    if ingredient_lower in ["говядина на косточке", "говядина на кости"]:
                        # Ищем "говядина" + "кость" или "говядина на"
                        if "говядина" in name and ("кост" in name or "на кости" in name):
                            is_exact = True
                        elif "говядина" in name:
                            is_partial = True
                    elif ingredient_lower in ["лук репчатый"]:
                        if name.startswith("лук репчатый"):
                            is_exact = True
                        elif "лук" in name and "зелен" not in name:
                            is_partial = True
                    elif ingredient_lower in ["томатная паста"]:
                        if name.startswith("томатная паста"):
                            is_exact = True
                        elif "томат" in name and "сок" not in name and "кетчуп" not in name:
                            is_partial = True
                    elif ingredient_lower in ["белокочанная капуста"]:
                        if "капуста" in name and "белокочан" in name:
                            is_exact = True
                        elif "капуста" in name:
                            is_partial = True
                    elif ingredient_lower in ["лавровый лист"]:
                        if name.startswith("лавровый"):
                            is_exact = True
                        elif "лавр" in name:
                            is_partial = True
                else:
                    # Однокомпонентный ингредиент - точное совпадение начала
                    if name.startswith(ingredient_lower) or ingredient_lower in name.split()[0]:
                        is_exact = True
                    # Для овощей и мяса — частичное совпадение
                    elif ingredient_lower in ["говядина", "мясо"] and any(m in name for m in ["говядина", "мясо", "телятина"]):
                        is_partial = True
                    elif ingredient_lower in ["морковь"] and name.startswith("морковь"):
                        is_exact = True
                    elif ingredient_lower in ["свёкла"] and name.startswith("свёкл"):
                        is_exact = True
                    elif ingredient_lower in ["капуста"] and name.startswith("капуста"):
                        is_exact = True
                    elif ingredient_lower in ["картофель"] and name.startswith("картофель"):
                        is_exact = True
                    elif ingredient_lower in ["чеснок"] and name.startswith("чеснок"):
                        is_exact = True
                    elif ingredient_lower in ["рис"] and name.startswith("рис"):
                        is_exact = True
                
                if is_exact:
                    exact_matches.append({
                        "id": result.get("id", ""),
                        "metadata": metadata,
                        "distance": distance
                    })
                elif is_partial:
                    partial_matches.append({
                        "id": result.get("id", ""),
                        "metadata": metadata,
                        "distance": distance
                    })
        
        # Сортируем: сначала точные совпадения, потом по distance
        exact_matches.sort(key=lambda x: x["distance"])
        partial_matches.sort(key=lambda x: x["distance"])
        found[ingredient] = exact_matches + partial_matches
    
    return found


def load_writer_prompt() -> str:
    """Загрузить промпт для writer агента"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_recipe_prompt() -> str:
    """Загрузить промпт для recipe writer агента"""
    with open(RECIPE_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def create_writer_chain():
    """Создать цепочку для writer агента"""
    prompt_text = load_writer_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", """
Запрос пользователя: {query}

Найденные товары:
{search_results}

Сформируй ответ для пользователя.
""")
    ])

    llm = get_llm()
    parser = JsonOutputParser()

    return prompt | llm | parser


def create_recipe_writer_chain():
    """Создать цепочку для recipe writer агента"""
    prompt_text = load_recipe_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Запрос: {query}\n\nВерни JSON с рецептом.")
    ])

    llm = get_llm()
    parser = JsonOutputParser()

    return prompt | llm | parser


def detect_sort_from_query(query: str) -> str | None:
    """Определить сортировку из запроса по ключевым словам"""
    query_lower = query.lower()
    
    # Дешевый/недорогой/дешевле
    if any(word in query_lower for word in ["дешев", "недорог", "дешевле", "эконом"]):
        return "price_asc"
    
    # Дорогой/дороже
    if any(word in query_lower for word in ["дорог", "дороже", "премиум"]):
        return "price_desc"
    
    return None


def detect_volume_from_query(query: str) -> tuple[float, float] | None:
    """Определить требуемый объем из запроса (возвращает мин/макс в мл)"""
    query_lower = query.lower()
    
    # "литр", "1л", "1 л"
    if "литр" in query_lower or "1л" in query_lower.replace(" ", "") or "1 л" in query_lower:
        return (900, 1100)  # 900-1100 мл
    
    # "поллитра", "0.5л", "500 мл"
    if "поллитра" in query_lower or "0.5л" in query_lower.replace(" ", "") or "500 мл" in query_lower:
        return (400, 600)  # 400-600 мл
    
    return None


def parse_volume_from_name(name: str) -> float | None:
    """Извлечь объем из названия товара в мл"""
    import re
    name_lower = name.lower()
    
    # Ищем "1 л", "1л", "1.5 л" и т.д.
    match = re.search(r'(\d+\.?\d*)\s*л\b', name_lower)
    if match:
        return float(match.group(1)) * 1000
    
    # Ищем "900 мл", "500мл" и т.д.
    match = re.search(r'(\d+)\s*мл\b', name_lower)
    if match:
        return float(match.group(1))
    
    # Ищем "1 кг" (для сыров и т.д.)
    match = re.search(r'(\d+)\s*кг\b', name_lower)
    if match:
        return float(match.group(1)) * 1000
    
    # Ищем просто число грамм (160 г, 225 г)
    match = re.search(r'(\d+)\s*г\b', name_lower)
    if match:
        return float(match.group(1))
    
    return None


def filter_by_volume(results: list[dict], volume_range: tuple[float, float]) -> list[dict]:
    """Отфильтровать товары по объему"""
    min_vol, max_vol = volume_range
    filtered = []
    
    for r in results:
        name = r.get('metadata', {}).get('name', '')
        volume = parse_volume_from_name(name)
        
        if volume and min_vol <= volume <= max_vol:
            filtered.append(r)
    
    return filtered if filtered else results  # Если ничего не найдено, возвращаем все


def generate_items_for_query(query: str) -> dict:
    """Сгенерировать список товаров для поиска через LLM для обычных запросов"""
    print(f"[LLM Items] Запрос: {query}")
    
    # Сначала определяем сортировку по ключевым словам
    sort_order = detect_sort_from_query(query)
    print(f"[LLM Items] Сортировка из ключевых слов: {sort_order}")
    
    # Определяем требуемый объем
    volume_range = detect_volume_from_query(query)
    print(f"[LLM Items] Объем из ключевых слов: {volume_range}")
    
    prompt_text = load_writer_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Запрос: {query}\n\nВерни JSON.")
    ])
    
    llm = get_llm()
    parser = JsonOutputParser()
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"query": query})
        print(f"[LLM Items] Результат LLM: {result}")
        
        # Валидация результата
        if result and isinstance(result, dict):
            items = result.get("items_to_search", [])
            if items and len(items) > 0:
                print(f"[LLM Items] Товары для поиска: {items}")
                return {
                    "items_to_search": items,
                    "sort": sort_order or result.get("sort"),
                    "max_price": result.get("max_price"),
                    "volume_range": volume_range,
                    "confidence": result.get("confidence", "medium")
                }
            else:
                print(f"[LLM Items] Пустой items_to_search")
        else:
            print(f"[LLM Items] Результат не dict: {type(result)}")
    except Exception as e:
        print(f"[LLM Items] Ошибка: {e}")
    
    # Фолбэк — используем исходный запрос как товар для поиска
    print(f"[LLM Items] Фолбэк: используем запрос как товар")
    return {
        "items_to_search": [query],
        "sort": sort_order,
        "max_price": None,
        "volume_range": volume_range,
        "confidence": "low"
    }


def format_recipe_availability(found_ingredients: dict[str, list[dict]]) -> tuple[str, int, int]:
    """
    Форматировать наличие ингредиентов для рецепта.
    
    Args:
        found_ingredients: Словарь {ингредиент: [найденные товары]}
        
    Returns:
        Кортеж (текст, количество доступных, количество отсутствующих)
    """
    lines = []
    available_count = 0
    missing_count = 0
    
    for ingredient, products in found_ingredients.items():
        if products:
            available_count += 1
            # Берём первый товар (точное совпадение идёт первым)
            best = products[0]
            metadata = best.get('metadata', {})
            name = metadata.get('name', 'N/A')
            price = metadata.get('price', 'N/A')
            lines.append(f"[ЕСТЬ] {ingredient}: {name} — {price} руб.")
        else:
            missing_count += 1
            lines.append(f"[НЕТ] {ingredient}: нет в наличии")
    
    return "\n".join(lines), available_count, missing_count


def writer_agent(state: AgentState) -> AgentState:
    """
    Writer агент: формирует финальный ответ пользователю.
    """
    query = state["user_query"]
    search_results = state.get("search_results") or []

    if search_results is None:
        search_results = []

    try:
        if is_recipe_query(query):
            # Для рецептов — генерируем рецепт через LLM
            recipe = generate_recipe_with_llm(query)

            if not recipe.get("ingredients"):
                # Неизвестный рецепт (LLM не смог сгенерировать)
                writer_output: WriterOutput = {
                    "response_text": f"К сожалению, у меня пока нет рецепта для блюда \"{query}\". Попробуйте спросить про борщ, плов или другое популярное блюдо!",
                    "products_mentioned": [],
                    "has_recommendations": False,
                    "confidence": "medium"
                }
            else:
                # Ищем каждый ингредиент НАПРЯМУЮ через vector_store
                from src.tools.vector_search import get_vector_store
                store = get_vector_store()

                all_search_results = []

                # Собираем все результаты поиска
                for ingredient in recipe["ingredients"]:
                    # Очищаем название от количества для лучшего поиска
                    clean_name = clean_ingredient_name(ingredient)
                    search_query = clean_name if clean_name else ingredient
                    results = store.search(query=search_query, n_results=20)
                    all_search_results.extend(results)

                # Используем find_ingredients_in_results для фильтрации
                found_ingredients = find_ingredients_in_results(
                    recipe["ingredients"],
                    all_search_results,
                    threshold=0.60  # Увеличенный порог для мультиязычной модели
                )
                
                # Формируем ответ
                lines = []
                lines.append(f"# {recipe['recipe_name']}\n")
                
                if recipe.get("intro"):
                    lines.append(f"{recipe['intro']}\n")
                
                lines.append("## Ингредиенты и наличие в магазине\n")
                availability_text, available_count, missing_count = format_recipe_availability(found_ingredients)
                lines.append(availability_text)
                lines.append("")
                
                lines.append(f"\n**Найдено {available_count} из {len(recipe['ingredients'])} ингредиентов**\n")
                
                if recipe.get("steps"):
                    lines.append("## Приготовление\n")
                    for i, step in enumerate(recipe["steps"], 1):
                        lines.append(f"{i}. {step}")
                    lines.append("")
                
                if recipe.get("tips"):
                    lines.append("## Советы\n")
                    for tip in recipe["tips"]:
                        lines.append(f"- {tip}")
                
                response_text = "\n".join(lines)
                
                # Собираем упомянутые продукты
                products_mentioned = []
                for ingredient, products in found_ingredients.items():
                    for p in products:
                        products_mentioned.append(p.get("metadata", {}).get("name", ""))

                writer_output: WriterOutput = {
                    "response_text": response_text,
                    "products_mentioned": products_mentioned,
                    "has_recommendations": available_count > 0,
                    "confidence": "high" if available_count > len(recipe["ingredients"]) / 2 else "medium"
                }

                # Возвращаем только найденные ингредиенты (не все search_results)
                found_products = []
                for ingredient, products in found_ingredients.items():
                    if products:
                        # Берём первый (лучший) товар для каждого ингредиента
                        found_products.append(products[0])

                # Обновляем search_results для ответа
                return {
                    **state,
                    "writer_output": writer_output,
                    "search_results": found_products,
                    "is_complete": True
                }
        else:
            # Для обычных запросов — используем LLM для определения товаров
            print(f"[Writer] Обычный запрос: {query}")
            
            # Генерируем список товаров через LLM
            items_data = generate_items_for_query(query)
            
            if items_data.get("items_to_search"):
                # Ищем каждый товар в базе
                from src.tools.vector_search import get_vector_store
                store = get_vector_store()
                
                all_search_results = []
                for item in items_data["items_to_search"]:
                    results = store.search(query=item, n_results=10)
                    all_search_results.extend(results)
                
                # Фильтруем релевантные (distance < 0.95)
                relevant_results = [r for r in all_search_results if r.get('distance', 1.0) < 0.95]
                
                # Фильтруем по объему если указано
                volume_range = items_data.get("volume_range")
                if volume_range:
                    relevant_results = filter_by_volume(relevant_results, volume_range)
                    print(f"[Writer] После фильтрации по объему: {len(relevant_results)} товаров")
                
                # Сортируем если указано в запросе
                sort_order = items_data.get("sort")
                if sort_order == "price_asc":
                    relevant_results.sort(key=lambda x: float(x.get('metadata', {}).get('price', 9999)))
                elif sort_order == "price_desc":
                    relevant_results.sort(key=lambda x: float(x.get('metadata', {}).get('price', 0)), reverse=True)
                
                # Фильтруем по max_price если указано
                max_price = items_data.get("max_price")
                if max_price:
                    relevant_results = [r for r in relevant_results if float(r.get('metadata', {}).get('price', 9999)) <= max_price]
                
                if relevant_results:
                    # Группируем по категориям
                    by_category = {}
                    for r in relevant_results:
                        cat = r.get("metadata", {}).get("category", "Другое")
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(r)
                    
                    lines = []
                    # Добавляем информацию о сортировке и объеме
                    volume_info = ""
                    if volume_range:
                        min_v, max_v = volume_range
                        if min_v >= 1000:
                            volume_info = f" (объем ~{int(min_v/1000)} л)"
                        else:
                            volume_info = f" (объем {int(min_v)}-{int(max_v)} мл)"
                    
                    if sort_order == "price_asc":
                        lines.append(f"По запросу \"{query}\"{volume_info} (сортировка: от дешевых к дорогим) найдено {len(relevant_results)} товаров:\n")
                    elif sort_order == "price_desc":
                        lines.append(f"По запросу \"{query}\"{volume_info} (сортировка: от дорогих к дешевым) найдено {len(relevant_results)} товаров:\n")
                    elif max_price:
                        lines.append(f"По запросу \"{query}\"{volume_info} (до {max_price} руб.) найдено {len(relevant_results)} товаров:\n")
                    else:
                        lines.append(f"По запросу \"{query}\"{volume_info} найдено {len(relevant_results)} товаров:\n")
                    
                    products_mentioned = []
                    for cat, items in by_category.items():
                        lines.append(f"\n**{cat}:**")
                        for item in items[:5]:
                            name = item.get("metadata", {}).get("name", "N/A")
                            price = item.get("metadata", {}).get("price", "N/A")
                            lines.append(f"- {name} — {price} руб.")
                            products_mentioned.append(name)
                    
                    if len(relevant_results) > 10:
                        lines.append(f"\n... и ещё {len(relevant_results) - 10} товаров.")
                    
                    # Добавляем рекомендацию для запросов с сортировкой
                    if sort_order == "price_asc" and relevant_results:
                        cheapest = relevant_results[0]
                        lines.append(f"\n💰 Самый дешевый: {cheapest.get('metadata', {}).get('name')} — {cheapest.get('metadata', {}).get('price')} руб.")
                    elif sort_order == "price_desc" and relevant_results:
                        expensive = relevant_results[0]
                        lines.append(f"\n💎 Самый дорогой: {expensive.get('metadata', {}).get('name')} — {expensive.get('metadata', {}).get('price')} руб.")
                    
                    response_text = "\n".join(lines)
                    confidence = "high"
                    
                    # Возвращаем найденные товары
                    found_products = relevant_results[:10]
                    
                    writer_output: WriterOutput = {
                        "response_text": response_text,
                        "products_mentioned": products_mentioned,
                        "has_recommendations": True,
                        "confidence": confidence
                    }
                    
                    return {
                        **state,
                        "writer_output": writer_output,
                        "search_results": found_products,
                        "is_complete": True
                    }
            
            # Фолбэк — используем search_results из state
            relevant_results = [r for r in search_results if r.get('distance', 1.0) < 0.95]

            if not relevant_results:
                response_text = "К сожалению, по вашему запросу ничего не найдено. Попробуйте уточнить запрос."
                products_mentioned = []
                confidence = "low"
            else:
                # Группируем товары по категориям
                by_category = {}
                for r in relevant_results:
                    cat = r.get("metadata", {}).get("category", "Другое")
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(r)

                lines = []
                lines.append(f"По запросу \"{query}\" найдено {len(relevant_results)} товаров:\n")

                products_mentioned = []
                for cat, items in by_category.items():
                    lines.append(f"\n**{cat}:**")
                    for item in items[:5]:  # Показываем до 5 товаров в категории
                        name = item.get("metadata", {}).get("name", "N/A")
                        price = item.get("metadata", {}).get("price", "N/A")
                        lines.append(f"- {name} — {price} руб.")
                        products_mentioned.append(name)

                if len(relevant_results) > 10:
                    lines.append(f"\n... и ещё {len(relevant_results) - 10} товаров.")

                response_text = "\n".join(lines)
                confidence = "high" if len(relevant_results) > 0 else "medium"

            writer_output: WriterOutput = {
                "response_text": response_text,
                "products_mentioned": products_mentioned,
                "has_recommendations": len(relevant_results) > 0,
                "confidence": confidence
            }

        return {
            **state,
            "writer_output": writer_output,
            "is_complete": True
        }

    except Exception as e:
        import traceback
        error_msg = f"Ошибка: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            **state,
            "writer_output": {
                "response_text": error_msg,
                "products_mentioned": [],
                "has_recommendations": False,
                "confidence": "low"
            },
            "is_complete": True,
            "error": str(e)
        }
