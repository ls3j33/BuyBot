# BuyBot - Умный помощник покупок

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM-чат-бот с RAG (Retrieval-Augmented Generation) для поиска товаров в ассортименте российских супермаркетов.

## 📊 Статус проекта

✅ **Продакшн-готов** — все тесты проходят  
✅ **Мультиагентная архитектура** — LangGraph  
✅ **Векторный поиск** — ChromaDB + эмбеддинги  
✅ **Полная документация** — API, тесты, примеры

## Возможности

- 🛒 **Умный поиск товаров** — LLM определяет конкретные товары для поиска
- 💰 **Сортировка по цене** — запросы "дешевый", "дорогой"
- 📏 **Фильтр по объему** — "литр молока", "500 мл"
- 🍳 **Поиск по рецептам** — полные рецепты с проверкой наличия ингредиентов:
  - LLM генерирует список ингредиентов для блюда
  - Поиск каждого ингредиента в базе супермаркета
  - Показывает какие продукты есть в наличии с ценами
  - Отмечает отсутствующие ингредиенты
  - Примеры: "рецепт борща", "рецепт плова", "как приготовить лазанью"
- 🧠 **Мультиагентная система на LangGraph**:
  - **Router Agent** — определяет тип запроса (рецепт или поиск)
  - **Recipe Agent** — обработка запросов рецептов
  - **Search Agent** — поиск товаров с фильтрами и сортировкой
- 🔍 **Векторный поиск** на основе эмбеддингов (ChromaDB + косинусное расстояние)
- 📦 **Ingestion pipeline** — импорт данных из CSV файла супермаркетов
- 📝 **Логирование** — централизованное логирование всех операций

## Быстрый старт

### 1. Установка зависимостей

```bash
# Через uv (рекомендуется)
uv sync

# Или через pip
pip install -r requirements.txt
```

### 2. Настройка окружения

Скопируйте `.env.example` в `.env` и настройте переменные:

```bash
# LLM Модель (Ollama)
OLLAMA_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=qwen3:4b

# Embedding модель
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# ChromaDB
CHROMA_DB_PATH=./chroma_db_e5

# FastAPI
HOST=0.0.0.0
PORT=8000
```

### 3. Запуск Ollama с моделью

```bash
# Установите Ollama: https://ollama.ai
ollama pull qwen3:4b
ollama serve
```

### 4. Импорт данных

```bash
# Импорт из CSV файла с данными супермаркетов
uv run python scripts/ingest_csv.py --limit 1000

# Сбросить коллекцию и загрузить заново
uv run python scripts/ingest_csv.py --reset
```

### 5. Запуск приложения

```bash
# Через run.bat (Windows)
run.bat

# Или напрямую через uv
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /api/v1/chat

Отправить запрос чат-боту.

**Request:**
```json
{
  "query": "самый дешевый литр молока"
}
```

**Response:**
```json
{
  "response": "По запросу \"самый дешевый литр молока\" (объем 900-1100 мл) (сортировка: от дешевых к дорогим) найдено 5 товаров:\n\n**Молоко:**\n- Молоко «Правильное» 2,5%, 900 мл — 69.9 руб.\n- Молоко EkoNiva ультрапастеризованное 3,2%, 1 л — 79.9 руб.\n\n💰 Самый дешевый: Молоко «Правильное» 2,5%, 900 мл — 69.9 руб.",
  "search_results": [
    {
      "id": "...",
      "name": "Молоко «Правильное» 2,5%, 900 мл",
      "category": "Молоко",
      "price": "69.9",
      "distance": 0.18
    }
  ],
  "confidence": "high",
  "success": true
}
```

### GET /api/v1/stats

Получить статистику базы товаров.

**Response:**
```json
{
  "collection_name": "products",
  "total_documents": 3092,
  "is_ready": true
}
```

### GET /api/v1/health

Проверка здоровья сервиса.

**Response:**
```json
{
  "status": "ok",
  "service": "buybot"
}
```

## Примеры запросов

### 🔍 Поиск товаров
- `"сыр для пиццы"` → найдет Моцареллу, Пармезан, Чеддер (поиск по категории)
- `"какой сыр лучше для пиццы"` → найдет сыры для пиццы с рекомендациями
- `"завтрак"` → найдет яйца, молоко, хлеб, сыр
- `"ужин"` → найдет мясо, рыбу, овощи, гарнир

### 💰 Сортировка по цене
- `"самый дешевый литр молока"` → сортировка от дешевых к дорогим + фильтр объема ~1л
- `"самый дорогой сыр"` → сортировка от дорогих к дешевым

### 📏 Фильтр по объему
- `"литр молока"` → только товары 900-1100 мл
- `"500 мл"` → только товары 400-600 мл

### 🍳 Рецепты
- `"рецепт борща"` → полный рецепт с проверкой наличия ингредиентов:
  ```
  # Борщ

  ## Ингредиенты и наличие в магазине
  [ЕСТЬ] говядина на косточке — Говядина тушеная «Курганский Стандарт», 338 г — 154.9 руб.
  [ЕСТЬ] свёкла — Свёкла свежая — 89.99 руб.
  [ЕСТЬ] капуста — Капуста белокочанная — 79.99 руб.

  **Найдено 17 из 17 ингредиентов**

  ## Приготовление
  1. Промыть говядину, залить холодной водой...
  ```
- `"рецепт пиццы"` → рецепт Пиццы Маргарита с моцареллой и пармезаном
- `"рецепт плова"` → рецепт с доступными продуктами (рис, говядина, морковь, лук)
- `"как приготовить лазанью"` → пошаговый рецепт с ингредиентами

### 📊 Статистика
- `GET /api/v1/stats` → количество товаров в базе

## Структура проекта

```
BuyBot/
├── app/                    # FastAPI приложение
│   └── main.py            # Точка входа (lifespan, middleware, CORS)
├── src/
│   ├── agents/            # Агенты LangGraph
│   │   ├── __init__.py    # Инициализация и регистрация агентов
│   │   ├── state.py       # AgentState (TypedDict для LangGraph)
│   │   ├── graph.py       # LangGraph граф (оркестрация)
│   │   ├── registry.py    # Реестр агентов (динамическая регистрация)
│   │   ├── router_agent.py # Роутер (LLM-классификация запросов)
│   │   ├── search_agent.py # Поиск товаров (LLM + векторный поиск + фильтры)
│   │   └── recipe_agent.py # Генерация рецептов (LLM + поиск ингредиентов)
│   ├── api/               # API маршруты
│   │   ├── routes.py      # Endpoints (/chat, /stats, /health)
│   │   └── schemas.py     # Pydantic модели для API (ChatRequest, ChatResponse)
│   ├── config/            # Конфигурация
│   │   ├── settings.py    # Настройки приложения (env variables)
│   │   └── llm.py         # LLM конфигурация (Ollama)
│   ├── ingestion/         # Pipeline импорта данных
│   │   └── vector_store.py # ChromaDB векторное хранилище + эмбеддинги
│   ├── tools/             # Tools для агентов
│   │   └── vector_search.py # Утилиты поиска (get_vector_store singleton)
│   ├── models/            # Pydantic модели
│   │   └── schemas.py     # SearchIntent и др.
│   └── utils/             # Утилиты
│       └── logger.py      # Централизованное логирование
├── prompts/               # Промпты для LLM
│   └── recipe_writer.txt  # Промпт для генерации рецептов (с примерами)
├── tests/                 # Тесты (pytest)
│   ├── conftest.py        # Фикстуры
│   ├── test_api.py        # Тесты API endpoints (11 тестов)
│   ├── test_agents.py     # Тесты агентов (12 тестов)
│   └── test_vector_store.py # Тесты векторного хранилища (7 тестов, skip)
├── scripts/
│   └── ingest_csv.py      # Импорт данных из CSV в ChromaDB
├── data/                  # Данные (CSV файлы супермаркетов)
├── logs/                  # Логи (игнорируется в git)
├── chroma_db_e5/          # ChromaDB (игнорируется в git)
├── reports/               # HTML отчёты тестов (игнорируется в git)
├── htmlcov/               # Coverage отчёты (игнорируется в git)
├── .env.example           # Пример переменных окружения
├── .github/
│   └── workflows/
│       └── tests.yml      # GitHub Actions CI/CD
├── .gitignore
├── pyproject.toml         # Зависимости проекта (uv/pip)
├── README.md
└── run.bat                # Скрипт запуска (Windows)
```

## Тестирование

### Статус тестов

| Категория | Тестов | Пройдено | Пропущено | Время |
|-----------|--------|----------|-----------|-------|
| **API Tests** | 11 | ✅ 11 | 0 | ~30 сек |
| **Agent Tests** | 12 | ✅ 12 | 0 | ~10 сек |
| **Vector Tests** | 7 | 0 | ⏭️ 7 | 0 сек |
| **ВСЕГО** | **30** | **✅ 23** | **⏭️ 7** | **~40 сек** |


### Запуск тестов

```bash
# Быстрые тесты (рекомендуется для CI/CD)
uv run pytest -v -m "not slow"

# Все тесты (включая медленные с эмбеддингами)
uv run pytest -v

# С покрытием кода
uv run pytest --cov=src --cov-report=html -m "not slow"

# С HTML-отчётом
uv run pytest --html=reports/test_report.html -m "not slow"

# Конкретный тест
uv run pytest tests/test_api.py::TestHealthEndpoint -v

# Только API тесты (быстрые)
uv run pytest tests/test_api.py -v

# Только агент тесты (быстрые)
uv run pytest tests/test_agents.py -v
```

### Просмотр результатов

| Команда | Где результат |
|---------|---------------|
| `pytest` | Терминал |
| `pytest -v` | Терминал (подробно) |
| `pytest --html=reports/test_report.html` | `reports/test_report.html` |
| `pytest --cov=src --cov-report=html` | `htmlcov/index.html` |

### Примеры тестов

**Тест API:**
```python
def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "buybot"}
```

**Тест классификации:**
```python
def test_router_recipe_query():
    state = {"user_query": "рецепт борща", "is_complete": False}
    result = router_agent(state)
    assert result["router_decision"]["query_type"] == "recipe"
```

**Тест рекомендации:**
```python
def test_chat_recommendation_query():
    response = client.post("/api/v1/chat", json={"query": "какой сыр лучше для пиццы"})
    assert response.status_code == 200
    assert response.json()["success"] is True
    # Проверяем что найдены сыры, а не чипсы
    for result in response.json()["search_results"]:
        assert "сыр" in result["category"].lower() or "моцарелла" in result["name"].lower()
```

---

## CI/CD

### GitHub Actions

Для автоматического запуска тестов при каждом commit создайте файл `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: uv sync --extra dev
        
      - name: Run tests
        run: uv run pytest -v -m "not slow"
        
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

### Локальная проверка

```bash
# Перед commit проверьте тесты
uv run pytest -v -m "not slow"

# Проверьте покрытие
uv run pytest --cov=src -m "not slow"
```

---

## Технологии

- **Backend:** FastAPI, Python 3.11+
- **LLM:** Ollama (qwen3:4b)
- **Embeddings:** intfloat/multilingual-e5-large
- **Vector DB:** ChromaDB
- **Orchestration:** LangGraph/LangChain

## Требования

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (рекомендуется) или pip
- Ollama с моделью Qwen 3 (или другой LLM)
- CSV файл с данными супермаркетов (опционально)

## Лицензия

MIT
