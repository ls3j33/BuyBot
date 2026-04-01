from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config.llm import get_llm
from src.agents.state import AgentState, RAGDecision


PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "rag_agent.txt"


def load_rag_prompt() -> str:
    """Загрузить промпт для RAG агента"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def create_rag_chain():
    """Создать цепочку для RAG агента"""
    prompt_text = load_rag_prompt()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{query}")
    ])
    
    llm = get_llm()
    parser = JsonOutputParser()
    
    return prompt | llm | parser


def rag_agent(state: AgentState) -> AgentState:
    """
    RAG агент: определяет необходимость поиска в базе данных.

    Анализирует запрос и решает, нужно ли вызывать vector_search tool.
    """
    query = state["user_query"]
    print(f"[RAG] Запрос: {query}")

    try:
        chain = create_rag_chain()
        result = chain.invoke({"query": query})
        print(f"[RAG] Результат: {result}")

        # Валидация результата
        if result is None:
            # Если LLM не вернул результат, используем дефолт
            rag_decision: RAGDecision = {
                "should_search": True,
                "search_query": query,
                "search_category": None,
                "n_results": 10,
                "reason": "Поиск по умолчанию"
            }
        else:
            rag_decision: RAGDecision = {
                "should_search": result.get("should_search", True),
                "search_query": result.get("search_query", query),
                "search_category": result.get("search_category") if result else None,
                "n_results": result.get("n_results", 10) if result else 10,
                "reason": result.get("reason", "Поиск") if result else "Поиск"
            }

        print(f"[RAG] should_search: {rag_decision['should_search']}")
        return {
            **state,
            "rag_decision": rag_decision
        }

    except Exception as e:
        print(f"[RAG] Ошибка: {e}")
        # При ошибке — всё равно ищем
        return {
            **state,
            "rag_decision": {
                "should_search": True,
                "search_query": query,
                "search_category": None,
                "n_results": 10,
                "reason": f"Поиск по умолчанию (ошибка: {str(e)})"
            }
        }
