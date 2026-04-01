from src.agents.writer import writer_agent, is_recipe_query

async def process_query(query: str, conversation_history=None):
    """Обработать запрос через writer агента"""
    from src.agents.state import AgentState

    # Для всех запросов передаём пустые search_results
    # writer_agent сам определит что искать через LLM
    state: AgentState = {
        "user_query": query,
        "search_results": [],  # Пустые — writer_agent найдёт сам
        "rag_decision": None,
        "writer_output": None,
        "is_complete": False,
        "messages": []
    }

    result = writer_agent(state)

    output = result.get("writer_output", {})
    return {
        "response": output.get("response_text", ""),
        "search_results": result.get("search_results", []),
        "confidence": output.get("confidence", "low"),
        "success": True,
        "error": None
    }
