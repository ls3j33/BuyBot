@echo off
set EMBEDDING_MODEL=intfloat/multilingual-e5-large
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set LLM_MODEL=qwen3:4b
set CHROMA_DB_PATH=./chroma_db_e5
echo Starting server with EMBEDDING_MODEL=%EMBEDDING_MODEL%
uv run python start_server.py
