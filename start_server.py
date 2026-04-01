#!/usr/bin/env python3
"""Запуск сервера с загрузкой .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл ПЕРВЫМ
env_path = Path(__file__).parent / ".env"
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

# Проверяем что загрузилось
print(f"EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL')}")
print(f"CHROMA_DB_PATH: {os.environ.get('CHROMA_DB_PATH')}")
print(f"OLLAMA_BASE_URL: {os.environ.get('OLLAMA_BASE_URL')}")

# Запускаем uvicorn
import uvicorn
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False
)
