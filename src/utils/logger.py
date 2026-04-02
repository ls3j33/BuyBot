"""
Логгер для проекта BuyBot.
Централизованная настройка логирования.
"""
import logging
import sys
from pathlib import Path


# Глобальный логгер для приложения
# Создается только при первом использовании через get_app_logger()
_app_logger: logging.Logger | None = None


def get_app_logger() -> logging.Logger:
    """Получить или создать глобальный логгер приложения"""
    global _app_logger
    
    if _app_logger is None:
        _app_logger = logging.getLogger("buybot")
        _app_logger.setLevel(logging.INFO)

        # Формат сообщений
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        _app_logger.addHandler(console_handler)

        # File handler
        log_path = Path("logs/buybot.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        _app_logger.addHandler(file_handler)

    return _app_logger
