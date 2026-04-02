@echo off
echo ==================================================
echo BuyBot Server
echo ==================================================
echo.
echo Запуск сервера...
echo.

REM Запускаем в отдельном окне чтобы сервер не "засыпал"
start "BuyBot Server" cmd /k "uv run python start_server.py"

echo.
echo Сервер запущен в новом окне!
echo Swagger UI: http://localhost:8000/docs
echo ReDoc: http://localhost:8000/redoc
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
