@echo off
REM Windows One-Click Launcher for Knowledge AI Chatbot
echo ===================================================
echo   Starting Knowledge AI Documentation Chatbot...
echo ===================================================

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH! Please install Python 3.10+.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt --quiet
python run_server.py
pause
