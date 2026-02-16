@echo off
chcp 65001
echo ==========================================
echo       Upbit SMA Auto-Trading System
echo ==========================================
echo Starting the application...
echo.

python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed or failed to start.
    echo Please check if the virtual environment is active or dependencies are installed.
    pause
)
