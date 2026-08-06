@echo off
title VaaniFlow Launcher
echo ========================================================
echo               🎙️  Launching VaaniFlow  🎙️
echo ========================================================
echo.

:: 1. Check if venv exists
if not exist "venv" (
    echo [!] Virtual environment not found. Creating venv...
    python -m venv venv
)

:: 2. Activate venv
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

:: 3. Ensure pip and dependencies are installed
echo [*] Checking dependencies...
python -m pip install -e ".[dev]" --quiet

:: 4. Ensure spaCy model exists
echo [*] Checking spaCy model...
python -m spacy info en_core_web_sm >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Downloading spaCy model en_core_web_sm...
    python -m spacy download en_core_web_sm
)

:: 5. Open Browser automatically after 3 seconds
echo [*] Starting server at http://localhost:8000/ui ...
start "" "http://localhost:8000/ui"

:: 6. Launch FastAPI Uvicorn Server
python -m uvicorn api.main:app --reload --port 8000
