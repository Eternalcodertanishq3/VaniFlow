# VaaniFlow One-Click PowerShell Launcher
$Host.UI.RawUI.WindowTitle = "VaaniFlow Launcher"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "              🎙️  Launching VaaniFlow  🎙️              " -ForegroundColor BrightGreen
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check venv
if (-not (Test-Path "venv")) {
    Write-Host "[!] Virtual environment not found. Creating venv..." -ForegroundColor Yellow
    python -m venv venv
}

# 2. Activate venv
Write-Host "[*] Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# 3. Check dependencies
Write-Host "[*] Checking dependencies..." -ForegroundColor Cyan
python -m pip install -e ".[dev]" --quiet

# 4. Check spaCy model
Write-Host "[*] Checking spaCy model..." -ForegroundColor Cyan
$spacyCheck = python -m spacy info en_core_web_sm 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[*] Downloading spaCy model en_core_web_sm..." -ForegroundColor Yellow
    python -m spacy download en_core_web_sm
}

# 5. Open browser
Write-Host "[*] Starting server at http://localhost:8000 ..." -ForegroundColor Green
Start-Process "http://localhost:8000/docs"

# 6. Run server
python -m uvicorn api.main:app --reload --port 8000
