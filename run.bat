@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .env ( copy .env.example .env >nul & echo .env created from .env.example: set API_KEY, VLM_PATH, OWL_PATH & notepad .env )
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
echo === Forte service: real models (PIPELINE=vlm), models load ~1 min ===
echo Web page:  http://localhost:8000
echo API:       POST http://localhost:8000/api/v1/analyze   header X-API-Key = API_KEY from .env
echo Stop:      Ctrl+C
start "" http://localhost:8000
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
pause
