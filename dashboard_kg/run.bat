@echo off
echo Iniciando TDA Crypto Trading Dashboard...
cd /d "%~dp0.."
.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8501
pause
