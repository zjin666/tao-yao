@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
"D:\tao-yao-venv\Scripts\python.exe" run.py
pause
