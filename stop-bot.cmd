@echo off
cd /d "%~dp0"
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":6199" ^| findstr "LISTENING"') do (
  set FOUND=1
  echo stopping PID %%a on port 6199
  taskkill /F /PID %%a >nul 2>&1
)
if "%FOUND%"=="0" echo 6199 is free
ping 127.0.0.1 -n 2 >nul
