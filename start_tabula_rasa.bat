@echo off
title Tabula Rasa AI
cd /d "%~dp0"
cls

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        Tabula Rasa AI System Startup        ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ─── Kill old processes ───
echo [*] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8002 " ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000 " ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
ping -n 3 127.0.0.1 >nul
echo  [+] Ports 8000,8002 cleared.
echo.

:: ─── Start Tabula Rasa AI ───
echo [*] Starting Tabula Rasa AI (port 8002)...
echo  - Dashboard: http://localhost:8000
echo  - AI:        http://localhost:8002
echo.
start "Tabula Rasa AI" /MIN cmd /c "title Tabula Rasa AI && tabula-rasa serve"

:: ─── Wait ───
echo [*] Waiting for server to start...
ping -n 16 127.0.0.1 >nul
echo.

:: ─── Open dashboard ───
echo [*] Opening dashboard in browser...
start http://localhost:8000
echo  [+] Dashboard opened.
echo.

echo  ╔══════════════════════════════════════════════╗
echo  ║           All systems running!              ║
echo  ║                                            ║
echo  ║  Dashboard : http://localhost:8000          ║
echo  ║  AI API    : http://localhost:8002          ║
echo  ║                                            ║
echo  ║  pip install tabula-rasa-ai                ║
echo  ║  tabula-rasa train add --quick             ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo [*] Close the server window or press Ctrl+C to stop.
echo.
pause >nul
