@echo off
title Tabula Rasa AI - Training Manager
cd /d "C:\Users\Admin\tabula-rasa"

:: ─── Colors ───
set "GREEN=[92m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

:menu
cls
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        Tabula Rasa Training Manager         ║
echo  ║  Select a specialist to train               ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo   [1] Addition Specialist
echo   [2] Subtraction Specialist
echo   [3] Multiplication Specialist
echo   [4] Division Specialist
echo   [5] Run Sleep Cycle (once, now)
echo   [6] Start All Servers
echo   [7] Show Training Status
echo.
echo   [0] Exit
echo.

set /p choice="Select option (0-7): "

if "%choice%"=="1" goto train_addition
if "%choice%"=="2" goto train_subtraction
if "%choice%"=="3" goto train_multiplication
if "%choice%"=="4" goto train_division
if "%choice%"=="5" goto sleep_cycle
if "%choice%"=="6" goto start_servers
if "%choice%"=="7" goto show_status
if "%choice%"=="0" goto exit
goto menu

:train_addition
cls
echo.
echo  Starting Addition Specialist Training...
echo  This will take approximately 8 hours on CPU.
echo.
start "Addition Training" /B /MIN pythonw train_specialist.py add
echo  [+] Addition training started in background.
echo  Log: specialists\math\add\training.log
timeout /t 3 /nobreak >nul
goto menu

:train_subtraction
cls
echo.
echo  Starting Subtraction Specialist Training...
echo  This will take approximately 8 hours on CPU.
echo.
start "Subtraction Training" /B /MIN pythonw train_specialist.py sub
echo  [+] Subtraction training started in background.
echo  Log: specialists\math\sub\training.log
timeout /t 3 /nobreak >nul
goto menu

:train_multiplication
cls
echo.
echo  Starting Multiplication Specialist Training...
echo  This will take approximately 8 hours on CPU.
echo.
start "Multiplication Training" /B /MIN pythonw train_specialist.py mul
echo  [+] Multiplication training started in background.
echo  Log: specialists\math\mul\training.log
timeout /t 3 /nobreak >nul
goto menu

:train_division
cls
echo.
echo  Starting Division Specialist Training...
echo  This will take approximately 8 hours on CPU.
echo.
start "Division Training" /B /MIN pythonw train_specialist.py div
echo  [+] Division training started in background.
echo  Log: specialists\math\div\training.log
timeout /t 3 /nobreak >nul
goto menu

:sleep_cycle
cls
echo.
echo  Running Sleep Cycle (once)...
pythonw -c "from egefalos.sleep_cycle import run_all_cycles; run_all_cycles(200)"
echo  [+] Sleep cycle complete.
timeout /t 3 /nobreak >nul
goto menu

:start_servers
cls
echo.
echo  Starting all servers...
call start_tabula_rasa.bat
exit /b

:show_status
cls
echo.
echo  Current Training Status:
echo.
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr /i "train_specialist"
if %errorlevel%==0 (
    echo  [*] Training processes are running.
) else (
    echo  [*] No training processes are currently running.
)
echo.
echo  Training logs:
for %%d in (specialists\math\*) do (
    if exist %%d\training.log (
        for /f "skip=2 tokens=*" %%a in ('findstr /n "Step" %%d\training.log 2^>nul') do set last=%%a
        for %%a in ("%%d") do echo  %%~nxa: %last%
    )
)
echo.
pause
goto menu

:exit
exit /b
