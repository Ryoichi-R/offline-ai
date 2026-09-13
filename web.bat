@echo off
setlocal enabledelayedexpansion
REM === Offline AI Web UI Launcher ===
REM NOTE: Japanese text must NOT appear in .bat files (cmd.exe UTF-8 bug).
REM
REM Flow: web.bat -> pythonw web_launcher.py -> web_server.py (browser only)
REM Set OFFLINE_AI_WEB_CONSOLE=1 to retain console diagnostics and Ctrl+C stop.
REM Update trigger: Update this comment when dataflow, exit codes, or security design changes.

REM --- Detect Python (pin first resolved python.exe to avoid Store stub / shim mismatch) ---
set "PY="
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PY set "PY=%%i"
)
if not defined PY (
    echo [ERROR] Python not found. Install offline-ai from offline-ai-offline-package\scripts\install-offline.bat first.
    pause
    exit /b 1
)
"%PY%" -c "import sys; v=sys.version_info; exit(0 if v>=(3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 or later is required.
    "%PY%" --version
    pause
    exit /b 1
)

REM --- Start Web UI server ---
if defined OFFLINE_AI_WEB_CONSOLE goto :check_ollama
for %%i in ("%PY%") do set "PYW=%%~dpipythonw.exe"
if not exist "%PYW%" (
    echo [ERROR] pythonw.exe not found next to Python. Repair the Python installation.
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0_internal\web_launcher.py" %*
exit /b %ERRORLEVEL%

:check_ollama
REM --- Check Ollama is running ---
curl -s http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 goto :ollama_ready

REM --- Ollama not running; try to start ---
echo Starting Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not found. Install offline-ai from offline-ai-offline-package\scripts\install-offline.bat first.
    pause
    exit /b 1
)
start /b "" ollama serve 2>"%TEMP%\ollama_start.log"

REM --- Wait for startup (max 30 seconds) ---
set "WAIT_COUNT=0"
:wait_loop
timeout /t 2 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 goto :ollama_ready
set /a WAIT_COUNT+=2
if !WAIT_COUNT! geq 30 (
    echo [ERROR] Ollama startup timed out after !WAIT_COUNT! seconds.
    echo [INFO] Check log: %TEMP%\ollama_start.log
    echo [INFO] To retry: re-run this script or the offline package installer.
    pause
    exit /b 1
)
echo Waiting for Ollama... (!WAIT_COUNT!s)
goto :wait_loop

:ollama_ready
REM Carry the resolved Python path beyond endlocal.
endlocal & set "PY=%PY%"
echo Ollama: ready
echo.


:console_server
echo Starting offline-ai Web UI...
echo (Press Ctrl+C to stop)
echo.
"%PY%" "%~dp0_internal\web_server.py" %*
set "WEB_RC=%ERRORLEVEL%"

if not "%WEB_RC%"=="0" (
    echo.
    echo [ERROR] Web server exited with error: %WEB_RC%
    pause
    exit /b %WEB_RC%
)

exit /b 0
