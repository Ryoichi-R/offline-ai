@echo off
setlocal enabledelayedexpansion
REM === Offline AI Search Launcher ===
REM NOTE: Japanese text must NOT appear in .bat files (cmd.exe UTF-8 bug).
REM
REM Flow: search.bat -> _internal/search.py -> keyword or ready Embedding cache -> Ollama API
REM Embedding index construction is explicit: use index.bat build/resume.
REM Query is passed via temp file to avoid command injection. See _internal/ARCHITECTURE.md.
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

REM --- Cleanup stale temp files (older than 1 hour only) ---
powershell -NoProfile -Command "Get-ChildItem -Path $env:TEMP -Filter 'sq_*.tmp' -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } | Remove-Item -Force -ErrorAction SilentlyContinue"

REM --- Search loop (delayed expansion OFF to preserve ! in queries) ---
set "FAIL_COUNT=0"
:query_loop
set "QUERY="
set /p QUERY=Query:
if not defined QUERY (
    echo No query entered.
    goto :query_loop
)

echo.
echo Searching...
echo.

REM Write query to temp file through environment variables to avoid cmd metacharacter expansion.
REM File name uses RANDOM only (locale-independent)
set "QUERY_TMP=%TEMP%\sq_%RANDOM%%RANDOM%%RANDOM%.tmp"
powershell -NoProfile -Command "[IO.File]::WriteAllText($env:QUERY_TMP, $env:QUERY, [System.Text.UTF8Encoding]::new($false))"
"%PY%" "%~dp0_internal\search.py" --query-file "%QUERY_TMP%"
set "SEARCH_RC=%ERRORLEVEL%"
del "%QUERY_TMP%" >nul 2>&1
if exist "%QUERY_TMP%" echo [WARN] Failed to delete temp file: %QUERY_TMP%
if not "%SEARCH_RC%"=="0" (
    echo [WARN] Search exited with code %SEARCH_RC%.
    set /a FAIL_COUNT+=1
)
if "%SEARCH_RC%"=="0" set "FAIL_COUNT=0"

REM --- Consecutive failure guard (max 3) ---
if %FAIL_COUNT% geq 3 (
    echo [ERROR] Search failed 3 times consecutively.
    echo [INFO] Check Ollama status or re-run the offline package installer.
    pause
    exit /b 1
)
echo.

set "CONTINUE="
set /p CONTINUE=Continue? (y/n):
if /i "%CONTINUE%"=="y" (
    echo.
    goto :query_loop
)

echo.
echo Press any key to close...
pause >nul
exit /b 0
