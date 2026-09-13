@echo off
setlocal
REM === Offline AI Embedding Index Launcher ===
REM Index construction is explicit and never started by search.bat or web.bat.

set "PY="
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PY set "PY=%%i"
)
if not defined PY (
    echo [ERROR] Python not found. Install offline-ai first.
    exit /b 2
)
"%PY%" -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 or later is required.
    exit /b 2
)

"%PY%" "%~dp0_internal\index_cli.py" %*
exit /b %ERRORLEVEL%
