@echo off
REM === Offline AI Download-Only Package Launcher ===
REM NOTE: Japanese text must NOT appear in .bat files (cmd.exe UTF-8 bug).
REM       All Japanese messages are handled by the PowerShell script.
REM
REM Role: Delegates to _internal/download-offline-package.ps1 to collect files only.
REM This launcher does not install Python, Ollama, Git, PowerShell, or models.
REM See _internal/ARCHITECTURE.md for design details.

set "PS_CMD="
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    set "PS_CMD=pwsh"
) else (
    where powershell >nul 2>&1
    if %errorlevel% equ 0 (
        set "PS_CMD=powershell.exe"
    )
)

if "%PS_CMD%"=="" (
    echo [ERROR] PowerShell not found. Please use Windows 11 x64.
    pause
    exit /b 1
)

%PS_CMD% -ExecutionPolicy Bypass -NoProfile -File "%~dp0_internal\download-offline-package.ps1" %*
set "SETUP_EXIT=%errorlevel%"

if %SETUP_EXIT% neq 0 (
    echo.
    echo [ERROR] Download-only package creation failed with exit code: %SETUP_EXIT%
)

echo.
echo Press any key to close...
pause >nul
exit /b %SETUP_EXIT%
