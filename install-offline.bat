@echo off
REM === Offline AI Offline Package Installer ===
REM NOTE: Japanese text must NOT appear in .bat files (cmd.exe UTF-8 bug).
REM       All Japanese messages are handled by the PowerShell script.

set "PS_CMD="
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    set "PS_CMD=pwsh"
) else (
    where powershell.exe >nul 2>&1
    if %errorlevel% equ 0 (
        set "PS_CMD=powershell.exe"
    )
)

if "%PS_CMD%"=="" (
    echo [ERROR] PowerShell not found. Windows 11 x64 is required.
    pause
    exit /b 1
)

set "SCRIPT_PATH="
set "PACKAGE_ROOT="
if exist "%~dp0_internal\install-offline.ps1" (
    set "SCRIPT_PATH=%~dp0_internal\install-offline.ps1"
    set "PACKAGE_ROOT=%~dp0"
) else (
    if exist "%~dp0install-offline.ps1" (
        set "SCRIPT_PATH=%~dp0install-offline.ps1"
        set "PACKAGE_ROOT=%~dp0.."
    )
)

if "%SCRIPT_PATH%"=="" (
    echo [ERROR] install-offline.ps1 not found.
    pause
    exit /b 1
)

%PS_CMD% -ExecutionPolicy Bypass -NoProfile -File "%SCRIPT_PATH%" -PackageRoot "%PACKAGE_ROOT%"
set "SETUP_EXIT=%errorlevel%"

if %SETUP_EXIT% neq 0 (
    echo.
    echo [ERROR] Offline install failed with exit code: %SETUP_EXIT%
)

echo.
echo Press any key to close...
pause >nul
exit /b %SETUP_EXIT%
