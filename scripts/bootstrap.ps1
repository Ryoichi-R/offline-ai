#requires -Version 7.4
<#
.SYNOPSIS
    offline-ai の standalone開発環境（Python / pytest / Pester）を検証・準備する。

.DESCRIPTION
    親workspaceの scripts/bootstrap.ps1 には依存しない。offline-ai を独立repositoryとして
    checkoutした場合でも、このscriptだけでtest実行環境を再現できることを目的とする。
    Git/GitHub関連の設定は行わない。

.EXAMPLE
    pwsh -NoProfile -File scripts/bootstrap.ps1
    pwsh -NoProfile -File scripts/bootstrap.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$SkipPesterInstall,
    [switch]$SkipPipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$offlineAiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
)

# 再現用固定version（対応rangeはpyproject.tomlのdependency-groupsを参照: pytest>=8,<10）
$RequiredPesterVersion = '5.7.1'
$RequiredPytestVersion = '9.0.2'
$MinPythonVersion = [version]'3.11.0'
$MaxPythonVersionExclusive = [version]'3.13.0'

function Resolve-Application {
    param([Parameter(Mandatory)][string]$Name)
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { throw "Required application is not available: $Name" }
    return $command.Source
}

if ($WhatIfPreference) {
    Write-Host '[WhatIf] offline-ai bootstrap'
    Write-Host "[WhatIf] OfflineAiRoot: $offlineAiRoot"
    Write-Host "[WhatIf] Validate Python >= $MinPythonVersion and < $MaxPythonVersionExclusive"
    if (-not $SkipPipInstall) { Write-Host "[WhatIf] Run pip install pytest==$RequiredPytestVersion" }
    Write-Host "[WhatIf] Validate/install Pester $RequiredPesterVersion"
    Write-Host '[WhatIf] Git and GitHub configuration are intentionally skipped'
    return
}

$python = Resolve-Application -Name 'python'
$pythonVersionText = ((& $python --version) 2>&1 | Select-Object -First 1) -replace '^Python\s+', ''
$pythonVersion = $null
if (-not [version]::TryParse($pythonVersionText.Trim(), [ref]$pythonVersion)) {
    throw "Unable to parse Python version: $pythonVersionText"
}
if ($pythonVersion -lt $MinPythonVersion -or $pythonVersion -ge $MaxPythonVersionExclusive) {
    throw "Python >= $MinPythonVersion and < $MaxPythonVersionExclusive is required. Current: $pythonVersion"
}

if (-not $SkipPipInstall) {
    & $python -m pip install --quiet "pytest==$RequiredPytestVersion"
    if ($LASTEXITCODE -ne 0) { throw "pip install pytest==$RequiredPytestVersion failed with exit code $LASTEXITCODE" }
}
$pytestVersionText = (& $python -m pytest --version 2>&1 | Select-Object -First 1)
if ($pytestVersionText -notmatch [regex]::Escape($RequiredPytestVersion)) {
    throw "pytest $RequiredPytestVersion is required. Re-run without -SkipPipInstall or install it manually. Current: $pytestVersionText"
}

$pester = Get-Module -ListAvailable -Name Pester |
    Where-Object { $_.Version -eq [version]$RequiredPesterVersion } |
    Select-Object -First 1
if (-not $pester -and -not $SkipPesterInstall) {
    Install-Module Pester -RequiredVersion $RequiredPesterVersion -Scope CurrentUser -Force
    $pester = Get-Module -ListAvailable -Name Pester |
        Where-Object { $_.Version -eq [version]$RequiredPesterVersion } |
        Select-Object -First 1
}
if (-not $pester) {
    throw "Pester $RequiredPesterVersion is required. Re-run without -SkipPesterInstall or install it for CurrentUser."
}

Write-Host '[bootstrap] complete'
Write-Host "[bootstrap] offline-ai root: $offlineAiRoot"
Write-Host "[bootstrap] Python: $pythonVersion"
Write-Host "[bootstrap] pytest: $RequiredPytestVersion"
Write-Host "[bootstrap] Pester: $($pester.Version)"
Write-Host '[bootstrap] Git/GitHub: not configured (standalone bootstrap does not manage version control)'
