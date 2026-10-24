<#
.SYNOPSIS
    offline-ai 完全オフライン導入パッケージを作成する。

.DESCRIPTION
    オンラインPCで事前取得済みの Ollama モデルと、手動配置済みのインストーラーを
    1つの搬送用フォルダへまとめる。外部ファイルの自動ダウンロードは行わない。
#>
[CmdletBinding()]
param(
    [string]$OutputPath = "",
    [string]$InstallersPath = "",
    [string]$OllamaModelsPath = "",
    [switch]$SkipModelCheck,
    [switch]$IncludeAllLocalModels,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$CommonModulePath = Join-Path $PSScriptRoot "OfflineAi.Common.psm1"
Import-Module $CommonModulePath -Force

$DefaultChatModel = "qwen3.5:9b"
$DefaultEmbedModel = "bge-m3"
$LocalBuildCreatedBy = "offline-ai/build-offline-package.ps1"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
}

function Find-OllamaModelsPath {
    param([string]$ExplicitPath)
    try {
        return Get-OfflineAiOllamaModelsPath -ExplicitPath $ExplicitPath
    } catch {
        throw "Ollama モデル保存先を検出できません。-OllamaModelsPath を指定してください。"
    }
}

function Assert-ModelPulled {
    param(
        [Parameter(Mandatory = $true)][string[]]$ModelNames,
        [Parameter(Mandatory = $true)][string]$Model
    )
    if ($ModelNames -contains $Model -or $ModelNames -contains "${Model}:latest") {
        Write-Host "  モデル確認 OK: $Model" -ForegroundColor Green
        return
    }
    throw "モデルが取得済みではありません: $Model。オンラインPCで事前取得を完了してから再実行してください。"
}

function Assert-RequiredDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "必要ディレクトリが見つかりません: $Path"
    }
}

function Assert-BuildPackageOutputRoot {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )
    $resolvedPackageRoot = Resolve-OfflineAiFullPath -Path $PackageRoot
    $resolvedAppRoot = Resolve-OfflineAiFullPath -Path $AppRoot
    $resolvedOutputRoot = Assert-OfflineAiPathUnderRoot -Root $resolvedPackageRoot -Target $OutputRoot
    if ($resolvedOutputRoot -eq $resolvedPackageRoot) {
        throw "OutputPath に offline-package 直下そのものは指定できません: $resolvedOutputRoot"
    }
    if ($resolvedOutputRoot -eq $resolvedAppRoot) {
        throw "OutputPath に app root は指定できません: $resolvedOutputRoot"
    }
    $qualifier = Split-Path -Qualifier $resolvedOutputRoot
    if ($qualifier -and $resolvedOutputRoot.TrimEnd('\', '/') -eq $qualifier.TrimEnd('\', '/')) {
        throw "OutputPath にドライブ直下は指定できません: $resolvedOutputRoot"
    }
    return $resolvedOutputRoot
}

function Test-LocalBuildPackageRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )
    $resolvedPath = Assert-OfflineAiPathUnderRoot -Root $PackageRoot -Target $Path
    $manifestPath = Join-Path $resolvedPath "manifest.json"
    $appPath = Join-Path $resolvedPath "app\offline-ai"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $appPath -PathType Container)) { return $false }

    $manifest = $null
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $false
    }
    $kindProperty = $manifest.PSObject.Properties['packageKind']
    if ($kindProperty) {
        $packageKind = [string]$kindProperty.Value
        if ($packageKind -ne "local-build") { return $false }
        $createdByProperty = $manifest.PSObject.Properties['createdBy']
        $createdAtProperty = $manifest.PSObject.Properties['createdAt']
        return (
            $createdByProperty -and
            [string]$createdByProperty.Value -eq $LocalBuildCreatedBy -and
            $createdAtProperty -and
            -not [string]::IsNullOrWhiteSpace([string]$createdAtProperty.Value)
        )
    }

    # packageKind プロパティ自体がない旧build成果物だけを互換判定する。
    $legacyRequired = @("installers", "ollama-models", "ollama-models\blobs", "ollama-models\manifests")
    foreach ($relative in $legacyRequired) {
        if (-not (Test-Path -LiteralPath (Join-Path $resolvedPath $relative) -PathType Container)) {
            return $false
        }
    }
    return $true
}

function Find-Installer {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$Patterns
    )
    foreach ($pattern in $Patterns) {
        $hit = Get-ChildItem -LiteralPath $Root -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Get-OllamaManifestRelativePath {
    param([Parameter(Mandatory = $true)][string]$Model)
    $name = $Model
    $tag = "latest"
    if ($Model.Contains(":")) {
        $parts = $Model -split ":", 2
        $name = $parts[0]
        $tag = $parts[1]
    }
    $repoParts = @($name -split "/")
    if ($repoParts.Count -eq 1) {
        $repoParts = @("library") + $repoParts
    }
    return Join-Path (Join-Path "registry.ollama.ai" ($repoParts -join "\")) $tag
}

function Get-OllamaManifestBlobDigests {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $digests = @()
    if ($manifest.config -and $manifest.config.digest) {
        $digests += [string]$manifest.config.digest
    }
    foreach ($layer in @($manifest.layers)) {
        if ($layer.digest) {
            $digests += [string]$layer.digest
        }
    }
    return @($digests | Where-Object { $_ -match '^sha256:[0-9a-fA-F]{64}$' } | Select-Object -Unique)
}

function Copy-TargetOllamaModels {
    param(
        [Parameter(Mandatory = $true)][string]$SourceModelsRoot,
        [Parameter(Mandatory = $true)][string]$DestinationModelsRoot,
        [Parameter(Mandatory = $true)][string[]]$Models
    )
    foreach ($model in $Models) {
        $relativeManifest = Get-OllamaManifestRelativePath -Model $model
        $sourceManifest = Join-Path (Join-Path $SourceModelsRoot "manifests") $relativeManifest
        if (-not (Test-Path -LiteralPath $sourceManifest -PathType Leaf)) {
            throw "対象モデル manifest が見つかりません: $model ($sourceManifest)"
        }
        $destManifest = Join-Path (Join-Path $DestinationModelsRoot "manifests") $relativeManifest
        $destManifestDir = Split-Path -Parent $destManifest
        if (-not (Test-Path -LiteralPath $destManifestDir -PathType Container)) {
            New-Item -Path $destManifestDir -ItemType Directory -Force | Out-Null
        }
        Copy-Item -LiteralPath $sourceManifest -Destination $destManifest

        foreach ($digest in @(Get-OllamaManifestBlobDigests -ManifestPath $sourceManifest)) {
            $blobName = $digest.Replace(":", "-")
            $sourceBlob = Join-Path (Join-Path $SourceModelsRoot "blobs") $blobName
            if (-not (Test-Path -LiteralPath $sourceBlob -PathType Leaf)) {
                throw "対象モデル blob が見つかりません: $model ($blobName)"
            }
            $destBlobRoot = Join-Path $DestinationModelsRoot "blobs"
            if (-not (Test-Path -LiteralPath $destBlobRoot -PathType Container)) {
                New-Item -Path $destBlobRoot -ItemType Directory -Force | Out-Null
            }
            Copy-Item -LiteralPath $sourceBlob -Destination (Join-Path $destBlobRoot $blobName)
        }
    }
}

function Get-OllamaVersion {
    $ollama = Find-OfflineAiCommandPath -Name "ollama"
    if (-not $ollama) { return "" }
    try {
        return ((& $ollama --version 2>&1) | Out-String).Trim()
    } catch {
        return ""
    }
}

$appRoot = Resolve-OfflineAiFullPath -Path (Join-Path $PSScriptRoot "..")
$packageRoot = Join-Path $appRoot "offline-package"
if (-not $OutputPath) {
    $OutputPath = Join-Path $packageRoot "offline-ai-offline-package"
}
if (-not $InstallersPath) {
    $InstallersPath = Join-Path $packageRoot "installers"
}

$outputRoot = Assert-BuildPackageOutputRoot -PackageRoot $packageRoot -AppRoot $appRoot -OutputRoot $OutputPath
$installersRoot = Assert-OfflineAiPathUnderRoot -Root $packageRoot -Target $InstallersPath

try {
    Write-Step "1/5 事前検証"
    Write-Host "  OutputPath: $outputRoot"
    Write-Host "  InstallersPath: $installersRoot"
    Assert-RequiredDirectory -Path $installersRoot

    if (-not $SkipModelCheck) {
        $names = @(Get-OfflineAiOllamaModelNames)
        Assert-ModelPulled -ModelNames $names -Model $DefaultChatModel
        Assert-ModelPulled -ModelNames $names -Model $DefaultEmbedModel
    } else {
        Write-Host "  [警告] ollama list によるモデル取得済み確認をスキップします。" -ForegroundColor Yellow
    }

    $modelRoot = Find-OllamaModelsPath -ExplicitPath $OllamaModelsPath
    Assert-RequiredDirectory -Path (Join-Path $modelRoot "blobs")
    Assert-RequiredDirectory -Path (Join-Path $modelRoot "manifests")

    $ollamaInstaller = Find-Installer -Root $installersRoot -Patterns @("OllamaSetup.exe")
    $pythonInstaller = Find-Installer -Root $installersRoot -Patterns @("python-*-amd64.exe")
    $psInstaller = Find-Installer -Root $installersRoot -Patterns @("PowerShell-*-win-x64.msi")

    if (-not $ollamaInstaller) { throw "OllamaSetup.exe を installers に配置してください。" }
    if (-not $pythonInstaller) { throw "python-*-amd64.exe を installers に配置してください。" }
    if (-not $psInstaller) { Write-Host "  [情報] PowerShell 7 MSI は任意です。未同梱のまま続行します。" -ForegroundColor Yellow }
    Write-Step "2/5 出力フォルダ初期化"
    if (Test-Path -LiteralPath $outputRoot) {
        if (-not $Force) {
            throw "出力先が既に存在します。削除してよい場合のみ -Force を指定してください: $outputRoot"
        }
        if (-not (Test-LocalBuildPackageRoot -Path $outputRoot -PackageRoot $packageRoot)) {
            throw "安全確認に失敗したため -Force 削除を拒否します: $outputRoot"
        }
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }
    New-Item -Path $outputRoot -ItemType Directory -Force | Out-Null

    Write-Step "3/5 ファイル収集"
    Copy-OfflineAiDirectoryExact -SourceRoot $installersRoot -DestinationRoot (Join-Path $outputRoot "installers")
    if ($IncludeAllLocalModels) {
        Copy-OfflineAiDirectoryExact -SourceRoot (Join-Path $modelRoot "blobs") -DestinationRoot (Join-Path $outputRoot "ollama-models\blobs")
        Copy-OfflineAiDirectoryExact -SourceRoot (Join-Path $modelRoot "manifests") -DestinationRoot (Join-Path $outputRoot "ollama-models\manifests")
    } else {
        Copy-TargetOllamaModels -SourceModelsRoot $modelRoot -DestinationModelsRoot (Join-Path $outputRoot "ollama-models") -Models @($DefaultChatModel, $DefaultEmbedModel)
    }
    $distributionPolicy = Import-OfflineAiDistributionPolicy -PolicyPath (Join-Path $appRoot "_internal\distribution-policy.json")
    Copy-OfflineAiDirectoryByDistribution `
        -SourceRoot $appRoot `
        -DestinationRoot (Join-Path $outputRoot "app\offline-ai") `
        -Policy $distributionPolicy `
        -IncludeDistributions @("both", "runtime")

    $scriptsDir = Join-Path $outputRoot "scripts"
    New-Item -Path $scriptsDir -ItemType Directory -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $appRoot "install-offline.bat") -Destination (Join-Path $scriptsDir "install-offline.bat")
    Copy-Item -LiteralPath (Join-Path $appRoot "_internal\install-offline.ps1") -Destination (Join-Path $scriptsDir "install-offline.ps1")
    Copy-Item -LiteralPath (Join-Path $appRoot "_internal\OfflineAi.Common.psm1") -Destination (Join-Path $scriptsDir "OfflineAi.Common.psm1")

    Write-Step "4/5 manifest 作成"
    $installerInventory = @()
    foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $outputRoot "installers") -File -Recurse | Sort-Object FullName)) {
        $installerInventory += [PSCustomObject]@{
            relativePath = Get-OfflineAiRelativePath -BasePath $outputRoot -Path $file.FullName
            sizeBytes = $file.Length
            sha256 = Get-OfflineAiSha256 -Path $file.FullName
        }
    }

    $modelSourceKind = "default-user-profile"
    if ($OllamaModelsPath) {
        $modelSourceKind = "explicit"
    } elseif ($env:OLLAMA_MODELS) {
        $modelSourceKind = "OLLAMA_MODELS"
    }

    $modelLicenseInventory = @(Export-OfflineAiModelLicenses `
        -ModelsRoot (Join-Path $outputRoot "ollama-models") `
        -DestinationRoot (Join-Path $outputRoot "legal\models"))

    $manifest = [PSCustomObject]@{
        schemaVersion = 1
        packageKind = "local-build"
        productVersion = ([string](Get-Content -LiteralPath (Join-Path $appRoot "VERSION") -Raw -Encoding UTF8)).Trim()
        createdBy = $LocalBuildCreatedBy
        createdAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        os = Get-OfflineAiOsSummary
        ollamaVersion = Get-OllamaVersion
        models = @($DefaultChatModel, $DefaultEmbedModel)
        modelLicenses = $modelLicenseInventory
        ollamaModelsSource = [PSCustomObject]@{
            kind = $modelSourceKind
            contains = @("blobs", "manifests")
        }
        install = [PSCustomObject]@{
            defaultAppRoot = "app\offline-ai"
            defaultOllamaModels = "%USERPROFILE%\.ollama\models"
            modelConfigModeDefault = "OverwriteDefault"
        }
        verifiedSilentArgs = [PSCustomObject]@{
            python = @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0")
            powershell7 = @("/i", "<msi>", "/qn", "/norestart")
            ollama = @("/SILENT")
        }
        installers = $installerInventory
        notes = @(
            "モデル本体とインストーラーの再配布条件は一次配布元で別途確認してください。",
            "manifest にはユーザー名や環境変数全体を記録しません。"
        )
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRoot "manifest.json") -Encoding UTF8

    Write-Step "5/5 checksum 作成"
    $inventory = @(New-OfflineAiFileInventory -Root $outputRoot)
    Write-OfflineAiChecksums -Root $outputRoot -Inventory $inventory

    Write-Host ""
    Write-Host "完全オフライン導入パッケージを作成しました。" -ForegroundColor Green
    Write-Host "出力先: $outputRoot"
    exit 0
} catch {
    Write-Host ""
    Write-Host "[エラー] パッケージ作成に失敗しました。" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
