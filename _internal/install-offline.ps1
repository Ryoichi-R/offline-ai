<#
.SYNOPSIS
    完全オフライン導入パッケージを同梱ファイルだけで展開する。

.DESCRIPTION
    install-offline.bat または offline package の scripts/install-offline.bat から呼び出す。
    外部取得は行わず、manifest/checksum 検証、依存導入、Ollama モデル配置、
    offline-ai の標準モデル設定を書き込む。
#>
[CmdletBinding()]
param(
    [string]$PackageRoot = "",
    [string]$AppInstallPath = "",
    [ValidateSet("OverwriteDefault", "KeepExisting")]
    [string]$ModelConfigMode = "OverwriteDefault",
    [ValidateSet("Fail", "BackupAndReplace")]
    [string]$ModelConflictMode = "Fail",
    [switch]$DryRun,
    [switch]$SkipRunSmokeTest,
    [switch]$SkipEmbeddingSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$CommonModulePath = Join-Path $PSScriptRoot "OfflineAi.Common.psm1"
Import-Module $CommonModulePath -Force

$DefaultChatModel = "qwen3.5:9b"
$DefaultEmbedModel = "bge-m3"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
}

function Write-Plan {
    param([string]$Message)
    if ($DryRun) {
        Write-Host "  [DryRun] $Message" -ForegroundColor DarkCyan
    } else {
        Write-Host "  $Message"
    }
}

function Write-Utf8NoBomText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir -PathType Container)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }
    [System.IO.File]::WriteAllText(
        $Path,
        $Value,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Refresh-PathEnv {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$userPath;$machinePath"
}

function Resolve-PackageRoot {
    param([string]$Value)
    if ($Value) {
        return Resolve-OfflineAiFullPath -Path $Value
    }

    $scriptDir = $PSScriptRoot
    $candidateFromScripts = Resolve-OfflineAiFullPath -Path (Join-Path $scriptDir "..")
    if (Test-Path -LiteralPath (Join-Path $candidateFromScripts "manifest.json")) {
        return $candidateFromScripts
    }

    $appRoot = Resolve-OfflineAiFullPath -Path (Join-Path $PSScriptRoot "..")
    if (Test-Path -LiteralPath (Join-Path $appRoot "manifest.json")) {
        return $appRoot
    }
    return $appRoot
}

function Read-PackageManifest {
    param([Parameter(Mandatory = $true)][string]$Root)
    $manifestPath = Join-Path $Root "manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "manifest.json が見つかりません: $manifestPath"
    }
    return Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Test-Checksums {
    param([Parameter(Mandatory = $true)][string]$Root)
    $null = Test-OfflineAiPackageChecksums -Root $Root
    Write-Host "  checksum 検証: OK" -ForegroundColor Green
}

function Test-Fat32PackageMedia {
    param([Parameter(Mandatory = $true)][string]$Root)
    $isFat32 = Test-OfflineAiFat32PackageMedia -Root $Root
    if ($isFat32) {
        Write-Host "  [警告] 搬送媒体が FAT32 です。4GB 超のモデル blob を扱えません。exFAT または NTFS を使ってください。" -ForegroundColor Yellow
    } elseif ($null -eq $isFat32) {
        Write-Host "  [情報] ファイルシステム種別を確認できませんでした。" -ForegroundColor Yellow
    }
}

function Test-InstallPreflight {
    param(
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [UInt64]$MinimumFreeBytes = 20GB
    )
    if ($DryRun) {
        $result = Test-OfflineAiPreflight -RamBytes 16GB -FreeBytes $MinimumFreeBytes -MinimumFreeBytes $MinimumFreeBytes -DiskPath $DestinationPath
    } else {
        $result = Test-OfflineAiPreflight -MinimumFreeBytes $MinimumFreeBytes -DiskPath $DestinationPath
    }
    Write-Host "  preflight: RAM $($result.RamGB)GB, Disk $($result.FreeGB)GB ($DestinationPath)"
    if (-not $result.Passed) {
        throw "preflight failed: $($result.Errors -join ', ') が最小要件を満たしていません。"
    }
}

function Copy-FileNoConflict {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if ($DryRun) {
        Write-Plan "コピー: $Source -> $Destination"
    }
    $result = Copy-OfflineAiFileNoConflict -Source $Source -Destination $Destination -DryRun:$DryRun
    if ($result -eq "SkippedSameHash") {
        Write-Host "  既存同一ファイルをスキップ: $Destination"
    }
}

function Copy-TreeNoConflict {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )
    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        throw "コピー元ディレクトリが見つかりません: $SourceRoot"
    }
    $files = @(Get-ChildItem -LiteralPath $SourceRoot -Force -File -Recurse)
    foreach ($file in $files) {
        $rel = Get-OfflineAiRelativePath -BasePath $SourceRoot -Path $file.FullName
        $destination = Join-Path $DestinationRoot $rel
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $sourceHash = Get-OfflineAiSha256 -Path $file.FullName
            $destinationHash = Get-OfflineAiSha256 -Path $destination
            if ($sourceHash -ne $destinationHash) {
                throw "Destination file already exists with different content: $destination"
            }
        }
    }
    foreach ($file in $files) {
        $rel = Get-OfflineAiRelativePath -BasePath $SourceRoot -Path $file.FullName
        Copy-FileNoConflict -Source $file.FullName -Destination (Join-Path $DestinationRoot $rel)
    }
}

function Install-ExecutableIfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CommandName,
        [Parameter(Mandatory = $true)][string]$InstallerPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    if (Find-OfflineAiCommandPath -Name $CommandName) {
        Write-Host "  ${Name}: インストール済み" -ForegroundColor Green
        return
    }
    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
        throw "$Name の同梱インストーラーが見つかりません: $InstallerPath"
    }
    if ($DryRun) {
        Write-Plan "$Name インストール: $InstallerPath $($Arguments -join ' ')"
        return
    }
    Write-Host "  $Name を同梱インストーラーからインストールします..."
    $p = Start-Process -FilePath $InstallerPath -ArgumentList $Arguments -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "$Name インストーラーが失敗しました: exit $($p.ExitCode)"
    }
    Refresh-PathEnv
}

function Install-MsiIfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CommandName,
        [Parameter(Mandatory = $true)][string]$MsiPath,
        [Parameter(Mandatory = $true)][string[]]$MsiProperties
    )
    if (Find-OfflineAiCommandPath -Name $CommandName) {
        Write-Host "  ${Name}: インストール済み" -ForegroundColor Green
        return
    }
    if (-not (Test-Path -LiteralPath $MsiPath -PathType Leaf)) {
        Write-Host "  [情報] $Name の MSI が同梱されていないためスキップします: $MsiPath" -ForegroundColor Yellow
        return
    }
    $args = @("/i", $MsiPath, "/qn", "/norestart") + $MsiProperties
    if ($DryRun) {
        Write-Plan "$Name インストール: msiexec $($args -join ' ')"
        return
    }
    $p = Start-Process -FilePath "msiexec.exe" -ArgumentList $args -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "$Name MSI が失敗しました: exit $($p.ExitCode)"
    }
    Refresh-PathEnv
}

function Find-FirstFile {
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

function Resolve-AppRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$InstallPath
    )
    $packageApp = Join-Path $Root "app\offline-ai"
    if (-not (Test-Path -LiteralPath $packageApp -PathType Container)) {
        $sourceApp = Resolve-OfflineAiFullPath -Path (Join-Path $PSScriptRoot "..")
        if (Test-Path -LiteralPath (Join-Path $sourceApp "_internal")) {
            $packageApp = $sourceApp
        } else {
            throw "offline-ai アプリ本体が見つかりません。"
        }
    }
    if (-not $InstallPath) {
        return $packageApp
    }
    $dest = Resolve-OfflineAiFullPath -Path $InstallPath
    return $dest
}

function Get-DirectoryCopyRequirementBytes {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [UInt64]$MinimumBytes
    )
    $files = @(Get-ChildItem -LiteralPath $SourceRoot -File -Recurse -ErrorAction SilentlyContinue)
    [UInt64]$total = 0
    [UInt64]$largest = 0
    foreach ($file in $files) {
        $total += [UInt64]$file.Length
        if ([UInt64]$file.Length -gt $largest) { $largest = [UInt64]$file.Length }
    }
    $required = $total + $largest + 512MB
    if ($required -lt $MinimumBytes) { return [UInt64]$MinimumBytes }
    return [UInt64]$required
}

function Test-InstallVolumeRequirements {
    param([Parameter(Mandatory = $true)][object[]]$Requirements)
    $byVolume = @{}
    foreach ($requirement in $Requirements) {
        $fullPath = Resolve-OfflineAiFullPath -Path ([string]$requirement.Path)
        $volume = [System.IO.Path]::GetPathRoot($fullPath).ToUpperInvariant()
        if (-not $byVolume.ContainsKey($volume)) {
            $byVolume[$volume] = [PSCustomObject]@{ Path = $fullPath; Bytes = [UInt64]0; Labels = @() }
        }
        $byVolume[$volume].Bytes += [UInt64]$requirement.Bytes
        $byVolume[$volume].Labels += [string]$requirement.Label
    }
    foreach ($entry in $byVolume.Values) {
        Test-InstallPreflight -DestinationPath $entry.Path -MinimumFreeBytes $entry.Bytes
        Write-Host "  volume requirement: $($entry.Labels -join ' + ') = $([math]::Round($entry.Bytes / 1GB, 1))GB"
    }
}

function Assert-InstallerIdentities {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object]$Manifest
    )
    if (@($Manifest.PSObject.Properties.Name) -notcontains "downloads") {
        return
    }
    foreach ($download in @($Manifest.downloads)) {
        $names = @($download.PSObject.Properties.Name)
        if ($names -notcontains "expectedSha256" -or -not [string]$download.expectedSha256) { continue }
        $path = Join-Path (Join-Path $Root "installers") ([string]$download.relativeFile)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            if ([string]$download.status -notin @("ManualOnly", "OptionalSkipped")) {
                throw "期待したinstallerが見つかりません: $($download.id)"
            }
            continue
        }
        if ((Get-OfflineAiSha256 -Path $path) -ne ([string]$download.expectedSha256).ToLowerInvariant()) {
            throw "承認済みSHA-256と一致しません: $($download.id)"
        }
        if ($names -contains "expectedProductVersion" -and [string]$download.expectedProductVersion) {
            $version = ([System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)).ProductVersion
            if (-not $version -or -not $version.Trim().StartsWith([string]$download.expectedProductVersion, [StringComparison]::Ordinal)) {
                throw "承認済み製品バージョンと一致しません: $($download.id)"
            }
        }
        if ($names -contains "expectedSignerSubjectContains" -and [string]$download.expectedSignerSubjectContains) {
            $signature = Get-AuthenticodeSignature -LiteralPath $path
            $subject = if ($signature.SignerCertificate) { [string]$signature.SignerCertificate.Subject } else { "" }
            if ($signature.Status -ne "Valid" -or
                $subject.IndexOf([string]$download.expectedSignerSubjectContains, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
                throw "承認済み署名者と一致しません: $($download.id)"
            }
        }
    }
    Write-Host "  installer identity 検証: OK" -ForegroundColor Green
}

function Assert-SupportedArchitecture {
    $architecture = Get-OfflineAiNativeArchitecture
    if ($architecture -notin @("AMD64", "X64")) {
        throw "未対応のCPUアーキテクチャです: $architecture。Windows 11 x64を使用してください。"
    }
}

function Test-IsAdministrator {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

function Initialize-SkillSource {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $sourceRoot = Join-Path $AppRoot "skill-source"
    if ($DryRun) {
        Write-Plan "利用者資料フォルダー作成: $sourceRoot"
        return
    }
    New-Item -Path $sourceRoot -ItemType Directory -Force | Out-Null
    $readme = Join-Path $sourceRoot "README.txt"
    if (-not (Test-Path -LiteralPath $readme -PathType Leaf)) {
        Write-Utf8NoBomText -Path $readme -Value @"
このフォルダーに検索対象の資料を置いてください。
ここに置いた資料は利用者データです。バックアップは利用者が管理してください。
オフラインパッケージの作成時、このフォルダーの内容は常に除外されます。
"@
    }
}

function Write-InstallState {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][object]$State
    )
    if ($DryRun) {
        Write-Plan "install state記録: $AppRoot\_internal\install-state.json"
        return
    }
    $path = Join-Path $AppRoot "_internal\install-state.json"
    $directory = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -Path $directory -ItemType Directory -Force | Out-Null
    }
    $part = "$path.part-$([guid]::NewGuid().ToString('N'))"
    try {
        $json = $State | ConvertTo-Json -Depth 8
        [System.IO.File]::WriteAllText($part, $json, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $part -Destination $path -Force
    } finally {
        if (Test-Path -LiteralPath $part -PathType Leaf) {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
    }
}

function Read-InstallState {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $path = Join-Path $AppRoot "_internal\install-state.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw "既存install-state.jsonを読み取れません。削除せず内容を確認してください: $path"
    }
}

function Get-PreservedComponentDisposition {
    param(
        [object]$PreviousState,
        [Parameter(Mandatory = $true)][string]$Name,
        [bool]$CurrentlyExists,
        [bool]$Included
    )
    if ($PreviousState -and
        @($PreviousState.PSObject.Properties.Name) -contains "components" -and
        @($PreviousState.components.PSObject.Properties.Name) -contains $Name -and
        [string]$PreviousState.components.$Name -eq "Installed") {
        return "Installed"
    }
    if ($CurrentlyExists) { return "Existing" }
    if ($Included) { return "Planned" }
    return "NotIncluded"
}

function Resolve-OllamaModelsPath {
    if ($env:OLLAMA_MODELS) {
        return Resolve-OfflineAiFullPath -Path $env:OLLAMA_MODELS
    }
    if (-not $env:USERPROFILE) {
        throw "USERPROFILE が取得できないため Ollama モデル保存先を決定できません。"
    }
    return Resolve-OfflineAiFullPath -Path (Join-Path $env:USERPROFILE ".ollama\models")
}

function Copy-OllamaModels {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$DestinationModelsPath
    )
    $sourceModels = Join-Path $Root "ollama-models"
    $sourceBlobs = Join-Path $sourceModels "blobs"
    $sourceManifests = Join-Path $sourceModels "manifests"
    if (-not (Test-Path -LiteralPath $sourceBlobs -PathType Container)) {
        throw "ollama-models\blobs が見つかりません。"
    }
    if (-not (Test-Path -LiteralPath $sourceManifests -PathType Container)) {
        throw "ollama-models\manifests が見つかりません。"
    }
    Copy-TreeNoConflict -SourceRoot $sourceBlobs -DestinationRoot (Join-Path $DestinationModelsPath "blobs")
    $manifestDestinationRoot = Join-Path $DestinationModelsPath "manifests"
    if ($ModelConflictMode -eq "Fail") {
        try {
            Copy-TreeNoConflict -SourceRoot $sourceManifests -DestinationRoot $manifestDestinationRoot
        } catch {
            throw "$($_.Exception.Message) 同名モデルtagを置き換える場合は、既存モデルを確認して -ModelConflictMode BackupAndReplace を明示してください。"
        }
        return
    }

    foreach ($file in @(Get-ChildItem -LiteralPath $sourceManifests -Force -File -Recurse)) {
        $rel = Get-OfflineAiRelativePath -BasePath $sourceManifests -Path $file.FullName
        $destination = Join-Path $manifestDestinationRoot $rel
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $sourceHash = Get-OfflineAiSha256 -Path $file.FullName
            $destinationHash = Get-OfflineAiSha256 -Path $destination
            if ($sourceHash -ne $destinationHash) {
                $backup = "$destination.offline-ai-backup-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ'))"
                if ($DryRun) {
                    Write-Plan "既存モデルmanifest backup: $destination -> $backup"
                    Write-Plan "モデルmanifest replace: $($file.FullName) -> $destination"
                } else {
                    $part = "$destination.offline-ai-replace-$([guid]::NewGuid().ToString('N'))"
                    try {
                        $destinationDirectory = Split-Path -Parent $destination
                        if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
                            New-Item -Path $destinationDirectory -ItemType Directory -Force | Out-Null
                        }
                        Copy-Item -LiteralPath $file.FullName -Destination $part
                        if ((Get-OfflineAiSha256 -Path $part) -ne $sourceHash) {
                            throw "Temporary model manifest hash mismatch: $destination"
                        }
                        Move-Item -LiteralPath $destination -Destination $backup
                        try {
                            Move-Item -LiteralPath $part -Destination $destination
                        } catch {
                            if (-not (Test-Path -LiteralPath $destination) -and
                                (Test-Path -LiteralPath $backup -PathType Leaf)) {
                                Move-Item -LiteralPath $backup -Destination $destination
                            }
                            throw
                        }
                        Write-Host "  既存モデルmanifestを退避: $backup" -ForegroundColor Yellow
                    } finally {
                        if (Test-Path -LiteralPath $part -PathType Leaf) {
                            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
                        }
                    }
                }
                continue
            }
        }
        Copy-FileNoConflict -Source $file.FullName -Destination $destination
    }
}

function Install-OptionalComponents {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][object]$Manifest
    )
    $sourceRoot = Join-Path $Root "optional-components"
    $destinationRoot = Join-Path $AppRoot "_internal\optional-components"
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        if (Test-Path -LiteralPath $destinationRoot -PathType Container) {
            $existingDestinationFiles = @(Get-ChildItem -LiteralPath $destinationRoot -Force -File -Recurse)
            if ($existingDestinationFiles.Count -gt 0) {
                throw "optional component source なしで installed optional component file が存在します: $($existingDestinationFiles[0].FullName)"
            }
        }
        Write-Host "  [情報] optional-components は同梱されていません。標準構成で続行します。" -ForegroundColor Yellow
        return
    }
    if (@($Manifest.PSObject.Properties.Name) -notcontains "optionalComponents" -or
        @($Manifest.optionalComponents).Count -eq 0) {
        if (Test-Path -LiteralPath $destinationRoot -PathType Container) {
            $existingDestinationFiles = @(Get-ChildItem -LiteralPath $destinationRoot -Force -File -Recurse)
            if ($existingDestinationFiles.Count -gt 0) {
                throw "manifest 宣言なしで installed optional component file が存在します: $($existingDestinationFiles[0].FullName)"
            }
        }
        Write-Warning "optional-components はありますが manifest に宣言がないため配置しません。標準構成で続行します。"
        return
    }

    $declaredComponentRoots = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $declaredDestinationFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $componentPlans = @()
    foreach ($component in @($Manifest.optionalComponents)) {
        $componentProperties = @($component.PSObject.Properties.Name)
        $componentId = if ($componentProperties -contains "id") { [string]$component.id } else { "unknown" }
        try {
            foreach ($requiredProperty in @("packageRelativePath", "activation", "files")) {
                if ($componentProperties -notcontains $requiredProperty) {
                    throw "manifest optional component に $requiredProperty がありません。"
                }
            }
            $packageRelativePath = ([string]$component.packageRelativePath).Replace("/", "\")
            if ([System.IO.Path]::IsPathRooted($packageRelativePath)) {
                throw "packageRelativePath は相対パスである必要があります。"
            }
            $segments = @($packageRelativePath.Split([System.IO.Path]::DirectorySeparatorChar))
            if ($segments.Count -lt 2 -or
                @($segments | Where-Object { -not $_ -or $_ -in @(".", "..") }).Count -gt 0 -or
                $segments[0] -ne "optional-components") {
                throw "packageRelativePath は optional-components 配下である必要があります。"
            }
            $componentSource = Assert-OfflineAiPathUnderRoot `
                -Root $sourceRoot `
                -Target (Join-Path $Root $packageRelativePath)
            if (-not (Test-Path -LiteralPath $componentSource -PathType Container)) {
                throw "宣言された component directory がありません: $packageRelativePath"
            }
            $treeItems = @(
                Get-Item -LiteralPath $componentSource -Force
                Get-ChildItem -LiteralPath $componentSource -Force -Recurse
            )
            foreach ($item in $treeItems) {
                if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "reparse point は配置できません: $($item.FullName)"
                }
            }

            $fileEntries = @($component.files)
            if ($fileEntries.Count -eq 0) {
                throw "manifest optional component の files が空です。"
            }
            $declaredFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
            $componentDestinationFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
            $copyPlan = @()
            foreach ($fileEntry in $fileEntries) {
                $fileProperties = @($fileEntry.PSObject.Properties.Name)
                if ($fileProperties -notcontains "relativePath") {
                    throw "manifest optional component file に relativePath がありません。"
                }
                $relativePath = ([string]$fileEntry.relativePath).Replace("/", "\")
                if ([System.IO.Path]::IsPathRooted($relativePath)) {
                    throw "optional component file path は相対パスである必要があります: $relativePath"
                }
                $fileSegments = @($relativePath.Split([System.IO.Path]::DirectorySeparatorChar))
                if (@($fileSegments | Where-Object { -not $_ -or $_ -in @(".", "..") }).Count -gt 0) {
                    throw "optional component file path が不正です: $relativePath"
                }
                $source = Assert-OfflineAiPathUnderRoot -Root $componentSource -Target (Join-Path $Root $relativePath)
                if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
                    throw "宣言された optional component file がありません: $relativePath"
                }
                $canonicalRelativePath = $relativePath.Replace("\", "/")
                if (-not $declaredFiles.Add($canonicalRelativePath)) {
                    throw "optional component file が重複しています: $relativePath"
                }
                if ($fileProperties -contains "sha256" -and [string]$fileEntry.sha256) {
                    if ((Get-OfflineAiSha256 -Path $source) -ne ([string]$fileEntry.sha256).ToLowerInvariant()) {
                        throw "optional component manifest hash mismatch: $relativePath"
                    }
                }
                $relativeWithinComponent = Get-OfflineAiRelativePath -BasePath $componentSource -Path $source
                $destination = Assert-OfflineAiPathUnderRoot `
                    -Root $destinationRoot `
                    -Target (Join-Path (Join-Path $destinationRoot (Get-OfflineAiRelativePath -BasePath $sourceRoot -Path $componentSource)) $relativeWithinComponent)
                $copyPlan += [PSCustomObject]@{
                    source = $source
                    destination = $destination
                    isActivationFile = ($relativeWithinComponent -eq "command.txt")
                }
                $null = $componentDestinationFiles.Add($destination)
            }

            $actualFiles = @(Get-ChildItem -LiteralPath $componentSource -Force -File -Recurse)
            foreach ($actualFile in $actualFiles) {
                $actualRelative = Get-OfflineAiRelativePath -BasePath $Root -Path $actualFile.FullName
                if (-not $declaredFiles.Contains($actualRelative)) {
                    throw "manifest に未宣言の optional component file があります: $actualRelative"
                }
            }
            if ($actualFiles.Count -ne $declaredFiles.Count) {
                throw "optional component file inventory count mismatch"
            }

            $activation = [string]$component.activation
            if ($activation -ne "transport-only") {
                throw "未対応の optional component activation です: $activation"
            }
            foreach ($item in $copyPlan) {
                if (Test-Path -LiteralPath $item.destination -PathType Leaf) {
                    if ((Get-OfflineAiSha256 -Path $item.source) -ne (Get-OfflineAiSha256 -Path $item.destination)) {
                        throw "Destination file already exists with different content: $($item.destination)"
                    }
                }
            }
            $null = $declaredComponentRoots.Add($componentSource)
            foreach ($destinationFile in $componentDestinationFiles) {
                $null = $declaredDestinationFiles.Add($destinationFile)
            }
            $componentPlans += [PSCustomObject]@{
                id = $componentId
                activation = $activation
                files = $copyPlan
            }
        } catch {
            Write-Warning "optional component '$componentId' の配置をスキップします。標準構成で続行します: $($_.Exception.Message)"
        }
    }

    foreach ($directory in @(Get-ChildItem -LiteralPath $sourceRoot -Force -Directory)) {
        if (-not $declaredComponentRoots.Contains($directory.FullName)) {
            Write-Warning "manifest に未宣言の optional component directory は配置しません: $($directory.Name)"
        }
    }
    if (Test-Path -LiteralPath $destinationRoot -PathType Container) {
        foreach ($file in @(Get-ChildItem -LiteralPath $destinationRoot -Force -File -Recurse)) {
            if (-not $declaredDestinationFiles.Contains($file.FullName)) {
                throw "manifest に未宣言の installed optional component file があります: $($file.FullName)"
            }
        }
    }
    foreach ($plan in $componentPlans) {
        try {
            Write-Host "  optional component を配置します: $($plan.id)"
            foreach ($item in @($plan.files | Sort-Object isActivationFile)) {
                Copy-FileNoConflict -Source $item.source -Destination $item.destination
            }
            Write-Host "  [情報] reranker は搬送のみです。loopback backend は利用できますが、runtime/model は自動起動・自動有効化しません。" -ForegroundColor Yellow
        } catch {
            Write-Warning "optional component '$($plan.id)' の配置をスキップします。標準構成で続行します: $($_.Exception.Message)"
        }
    }
}

function Start-OllamaIfNeeded {
    param([Parameter(Mandatory = $true)][string]$OllamaExe)
    try {
        $null = Get-OfflineAiOllamaModelNames -OllamaExe $OllamaExe
        return
    } catch {
        Write-Host "  Ollama を起動します..."
    }
    if (-not $DryRun) {
        Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    }
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        try {
            $null = Get-OfflineAiOllamaModelNames -OllamaExe $OllamaExe
            return
        } catch {
        }
    }
    throw "Ollama の起動確認がタイムアウトしました。"
}

function Assert-ModelExists {
    param(
        [Parameter(Mandatory = $true)][string[]]$ModelNames,
        [Parameter(Mandatory = $true)][string]$ExpectedModel
    )
    if ($ModelNames -contains $ExpectedModel -or $ModelNames -contains "${ExpectedModel}:latest") {
        Write-Host "  モデル確認 OK: $ExpectedModel" -ForegroundColor Green
        return
    }
    throw "Ollama にモデルが認識されていません: $ExpectedModel"
}

function Write-ModelFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Mode
    )
    $existing = $null
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $existing = ([string](Get-Content -LiteralPath $Path -Raw -Encoding UTF8)).Trim()
    }
    if ($existing -and $existing -ne $Value -and $Mode -eq "KeepExisting") {
        Write-Host "  既存設定を維持します: $Path = $existing" -ForegroundColor Yellow
        return
    }
    if ($existing -and $existing -ne $Value) {
        Write-Host "  既存設定を標準値で更新します: $existing -> $Value" -ForegroundColor Yellow
    }
    if ($DryRun) {
        Write-Plan "モデル設定書き込み: $Path = $Value"
    } else {
        Write-Utf8NoBomText -Path $Path -Value $Value
    }
}

function Invoke-RunSmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$OllamaExe,
        [Parameter(Mandatory = $true)][string]$Model
    )
    if ($SkipRunSmokeTest) {
        Write-Host "  ollama run テスト: スキップ" -ForegroundColor Yellow
        return
    }
    if ($DryRun) {
        Write-Plan "ollama run $Model hello"
        return
    }
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $OllamaExe run $Model "hello" 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0 -or -not $output) {
        throw "ollama run テストに失敗しました: $output"
    }
    Write-Host "  ollama run テスト: OK" -ForegroundColor Green
}

function Invoke-EmbeddingSmokeTest {
    param([Parameter(Mandatory = $true)][string]$Model)
    if ($SkipEmbeddingSmokeTest) {
        Write-Host "  Embedding API テスト: スキップ" -ForegroundColor Yellow
        return
    }
    if ($DryRun) {
        Write-Plan "localhost Embedding API テスト: $Model"
        return
    }
    $body = @{
        model = $Model
        prompt = "hello"
    } | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $request = [System.Net.WebRequest]::Create("http://localhost:11434/api/embeddings")
    $request.Method = "POST"
    $request.ContentType = "application/json"
    $request.ContentLength = $bytes.Length
    $request.Timeout = 60000
    $stream = $request.GetRequestStream()
    try {
        $stream.Write($bytes, 0, $bytes.Length)
    } finally {
        $stream.Close()
    }
    $response = $request.GetResponse()
    try {
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        $json = $reader.ReadToEnd() | ConvertFrom-Json
        if (-not $json.embedding) {
            throw "embedding フィールドが空です。"
        }
    } finally {
        $response.Close()
    }
    Write-Host "  Embedding API テスト: OK" -ForegroundColor Green
}

function ConvertTo-OfflineAiPlacementObservation {
    param(
        [Parameter(Mandatory = $true)][object]$PsResponse,
        [Parameter(Mandatory = $true)][string]$Model
    )

    $matches = @($PsResponse.models | Where-Object {
        $properties = @($_.PSObject.Properties.Name)
        $name = if ($properties -contains 'name') { [string]$_.name } elseif ($properties -contains 'model') { [string]$_.model } else { '' }
        $name -eq $Model -or $name -eq "${Model}:latest"
    })
    if ($matches.Count -eq 0) {
        return [PSCustomObject]@{
            status = 'ModelNotLoaded'
            model = $Model
            processor = 'unknown'
            sizeBytes = 0
            sizeVramBytes = 0
            gpuPercent = $null
            contextLength = $null
        }
    }
    $entry = $matches[0]
    $properties = @($entry.PSObject.Properties.Name)
    [Int64]$sizeBytes = if ($properties -contains 'size') { [Int64]$entry.size } else { 0 }
    [Int64]$sizeVramBytes = if ($properties -contains 'size_vram') { [Int64]$entry.size_vram } else { 0 }
    $gpuPercent = if ($sizeBytes -gt 0) { [math]::Round(100 * [decimal]$sizeVramBytes / [decimal]$sizeBytes, 1) } else { $null }
    $processor = if ($sizeBytes -gt 0 -and $sizeVramBytes -ge $sizeBytes) {
        'full-gpu'
    } elseif ($sizeVramBytes -gt 0) {
        'partial-gpu'
    } else {
        'cpu'
    }
    return [PSCustomObject]@{
        status = 'Observed'
        model = $Model
        processor = $processor
        sizeBytes = $sizeBytes
        sizeVramBytes = $sizeVramBytes
        gpuPercent = $gpuPercent
        contextLength = if ($properties -contains 'context_length') { [Int64]$entry.context_length } else { $null }
    }
}

function Get-OfflineAiPlacementObservation {
    param([Parameter(Mandatory = $true)][string]$Model)

    if ($DryRun) {
        return [PSCustomObject]@{
            status = 'Planned'
            model = $Model
            processor = 'unknown'
            sizeBytes = 0
            sizeVramBytes = 0
            gpuPercent = $null
            contextLength = $null
        }
    }
    try {
        $request = [System.Net.WebRequest]::Create('http://localhost:11434/api/ps')
        $request.Method = 'GET'
        $request.Timeout = 10000
        $webResponse = $request.GetResponse()
        try {
            $reader = [IO.StreamReader]::new($webResponse.GetResponseStream())
            try {
                $response = $reader.ReadToEnd() | ConvertFrom-Json
            } finally {
                $reader.Dispose()
            }
        } finally {
            $webResponse.Dispose()
        }
        return ConvertTo-OfflineAiPlacementObservation -PsResponse $response -Model $Model
    } catch {
        return [PSCustomObject]@{
            status = 'Unavailable'
            model = $Model
            processor = 'unknown'
            sizeBytes = 0
            sizeVramBytes = 0
            gpuPercent = $null
            contextLength = $null
        }
    }
}

function Write-OfflineAiPlacementObservation {
    param([Parameter(Mandatory = $true)][object]$Observation)

    switch ([string]$Observation.status) {
        'Observed' {
            $context = if ($null -ne $Observation.contextLength) { ", context=$($Observation.contextLength)" } else { '' }
            Write-Host "  GPU配置確認: $($Observation.processor), GPU $($Observation.gpuPercent)%${context}" -ForegroundColor Green
        }
        'ModelNotLoaded' { Write-Host '  [情報] GPU配置確認: 対象モデルは現在loadされていません。ollama psで再確認してください。' -ForegroundColor Yellow }
        'Planned' { Write-Plan 'Ollama /api/ps でPROCESSOR相当のGPU配置を確認' }
        default { Write-Host '  [警告] GPU配置確認を取得できませんでした。導入後に ollama ps で確認してください。' -ForegroundColor Yellow }
    }
}

function Get-ModelConfigSnapshot {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [PSCustomObject]@{ state = 'missing'; value = $null }
    }
    $value = ([string](Get-Content -LiteralPath $Path -Raw -Encoding UTF8)).Trim()
    if (-not $value) { return [PSCustomObject]@{ state = 'empty'; value = $null } }
    return [PSCustomObject]@{ state = 'value'; value = $value }
}

function Get-SchemaOneModelNames {
    param([Parameter(Mandatory = $true)][object]$Manifest)
    $entries = @($Manifest.models)
    if ($entries.Count -eq 0) { throw 'schemaVersion 1 package models inventory is empty.' }
    $stringEntries = @($entries | Where-Object { $_ -is [string] })
    $objectEntries = @($entries | Where-Object { $_ -isnot [string] })
    if ($stringEntries.Count -gt 0 -and $objectEntries.Count -gt 0) {
        throw 'schemaVersion 1 package models inventory mixes string and object entries.'
    }
    $names = @(if ($stringEntries.Count -gt 0) {
        $stringEntries | ForEach-Object { ([string]$_).Trim() }
    } else {
        $objectEntries | ForEach-Object {
            if (@($_.PSObject.Properties.Name) -notcontains 'name') { throw 'schemaVersion 1 model object is missing name.' }
            ([string]$_.name).Trim()
        }
    })
    if (@($names | Where-Object { -not $_ }).Count -gt 0) { throw 'schemaVersion 1 package contains an empty model name.' }
    if (@($names | Sort-Object -Unique).Count -ne $names.Count) { throw 'schemaVersion 1 package contains duplicate model names.' }
    if (@($names | Where-Object { $_ -eq $DefaultChatModel }).Count -ne 1 -or
        @($names | Where-Object { $_ -eq $DefaultEmbedModel }).Count -ne 1) {
        throw "schemaVersion 1 package must contain $DefaultChatModel and $DefaultEmbedModel exactly once."
    }
    return $names
}

function Resolve-PackageModelContract {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object]$Manifest
    )
    $properties = @($Manifest.PSObject.Properties.Name)
    $kind = if ($properties -contains 'packageKind') { [string]$Manifest.packageKind } else { '' }
    $installable = if ($properties -contains 'installable') { [bool]$Manifest.installable } else { $true }
    if ($kind -in @('public-source', 'download-only-no-models') -or -not $installable) {
        throw "この成果物はインストールできません: packageKind=$kind installable=$installable"
    }
    if ([int]$Manifest.schemaVersion -eq 1) {
        $null = Get-SchemaOneModelNames -Manifest $Manifest
        return [PSCustomObject]@{
            chatModel = $DefaultChatModel
            embeddingModel = $DefaultEmbedModel
            unsupportedOverride = [PSCustomObject]@{ used = $false; reasonCodes = @(); confirmationMode = 'not-required' }
        }
    }
    if ([int]$Manifest.schemaVersion -ne 2) { throw "未対応の manifest schemaVersion です: $($Manifest.schemaVersion)" }
    foreach ($field in @('selection', 'models')) {
        if ($properties -notcontains $field) { throw "schemaVersion 2 package is missing $field." }
    }
    $modulePath = Join-Path $Root 'scripts\OfflineAi.ModelSelection.psm1'
    $catalogPath = Join-Path $Root 'app\offline-ai\_internal\download-manifest.json'
    if (-not (Test-Path -LiteralPath $modulePath -PathType Leaf)) { throw 'schemaVersion 2 package is missing OfflineAi.ModelSelection.psm1.' }
    if (-not (Test-Path -LiteralPath $catalogPath -PathType Leaf)) { throw 'schemaVersion 2 package is missing its model catalog snapshot.' }
    Import-Module $modulePath -Force
    $catalog = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $validated = Assert-OfflineAiModelCatalog -Catalog $catalog
    $selectionProperties = @($Manifest.selection.PSObject.Properties.Name)
    foreach ($field in @('chatModel', 'embeddingModel', 'unsupportedOverride')) {
        if ($selectionProperties -notcontains $field) { throw "schemaVersion 2 selection is missing $field." }
    }
    $chatModel = [string]$Manifest.selection.chatModel
    $embeddingModel = [string]$Manifest.selection.embeddingModel
    if (@($validated.selectableChats | Where-Object { $_.name -eq $chatModel }).Count -ne 1) {
        throw "package selected chat model is not selectable in catalog: $chatModel"
    }
    if ($embeddingModel -ne [string]$validated.requiredEmbedding.name) {
        throw "package embedding model does not match catalog required embedding: $embeddingModel"
    }
    $modelEntries = @($Manifest.models)
    if (@($modelEntries | Where-Object { $_ -is [string] }).Count -gt 0) { throw 'schemaVersion 2 models must use object entries.' }
    $modelNames = @($modelEntries | ForEach-Object {
        if (@($_.PSObject.Properties.Name) -notcontains 'name') { throw 'schemaVersion 2 model object is missing name.' }
        [string]$_.name
    })
    if (@($modelNames | Sort-Object -Unique).Count -ne $modelNames.Count) { throw 'schemaVersion 2 package contains duplicate model names.' }
    if ($modelNames.Count -ne 2 -or @($modelNames | Where-Object { $_ -eq $chatModel }).Count -ne 1 -or
        @($modelNames | Where-Object { $_ -eq $embeddingModel }).Count -ne 1) {
        throw 'schemaVersion 2 model inventory must contain exactly the selected chat and required embedding models.'
    }
    $null = Assert-OfflineAiUnsupportedOverrideReceipt -Receipt $Manifest.selection.unsupportedOverride
    return [PSCustomObject]@{
        chatModel = $chatModel
        embeddingModel = $embeddingModel
        unsupportedOverride = $Manifest.selection.unsupportedOverride
    }
}

$installState = $null
try {
    Write-Step "1/7 パッケージ検証"
    Assert-SupportedArchitecture
    $resolvedPackageRoot = Resolve-PackageRoot -Value $PackageRoot
    Write-Host "  PackageRoot: $resolvedPackageRoot"
    $manifest = Read-PackageManifest -Root $resolvedPackageRoot
    $earlyManifestProperties = @($manifest.PSObject.Properties.Name)
    $earlyPackageKind = if ($earlyManifestProperties -contains 'packageKind') { [string]$manifest.packageKind } else { '' }
    $earlyInstallable = if ($earlyManifestProperties -contains 'installable') { [bool]$manifest.installable } else { $true }
    if ($earlyPackageKind -in @('public-source', 'download-only-no-models') -or -not $earlyInstallable) {
        throw "この成果物はインストールできません: packageKind=$earlyPackageKind installable=$earlyInstallable"
    }
    Test-Checksums -Root $resolvedPackageRoot
    $modelContract = Resolve-PackageModelContract -Root $resolvedPackageRoot -Manifest $manifest
    $selectedChatModel = [string]$modelContract.chatModel
    $selectedEmbedModel = [string]$modelContract.embeddingModel
    if ([bool]$modelContract.unsupportedOverride.used) {
        Write-Host "  [警告] このpackageは対象PCのRAM最小要件未達を明示承認して作成されています。" -ForegroundColor Yellow
    }
    Assert-InstallerIdentities -Root $resolvedPackageRoot -Manifest $manifest
    Test-Fat32PackageMedia -Root $resolvedPackageRoot
    Write-Step "2/7 アプリ本体確認"
    $packageAppRoot = Resolve-AppRoot -Root $resolvedPackageRoot
    $appRoot = Resolve-AppRoot -Root $resolvedPackageRoot -InstallPath $AppInstallPath
    Write-Host "  AppRoot: $appRoot"
    $previousChatModel = Get-ModelConfigSnapshot -Path (Join-Path $appRoot '_internal\.model')
    $appliedChatModel = if ($ModelConfigMode -eq 'KeepExisting' -and $previousChatModel.state -eq 'value') {
        [string]$previousChatModel.value
    } else {
        $selectedChatModel
    }
    if ($ModelConfigMode -eq 'KeepExisting' -and $appliedChatModel -ne $selectedChatModel) {
        Write-Host "  [警告] package選択chat=$selectedChatModel、既存設定を維持して実効chat=$appliedChatModel" -ForegroundColor Yellow
    }
    $modelsDest = Resolve-OllamaModelsPath
    $modelSourceRoot = Join-Path $resolvedPackageRoot "ollama-models"
    $appRequirement = Get-DirectoryCopyRequirementBytes -SourceRoot $packageAppRoot -MinimumBytes 1GB
    $modelRequirement = Get-DirectoryCopyRequirementBytes -SourceRoot $modelSourceRoot -MinimumBytes 20GB
    $tempPath = [System.IO.Path]::GetTempPath()
    Test-InstallVolumeRequirements -Requirements @(
        [PSCustomObject]@{ Path = $appRoot; Bytes = $appRequirement; Label = "app" }
        [PSCustomObject]@{ Path = $modelsDest; Bytes = $modelRequirement; Label = "models+atomic-part" }
        [PSCustomObject]@{ Path = $tempPath; Bytes = [UInt64]2GB; Label = "temp+installer" }
    )

    $installersRoot = Join-Path $resolvedPackageRoot "installers"
    $pythonExisted = [bool](Find-OfflineAiCommandPath -Name "python")
    $pwshExisted = [bool](Find-OfflineAiCommandPath -Name "pwsh")
    $ollamaExisted = [bool](Find-OfflineAiCommandPath -Name "ollama")
    $pythonInstaller = Find-FirstFile -Root $installersRoot -Patterns @("python-*-amd64.exe")
    if (-not $pythonInstaller) { throw "Python インストーラーが見つかりません: installers\python-*-amd64.exe" }
    $psMsi = Find-FirstFile -Root $installersRoot -Patterns @("PowerShell-*-win-x64.msi")
    if ($psMsi -and -not (Find-OfflineAiCommandPath -Name "pwsh") -and -not $DryRun -and -not (Test-IsAdministrator)) {
        throw "PowerShell 7 MSI の導入には管理者権限が必要です。管理者として再実行するか、MSIをパッケージから外してください。"
    }
    $manifestProperties = @($manifest.PSObject.Properties.Name)
    $productVersion = if ($manifestProperties -contains "productVersion") {
        [string]$manifest.productVersion
    } elseif (Test-Path -LiteralPath (Join-Path $packageAppRoot "VERSION") -PathType Leaf) {
        ([string](Get-Content -LiteralPath (Join-Path $packageAppRoot "VERSION") -Raw -Encoding UTF8)).Trim()
    } else {
        "unknown"
    }
    $previousInstallState = Read-InstallState -AppRoot $appRoot
    $previousBackups = @()
    if ($previousInstallState -and
        @($previousInstallState.PSObject.Properties.Name) -contains "models" -and
        @($previousInstallState.models.PSObject.Properties.Name) -contains "manifestBackups") {
        $previousBackups = @($previousInstallState.models.manifestBackups)
    }
    $installState = [PSCustomObject]@{
        schemaVersion = 1
        productVersion = $productVersion
        recordedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        lastUpdatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        installationStatus = "InProgress"
        app = [PSCustomObject]@{
            path = $appRoot
            disposition = if ([string]::Equals($packageAppRoot, $appRoot, [StringComparison]::OrdinalIgnoreCase)) { "InPlace" } else { "Copied" }
        }
        components = [PSCustomObject]@{
            python = Get-PreservedComponentDisposition -PreviousState $previousInstallState -Name "python" -CurrentlyExists $pythonExisted -Included $true
            ollama = Get-PreservedComponentDisposition -PreviousState $previousInstallState -Name "ollama" -CurrentlyExists $ollamaExisted -Included $true
            powershell7 = Get-PreservedComponentDisposition -PreviousState $previousInstallState -Name "powershell7" -CurrentlyExists $pwshExisted -Included ([bool]$psMsi)
        }
        models = [PSCustomObject]@{
            path = $modelsDest
            names = @($selectedChatModel, $selectedEmbedModel)
            packageSelectedChatModel = $selectedChatModel
            appliedChatModel = $appliedChatModel
            previousChatModel = $previousChatModel
            unsupportedOverride = $modelContract.unsupportedOverride
            manifestBackups = $previousBackups
            placementObservation = $null
        }
    }
    Write-InstallState -AppRoot $appRoot -State $installState

    if (-not [string]::Equals($packageAppRoot, $appRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "  アプリ本体を配置します: $appRoot"
        Copy-TreeNoConflict -SourceRoot $packageAppRoot -DestinationRoot $appRoot
    }
    Initialize-SkillSource -AppRoot $appRoot

    Write-Step "3/7 依存ツール確認"
    Install-ExecutableIfMissing -Name "Python" -CommandName "python" -InstallerPath $pythonInstaller -Arguments @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0")
    if (-not $pythonExisted -and -not $DryRun) { $installState.components.python = "Installed" }
    Write-InstallState -AppRoot $appRoot -State $installState

    if ($psMsi) {
        Install-MsiIfMissing -Name "PowerShell 7" -CommandName "pwsh" -MsiPath $psMsi -MsiProperties @("ADD_EXPLORER_CONTEXT_MENU_OPENPOWERSHELL=0", "ENABLE_PSREMOTING=0", "REGISTER_MANIFEST=1", "USE_MU=0", "ENABLE_MU=0")
        if (-not $pwshExisted -and -not $DryRun) { $installState.components.powershell7 = "Installed" }
        Write-InstallState -AppRoot $appRoot -State $installState
    } else {
        Write-Host "  [情報] PowerShell 7 MSI は同梱されていません。Windows PowerShell で続行します。" -ForegroundColor Yellow
    }
    Write-InstallState -AppRoot $appRoot -State $installState

    Write-Step "4/7 Ollama 確認"
    $ollamaExe = Find-OfflineAiCommandPath -Name "ollama"
    if (-not $ollamaExe) {
        $ollamaInstaller = Join-Path $installersRoot "OllamaSetup.exe"
        Install-ExecutableIfMissing -Name "Ollama" -CommandName "ollama" -InstallerPath $ollamaInstaller -Arguments @("/SILENT")
        $ollamaExe = Find-OfflineAiCommandPath -Name "ollama"
    }
    if (-not $ollamaExisted -and -not $DryRun) { $installState.components.ollama = "Installed" }
    Write-InstallState -AppRoot $appRoot -State $installState
    if (-not $ollamaExe -and -not $DryRun) {
        throw "Ollama コマンドが見つかりません。"
    }
    if ($DryRun -and -not $ollamaExe) {
        $ollamaExe = "ollama"
    }

    Write-Step "5/7 Ollama モデル配置"
    Write-Host "  Ollama models: $modelsDest"
    Copy-OllamaModels -Root $resolvedPackageRoot -DestinationModelsPath $modelsDest
    $installState.models.manifestBackups = @(
        $previousBackups + @(
            Get-ChildItem -LiteralPath (Join-Path $modelsDest "manifests") -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object Name -Like "*.offline-ai-backup-*" |
                ForEach-Object FullName
        ) | Sort-Object -Unique
    )
    Write-InstallState -AppRoot $appRoot -State $installState

    Write-Step "6/7 モデル認識確認"
    if ($DryRun) {
        Write-Plan "Ollama 起動確認"
        Write-Plan "ollama list で $appliedChatModel / $selectedEmbedModel を確認"
    } else {
        Start-OllamaIfNeeded -OllamaExe $ollamaExe
        $names = @(Get-OfflineAiOllamaModelNames -OllamaExe $ollamaExe)
        Assert-ModelExists -ModelNames $names -ExpectedModel $appliedChatModel
        Assert-ModelExists -ModelNames $names -ExpectedModel $selectedEmbedModel
    }

    Write-Step "7/7 offline-ai 設定とローカル検証"
    Write-ModelFile -Path (Join-Path $appRoot "_internal\.model") -Value $selectedChatModel -Mode $ModelConfigMode
    Write-ModelFile -Path (Join-Path $appRoot "_internal\.model_embed") -Value $selectedEmbedModel -Mode $ModelConfigMode
    Install-OptionalComponents -Root $resolvedPackageRoot -AppRoot $appRoot -Manifest $manifest
    $installState.installationStatus = "ConfiguredBeforeSmokeTest"
    $installState.lastUpdatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-InstallState -AppRoot $appRoot -State $installState
    if (-not $DryRun) {
        Invoke-RunSmokeTest -OllamaExe $ollamaExe -Model $appliedChatModel
        Invoke-EmbeddingSmokeTest -Model $selectedEmbedModel
    } else {
        Invoke-RunSmokeTest -OllamaExe $ollamaExe -Model $appliedChatModel
        Invoke-EmbeddingSmokeTest -Model $selectedEmbedModel
    }
    $placementObservation = Get-OfflineAiPlacementObservation -Model $appliedChatModel
    Write-OfflineAiPlacementObservation -Observation $placementObservation
    $installState.models.placementObservation = $placementObservation

    $installState.installationStatus = "Completed"
    $installState.lastUpdatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-InstallState -AppRoot $appRoot -State $installState

    Write-Host ""
    Write-Host "完全オフライン導入が完了しました。" -ForegroundColor Green
    Write-Host "起動: $appRoot\search.bat / $appRoot\web.bat"
    exit 0
} catch {
    $failure = $_
    if ($installState -and -not $DryRun) {
        try {
            $installState.installationStatus = "Failed"
            $installState.lastUpdatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            Write-InstallState -AppRoot ([string]$installState.app.path) -State $installState
        } catch {
            Write-Host "[警告] install-stateの失敗状態を記録できませんでした。" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "[エラー] 完全オフライン導入に失敗しました。" -ForegroundColor Red
    Write-Host $failure.Exception.Message -ForegroundColor Red
    exit 1
}
