<#
.SYNOPSIS
    offline-ai のダウンロード専用オフライン導入パッケージを作成する。

.DESCRIPTION
    オンラインPCに Python、Ollama、Git、PowerShell 7、モデルをインストールせず、
    搬送用フォルダに必要ファイルだけを収集する。DryRun ではネットワーク取得も
    ファイル作成も行わない。
#>
[CmdletBinding()]
param(
    [string]$OutputPath = "",
    [string]$InstallersPath = "",
    [string]$DownloadManifestPath = "",
    [string]$RegistryRoot = "",
    [string]$OptionalComponentsPath = "",
    [switch]$SkipInstallers,
    [switch]$SkipModels,
    [switch]$IncludeReranker,
    [switch]$PublicRelease,
    [string]$TargetGpuName = '',
    [ValidateSet('Nvidia', 'Amd', 'Intel', 'Other', 'None', 'Unknown')]
    [string]$TargetGpuVendor = 'Unknown',
    [Nullable[decimal]]$TargetVramGiB,
    [Nullable[decimal]]$TargetRamGiB,
    [ValidateSet('SingleVolume', 'ModelsSeparate', 'AllSeparate')]
    [string]$TargetStorageLayout = '',
    [Nullable[decimal]]$TargetAppFreeDiskGiB,
    [Nullable[decimal]]$TargetModelsFreeDiskGiB,
    [Nullable[decimal]]$TargetTempFreeDiskGiB,
    [ValidateSet('Speed', 'Balanced', 'Quality')]
    [string]$Preference = 'Balanced',
    [string]$ChatModel = '',
    [switch]$AllowUnsupportedChatModel,
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$InvocationParameters = @{} + $PSBoundParameters
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$CommonModulePath = Join-Path $PSScriptRoot "OfflineAi.Common.psm1"
$RegistryModulePath = Join-Path $PSScriptRoot "OllamaRegistryDownloader.psm1"
$ModelSelectionModulePath = Join-Path $PSScriptRoot "OfflineAi.ModelSelection.psm1"
Import-Module $CommonModulePath -Force
Import-Module $RegistryModulePath -Force
Import-Module $ModelSelectionModulePath -Force

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

function Read-DownloadManifest {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "download manifest が見つかりません: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Assert-PublicReleaseReady {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    foreach ($name in @("LICENSE", "THIRD-PARTY-NOTICES.md", "README.md", "SUPPORT.md", "UNINSTALL.md", "VERSION", "release-metadata.json")) {
        $path = Join-Path $AppRoot $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "public release blocker: required file is missing: $name"
        }
    }
    $licensePath = Join-Path $AppRoot "LICENSE"
    $licenseText = [string](Get-Content -LiteralPath $licensePath -Raw -Encoding UTF8)
    if ([string]::IsNullOrWhiteSpace($licenseText)) {
        throw "public release blocker: LICENSE must not be empty"
    }
    $noticesText = [string](Get-Content -LiteralPath (Join-Path $AppRoot "THIRD-PARTY-NOTICES.md") -Raw -Encoding UTF8)
    if ([string]::IsNullOrWhiteSpace($noticesText)) {
        throw "public release blocker: THIRD-PARTY-NOTICES.md must not be empty"
    }
    $metadata = Get-Content -LiteralPath (Join-Path $AppRoot "release-metadata.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $metadataProperties = @($metadata.PSObject.Properties.Name)
    $supportUri = $null
    $supportAddress = $null
    $supportUriValid = [Uri]::TryCreate([string]$metadata.supportUrl, [UriKind]::Absolute, [ref]$supportUri)
    $supportIsLoopback = $supportUriValid -and
        [System.Net.IPAddress]::TryParse($supportUri.Host, [ref]$supportAddress) -and
        [System.Net.IPAddress]::IsLoopback($supportAddress)
    if (-not $metadata.supportUrl -or
        -not $supportUriValid -or
        $supportUri.Scheme -ne "https" -or
        $supportUri.Host -in @("example.com", "example.org", "example.net", "localhost") -or
        $supportUri.Host.EndsWith(".example", [StringComparison]::OrdinalIgnoreCase) -or
        $supportIsLoopback) {
        throw "public release blocker: release-metadata.json supportUrl must be an actual HTTPS URL"
    }
    $version = ([string](Get-Content -LiteralPath (Join-Path $AppRoot "VERSION") -Raw -Encoding UTF8)).Trim()
    if (-not $version -or [string]$metadata.productVersion -ne $version) {
        throw "public release blocker: VERSION and release-metadata.json productVersion must match"
    }
    $licenseHash = Get-OfflineAiSha256 -Path $licensePath
    if ($metadataProperties -notcontains "licenseSha256" -or [string]$metadata.licenseSha256 -ne $licenseHash) {
        throw "public release blocker: approved LICENSE SHA-256 is missing or mismatched"
    }
    $approvedAt = [DateTimeOffset]::MinValue
    if ($metadataProperties -notcontains "approvedBy" -or [string]::IsNullOrWhiteSpace([string]$metadata.approvedBy) -or
        $metadataProperties -notcontains "approvedAt" -or
        -not [DateTimeOffset]::TryParse([string]$metadata.approvedAt, [ref]$approvedAt)) {
        throw "public release blocker: owner/legal approval record is incomplete"
    }
    if (-not $metadata.licenseApproved -or -not $metadata.redistributionApproved) {
        throw "public release blocker: owner/legal approval is not recorded"
    }
}

function Assert-SupportedDownloadArchitecture {
    $architecture = Get-OfflineAiNativeArchitecture
    if ($architecture -notin @("AMD64", "X64")) {
        throw "offline-ai package作成はWindows 11 x64のみ対応しています。検出architecture: $architecture"
    }
}

function Download-FileVerified {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$ExpectedSha256 = "",
        [string]$ExpectedProductVersion = "",
        [string]$ExpectedSignerSubjectContains = "",
        [switch]$DryRun
    )

    function Assert-FileIdentity {
        param([Parameter(Mandatory = $true)][string]$Path)
        if ($ExpectedProductVersion) {
            $actualVersion = ([System.Diagnostics.FileVersionInfo]::GetVersionInfo($Path)).ProductVersion
            if (-not $actualVersion -or -not $actualVersion.Trim().StartsWith($ExpectedProductVersion, [StringComparison]::Ordinal)) {
                throw "製品バージョンが期待値と一致しません: $Id (expected=$ExpectedProductVersion, actual=$actualVersion)"
            }
        }
        if ($ExpectedSignerSubjectContains) {
            if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
                throw "Authenticode署名検証はWindows上で実行してください: $Id"
            }
            $signature = Get-AuthenticodeSignature -LiteralPath $Path
            $subject = if ($signature.SignerCertificate) { [string]$signature.SignerCertificate.Subject } else { "" }
            if ($signature.Status -ne "Valid" -or
                $subject.IndexOf($ExpectedSignerSubjectContains, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
                throw "Authenticode署名者が期待値と一致しません: $Id"
            }
        }
    }

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $actualHash = Get-OfflineAiSha256 -Path $Destination
        if ($ExpectedSha256 -and $actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "既存ファイルの SHA-256 が expectedSha256 と一致しません: $Id"
        }
        Assert-FileIdentity -Path $Destination
        return [PSCustomObject]@{
            id = $Id
            relativeFile = Split-Path -Leaf $Destination
            status = "Existing"
            sizeBytes = (Get-Item -LiteralPath $Destination).Length
            sha256 = $actualHash
            expectedSha256 = $ExpectedSha256
            expectedProductVersion = $ExpectedProductVersion
            expectedSignerSubjectContains = $ExpectedSignerSubjectContains
            source = $Url
        }
    }

    if ($DryRun) {
        return [PSCustomObject]@{
            id = $Id
            relativeFile = Split-Path -Leaf $Destination
            status = "Planned"
            sizeBytes = 0
            sha256 = ""
            expectedSha256 = $ExpectedSha256
            expectedProductVersion = $ExpectedProductVersion
            expectedSignerSubjectContains = $ExpectedSignerSubjectContains
            source = $Url
        }
    }

    $directory = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -Path $directory -ItemType Directory -Force | Out-Null
    }

    $part = "$Destination.part-$([guid]::NewGuid().ToString('N'))"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing -ErrorAction Stop
        $actualHash = Get-OfflineAiSha256 -Path $part
        if ($ExpectedSha256 -and $actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "ダウンロードファイルの SHA-256 が expectedSha256 と一致しません: $Id"
        }
        Assert-FileIdentity -Path $part
        if (Test-Path -LiteralPath $Destination -PathType Leaf) {
            throw "Download target appeared during transfer: $Destination"
        }
        Move-Item -LiteralPath $part -Destination $Destination
        return [PSCustomObject]@{
            id = $Id
            relativeFile = Split-Path -Leaf $Destination
            status = "Downloaded"
            sizeBytes = (Get-Item -LiteralPath $Destination).Length
            sha256 = $actualHash
            expectedSha256 = $ExpectedSha256
            expectedProductVersion = $ExpectedProductVersion
            expectedSignerSubjectContains = $ExpectedSignerSubjectContains
            source = $Url
        }
    } finally {
        if (Test-Path -LiteralPath $part -PathType Leaf) {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-OptionalComponentSelected {
    param([Parameter(Mandatory = $true)][object]$Item)
    $propertyNames = @($Item.PSObject.Properties.Name)
    if ($propertyNames -notcontains "includeFlag") {
        return $true
    }
    $flag = [string]$Item.includeFlag
    if (-not $flag) {
        return $true
    }
    if ($flag -eq "IncludeReranker") {
        return [bool]$IncludeReranker
    }
    return $false
}

function Get-SelectedOptionalComponentDefinitions {
    param([Parameter(Mandatory = $true)][object]$Manifest)

    $items = @($Manifest.tools) + @($Manifest.models)
    foreach ($item in $items) {
        $propertyNames = @($item.PSObject.Properties.Name)
        if ($propertyNames -notcontains "includeFlag") { continue }
        if (-not (Test-OptionalComponentSelected -Item $item)) { continue }

        $componentId = if ($propertyNames -contains "id") { [string]$item.id } else { [string]$item.name }
        foreach ($requiredProperty in @("stagingDirectory", "packageRelativePath", "componentKind", "provider", "activation")) {
            if ($propertyNames -notcontains $requiredProperty -or
                [string]::IsNullOrWhiteSpace([string]$item.$requiredProperty)) {
                throw "optional component definition is missing ${requiredProperty}: $componentId"
            }
        }

        $rawPackageRelativePath = [string]$item.packageRelativePath
        $packageRelativePath = $rawPackageRelativePath.Replace("/", "\")
        if ([System.IO.Path]::IsPathRooted($rawPackageRelativePath) -or
            [System.IO.Path]::IsPathRooted($packageRelativePath)) {
            throw "optional component packageRelativePath must be relative: $componentId"
        }
        $pathSegments = @($packageRelativePath.Split([System.IO.Path]::DirectorySeparatorChar))
        if ($pathSegments.Count -lt 2 -or
            @($pathSegments | Where-Object { -not $_ -or $_ -in @(".", "..") }).Count -gt 0 -or
            $pathSegments[0] -ne "optional-components") {
            throw "optional component packageRelativePath must be under optional-components: $componentId"
        }
        $requiredFiles = @()
        if ($propertyNames -contains "requiredFiles") {
            $requiredFiles = @($item.requiredFiles | ForEach-Object { [string]$_ })
        }
        $licenseReviewRequired = $false
        if ($propertyNames -contains "licenseReviewRequired") {
            $licenseReviewRequired = [bool]$item.licenseReviewRequired
        }

        [PSCustomObject]@{
            id = $componentId
            stagingDirectory = [string]$item.stagingDirectory
            packageRelativePath = $packageRelativePath
            requiredFiles = $requiredFiles
            componentKind = [string]$item.componentKind
            provider = [string]$item.provider
            activation = [string]$item.activation
            licenseReviewRequired = $licenseReviewRequired
        }
    }
}

function Read-TargetDecimal {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [switch]$Optional
    )
    while ($true) {
        $value = Read-Host $Prompt
        if ($Optional -and [string]::IsNullOrWhiteSpace($value)) { return $null }
        [decimal]$parsed = 0
        if ([decimal]::TryParse($value, [Globalization.NumberStyles]::Number, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -and $parsed -ge 0) {
            return $parsed
        }
        Write-Host "  0以上の数値を入力してください。" -ForegroundColor Yellow
    }
}

function Confirm-OfflineAiUnsupportedChatModel {
    param([Parameter(Mandatory = $true)][string]$ChatModel)

    $answer = Read-Host "RAM最小要件未満の $ChatModel は実行非推奨です。選択に続く2回目の確認として、承知して収録しますか? [y/N]"
    if ($answer -notmatch '^(?i:y|yes)$') {
        throw 'RAM不足モデルの2回目の確認が得られませんでした。'
    }
    return $true
}

function Write-ModelRecommendationTable {
    param([Parameter(Mandatory = $true)][object[]]$Recommendations)
    Write-Host ''
    Write-Host '  推薦は8K context・単一利用の見込みです。実機、driver、Ollama版で変わり、性能を保証しません。' -ForegroundColor Yellow
    foreach ($item in $Recommendations) {
        $downloadGiB = [math]::Round([Int64]$item.transferEstimateBytes / 1GB, 2)
        Write-Host ("  [{0}] {1} ({2}) / {3} [{4}] / 搬送見積 {5} GiB / RAM {6}+ GiB / VRAM partial {7}+・full {8}+ GiB" -f `
            $item.rank, $item.model, $item.tier, $item.recommendationClassLabel, $item.recommendationClass, $downloadGiB, $item.minimumRamGiB, $item.partialGpuVramGiB, $item.fullGpuVramGiB)
        Write-Host "      推薦理由: $($item.recommendationReason)"
        foreach ($shortage in @($item.storage.shortages)) {
            Write-Host ("      容量不足: {0}が {1} GiB不足（必要 {2} / 空き {3} GiB）" -f `
                $shortage.labelJa,
                [math]::Round([decimal]$shortage.shortageBytes / 1GB, 2),
                [math]::Round([decimal]$shortage.requiredBytes / 1GB, 2),
                [math]::Round([decimal]$shortage.availableBytes / 1GB, 2)) -ForegroundColor Yellow
        }
        foreach ($notice in @($item.compatibilityNotices)) { Write-Host "      GPU互換性注意: $notice" -ForegroundColor Yellow }
        foreach ($notice in @($item.notices)) { Write-Host "      注意: $notice" }
    }
    if (@($Recommendations | Where-Object supported).Count -eq 0) {
        $minimum = @($Recommendations | Sort-Object estimatedDownloadBytes | Select-Object -First 1)[0]
        Write-Host "  [警告] 適合候補がありません。最小候補は $($minimum.model) です。" -ForegroundColor Yellow
        Write-Host "      不適合理由: $($minimum.recommendationReason)" -ForegroundColor Yellow
    }
}

function Test-SelectionParameterSpecified {
    return [bool]($TargetGpuName -or $InvocationParameters.ContainsKey('TargetGpuVendor') -or
        $InvocationParameters.ContainsKey('TargetVramGiB') -or $InvocationParameters.ContainsKey('TargetRamGiB') -or
        $TargetStorageLayout -or $InvocationParameters.ContainsKey('TargetAppFreeDiskGiB') -or
        $InvocationParameters.ContainsKey('TargetModelsFreeDiskGiB') -or $InvocationParameters.ContainsKey('TargetTempFreeDiskGiB') -or
        $InvocationParameters.ContainsKey('Preference') -or $ChatModel -or $AllowUnsupportedChatModel)
}

$appRoot = Resolve-OfflineAiFullPath -Path (Join-Path $PSScriptRoot "..")
$packageRoot = Join-Path $appRoot "offline-package"
if (-not $OutputPath) {
    $OutputPath = Join-Path $packageRoot "offline-ai-offline-package"
}
if (-not $InstallersPath) {
    $InstallersPath = Join-Path $packageRoot "installers"
}
if (-not $DownloadManifestPath) {
    $DownloadManifestPath = Join-Path $PSScriptRoot "download-manifest.json"
}
if (-not $OptionalComponentsPath) {
    $OptionalComponentsPath = Join-Path $packageRoot "optional-components"
}

$resolvedPackageRoot = Resolve-OfflineAiFullPath -Path $packageRoot
$outputRoot = Assert-OfflineAiPathUnderRoot -Root $resolvedPackageRoot -Target $OutputPath
$installersRoot = Assert-OfflineAiPathUnderRoot -Root $resolvedPackageRoot -Target $InstallersPath
$downloadManifestFullPath = Resolve-OfflineAiFullPath -Path $DownloadManifestPath
$optionalComponentsRoot = Resolve-OfflineAiFullPath -Path $OptionalComponentsPath

try {
    Write-Step "1/7 download-only 事前検証"
    Assert-SupportedDownloadArchitecture
    Write-Host "  AppRoot: $appRoot"
    Write-Host "  OutputPath: $outputRoot"
    Write-Host "  InstallersPath: $installersRoot"
    Write-Host "  OptionalComponentsPath: $optionalComponentsRoot"
    Write-Host "  DownloadManifest: $downloadManifestFullPath"
    Write-Host "  Mode: インストールなし / PATH変更なし / サービス登録なし"

    $downloadManifest = Read-DownloadManifest -Path $downloadManifestFullPath
    $catalogValidation = Assert-OfflineAiModelCatalog -Catalog $downloadManifest
    if (-not $PublicRelease -and -not $SkipModels) {
        $distributionPolicy = Import-OfflineAiDistributionPolicy -PolicyPath (Join-Path $appRoot '_internal\distribution-policy.json')
        $distributionInventory = Get-OfflineAiDistributionInventory -SourceRoot $appRoot -Policy $distributionPolicy
        if ($distributionInventory.unmatchedPaths.Count -gt 0) {
            throw "app payload inventory contains unclassified path(s): $($distributionInventory.unmatchedPaths -join ', ')"
        }
        $appPayloadGate = Test-OfflineAiAppPayloadEstimate -Inventory $distributionInventory -Expected $catalogValidation.appPayloadEstimate
        if (-not $appPayloadGate.passed) {
            throw "appPayloadEstimate drift detected: expected total=$($appPayloadGate.expected.totalBytes), largest=$($appPayloadGate.expected.largestFileBytes) ($($appPayloadGate.expected.largestFile)), files=$($appPayloadGate.expected.fileCount); actual total=$($appPayloadGate.actual.totalBytes), largest=$($appPayloadGate.actual.largestFileBytes) ($($appPayloadGate.actual.largestFile)), files=$($appPayloadGate.actual.fileCount)"
        }
    }
    $nonInteractive = [bool]($Yes -or $DryRun -or [Console]::IsInputRedirected)
    $targetRequirements = $null
    $selectionResult = $null
    $recommendations = @()

    if ($PublicRelease) {
        if (Test-SelectionParameterSpecified) {
            throw 'PublicRelease と対象PCスペック・モデル選択引数は併用できません。'
        }
        Assert-PublicReleaseReady -AppRoot $appRoot
        $SkipInstallers = $true
        $SkipModels = $true
        if ($IncludeReranker) {
            throw "public source release must not include optional binary/model components"
        }
        Write-Host "  Profile: public source release (third-party binaries/models excluded)" -ForegroundColor Green
    } elseif ($SkipModels) {
        if (Test-SelectionParameterSpecified) {
            throw 'SkipModels と対象PCスペック・モデル選択引数は併用できません。'
        }
    } else {
        if ($AllowUnsupportedChatModel -and -not $nonInteractive) {
            throw 'AllowUnsupportedChatModel は -Yes または標準入力redirectの非対話実行専用です。'
        }
        if ($AllowUnsupportedChatModel -and -not $ChatModel) {
            throw 'AllowUnsupportedChatModel には ChatModel の指定が必要です。'
        }
        if (-not $nonInteractive) {
            if (-not $InvocationParameters.ContainsKey('TargetGpuName')) {
                $TargetGpuName = Read-Host '対象オフラインPCのGPU名（任意、空欄可）'
            }
            if (-not $InvocationParameters.ContainsKey('TargetGpuVendor')) {
                $value = (Read-Host 'GPUベンダー [Nvidia/Amd/Intel/Other/None/Unknown] (既定 Unknown)').Trim()
                if ($value) {
                    $canonical = @('Nvidia', 'Amd', 'Intel', 'Other', 'None', 'Unknown') | Where-Object { $_ -ieq $value } | Select-Object -First 1
                    if (-not $canonical) { throw "不正なGPUベンダーです: $value" }
                    $TargetGpuVendor = $canonical
                }
            }
            if (-not $InvocationParameters.ContainsKey('TargetVramGiB')) {
                Write-Host '  タスク マネージャーの「専用GPUメモリ」を入力してください。共有GPUメモリは加算しません。' -ForegroundColor Yellow
                $TargetVramGiB = Read-TargetDecimal -Prompt '専用VRAM GiB（不明なら空欄）' -Optional
            }
            if ($null -eq $TargetRamGiB) { $TargetRamGiB = Read-TargetDecimal -Prompt 'システムRAM GiB' }
            if (-not $TargetStorageLayout) {
                $TargetStorageLayout = (Read-Host '保存先構成 [SingleVolume/ModelsSeparate/AllSeparate]').Trim()
            }
            if ($null -eq $TargetAppFreeDiskGiB) { $TargetAppFreeDiskGiB = Read-TargetDecimal -Prompt 'app保存先の空き容量 GiB' }
            if ($null -eq $TargetModelsFreeDiskGiB) { $TargetModelsFreeDiskGiB = Read-TargetDecimal -Prompt 'models保存先の空き容量 GiB' }
            if ($null -eq $TargetTempFreeDiskGiB) { $TargetTempFreeDiskGiB = Read-TargetDecimal -Prompt 'temp保存先の空き容量 GiB' }
        } else {
            $missing = @()
            if ($null -eq $TargetRamGiB) { $missing += 'TargetRamGiB' }
            if (-not $TargetStorageLayout) { $missing += 'TargetStorageLayout' }
            if ($null -eq $TargetAppFreeDiskGiB) { $missing += 'TargetAppFreeDiskGiB' }
            if ($null -eq $TargetModelsFreeDiskGiB) { $missing += 'TargetModelsFreeDiskGiB' }
            if ($null -eq $TargetTempFreeDiskGiB) { $missing += 'TargetTempFreeDiskGiB' }
            if (-not $DryRun -and -not $ChatModel) { $missing += 'ChatModel' }
            if ($missing.Count -gt 0) { throw "非対話実行に必要な引数が不足しています: $($missing -join ', ')" }
        }
        $targetRequirements = ConvertTo-OfflineAiTargetRequirements `
            -GpuName $TargetGpuName -GpuVendor $TargetGpuVendor -VramGiB $TargetVramGiB `
            -RamGiB $TargetRamGiB -StorageLayout $TargetStorageLayout `
            -AppFreeDiskGiB $TargetAppFreeDiskGiB -ModelsFreeDiskGiB $TargetModelsFreeDiskGiB `
            -TempFreeDiskGiB $TargetTempFreeDiskGiB -Preference $Preference
        $recommendations = @(Get-OfflineAiModelRecommendations -Catalog $downloadManifest -Target $targetRequirements)
        Write-ModelRecommendationTable -Recommendations $recommendations

        if (-not $ChatModel -and -not $DryRun) {
            $choice = Read-Host '収録するchatモデルの番号'
            [int]$choiceNumber = 0
            if (-not [int]::TryParse($choice, [ref]$choiceNumber)) { throw 'モデル選択番号が不正です。' }
            $chosen = @($recommendations | Where-Object { $_.rank -eq $choiceNumber })
            if ($chosen.Count -ne 1) { throw 'モデル選択番号が範囲外です。' }
            $ChatModel = [string]$chosen[0].model
        }
        if ($ChatModel) {
            $selectedPreview = @($recommendations | Where-Object { $_.model -eq $ChatModel })
            if ($selectedPreview.Count -ne 1) { throw "ChatModel はmanifest上の選択可能chatモデルではありません: $ChatModel" }
            $allowUnsupported = [bool]$AllowUnsupportedChatModel
            $confirmationMode = 'explicit-switch'
            if (-not $selectedPreview[0].ramSupported -and -not $nonInteractive) {
                $null = Confirm-OfflineAiUnsupportedChatModel -ChatModel $ChatModel
                $allowUnsupported = $true
                $confirmationMode = 'interactive-second-confirmation'
            }
            $selectionResult = Resolve-OfflineAiChatSelection -Catalog $downloadManifest -Target $targetRequirements `
                -ChatModel $ChatModel -AllowUnsupported:$allowUnsupported -OverrideConfirmationMode $confirmationMode
        }

        if (-not $DryRun -and -not $Yes -and -not [Console]::IsInputRedirected) {
            $selectedGiB = [math]::Round([Int64]$selectionResult.recommendation.transferEstimateBytes / 1GB, 2)
            Write-Host "  選択モデル: $ChatModel / chat+Embedding搬送見積: $selectedGiB GiB" -ForegroundColor Yellow
            Write-Host "  取得元: registry.ollama.ai。モデルlicenseをパッケージへ記録します。"
        }
    }

    if (-not $DryRun -and -not $Yes -and (-not $SkipInstallers -or -not $SkipModels) -and -not [Console]::IsInputRedirected) {
        Write-Host "  検索語は既定ログへ保存しません。取得物は自身のオフラインPCへの搬送用です。"
        $answer = Read-Host "ダウンロードを開始しますか? [y/N]"
        if ($answer -notmatch '^(?i:y|yes)$') {
            throw "利用者がダウンロードをキャンセルしました。"
        }
    }

    if (Test-Path -LiteralPath $outputRoot) {
        if (-not $Force) {
            throw "出力先が既に存在します。作り直す場合だけ -Force を指定してください: $outputRoot"
        }
        Write-Plan "出力先を再作成: $outputRoot"
        if (-not $DryRun) {
            Remove-Item -LiteralPath $outputRoot -Recurse -Force
        }
    }

    if ($DryRun) {
        Write-Plan "ファイル作成とネットワーク取得は行いません。"
    } else {
        New-Item -Path $outputRoot -ItemType Directory -Force | Out-Null
        New-Item -Path $installersRoot -ItemType Directory -Force | Out-Null
    }

    Write-Step "2/7 インストーラー収集"
    $toolResults = @()
    if ($SkipInstallers) {
        Write-Host "  [警告] インストーラー収集をスキップします。" -ForegroundColor Yellow
    } else {
        foreach ($tool in @($downloadManifest.tools)) {
            $id = [string]$tool.id
            if (-not (Test-OptionalComponentSelected -Item $tool)) {
                Write-Host "  ${id}: OptionalSkipped"
                continue
            }
            $downloadUrl = [string]$tool.downloadUrl
            $required = [bool]$tool.required
            if (-not $downloadUrl) {
                $status = if ($required) { "MissingDownloadUrl" } else { "ManualOnly" }
                Write-Host "  ${id}: $status"
                if ($required) {
                    throw "必須ツールに downloadUrl がありません: $id"
                }
                $toolResults += [PSCustomObject]@{
                    id = $id
                    relativeFile = [string]$tool.fileName
                    status = $status
                    sizeBytes = 0
                    sha256 = ""
                    expectedSha256 = [string]$tool.expectedSha256
                    source = [string]$tool.source
                }
                continue
            }

            $destination = Join-Path $installersRoot ([string]$tool.fileName)
            Write-Plan "$id を取得: $downloadUrl"
            $propertyNames = @($tool.PSObject.Properties.Name)
            $expectedVersion = if ($propertyNames -contains "expectedProductVersion") { [string]$tool.expectedProductVersion } else { "" }
            $expectedSigner = if ($propertyNames -contains "expectedSignerSubjectContains") { [string]$tool.expectedSignerSubjectContains } else { "" }
            $toolResults += Download-FileVerified `
                -Id $id `
                -Url $downloadUrl `
                -Destination $destination `
                -ExpectedSha256 ([string]$tool.expectedSha256) `
                -ExpectedProductVersion $expectedVersion `
                -ExpectedSignerSubjectContains $expectedSigner `
                -DryRun:$DryRun
        }
    }

    Write-Step "3/7 Ollama モデル直接取得"
    $modelResults = @()
    if ($SkipModels) {
        Write-Host "  [警告] モデル取得をスキップします。" -ForegroundColor Yellow
    } elseif (-not $selectionResult) {
        Write-Host "  [DryRun] ChatModel未指定のため、順位表のみを表示し取得対象は未確定です。" -ForegroundColor Yellow
    } else {
        $modelsToAcquire = @(
            @($catalogValidation.selectableChats | Where-Object { $_.name -eq $ChatModel })
            $catalogValidation.requiredEmbedding
        )
        foreach ($model in $modelsToAcquire) {
            $modelName = [string]$model.name
            if (-not (Test-OptionalComponentSelected -Item $model)) {
                Write-Host "  ${modelName}: OptionalSkipped"
                continue
            }
            $registry = [string]$model.registry
            $manualOnly = $false
            if (@($model.PSObject.Properties.Name) -contains "manualOnly") {
                $manualOnly = [bool]$model.manualOnly
            }
            if ($manualOnly -or -not $registry) {
                $modelResults += [PSCustomObject]@{
                    name = $modelName
                    repository = [string]$model.repository
                    tag = [string]$model.tag
                    registry = $registry
                    status = "ManualOnly"
                    blobCount = 0
                    totalBytes = 0
                    manifestDigest = ""
                }
                Write-Host "  ${modelName}: ManualOnly"
                continue
            }
            Write-Plan "$modelName を registry から取得: $registry"
            if ($DryRun) {
                $reference = Resolve-OllamaModelReference -Name $modelName
                $modelResults += [PSCustomObject]@{
                    name = $modelName
                    repository = $reference.repository
                    tag = $reference.tag
                    registry = $registry
                    status = "Planned"
                    blobCount = 0
                    totalBytes = [Int64]$model.recommendation.estimatedDownloadBytes
                    largestBlobBytes = [Int64]$model.recommendation.estimatedLargestBlobBytes
                    manifestDigest = ""
                }
            } else {
                $modelResults += Save-OllamaModelFromRegistry `
                    -ModelName $modelName `
                    -DestinationModelsRoot (Join-Path $outputRoot "ollama-models") `
                    -RegistryBaseUrl $registry `
                    -RegistryRoot $RegistryRoot `
                    -ExpectedManifestDigest ([string]$model.expectedManifestDigest) `
                    -ExpectedConfigDigest ([string]$model.expectedConfigDigest) `
                    -ExpectedLicenseLayerDigest ([string]$model.expectedLicenseLayerDigest)
            }
        }
    }

    Write-Step "4/7 任意コンポーネント収集"
    $optionalComponentResults = @()
    $optionalDefinitions = @(
        if (-not $PublicRelease) {
            Get-SelectedOptionalComponentDefinitions -Manifest $downloadManifest
        }
    )
    if ($optionalDefinitions.Count -eq 0) {
        Write-Host "  任意コンポーネントは選択されていません。標準構成を維持します。"
    }
    foreach ($definition in $optionalDefinitions) {
        $copyResult = Copy-OfflineAiOptionalComponent `
            -SourceRoot $optionalComponentsRoot `
            -DestinationRoot $outputRoot `
            -ComponentId $definition.id `
            -StagingDirectory $definition.stagingDirectory `
            -PackageRelativePath $definition.packageRelativePath `
            -RequiredFiles $definition.requiredFiles `
            -DryRun:$DryRun
        $optionalComponentResults += [PSCustomObject]@{
            id = $definition.id
            componentKind = $definition.componentKind
            provider = $definition.provider
            activation = $definition.activation
            licenseReviewRequired = $definition.licenseReviewRequired
            packageRelativePath = $copyResult.packageRelativePath
            status = $copyResult.status
            fileCount = $copyResult.fileCount
            totalBytes = $copyResult.totalBytes
            files = $copyResult.files
        }
        Write-Host "  $($definition.id): $($copyResult.status) ($($copyResult.fileCount) files)"
    }

    if ($DryRun) {
        Write-Step "5/7 DryRun 結果"
        Write-Host "  作成予定: $outputRoot"
        Write-Host "  インストーラー保存先: $installersRoot"
        Write-Host "  搬送媒体は exFAT または NTFS を推奨します。FAT32 は 4GB 超 blob を扱えません。" -ForegroundColor Yellow
        exit 0
    }

    Write-Step "5/7 アプリ本体と導入スクリプト収集"
    if (-not $PublicRelease) {
        Copy-OfflineAiDirectoryExact -SourceRoot $installersRoot -DestinationRoot (Join-Path $outputRoot "installers")
    }
    $distributionPolicy = Import-OfflineAiDistributionPolicy -PolicyPath (Join-Path $appRoot "_internal\distribution-policy.json")
    # -PublicRelease: GitHub等へ渡すpublic source artifact（tests/CI/community文書を含む）。
    # 通常経路: 利用者向けruntime app payload（tests/CI/開発文書を含めない）。
    $appIncludeDistributions = if ($PublicRelease) { @("both", "public-source", "release-only") } else { @("both", "runtime") }
    Copy-OfflineAiDirectoryByDistribution `
        -SourceRoot $appRoot `
        -DestinationRoot (Join-Path $outputRoot "app\offline-ai") `
        -Policy $distributionPolicy `
        -IncludeDistributions $appIncludeDistributions

    $scriptsDir = Join-Path $outputRoot "scripts"
    New-Item -Path $scriptsDir -ItemType Directory -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $appRoot "install-offline.bat") -Destination (Join-Path $scriptsDir "install-offline.bat")
    Copy-Item -LiteralPath (Join-Path $appRoot "_internal\install-offline.ps1") -Destination (Join-Path $scriptsDir "install-offline.ps1")
    Copy-Item -LiteralPath (Join-Path $appRoot "_internal\OfflineAi.Common.psm1") -Destination (Join-Path $scriptsDir "OfflineAi.Common.psm1")
    if (-not $PublicRelease) {
        Copy-Item -LiteralPath (Join-Path $appRoot "_internal\OfflineAi.ModelSelection.psm1") -Destination (Join-Path $scriptsDir "OfflineAi.ModelSelection.psm1")
    }

    $modelLicenseInventory = @(Export-OfflineAiModelLicenses `
        -ModelsRoot (Join-Path $outputRoot "ollama-models") `
        -DestinationRoot (Join-Path $outputRoot "legal\models"))

    Write-Step "6/7 manifest 作成"
    $installerInventory = @()
    if (Test-Path -LiteralPath (Join-Path $outputRoot "installers") -PathType Container) {
        foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $outputRoot "installers") -File -Recurse | Sort-Object FullName)) {
            $installerInventory += [PSCustomObject]@{
                relativePath = Get-OfflineAiRelativePath -BasePath $outputRoot -Path $file.FullName
                sizeBytes = $file.Length
                sha256 = Get-OfflineAiSha256 -Path $file.FullName
            }
        }
    }

    $manifestData = [ordered]@{
        schemaVersion = 2
        packageKind = if ($PublicRelease) { "public-source" } elseif ($SkipModels) { 'download-only-no-models' } else { "download-only" }
        installable = [bool](-not $PublicRelease -and -not $SkipModels)
        productVersion = ([string](Get-Content -LiteralPath (Join-Path $appRoot "VERSION") -Raw -Encoding UTF8)).Trim()
        createdAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        os = Get-OfflineAiOsSummary
        models = @($modelResults | ForEach-Object {
            $modelResult = $_
            $definition = @($downloadManifest.models | Where-Object { $_.name -eq $modelResult.name } | Select-Object -First 1)
            [PSCustomObject]@{
                name = $modelResult.name
                repository = $modelResult.repository
                tag = $modelResult.tag
                registry = $modelResult.registry
                manifestDigest = $modelResult.manifestDigest
                expectedManifestDigest = if ($definition.Count -eq 1) { [string]$definition[0].expectedManifestDigest } else { '' }
                expectedLicenseLayerDigest = if ($definition.Count -eq 1) { [string]$definition[0].expectedLicenseLayerDigest } else { '' }
                blobCount = $modelResult.blobCount
                totalBytes = $modelResult.totalBytes
                largestBlobBytes = $modelResult.largestBlobBytes
            }
        })
        modelLicenses = $modelLicenseInventory
        distributionProfile = if ($PublicRelease) {
            [PSCustomObject]@{
                kind = "public-source"
                thirdPartyBinariesIncluded = $false
                modelsIncluded = $false
                notes = @("利用者はonline staging PCでuser-built transport packageを別途作成します。")
            }
        } else {
            [PSCustomObject]@{
                kind = "user-built-transport"
                installedOnDownloadPc = $false
                usesOllamaPull = $false
                notes = @(
                    "このパッケージ作成処理はオンラインPCへ Python、Ollama、Git、PowerShell 7、モデルをインストールしません。",
                    "Ollama モデルは registry.ollama.ai の manifest/blob 構造を直接保存します。公式のローカル /api/pull とは別経路です。"
                )
            }
        }
        install = [PSCustomObject]@{
            defaultAppRoot = "app\offline-ai"
            defaultOllamaModels = "%USERPROFILE%\.ollama\models"
            modelConfigModeDefault = "OverwriteDefault"
        }
        installers = $installerInventory
        downloads = $toolResults
        optionalComponents = $optionalComponentResults
        notes = @(
            "モデル本体とインストーラーの再配布条件は一次配布元で別途確認してください。",
            "manifest にはユーザー名や環境変数全体を記録しません。",
            "Python は download-manifest.json の必須ツール定義に従い 3.11.9 に固定しています。"
        )
    }
    if ($selectionResult) {
        $manifestData.selection = [PSCustomObject]@{
            chatModel = [string]$selectionResult.recommendation.model
            embeddingModel = [string]$catalogValidation.requiredEmbedding.name
            preference = [string]$targetRequirements.preference
            recommendationClass = [string]$selectionResult.recommendation.recommendationClass
            recommendationClassLabel = [string]$selectionResult.recommendation.recommendationClassLabel
            recommendationReason = [string]$selectionResult.recommendation.recommendationReason
            compatibilityNotices = @($selectionResult.recommendation.compatibilityNotices)
            catalogReviewedAt = [string]$selectionResult.recommendation.catalogReviewedAt
            unsupportedOverride = $selectionResult.unsupportedOverride
            storageAssessment = [PSCustomObject]@{
                appRequiredBytes = [Int64]$selectionResult.recommendation.storage.appRequiredBytes
                modelsRequiredBytes = [Int64]$selectionResult.recommendation.storage.modelsRequiredBytes
                tempRequiredBytes = [Int64]$selectionResult.recommendation.storage.tempRequiredBytes
                shortages = @($selectionResult.recommendation.storage.shortages)
            }
        }
        $manifestData.targetRequirementsInput = [PSCustomObject]@{
            gpuVendor = [string]$targetRequirements.gpuVendor
            gpuName = [string]$targetRequirements.gpuName
            vramGiB = $targetRequirements.vramGiB
            ramGiB = $targetRequirements.ramGiB
            storageLayout = [string]$targetRequirements.storageLayout
            appFreeDiskGiB = $targetRequirements.appFreeDiskGiB
            modelsFreeDiskGiB = $targetRequirements.modelsFreeDiskGiB
            tempFreeDiskGiB = $targetRequirements.tempFreeDiskGiB
        }
    }
    $manifest = [PSCustomObject]$manifestData
    if ($PublicRelease) {
        $manifest = New-OfflineAiPublicSourceManifest -SourceManifest $manifest
    }
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outputRoot "manifest.json") -Encoding UTF8

    if ($PublicRelease) {
        Assert-OfflineAiPublicArtifactContents -Root $outputRoot | Out-Null
    }

    Write-Step "7/7 checksum 作成"
    $inventory = @(New-OfflineAiFileInventory -Root $outputRoot)
    Write-OfflineAiChecksums -Root $outputRoot -Inventory $inventory

    $isFat32 = Test-OfflineAiFat32PackageMedia -Root $outputRoot
    if ($isFat32) {
        Write-Host "  [警告] 出力先が FAT32 です。4GB 超のモデル blob を扱えません。exFAT または NTFS を使ってください。" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "ダウンロード専用オフライン導入パッケージを作成しました。" -ForegroundColor Green
    Write-Host "出力先: $outputRoot"
    exit 0
} catch {
    Write-Host ""
    Write-Host "[エラー] ダウンロード専用パッケージ作成に失敗しました。" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
