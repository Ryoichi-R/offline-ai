BeforeAll {
    # 子scriptはUTF-8で出力する。親の既定code page（日本語Windowsではcp932）で
    # 取り込むと日本語が化けてassertionが失敗するため、実行中だけUTF-8へ固定する。
    $script:OriginalConsoleOutputEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:DownloadPackagePath = Join-Path $script:OfflineAiRoot '_internal\download-offline-package.ps1'
    $script:InstallOfflinePath = Join-Path $script:OfflineAiRoot '_internal\install-offline.ps1'
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    $script:CanonicalCatalogPath = Join-Path $script:OfflineAiRoot '_internal\download-manifest.json'
    $script:IsWindowsRuntime = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    $script:IsX64Runtime = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [System.Runtime.InteropServices.Architecture]::X64
    $script:PowerShellHost = (Get-Process -Id $PID).Path
    Import-Module $script:CommonModulePath -Force

    function Get-FixtureSha256 {
        param([Parameter(Mandatory = $true)][byte[]]$Bytes)

        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
    }

    function New-E2eRegistryModel {
        param(
            [Parameter(Mandatory = $true)][string]$RegistryRoot,
            [Parameter(Mandatory = $true)][string]$Repository,
            [Parameter(Mandatory = $true)][string]$Tag,
            [Parameter(Mandatory = $true)][string]$Stem
        )

        $repoRoot = Join-Path $RegistryRoot 'v2'
        foreach ($segment in @($Repository -split '/')) { $repoRoot = Join-Path $repoRoot $segment }
        $manifestRoot = Join-Path $repoRoot 'manifests'
        $blobRoot = Join-Path $repoRoot 'blobs'
        New-Item -Path $manifestRoot -ItemType Directory -Force | Out-Null
        New-Item -Path $blobRoot -ItemType Directory -Force | Out-Null

        $configBytes = [Text.Encoding]::UTF8.GetBytes("{`"model`":`"$Stem`"}")
        $modelBytes = [Text.Encoding]::UTF8.GetBytes("fixture-model-$Stem")
        $licenseBytes = [Text.Encoding]::UTF8.GetBytes("Apache License 2.0 fixture for $Stem")
        $configHash = Get-FixtureSha256 -Bytes $configBytes
        $modelHash = Get-FixtureSha256 -Bytes $modelBytes
        $licenseHash = Get-FixtureSha256 -Bytes $licenseBytes
        [IO.File]::WriteAllBytes((Join-Path $blobRoot "sha256-$configHash"), $configBytes)
        [IO.File]::WriteAllBytes((Join-Path $blobRoot "sha256-$modelHash"), $modelBytes)
        [IO.File]::WriteAllBytes((Join-Path $blobRoot "sha256-$licenseHash"), $licenseBytes)

        $manifest = [ordered]@{
            schemaVersion = 2
            mediaType = 'application/vnd.docker.distribution.manifest.v2+json'
            config = [ordered]@{
                mediaType = 'application/vnd.docker.container.image.v1+json'
                digest = "sha256:$configHash"
                size = $configBytes.Length
            }
            layers = @(
                [ordered]@{ mediaType = 'application/vnd.ollama.image.model'; digest = "sha256:$modelHash"; size = $modelBytes.Length }
                [ordered]@{ mediaType = 'application/vnd.ollama.image.license'; digest = "sha256:$licenseHash"; size = $licenseBytes.Length }
            )
        } | ConvertTo-Json -Depth 8 -Compress
        $manifestBytes = [Text.Encoding]::UTF8.GetBytes($manifest)
        [IO.File]::WriteAllBytes((Join-Path $manifestRoot $Tag), $manifestBytes)
        return [PSCustomObject]@{
            manifestDigest = "sha256:$(Get-FixtureSha256 -Bytes $manifestBytes)"
            configDigest = "sha256:$configHash"
            licenseDigest = "sha256:$licenseHash"
        }
    }

    function Set-PackageChecksums {
        param([Parameter(Mandatory = $true)][string]$PackageRoot)

        $inventory = @(New-OfflineAiFileInventory -Root $PackageRoot)
        Write-OfflineAiChecksums -Root $PackageRoot -Inventory $inventory
    }
}

AfterAll {
    if ($null -ne $script:OriginalConsoleOutputEncoding) {
        [Console]::OutputEncoding = $script:OriginalConsoleOutputEncoding
    }
}

Describe 'offline-ai target-spec schema v2 package to installer E2E' -Tag 'WindowsOnly' {
    It 'propagates a nondefault selection and preserves install state on noninstallable rejection' {
        if (-not $script:IsWindowsRuntime -or -not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because 'install-offline.ps1 targets Windows 11 x64'
            return
        }

        $id = [guid]::NewGuid().ToString('N')
        $packageBase = Join-Path $script:OfflineAiRoot 'offline-package'
        $packageRoot = Join-Path $packageBase "target-spec-e2e-$id"
        $overridePackageRoot = Join-Path $packageBase "target-spec-e2e-override-$id"
        $dryRunOutputPath = Join-Path $packageBase "target-spec-e2e-dryrun-$id"
        $staleDryRunOutputPath = Join-Path $packageBase "target-spec-e2e-stale-$id"
        $emptyInstallers = Join-Path $packageBase "target-spec-e2e-installers-$id"
        $workRoot = Join-Path $TestDrive $id
        $registryRoot = Join-Path $workRoot 'registry'
        $catalogPath = Join-Path $workRoot 'download-manifest.fixture.json'
        $installRoot = Join-Path $workRoot 'installed-app'
        $modelsRoot = Join-Path $workRoot 'installed-models'
        $fakeBin = Join-Path $workRoot 'fake-bin'
        $oldPath = $env:PATH
        $oldModels = $env:OLLAMA_MODELS
        $oldConsoleOutputEncoding = [Console]::OutputEncoding
        try {
            # Child scripts emit UTF-8; native stdout decoding follows the parent console encoding.
            [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
            New-Item -Path $emptyInstallers -ItemType Directory -Force | Out-Null
            New-Item -Path $workRoot -ItemType Directory -Force | Out-Null
            $chatFixture = New-E2eRegistryModel -RegistryRoot $registryRoot -Repository 'library/qwen3.5' -Tag '4b' -Stem 'qwen3.5-4b'
            $overrideChatFixture = New-E2eRegistryModel -RegistryRoot $registryRoot -Repository 'library/qwen3.5' -Tag '27b' -Stem 'qwen3.5-27b'
            $embedFixture = New-E2eRegistryModel -RegistryRoot $registryRoot -Repository 'library/bge-m3' -Tag 'latest' -Stem 'bge-m3'

            $catalog = Get-Content -LiteralPath $script:CanonicalCatalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $catalog.tools = @()
            $chatDefinition = @($catalog.models | Where-Object name -eq 'qwen3.5:4b')[0]
            $chatDefinition.expectedManifestDigest = $chatFixture.manifestDigest
            $chatDefinition.expectedConfigDigest = $chatFixture.configDigest
            $chatDefinition.expectedLicenseLayerDigest = $chatFixture.licenseDigest
            $overrideChatDefinition = @($catalog.models | Where-Object name -eq 'qwen3.5:27b')[0]
            $overrideChatDefinition.expectedManifestDigest = $overrideChatFixture.manifestDigest
            $overrideChatDefinition.expectedConfigDigest = $overrideChatFixture.configDigest
            $overrideChatDefinition.expectedLicenseLayerDigest = $overrideChatFixture.licenseDigest
            $embedDefinition = @($catalog.models | Where-Object name -eq 'bge-m3')[0]
            $embedDefinition.expectedManifestDigest = $embedFixture.manifestDigest
            $embedDefinition.expectedConfigDigest = $embedFixture.configDigest
            $embedDefinition.expectedLicenseLayerDigest = $embedFixture.licenseDigest
            [IO.File]::WriteAllText($catalogPath, ($catalog | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))

            $staleCatalogPath = Join-Path $workRoot 'download-manifest.stale.json'
            $staleCatalog = $catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
            $staleCatalog.appPayloadEstimate.totalBytes = [Int64]$staleCatalog.appPayloadEstimate.totalBytes - 1
            [IO.File]::WriteAllText($staleCatalogPath, ($staleCatalog | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
            $staleOutput = & $script:PowerShellHost -NoProfile -File $script:DownloadPackagePath `
                -OutputPath $staleDryRunOutputPath `
                -InstallersPath $emptyInstallers `
                -DownloadManifestPath $staleCatalogPath `
                -SkipInstallers `
                -TargetGpuVendor Unknown `
                -TargetVramGiB 12 `
                -TargetRamGiB 32 `
                -TargetStorageLayout AllSeparate `
                -TargetAppFreeDiskGiB 10 `
                -TargetModelsFreeDiskGiB 40 `
                -TargetTempFreeDiskGiB 4 `
                -DryRun 2>&1
            $staleText = $staleOutput | Out-String
            $LASTEXITCODE | Should -Be 1 -Because $staleText
            $staleText | Should -Match 'appPayloadEstimate drift detected'
            Test-Path -LiteralPath $staleDryRunOutputPath | Should -BeFalse

            $dryRunOutput = & $script:PowerShellHost -NoProfile -File $script:DownloadPackagePath `
                -OutputPath $dryRunOutputPath `
                -InstallersPath $emptyInstallers `
                -DownloadManifestPath $catalogPath `
                -SkipInstallers `
                -TargetGpuVendor Unknown `
                -TargetVramGiB 12 `
                -TargetRamGiB 32 `
                -TargetStorageLayout AllSeparate `
                -TargetAppFreeDiskGiB 0 `
                -TargetModelsFreeDiskGiB 0 `
                -TargetTempFreeDiskGiB 0 `
                -DryRun 2>&1
            $dryRunText = $dryRunOutput | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $dryRunText
            $dryRunText | Should -Match '適合候補がありません'
            $dryRunText | Should -Match '容量不足:'
            $dryRunText | Should -Match '保存先'
            Test-Path -LiteralPath $dryRunOutputPath | Should -BeFalse

            $overrideOutput = & $script:PowerShellHost -NoProfile -File $script:DownloadPackagePath `
                -OutputPath $overridePackageRoot `
                -InstallersPath $emptyInstallers `
                -DownloadManifestPath $catalogPath `
                -RegistryRoot $registryRoot `
                -SkipInstallers `
                -TargetGpuVendor Unknown `
                -TargetVramGiB 8 `
                -TargetRamGiB 8 `
                -TargetStorageLayout AllSeparate `
                -TargetAppFreeDiskGiB 10 `
                -TargetModelsFreeDiskGiB 40 `
                -TargetTempFreeDiskGiB 4 `
                -ChatModel 'qwen3.5:27b' `
                -AllowUnsupportedChatModel `
                -Yes 2>&1
            $overrideText = $overrideOutput | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $overrideText
            $overrideManifest = Get-Content -LiteralPath (Join-Path $overridePackageRoot 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            $overrideManifest.selection.unsupportedOverride.used | Should -BeTrue
            @($overrideManifest.selection.unsupportedOverride.reasonCodes) | Should -Be @('ram-below-minimum')
            $overrideManifest.selection.unsupportedOverride.confirmationMode | Should -Be 'explicit-switch'

            $packageOutput = & $script:PowerShellHost -NoProfile -File $script:DownloadPackagePath `
                -OutputPath $packageRoot `
                -InstallersPath $emptyInstallers `
                -DownloadManifestPath $catalogPath `
                -RegistryRoot $registryRoot `
                -SkipInstallers `
                -TargetGpuVendor Unknown `
                -TargetVramGiB 12 `
                -TargetRamGiB 32 `
                -TargetStorageLayout AllSeparate `
                -TargetAppFreeDiskGiB 10 `
                -TargetModelsFreeDiskGiB 40 `
                -TargetTempFreeDiskGiB 4 `
                -ChatModel 'qwen3.5:4b' `
                -Yes 2>&1
            $packageText = $packageOutput | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $packageText
            $packageText | Should -Match '推薦理由:'
            $packageText | Should -Match 'GPU互換性注意:'

            $packageManifest = Get-Content -LiteralPath (Join-Path $packageRoot 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            $packageManifest.schemaVersion | Should -Be 2
            $packageManifest.selection.chatModel | Should -Be 'qwen3.5:4b'
            $packageManifest.selection.embeddingModel | Should -Be 'bge-m3'
            $packageManifest.selection.recommendationClass | Should -Be 'full-gpu-likely'
            $packageManifest.selection.compatibilityNotices -join ' ' | Should -Match 'GPUベンダー未指定'
            @($packageManifest.models).Count | Should -Be 2
            @($packageManifest.models.name) | Should -Contain 'qwen3.5:4b'
            @($packageManifest.models.name) | Should -Contain 'bge-m3'
            Test-Path -LiteralPath (Join-Path $packageRoot 'ollama-models\manifests\registry.ollama.ai\library\qwen3.5\9b') | Should -BeFalse
            $checksums = Get-Content -LiteralPath (Join-Path $packageRoot 'checksums.sha256') -Raw -Encoding UTF8
            $checksums | Should -Match 'scripts/OfflineAi.ModelSelection.psm1'
            $checksums | Should -Match 'app/offline-ai/_internal/download-manifest.json'

            New-Item -Path (Join-Path $packageRoot 'installers') -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $packageRoot 'installers\python-3.11.9-amd64.exe') -Value 'fixture-python-installer' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $packageRoot 'installers\OllamaSetup.exe') -Value 'fixture-ollama-installer' -Encoding ASCII
            Set-PackageChecksums -PackageRoot $packageRoot

            New-Item -Path $fakeBin -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $fakeBin 'python.cmd') -Encoding ASCII -Value @('@echo off', 'exit /b 0')
            Set-Content -LiteralPath (Join-Path $fakeBin 'ollama.cmd') -Encoding ASCII -Value @(
                '@echo off',
                'if /I "%1"=="list" (',
                '  echo NAME ID SIZE MODIFIED',
                '  echo qwen3.5:4b fixture 1 GB now',
                '  echo qwen3.5:9b fixture 1 GB now',
                '  echo bge-m3:latest fixture 1 GB now',
                '  exit /b 0',
                ')',
                'exit /b 0'
            )
            New-Item -Path (Join-Path $installRoot '_internal') -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $installRoot '_internal\.model') -Value 'qwen3.5:9b' -Encoding UTF8
            New-Item -Path $modelsRoot -ItemType Directory -Force | Out-Null
            $env:PATH = "$fakeBin;$oldPath"
            $env:OLLAMA_MODELS = $modelsRoot

            $keepOutput = & $script:PowerShellHost -NoProfile -File $script:InstallOfflinePath `
                -PackageRoot $packageRoot `
                -AppInstallPath $installRoot `
                -ModelConfigMode KeepExisting `
                -SkipRunSmokeTest `
                -SkipEmbeddingSmokeTest 2>&1
            $keepText = $keepOutput | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $keepText
            (Get-Content -LiteralPath (Join-Path $installRoot '_internal\.model') -Raw -Encoding UTF8).Trim() | Should -Be 'qwen3.5:9b'
            $keepState = Get-Content -LiteralPath (Join-Path $installRoot '_internal\install-state.json') -Raw -Encoding UTF8 | ConvertFrom-Json
            $keepState.installationStatus | Should -Be 'Completed'
            $keepState.models.packageSelectedChatModel | Should -Be 'qwen3.5:4b'
            $keepState.models.appliedChatModel | Should -Be 'qwen3.5:9b'
            $keepState.models.placementObservation.status | Should -BeIn @('Observed', 'ModelNotLoaded', 'Unavailable')

            $overwriteOutput = & $script:PowerShellHost -NoProfile -File $script:InstallOfflinePath `
                -PackageRoot $packageRoot `
                -AppInstallPath $installRoot `
                -ModelConfigMode OverwriteDefault `
                -SkipRunSmokeTest `
                -SkipEmbeddingSmokeTest 2>&1
            $overwriteText = $overwriteOutput | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $overwriteText
            (Get-Content -LiteralPath (Join-Path $installRoot '_internal\.model') -Raw -Encoding UTF8).Trim() | Should -Be 'qwen3.5:4b'
            $overwriteStatePath = Join-Path $installRoot '_internal\install-state.json'
            $overwriteState = Get-Content -LiteralPath $overwriteStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $overwriteState.models.appliedChatModel | Should -Be 'qwen3.5:4b'
            $stateHashBeforeReject = (Get-FileHash -LiteralPath $overwriteStatePath -Algorithm SHA256).Hash

            $packageManifest.installable = $false
            [IO.File]::WriteAllText((Join-Path $packageRoot 'manifest.json'), ($packageManifest | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
            Set-PackageChecksums -PackageRoot $packageRoot
            $rejectOutput = & $script:PowerShellHost -NoProfile -File $script:InstallOfflinePath `
                -PackageRoot $packageRoot `
                -AppInstallPath $installRoot `
                -SkipRunSmokeTest `
                -SkipEmbeddingSmokeTest 2>&1
            $rejectText = $rejectOutput | Out-String
            $LASTEXITCODE | Should -Be 1 -Because $rejectText
            $rejectText | Should -Match 'インストールできません'
            (Get-FileHash -LiteralPath $overwriteStatePath -Algorithm SHA256).Hash | Should -Be $stateHashBeforeReject
        } finally {
            [Console]::OutputEncoding = $oldConsoleOutputEncoding
            $env:PATH = $oldPath
            $env:OLLAMA_MODELS = $oldModels
            $packageBaseFull = [IO.Path]::GetFullPath($packageBase).TrimEnd('\')
            foreach ($path in @($packageRoot, $overridePackageRoot, $dryRunOutputPath, $staleDryRunOutputPath, $emptyInstallers)) {
                $full = [IO.Path]::GetFullPath($path)
                if ($full.StartsWith($packageBaseFull + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
                    (Test-Path -LiteralPath $full)) {
                    Remove-Item -LiteralPath $full -Recurse -Force
                }
            }
        }
    }
}

Describe 'offline-ai existing qwen model continuity' -Tag 'WindowsOnly' {
    It 'resolves an existing qwen3.5:9b config without catalog selection or file mutation' {
        if (-not $script:IsWindowsRuntime -or -not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because 'runtime continuity check targets Windows 11 x64'
            return
        }

        $catalog = Get-Content -LiteralPath $script:CanonicalCatalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
        @($catalog.models | Where-Object name -eq 'qwen3.5:9b').Count | Should -Be 0
        @($catalog.legacyMigrationModels) | Should -Not -Contain 'qwen3.5:9b'

        $modelPath = Join-Path $TestDrive '.model'
        [IO.File]::WriteAllText($modelPath, "qwen3.5:9b`n", [Text.UTF8Encoding]::new($false))
        $before = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
        $internal = ((Join-Path $script:OfflineAiRoot '_internal') -replace '\\', '\\\\')
        $model = ($modelPath -replace '\\', '\\\\')
        $code = "import sys; from pathlib import Path; sys.path.insert(0, r'$internal'); from model_config import detect_model_config; print(detect_model_config(Path(r'$model')))"
        $resolved = & python -c $code 2>&1 | Out-String
        $LASTEXITCODE | Should -Be 0 -Because $resolved
        $resolved.Trim() | Should -Match 'qwen3\.5:9b$'
        (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash | Should -Be $before
    }
}
