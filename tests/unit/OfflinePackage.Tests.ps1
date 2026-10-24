Set-StrictMode -Version Latest

BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:InstallOfflinePath = Join-Path $script:OfflineAiRoot '_internal\install-offline.ps1'
    $script:BuildPackagePath = Join-Path $script:OfflineAiRoot '_internal\build-offline-package.ps1'
    $script:DownloadPackageBatPath = Join-Path $script:OfflineAiRoot 'download-package.bat'
    $script:DownloadPackagePath = Join-Path $script:OfflineAiRoot '_internal\download-offline-package.ps1'
    $script:RegistryDownloaderPath = Join-Path $script:OfflineAiRoot '_internal\OllamaRegistryDownloader.psm1'
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    $script:CollectSkillSourcePath = Join-Path $script:OfflineAiRoot '_internal\scripts\collect_skill_source.ps1'
    $script:IsWindowsRuntime = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    $script:IsX64Runtime = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [System.Runtime.InteropServices.Architecture]::X64
    Import-Module $script:CommonModulePath -Force
    $script:DefaultChatModel = 'qwen3.5:9b'
    $script:DefaultEmbedModel = 'bge-m3'
    $installTokens = $null
    $installParseErrors = $null
    $installAst = [System.Management.Automation.Language.Parser]::ParseFile($script:InstallOfflinePath, [ref]$installTokens, [ref]$installParseErrors)
    $installParseErrors.Count | Should -Be 0
    foreach ($functionName in @('Get-SchemaOneModelNames', 'Resolve-PackageModelContract', 'ConvertTo-OfflineAiPlacementObservation')) {
        $functionAst = $installAst.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $functionName
        }, $true)
        . ([scriptblock]::Create($functionAst.Extent.Text))
    }
}

Describe 'offline-ai installer model selection contract' {
    It 'normalizes both schemaVersion 1 inventory shapes and ignores unique optional extras' {
        Get-SchemaOneModelNames -Manifest ([PSCustomObject]@{ models = @('qwen3.5:9b', 'bge-m3', 'optional-reranker') }) |
            Should -Contain 'qwen3.5:9b'
        Get-SchemaOneModelNames -Manifest ([PSCustomObject]@{ models = @([PSCustomObject]@{ name = 'qwen3.5:9b' }, [PSCustomObject]@{ name = 'bge-m3' }) }) |
            Should -Contain 'bge-m3'
    }

    It 'rejects mixed, duplicate, and incomplete schemaVersion 1 inventories' {
        { Get-SchemaOneModelNames -Manifest ([PSCustomObject]@{ models = @('qwen3.5:9b', [PSCustomObject]@{ name = 'bge-m3' }) }) } | Should -Throw '*mixes*'
        { Get-SchemaOneModelNames -Manifest ([PSCustomObject]@{ models = @('qwen3.5:9b', 'bge-m3', 'bge-m3') }) } | Should -Throw '*duplicate*'
        { Get-SchemaOneModelNames -Manifest ([PSCustomObject]@{ models = @('qwen3.5:9b') }) } | Should -Throw '*must contain*'
    }

    It 'rejects noninstallable artifacts before package mutation' {
        $manifest = [PSCustomObject]@{ schemaVersion = 2; packageKind = 'download-only-no-models'; installable = $false; models = @() }
        { Resolve-PackageModelContract -Root $TestDrive -Manifest $manifest } | Should -Throw '*インストールできません*'
    }

    It 'resolves a nondefault schemaVersion 2 chat model from the packaged module and catalog' {
        $root = Join-Path $TestDrive 'schema2'
        New-Item -Path (Join-Path $root 'scripts') -ItemType Directory -Force | Out-Null
        New-Item -Path (Join-Path $root 'app\offline-ai\_internal') -ItemType Directory -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\OfflineAi.ModelSelection.psm1') -Destination (Join-Path $root 'scripts\OfflineAi.ModelSelection.psm1')
        Copy-Item -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\download-manifest.json') -Destination (Join-Path $root 'app\offline-ai\_internal\download-manifest.json')
        $manifest = [PSCustomObject]@{
            schemaVersion = 2
            packageKind = 'download-only'
            installable = $true
            selection = [PSCustomObject]@{
                chatModel = 'qwen3.5:4b'
                embeddingModel = 'bge-m3'
                unsupportedOverride = [PSCustomObject]@{ used = $false; reasonCodes = @(); confirmationMode = 'not-required' }
            }
            models = @([PSCustomObject]@{ name = 'qwen3.5:4b' }, [PSCustomObject]@{ name = 'bge-m3' })
        }
        $result = Resolve-PackageModelContract -Root $root -Manifest $manifest
        $result.chatModel | Should -Be 'qwen3.5:4b'
        $result.embeddingModel | Should -Be 'bge-m3'

        $manifest.models = @([PSCustomObject]@{ name = 'qwen3.5:9b' }, [PSCustomObject]@{ name = 'bge-m3' })
        { Resolve-PackageModelContract -Root $root -Manifest $manifest } | Should -Throw '*exactly the selected chat*'
    }

    It 'classifies full, partial, and CPU placement from the Ollama ps API shape' {
        $full = ConvertTo-OfflineAiPlacementObservation -Model 'qwen3.5:4b' -PsResponse ([PSCustomObject]@{
                models = @([PSCustomObject]@{ name = 'qwen3.5:4b'; size = 1000; size_vram = 1000; context_length = 8192 })
            })
        $partial = ConvertTo-OfflineAiPlacementObservation -Model 'qwen3.5:4b' -PsResponse ([PSCustomObject]@{
                models = @([PSCustomObject]@{ name = 'qwen3.5:4b'; size = 1000; size_vram = 400; context_length = 4096 })
            })
        $cpu = ConvertTo-OfflineAiPlacementObservation -Model 'qwen3.5:4b' -PsResponse ([PSCustomObject]@{
                models = @([PSCustomObject]@{ name = 'qwen3.5:4b'; size = 1000; size_vram = 0 })
            })
        $full.processor | Should -Be 'full-gpu'
        $full.gpuPercent | Should -Be 100
        $partial.processor | Should -Be 'partial-gpu'
        $partial.gpuPercent | Should -Be 40
        $cpu.processor | Should -Be 'cpu'
    }
}

Describe "offline-ai offline package scripts" {
    It "adds the offline installer entrypoints" {
        Test-Path (Join-Path $script:OfflineAiRoot 'install-offline.bat') | Should -BeTrue
        Test-Path $script:InstallOfflinePath | Should -BeTrue
        Test-Path $script:BuildPackagePath | Should -BeTrue
        Test-Path $script:DownloadPackageBatPath | Should -BeTrue
        Test-Path $script:DownloadPackagePath | Should -BeTrue
        Test-Path $script:RegistryDownloaderPath | Should -BeTrue
        Test-Path $script:CommonModulePath | Should -BeTrue
    }

    It "does not keep the removed normal setup entrypoints" {
        Test-Path (Join-Path $script:OfflineAiRoot 'install.bat') | Should -BeFalse
        Test-Path (Join-Path $script:OfflineAiRoot '_internal\setup-offline-ai.ps1') | Should -BeFalse
    }

    It "keeps install-offline free of external acquisition commands" {
        $content = @(
            Get-Content -LiteralPath $script:InstallOfflinePath -Raw
            Get-Content -LiteralPath $script:CommonModulePath -Raw
        ) -join "`n"
        $content | Should -Not -Match 'Invoke-WebRequest'
        $content | Should -Not -Match 'Invoke-RestMethod'
        $content | Should -Not -Match '\bcurl\b'
        $content | Should -Not -Match '\bwget\b'
        $content | Should -Not -Match '\bwinget\b'
        $content | Should -Not -Match '\bollama\s+pull\b'
        $content | Should -Not -Match 'https?://(?!localhost\b)'
    }

    It "keeps install-offline dry-run independent of host hardware preflight" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match 'if \(\$DryRun\)'
        $content | Should -Match 'Test-OfflineAiPreflight -RamBytes 16GB -FreeBytes \$MinimumFreeBytes'
    }

    It "keeps download-only entrypoints free of local installation commands" {
        $content = @(
            Get-Content -LiteralPath $script:DownloadPackageBatPath -Raw
            Get-Content -LiteralPath $script:DownloadPackagePath -Raw
            Get-Content -LiteralPath $script:RegistryDownloaderPath -Raw
        ) -join "`n"
        $content | Should -Not -Match '\bwinget\s+install\b'
        $content | Should -Not -Match '\bStart-Process\b'
        $content | Should -Not -Match '\bollama\s+pull\b'
        $content | Should -Not -Match '\bollama\s+serve\b'
        $content | Should -Not -Match '\bsetx(\.exe)?\b'
        $content | Should -Not -Match 'SetEnvironmentVariable'
        $content | Should -Not -Match 'Refresh-PathEnv'
    }

    It "uses qwen3.5:9b and bge-m3 as offline defaults" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match '\$DefaultChatModel\s*=\s*"qwen3\.5:9b"'
        $content | Should -Match '\$DefaultEmbedModel\s*=\s*"bge-m3"'
        $content | Should -Match '\.model_embed'
    }

    It "writes model and one-line config files as UTF-8 without BOM" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match 'function Write-Utf8NoBomText'
        $content | Should -Match '\[System\.Text\.UTF8Encoding\]::new\(\$false\)'
        $content | Should -Not -Match 'Set-Content\s+-LiteralPath\s+\$Path\s+-Value\s+\$Value\s+-Encoding\s+UTF8\s+-NoNewline'
    }

    It "keeps PS5.1-targeted Japanese PowerShell scripts as UTF-8 with BOM" {
        $ps51ScriptPaths = @(
            $script:InstallOfflinePath
            $script:BuildPackagePath
            $script:DownloadPackagePath
            $script:RegistryDownloaderPath
            $script:CommonModulePath
            (Join-Path $script:OfflineAiRoot '_internal\OfflineAi.ModelSelection.psm1')
            $script:CollectSkillSourcePath
        )

        foreach ($path in $ps51ScriptPaths) {
            $bytes = [System.IO.File]::ReadAllBytes($path)
            $hasUtf8Bom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
            $hasUtf8Bom | Should -BeTrue -Because "$path must run under Windows PowerShell 5.1 without mojibake"
        }
    }

    It "guards build package output and force deletion with a local-build sentinel" {
        $content = Get-Content -LiteralPath $script:BuildPackagePath -Raw
        $content | Should -Match 'Assert-BuildPackageOutputRoot'
        $content | Should -Match 'Test-LocalBuildPackageRoot'
        $content | Should -Match 'packageKind\s+=\s+"local-build"'
        $content | Should -Match 'createdBy\s+=\s+\$LocalBuildCreatedBy'
        $content | Should -Match 'Assert-OfflineAiPathUnderRoot\s+-Root\s+\$packageRoot'
    }

    It "uses targeted Ollama model copy by default in build packages" {
        $content = Get-Content -LiteralPath $script:BuildPackagePath -Raw
        $content | Should -Match 'Copy-TargetOllamaModels'
        $content | Should -Match 'IncludeAllLocalModels'
        $content | Should -Match '対象モデル manifest が見つかりません'
    }

    It "propagates PowerShell script exit codes from batch launchers" {
        $installBat = Get-Content -LiteralPath (Join-Path $script:OfflineAiRoot 'install-offline.bat') -Raw
        $downloadBat = Get-Content -LiteralPath $script:DownloadPackageBatPath -Raw
        $installBat | Should -Match 'where pwsh'
        $installBat | Should -Match '-File\s+"%SCRIPT_PATH%"\s+-PackageRoot\s+"%PACKAGE_ROOT%"'
        $installBat | Should -Not -Match "'%SCRIPT_PATH%'"
        $downloadBat | Should -Match '-File\s+"%~dp0_internal\\download-offline-package\.ps1"\s+%\*'
        $downloadBat | Should -Not -Match "'%~dp0_internal"
    }

    It "excludes user documents and internal infographic from package collection" {
        foreach ($path in @($script:BuildPackagePath, $script:DownloadPackagePath)) {
            $content = Get-Content -LiteralPath $path -Raw
            $content | Should -Match 'Import-OfflineAiDistributionPolicy'
            $content | Should -Match 'Copy-OfflineAiDirectoryByDistribution'
        }
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath (Join-Path $script:OfflineAiRoot '_internal\distribution-policy.json')
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'skill-source/private.md' | Should -Be 'excluded'
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'infographic/offline-ai-infographic.png' | Should -Be 'excluded'
    }

    It "rejects reparse points from all common package collectors" {
        $content = Get-Content -LiteralPath $script:CommonModulePath -Raw
        $content | Should -Match 'function Assert-OfflineAiNoReparsePoints'
        ([regex]::Matches($content, 'Assert-OfflineAiNoReparsePoints -Root \$SourceRoot')).Count | Should -BeGreaterOrEqual 2
    }

    It "keeps batch launchers ASCII-only for cmd.exe compatibility" {
        foreach ($path in @(Get-ChildItem -LiteralPath $script:OfflineAiRoot -Filter '*.bat' -File)) {
            $bytes = [System.IO.File]::ReadAllBytes($path.FullName)
            @($bytes | Where-Object { $_ -gt 0x7F }).Count |
                Should -Be 0 -Because "$($path.FullName) must remain ASCII-only"
        }
    }

    It "creates skill-source during offline install" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match 'Initialize-SkillSource'
        $content | Should -Match '利用者資料フォルダー作成'
        $content | Should -Match 'Write-InstallState'
        $content | Should -Match 'install-state\.json'
        $content | Should -Match 'installationStatus = "InProgress"'
        $content | Should -Match 'installationStatus = "Completed"'
        $content | Should -Match 'installationStatus = "Failed"'
        $content | Should -Match 'Get-PreservedComponentDisposition'
    }

    It "uses destination-aware preflight and explicit model conflict recovery" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match 'Test-InstallVolumeRequirements -Requirements'
        $content | Should -Match 'models\+atomic-part'
        $content | Should -Match 'temp\+installer'
        $content | Should -Match 'BackupAndReplace'
        $content | Should -Match 'offline-ai-backup-'
        $content | Should -Match 'http://localhost:11434/api/ps'
        $content | Should -Match 'placementObservation'
    }

    It "gates public release on complete owner approval metadata" {
        $content = Get-Content -LiteralPath $script:DownloadPackagePath -Raw
        $content | Should -Match 'Assert-PublicReleaseReady'
        $content | Should -Match 'licenseApproved'
        $content | Should -Match 'redistributionApproved'
        $metadata = Get-Content -LiteralPath (Join-Path $script:OfflineAiRoot 'release-metadata.json') -Raw | ConvertFrom-Json
        $metadata.licenseApproved | Should -BeTrue
        $metadata.redistributionApproved | Should -BeTrue
        # 承認を記録する以上、gateが要求する付随metadataも揃っていること。
        $metadata.supportUrl | Should -Not -BeNullOrEmpty
        $metadata.approvedBy | Should -Not -BeNullOrEmpty
        $metadata.approvedAt | Should -Not -BeNullOrEmpty
        [DateTimeOffset]::TryParse([string]$metadata.approvedAt, [ref]([DateTimeOffset]::MinValue)) | Should -BeTrue
    }

    It "excludes the removed conversion runtime from the source tree" {
        foreach ($relative in @(
                'convert.bat'
                '_internal\convert.py'
                '_internal\structured_parser.py'
                '_internal\scripts\extract_text.ps1'
                '_internal\scripts\verify_conversion.ps1'
            )) {
            Test-Path (Join-Path $script:OfflineAiRoot $relative) | Should -BeFalse
        }
    }

    It "installs optional components only when package directory exists" {
        $content = Get-Content -LiteralPath $script:InstallOfflinePath -Raw
        $content | Should -Match 'function Install-OptionalComponents'
        $content | Should -Match 'optional-components'
        $content | Should -Match '標準構成で続行します'
        $content | Should -Match '_internal\\optional-components'
        $content | Should -Match 'Install-OptionalComponents\s+-Root\s+\$resolvedPackageRoot\s+-AppRoot\s+\$appRoot\s+-Manifest\s+\$manifest'
        $content | Should -Match 'manifest に未宣言の optional component'
    }

    It "runs install-offline -DryRun against a minimal package" {
        if (-not $script:IsWindowsRuntime -or -not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because "install-offline.ps1 targets Windows 11 x64"
            return
        }
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-package-test-$([guid]::NewGuid().ToString('N'))"
        $modelsDest = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-models-$([guid]::NewGuid().ToString('N'))"
        $oldModels = $env:OLLAMA_MODELS
        try {
            New-Item -Path (Join-Path $tmp 'installers') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'ollama-models\blobs') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'ollama-models\manifests\registry.ollama.ai\library\qwen3.5') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'app\offline-ai\_internal') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'optional-components\reranker') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'optional-components\undeclared') -ItemType Directory -Force | Out-Null
            New-Item -Path $modelsDest -ItemType Directory -Force | Out-Null

            Set-Content -LiteralPath (Join-Path $tmp 'installers\OllamaSetup.exe') -Value 'dummy-ollama' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'installers\python-3.11.9-amd64.exe') -Value 'dummy-python' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'ollama-models\blobs\sha256-deadbeef') -Value 'dummy-blob' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'ollama-models\manifests\registry.ollama.ai\library\qwen3.5\9b') -Value 'dummy-manifest' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'optional-components\reranker\model.bin') -Value 'source-reranker' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'optional-components\undeclared\payload.exe') -Value 'undeclared' -Encoding ASCII
            $rerankerRelative = 'optional-components\reranker\model.bin'
            $manifest = @{
                schemaVersion = 1
                createdAt = '2026-06-29T00:00:00Z'
                models = @('qwen3.5:9b', 'bge-m3')
                optionalComponents = @(
                    @{
                        id = 'local-reranker'
                        packageRelativePath = 'optional-components\reranker'
                        activation = 'transport-only'
                        files = @(
                            @{ relativePath = $rerankerRelative; sha256 = (Get-FileHash -LiteralPath (Join-Path $tmp $rerankerRelative) -Algorithm SHA256).Hash.ToLowerInvariant() }
                        )
                    }
                )
            } | ConvertTo-Json -Depth 4
            Set-Content -LiteralPath (Join-Path $tmp 'manifest.json') -Value $manifest -Encoding UTF8

            $lines = @()
            foreach ($file in Get-ChildItem -LiteralPath $tmp -File -Recurse | Sort-Object FullName) {
                if ($file.Name -eq 'checksums.sha256') { continue }
                $relative = $file.FullName.Substring($tmp.Length).TrimStart('\')
                if ($relative.StartsWith('app\offline-ai\_internal\optional-components\', [System.StringComparison]::OrdinalIgnoreCase)) { continue }
                $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                $lines += "$hash *$relative"
            }
            Set-Content -LiteralPath (Join-Path $tmp 'checksums.sha256') -Value $lines -Encoding UTF8

            $env:OLLAMA_MODELS = $modelsDest
            $result = & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $script:InstallOfflinePath -PackageRoot $tmp -DryRun -SkipRunSmokeTest -SkipEmbeddingSmokeTest 2>&1
            $resultText = $result | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $resultText
            $resultText.Length | Should -BeGreaterThan 0
            # Native child output decoding can depend on the host console code
            # page. Match the stable ASCII contract fragments so the behavior
            # check remains deterministic across PowerShell hosts.
            $resultText | Should -Match 'reranker'
            $resultText | Should -Match 'optional component directory.*undeclared'
        } finally {
            $env:OLLAMA_MODELS = $oldModels
            foreach ($path in @($tmp, $modelsDest)) {
                if (Test-Path -LiteralPath $path) {
                    Remove-Item -LiteralPath $path -Recurse -Force
                }
            }
        }
    }

    It "reports a model blob collision during install-offline -DryRun" {
        if (-not $script:IsWindowsRuntime -or -not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because "install-offline.ps1 targets Windows 11 x64"
            return
        }
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-package-test-$([guid]::NewGuid().ToString('N'))"
        $modelsDest = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-models-$([guid]::NewGuid().ToString('N'))"
        $oldModels = $env:OLLAMA_MODELS
        try {
            New-Item -Path (Join-Path $tmp 'installers') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'ollama-models\blobs') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'ollama-models\manifests\registry.ollama.ai\library\qwen3.5') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $tmp 'app\offline-ai\_internal') -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $modelsDest 'blobs') -ItemType Directory -Force | Out-Null

            Set-Content -LiteralPath (Join-Path $tmp 'installers\OllamaSetup.exe') -Value 'dummy-ollama' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'installers\python-3.11.9-amd64.exe') -Value 'dummy-python' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'ollama-models\blobs\sha256-deadbeef') -Value 'source-blob' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $modelsDest 'blobs\sha256-deadbeef') -Value 'different-blob' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'ollama-models\manifests\registry.ollama.ai\library\qwen3.5\9b') -Value 'dummy-manifest' -Encoding ASCII
            $manifest = @{
                schemaVersion = 1
                createdAt = '2026-06-29T00:00:00Z'
                models = @('qwen3.5:9b', 'bge-m3')
            } | ConvertTo-Json -Depth 4
            Set-Content -LiteralPath (Join-Path $tmp 'manifest.json') -Value $manifest -Encoding UTF8

            $lines = @()
            foreach ($file in Get-ChildItem -LiteralPath $tmp -File -Recurse | Sort-Object FullName) {
                if ($file.Name -eq 'checksums.sha256') { continue }
                $relative = $file.FullName.Substring($tmp.Length).TrimStart('\')
                $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                $lines += "$hash *$relative"
            }
            Set-Content -LiteralPath (Join-Path $tmp 'checksums.sha256') -Value $lines -Encoding UTF8

            $env:OLLAMA_MODELS = $modelsDest
            $result = & powershell.exe -ExecutionPolicy Bypass -NoProfile -File $script:InstallOfflinePath -PackageRoot $tmp -DryRun -SkipRunSmokeTest -SkipEmbeddingSmokeTest 2>&1
            $resultText = $result | Out-String
            $LASTEXITCODE | Should -Be 1 -Because $resultText
            $resultText | Should -Match 'different content'
        } finally {
            $env:OLLAMA_MODELS = $oldModels
            foreach ($path in @($tmp, $modelsDest)) {
                if (Test-Path -LiteralPath $path) {
                    Remove-Item -LiteralPath $path -Recurse -Force
                }
            }
        }
    }
}

Describe "offline-ai remediation dynamic regressions" -Tag 'WindowsOnly' {
    BeforeAll {
        $script:SearchBatPath = Join-Path $script:OfflineAiRoot 'search.bat'
        $script:DownloadBatPath = Join-Path $script:OfflineAiRoot 'download-package.bat'
        Import-Module $script:CommonModulePath -Force
        $script:IsX64Runtime = (Get-OfflineAiNativeArchitecture) -eq 'X64'

        $tokens = $null
        $parseErrors = $null
        $buildAst = [System.Management.Automation.Language.Parser]::ParseFile(
            $script:BuildPackagePath,
            [ref]$tokens,
            [ref]$parseErrors
        )
        $parseErrors.Count | Should -Be 0
        foreach ($functionName in @(
            'Test-LocalBuildPackageRoot',
            'Get-OllamaManifestRelativePath',
            'Get-OllamaManifestBlobDigests',
            'Copy-TargetOllamaModels'
        )) {
            $functionAst = $buildAst.Find({
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq $functionName
            }, $true)
            $functionAst | Should -Not -BeNullOrEmpty
            . ([scriptblock]::Create($functionAst.Extent.Text))
        }
        $downloadTokens = $null
        $downloadParseErrors = $null
        $downloadAst = [System.Management.Automation.Language.Parser]::ParseFile(
            $script:DownloadPackagePath,
            [ref]$downloadTokens,
            [ref]$downloadParseErrors
        )
        $downloadParseErrors.Count | Should -Be 0
        foreach ($functionName in @('Assert-PublicReleaseReady')) {
            $functionAst = $downloadAst.Find({
                param($node)
                $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq $functionName
            }, $true)
            $functionAst | Should -Not -BeNullOrEmpty
            . ([scriptblock]::Create($functionAst.Extent.Text))
        }
        $script:LocalBuildCreatedBy = 'offline-ai/build-offline-package.ps1'
    }

    It "uses common package collection helpers without local duplicate definitions" {
        $buildContent = Get-Content -LiteralPath $script:BuildPackagePath -Raw
        $downloadContent = Get-Content -LiteralPath $script:DownloadPackagePath -Raw
        $commonContent = Get-Content -LiteralPath $script:CommonModulePath -Raw

        foreach ($name in @(
            'Copy-OfflineAiDirectoryByDistribution',
            'Copy-OfflineAiDirectoryExact',
            'New-OfflineAiFileInventory',
            'Write-OfflineAiChecksums',
            'Get-OfflineAiOsSummary'
        )) {
            $buildContent | Should -Match ([regex]::Escape($name))
            $downloadContent | Should -Match ([regex]::Escape($name))
            $commonContent | Should -Match ("function\s+" + [regex]::Escape($name))
        }
        $buildContent | Should -Not -Match 'function\s+Copy-DirectoryFiltered'
        $downloadContent | Should -Not -Match 'function\s+Copy-DirectoryFiltered'
    }

    It "requires the exact createdBy sentinel before a local-build package is replaceable" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-sentinel-$([guid]::NewGuid().ToString('N'))"
        $output = Join-Path $tmp 'output'
        try {
            New-Item -Path (Join-Path $output 'app\offline-ai') -ItemType Directory -Force | Out-Null
            $manifestPath = Join-Path $output 'manifest.json'
            @{
                schemaVersion = 1
                packageKind = 'local-build'
                createdAt = '2026-07-11T00:00:00Z'
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8

            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeFalse

            @{
                schemaVersion = 1
                packageKind = 'local-build'
                createdBy = 'untrusted-script'
                createdAt = '2026-07-11T00:00:00Z'
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeFalse

            foreach ($relative in @('installers', 'ollama-models\blobs', 'ollama-models\manifests')) {
                New-Item -Path (Join-Path $output $relative) -ItemType Directory -Force | Out-Null
            }
            @{
                schemaVersion = 1
                packageKind = 'foreign-artifact'
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeFalse

            @{
                schemaVersion = 1
                packageKind = ''
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeFalse

            @{
                schemaVersion = 1
                createdAt = '2026-07-01T00:00:00Z'
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeTrue

            @{
                schemaVersion = 1
                packageKind = 'local-build'
                createdBy = $script:LocalBuildCreatedBy
                createdAt = '2026-07-11T00:00:00Z'
            } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            Test-LocalBuildPackageRoot -Path $output -PackageRoot $tmp | Should -BeTrue
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "rejects staged third-party payloads from a public source artifact" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-public-guard-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path (Join-Path $tmp 'app\offline-ai') -ItemType Directory -Force | Out-Null
            @{ schemaVersion = 2; packageKind = 'public-source'; installable = $false; models = @() } |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $tmp 'manifest.json') -Encoding UTF8
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Not -Throw

            New-Item -Path (Join-Path $tmp 'installers') -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $tmp 'installers\sentinel.exe') -Value 'not-a-real-binary' -Encoding ASCII
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*third-party payload directory*'

            Remove-Item -LiteralPath (Join-Path $tmp 'installers') -Recurse -Force
            Set-Content -LiteralPath (Join-Path $tmp '.env') -Value 'SECRET_VALUE=sentinel' -Encoding ASCII
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*sensitive file*'
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "rejects empty licenses and placeholder support URLs before public release" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-public-ready-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            foreach ($name in @('LICENSE', 'THIRD-PARTY-NOTICES.md', 'README.md', 'SUPPORT.md', 'UNINSTALL.md', 'VERSION')) {
                Set-Content -LiteralPath (Join-Path $tmp $name) -Value '' -Encoding UTF8
            }
            Set-Content -LiteralPath (Join-Path $tmp 'VERSION') -Value '0.1.0' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'THIRD-PARTY-NOTICES.md') -Value 'notices' -Encoding UTF8
            @{
                schemaVersion = 1
                productVersion = '0.1.0'
                supportUrl = 'https://example.com/support'
                licenseSha256 = ''
                approvedBy = 'owner'
                approvedAt = '2026-07-12T00:00:00Z'
                licenseApproved = $true
                redistributionApproved = $true
            } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $tmp 'release-metadata.json') -Encoding UTF8

            { Assert-PublicReleaseReady -AppRoot $tmp } | Should -Throw '*LICENSE must not be empty*'

            Set-Content -LiteralPath (Join-Path $tmp 'LICENSE') -Value 'Approved test license text' -Encoding UTF8
            $metadata = Get-Content -LiteralPath (Join-Path $tmp 'release-metadata.json') -Raw | ConvertFrom-Json
            $metadata.licenseSha256 = Get-OfflineAiSha256 -Path (Join-Path $tmp 'LICENSE')
            $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $tmp 'release-metadata.json') -Encoding UTF8
            { Assert-PublicReleaseReady -AppRoot $tmp } | Should -Throw '*actual HTTPS URL*'

            $metadata.supportUrl = 'https://github.com/example-project/issues'
            $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $tmp 'release-metadata.json') -Encoding UTF8
            { Assert-PublicReleaseReady -AppRoot $tmp } | Should -Not -Throw
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "builds a public source artifact without cached installers, models, optional payloads, or user documents" {
        if (-not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because "download-offline-package.ps1 targets Windows 11 x64"
            return
        }
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-public-build-$([guid]::NewGuid().ToString('N'))"
        $app = Join-Path $tmp 'offline-ai'
        $internal = Join-Path $app '_internal'
        $packageRoot = Join-Path $app 'offline-package'
        $output = Join-Path $packageRoot 'public-source'
        try {
            foreach ($path in @(
                $internal,
                (Join-Path $packageRoot 'installers'),
                (Join-Path $packageRoot 'optional-components\reranker'),
                (Join-Path $app 'skill-source'),
                (Join-Path $app 'infographic')
            )) {
                New-Item -Path $path -ItemType Directory -Force | Out-Null
            }
            foreach ($name in @('download-offline-package.ps1', 'OfflineAi.Common.psm1', 'OfflineAi.ModelSelection.psm1', 'OllamaRegistryDownloader.psm1')) {
                Copy-Item -LiteralPath (Join-Path $script:OfflineAiRoot "_internal\$name") -Destination (Join-Path $internal $name)
            }
            Set-Content -LiteralPath (Join-Path $internal 'install-offline.ps1') -Value '# public source fixture' -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $app 'install-offline.bat') -Value '@echo off' -Encoding ASCII
            foreach ($name in @('README.md', 'SUPPORT.md', 'UNINSTALL.md', 'THIRD-PARTY-NOTICES.md')) {
                Set-Content -LiteralPath (Join-Path $app $name) -Value "$name fixture" -Encoding UTF8
            }
            Set-Content -LiteralPath (Join-Path $app 'VERSION') -Value '0.1.0' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $app 'LICENSE') -Value 'Approved fixture license text' -Encoding UTF8
            $licenseHash = (Get-FileHash -LiteralPath (Join-Path $app 'LICENSE') -Algorithm SHA256).Hash.ToLowerInvariant()
            @{
                schemaVersion = 1
                productVersion = '0.1.0'
                supportUrl = 'https://github.com/example-project/issues'
                licenseSha256 = $licenseHash
                approvedBy = 'fixture-owner'
                approvedAt = '2026-07-12T00:00:00Z'
                licenseApproved = $true
                redistributionApproved = $true
            } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $app 'release-metadata.json') -Encoding UTF8
            Copy-Item -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\download-manifest.json') -Destination (Join-Path $internal 'download-manifest.json')
            @{
                schema_version = '1.0'
                entries = @(
                    @{ pattern = 'README.md'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'SUPPORT.md'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'UNINSTALL.md'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'THIRD-PARTY-NOTICES.md'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'VERSION'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'LICENSE'; distribution = 'release-only'; reason = 'fixture'; boundary_row = 'user-built-transport' }
                    @{ pattern = 'release-metadata.json'; distribution = 'both'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'install-offline.bat'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/install-offline.ps1'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/download-offline-package.ps1'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/OfflineAi.Common.psm1'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/OfflineAi.ModelSelection.psm1'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/OllamaRegistryDownloader.psm1'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/download-manifest.json'; distribution = 'both'; reason = 'fixture'; boundary_row = 'runtime-app-payload' }
                    @{ pattern = '_internal/distribution-policy.json'; distribution = 'public-source'; reason = 'fixture'; boundary_row = 'public-source-artifact' }
                    @{ pattern = 'skill-source/**'; distribution = 'excluded'; reason = 'fixture'; boundary_row = 'generated-excluded' }
                    @{ pattern = 'infographic/**'; distribution = 'excluded'; reason = 'fixture'; boundary_row = 'generated-excluded' }
                    @{ pattern = 'offline-package/**'; distribution = 'excluded'; reason = 'fixture'; boundary_row = 'generated-excluded' }
                )
            } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $internal 'distribution-policy.json') -Encoding UTF8

            $sentinel = 'DO_NOT_COPY_SENTINEL'
            Set-Content -LiteralPath (Join-Path $packageRoot 'installers\cached.exe') -Value $sentinel -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $packageRoot 'optional-components\reranker\payload.bin') -Value $sentinel -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $app 'skill-source\private.txt') -Value $sentinel -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $app 'infographic\internal.txt') -Value $sentinel -Encoding ASCII

            $result = & powershell.exe -ExecutionPolicy Bypass -NoProfile -File (Join-Path $internal 'download-offline-package.ps1') `
                -OutputPath $output `
                -DownloadManifestPath (Join-Path $internal 'download-manifest.json') `
                -PublicRelease `
                -Yes 2>&1
            $resultText = $result | Out-String
            $LASTEXITCODE | Should -Be 0 -Because $resultText
            foreach ($relative in @('installers', 'ollama-models', 'optional-components')) {
                Test-Path -LiteralPath (Join-Path $output $relative) | Should -BeFalse
            }
            $matches = @(
                Get-ChildItem -LiteralPath $output -File -Recurse |
                    Select-String -SimpleMatch $sentinel -List
            )
            $matches.Count | Should -Be 0
            Test-Path -LiteralPath (Join-Path $output 'checksums.sha256') -PathType Leaf | Should -BeTrue
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "copies only blobs referenced by the selected Ollama model manifests" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-targeted-models-$([guid]::NewGuid().ToString('N'))"
        $source = Join-Path $tmp 'source'
        $destination = Join-Path $tmp 'destination'
        $hashA = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        $hashB = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        $hashUnused = 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
        try {
            $qwenManifest = Join-Path $source 'manifests\registry.ollama.ai\library\qwen3.5\9b'
            $embedManifest = Join-Path $source 'manifests\registry.ollama.ai\library\bge-m3\latest'
            New-Item -Path (Split-Path -Parent $qwenManifest) -ItemType Directory -Force | Out-Null
            New-Item -Path (Split-Path -Parent $embedManifest) -ItemType Directory -Force | Out-Null
            New-Item -Path (Join-Path $source 'blobs') -ItemType Directory -Force | Out-Null
            @{ config = @{ digest = "sha256:$hashA" }; layers = @() } |
                ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $qwenManifest -Encoding UTF8
            @{ config = @{ digest = "sha256:$hashB" }; layers = @() } |
                ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $embedManifest -Encoding UTF8
            foreach ($hash in @($hashA, $hashB, $hashUnused)) {
                Set-Content -LiteralPath (Join-Path $source "blobs\sha256-$hash") -Value $hash -Encoding ASCII
            }

            Copy-TargetOllamaModels -SourceModelsRoot $source -DestinationModelsRoot $destination -Models @('qwen3.5:9b', 'bge-m3')

            Test-Path -LiteralPath (Join-Path $destination "blobs\sha256-$hashA") | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $destination "blobs\sha256-$hashB") | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $destination "blobs\sha256-$hashUnused") | Should -BeFalse
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "preserves metacharacter queries without executing injected cmd fragments" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-search-bat-$([guid]::NewGuid().ToString('N'))"
        $oldPath = $env:PATH
        $oldCapture = $env:SEARCH_CAPTURE
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $capture = Join-Path $tmp 'captured.txt'
            $marker = Join-Path $tmp 'injected.txt'
            $input = Join-Path $tmp 'input.txt'
            $query = 'x" & echo injected>"' + $marker + '" & rem " | bang!'
            [System.IO.File]::WriteAllLines($input, @($query, 'n', 'x'), [System.Text.Encoding]::ASCII)
            $fakePythonSource = @'
using System;
using System.IO;

public static class FakePython
{
    public static int Main(string[] args)
    {
        string executable = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0]);
        if (string.Equals(executable, "curl", StringComparison.OrdinalIgnoreCase)) return 0;
        if (args.Length > 0 && args[0] == "-c") return 0;
        if (args.Length < 3) return 2;
        File.Copy(args[2], Environment.GetEnvironmentVariable("SEARCH_CAPTURE"), true);
        return 0;
    }
}
'@
            $compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
            if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
                Set-ItResult -Skipped -Because 'Windows .NET Framework C# compiler is unavailable'
                return
            }
            $sourcePath = Join-Path $tmp 'FakePython.cs'
            $fakePythonPath = Join-Path $tmp 'python.exe'
            [System.IO.File]::WriteAllText($sourcePath, $fakePythonSource, [System.Text.UTF8Encoding]::new($false))
            $compilerOutput = & $compiler /nologo /target:exe "/out:$fakePythonPath" $sourcePath 2>&1
            $LASTEXITCODE | Should -Be 0 -Because ($compilerOutput -join "`n")
            Copy-Item -LiteralPath $fakePythonPath -Destination (Join-Path $tmp 'curl.exe')
            $env:PATH = "$tmp;$oldPath"
            $env:SEARCH_CAPTURE = $capture
            $cmdLine = 'call "' + $script:SearchBatPath + '" < "' + $input + '"'

            $output = & cmd.exe /d /c $cmdLine 2>&1
            $exitCode = $LASTEXITCODE

            $exitCode | Should -Be 0 -Because ($output -join "`n")
            Test-Path -LiteralPath $marker | Should -BeFalse
            Test-Path -LiteralPath $capture | Should -BeTrue -Because ($output -join "`n")
            [System.IO.File]::ReadAllText($capture, [System.Text.Encoding]::UTF8) | Should -Be $query
        } finally {
            $env:PATH = $oldPath
            $env:SEARCH_CAPTURE = $oldCapture
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "propagates success and failure through download-package.bat" {
        if (-not $script:IsX64Runtime) {
            Set-ItResult -Skipped -Because "download-package.bat delegates to a Windows 11 x64 package builder"
            return
        }
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-bat-exit-$([guid]::NewGuid().ToString('N'))"
        $outputPath = Join-Path $script:OfflineAiRoot "offline-package\dryrun-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $input = Join-Path $tmp 'input.txt'
            [System.IO.File]::WriteAllLines($input, @('x'), [System.Text.Encoding]::ASCII)

            $successCommand = 'call "' + $script:DownloadBatPath + '" -OutputPath "' + $outputPath + '" -SkipInstallers -SkipModels -DryRun < "' + $input + '"'
            $successOutput = & cmd.exe /d /c $successCommand 2>&1
            $successExit = $LASTEXITCODE
            $successExit | Should -Be 0 -Because ($successOutput -join "`n")

            $failureCommand = 'call "' + $script:DownloadBatPath + '" -OutputPath "' + $script:OfflineAiRoot + '" -SkipInstallers -SkipModels -DryRun < "' + $input + '"'
            $failureOutput = & cmd.exe /d /c $failureCommand 2>&1
            $failureExit = $LASTEXITCODE
            $failureExit | Should -Not -Be 0 -Because ($failureOutput -join "`n")
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $outputPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
