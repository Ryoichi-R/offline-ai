BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:RegistryDownloaderPath = Join-Path $script:OfflineAiRoot '_internal\OllamaRegistryDownloader.psm1'
    $script:DownloadPackagePath = Join-Path $script:OfflineAiRoot '_internal\download-offline-package.ps1'
    $script:DownloadManifestPath = Join-Path $script:OfflineAiRoot '_internal\download-manifest.json'
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    $script:IsWindowsRuntime = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    Import-Module $script:RegistryDownloaderPath -Force
    Import-Module $script:CommonModulePath -Force

    $tokens = $null
    $parseErrors = $null
    $downloadAst = [System.Management.Automation.Language.Parser]::ParseFile(
        $script:DownloadPackagePath,
        [ref]$tokens,
        [ref]$parseErrors
    )
    $parseErrors.Count | Should -Be 0
    foreach ($functionName in @('Download-FileVerified', 'Confirm-OfflineAiUnsupportedChatModel')) {
        $downloadFunctionAst = $downloadAst.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                $node.Name -eq $functionName
        }, $true)
        $downloadFunctionAst | Should -Not -BeNullOrEmpty
        . ([scriptblock]::Create($downloadFunctionAst.Extent.Text))
    }

    function Get-TestSha256 {
        param([Parameter(Mandatory = $true)][byte[]]$Bytes)
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
    }

    function New-TestRegistryFixture {
        param([Parameter(Mandatory = $true)][string]$Root)

        $repoRoot = Join-Path (Join-Path (Join-Path $Root 'v2') 'library') 'bge-m3'
        New-Item -Path (Join-Path $repoRoot 'manifests') -ItemType Directory -Force | Out-Null
        New-Item -Path (Join-Path $repoRoot 'blobs') -ItemType Directory -Force | Out-Null

        $configBytes = [System.Text.Encoding]::UTF8.GetBytes('{"model":"dummy-config"}')
        $layerBytes = [System.Text.Encoding]::UTF8.GetBytes('dummy model blob')
        $licenseBytes = [System.Text.Encoding]::UTF8.GetBytes('Apache License 2.0 fixture')
        $configHash = Get-TestSha256 -Bytes $configBytes
        $layerHash = Get-TestSha256 -Bytes $layerBytes
        $licenseHash = Get-TestSha256 -Bytes $licenseBytes

        [System.IO.File]::WriteAllBytes((Join-Path (Join-Path $repoRoot 'blobs') "sha256-$configHash"), $configBytes)
        [System.IO.File]::WriteAllBytes((Join-Path (Join-Path $repoRoot 'blobs') "sha256-$layerHash"), $layerBytes)
        [System.IO.File]::WriteAllBytes((Join-Path (Join-Path $repoRoot 'blobs') "sha256-$licenseHash"), $licenseBytes)

        $manifest = @{
            schemaVersion = 2
            mediaType = 'application/vnd.docker.distribution.manifest.v2+json'
            config = @{
                mediaType = 'application/vnd.ollama.image.model'
                digest = "sha256:$configHash"
                size = $configBytes.Length
            }
            layers = @(
                @{
                    mediaType = 'application/vnd.ollama.image.layer'
                    digest = "sha256:$layerHash"
                    size = $layerBytes.Length
                },
                @{
                    mediaType = 'application/vnd.ollama.image.license'
                    digest = "sha256:$licenseHash"
                    size = $licenseBytes.Length
                }
            )
        } | ConvertTo-Json -Depth 8
        $manifestPath = Join-Path (Join-Path $repoRoot 'manifests') 'latest'
        Set-Content -LiteralPath $manifestPath -Value $manifest -Encoding UTF8

        return [PSCustomObject]@{
            configHash = $configHash
            layerHash = $layerHash
            licenseHash = $licenseHash
            manifestDigest = "sha256:$((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant())"
            largestBlobBytes = [Math]::Max([Math]::Max($configBytes.Length, $layerBytes.Length), $licenseBytes.Length)
        }
    }
}

Describe "offline-ai download-only package helpers" {
    It 'requires a second affirmative interactive confirmation for a RAM-unsupported chat model' {
        Mock Read-Host { 'yes' }
        Confirm-OfflineAiUnsupportedChatModel -ChatModel 'qwen3.5:27b' | Should -BeTrue
        Should -Invoke Read-Host -Times 1 -Exactly -ParameterFilter { $Prompt -match '2回目の確認' }
    }

    It 'rejects a negative second interactive confirmation for a RAM-unsupported chat model' {
        Mock Read-Host { 'n' }
        { Confirm-OfflineAiUnsupportedChatModel -ChatModel 'qwen3.5:27b' } | Should -Throw '*2回目の確認*'
    }

    It "resolves Ollama model names to repository and tag" {
        $qwen = Resolve-OllamaModelReference -Name 'qwen3.5:9b'
        $qwen.repository | Should -Be 'library/qwen3.5'
        $qwen.tag | Should -Be '9b'

        $embed = Resolve-OllamaModelReference -Name 'bge-m3'
        $embed.repository | Should -Be 'library/bge-m3'
        $embed.tag | Should -Be 'latest'
    }

    It "converts sha256 digests to Ollama blob file names" {
        $hash = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        Convert-OllamaDigestToBlobFileName -Digest "sha256:$hash" | Should -Be "sha256-$hash"
    }

    It "extracts config and layer descriptors from a manifest" {
        $hashA = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        $hashB = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        $manifest = [PSCustomObject]@{
            config = [PSCustomObject]@{ digest = "sha256:$hashA"; size = 10; mediaType = 'config/type' }
            layers = @([PSCustomObject]@{ digest = "sha256:$hashB"; size = 20; mediaType = 'layer/type' })
        }

        $items = @(Get-OllamaManifestDescriptors -ManifestJson $manifest)
        $items.Count | Should -Be 2
        $items[0].kind | Should -Be 'config'
        $items[1].kind | Should -Be 'layer'
    }

    It "creates Ollama offline model manifests and blobs from a fixture registry" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-registry-test-$([guid]::NewGuid().ToString('N'))"
        $registryRoot = Join-Path $tmp 'registry'
        $modelsRoot = Join-Path (Join-Path $tmp 'package') 'ollama-models'
        try {
            $fixture = New-TestRegistryFixture -Root $registryRoot
            $result = Save-OllamaModelFromRegistry -ModelName 'bge-m3' -DestinationModelsRoot $modelsRoot -RegistryBaseUrl 'https://registry.ollama.ai' -RegistryRoot $registryRoot `
                -ExpectedManifestDigest $fixture.manifestDigest `
                -ExpectedConfigDigest "sha256:$($fixture.configHash)" `
                -ExpectedLicenseLayerDigest "sha256:$($fixture.licenseHash)"

            $result.name | Should -Be 'bge-m3'
            $result.repository | Should -Be 'library/bge-m3'
            $result.tag | Should -Be 'latest'
            $result.blobCount | Should -Be 3
            $result.largestBlobBytes | Should -Be $fixture.largestBlobBytes
            $manifestPath = Join-Path (Join-Path (Join-Path (Join-Path (Join-Path $modelsRoot 'manifests') 'registry.ollama.ai') 'library') 'bge-m3') 'latest'
            Test-Path $manifestPath | Should -BeTrue
            Test-Path (Join-Path (Join-Path $modelsRoot 'blobs') "sha256-$($fixture.configHash)") | Should -BeTrue
            Test-Path (Join-Path (Join-Path $modelsRoot 'blobs') "sha256-$($fixture.layerHash)") | Should -BeTrue
            Test-Path (Join-Path (Join-Path $modelsRoot 'blobs') "sha256-$($fixture.licenseHash)") | Should -BeTrue
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects manifest tag drift before creating model output" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-registry-drift-$([guid]::NewGuid().ToString('N'))"
        $registryRoot = Join-Path $tmp 'registry'
        $modelsRoot = Join-Path (Join-Path $tmp 'package') 'ollama-models'
        try {
            $fixture = New-TestRegistryFixture -Root $registryRoot
            { Save-OllamaModelFromRegistry -ModelName 'bge-m3' -DestinationModelsRoot $modelsRoot -RegistryBaseUrl 'https://registry.ollama.ai' -RegistryRoot $registryRoot -ExpectedManifestDigest "sha256:$('0' * 64)" -ExpectedLicenseLayerDigest "sha256:$($fixture.licenseHash)" } |
                Should -Throw '*manifest digest drift*'
            Test-Path -LiteralPath $modelsRoot | Should -BeFalse
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "does not create output files during registry DryRun" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-registry-dryrun-$([guid]::NewGuid().ToString('N'))"
        $registryRoot = Join-Path $tmp 'registry'
        $modelsRoot = Join-Path (Join-Path $tmp 'package') 'ollama-models'
        try {
            $null = New-TestRegistryFixture -Root $registryRoot
            $result = Save-OllamaModelFromRegistry -ModelName 'bge-m3' -DestinationModelsRoot $modelsRoot -RegistryBaseUrl 'https://registry.ollama.ai' -RegistryRoot $registryRoot -DryRun

            $result.name | Should -Be 'bge-m3'
            Test-Path -LiteralPath $modelsRoot | Should -BeFalse
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "keeps expectedSha256 verification wired for installer downloads" {
        $scriptContent = Get-Content -LiteralPath $script:DownloadPackagePath -Raw
        $manifest = Get-Content -LiteralPath $script:DownloadManifestPath -Raw | ConvertFrom-Json

        $scriptContent | Should -Match 'ExpectedSha256'
        $scriptContent | Should -Match 'expectedSha256 と一致しません'
        foreach ($tool in @($manifest.tools)) {
            $tool.PSObject.Properties.Name | Should -Contain 'expectedSha256'
        }
    }

    It "pins required installers by hash, product version, and signer" {
        $manifest = Get-Content -LiteralPath $script:DownloadManifestPath -Raw | ConvertFrom-Json
        foreach ($tool in @($manifest.tools | Where-Object required)) {
            $tool.expectedSha256 | Should -Match '^[0-9a-f]{64}$'
            $tool.expectedProductVersion | Should -Not -BeNullOrEmpty
            $tool.expectedSignerSubjectContains | Should -Not -BeNullOrEmpty
        }
        $content = Get-Content -LiteralPath $script:DownloadPackagePath -Raw
        $content | Should -Match 'Get-AuthenticodeSignature'
        $content | Should -Match 'expectedProductVersion'
    }

    It "rejects an existing installer whose hash differs from expectedSha256" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-hash-existing-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $destination = Join-Path $tmp 'installer.exe'
            Set-Content -LiteralPath $destination -Value 'unexpected' -Encoding ASCII

            { Download-FileVerified -Id 'fixture' -Url 'https://example.invalid/installer.exe' -Destination $destination -ExpectedSha256 ('0' * 64) } |
                Should -Throw '*expectedSha256*'
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "removes a downloaded part file when expectedSha256 verification fails" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-hash-download-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $destination = Join-Path $tmp 'installer.exe'
            Mock Invoke-WebRequest {
                param($Uri, $OutFile)
                Set-Content -LiteralPath $OutFile -Value 'unexpected' -Encoding ASCII
            }

            { Download-FileVerified -Id 'fixture' -Url 'https://example.invalid/installer.exe' -Destination $destination -ExpectedSha256 ('0' * 64) } |
                Should -Throw '*expectedSha256*'
            Test-Path -LiteralPath $destination | Should -BeFalse
            @(Get-ChildItem -LiteralPath $tmp -Filter '*.part-*' -File).Count | Should -Be 0
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "retries a blob after digest mismatch and accepts a later valid download" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-blob-retry-$([guid]::NewGuid().ToString('N'))"
        $goodBytes = [System.Text.Encoding]::UTF8.GetBytes('valid blob')
        $hash = Get-TestSha256 -Bytes $goodBytes
        $attempts = [System.Collections.Generic.List[int]]::new()
        try {
            Mock Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader {
                param($Destination)
                $attempts.Add($attempts.Count + 1)
                $bytes = if ($attempts.Count -eq 1) {
                    [System.Text.Encoding]::UTF8.GetBytes('bad blob')
                } else {
                    $goodBytes
                }
                [System.IO.File]::WriteAllBytes($Destination, $bytes)
            }
            Mock Start-Sleep -ModuleName OllamaRegistryDownloader {}

            $result = Save-OllamaBlob -RegistryBaseUrl 'https://registry.example.invalid' -Repository 'library/test' -Digest "sha256:$hash" -DestinationModelsRoot $tmp -RetryCount 3

            $result.status | Should -Be 'Saved'
            $attempts.Count | Should -Be 2
            Should -Invoke -CommandName Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader -Times 2 -Exactly
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "fails after the configured number of blob digest mismatches" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-blob-retry-fail-$([guid]::NewGuid().ToString('N'))"
        $goodBytes = [System.Text.Encoding]::UTF8.GetBytes('valid blob')
        $hash = Get-TestSha256 -Bytes $goodBytes
        try {
            Mock Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader {
                param($Destination)
                [System.IO.File]::WriteAllText($Destination, 'always bad')
            }
            Mock Start-Sleep -ModuleName OllamaRegistryDownloader {}

            { Save-OllamaBlob -RegistryBaseUrl 'https://registry.example.invalid' -Repository 'library/test' -Digest "sha256:$hash" -DestinationModelsRoot $tmp -RetryCount 3 } |
                Should -Throw '*Blob digest mismatch*'
            Should -Invoke -CommandName Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader -Times 3 -Exactly
            @(Get-ChildItem -LiteralPath $tmp -Filter '*.part-*' -File -Recurse -ErrorAction SilentlyContinue).Count | Should -Be 0
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "does not redownload a verified blob when final placement fails" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-blob-move-fail-$([guid]::NewGuid().ToString('N'))"
        $goodBytes = [System.Text.Encoding]::UTF8.GetBytes('valid blob')
        $hash = Get-TestSha256 -Bytes $goodBytes
        try {
            Mock Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader {
                param($Destination)
                [System.IO.File]::WriteAllBytes($Destination, $goodBytes)
            }
            Mock Move-Item -ModuleName OllamaRegistryDownloader {
                throw 'simulated placement failure'
            }

            { Save-OllamaBlob -RegistryBaseUrl 'https://registry.example.invalid' -Repository 'library/test' -Digest "sha256:$hash" -DestinationModelsRoot $tmp -RetryCount 3 } |
                Should -Throw '*simulated placement failure*'
            Should -Invoke -CommandName Save-OllamaRegistryContentToFile -ModuleName OllamaRegistryDownloader -Times 1 -Exactly
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "keeps the reranker optional unless explicitly included" {
        $scriptContent = Get-Content -LiteralPath $script:DownloadPackagePath -Raw
        $manifest = Get-Content -LiteralPath $script:DownloadManifestPath -Raw | ConvertFrom-Json

        $reranker = @($manifest.models) |
            Where-Object { $_.PSObject.Properties['includeFlag'] -and $_.includeFlag -eq 'IncludeReranker' } |
            Select-Object -First 1

        @($manifest.tools | Where-Object { $_.id -in @('pdftotext', 'git-for-windows', 'structured-parser') }).Count | Should -Be 0
        $reranker.manualOnly | Should -BeTrue
        $reranker.stagingDirectory | Should -Be 'reranker'
        $reranker.packageRelativePath | Should -Be 'optional-components/reranker'
        $reranker.activation | Should -Be 'transport-only'
        $reranker.licenseReviewRequired | Should -BeTrue
        $scriptContent | Should -Match 'Test-OptionalComponentSelected'
        $scriptContent | Should -Match 'Copy-OfflineAiOptionalComponent'
        $scriptContent | Should -Match 'optionalComponents\s*=\s*\$optionalComponentResults'
        $scriptContent | Should -Match 'OptionalSkipped'
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath (Join-Path $script:OfflineAiRoot '_internal\distribution-policy.json')
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '_internal/optional-components/reranker/model.bin' | Should -Be 'excluded'
    }

    It "uses retry and progress suppression for registry downloads" {
        $content = Get-Content -LiteralPath $script:RegistryDownloaderPath -Raw
        $content | Should -Match 'Invoke-OllamaRegistryWebRequest'
        $content | Should -Match "ProgressPreference = 'SilentlyContinue'"
        $content | Should -Match 'Tls12'
    }
}
