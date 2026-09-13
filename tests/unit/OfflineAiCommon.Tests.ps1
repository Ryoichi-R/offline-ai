BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    Import-Module $script:CommonModulePath -Force
}

Describe "OfflineAi.Common path and checksum helpers" {
    It "allows a target under the guarded root" {
        $root = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-common-root"
        $target = Join-Path $root "child\file.txt"
        $resolved = Assert-OfflineAiPathUnderRoot -Root $root -Target $target
        $resolved | Should -Match ([regex]::Escape("offline-ai-common-root"))
    }

    It "rejects a target outside the guarded root" {
        $root = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-common-root"
        $target = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-common-outside\file.txt"
        { Assert-OfflineAiPathUnderRoot -Root $root -Target $target } | Should -Throw "*outside root*"
    }

    It "rejects a case-different sibling path on case-sensitive platforms" {
        if ([System.IO.Path]::DirectorySeparatorChar -eq '\') {
            Set-ItResult -Skipped -Because "Windows path comparison is case-insensitive"
            return
        }
        $root = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-case-root"
        $target = Join-Path ([System.IO.Path]::GetTempPath()) "OFFLINE-AI-CASE-ROOT"

        { Assert-OfflineAiPathUnderRoot -Root $root -Target $target } | Should -Throw "*outside root*"
    }

    It "validates checksums.sha256 entries" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-checksum-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $file = Join-Path $tmp "file.txt"
            Set-Content -LiteralPath $file -Value "ok" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $file
            Set-Content -LiteralPath (Join-Path $tmp "checksums.sha256") -Value "$hash *file.txt" -Encoding UTF8

            Test-OfflineAiPackageChecksums -Root $tmp | Should -BeTrue
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "fails when a checksum entry does not match the file content" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-checksum-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $tmp "file.txt") -Value "changed" -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp "checksums.sha256") -Value "$('0' * 64) *file.txt" -Encoding UTF8

            { Test-OfflineAiPackageChecksums -Root $tmp } | Should -Throw "*Checksum mismatch*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects package files missing from the checksum inventory" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-checksum-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $listed = Join-Path $tmp "listed.txt"
            Set-Content -LiteralPath $listed -Value "listed" -Encoding ASCII
            New-Item -Path (Join-Path $tmp "optional-components") -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $tmp "optional-components/injected.txt") -Value "injected" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $listed
            Set-Content -LiteralPath (Join-Path $tmp "checksums.sha256") -Value "$hash *listed.txt" -Encoding UTF8

            { Test-OfflineAiPackageChecksums -Root $tmp } | Should -Throw "*not declared*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "allows an installed optional copy only when it matches the inventoried source" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-checksum-$([guid]::NewGuid().ToString('N'))"
        try {
            $source = Join-Path $tmp "optional-components/reranker/model.bin"
            $installed = Join-Path $tmp "app/offline-ai/_internal/optional-components/reranker/model.bin"
            New-Item -Path (Split-Path -Parent $source) -ItemType Directory -Force | Out-Null
            New-Item -Path (Split-Path -Parent $installed) -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath $source -Value "same" -Encoding ASCII
            Set-Content -LiteralPath $installed -Value "same" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $source
            Set-Content -LiteralPath (Join-Path $tmp "checksums.sha256") -Value "$hash *optional-components\reranker\model.bin" -Encoding UTF8

            Test-OfflineAiPackageChecksums -Root $tmp | Should -BeTrue
            Set-Content -LiteralPath $installed -Value "different" -Encoding ASCII
            { Test-OfflineAiPackageChecksums -Root $tmp } | Should -Throw "*does not match*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }
}

Describe "OfflineAi.Common Ollama model path helper" {
    It "uses an explicit Ollama models path first" {
        $old = $env:OLLAMA_MODELS
        try {
            $env:OLLAMA_MODELS = Join-Path ([System.IO.Path]::GetTempPath()) "env-models"
            $explicit = Join-Path ([System.IO.Path]::GetTempPath()) "explicit-models"
            Get-OfflineAiOllamaModelsPath -ExplicitPath $explicit | Should -Be (Resolve-OfflineAiFullPath -Path $explicit)
        } finally {
            $env:OLLAMA_MODELS = $old
        }
    }

    It "uses OLLAMA_MODELS when no explicit path is provided" {
        $old = $env:OLLAMA_MODELS
        try {
            $env:OLLAMA_MODELS = Join-Path ([System.IO.Path]::GetTempPath()) "env-models"
            Get-OfflineAiOllamaModelsPath | Should -Be (Resolve-OfflineAiFullPath -Path $env:OLLAMA_MODELS)
        } finally {
            $env:OLLAMA_MODELS = $old
        }
    }
}

Describe "OfflineAi.Common no-conflict copy helper" {
    It "skips an existing destination when hashes match" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-copy-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $source = Join-Path $tmp "source.txt"
            $dest = Join-Path $tmp "dest.txt"
            Set-Content -LiteralPath $source -Value "same" -Encoding ASCII
            Set-Content -LiteralPath $dest -Value "same" -Encoding ASCII

            Copy-OfflineAiFileNoConflict -Source $source -Destination $dest | Should -Be "SkippedSameHash"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects an existing destination when hashes differ" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-copy-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $source = Join-Path $tmp "source.txt"
            $dest = Join-Path $tmp "dest.txt"
            Set-Content -LiteralPath $source -Value "source" -Encoding ASCII
            Set-Content -LiteralPath $dest -Value "dest" -Encoding ASCII

            { Copy-OfflineAiFileNoConflict -Source $source -Destination $dest } | Should -Throw "*different content*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }
}

Describe "OfflineAi.Common model license export" {
    It "exports a verified Ollama license layer into a readable legal inventory" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-license-$([guid]::NewGuid().ToString('N'))"
        try {
            $modelsRoot = Join-Path $tmp "models"
            $manifestPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\qwen3.5\9b"
            $blobRoot = Join-Path $modelsRoot "blobs"
            $legalRoot = Join-Path $tmp "legal\models"
            New-Item -Path (Split-Path -Parent $manifestPath) -ItemType Directory -Force | Out-Null
            New-Item -Path $blobRoot -ItemType Directory -Force | Out-Null

            $licenseSource = Join-Path $tmp "license.txt"
            Set-Content -LiteralPath $licenseSource -Value "Apache License 2.0" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $licenseSource
            Copy-Item -LiteralPath $licenseSource -Destination (Join-Path $blobRoot "sha256-$hash")
            @{
                schemaVersion = 2
                layers = @(@{
                    mediaType = "application/vnd.ollama.image.license"
                    digest = "sha256:$hash"
                    size = (Get-Item -LiteralPath $licenseSource).Length
                })
            } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

            $result = @(Export-OfflineAiModelLicenses -ModelsRoot $modelsRoot -DestinationRoot $legalRoot)

            $result.Count | Should -Be 1
            $result[0].licenseDigest | Should -Be "sha256:$hash"
            $exported = Join-Path $legalRoot $result[0].relativePath
            Test-Path -LiteralPath $exported -PathType Leaf | Should -BeTrue
            (Get-OfflineAiSha256 -Path $exported) | Should -Be $hash
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects reparse points before validating an install package" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-reparse-$([guid]::NewGuid().ToString('N'))"
        $outside = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-reparse-outside-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            New-Item -Path $outside -ItemType Directory -Force | Out-Null
            $listed = Join-Path $tmp "listed.txt"
            Set-Content -LiteralPath $listed -Value "listed" -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $outside "outside.txt") -Value "outside" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $listed
            Set-Content -LiteralPath (Join-Path $tmp "checksums.sha256") -Value "$hash *listed.txt" -Encoding UTF8
            try {
                $linkItemType = if ($IsWindows) { "Junction" } else { "SymbolicLink" }
                New-Item -Path (Join-Path $tmp "linked") -ItemType $linkItemType -Target $outside -ErrorAction Stop | Out-Null
            } catch {
                Set-ItResult -Skipped -Because "junction creation is unavailable: $($_.Exception.Message)"
                return
            }

            { Test-OfflineAiPackageChecksums -Root $tmp } | Should -Throw "*Reparse point*"
        } finally {
            foreach ($path in @($tmp, $outside)) {
                if (Test-Path -LiteralPath $path) {
                    Remove-Item -LiteralPath $path -Recurse -Force
                }
            }
        }
    }

    It "fails closed when an Ollama model manifest has no license layer" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-license-$([guid]::NewGuid().ToString('N'))"
        try {
            $modelsRoot = Join-Path $tmp "models"
            $manifestPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\unknown\latest"
            New-Item -Path (Split-Path -Parent $manifestPath) -ItemType Directory -Force | Out-Null
            @{ schemaVersion = 2; layers = @() } |
                ConvertTo-Json -Depth 5 |
                Set-Content -LiteralPath $manifestPath -Encoding UTF8

            {
                Export-OfflineAiModelLicenses -ModelsRoot $modelsRoot -DestinationRoot (Join-Path $tmp "legal")
            } | Should -Throw "*no license layer*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "fails closed when the model license blob does not match its digest" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-license-$([guid]::NewGuid().ToString('N'))"
        try {
            $modelsRoot = Join-Path $tmp "models"
            $manifestPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\unknown\latest"
            $blobRoot = Join-Path $modelsRoot "blobs"
            New-Item -Path (Split-Path -Parent $manifestPath) -ItemType Directory -Force | Out-Null
            New-Item -Path $blobRoot -ItemType Directory -Force | Out-Null
            $claimedHash = "0" * 64
            Set-Content -LiteralPath (Join-Path $blobRoot "sha256-$claimedHash") -Value "different" -Encoding ASCII
            @{
                schemaVersion = 2
                layers = @(@{
                    mediaType = "application/vnd.ollama.image.license"
                    digest = "sha256:$claimedHash"
                    size = (Get-Item -LiteralPath (Join-Path $blobRoot "sha256-$claimedHash")).Length
                })
            } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

            {
                Export-OfflineAiModelLicenses -ModelsRoot $modelsRoot -DestinationRoot (Join-Path $tmp "legal")
            } | Should -Throw "*digest mismatch*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "fails closed when the model license layer size does not match the blob" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-license-$([guid]::NewGuid().ToString('N'))"
        try {
            $modelsRoot = Join-Path $tmp "models"
            $manifestPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\unknown\latest"
            $blobRoot = Join-Path $modelsRoot "blobs"
            New-Item -Path (Split-Path -Parent $manifestPath) -ItemType Directory -Force | Out-Null
            New-Item -Path $blobRoot -ItemType Directory -Force | Out-Null
            $source = Join-Path $tmp "license.txt"
            Set-Content -LiteralPath $source -Value "MIT" -Encoding ASCII
            $hash = Get-OfflineAiSha256 -Path $source
            Copy-Item -LiteralPath $source -Destination (Join-Path $blobRoot "sha256-$hash")
            @{
                schemaVersion = 2
                layers = @(@{
                    mediaType = "application/vnd.ollama.image.license"
                    digest = "sha256:$hash"
                    size = (Get-Item -LiteralPath $source).Length + 1
                })
            } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

            {
                Export-OfflineAiModelLicenses -ModelsRoot $modelsRoot -DestinationRoot (Join-Path $tmp "legal")
            } | Should -Throw "*size mismatch*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }
}

Describe "OfflineAi.Common optional component copy helper" {
    It "rejects symbolic links in optional component input" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-optional-link-$([guid]::NewGuid().ToString('N'))"
        $sourceRoot = Join-Path $tmp "staging"
        $componentRoot = Join-Path $sourceRoot "reranker"
        $outside = Join-Path $tmp "outside.txt"
        try {
            New-Item -Path $componentRoot -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $componentRoot "model.bin") -Value "fake" -Encoding ASCII
            Set-Content -LiteralPath $outside -Value "outside" -Encoding ASCII
            try {
                New-Item -Path (Join-Path $componentRoot "linked.txt") -ItemType SymbolicLink -Target $outside -ErrorAction Stop | Out-Null
            } catch {
                Set-ItResult -Skipped -Because "symbolic link creation is unavailable: $($_.Exception.Message)"
                return
            }

            {
                Copy-OfflineAiOptionalComponent `
                    -SourceRoot $sourceRoot `
                    -DestinationRoot (Join-Path $tmp "package") `
                    -ComponentId "reranker" `
                    -StagingDirectory "reranker" `
                    -PackageRelativePath "optional-components/reranker" `
                    -RequiredFiles @("model.bin") `
                    -DryRun
            } | Should -Throw "*Reparse point*"
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "copies a validated fake component into its package path" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-optional-$([guid]::NewGuid().ToString('N'))"
        $sourceRoot = Join-Path $tmp "staging"
        $componentRoot = Join-Path $sourceRoot "reranker"
        $destinationRoot = Join-Path $tmp "package"
        try {
            New-Item -Path $componentRoot -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $componentRoot "model.bin") -Value '"{component_dir}\weights.bin" {input} {output_dir}' -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $componentRoot "weights.bin") -Value "fake" -Encoding ASCII

            $result = Copy-OfflineAiOptionalComponent `
                -SourceRoot $sourceRoot `
                -DestinationRoot $destinationRoot `
                -ComponentId "reranker" `
                -StagingDirectory "reranker" `
                -PackageRelativePath "optional-components\reranker" `
                -RequiredFiles @("model.bin")

            $result.status | Should -Be "Copied"
            $result.fileCount | Should -Be 2
            $result.totalBytes | Should -BeGreaterThan 0
            Test-Path -LiteralPath (Join-Path $destinationRoot "optional-components\reranker\model.bin") | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $destinationRoot "optional-components\reranker\weights.bin") | Should -BeTrue
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "validates a component during DryRun without creating package files" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-optional-$([guid]::NewGuid().ToString('N'))"
        $sourceRoot = Join-Path $tmp "staging"
        $componentRoot = Join-Path $sourceRoot "reranker"
        $destinationRoot = Join-Path $tmp "package"
        try {
            New-Item -Path $componentRoot -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $componentRoot "model.bin") -Value "fake" -Encoding ASCII

            $result = Copy-OfflineAiOptionalComponent `
                -SourceRoot $sourceRoot `
                -DestinationRoot $destinationRoot `
                -ComponentId "local-reranker" `
                -StagingDirectory "reranker" `
                -PackageRelativePath "optional-components\reranker" `
                -DryRun

            $result.status | Should -Be "Planned"
            $result.fileCount | Should -Be 1
            Test-Path -LiteralPath $destinationRoot | Should -BeFalse
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects missing required files and sensitive file names" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-optional-$([guid]::NewGuid().ToString('N'))"
        $sourceRoot = Join-Path $tmp "staging"
        $componentRoot = Join-Path $sourceRoot "reranker"
        $destinationRoot = Join-Path $tmp "package"
        try {
            New-Item -Path $componentRoot -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $componentRoot "parser.exe") -Value "fake" -Encoding ASCII

            {
                Copy-OfflineAiOptionalComponent `
                    -SourceRoot $sourceRoot `
                    -DestinationRoot $destinationRoot `
                    -ComponentId "reranker" `
                    -StagingDirectory "reranker" `
                    -PackageRelativePath "optional-components\reranker" `
                    -RequiredFiles @("model.bin")
            } | Should -Throw "*required file*"

            Set-Content -LiteralPath (Join-Path $componentRoot "model.bin") -Value "fake" -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $componentRoot ".env") -Value "blocked" -Encoding ASCII
            {
                Copy-OfflineAiOptionalComponent `
                    -SourceRoot $sourceRoot `
                    -DestinationRoot $destinationRoot `
                    -ComponentId "reranker" `
                    -StagingDirectory "reranker" `
                    -PackageRelativePath "optional-components\reranker" `
                    -RequiredFiles @("model.bin")
            } | Should -Throw "*sensitive file name*"
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }

    It "rejects optional destination traversal and the full sensitive denylist" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-optional-$([guid]::NewGuid().ToString('N'))"
        $sourceRoot = Join-Path $tmp "staging"
        $componentRoot = Join-Path $sourceRoot "reranker"
        $destinationRoot = Join-Path $tmp "package"
        try {
            New-Item -Path $componentRoot -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $componentRoot "model.bin") -Value "fake" -Encoding ASCII
            {
                Copy-OfflineAiOptionalComponent `
                    -SourceRoot $sourceRoot `
                    -DestinationRoot $destinationRoot `
                    -ComponentId "reranker" `
                    -StagingDirectory "reranker" `
                    -PackageRelativePath "optional-components\..\app\offline-ai\_internal\optional-components\reranker" `
                    -RequiredFiles @("model.bin") `
                    -DryRun
            } | Should -Throw "*outside root*"

            foreach ($relative in @("private/model.bin", "service-account-prod.json", "client-credentials.json", "client.jks", "client.keystore")) {
                $sensitivePath = Join-Path $componentRoot $relative
                New-Item -Path (Split-Path -Parent $sensitivePath) -ItemType Directory -Force | Out-Null
                Set-Content -LiteralPath $sensitivePath -Value "blocked" -Encoding ASCII
                {
                    Copy-OfflineAiOptionalComponent `
                        -SourceRoot $sourceRoot `
                        -DestinationRoot $destinationRoot `
                        -ComponentId "reranker" `
                        -StagingDirectory "reranker" `
                        -PackageRelativePath "optional-components\reranker" `
                        -RequiredFiles @("model.bin") `
                        -DryRun
                } | Should -Throw "*sensitive file name*"
                Remove-Item -LiteralPath $sensitivePath -Force
            }
        } finally {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }
}

Describe "OfflineAi.Common local detector helpers" {
    It "detects FAT32 disk info" {
        $disk = [PSCustomObject]@{ FileSystem = "FAT32" }
        Test-OfflineAiFat32PackageMedia -Root "C:\" -DiskInfo $disk | Should -BeTrue
    }

    It "does not report non-FAT32 disk info as FAT32" {
        $disk = [PSCustomObject]@{ FileSystem = "NTFS" }
        Test-OfflineAiFat32PackageMedia -Root "C:\" -DiskInfo $disk | Should -BeFalse
    }

    It "parses NVIDIA GPU memory from nvidia-smi style output" {
        $gpu = Get-OfflineAiNvidiaGpu -Query { "12288" }
        $gpu.Available | Should -BeTrue
        $gpu.VramGB | Should -Be 12
    }

}

Describe "OfflineAi.Common package collection helpers" {
    It "filters skipped directories, prefixes, and file names during copy" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-common-copy-$([guid]::NewGuid().ToString('N'))"
        $source = Join-Path $tmp 'source'
        $destination = Join-Path $tmp 'destination'
        try {
            foreach ($relative in @('keep\ok.txt', '.hidden\secret.txt', 'node_modules\dep.txt', 'cache\old.txt', 'keep\skip.txt')) {
                $path = Join-Path $source $relative
                New-Item -Path (Split-Path -Parent $path) -ItemType Directory -Force | Out-Null
                Set-Content -LiteralPath $path -Value $relative -Encoding ASCII
            }

            Copy-OfflineAiDirectoryFiltered `
                -SourceRoot $source `
                -DestinationRoot $destination `
                -SkipRelativePrefixes @('cache\') `
                -SkipFileNames @('skip.txt') `
                -SkipDirectoryNames @('node_modules') `
                -SkipDotDirectories

            Test-Path -LiteralPath (Join-Path $destination 'keep\ok.txt') | Should -BeTrue
            Test-Path -LiteralPath (Join-Path $destination '.hidden\secret.txt') | Should -BeFalse
            Test-Path -LiteralPath (Join-Path $destination 'node_modules\dep.txt') | Should -BeFalse
            Test-Path -LiteralPath (Join-Path $destination 'cache\old.txt') | Should -BeFalse
            Test-Path -LiteralPath (Join-Path $destination 'keep\skip.txt') | Should -BeFalse
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "creates a deterministic inventory and matching checksum file" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-common-inventory-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $tmp 'b.txt') -Value 'b' -Encoding ASCII
            Set-Content -LiteralPath (Join-Path $tmp 'a.txt') -Value 'a' -Encoding ASCII

            $inventory = @(New-OfflineAiFileInventory -Root $tmp)
            Write-OfflineAiChecksums -Root $tmp -Inventory $inventory

            @($inventory.relativePath) | Should -Be @('a.txt', 'b.txt')
            $checksumLines = @(Get-Content -LiteralPath (Join-Path $tmp 'checksums.sha256'))
            $checksumLines.Count | Should -Be 2
            Test-OfflineAiPackageChecksums -Root $tmp | Should -BeTrue
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Describe 'OfflineAi.Common public source boundary' {
    It 'projects an allowlisted schemaVersion 2 manifest without target input fields' {
        $source = [PSCustomObject]@{
            schemaVersion = 2
            packageKind = 'download-only'
            installable = $true
            productVersion = '1.0.0'
            createdAt = '2026-08-12T00:00:00Z'
            models = @([PSCustomObject]@{ name = 'qwen3.5:9b' })
            selection = [PSCustomObject]@{ chatModel = 'qwen3.5:9b' }
            targetRequirementsInput = [PSCustomObject]@{ gpuName = 'private fixture gpu' }
        }
        $public = New-OfflineAiPublicSourceManifest -SourceManifest $source
        $public.schemaVersion | Should -Be 2
        $public.packageKind | Should -Be 'public-source'
        $public.installable | Should -BeFalse
        @($public.models).Count | Should -Be 0
        $public.PSObject.Properties.Name | Should -Not -Contain 'selection'
        $public.PSObject.Properties.Name | Should -Not -Contain 'targetRequirementsInput'
    }

    It 'rejects parsed manifest contamination, binary directories, sensitive files, and disguised PE headers' {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-public-boundary-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $manifestPath = Join-Path $tmp 'manifest.json'
            @{ schemaVersion = 2; packageKind = 'public-source'; installable = $false; models = @() } |
                ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Not -Throw

            @{ schemaVersion = 2; packageKind = 'public-source'; installable = $false; models = @(); targetRequirementsInput = @{ ramGiB = 16 } } |
                ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*target input field*'

            @{ schemaVersion = 2; packageKind = 'public-source'; installable = $false; models = @() } |
                ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8
            New-Item -Path (Join-Path $tmp 'installers') -ItemType Directory -Force | Out-Null
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*third-party payload directory*'
            Remove-Item -LiteralPath (Join-Path $tmp 'installers') -Recurse -Force

            Set-Content -LiteralPath (Join-Path $tmp '.env') -Value 'fixture' -Encoding ASCII
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*sensitive file*'
            Remove-Item -LiteralPath (Join-Path $tmp '.env') -Force

            [System.IO.File]::WriteAllBytes((Join-Path $tmp 'innocent.txt'), [byte[]](0x4D, 0x5A, 0x00))
            { Assert-OfflineAiPublicArtifactContents -Root $tmp } | Should -Throw '*PE binary payload*'
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
