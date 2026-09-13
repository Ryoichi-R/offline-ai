BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    $script:PolicyPath = Join-Path $script:OfflineAiRoot '_internal\distribution-policy.json'
    $script:VerifyScriptPath = Join-Path $script:OfflineAiRoot '_internal\scripts\Test-DistributionPolicy.ps1'
    Import-Module $script:CommonModulePath -Force
}

Describe "distribution-policy pattern matching" {
    It "matches an exact-path pattern" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'README.md' | Should -Be 'both'
    }

    It "matches a double-star directory pattern" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'tests/unit/test_config.py' | Should -Be 'public-source'
    }

    It "returns null for an unmatched path" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '__never_registered__.txt' | Should -Be $null
    }

    It "excludes Embedding checkpoint, lock, and atomic temporary files" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '_internal/embed_cache.checkpoints/state.json' | Should -Be 'excluded'
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '_internal/embed_cache.checkpoints/batch-00000001.json.tmp' | Should -Be 'excluded'
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '_internal/embed_cache.lock' | Should -Be 'excluded'
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath '_internal/embed_cache.json.tmp' | Should -Be 'excluded'
    }

    It "gives excluded precedence over an overlapping allow pattern" {
        $policy = [PSCustomObject]@{
            entries = @(
                [PSCustomObject]@{ pattern = 'dir/*'; distribution = 'both'; regex = (ConvertTo-OfflineAiPolicyRegex -Pattern 'dir/*') }
                [PSCustomObject]@{ pattern = 'dir/secret.txt'; distribution = 'excluded'; regex = (ConvertTo-OfflineAiPolicyRegex -Pattern 'dir/secret.txt') }
            )
        }
        Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'dir/secret.txt' | Should -Be 'excluded'
    }

    It "throws on ambiguous non-excluded classification" {
        $policy = [PSCustomObject]@{
            entries = @(
                [PSCustomObject]@{ pattern = 'dir/*'; distribution = 'both'; regex = (ConvertTo-OfflineAiPolicyRegex -Pattern 'dir/*') }
                [PSCustomObject]@{ pattern = 'dir/x.txt'; distribution = 'runtime'; regex = (ConvertTo-OfflineAiPolicyRegex -Pattern 'dir/x.txt') }
            )
        }
        { Get-OfflineAiDistributionClassification -Policy $policy -RelativePath 'dir/x.txt' } | Should -Throw '*ambiguous*'
    }
}

Describe "distribution-policy path safety" {
    It "rejects a traversal segment" {
        { Test-OfflineAiRelativePathSafe -RelativePath 'a/../b.txt' } | Should -Throw '*traversal*'
    }

    It "rejects a colon (ADS-like) path" {
        { Test-OfflineAiRelativePathSafe -RelativePath 'a/file.txt:stream' } | Should -Throw '*colon*'
    }

    It "rejects a reserved device name segment" {
        { Test-OfflineAiRelativePathSafe -RelativePath 'a/CON.txt' } | Should -Throw '*reserved device name*'
    }

    It "accepts a normal relative path" {
        Test-OfflineAiRelativePathSafe -RelativePath 'rules/leave.md' | Should -BeTrue
    }
}

Describe "distribution inventory locked files" -Tag 'WindowsOnly' {
    It "classifies excluded locked content without reading it: <Relative>" -TestCases @(
        @{ Relative = 'offline-package/tool.part' }
        @{ Relative = '.git/objects/temporary-pack' }
        @{ Relative = '.test-results/evaluation.json' }
    ) {
        param($Relative)
        $fixture = Join-Path $TestDrive ('excluded-download-' + [guid]::NewGuid().ToString('N'))
        $path = Join-Path $fixture $Relative
        New-Item -ItemType Directory -Path (Split-Path $path) -Force | Out-Null
        [IO.File]::WriteAllText($path, 'download in progress')
        $lock = [IO.File]::Open($path, 'Open', 'ReadWrite', 'None')
        try {
            $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
            $inventory = Get-OfflineAiDistributionInventory -SourceRoot $fixture -Policy $policy
            $inventory.unmatchedPaths | Should -BeNullOrEmpty
            $inventory.classified.Count | Should -Be 1
            $inventory.classified[0].distribution | Should -Be 'excluded'
            $inventory.classified[0].sha256 | Should -BeNullOrEmpty
        } finally { $lock.Dispose() }
    }

    It "still rejects a locked runtime file whose hash cannot be verified" {
        $fixture = Join-Path $TestDrive 'locked-runtime'
        New-Item -ItemType Directory -Path $fixture -Force | Out-Null
        $path = Join-Path $fixture 'README.md'
        [IO.File]::WriteAllText($path, 'runtime')
        $lock = [IO.File]::Open($path, 'Open', 'ReadWrite', 'None')
        try {
            $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
            { Get-OfflineAiDistributionInventory -SourceRoot $fixture -Policy $policy -ErrorAction Stop } | Should -Throw
        } finally { $lock.Dispose() }
    }
}

Describe "distribution-policy coverage on the real offline-ai tree" {
    It "classifies every file under offline-ai with zero unmatched paths" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        $inventory = Get-OfflineAiDistributionInventory -SourceRoot $script:OfflineAiRoot -Policy $policy
        $inventory.unmatchedPaths | Should -BeNullOrEmpty
    }

    It "matches catalog appPayloadEstimate to the runtime and both inventory" {
        $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
        $inventory = Get-OfflineAiDistributionInventory -SourceRoot $script:OfflineAiRoot -Policy $policy
        $catalog = Get-Content -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\download-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $gate = Test-OfflineAiAppPayloadEstimate -Inventory $inventory -Expected $catalog.appPayloadEstimate
        $gate.passed | Should -BeTrue
        $gate.actual.fileCount | Should -Be 36
    }

    It "fails closed when an unregistered file is added" {
        $dummy = Join-Path $script:OfflineAiRoot '_internal\__dummy_policy_negative_test.txt'
        try {
            Set-Content -LiteralPath $dummy -Value 'dummy' -Encoding ASCII
            $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
            $inventory = Get-OfflineAiDistributionInventory -SourceRoot $script:OfflineAiRoot -Policy $policy
            @($inventory.unmatchedPaths) | Should -Contain '_internal/__dummy_policy_negative_test.txt'
        } finally {
            Remove-Item -LiteralPath $dummy -Force -ErrorAction SilentlyContinue
        }
    }

    It "rejects reparse points during inventory collection" {
        $linkPath = Join-Path $script:OfflineAiRoot '_internal\__reparse_negative_test'
        $targetPath = Join-Path $script:OfflineAiRoot '_internal\__reparse_negative_target'
        try {
            New-Item -Path $targetPath -ItemType Directory -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $targetPath 'x.txt') -Value 'x' -Encoding ASCII
            try {
                $linkItemType = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
                New-Item -Path $linkPath -ItemType $linkItemType -Target $targetPath -ErrorAction Stop | Out-Null
            } catch {
                Set-ItResult -Skipped -Because "junction/symlink creation is unavailable: $($_.Exception.Message)"
                return
            }
            $policy = Import-OfflineAiDistributionPolicy -PolicyPath $script:PolicyPath
            { Get-OfflineAiDistributionInventory -SourceRoot $script:OfflineAiRoot -Policy $policy } | Should -Throw '*Reparse point*'
        } finally {
            if (Test-Path -LiteralPath $linkPath) { Remove-Item -LiteralPath $linkPath -Force -Recurse -ErrorAction SilentlyContinue }
            if (Test-Path -LiteralPath $targetPath) { Remove-Item -LiteralPath $targetPath -Force -Recurse -ErrorAction SilentlyContinue }
        }
    }
}

Describe "Test-DistributionPolicy.ps1 candidate digest" {
    It "produces the same candidate_digest across two runs with a different generated_at" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-policy-digest-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $receipt1Path = Join-Path $tmp 'receipt1.json'
            $receipt2Path = Join-Path $tmp 'receipt2.json'
            & pwsh -NoProfile -File $script:VerifyScriptPath -Audience public-source -ReceiptPath $receipt1Path | Out-Null
            Start-Sleep -Seconds 1
            & pwsh -NoProfile -File $script:VerifyScriptPath -Audience public-source -ReceiptPath $receipt2Path | Out-Null

            $receipt1 = Get-Content -LiteralPath $receipt1Path -Raw | ConvertFrom-Json
            $receipt2 = Get-Content -LiteralPath $receipt2Path -Raw | ConvertFrom-Json

            $receipt1.status | Should -Be 'READY'
            $receipt1.sensitive_scan_performed | Should -BeFalse
            $receipt1.candidate_digest | Should -Be $receipt2.candidate_digest
            $receipt1.generated_at | Should -Not -Be $receipt2.generated_at
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "excludes public-source-only files from the runtime audience" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-policy-runtime-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path $tmp -ItemType Directory -Force | Out-Null
            $receiptPath = Join-Path $tmp 'receipt.json'
            & pwsh -NoProfile -File $script:VerifyScriptPath -Audience runtime -ReceiptPath $receiptPath | Out-Null
            $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json

            $paths = @($receipt.files | ForEach-Object { $_.path })
            $paths | Should -Not -Contain 'CONTRIBUTING.md'
            $paths | Should -Not -Contain 'pyproject.toml'
            @($paths | Where-Object { $_ -like 'tests/*' }).Count | Should -Be 0
            $paths | Should -Contain 'search.bat'
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    It "fails closed when appPayloadEstimate is stale" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-policy-app-estimate-$([guid]::NewGuid().ToString('N'))"
        try {
            New-Item -Path (Join-Path $tmp '_internal\scripts') -ItemType Directory -Force | Out-Null
            Copy-Item -LiteralPath $script:CommonModulePath -Destination (Join-Path $tmp '_internal\OfflineAi.Common.psm1')
            Copy-Item -LiteralPath $script:PolicyPath -Destination (Join-Path $tmp '_internal\distribution-policy.json')
            Copy-Item -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\download-manifest.json') -Destination (Join-Path $tmp '_internal\download-manifest.json')
            $output = & pwsh -NoProfile -File $script:VerifyScriptPath -SourceRoot $tmp 2>&1
            $LASTEXITCODE | Should -Be 1
            $output | Out-String | Should -Match 'V-6'
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
