BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    Import-Module (Join-Path $script:OfflineAiRoot '_internal\OfflineAi.ModelSelection.psm1') -Force
    $script:Catalog = Get-Content -LiteralPath (Join-Path $script:OfflineAiRoot '_internal\download-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json

    function New-Target {
        param(
            [object]$Vram = 12,
            [object]$Ram = 32,
            [string]$Layout = 'AllSeparate',
            [object]$App = 40,
            [object]$Models = 40,
            [object]$Temp = 4,
            [string]$Preference = 'Balanced',
            [string]$Vendor = 'Nvidia'
        )
        ConvertTo-OfflineAiTargetRequirements -GpuName 'fixture gpu' -GpuVendor $Vendor -VramGiB $Vram -RamGiB $Ram `
            -StorageLayout $Layout -AppFreeDiskGiB $App -ModelsFreeDiskGiB $Models -TempFreeDiskGiB $Temp -Preference $Preference
    }
}

Describe 'offline-ai model catalog contract' {
    It 'accepts the canonical three-tier catalog and one embedding' {
        $result = Assert-OfflineAiModelCatalog -Catalog $script:Catalog
        $result.selectableChats.Count | Should -Be 3
        $result.requiredEmbedding.name | Should -Be 'bge-m3'
    }

    It 'uses gpt-oss:20b for balanced and leaves qwen3.5:9b out of the catalog' {
        @($script:Catalog.models | Where-Object name -eq 'gpt-oss:20b')[0].recommendation.tier | Should -Be 'balanced'
        @($script:Catalog.models | Where-Object name -eq 'gpt-oss:20b')[0].selectable | Should -BeTrue
        @($script:Catalog.models | Where-Object name -eq 'qwen3.5:9b').Count | Should -Be 0
        @($script:Catalog.legacyMigrationModels) | Should -Not -Contain 'qwen3.5:9b'
    }

    It 'rejects selectable chat overlap with the legacy denylist' {
        $copy = $script:Catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
        $copy.legacyMigrationModels[0] = 'qwen3.5:4b'
        { Assert-OfflineAiModelCatalog -Catalog $copy } | Should -Throw '*overlaps legacyMigrationModels*'
    }

    It 'rejects duplicate tiers' {
        $copy = $script:Catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
        $copy.models[1].recommendation.tier = 'compact'
        { Assert-OfflineAiModelCatalog -Catalog $copy } | Should -Throw '*duplicate selectable tier*'
    }

    It 'rejects a missing largest blob estimate' {
        $copy = $script:Catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
        $copy.models[0].recommendation.estimatedLargestBlobBytes = 0
        { Assert-OfflineAiModelCatalog -Catalog $copy } | Should -Throw '*estimatedLargestBlobBytes*'
    }

    It 'requires pinned manifest, config, and license digests' {
        $copy = $script:Catalog | ConvertTo-Json -Depth 20 | ConvertFrom-Json
        $copy.models[0].expectedManifestDigest = ''
        { Assert-OfflineAiModelCatalog -Catalog $copy } | Should -Throw '*expectedManifestDigest*'
    }
}

Describe 'offline-ai target input normalization' {
    It 'rejects RAM below four GiB and control characters in GPU name' {
        { ConvertTo-OfflineAiTargetRequirements -RamGiB 3 -StorageLayout AllSeparate -AppFreeDiskGiB 1 -ModelsFreeDiskGiB 1 -TempFreeDiskGiB 1 } | Should -Throw '*at least 4*'
        { ConvertTo-OfflineAiTargetRequirements -GpuName "bad`nname" -RamGiB 16 -StorageLayout AllSeparate -AppFreeDiskGiB 1 -ModelsFreeDiskGiB 20 -TempFreeDiskGiB 2 } | Should -Throw '*control*'
    }

    It 'allows unknown VRAM without inventing a value' {
        $target = ConvertTo-OfflineAiTargetRequirements -GpuVendor Unknown -RamGiB 16 -StorageLayout AllSeparate -AppFreeDiskGiB 20 -ModelsFreeDiskGiB 30 -TempFreeDiskGiB 2
        $target.vramGiB | Should -BeNullOrEmpty
    }
}

Describe 'offline-ai recommendation ranking and boundaries' {
    It 'ranks balanced first for a supported full GPU target' {
        $items = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target))
        $items[0].model | Should -Be 'qwen3.5:4b'
        $items[0].recommendationClass | Should -Be 'full-gpu-likely'
    }

    It 'changes the first same-class tier with preference' {
        $speed = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target -Vram 32 -Ram 64 -Preference Speed))
        $quality = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target -Vram 32 -Ram 64 -Preference Quality))
        $speed[0].tier | Should -Be 'compact'
        $quality[0].tier | Should -Be 'quality'
    }

    It 'classifies unknown VRAM without a GPU guarantee' {
        $items = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target -Vram $null -Vendor Unknown))
        $items[0].recommendationClass | Should -Be 'gpu-unknown'
    }

    It 'uses VRAM for placement class even when GPU vendor is unknown' {
        $items = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target -Vram 12 -Vendor Unknown))
        (@($items | Where-Object model -eq 'gpt-oss:20b'))[0].recommendationClass | Should -Be 'partial-gpu-likely'
        (@($items | Where-Object model -eq 'qwen3.5:27b'))[0].recommendationClass | Should -Be 'partial-gpu-likely'
        $items[0].compatibilityNotices -join ' ' | Should -Match 'GPUベンダー未指定'
    }

    It 'marks RAM shortage separately' {
        $items = @(Get-OfflineAiModelRecommendations -Catalog $script:Catalog -Target (New-Target -Vram 24 -Ram 8))
        (@($items | Where-Object model -eq 'qwen3.5:27b'))[0].recommendationClass | Should -Be 'ram-unsupported'
    }

    It 'accepts exact AllSeparate capacity and rejects one byte less' {
        $target = New-Target
        $validated = Assert-OfflineAiModelCatalog -Catalog $script:Catalog
        $chat = @($validated.selectableChats | Where-Object name -eq 'gpt-oss:20b')[0]
        $assessment = Get-OfflineAiStorageAssessment -Target $target -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $exact = ConvertTo-OfflineAiTargetRequirements -GpuVendor None -VramGiB 0 -RamGiB 32 -StorageLayout AllSeparate `
            -AppFreeDiskGiB ([decimal]$assessment.appRequiredBytes / 1GB) `
            -ModelsFreeDiskGiB ([decimal]$assessment.modelsRequiredBytes / 1GB) `
            -TempFreeDiskGiB ([decimal]$assessment.tempRequiredBytes / 1GB)
        (Get-OfflineAiStorageAssessment -Target $exact -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate).passed | Should -BeTrue
        $exact.modelsFreeDiskGiB -= ([decimal]1 / 1GB)
        $short = Get-OfflineAiStorageAssessment -Target $exact -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $short.passed | Should -BeFalse
        $short.shortages[0].labelJa | Should -Be 'models保存先'
        $short.shortages[0].shortageBytes | Should -Be 1
    }

    It 'uses app total plus largest file plus atomic margin above the one GiB floor' {
        $target = New-Target
        $validated = Assert-OfflineAiModelCatalog -Catalog $script:Catalog
        $chat = @($validated.selectableChats | Where-Object name -eq 'gpt-oss:20b')[0]
        $largeApp = [PSCustomObject]@{ totalBytes = [Int64]2GB; largestFileBytes = [Int64]512MB }
        $assessment = Get-OfflineAiStorageAssessment -Target $target -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $largeApp
        $assessment.appRequiredBytes | Should -Be ([Int64]3GB)
    }

    It 'accepts exact SingleVolume capacity and rejects one byte less' {
        $validated = Assert-OfflineAiModelCatalog -Catalog $script:Catalog
        $chat = @($validated.selectableChats | Where-Object name -eq 'gpt-oss:20b')[0]
        $seed = Get-OfflineAiStorageAssessment -Target (New-Target -Layout SingleVolume -App 80 -Models 0 -Temp 0) `
            -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        [decimal]$required = $seed.appRequiredBytes + $seed.modelsRequiredBytes + $seed.tempRequiredBytes
        $exact = New-Target -Layout SingleVolume -App ($required / 1GB) -Models 0 -Temp 0
        (Get-OfflineAiStorageAssessment -Target $exact -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate).passed | Should -BeTrue
        $exact.appFreeDiskGiB -= ([decimal]1 / 1GB)
        $short = Get-OfflineAiStorageAssessment -Target $exact -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $short.passed | Should -BeFalse
        $short.shortages[0].label | Should -Be 'app+models+temp'
        $short.shortages[0].shortageBytes | Should -Be 1
    }

    It 'accepts exact ModelsSeparate capacity and rejects either volume one byte short' {
        $validated = Assert-OfflineAiModelCatalog -Catalog $script:Catalog
        $chat = @($validated.selectableChats | Where-Object name -eq 'gpt-oss:20b')[0]
        $seed = Get-OfflineAiStorageAssessment -Target (New-Target -Layout ModelsSeparate -App 20 -Models 40 -Temp 0) `
            -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $exact = New-Target -Layout ModelsSeparate -App (($seed.appRequiredBytes + $seed.tempRequiredBytes) / 1GB) `
            -Models ($seed.modelsRequiredBytes / 1GB) -Temp 0
        (Get-OfflineAiStorageAssessment -Target $exact -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate).passed | Should -BeTrue

        $appShort = New-Target -Layout ModelsSeparate -App ($exact.appFreeDiskGiB - ([decimal]1 / 1GB)) -Models $exact.modelsFreeDiskGiB -Temp 0
        $appAssessment = Get-OfflineAiStorageAssessment -Target $appShort -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $appAssessment.shortages[0].label | Should -Be 'app+temp'
        $appAssessment.shortages[0].shortageBytes | Should -Be 1

        $modelsShort = New-Target -Layout ModelsSeparate -App $exact.appFreeDiskGiB -Models ($exact.modelsFreeDiskGiB - ([decimal]1 / 1GB)) -Temp 0
        $modelsAssessment = Get-OfflineAiStorageAssessment -Target $modelsShort -ChatModel $chat -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $modelsAssessment.shortages[0].label | Should -Be 'models'
        $modelsAssessment.shortages[0].shortageBytes | Should -Be 1
    }
}

Describe 'offline-ai explicit model selection' {
    It 'allows only catalog chat models' {
        { Resolve-OfflineAiChatSelection -Catalog $script:Catalog -Target (New-Target) -ChatModel bge-m3 } | Should -Throw '*not a selectable*'
    }

    It 'allows only RAM shortage through the explicit override receipt' {
        $target = New-Target -Ram 8 -Vram 8
        { Resolve-OfflineAiChatSelection -Catalog $script:Catalog -Target $target -ChatModel 'qwen3.5:27b' } | Should -Throw '*minimum RAM*'
        $result = Resolve-OfflineAiChatSelection -Catalog $script:Catalog -Target $target -ChatModel 'qwen3.5:27b' -AllowUnsupported
        $result.unsupportedOverride.used | Should -BeTrue
        $result.unsupportedOverride.reasonCodes | Should -Be @('ram-below-minimum')
        $result.unsupportedOverride.confirmationMode | Should -Be 'explicit-switch'
    }

    It 'never lets RAM override bypass disk shortage' {
        $target = New-Target -Ram 8 -Vram 8 -App 0 -Models 0 -Temp 0
        { Resolve-OfflineAiChatSelection -Catalog $script:Catalog -Target $target -ChatModel 'qwen3.5:27b' -AllowUnsupported } | Should -Throw '*disk requirements*'
    }

    It 'rejects inconsistent override receipts' {
        $bad = [PSCustomObject]@{ used = $false; reasonCodes = @('ram-below-minimum'); confirmationMode = 'explicit-switch' }
        { Assert-OfflineAiUnsupportedOverrideReceipt -Receipt $bad } | Should -Throw '*inconsistent*'
    }

    It 'serializes an unused override receipt with an empty reason array' {
        $json = New-OfflineAiUnsupportedOverrideReceipt -Used:$false -ConfirmationMode not-required | ConvertTo-Json -Compress
        $json | Should -Match '"reasonCodes":\[\]'
        { Assert-OfflineAiUnsupportedOverrideReceipt -Receipt ($json | ConvertFrom-Json) } | Should -Not -Throw
    }
}
