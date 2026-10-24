Set-StrictMode -Version Latest

$script:GiB = [Int64]1073741824
$script:ModelNamePattern = '^[a-z0-9][a-z0-9._/-]*(?::[a-z0-9][a-z0-9._-]*)?$'
$script:Sha256DigestPattern = '^sha256:[0-9a-f]{64}$'
$script:SupportedCatalogSchemaVersion = 2

function Get-OfflineAiObjectPropertyNames {
    param([Parameter(Mandatory = $true)][object]$Value)
    return @($Value.PSObject.Properties.Name)
}

function Get-OfflineAiRequiredProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Context
    )
    if ((Get-OfflineAiObjectPropertyNames -Value $Value) -notcontains $Name) {
        throw "$Context is missing required property '$Name'."
    }
    return $Value.$Name
}

function ConvertTo-OfflineAiNonNegativeDecimal {
    param(
        [AllowNull()][object]$Value,
        [Parameter(Mandatory = $true)][string]$Name,
        [switch]$AllowNull
    )
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        if ($AllowNull) { return $null }
        throw "$Name is required."
    }
    [decimal]$parsed = 0
    if (-not [decimal]::TryParse([string]$Value, [Globalization.NumberStyles]::Number, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        throw "$Name must be a decimal value."
    }
    if ($parsed -lt 0) { throw "$Name must be zero or greater." }
    return $parsed
}

function Assert-OfflineAiModelName {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Context
    )
    if ([string]::IsNullOrWhiteSpace($Name) -or $Name -cnotmatch $script:ModelNamePattern) {
        throw "$Context contains an invalid model name: '$Name'."
    }
}

function Assert-OfflineAiModelCatalog {
    param([Parameter(Mandatory = $true)][object]$Catalog)

    $schemaVersion = Get-OfflineAiRequiredProperty -Value $Catalog -Name 'schemaVersion' -Context 'catalog'
    if ([int]$schemaVersion -ne $script:SupportedCatalogSchemaVersion) {
        throw "Unsupported download manifest schemaVersion: $schemaVersion"
    }
    $appPayloadEstimate = Get-OfflineAiRequiredProperty -Value $Catalog -Name 'appPayloadEstimate' -Context 'catalog'
    foreach ($field in @('totalBytes', 'largestFileBytes', 'reviewedAt')) {
        $null = Get-OfflineAiRequiredProperty -Value $appPayloadEstimate -Name $field -Context 'appPayloadEstimate'
    }
    if ([Int64]$appPayloadEstimate.totalBytes -lt 0 -or [Int64]$appPayloadEstimate.largestFileBytes -lt 0) {
        throw 'appPayloadEstimate byte fields must be zero or greater.'
    }
    if ([Int64]$appPayloadEstimate.largestFileBytes -gt [Int64]$appPayloadEstimate.totalBytes) {
        throw 'appPayloadEstimate.largestFileBytes must not exceed totalBytes.'
    }
    $appPayloadReviewedAt = [DateTime]::MinValue
    if (-not [DateTime]::TryParseExact([string]$appPayloadEstimate.reviewedAt, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$appPayloadReviewedAt)) {
        throw 'appPayloadEstimate.reviewedAt must be YYYY-MM-DD.'
    }
    $legacyValues = @(Get-OfflineAiRequiredProperty -Value $Catalog -Name 'legacyMigrationModels' -Context 'catalog')
    $legacySet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($legacyValue in $legacyValues) {
        $legacyName = [string]$legacyValue
        Assert-OfflineAiModelName -Name $legacyName -Context 'legacyMigrationModels'
        if (-not $legacySet.Add($legacyName)) { throw "legacyMigrationModels contains duplicate model: $legacyName" }
    }

    $models = @(Get-OfflineAiRequiredProperty -Value $Catalog -Name 'models' -Context 'catalog')
    $modelSet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $tierSet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $selectableChats = @()
    $requiredEmbeddings = @()
    foreach ($model in $models) {
        $properties = Get-OfflineAiObjectPropertyNames -Value $model
        $name = [string](Get-OfflineAiRequiredProperty -Value $model -Name 'name' -Context 'model')
        Assert-OfflineAiModelName -Name $name -Context 'model'
        if (-not $modelSet.Add($name)) { throw "catalog contains duplicate model: $name" }

        $role = [string](Get-OfflineAiRequiredProperty -Value $model -Name 'role' -Context "model '$name'")
        if ($role -notin @('chat', 'embedding', 'optional')) { throw "model '$name' has unknown role: $role" }
        $selectable = $properties -contains 'selectable' -and [bool]$model.selectable
        $required = $properties -contains 'required' -and [bool]$model.required
        if ($role -eq 'chat' -and $selectable) {
            foreach ($field in @('registry', 'repository', 'tag', 'expectedManifestDigest', 'expectedConfigDigest', 'expectedLicenseLayerDigest', 'parameterSize', 'quantization', 'minimumOllamaVersion', 'adoptionReceipt', 'recommendation')) {
                $null = Get-OfflineAiRequiredProperty -Value $model -Name $field -Context "selectable chat '$name'"
            }
            foreach ($field in @('expectedManifestDigest', 'expectedConfigDigest', 'expectedLicenseLayerDigest')) {
                if ([string]$model.$field -cnotmatch $script:Sha256DigestPattern) {
                    throw "selectable chat '$name' field '$field' must be a lowercase sha256 digest."
                }
            }
            foreach ($field in @('parameterSize', 'quantization', 'minimumOllamaVersion', 'adoptionReceipt')) {
                if ([string]::IsNullOrWhiteSpace([string]$model.$field)) {
                    throw "selectable chat '$name' field '$field' must not be empty."
                }
            }
            if ($legacySet.Contains($name)) { throw "selectable chat overlaps legacyMigrationModels: $name" }
            $recommendation = $model.recommendation
            $tier = [string](Get-OfflineAiRequiredProperty -Value $recommendation -Name 'tier' -Context "recommendation '$name'")
            if ($tier -notin @('compact', 'balanced', 'quality')) { throw "model '$name' has invalid tier: $tier" }
            if (-not $tierSet.Add($tier)) { throw "catalog contains duplicate selectable tier: $tier" }
            foreach ($field in @('minimumRamGiB', 'fullGpuVramGiB', 'partialGpuVramGiB', 'estimatedDownloadBytes', 'estimatedLargestBlobBytes', 'contextTokensAssumed')) {
                $number = Get-OfflineAiRequiredProperty -Value $recommendation -Name $field -Context "recommendation '$name'"
                if ([decimal]$number -le 0) { throw "recommendation '$name' field '$field' must be greater than zero." }
            }
            if ([decimal]$recommendation.partialGpuVramGiB -gt [decimal]$recommendation.fullGpuVramGiB) {
                throw "recommendation '$name' partialGpuVramGiB exceeds fullGpuVramGiB."
            }
            $reviewedAt = [DateTime]::MinValue
            if (-not [DateTime]::TryParseExact([string]$recommendation.catalogReviewedAt, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$reviewedAt)) {
                throw "recommendation '$name' catalogReviewedAt must be YYYY-MM-DD."
            }
            if (@($recommendation.evidence).Count -eq 0) { throw "recommendation '$name' requires evidence." }
            $selectableChats += $model
        } elseif ($role -eq 'chat' -and $selectable -eq $false) {
            throw "chat model '$name' must be selectable or use role optional."
        }
        if ($role -eq 'embedding' -and $required) {
            if ($selectable) { throw "required embedding '$name' must not be selectable." }
            $requiredEmbeddings += $model
        }
    }
    if ($selectableChats.Count -lt 3) { throw 'catalog must contain at least three selectable chat models.' }
    foreach ($tier in @('compact', 'balanced', 'quality')) {
        if (-not $tierSet.Contains($tier)) { throw "catalog is missing selectable tier: $tier" }
    }
    if ($requiredEmbeddings.Count -ne 1) { throw 'catalog must contain exactly one required embedding model.' }
    $embedding = $requiredEmbeddings[0]
    foreach ($field in @('registry', 'repository', 'tag', 'expectedManifestDigest', 'expectedConfigDigest', 'expectedLicenseLayerDigest', 'parameterSize', 'quantization', 'adoptionReceipt')) {
        $null = Get-OfflineAiRequiredProperty -Value $embedding -Name $field -Context "required embedding '$($embedding.name)'"
    }
    foreach ($field in @('expectedManifestDigest', 'expectedConfigDigest', 'expectedLicenseLayerDigest')) {
        if ([string]$embedding.$field -cnotmatch $script:Sha256DigestPattern) {
            throw "required embedding '$($embedding.name)' field '$field' must be a lowercase sha256 digest."
        }
    }
    foreach ($field in @('parameterSize', 'quantization', 'adoptionReceipt')) {
        if ([string]::IsNullOrWhiteSpace([string]$embedding.$field)) {
            throw "required embedding '$($embedding.name)' field '$field' must not be empty."
        }
    }
    if (-not (Get-OfflineAiObjectPropertyNames -Value $embedding.recommendation)) { throw "required embedding '$($embedding.name)' requires recommendation metadata." }
    foreach ($field in @('estimatedDownloadBytes', 'estimatedLargestBlobBytes')) {
        if ([Int64](Get-OfflineAiRequiredProperty -Value $embedding.recommendation -Name $field -Context "required embedding '$($embedding.name)'") -le 0) {
            throw "required embedding '$($embedding.name)' field '$field' must be greater than zero."
        }
    }
    return [PSCustomObject]@{
        selectableChats = $selectableChats
        requiredEmbedding = $embedding
        legacyMigrationModels = @($legacySet)
        appPayloadEstimate = $appPayloadEstimate
    }
}

function ConvertTo-OfflineAiTargetRequirements {
    param(
        [AllowEmptyString()][string]$GpuName = '',
        [ValidateSet('Nvidia', 'Amd', 'Intel', 'Other', 'None', 'Unknown')][string]$GpuVendor = 'Unknown',
        [AllowNull()][object]$VramGiB = $null,
        [AllowNull()][object]$RamGiB,
        [ValidateSet('SingleVolume', 'ModelsSeparate', 'AllSeparate')][string]$StorageLayout,
        [AllowNull()][object]$AppFreeDiskGiB,
        [AllowNull()][object]$ModelsFreeDiskGiB,
        [AllowNull()][object]$TempFreeDiskGiB,
        [ValidateSet('Speed', 'Balanced', 'Quality')][string]$Preference = 'Balanced'
    )
    $normalizedName = $GpuName.Trim()
    if ($normalizedName.Length -gt 120 -or $normalizedName -match '[\x00-\x1F\x7F]') {
        throw 'TargetGpuName must be at most 120 characters and contain no control characters.'
    }
    $ram = ConvertTo-OfflineAiNonNegativeDecimal -Value $RamGiB -Name 'TargetRamGiB'
    if ($ram -lt 4) { throw 'TargetRamGiB must be at least 4.' }
    return [PSCustomObject]@{
        gpuName = $normalizedName
        gpuVendor = $GpuVendor
        vramGiB = ConvertTo-OfflineAiNonNegativeDecimal -Value $VramGiB -Name 'TargetVramGiB' -AllowNull
        ramGiB = $ram
        storageLayout = $StorageLayout
        appFreeDiskGiB = ConvertTo-OfflineAiNonNegativeDecimal -Value $AppFreeDiskGiB -Name 'TargetAppFreeDiskGiB'
        modelsFreeDiskGiB = ConvertTo-OfflineAiNonNegativeDecimal -Value $ModelsFreeDiskGiB -Name 'TargetModelsFreeDiskGiB'
        tempFreeDiskGiB = ConvertTo-OfflineAiNonNegativeDecimal -Value $TempFreeDiskGiB -Name 'TargetTempFreeDiskGiB'
        preference = $Preference
    }
}

function Get-OfflineAiStorageAssessment {
    param(
        [Parameter(Mandatory = $true)][object]$Target,
        [Parameter(Mandatory = $true)][object]$ChatModel,
        [Parameter(Mandatory = $true)][object]$EmbeddingModel,
        [Parameter(Mandatory = $true)][object]$AppPayloadEstimate
    )
    [Int64]$chatBytes = [Int64]$ChatModel.recommendation.estimatedDownloadBytes
    [Int64]$embeddingBytes = [Int64]$EmbeddingModel.recommendation.estimatedDownloadBytes
    [Int64]$largest = [Math]::Max([Int64]$ChatModel.recommendation.estimatedLargestBlobBytes, [Int64]$EmbeddingModel.recommendation.estimatedLargestBlobBytes)
    if ($chatBytes -le 0 -or $embeddingBytes -le 0 -or $largest -le 0) { throw 'catalog contains invalid model size estimates.' }
    [Int64]$appTotal = [Int64]$AppPayloadEstimate.totalBytes
    [Int64]$appLargest = [Int64]$AppPayloadEstimate.largestFileBytes
    if ($appTotal -lt 0 -or $appLargest -lt 0) { throw 'catalog contains invalid app payload estimates.' }
    [Int64]$appRequired = [Math]::Max($appTotal + $appLargest + 512MB, 1GB)
    [Int64]$modelsRequired = [Math]::Max($chatBytes + $embeddingBytes + $largest + 512MB, 20GB)
    [Int64]$tempRequired = 2GB
    $requirements = @()
    switch ($Target.storageLayout) {
        'SingleVolume' {
            $requirements = @([PSCustomObject]@{ label = 'app+models+temp'; requiredBytes = $appRequired + $modelsRequired + $tempRequired; availableBytes = [decimal]::Round([decimal]$Target.appFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) })
        }
        'ModelsSeparate' {
            $requirements = @(
                [PSCustomObject]@{ label = 'app+temp'; requiredBytes = $appRequired + $tempRequired; availableBytes = [decimal]::Round([decimal]$Target.appFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) }
                [PSCustomObject]@{ label = 'models'; requiredBytes = $modelsRequired; availableBytes = [decimal]::Round([decimal]$Target.modelsFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) }
            )
        }
        'AllSeparate' {
            $requirements = @(
                [PSCustomObject]@{ label = 'app'; requiredBytes = $appRequired; availableBytes = [decimal]::Round([decimal]$Target.appFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) }
                [PSCustomObject]@{ label = 'models'; requiredBytes = $modelsRequired; availableBytes = [decimal]::Round([decimal]$Target.modelsFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) }
                [PSCustomObject]@{ label = 'temp'; requiredBytes = $tempRequired; availableBytes = [decimal]::Round([decimal]$Target.tempFreeDiskGiB * $script:GiB, 0, [MidpointRounding]::AwayFromZero) }
            )
        }
        default { throw "Unsupported storage layout: $($Target.storageLayout)" }
    }
    $labels = @{
        app = 'app保存先'
        models = 'models保存先'
        temp = 'temp保存先'
        'app+temp' = 'app・temp共通保存先'
        'app+models+temp' = 'app・models・temp共通保存先'
    }
    $shortages = @($requirements | Where-Object { [decimal]$_.availableBytes -lt [decimal]$_.requiredBytes } | ForEach-Object {
        [PSCustomObject]@{
            label = [string]$_.label
            labelJa = [string]$labels[[string]$_.label]
            requiredBytes = [decimal]$_.requiredBytes
            availableBytes = [decimal]$_.availableBytes
            shortageBytes = [decimal]$_.requiredBytes - [decimal]$_.availableBytes
        }
    })
    return [PSCustomObject]@{
        passed = $shortages.Count -eq 0
        appRequiredBytes = $appRequired
        modelsRequiredBytes = $modelsRequired
        tempRequiredBytes = $tempRequired
        transferEstimateBytes = $chatBytes + $embeddingBytes
        comparisons = $requirements
        shortages = $shortages
    }
}

function Get-OfflineAiGpuCompatibilityNotices {
    param([Parameter(Mandatory = $true)][object]$Target)

    switch ([string]$Target.gpuVendor) {
        'Unknown' { return @('GPUベンダー未指定です。VRAMによる容量分類とは別に、GPU・OS・driverがOllamaの対応範囲か確認してください。') }
        'Amd' { return @('AMD GPUはOS・機種・ROCm/Vulkan driverで対応範囲が異なります。Ollama公式hardware supportを確認してください。') }
        'Intel' { return @('Intel GPUのaccelerator利用可否は機種・driver・Vulkan対応で異なります。非対応時はCPU実行になります。') }
        'Other' { return @('指定GPUのOllama対応は未判定です。公式hardware supportとdriver要件を確認してください。') }
        'Nvidia' { return @('NVIDIA GPUはcompute capabilityとdriver要件を満たす必要があります。') }
        default { return @() }
    }
}

function Get-OfflineAiRecommendationPresentation {
    param(
        [Parameter(Mandatory = $true)][string]$Class,
        [Parameter(Mandatory = $true)][object]$Target,
        [Parameter(Mandatory = $true)][object]$Model,
        [Parameter(Mandatory = $true)][object]$Storage
    )

    switch ($Class) {
        'full-gpu-likely' {
            return [PSCustomObject]@{ label = 'GPU全載せ見込み'; reason = "専用VRAM $($Target.vramGiB) GiBがfull目安 $($Model.recommendation.fullGpuVramGiB) GiB以上です。" }
        }
        'partial-gpu-likely' {
            return [PSCustomObject]@{ label = 'GPU一部offload見込み'; reason = "専用VRAM $($Target.vramGiB) GiBはpartial目安以上、full目安未満です。" }
        }
        'cpu-likely' {
            return [PSCustomObject]@{ label = 'CPU中心見込み'; reason = if ($Target.gpuVendor -eq 'None') { 'GPUなしが指定されています。' } else { "専用VRAM $($Target.vramGiB) GiBがpartial目安 $($Model.recommendation.partialGpuVramGiB) GiB未満です。" } }
        }
        'gpu-unknown' {
            return [PSCustomObject]@{ label = 'GPU配置不明'; reason = '専用VRAMが未入力のため、GPU配置を数値判定できません。' }
        }
        'ram-unsupported' {
            return [PSCustomObject]@{ label = 'RAM要件未達'; reason = "RAM $($Target.ramGiB) GiBが最小要件 $($Model.recommendation.minimumRamGiB) GiB未満です。" }
        }
        'disk-insufficient' {
            $shortageText = @($Storage.shortages | ForEach-Object { "$($_.labelJa)が$([Math]::Round([decimal]$_.shortageBytes / $script:GiB, 2)) GiB不足" }) -join '、'
            return [PSCustomObject]@{ label = '保存容量不足'; reason = $shortageText }
        }
        default { throw "Unknown recommendation class: $Class" }
    }
}

function Get-OfflineAiPreferenceScore {
    param([string]$Preference, [string]$Tier)
    $orders = @{
        Speed = @('compact', 'balanced', 'quality')
        Balanced = @('balanced', 'compact', 'quality')
        Quality = @('quality', 'balanced', 'compact')
    }
    return [Array]::IndexOf($orders[$Preference], $Tier)
}

function Get-OfflineAiModelRecommendations {
    param(
        [Parameter(Mandatory = $true)][object]$Catalog,
        [Parameter(Mandatory = $true)][object]$Target
    )
    $validated = Assert-OfflineAiModelCatalog -Catalog $Catalog
    $results = foreach ($model in $validated.selectableChats) {
        $storage = Get-OfflineAiStorageAssessment -Target $Target -ChatModel $model -EmbeddingModel $validated.requiredEmbedding -AppPayloadEstimate $validated.appPayloadEstimate
        $minimumRam = [decimal]$model.recommendation.minimumRamGiB
        $ramSupported = [decimal]$Target.ramGiB -ge $minimumRam
        $class = ''
        $classRank = 0
        if (-not $storage.passed) { $class = 'disk-insufficient'; $classRank = 5 }
        elseif (-not $ramSupported) { $class = 'ram-unsupported'; $classRank = 4 }
        elseif ($null -eq $Target.vramGiB) { $class = 'gpu-unknown'; $classRank = 3 }
        elseif ($Target.gpuVendor -eq 'None' -or [decimal]$Target.vramGiB -lt [decimal]$model.recommendation.partialGpuVramGiB) { $class = 'cpu-likely'; $classRank = 2 }
        elseif ([decimal]$Target.vramGiB -ge [decimal]$model.recommendation.fullGpuVramGiB) { $class = 'full-gpu-likely'; $classRank = 0 }
        else { $class = 'partial-gpu-likely'; $classRank = 1 }
        $presentation = Get-OfflineAiRecommendationPresentation -Class $class -Target $Target -Model $model -Storage $storage
        [PSCustomObject]@{
            model = [string]$model.name
            tier = [string]$model.recommendation.tier
            recommendationClass = $class
            recommendationClassLabel = [string]$presentation.label
            recommendationReason = [string]$presentation.reason
            supported = $ramSupported -and $storage.passed
            ramSupported = $ramSupported
            diskSupported = $storage.passed
            minimumRamGiB = $minimumRam
            fullGpuVramGiB = [decimal]$model.recommendation.fullGpuVramGiB
            partialGpuVramGiB = [decimal]$model.recommendation.partialGpuVramGiB
            estimatedDownloadBytes = [Int64]$model.recommendation.estimatedDownloadBytes
            transferEstimateBytes = [Int64]$storage.transferEstimateBytes
            storage = $storage
            catalogReviewedAt = [string]$model.recommendation.catalogReviewedAt
            notices = @($model.recommendation.notices)
            compatibilityNotices = @(Get-OfflineAiGpuCompatibilityNotices -Target $Target)
            sortClass = $classRank
            sortPreference = Get-OfflineAiPreferenceScore -Preference $Target.preference -Tier ([string]$model.recommendation.tier)
        }
    }
    $rank = 0
    return @($results | Sort-Object sortClass, sortPreference, estimatedDownloadBytes | ForEach-Object {
        $rank++
        $_ | Add-Member -NotePropertyName rank -NotePropertyValue $rank -PassThru
    })
}

function New-OfflineAiUnsupportedOverrideReceipt {
    param(
        [bool]$Used,
        [ValidateSet('not-required', 'interactive-second-confirmation', 'explicit-switch')][string]$ConfirmationMode = 'not-required'
    )
    if ($Used -and $ConfirmationMode -eq 'not-required') { throw 'unsupported override requires an explicit confirmation mode.' }
    if (-not $Used -and $ConfirmationMode -ne 'not-required') { throw 'unused override must use not-required confirmation mode.' }
    [object[]]$reasonCodes = @()
    if ($Used) { $reasonCodes = @('ram-below-minimum') }
    return [PSCustomObject]@{
        used = $Used
        reasonCodes = $reasonCodes
        confirmationMode = $ConfirmationMode
    }
}

function Assert-OfflineAiUnsupportedOverrideReceipt {
    param([Parameter(Mandatory = $true)][object]$Receipt)
    $used = [bool](Get-OfflineAiRequiredProperty -Value $Receipt -Name 'used' -Context 'unsupportedOverride')
    $reasonValue = Get-OfflineAiRequiredProperty -Value $Receipt -Name 'reasonCodes' -Context 'unsupportedOverride'
    [object[]]$reasons = @()
    if ($null -ne $reasonValue) { $reasons = @($reasonValue) }
    $mode = [string](Get-OfflineAiRequiredProperty -Value $Receipt -Name 'confirmationMode' -Context 'unsupportedOverride')
    if ($used) {
        if ($reasons.Count -ne 1 -or [string]$reasons[0] -ne 'ram-below-minimum') { throw 'unsupportedOverride contains an unsupported reason.' }
        if ($mode -notin @('interactive-second-confirmation', 'explicit-switch')) { throw 'unsupportedOverride contains an invalid confirmation mode.' }
    } elseif ($reasons.Count -ne 0 -or $mode -ne 'not-required') {
        throw 'unused unsupportedOverride receipt is internally inconsistent.'
    }
    return $true
}

function Resolve-OfflineAiChatSelection {
    param(
        [Parameter(Mandatory = $true)][object]$Catalog,
        [Parameter(Mandatory = $true)][object]$Target,
        [Parameter(Mandatory = $true)][string]$ChatModel,
        [switch]$AllowUnsupported,
        [ValidateSet('interactive-second-confirmation', 'explicit-switch')][string]$OverrideConfirmationMode = 'explicit-switch'
    )
    $recommendations = @(Get-OfflineAiModelRecommendations -Catalog $Catalog -Target $Target)
    $selected = @($recommendations | Where-Object { $_.model -eq $ChatModel })
    if ($selected.Count -ne 1) { throw "ChatModel is not a selectable catalog chat model: $ChatModel" }
    $item = $selected[0]
    if (-not $item.diskSupported) { throw "Selected chat model does not meet disk requirements: $ChatModel" }
    if (-not $item.ramSupported -and -not $AllowUnsupported) { throw "Selected chat model is below minimum RAM: $ChatModel" }
    $receipt = New-OfflineAiUnsupportedOverrideReceipt -Used:(-not $item.ramSupported) -ConfirmationMode $(if ($item.ramSupported) { 'not-required' } else { $OverrideConfirmationMode })
    return [PSCustomObject]@{ recommendation = $item; unsupportedOverride = $receipt }
}

Export-ModuleMember -Function @(
    'Assert-OfflineAiModelCatalog',
    'ConvertTo-OfflineAiTargetRequirements',
    'Get-OfflineAiStorageAssessment',
    'Get-OfflineAiModelRecommendations',
    'New-OfflineAiUnsupportedOverrideReceipt',
    'Assert-OfflineAiUnsupportedOverrideReceipt',
    'Resolve-OfflineAiChatSelection'
)
