Set-StrictMode -Version Latest

function Resolve-OfflineAiFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-OfflineAiPathComparison {
    if ([System.IO.Path]::DirectorySeparatorChar -eq '\') {
        return [System.StringComparison]::OrdinalIgnoreCase
    }
    return [System.StringComparison]::Ordinal
}

function ConvertTo-OfflineAiNativeRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $separator = [string][System.IO.Path]::DirectorySeparatorChar
    return $Path.Replace('\', $separator).Replace('/', $separator)
}

function ConvertTo-OfflineAiCanonicalRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return $Path.Replace('\', '/')
}

function Assert-OfflineAiPathUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Target
    )
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $fullTarget = [System.IO.Path]::GetFullPath($Target)
    $pathComparison = Get-OfflineAiPathComparison
    if ([string]::Equals($fullTarget, $fullRoot, $pathComparison)) {
        return $fullTarget
    }
    $separator = [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullTarget.StartsWith($fullRoot + $separator, $pathComparison)) {
        throw "Refusing to write outside root: $fullTarget"
    }
    return $fullTarget
}

function Get-OfflineAiSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Find-OfflineAiCommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-OfflineAiRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $base = Resolve-OfflineAiFullPath -Path $BasePath
    $target = Resolve-OfflineAiFullPath -Path $Path
    if (-not $base.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $base += [System.IO.Path]::DirectorySeparatorChar
    }
    if (-not $target.StartsWith($base, (Get-OfflineAiPathComparison))) {
        throw "Path is outside base path: $target"
    }
    return ConvertTo-OfflineAiCanonicalRelativePath -Path ($target.Substring($base.Length))
}

function Get-OfflineAiNativeArchitecture {
    try {
        $architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        if ($architecture) {
            return $architecture.ToUpperInvariant()
        }
    } catch {
        # Windows PowerShell on older .NET Framework may not expose RuntimeInformation.
    }
    if ($env:PROCESSOR_ARCHITEW6432) {
        return ([string]$env:PROCESSOR_ARCHITEW6432).ToUpperInvariant()
    }
    if ($env:PROCESSOR_ARCHITECTURE) {
        return ([string]$env:PROCESSOR_ARCHITECTURE).ToUpperInvariant()
    }
    return "UNKNOWN"
}

function Get-OfflineAiTreeItems {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$SkipDirectoryNames = @()
    )

    $rootItem = Get-Item -LiteralPath $Root -Force -ErrorAction Stop
    $pending = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
    $pending.Push($rootItem)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in @(Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction Stop)) {
            $item
            if (-not $item.PSIsContainer) {
                continue
            }

            $linkTypeProperty = $item.PSObject.Properties['LinkType']
            $linkType = if ($null -ne $linkTypeProperty) { [string]$linkTypeProperty.Value } else { '' }
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or
                -not [string]::IsNullOrWhiteSpace($linkType)) {
                continue
            }
            if ($SkipDirectoryNames -notcontains $item.Name) {
                $pending.Push($item)
            }
        }
    }
}

function Assert-OfflineAiNoReparsePoints {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$SkipDirectoryNames = @()
    )
    $rootItem = Get-Item -LiteralPath $Root -Force -ErrorAction Stop
    $items = @($rootItem) + @(Get-OfflineAiTreeItems -Root $Root -SkipDirectoryNames $SkipDirectoryNames)
    foreach ($item in $items) {
        $linkTypeProperty = $item.PSObject.Properties['LinkType']
        $linkType = if ($null -ne $linkTypeProperty) { [string]$linkTypeProperty.Value } else { '' }
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or
            -not [string]::IsNullOrWhiteSpace($linkType)) {
            throw "Reparse point is not allowed in package input: $($item.FullName)"
        }
    }
}

function Copy-OfflineAiDirectoryFiltered {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [string[]]$SkipRelativePrefixes = @(),
        [string[]]$SkipFileNames = @(),
        [string[]]$SkipDirectoryNames = @(),
        [switch]$SkipDotDirectories
    )

    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        throw "コピー元ディレクトリが見つかりません: $SourceRoot"
    }
    Assert-OfflineAiNoReparsePoints -Root $SourceRoot
    foreach ($file in @(Get-ChildItem -LiteralPath $SourceRoot -File -Recurse)) {
        $relativePath = Get-OfflineAiRelativePath -BasePath $SourceRoot -Path $file.FullName
        $normalized = ConvertTo-OfflineAiCanonicalRelativePath -Path $relativePath
        $skip = $false
        $segments = @($normalized -split "/")
        $directorySegments = @()
        if ($segments.Count -gt 1) {
            $directorySegments = @($segments[0..($segments.Count - 2)])
        }
        foreach ($segment in $directorySegments) {
            if (($SkipDotDirectories -and $segment.StartsWith(".", [StringComparison]::Ordinal)) -or
                ($SkipDirectoryNames -contains $segment)) {
                $skip = $true
                break
            }
        }
        foreach ($prefix in $SkipRelativePrefixes) {
            $normalizedPrefix = ConvertTo-OfflineAiCanonicalRelativePath -Path $prefix
            if ($normalized.StartsWith($normalizedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                $skip = $true
                break
            }
        }
        if ($SkipFileNames -contains $file.Name) { $skip = $true }
        if ($skip) { continue }
        if (Test-OfflineAiSensitivePackageFile -RelativePath $relativePath) {
            throw "Sensitive file is not allowed in package input: $relativePath"
        }

        $destination = Join-Path $DestinationRoot $relativePath
        $destinationDirectory = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
            New-Item -Path $destinationDirectory -ItemType Directory -Force | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
    }
}

function Copy-OfflineAiDirectoryExact {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        throw "コピー元ディレクトリが見つかりません: $SourceRoot"
    }
    Assert-OfflineAiNoReparsePoints -Root $SourceRoot
    foreach ($file in @(Get-ChildItem -LiteralPath $SourceRoot -File -Recurse)) {
        $relativePath = Get-OfflineAiRelativePath -BasePath $SourceRoot -Path $file.FullName
        if (Test-OfflineAiSensitivePackageFile -RelativePath $relativePath) {
            throw "Sensitive file is not allowed in package input: $relativePath"
        }
        $destination = Join-Path $DestinationRoot $relativePath
        $destinationDirectory = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
            New-Item -Path $destinationDirectory -ItemType Directory -Force | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
    }
}

function ConvertTo-OfflineAiPolicyRegex {
    param([Parameter(Mandatory = $true)][string]$Pattern)
    $normalized = $Pattern.Replace('\', '/')
    $placeholder = 'OFFLINEAI_DOUBLESTAR_TOKEN'
    $escaped = [regex]::Escape($normalized)
    $escaped = $escaped.Replace('\*\*', $placeholder)
    $escaped = $escaped.Replace('\*', '[^/]*')
    $escaped = $escaped.Replace($placeholder, '.*')
    return '^' + $escaped + '$'
}

function Import-OfflineAiDistributionPolicy {
    param([Parameter(Mandatory = $true)][string]$PolicyPath)
    if (-not (Test-Path -LiteralPath $PolicyPath -PathType Leaf)) {
        throw "Distribution policy was not found: $PolicyPath"
    }
    $raw = Get-Content -LiteralPath $PolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$raw.schema_version -ne '1.0') {
        throw "Unsupported distribution policy schema_version: $($raw.schema_version)"
    }
    $entries = @()
    $seenPatterns = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($item in @($raw.entries)) {
        $pattern = [string]$item.pattern
        if ([string]::IsNullOrWhiteSpace($pattern)) {
            throw "Distribution policy entry has an empty pattern."
        }
        if (-not $seenPatterns.Add($pattern)) {
            throw "Distribution policy has a duplicate pattern: $pattern"
        }
        $distribution = [string]$item.distribution
        if ($distribution -notin @('both', 'public-source', 'runtime', 'excluded', 'release-only')) {
            throw "Distribution policy entry has an invalid distribution value: $distribution ($pattern)"
        }
        if ([string]::IsNullOrWhiteSpace([string]$item.reason)) {
            throw "Distribution policy entry is missing reason: $pattern"
        }
        if ([string]::IsNullOrWhiteSpace([string]$item.boundary_row)) {
            throw "Distribution policy entry is missing boundary_row: $pattern"
        }
        $entries += [PSCustomObject]@{
            pattern = $pattern
            distribution = $distribution
            reason = [string]$item.reason
            boundaryRow = [string]$item.boundary_row
            regex = ConvertTo-OfflineAiPolicyRegex -Pattern $pattern
        }
    }
    if ($entries.Count -eq 0) {
        throw "Distribution policy has no entries: $PolicyPath"
    }
    return [PSCustomObject]@{
        schemaVersion = [string]$raw.schema_version
        entries = $entries
        sourcePath = (Resolve-OfflineAiFullPath -Path $PolicyPath)
    }
}

function Get-OfflineAiDistributionClassification {
    param(
        [Parameter(Mandatory = $true)][object]$Policy,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    $canonical = ConvertTo-OfflineAiCanonicalRelativePath -Path $RelativePath
    $normalizedForMatch = $canonical.Normalize([System.Text.NormalizationForm]::FormC)
    $matched = @()
    foreach ($entry in $Policy.entries) {
        if ([regex]::IsMatch($normalizedForMatch, $entry.regex, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            $matched += $entry
        }
    }
    if ($matched.Count -eq 0) {
        return $null
    }
    if (@($matched | Where-Object { $_.distribution -eq 'excluded' }).Count -gt 0) {
        return 'excluded'
    }
    $distinctValues = @($matched.distribution | Select-Object -Unique)
    if ($distinctValues.Count -gt 1) {
        throw "Distribution policy is ambiguous for path '$canonical': $($distinctValues -join ', ')"
    }
    return $distinctValues[0]
}

function Test-OfflineAiRelativePathSafe {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $canonical = ConvertTo-OfflineAiCanonicalRelativePath -Path $RelativePath
    if ($canonical.Contains(':')) {
        throw "Path contains a colon (possible Alternate Data Stream or drive spec): $canonical"
    }
    if ($canonical.StartsWith('/')) {
        throw "Path must be relative: $canonical"
    }
    $reservedNames = @('CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9')
    foreach ($segment in @($canonical.Split('/'))) {
        if ($segment -eq '..' -or $segment -eq '.') {
            throw "Path contains a traversal segment: $canonical"
        }
        if ([string]::IsNullOrEmpty($segment)) {
            throw "Path contains an empty segment: $canonical"
        }
        $baseName = $segment.Split('.')[0]
        if ($reservedNames -contains $baseName.ToUpperInvariant()) {
            throw "Path contains a reserved device name segment: $canonical"
        }
    }
    return $true
}

function Get-OfflineAiDistributionInventory {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][object]$Policy
    )
    $generatedCacheDirectories = @('.pytest_cache')
    Assert-OfflineAiNoReparsePoints -Root $SourceRoot -SkipDirectoryNames $generatedCacheDirectories
    $files = @(Get-OfflineAiTreeItems -Root $SourceRoot -SkipDirectoryNames $generatedCacheDirectories |
        Where-Object { -not $_.PSIsContainer } |
        Sort-Object FullName)
    $classified = @()
    $unmatched = @()
    $seenCanonical = New-Object 'System.Collections.Generic.Dictionary[string,string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $files) {
        $relative = Get-OfflineAiRelativePath -BasePath $SourceRoot -Path $file.FullName
        Test-OfflineAiRelativePathSafe -RelativePath $relative | Out-Null
        $normalizedKey = $relative.Normalize([System.Text.NormalizationForm]::FormC)
        if ($seenCanonical.ContainsKey($normalizedKey) -and $seenCanonical[$normalizedKey] -cne $relative) {
            throw "Case-collision between paths that normalize to the same value: '$($seenCanonical[$normalizedKey])' and '$relative'"
        }
        $seenCanonical[$normalizedKey] = $relative
        $distribution = Get-OfflineAiDistributionClassification -Policy $Policy -RelativePath $relative
        if (-not $distribution) {
            $unmatched += $relative
            continue
        }
        $classified += [PSCustomObject]@{
            relativePath = $relative
            distribution = $distribution
            sizeBytes = $file.Length
            sha256 = if ($distribution -eq 'excluded') { $null } else { Get-OfflineAiSha256 -Path $file.FullName }
        }
    }
    return [PSCustomObject]@{
        classified = $classified
        unmatchedPaths = $unmatched
    }
}

function Get-OfflineAiRuntimePayloadEstimate {
    param([Parameter(Mandatory = $true)][object]$Inventory)

    $runtimeFiles = @($Inventory.classified |
        Where-Object { $_.distribution -in @('both', 'runtime') } |
        Sort-Object relativePath)
    if ($runtimeFiles.Count -eq 0) {
        throw 'Runtime payload inventory is empty.'
    }
    [Int64]$totalBytes = 0
    foreach ($item in $runtimeFiles) { $totalBytes += [Int64]$item.sizeBytes }
    $largest = @($runtimeFiles | Sort-Object @{ Expression = { [Int64]$_.sizeBytes }; Descending = $true }, relativePath)[0]
    return [PSCustomObject]@{
        totalBytes = $totalBytes
        largestFileBytes = [Int64]$largest.sizeBytes
        largestFile = [string]$largest.relativePath
        fileCount = $runtimeFiles.Count
    }
}

function Test-OfflineAiAppPayloadEstimate {
    param(
        [Parameter(Mandatory = $true)][object]$Inventory,
        [Parameter(Mandatory = $true)][object]$Expected
    )

    $actual = Get-OfflineAiRuntimePayloadEstimate -Inventory $Inventory
    foreach ($field in @('totalBytes', 'largestFileBytes', 'largestFile', 'fileCount')) {
        if ($Expected.PSObject.Properties.Name -notcontains $field) {
            throw "appPayloadEstimate is missing required field: $field"
        }
    }
    $passed = (
        [Int64]$Expected.totalBytes -eq $actual.totalBytes -and
        [Int64]$Expected.largestFileBytes -eq $actual.largestFileBytes -and
        [string]$Expected.largestFile -ceq $actual.largestFile -and
        [int]$Expected.fileCount -eq $actual.fileCount
    )
    return [PSCustomObject]@{
        passed = $passed
        expected = [PSCustomObject]@{
            totalBytes = [Int64]$Expected.totalBytes
            largestFileBytes = [Int64]$Expected.largestFileBytes
            largestFile = [string]$Expected.largestFile
            fileCount = [int]$Expected.fileCount
        }
        actual = $actual
    }
}

function Copy-OfflineAiDirectoryByDistribution {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][object]$Policy,
        [string[]]$IncludeDistributions = @('both', 'runtime')
    )

    $inventory = Get-OfflineAiDistributionInventory -SourceRoot $SourceRoot -Policy $Policy
    if ($inventory.unmatchedPaths.Count -gt 0) {
        throw "Distribution policy does not classify the following path(s). Add an entry to distribution-policy.json before packaging: $($inventory.unmatchedPaths -join ', ')"
    }

    foreach ($item in $inventory.classified) {
        if ($IncludeDistributions -notcontains $item.distribution) { continue }
        $sourceFile = Join-Path $SourceRoot (ConvertTo-OfflineAiNativeRelativePath -Path $item.relativePath)
        if (Test-OfflineAiSensitivePackageFile -RelativePath $item.relativePath) {
            throw "Sensitive file is not allowed in package input: $($item.relativePath)"
        }
        $destination = Join-Path $DestinationRoot (ConvertTo-OfflineAiNativeRelativePath -Path $item.relativePath)
        $destinationDirectory = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationDirectory -PathType Container)) {
            New-Item -Path $destinationDirectory -ItemType Directory -Force | Out-Null
        }
        Copy-Item -LiteralPath $sourceFile -Destination $destination
    }

    return [PSCustomObject]@{
        copiedCount = @($inventory.classified | Where-Object { $IncludeDistributions -contains $_.distribution }).Count
        totalCount = $inventory.classified.Count
    }
}

function New-OfflineAiFileInventory {
    param([Parameter(Mandatory = $true)][string]$Root)

    $items = @()
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -File -Recurse | Sort-Object FullName)) {
        $relativePath = Get-OfflineAiRelativePath -BasePath $Root -Path $file.FullName
        if ($relativePath -eq "checksums.sha256") { continue }
        $items += [PSCustomObject]@{
            relativePath = $relativePath
            sizeBytes = $file.Length
            sha256 = Get-OfflineAiSha256 -Path $file.FullName
        }
    }
    return $items
}

function Write-OfflineAiChecksums {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object[]]$Inventory
    )

    $lines = @()
    foreach ($item in $Inventory) {
        $lines += "$($item.sha256) *$($item.relativePath)"
    }
    Set-Content -LiteralPath (Join-Path $Root "checksums.sha256") -Value $lines -Encoding UTF8
}

function Export-OfflineAiModelLicenses {
    param(
        [Parameter(Mandatory = $true)][string]$ModelsRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )
    $manifestRoot = Join-Path $ModelsRoot "manifests"
    $blobRoot = Join-Path $ModelsRoot "blobs"
    if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
        return @()
    }
    $results = @()
    foreach ($manifestFile in @(Get-ChildItem -LiteralPath $manifestRoot -File -Recurse | Sort-Object FullName)) {
        $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        $licenseLayer = @($manifest.layers | Where-Object mediaType -eq "application/vnd.ollama.image.license" | Select-Object -First 1)
        if ($licenseLayer.Count -eq 0) {
            throw "Model manifest has no license layer: $($manifestFile.FullName)"
        }
        $digest = [string]$licenseLayer[0].digest
        if ($digest -notmatch '^sha256:[0-9a-fA-F]{64}$') {
            throw "Model license digest is invalid: $digest"
        }
        $blob = Join-Path $blobRoot $digest.Replace(":", "-").ToLowerInvariant()
        if (-not (Test-Path -LiteralPath $blob -PathType Leaf)) {
            throw "Model license blob was not found: $digest"
        }
        $actualLicenseHash = Get-OfflineAiSha256 -Path $blob
        if ($actualLicenseHash -ne $digest.Substring(7).ToLowerInvariant()) {
            throw "Model license blob digest mismatch: $digest"
        }
        $layerProperties = @($licenseLayer[0].PSObject.Properties.Name)
        if ($layerProperties -notcontains "size" -or [Int64]$licenseLayer[0].size -ne (Get-Item -LiteralPath $blob).Length) {
            throw "Model license blob size mismatch: $digest"
        }
        $manifestRelative = Get-OfflineAiRelativePath -BasePath $manifestRoot -Path $manifestFile.FullName
        $destination = Join-Path (Join-Path $DestinationRoot $manifestRelative) "LICENSE.txt"
        $null = Copy-OfflineAiFileNoConflict -Source $blob -Destination $destination
        $results += [PSCustomObject]@{
            modelManifest = $manifestRelative
            licenseDigest = $digest
            relativePath = Get-OfflineAiRelativePath -BasePath $DestinationRoot -Path $destination
            sha256 = Get-OfflineAiSha256 -Path $destination
            sizeBytes = (Get-Item -LiteralPath $destination).Length
        }
    }
    return $results
}

function Get-OfflineAiOsSummary {
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        return "$($os.Caption) $($os.Version)"
    } catch {
        return [System.Environment]::OSVersion.VersionString
    }
}

function Get-OfflineAiOllamaModelsPath {
    param([string]$ExplicitPath)
    if ($ExplicitPath) {
        return Resolve-OfflineAiFullPath -Path $ExplicitPath
    }
    if ($env:OLLAMA_MODELS) {
        return Resolve-OfflineAiFullPath -Path $env:OLLAMA_MODELS
    }
    if ($env:USERPROFILE) {
        return Resolve-OfflineAiFullPath -Path (Join-Path $env:USERPROFILE ".ollama\models")
    }
    throw "Unable to determine Ollama models path."
}

function Get-OfflineAiOllamaModelNames {
    param([string]$OllamaExe)
    if (-not $OllamaExe) {
        $OllamaExe = Find-OfflineAiCommandPath -Name "ollama"
    }
    if (-not $OllamaExe) {
        throw "ollama command was not found."
    }
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $OllamaExe list 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0) {
        throw "ollama list failed: $output"
    }
    $names = @()
    foreach ($line in @($output)) {
        $text = [string]$line
        if ($text -match '^\s*NAME\s+') { continue }
        if ($text -match '^\s*(\S+)') { $names += $matches[1] }
    }
    return $names
}

function Test-OfflineAiPackageChecksums {
    param([Parameter(Mandatory = $true)][string]$Root)
    Assert-OfflineAiNoReparsePoints -Root $Root
    $checksumPath = Join-Path $Root "checksums.sha256"
    if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
        throw "checksums.sha256 was not found: $checksumPath"
    }

    $lines = @(Get-Content -LiteralPath $checksumPath -Encoding UTF8 | Where-Object { $_.Trim() })
    $pathComparison = Get-OfflineAiPathComparison
    $pathComparer = if ($pathComparison -eq [System.StringComparison]::OrdinalIgnoreCase) {
        [System.StringComparer]::OrdinalIgnoreCase
    } else {
        [System.StringComparer]::Ordinal
    }
    $declaredPaths = New-Object 'System.Collections.Generic.HashSet[string]' ($pathComparer)
    $optionalSourcePrefix = "optional-components/"
    $optionalInstalledPrefix = "app/offline-ai/_internal/optional-components/"
    foreach ($line in $lines) {
        if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*(.+)$') {
            throw "Invalid checksums.sha256 line: $line"
        }
        $expected = $matches[1].ToLowerInvariant()
        $relative = ConvertTo-OfflineAiCanonicalRelativePath -Path ([string]$matches[2])
        if ([System.IO.Path]::IsPathRooted($relative) -or
            $relative -match '^[A-Za-z]:/' -or
            $relative.StartsWith('//', [System.StringComparison]::Ordinal)) {
            throw "Checksum path must be relative: $relative"
        }
        $segments = @($relative.Split('/'))
        if ($segments.Count -eq 0 -or @($segments | Where-Object { -not $_ -or $_ -in @(".", "..") }).Count -gt 0) {
            throw "Checksum path is not canonical: $relative"
        }
        $nativeRelative = ConvertTo-OfflineAiNativeRelativePath -Path $relative
        $target = Assert-OfflineAiPathUnderRoot -Root $Root -Target (Join-Path $Root $nativeRelative)
        $canonicalRelative = Get-OfflineAiRelativePath -BasePath $Root -Path $target
        if (-not [string]::Equals($canonicalRelative, $relative, [System.StringComparison]::Ordinal)) {
            throw "Checksum path is not canonical: $relative"
        }
        if ($canonicalRelative.StartsWith($optionalInstalledPrefix, $pathComparison)) {
            throw "Installed optional component copies must not be declared in package checksums: $canonicalRelative"
        }
        if (-not $declaredPaths.Add($canonicalRelative)) {
            throw "Duplicate checksum path: $canonicalRelative"
        }
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "Checksum target file was not found: $canonicalRelative"
        }
        $actual = Get-OfflineAiSha256 -Path $target
        if ($actual -ne $expected) {
            throw "Checksum mismatch: $canonicalRelative"
        }
    }

    foreach ($file in @(Get-ChildItem -LiteralPath $Root -Force -File -Recurse)) {
        $relative = Get-OfflineAiRelativePath -BasePath $Root -Path $file.FullName
        if ([string]::Equals($relative, "checksums.sha256", $pathComparison)) { continue }
        if ($declaredPaths.Contains($relative)) { continue }
        if ($relative.StartsWith($optionalSourcePrefix, $pathComparison)) {
            throw "Package optional component file is not declared in checksums.sha256: $relative"
        }
        if ($relative.StartsWith($optionalInstalledPrefix, $pathComparison)) {
            $sourceRelative = $optionalSourcePrefix + $relative.Substring($optionalInstalledPrefix.Length)
            $sourcePath = Join-Path $Root $sourceRelative
            if (-not $declaredPaths.Contains($sourceRelative) -or
                -not (Test-Path -LiteralPath $sourcePath -PathType Leaf) -or
                (Get-OfflineAiSha256 -Path $sourcePath) -ne (Get-OfflineAiSha256 -Path $file.FullName)) {
                throw "Installed optional component copy does not match package inventory: $relative"
            }
        }
    }
    return $true
}

function Test-OfflineAiFat32PackageMedia {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [object]$DiskInfo
    )
    try {
        if (-not $DiskInfo) {
            $qualifier = Split-Path -Qualifier (Resolve-OfflineAiFullPath -Path $Root)
            if (-not $qualifier) { return $false }
            $DiskInfo = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$qualifier'" -ErrorAction SilentlyContinue
        }
        if ($DiskInfo -and $DiskInfo.FileSystem -eq "FAT32") {
            return $true
        }
        return $false
    } catch {
        return $false
    }
}

function Copy-OfflineAiFileNoConflict {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [switch]$DryRun
    )
    $destDir = Split-Path -Parent $Destination
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $sourceHash = Get-OfflineAiSha256 -Path $Source
        $destHash = Get-OfflineAiSha256 -Path $Destination
        if ($sourceHash -eq $destHash) {
            return "SkippedSameHash"
        }
        throw "Destination file already exists with different content: $Destination"
    }
    if (-not (Test-Path -LiteralPath $destDir -PathType Container)) {
        if (-not $DryRun) {
            New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        }
    }
    if (-not $DryRun) {
        $part = "$Destination.part-$([guid]::NewGuid().ToString('N'))"
        try {
            Copy-Item -LiteralPath $Source -Destination $part
            $sourceHash = Get-OfflineAiSha256 -Path $Source
            $partHash = Get-OfflineAiSha256 -Path $part
            if ($sourceHash -ne $partHash) {
                throw "Temporary copy hash mismatch: $Destination"
            }
            if (Test-Path -LiteralPath $Destination) {
                throw "Destination appeared during copy: $Destination"
            }
            Move-Item -LiteralPath $part -Destination $Destination
        } finally {
            if (Test-Path -LiteralPath $part -PathType Leaf) {
                Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            }
        }
    }
    return "Copied"
}

function Test-OfflineAiSensitivePackageFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = ConvertTo-OfflineAiCanonicalRelativePath -Path $RelativePath
    $segments = @($normalized.Split('/'))
    $leaf = $segments[-1].ToLowerInvariant()
    $directorySegments = @()
    if ($segments.Count -gt 1) {
        $directorySegments = @($segments[0..($segments.Count - 2)] | ForEach-Object { $_.ToLowerInvariant() })
    }
    if (@($directorySegments | Where-Object { $_ -in @("secret", "secrets", "private") }).Count -gt 0) {
        return $true
    }
    if ($leaf -eq ".env" -or
        $leaf.StartsWith(".env.", [System.StringComparison]::Ordinal) -or
        $leaf -match '^secrets?(\.|$)' -or
        $leaf -match '^credentials?(\.|$)' -or
        $leaf -like 'service-account*.json' -or
        $leaf -like '*-credentials.json' -or
        $leaf -in @("id_rsa", "id_ed25519")) {
        return $true
    }

    $extension = [System.IO.Path]::GetExtension($leaf)
    return $extension -in @(".key", ".pem", ".pfx", ".p12", ".jks", ".keystore")
}

function Copy-OfflineAiOptionalComponent {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][string]$ComponentId,
        [Parameter(Mandatory = $true)][string]$StagingDirectory,
        [Parameter(Mandatory = $true)][string]$PackageRelativePath,
        [string[]]$RequiredFiles = @(),
        [switch]$DryRun
    )

    $resolvedSourceRoot = Resolve-OfflineAiFullPath -Path $SourceRoot
    $componentSource = Assert-OfflineAiPathUnderRoot `
        -Root $resolvedSourceRoot `
        -Target (Join-Path $resolvedSourceRoot (ConvertTo-OfflineAiNativeRelativePath -Path $StagingDirectory))
    if (-not (Test-Path -LiteralPath $componentSource -PathType Container)) {
        throw "Optional component staging directory was not found: $ComponentId ($componentSource)"
    }

    Assert-OfflineAiNoReparsePoints -Root $componentSource

    $files = @(Get-ChildItem -LiteralPath $componentSource -Force -File -Recurse | Sort-Object FullName)
    if ($files.Count -eq 0) {
        throw "Optional component staging directory is empty: $ComponentId"
    }

    foreach ($requiredFile in @($RequiredFiles)) {
        if ([string]::IsNullOrWhiteSpace($requiredFile)) {
            throw "Optional component requiredFiles contains an empty path: $ComponentId"
        }
        $requiredPath = Assert-OfflineAiPathUnderRoot `
            -Root $componentSource `
            -Target (Join-Path $componentSource (ConvertTo-OfflineAiNativeRelativePath -Path $requiredFile))
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Optional component required file was not found: $ComponentId ($requiredFile)"
        }
    }

    $resolvedDestinationRoot = Resolve-OfflineAiFullPath -Path $DestinationRoot
    $optionalDestinationRoot = Assert-OfflineAiPathUnderRoot `
        -Root $resolvedDestinationRoot `
        -Target (Join-Path $resolvedDestinationRoot "optional-components")
    $componentDestination = Assert-OfflineAiPathUnderRoot `
        -Root $optionalDestinationRoot `
        -Target (Join-Path $resolvedDestinationRoot (ConvertTo-OfflineAiNativeRelativePath -Path $PackageRelativePath))
    if ($componentDestination -eq $optionalDestinationRoot) {
        throw "Optional component destination must be below optional-components: $ComponentId"
    }
    $copyPlan = @()
    [Int64]$totalBytes = 0
    foreach ($file in $files) {
        $relative = Get-OfflineAiRelativePath -BasePath $componentSource -Path $file.FullName
        if (Test-OfflineAiSensitivePackageFile -RelativePath $relative) {
            throw "Optional component contains a sensitive file name: $ComponentId ($relative)"
        }
        $destination = Assert-OfflineAiPathUnderRoot `
            -Root $componentDestination `
            -Target (Join-Path $componentDestination $relative)
        $totalBytes += [Int64]$file.Length
        $copyPlan += [PSCustomObject]@{
            source = $file.FullName
            destination = $destination
            relativePath = Get-OfflineAiRelativePath -BasePath $resolvedDestinationRoot -Path $destination
            sizeBytes = [Int64]$file.Length
            sha256 = Get-OfflineAiSha256 -Path $file.FullName
        }
    }

    if (-not $DryRun) {
        foreach ($item in $copyPlan) {
            if (Test-Path -LiteralPath $item.destination -PathType Leaf) {
                $destinationHash = Get-OfflineAiSha256 -Path $item.destination
                if ($destinationHash -ne $item.sha256) {
                    throw "Destination file already exists with different content: $($item.destination)"
                }
            }
        }
        foreach ($item in $copyPlan) {
            $null = Copy-OfflineAiFileNoConflict -Source $item.source -Destination $item.destination
        }
    }

    return [PSCustomObject]@{
        id = $ComponentId
        status = if ($DryRun) { "Planned" } else { "Copied" }
        packageRelativePath = Get-OfflineAiRelativePath -BasePath $resolvedDestinationRoot -Path $componentDestination
        fileCount = $copyPlan.Count
        totalBytes = $totalBytes
        files = @($copyPlan | Select-Object relativePath, sizeBytes, sha256)
    }
}

function New-OfflineAiPublicSourceManifest {
    param([Parameter(Mandatory = $true)][object]$SourceManifest)

    $properties = @($SourceManifest.PSObject.Properties.Name)
    return [PSCustomObject]@{
        schemaVersion = 2
        packageKind = 'public-source'
        installable = $false
        productVersion = if ($properties -contains 'productVersion') { [string]$SourceManifest.productVersion } else { '' }
        createdAt = if ($properties -contains 'createdAt') { [string]$SourceManifest.createdAt } else { '' }
        os = if ($properties -contains 'os') { $SourceManifest.os } else { $null }
        models = @()
        modelLicenses = @()
        distributionProfile = [PSCustomObject]@{
            kind = 'public-source'
            thirdPartyBinariesIncluded = $false
            modelsIncluded = $false
            notes = @('利用者はonline staging PCでuser-built transport packageを別途作成します。')
        }
        install = if ($properties -contains 'install') { $SourceManifest.install } else { $null }
        installers = @()
        downloads = @()
        optionalComponents = @()
        notes = if ($properties -contains 'notes') { @($SourceManifest.notes) } else { @() }
    }
}

function Test-OfflineAiManifestForbiddenUserInput {
    param(
        [AllowNull()][object]$Value,
        [string]$Path = 'manifest'
    )
    if ($null -eq $Value) { return }
    $forbidden = @(
        'selection', 'targetRequirementsInput', 'gpuName', 'gpuVendor', 'vramGiB', 'ramGiB',
        'appFreeDiskGiB', 'modelsFreeDiskGiB', 'tempFreeDiskGiB', 'storageLayout', 'preference'
    )
    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) {
            if ([string]$key -in $forbidden) { throw "public release blocker: manifest contains target input field: $Path.$key" }
            Test-OfflineAiManifestForbiddenUserInput -Value $Value[$key] -Path "$Path.$key"
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        $index = 0
        foreach ($item in $Value) {
            Test-OfflineAiManifestForbiddenUserInput -Value $item -Path "$Path[$index]"
            $index++
        }
        return
    }
    if ($Value -is [PSCustomObject]) {
        foreach ($property in $Value.PSObject.Properties) {
            if ($property.Name -in $forbidden) { throw "public release blocker: manifest contains target input field: $Path.$($property.Name)" }
            Test-OfflineAiManifestForbiddenUserInput -Value $property.Value -Path "$Path.$($property.Name)"
        }
    }
}

function Assert-OfflineAiPublicArtifactContents {
    param([Parameter(Mandatory = $true)][string]$Root)

    foreach ($relative in @('installers', 'ollama-models', 'optional-components')) {
        if (Test-Path -LiteralPath (Join-Path $Root $relative)) {
            throw "public release blocker: third-party payload directory is present: $relative"
        }
    }
    $manifestPath = Join-Path $Root 'manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw 'public release blocker: manifest.json is missing'
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$manifest.schemaVersion -ne 2 -or [string]$manifest.packageKind -ne 'public-source' -or [bool]$manifest.installable) {
        throw 'public release blocker: public-source manifest contract is invalid'
    }
    Test-OfflineAiManifestForbiddenUserInput -Value $manifest
    $binary = Get-ChildItem -LiteralPath $Root -File -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension.ToLowerInvariant() -in @('.exe', '.msi', '.dll', '.bin') } |
        Select-Object -First 1
    if ($binary) { throw "public release blocker: binary payload is present: $($binary.FullName)" }
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -File -Recurse -Force -ErrorAction Stop)) {
        $relative = Get-OfflineAiRelativePath -BasePath $Root -Path $file.FullName
        if (Test-OfflineAiSensitivePackageFile -RelativePath $relative) {
            throw "public release blocker: sensitive file is present: $relative"
        }
        if ($file.Length -ge 2) {
            $stream = [System.IO.File]::OpenRead($file.FullName)
            try {
                if ($stream.ReadByte() -eq 0x4D -and $stream.ReadByte() -eq 0x5A) {
                    throw "public release blocker: PE binary payload is present: $relative"
                }
            } finally { $stream.Dispose() }
        }
    }
    return $true
}

function Test-OfflineAiPreflight {
    param(
        [Nullable[UInt64]]$RamBytes,
        [Nullable[UInt64]]$FreeBytes,
        [UInt64]$MinimumRamBytes = 16GB,
        [UInt64]$MinimumFreeBytes = 20GB,
        [string]$DiskPath = "C:\"
    )
    if ($null -eq $RamBytes) {
        $RamBytes = [UInt64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    }
    if ($null -eq $FreeBytes) {
        $resolvedDiskPath = Resolve-OfflineAiFullPath -Path $DiskPath
        $root = [System.IO.Path]::GetPathRoot($resolvedDiskPath)
        if (-not $root) {
            throw "Unable to determine destination volume: $DiskPath"
        }
        $FreeBytes = [UInt64]([System.IO.DriveInfo]::new($root)).AvailableFreeSpace
    }

    $errors = @()
    if ($RamBytes -lt $MinimumRamBytes) {
        $errors += "RAM"
    }
    if ($FreeBytes -lt $MinimumFreeBytes) {
        $errors += "Disk"
    }

    return [PSCustomObject]@{
        Passed = ($errors.Count -eq 0)
        Errors = $errors
        RamGB = [math]::Round($RamBytes / 1GB, 1)
        FreeGB = [math]::Round($FreeBytes / 1GB, 1)
        DiskPath = $DiskPath
    }
}

function Get-OfflineAiNvidiaGpu {
    param([scriptblock]$Query)
    $available = $false
    $vramGB = 0
    try {
        if ($Query) {
            $vramMB = & $Query
            $exitCode = 0
        } else {
            $vramMB = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null)
            $exitCode = $LASTEXITCODE
        }
        if ($exitCode -eq 0 -and $vramMB) {
            $first = @($vramMB)[0]
            $vramGB = [math]::Floor([int]([string]$first).Trim() / 1024)
            $available = $true
        }
    } catch {
        $available = $false
        $vramGB = 0
    }
    return [PSCustomObject]@{
        Available = $available
        VramGB = $vramGB
    }
}

Export-ModuleMember -Function @(
    'Resolve-OfflineAiFullPath',
    'Assert-OfflineAiPathUnderRoot',
    'Get-OfflineAiSha256',
    'Find-OfflineAiCommandPath',
    'Get-OfflineAiNativeArchitecture',
    'Get-OfflineAiRelativePath',
    'Assert-OfflineAiNoReparsePoints',
    'Copy-OfflineAiDirectoryFiltered',
    'Copy-OfflineAiDirectoryExact',
    'ConvertTo-OfflineAiPolicyRegex',
    'Import-OfflineAiDistributionPolicy',
    'Get-OfflineAiDistributionClassification',
    'Test-OfflineAiRelativePathSafe',
    'Get-OfflineAiDistributionInventory',
    'Get-OfflineAiRuntimePayloadEstimate',
    'Test-OfflineAiAppPayloadEstimate',
    'Copy-OfflineAiDirectoryByDistribution',
    'New-OfflineAiFileInventory',
    'Write-OfflineAiChecksums',
    'Export-OfflineAiModelLicenses',
    'Get-OfflineAiOsSummary',
    'Get-OfflineAiOllamaModelsPath',
    'Get-OfflineAiOllamaModelNames',
    'Test-OfflineAiPackageChecksums',
    'Test-OfflineAiFat32PackageMedia',
    'Copy-OfflineAiFileNoConflict',
    'Test-OfflineAiSensitivePackageFile',
    'Copy-OfflineAiOptionalComponent',
    'New-OfflineAiPublicSourceManifest',
    'Assert-OfflineAiPublicArtifactContents',
    'Test-OfflineAiPreflight',
    'Get-OfflineAiNvidiaGpu'
)
