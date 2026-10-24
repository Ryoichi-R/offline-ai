Set-StrictMode -Version Latest

function Resolve-OllamaModelReference {
    param([Parameter(Mandatory = $true)][string]$Name)

    $trimmed = $Name.Trim()
    if (-not $trimmed) {
        throw "Model name is empty."
    }

    $modelPart = $trimmed
    $tag = "latest"
    $colonIndex = $trimmed.LastIndexOf(":")
    if ($colonIndex -gt 0) {
        $modelPart = $trimmed.Substring(0, $colonIndex)
        $tag = $trimmed.Substring($colonIndex + 1)
    }
    if (-not $modelPart -or -not $tag) {
        throw "Invalid Ollama model reference: $Name"
    }

    $repository = $modelPart
    if ($repository.IndexOf("/") -lt 0) {
        $repository = "library/$repository"
    }

    return [PSCustomObject]@{
        name = $trimmed
        repository = $repository
        tag = $tag
    }
}

function Convert-OllamaDigestToBlobFileName {
    param([Parameter(Mandatory = $true)][string]$Digest)

    if ($Digest -notmatch '^sha256:([0-9a-fA-F]{64})$') {
        throw "Unsupported or invalid digest: $Digest"
    }
    return "sha256-$($matches[1].ToLowerInvariant())"
}

function Get-BytesSha256 {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-FileSha256Lower {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Join-RegistryFixturePath {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryRoot,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][ValidateSet("manifests", "blobs")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $path = Join-Path $RegistryRoot "v2"
    foreach ($segment in @($Repository -split '/')) {
        $path = Join-Path $path $segment
    }
    $path = Join-Path $path $Kind
    if ($Kind -eq "blobs") {
        $path = Join-Path $path (Convert-OllamaDigestToBlobFileName -Digest $Name)
    } else {
        $path = Join-Path $path $Name
    }
    return $path
}

function Get-OllamaRegistryUri {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryBaseUrl,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][ValidateSet("manifests", "blobs")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $base = $RegistryBaseUrl.TrimEnd("/")
    return "$base/v2/$Repository/$Kind/$Name"
}

function Invoke-OllamaRegistryWebRequest {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][hashtable]$Headers,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$RetryCount = 3
    )
    $oldProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        } catch {
        }
        for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
            try {
                Invoke-WebRequest -Uri $Uri -Headers $Headers -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
                return
            } catch {
                if ($attempt -ge $RetryCount) {
                    throw
                }
                Start-Sleep -Seconds ([Math]::Min(5, $attempt * 2))
            }
        }
    } finally {
        $ProgressPreference = $oldProgress
    }
}

function Read-OllamaRegistryBytes {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryBaseUrl,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][ValidateSet("manifests", "blobs")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$RegistryRoot = ""
    )

    if ($RegistryRoot) {
        $fixturePath = Join-RegistryFixturePath -RegistryRoot $RegistryRoot -Repository $Repository -Kind $Kind -Name $Name
        if (-not (Test-Path -LiteralPath $fixturePath -PathType Leaf)) {
            throw "Registry fixture was not found: $fixturePath"
        }
        return [System.IO.File]::ReadAllBytes($fixturePath)
    }

    $uri = Get-OllamaRegistryUri -RegistryBaseUrl $RegistryBaseUrl -Repository $Repository -Kind $Kind -Name $Name
    $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-registry-$([guid]::NewGuid().ToString('N')).tmp"
    try {
        $headers = @{
            Accept = "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/json, application/octet-stream"
        }
        Invoke-OllamaRegistryWebRequest -Uri $uri -Headers $headers -OutFile $tempFile
        return [System.IO.File]::ReadAllBytes($tempFile)
    } finally {
        if (Test-Path -LiteralPath $tempFile -PathType Leaf) {
            Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
        }
    }
}

function Save-OllamaRegistryContentToFile {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryBaseUrl,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][ValidateSet("manifests", "blobs")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$RegistryRoot = "",
        [int]$RetryCount = 3
    )

    if ($RegistryRoot) {
        $fixturePath = Join-RegistryFixturePath -RegistryRoot $RegistryRoot -Repository $Repository -Kind $Kind -Name $Name
        if (-not (Test-Path -LiteralPath $fixturePath -PathType Leaf)) {
            throw "Registry fixture was not found: $fixturePath"
        }
        Copy-Item -LiteralPath $fixturePath -Destination $Destination
        return
    }

    $uri = Get-OllamaRegistryUri -RegistryBaseUrl $RegistryBaseUrl -Repository $Repository -Kind $Kind -Name $Name
    $headers = @{
        Accept = "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/json, application/octet-stream"
    }
    Invoke-OllamaRegistryWebRequest -Uri $uri -Headers $headers -OutFile $Destination -RetryCount $RetryCount
}

function Get-OllamaRegistryManifest {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryBaseUrl,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Tag,
        [string]$RegistryRoot = ""
    )

    $bytes = Read-OllamaRegistryBytes -RegistryBaseUrl $RegistryBaseUrl -Repository $Repository -Kind "manifests" -Name $Tag -RegistryRoot $RegistryRoot
    $jsonText = [System.Text.Encoding]::UTF8.GetString($bytes)
    $json = $jsonText | ConvertFrom-Json
    return [PSCustomObject]@{
        bytes = $bytes
        json = $json
        sha256 = Get-BytesSha256 -Bytes $bytes
    }
}

function Get-OllamaManifestDescriptors {
    param([Parameter(Mandatory = $true)]$ManifestJson)

    $items = @()
    if ($ManifestJson.config -and $ManifestJson.config.digest) {
        $items += [PSCustomObject]@{
            kind = "config"
            digest = [string]$ManifestJson.config.digest
            size = if ($null -ne $ManifestJson.config.size) { [Int64]$ManifestJson.config.size } else { 0 }
            mediaType = [string]$ManifestJson.config.mediaType
        }
    }
    foreach ($layer in @($ManifestJson.layers)) {
        if (-not $layer.digest) { continue }
        $items += [PSCustomObject]@{
            kind = "layer"
            digest = [string]$layer.digest
            size = if ($null -ne $layer.size) { [Int64]$layer.size } else { 0 }
            mediaType = [string]$layer.mediaType
        }
    }

    foreach ($item in $items) {
        $null = Convert-OllamaDigestToBlobFileName -Digest $item.digest
    }
    return $items
}

function Save-OllamaManifestForOfflineInstall {
    param(
        [Parameter(Mandatory = $true)][byte[]]$Bytes,
        [Parameter(Mandatory = $true)][string]$DestinationModelsRoot,
        [Parameter(Mandatory = $true)][string]$RegistryHost,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Tag,
        [switch]$DryRun
    )

    $destination = Join-Path $DestinationModelsRoot "manifests"
    $destination = Join-Path $destination $RegistryHost
    foreach ($segment in @($Repository -split '/')) {
        $destination = Join-Path $destination $segment
    }
    $destination = Join-Path $destination $Tag

    if ($DryRun) {
        return [PSCustomObject]@{ path = $destination; status = "Planned"; sha256 = Get-BytesSha256 -Bytes $Bytes }
    }

    $directory = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -Path $directory -ItemType Directory -Force | Out-Null
    }
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $existing = Get-FileSha256Lower -Path $destination
        $incoming = Get-BytesSha256 -Bytes $Bytes
        if ($existing -eq $incoming) {
            return [PSCustomObject]@{ path = $destination; status = "SkippedSameHash"; sha256 = $incoming }
        }
        throw "Manifest already exists with different content: $destination"
    }
    [System.IO.File]::WriteAllBytes($destination, $Bytes)
    return [PSCustomObject]@{ path = $destination; status = "Saved"; sha256 = Get-BytesSha256 -Bytes $Bytes }
}

function Save-OllamaBlob {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryBaseUrl,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Digest,
        [Parameter(Mandatory = $true)][string]$DestinationModelsRoot,
        [string]$RegistryRoot = "",
        [int]$RetryCount = 3,
        [switch]$DryRun
    )

    $fileName = Convert-OllamaDigestToBlobFileName -Digest $Digest
    $expected = $fileName.Substring("sha256-".Length)
    $destination = Join-Path (Join-Path $DestinationModelsRoot "blobs") $fileName

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $actualExisting = Get-FileSha256Lower -Path $destination
        if ($actualExisting -eq $expected) {
            return [PSCustomObject]@{ path = $destination; digest = $Digest; status = "SkippedSameHash"; sizeBytes = (Get-Item -LiteralPath $destination).Length }
        }
        throw "Blob already exists with different content: $destination"
    }

    if ($DryRun) {
        return [PSCustomObject]@{ path = $destination; digest = $Digest; status = "Planned"; sizeBytes = 0 }
    }

    $directory = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -Path $directory -ItemType Directory -Force | Out-Null
    }

    if ($RetryCount -lt 1) {
        throw "RetryCount must be at least 1."
    }

    $part = "$destination.part-$([guid]::NewGuid().ToString('N'))"
    try {
        $verified = $false
        for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
            try {
                if (Test-Path -LiteralPath $part -PathType Leaf) {
                    Remove-Item -LiteralPath $part -Force
                }
                Save-OllamaRegistryContentToFile -RegistryBaseUrl $RegistryBaseUrl -Repository $Repository -Kind "blobs" -Name $Digest -Destination $part -RegistryRoot $RegistryRoot -RetryCount 1
            } catch {
                if (Test-Path -LiteralPath $part -PathType Leaf) {
                    Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
                }
                if ($attempt -ge $RetryCount) {
                    throw
                }
                if (-not $RegistryRoot) {
                    Start-Sleep -Seconds ([Math]::Min(5, $attempt * 2))
                }
                continue
            }

            # hash読取り自体のローカルI/O失敗は再取得せず、そのまま伝播する。
            $actual = Get-FileSha256Lower -Path $part
            if ($actual -eq $expected) {
                $verified = $true
                break
            }
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            if ($attempt -ge $RetryCount) {
                throw "Blob digest mismatch: $Digest"
            }
            if (-not $RegistryRoot) {
                Start-Sleep -Seconds ([Math]::Min(5, $attempt * 2))
            }
        }
        if (-not $verified) {
            throw "Blob digest mismatch: $Digest"
        }
        # 最終配置のローカルI/O失敗は巨大blobの再取得対象にしない。
        Move-Item -LiteralPath $part -Destination $destination
    } finally {
        if (Test-Path -LiteralPath $part -PathType Leaf) {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
    }

    return [PSCustomObject]@{ path = $destination; digest = $Digest; status = "Saved"; sizeBytes = (Get-Item -LiteralPath $destination).Length }
}

function Save-OllamaModelFromRegistry {
    param(
        [Parameter(Mandatory = $true)][string]$ModelName,
        [Parameter(Mandatory = $true)][string]$DestinationModelsRoot,
        [string]$RegistryBaseUrl = "https://registry.ollama.ai",
        [string]$RegistryRoot = "",
        [string]$ExpectedManifestDigest = "",
        [string]$ExpectedConfigDigest = "",
        [string]$ExpectedLicenseLayerDigest = "",
        [switch]$DryRun
    )

    $reference = Resolve-OllamaModelReference -Name $ModelName
    $registryUri = [System.Uri]$RegistryBaseUrl
    $registryHost = $registryUri.Host
    if (-not $registryHost) {
        $registryHost = "registry.ollama.ai"
    }

    $manifest = Get-OllamaRegistryManifest -RegistryBaseUrl $RegistryBaseUrl -Repository $reference.repository -Tag $reference.tag -RegistryRoot $RegistryRoot
    $actualManifestDigest = "sha256:$($manifest.sha256)"
    if ($ExpectedManifestDigest -and $ExpectedManifestDigest -cnotmatch '^sha256:[0-9a-f]{64}$') {
        throw "Expected manifest digest is invalid: $ExpectedManifestDigest"
    }
    if ($ExpectedManifestDigest -and $actualManifestDigest -cne $ExpectedManifestDigest) {
        throw "Registry manifest digest drift detected for ${ModelName}: expected $ExpectedManifestDigest, actual $actualManifestDigest"
    }
    if ($ExpectedConfigDigest -and $ExpectedConfigDigest -cnotmatch '^sha256:[0-9a-f]{64}$') {
        throw "Expected config digest is invalid: $ExpectedConfigDigest"
    }
    $actualConfigDigest = if ($manifest.json.config) { [string]$manifest.json.config.digest } else { '' }
    if ($ExpectedConfigDigest -and $actualConfigDigest -cne $ExpectedConfigDigest) {
        throw "Registry config digest drift detected for ${ModelName}: expected $ExpectedConfigDigest, actual $actualConfigDigest"
    }
    if ($ExpectedLicenseLayerDigest -and $ExpectedLicenseLayerDigest -cnotmatch '^sha256:[0-9a-f]{64}$') {
        throw "Expected license layer digest is invalid: $ExpectedLicenseLayerDigest"
    }
    if ($ExpectedLicenseLayerDigest) {
        $licenseLayers = @($manifest.json.layers | Where-Object mediaType -eq 'application/vnd.ollama.image.license')
        if ($licenseLayers.Count -ne 1 -or [string]$licenseLayers[0].digest -cne $ExpectedLicenseLayerDigest) {
            $actualLicenseDigest = if ($licenseLayers.Count -eq 1) { [string]$licenseLayers[0].digest } else { "count=$($licenseLayers.Count)" }
            throw "Registry license layer drift detected for ${ModelName}: expected $ExpectedLicenseLayerDigest, actual $actualLicenseDigest"
        }
    }
    $descriptors = @(Get-OllamaManifestDescriptors -ManifestJson $manifest.json)
    if ($descriptors.Count -eq 0) {
        throw "Registry manifest contains no descriptors: $ModelName"
    }
    $manifestSave = Save-OllamaManifestForOfflineInstall -Bytes $manifest.bytes -DestinationModelsRoot $DestinationModelsRoot -RegistryHost $registryHost -Repository $reference.repository -Tag $reference.tag -DryRun:$DryRun

    $blobResults = @()
    foreach ($descriptor in $descriptors) {
        $blobResults += Save-OllamaBlob -RegistryBaseUrl $RegistryBaseUrl -Repository $reference.repository -Digest $descriptor.digest -DestinationModelsRoot $DestinationModelsRoot -RegistryRoot $RegistryRoot -DryRun:$DryRun
    }

    $totalBytes = 0
    $largestBlobBytes = 0
    foreach ($descriptor in $descriptors) {
        $totalBytes += [Int64]$descriptor.size
        if ([Int64]$descriptor.size -gt $largestBlobBytes) {
            $largestBlobBytes = [Int64]$descriptor.size
        }
    }
    if ($largestBlobBytes -le 0) {
        throw "Registry manifest descriptor sizes are invalid: $ModelName"
    }

    return [PSCustomObject]@{
        name = $reference.name
        repository = $reference.repository
        tag = $reference.tag
        registry = $RegistryBaseUrl
        manifestDigest = $actualManifestDigest
        manifest = $manifestSave
        descriptors = $descriptors
        blobs = $blobResults
        blobCount = $descriptors.Count
        totalBytes = $totalBytes
        largestBlobBytes = $largestBlobBytes
    }
}

Export-ModuleMember -Function @(
    'Resolve-OllamaModelReference',
    'Convert-OllamaDigestToBlobFileName',
    'Get-OllamaManifestDescriptors',
    'Save-OllamaManifestForOfflineInstall',
    'Save-OllamaBlob',
    'Save-OllamaModelFromRegistry'
)
