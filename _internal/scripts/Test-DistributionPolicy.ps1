<#
.SYNOPSIS
    offline-ai の配布policy検証（V-1/V-2/V-4/V-5/V-6）を実行し、run receiptを出力する。

.DESCRIPTION
    秘密情報scan（V-3相当）はこのscriptの責務外とする。formal audit側のconsent後gateへ委譲し、
    sensitive_scan_performed=false をreceiptへ記録する。raw scanner出力・検出値は扱わない。

    V-4（path traversal / 絶対path / case collision / ADS / 予約device名拒否）と
    V-5（reparse point拒否）は、内部で呼び出す Get-OfflineAiDistributionInventory /
    Test-OfflineAiRelativePathSafe / Assert-OfflineAiNoReparsePoints が例外で検証する。
    例外はここで握りつぶさず、そのまま呼び出し元へ伝播させ fail-closedにする。

    V-6はdownload-manifest.jsonのappPayloadEstimateと、distributionがruntime/bothの
    現行inventory（total、largest file、file count）を完全一致で照合する。

.PARAMETER SourceRoot
    検証対象のoffline-ai root。省略時はこのscriptの2階層上（offline-ai/）。

.PARAMETER Audience
    'public-source' | 'runtime' | 'release'。candidateとして許可するdistribution集合を決める。

.PARAMETER ReceiptPath
    run receiptの出力先。省略時は標準出力のみ。

.EXAMPLE
    pwsh -NoProfile -File _internal/scripts/Test-DistributionPolicy.ps1 -Audience public-source
#>
[CmdletBinding()]
param(
    [string]$SourceRoot,
    [ValidateSet('public-source', 'runtime', 'release')]
    [string]$Audience = 'public-source',
    [string]$ReceiptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = $PSScriptRoot
$offlineAiRoot = if ($SourceRoot) { [IO.Path]::GetFullPath($SourceRoot) } else { [IO.Path]::GetFullPath((Join-Path $scriptRoot '..\..')) }
$policyPath = Join-Path $offlineAiRoot '_internal\distribution-policy.json'

Import-Module (Join-Path $offlineAiRoot '_internal\OfflineAi.Common.psm1') -Force

$allowedByAudience = @{
    'public-source' = @('both', 'public-source')
    'runtime'       = @('both', 'runtime')
    'release'       = @('both', 'runtime', 'release-only')
}

$policy = Import-OfflineAiDistributionPolicy -PolicyPath $policyPath
$inventory = Get-OfflineAiDistributionInventory -SourceRoot $offlineAiRoot -Policy $policy

$v1Pass = ($inventory.unmatchedPaths.Count -eq 0)
$catalogPath = Join-Path $offlineAiRoot '_internal\download-manifest.json'
$catalog = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
$appPayloadGate = Test-OfflineAiAppPayloadEstimate -Inventory $inventory -Expected $catalog.appPayloadEstimate
$allowed = $allowedByAudience[$Audience]
$candidateFiles = @($inventory.classified | Where-Object { $allowed -contains $_.distribution } | Sort-Object relativePath)

# candidate_digest（決定的）: policy digest + sorted file listのみで構成する。
# generated_at・tool versionはrun receipt側にのみ含め、digest対象から除外する。
$policyDigestInput = (@($policy.entries | Sort-Object pattern | ForEach-Object {
    "$($_.pattern)|$($_.distribution)|$($_.reason)|$($_.boundaryRow)"
})) -join "`n"
$canonicalPayload = (@($candidateFiles | ForEach-Object {
    "$($_.relativePath)|$($_.sizeBytes)|$($_.sha256)"
})) -join "`n"

$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $policyDigest = 'sha256:' + ([System.BitConverter]::ToString($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($policyDigestInput)))).Replace('-', '').ToLowerInvariant()
    $candidateDigest = 'sha256:' + ([System.BitConverter]::ToString($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes("$policyDigest`n$canonicalPayload")))).Replace('-', '').ToLowerInvariant()
} finally {
    $sha256.Dispose()
}

$status = if ($v1Pass -and $appPayloadGate.passed) { 'READY' } else { 'BLOCKED' }

$result = [ordered]@{
    schema_version = '1.0'
    status = $status
    audience = $Audience
    source_root = $offlineAiRoot
    generated_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    policy_digest = $policyDigest
    candidate_digest = $candidateDigest
    file_count = $candidateFiles.Count
    sensitive_scan_performed = $false
    gates = [ordered]@{
        'V-1_classification_coverage' = [ordered]@{
            status = if ($v1Pass) { 'PASS' } else { 'FAIL' }
            unmatched_paths = @($inventory.unmatchedPaths)
        }
        'V-2_allowlist_selection' = [ordered]@{
            status = 'PASS'
            allowed_distributions = $allowed
        }
        'V-4_path_safety' = [ordered]@{ status = 'PASS' }
        'V-5_reparse_point' = [ordered]@{ status = 'PASS' }
        'V-6_app_payload_estimate' = [ordered]@{
            status = if ($appPayloadGate.passed) { 'PASS' } else { 'FAIL' }
            expected = $appPayloadGate.expected
            actual = $appPayloadGate.actual
        }
        'V-3_sensitive_scan' = [ordered]@{
            status = 'NOT_RUN'
            note = '本文scanは行わない。formal audit側のconsent後gateへ委譲する。'
        }
    }
    files = @($candidateFiles | ForEach-Object {
        [ordered]@{ path = $_.relativePath; size = $_.sizeBytes; sha256 = $_.sha256; distribution = $_.distribution }
    })
}

$json = ($result | ConvertTo-Json -Depth 10) -replace "`r`n", "`n"

if ($ReceiptPath) {
    $parent = Split-Path -Parent $ReceiptPath
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -Path $parent -ItemType Directory -Force | Out-Null
    }
    [IO.File]::WriteAllText($ReceiptPath, $json + "`n", [Text.UTF8Encoding]::new($false))
}

Write-Output $json

if (-not $v1Pass) {
    throw "Distribution policy verification failed (V-1): $($inventory.unmatchedPaths.Count) unclassified path(s). See gates.V-1_classification_coverage.unmatched_paths."
}
if (-not $appPayloadGate.passed) {
    throw 'Distribution policy verification failed (V-6): appPayloadEstimate does not match the runtime/both inventory.'
}
