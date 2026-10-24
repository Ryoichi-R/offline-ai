Set-StrictMode -Version Latest

function Get-OfflineAiRootFromTests {
    param([Parameter(Mandatory = $true)][string]$TestScriptRoot)
    return (Resolve-Path (Join-Path $TestScriptRoot '..\..')).Path
}

Export-ModuleMember -Function 'Get-OfflineAiRootFromTests'
