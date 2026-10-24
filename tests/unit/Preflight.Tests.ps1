BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:CommonModulePath = Join-Path $script:OfflineAiRoot '_internal\OfflineAi.Common.psm1'
    Import-Module $script:CommonModulePath -Force
}

Describe "Test-OfflineAiPreflight" {
    It "preserves the legacy four positional argument contract" {
        $result = Test-OfflineAiPreflight 16GB 20GB 8GB 10GB
        $result.Passed | Should -BeTrue
        $result.DiskPath | Should -Be 'C:\'
    }

    It "fails when RAM is below the minimum" {
        $result = Test-OfflineAiPreflight -RamBytes 8GB -FreeBytes 100GB
        $result.Passed | Should -BeFalse
        $result.Errors | Should -Contain "RAM"
    }

    It "fails when disk free space is below the minimum" {
        $result = Test-OfflineAiPreflight -RamBytes 32GB -FreeBytes 10GB
        $result.Passed | Should -BeFalse
        $result.Errors | Should -Contain "Disk"
    }

    It "passes when RAM and disk free space meet the minimums" {
        $result = Test-OfflineAiPreflight -RamBytes 32GB -FreeBytes 100GB
        $result.Passed | Should -BeTrue
        $result.Errors.Count | Should -Be 0
    }

    It "uses 16GB RAM and 20GB free disk as the default thresholds" {
        $result = Test-OfflineAiPreflight -RamBytes 16GB -FreeBytes 20GB
        $result.Passed | Should -BeTrue
        $result.Errors.Count | Should -Be 0
    }

    It "fails one byte below the default RAM threshold" {
        $result = Test-OfflineAiPreflight -RamBytes ([UInt64](16GB - 1)) -FreeBytes 20GB
        $result.Passed | Should -BeFalse
        $result.Errors | Should -Contain "RAM"
        $result.Errors | Should -Not -Contain "Disk"
    }

    It "fails one byte below the default disk threshold" {
        $result = Test-OfflineAiPreflight -RamBytes 16GB -FreeBytes ([UInt64](20GB - 1))
        $result.Passed | Should -BeFalse
        $result.Errors | Should -Not -Contain "RAM"
        $result.Errors | Should -Contain "Disk"
    }

    It "reports both RAM and disk errors" {
        $result = Test-OfflineAiPreflight -RamBytes 8GB -FreeBytes 10GB
        $result.Passed | Should -BeFalse
        $result.Errors | Should -Contain "RAM"
        $result.Errors | Should -Contain "Disk"
    }

    It "honors injected minimums" {
        $result = Test-OfflineAiPreflight -RamBytes 12GB -FreeBytes 12GB -MinimumRamBytes 8GB -MinimumFreeBytes 8GB
        $result.Passed | Should -BeTrue
        $result.Errors.Count | Should -Be 0
    }
}
