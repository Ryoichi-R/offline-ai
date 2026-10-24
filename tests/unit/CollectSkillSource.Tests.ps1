BeforeAll {
    Import-Module (Join-Path $PSScriptRoot '..\support\TestPaths.psm1') -Force
    $script:OfflineAiRoot = Get-OfflineAiRootFromTests -TestScriptRoot $PSScriptRoot
    $script:CollectSkillSourcePath = Join-Path $script:OfflineAiRoot '_internal\scripts\collect_skill_source.ps1'
}

Describe 'offline-ai collect skill source scoring' {
    It 'caps multi-keyword heading and phrase bonuses relative to the base score' {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "offline-ai-score-$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Path $tmp -Force | Out-Null
        try {
            [System.IO.File]::WriteAllText(
                (Join-Path $tmp 'primary.md'),
                "# alpha beta`nalpha beta details",
                [System.Text.UTF8Encoding]::new($false)
            )
            [System.IO.File]::WriteAllText(
                (Join-Path $tmp 'secondary.md'),
                "# reference`nalpha only",
                [System.Text.UTF8Encoding]::new($false)
            )

            $output = & pwsh -NoProfile -File $script:CollectSkillSourcePath -Query 'alpha beta' -OriginalQuery 'alpha beta' -SourceRoot $tmp 2>&1
            $exitCode = $LASTEXITCODE
            $result = ($output -join "`n") | ConvertFrom-Json
            $primary = @($result.matches) | Where-Object { $_.path -eq 'primary.md' } | Select-Object -First 1
            $baseScore = [Math]::Log(2.0)

            $exitCode | Should -Be 0
            $primary | Should -Not -BeNullOrEmpty
            [double]$primary.idfScore | Should -BeLessOrEqual ($baseScore * 5.0 + 0.000001)
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
