<#
.SYNOPSIS
    skill-source ディレクトリ内のファイルをキーワード検索し、JSON で結果を返す。
.PARAMETER Query
    スペース区切りのキーワード文字列
.PARAMETER SourceRoot
    検索対象のルートディレクトリ
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Query,

    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $false)]
    [string]$OriginalQuery = "",

    [switch]$EnablePhraseBonus = $true,

    [switch]$EnableDenseSnippet = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# --- 出力用オブジェクト ---
$result = @{
    status          = "ok"
    matches         = @()
    scannedFileCount = 0
    message         = ""
}

# $SourceRoot を正規化（1回だけ実行）
$resolvedSourceRoot = (Resolve-Path $SourceRoot -ErrorAction Stop).Path

function Test-SafeHitPath {
    param([string]$HitPath)
    $joined = Join-Path $resolvedSourceRoot $HitPath
    try {
        $resolved = (Resolve-Path -LiteralPath $joined -ErrorAction Stop).Path
    } catch {
        return $null  # パスが存在しない
    }
    # 正規化: [IO.Path]::GetFullPath で ../ 等を解決
    $normalizedResolved = [System.IO.Path]::GetFullPath($resolved)
    # 境界付きプレフィクス検証: ディレクトリ区切り文字を末尾に付与して比較
    $sep = [System.IO.Path]::DirectorySeparatorChar
    $rootPrefix = $resolvedSourceRoot.TrimEnd($sep, [System.IO.Path]::AltDirectorySeparatorChar) + $sep
    if (-not $normalizedResolved.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Warning "SourceRoot外のパスを拒否: $HitPath -> $normalizedResolved"
        return $null
    }
    return $normalizedResolved
}

# SourceRoot 存在チェック
if (-not (Test-Path $SourceRoot -PathType Container)) {
    $result.status  = "empty"
    $result.message = "skill-source ディレクトリが見つかりません: $SourceRoot"
    $result | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

# 対象拡張子
$extensions = @("*.md", "*.txt", "*.csv", "*.json", "*.yaml", "*.yml", "*.html", "*.htm")
$excludedDirs = @("__pycache__", "node_modules")
$excludedFilePatterns = @(".env", ".env.*", "secrets.*", "*.secret.*", "*.pem", "*.key", "*.pfx", "*.p12")

function ConvertTo-SourceRelativePath {
    param([System.IO.FileInfo]$File)
    return $File.FullName.Substring($resolvedSourceRoot.Length).TrimStart('\', '/') -replace '\\', '/'
}

function Test-AllowedSourceFile {
    param([System.IO.FileInfo]$File)
    $relPath = ConvertTo-SourceRelativePath -File $File
    $parts = @($relPath -split '/')
    if ($parts.Count -eq 0) { return $false }
    foreach ($dir in $parts[0..([Math]::Max($parts.Count - 2, 0))]) {
        if ([string]::IsNullOrWhiteSpace($dir)) { continue }
        if ($dir.StartsWith(".")) { return $false }
        if ($excludedDirs -contains $dir.ToLowerInvariant()) { return $false }
    }
    $name = $File.Name.ToLowerInvariant()
    foreach ($pattern in $excludedFilePatterns) {
        if ($name -like $pattern) { return $false }
    }
    return $true
}

# 全ファイル取得
$files = @()
foreach ($ext in $extensions) {
    $files += Get-ChildItem -Path $SourceRoot -Recurse -Filter $ext -File -ErrorAction SilentlyContinue
}
$files = @($files | Where-Object { Test-AllowedSourceFile -File $_ })

$result.scannedFileCount = $files.Count

if ($files.Count -eq 0) {
    $result.status  = "empty"
    $result.message = "skill-source 内に対象ファイルがありません。"
    $result | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

# キーワード分割
$keywords = @($Query -split '\s+' | Where-Object { $_.Length -gt 0 })

if ($keywords.Count -eq 0) {
    $result.status  = "empty"
    $result.message = "検索キーワードが空です。"
    $result | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

# 各ファイルをキーワード検索
$hitResults = [System.Collections.ArrayList]::new()

foreach ($file in $files) {
    try {
        $content = Get-Content -Path $file.FullName -Raw -Encoding utf8 -ErrorAction Stop
    }
    catch {
        continue
    }

    if (-not $content) { continue }

    $hitKeywords = @()
    foreach ($kw in $keywords) {
        $escaped = [regex]::Escape($kw)
        if ([regex]::IsMatch($content, $escaped, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            $hitKeywords += $kw
        }
    }

    if ($hitKeywords.Count -gt 0) {
        $relPath = ConvertTo-SourceRelativePath -File $file

        # スニペット抽出（仮: 後段の IDF 再抽出で上書きされる）
        $snippet = ""

        [void]$hitResults.Add([PSCustomObject]@{
            path        = $relPath
            hitKeywords = $hitKeywords
            hitCount    = $hitKeywords.Count
            snippet     = $snippet
        })
    }
}

# --- IDF ベーススコアリング ---
# 各キーワードの文書頻度 (DF) を計算
$keywordDF = @{}
foreach ($kw in $keywords) {
    $keywordDF[$kw] = 0
}
foreach ($hit in $hitResults) {
    foreach ($kw in $hit.hitKeywords) {
        $keywordDF[$kw] = $keywordDF[$kw] + 1
    }
}

# IDF = log(N / DF) で重み付け（N = 全ファイル数、DF = そのキーワードを含むファイル数）
$totalFiles = [Math]::Max($files.Count, 1)
foreach ($hit in $hitResults) {
    $idfScore = 0.0
    foreach ($kw in $hit.hitKeywords) {
        $df = [Math]::Max($keywordDF[$kw], 1)
        $idfScore += [Math]::Log($totalFiles / $df)
    }
    $hit | Add-Member -NotePropertyName idfScore -NotePropertyValue $idfScore -Force
}

# --- タイトル/ファイル名マッチボーナス ---
# ファイル名または先頭見出し（# で始まる行）にキーワードが含まれる場合、
# その文書はキーワードの「主題」である可能性が高いためスコアを大幅に加算する。
foreach ($hit in $hitResults) {
    $titleBonus = 0.0
    $fileName = [System.IO.Path]::GetFileNameWithoutExtension($hit.path)
    foreach ($kw in $hit.hitKeywords) {
        $escaped = [regex]::Escape($kw)
        # ファイル名にキーワードが含まれる → 強いボーナス
        if ($fileName -match $escaped) {
            $titleBonus += $hit.idfScore * 3.0
        }
    }
    if ($titleBonus -gt 0) {
        $hit.idfScore += $titleBonus
    }
}

# --- フレーズマッチボーナス（-EnablePhraseBonus フラグで制御） ---
# ファイル内容キャッシュ: Step 3 のスニペット抽出で再利用し、重複 I/O を排除する
$fileContentCache = @{}

if ($EnablePhraseBonus) {
    # 主フレーズソース: LLM 抽出済みキーワード（$Query、スペース区切り）を優先使用
    # 補助フレーズソース: OriginalQuery から3文字以上の連続部分を追加候補とする
    $primaryPhrases = @($Query -split '\s+' | Where-Object { $_.Length -ge 3 })
    $supplementalPhrases = @()
    if ($OriginalQuery -and $OriginalQuery -ne $Query) {
        $oqTokens = @($OriginalQuery -split '\s+' | Where-Object { $_.Length -ge 3 })
        # 入力長ガード: 128文字超の OriginalQuery は切り捨て
        $maxQueryLen = 128
        $maxNgramCount = 64
        $oqTruncated = $OriginalQuery
        if ($OriginalQuery.Length -gt $maxQueryLen) {
            $oqTruncated = $OriginalQuery.Substring(0, $maxQueryLen)
            Write-Warning "OriginalQuery truncated ($($OriginalQuery.Length) -> $maxQueryLen chars)"
        }
        if ($oqTokens.Count -le 1 -and $oqTruncated.Length -ge 3) {
            # 非分かち書き日本語: スライディングウィンドウで N-gram 生成（上限: $maxNgramCount 件）
            $ngramPhrases = [System.Collections.ArrayList]::new()
            for ($nLen = [Math]::Min(8, $oqTruncated.Length); $nLen -ge 3; $nLen--) {
                for ($pos = 0; $pos -le ($oqTruncated.Length - $nLen); $pos++) {
                    $ngram = $oqTruncated.Substring($pos, $nLen)
                    if ($ngram -notin $primaryPhrases) {
                        [void]$ngramPhrases.Add($ngram)
                    }
                    if ($ngramPhrases.Count -ge $maxNgramCount) { break }
                }
                if ($ngramPhrases.Count -ge $maxNgramCount) { break }
            }
            $supplementalPhrases = $ngramPhrases | Select-Object -Unique
        } elseif ($oqTokens.Count -gt 0) {
            $supplementalPhrases = $oqTokens | Where-Object { $_ -notin $primaryPhrases }
        }
    }
    $phraseKeywords = @($primaryPhrases) + @($supplementalPhrases) | Select-Object -Unique

    # ボーナス上限定数: 1文書あたり最大 idfScore * 2.0 まで
    $maxBonusRatio = 2.0

    foreach ($hit in $hitResults) {
        $filePath = Test-SafeHitPath -HitPath $hit.path
        if (-not $filePath) { continue }
        try {
            $fileContent = Get-Content -Path $filePath -Raw -Encoding utf8 -ErrorAction Stop
            $fileContentCache[$hit.path] = $fileContent
        } catch { continue }
        if (-not $fileContent) { continue }

        $baseScore = [Math]::Max([double]$hit.idfScore, 0.0)

        # 先頭見出しボーナス: 先頭5行以内の見出し行にキーワードが含まれれば加算
        $headLines = @($fileContent -split "`n" | Select-Object -First 5)
        $headingBonus = 0.0
        foreach ($kw in $hit.hitKeywords) {
            $kwEsc = [regex]::Escape($kw)
            foreach ($line in $headLines) {
                if ($line -match '^#' -and [regex]::IsMatch($line, $kwEsc, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
                    $headingBonus += $baseScore * 2.0
                    break
                }
            }
        }
        if ($headingBonus -gt 0) {
            $hit.idfScore += [Math]::Min($headingBonus, $baseScore * 2.0)
        }

        $totalBonus = 0.0
        $maxBonus = $baseScore * $maxBonusRatio

        foreach ($phrase in $phraseKeywords) {
            $escaped = [regex]::Escape($phrase)
            if ([regex]::IsMatch($fileContent, $escaped, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
                $bonus = [Math]::Min($baseScore * ($phrase.Length / 3.0), $maxBonus - $totalBonus)
                if ($bonus -le 0) { break }
                $totalBonus += $bonus
            }
        }
        $hit.idfScore += $totalBonus
    }
}

# スニペット抽出
# 総バジェット 12000 文字をマッチ数で按分
$snippetBudget = [Math]::Max(1500, [Math]::Floor(12000 / [Math]::Max($hitResults.Count, 1)))
$snippetBudget = [Math]::Min($snippetBudget, 4000)  # 1ファイル上限 4000 文字

# --- スニペット抽出（-EnableDenseSnippet フラグで制御） ---
if ($EnableDenseSnippet) {
    # Step 2 で構築済みの $fileContentCache を再利用（キー: $hit.path、値: ファイル内容文字列）
    foreach ($hit in $hitResults) {
        $filePath = Test-SafeHitPath -HitPath $hit.path
        if (-not $filePath) { continue }

        # キャッシュから取得、なければ読み込んでキャッシュに追加
        $fileContent = $null
        if ($fileContentCache.ContainsKey($hit.path)) {
            $fileContent = $fileContentCache[$hit.path]
        } else {
            try {
                $fileContent = Get-Content -Path $filePath -Raw -Encoding utf8 -ErrorAction Stop
                $fileContentCache[$hit.path] = $fileContent
            } catch { continue }
        }
        if (-not $fileContent) { continue }

        # 全キーワードの出現位置を収集
        $allPositions = [System.Collections.ArrayList]::new()
        foreach ($kw in $hit.hitKeywords) {
            $searchStart = 0
            while ($searchStart -lt $fileContent.Length) {
                $idx = $fileContent.IndexOf($kw, $searchStart, [System.StringComparison]::OrdinalIgnoreCase)
                if ($idx -lt 0) { break }
                [void]$allPositions.Add([PSCustomObject]@{ Position = $idx; Keyword = $kw; Length = $kw.Length })
                $searchStart = $idx + $kw.Length
            }
        }

        if ($allPositions.Count -eq 0) { continue }

        # 位置でソート
        $sortedPositions = @($allPositions | Sort-Object Position)

        # スライディングウィンドウで最高密度区間を選択
        $bestStart = 0
        $bestCount = 0
        $windowSize = [Math]::Min($snippetBudget, 2000)

        for ($i = 0; $i -lt $sortedPositions.Count; $i++) {
            $wStart = $sortedPositions[$i].Position
            $wEnd = $wStart + $windowSize
            $count = 0
            $uniqueKws = @{}
            for ($j = $i; $j -lt $sortedPositions.Count; $j++) {
                if ($sortedPositions[$j].Position -gt $wEnd) { break }
                $uniqueKws[$sortedPositions[$j].Keyword] = $true
                $count++
            }
            # ユニークキーワード数を優先、同数なら総出現数で比較
            $score = $uniqueKws.Count * 1000 + $count
            if ($score -gt $bestCount) {
                $bestCount = $score
                $bestStart = $wStart
            }
        }

        # 最高密度区間を中心にスニペット抽出
        $segStart = [Math]::Max(0, $bestStart - 200)
        $segEnd = [Math]::Min($fileContent.Length, $bestStart + $snippetBudget - 200)
        $hit.snippet = $fileContent.Substring($segStart, $segEnd - $segStart) -replace '[\r\n]+', "`n"
    }
} else {
    # 旧ロジック: 最レアキーワード1語ベースのスニペット抽出
    foreach ($hit in $hitResults) {
        $sortedKws = @($hit.hitKeywords | Sort-Object { $keywordDF[$_] })
        $rarestKw = $sortedKws[0]

        $filePath = Join-Path $SourceRoot $hit.path
        if (-not (Test-Path $filePath)) { continue }

        try {
            $fileContent = Get-Content -Path $filePath -Raw -Encoding utf8 -ErrorAction Stop
        }
        catch { continue }

        if (-not $fileContent) { continue }

        $positions = [System.Collections.ArrayList]::new()
        $searchStart = 0
        while ($searchStart -lt $fileContent.Length) {
            $idx = $fileContent.IndexOf($rarestKw, $searchStart, [System.StringComparison]::OrdinalIgnoreCase)
            if ($idx -lt 0) { break }
            [void]$positions.Add($idx)
            $searchStart = $idx + $rarestKw.Length
        }

        if ($positions.Count -eq 0) { continue }

        $windowPerHit = [Math]::Floor($snippetBudget / [Math]::Max($positions.Count, 1))
        $windowPerHit = [Math]::Max($windowPerHit, 400)
        $halfWindow = [Math]::Floor($windowPerHit / 2)

        $segments = [System.Collections.ArrayList]::new()
        $usedChars = 0

        foreach ($pos in $positions) {
            if ($usedChars -ge $snippetBudget) { break }

            $segStart = [Math]::Max(0, $pos - $halfWindow)
            $segEnd   = [Math]::Min($fileContent.Length, $pos + $halfWindow)
            $segText  = $fileContent.Substring($segStart, $segEnd - $segStart) -replace '[\r\n]+', "`n"

            if ($segments.Count -gt 0) {
                $lastSeg = $segments[$segments.Count - 1]
                if ($segText.Length -gt 40 -and $lastSeg.Contains($segText.Substring(0, 40))) {
                    continue
                }
            }

            [void]$segments.Add($segText)
            $usedChars += $segText.Length
        }

        $hit.snippet = $segments -join "`n[...]`n"
    }
}

# IDF スコア降順でソート（同スコアならヒット数降順）
$sorted = $hitResults | Sort-Object -Property @{Expression={$_.idfScore};Descending=$true}, @{Expression={$_.hitCount};Descending=$true}

$result.matches = @($sorted)

$result | ConvertTo-Json -Depth 5 -Compress
