# Contributing to offline-ai

このリポジトリは、ローカル資料を検索し、Ollamaモデルで根拠付き回答を生成する完全オフライン向けツールの開発版です。

## 対応環境

- Windows 11 x64
- Python 3.11〜3.12（CI matrixとこの範囲で検証する。搬送物に同梱するPythonは3.11.9固定）
- PowerShell 7.4以降
- Pester 5.7.1（固定版。`scripts/bootstrap.ps1`が導入する）

## 開発環境のセットアップ

親workspaceの`scripts/bootstrap.ps1`には依存しない。offline-ai単体のcheckoutでも、次のcommandだけで開発環境を再現できる。

```powershell
pwsh -NoProfile -File scripts/bootstrap.ps1
```

`-WhatIf`を付けると、実際の変更を行わずに検証内容を確認できる。

## Test の実行

```powershell
# Python テスト（offline-ai/ をrootとして実行する）
python -m pytest

# Pester テスト
pwsh -NoProfile -Command '$c = New-PesterConfiguration; $c.Run.Path = @("tests/unit", "tests/integration"); $c.Run.Exit = $true; Invoke-Pester -Configuration $c'
```

上記はこのrepositoryのrootから実行する。評価receiptなどの一時成果物は`.test-results/`へ保存し、Gitの追跡対象に含めない。

## 依存関係

- **runtime依存: 0件。** `_internal/`配下のPython実装は標準ライブラリのみで動作する。新しい外部パッケージをruntime経路へ追加しない。
- **development/test依存**: `pyproject.toml`の`[dependency-groups]`を参照する。対応range（互換性の宣言）と、`scripts/bootstrap.ps1`が導入する再現用固定versionを分けて記載している。

## コーディングスタイル

このrepositoryでは以下を必須とする。

- インデント: 4スペース（PowerShell/Python）
- **PowerShellモジュール（`*.psm1`）のトップレベル**では、function定義・class定義・`Set-StrictMode`・`Export-ModuleMember`・純粋な定数代入のみを書く。`Add-Type`、`[Environment]::GetFolderPath(...)`などWindows専用API呼び出し、I/Oを伴う関数呼び出しをモジュールスコープで実行しない。必要ならWindowsガード付きの`Initialize-*`関数へ切り出す。
- WinForms/System.Drawing/Win32 P/Invoke等Windows専用のPester testには`-Tag 'WindowsOnly'`を付与する。ただしmodule読み込み自体は全environmentで成功させる。

## `.github/workflows/` の静的検証（actionlint）

workflow定義の静的検証はactionlintを正規手段とする。YAML parseやkey存在チェックを代替PASSとして扱わない。

| 項目           | 方針                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| version        | 実施時点の[公式releaseページ](https://github.com/rhysd/actionlint/releases)で最新安定版を確認し、明示的にpinする。自動最新追従にしない |
| architecture   | 実行機に合わせWindows x64 / ARM64を選択し、選択結果をreceiptへ記録する                                                                 |
| 取得元         | 上流公式release（`rhysd/actionlint`）のみ。ミラーやサードパーティ配布を使わない                                                        |
| 完全性検証     | SHA-256照合を必須とする。可能なら`gh attestation verify -R rhysd/actionlint`を併用する                                                 |
| license        | 取得時にlicenseを確認する。配布物へ同梱しないため`THIRD-PARTY-NOTICES.md`には原則不掲載だが、判定結果を記録する                        |
| cache位置      | workspace内の一時領域（例: スクラッチ領域）。`offline-ai/`配下へは置かない                                                             |
| 実行オプション | `-shellcheck= -pyflakes=`（外部tool不在による誤失敗を避ける）                                                                          |
| 更新方針       | pin版を明示更新する。更新時はSHA-256とattestationを再確認する                                                                          |

実行例（`<version>`と`<arch>`は取得時に確定した値へ置き換える）:

```powershell
actionlint -shellcheck= -pyflakes= .github/workflows/*.yml
```

静的検証PASSと、切り出し後の実CI PASSは別gateである。actionlintの実行version・SHA-256・architectureは、実施時にrun receiptへ記録する。

## `_internal/` について

`_internal/`は利用者が直接触らない実装本体であり、OSSにおける`src/`に相当する。正式な検索・索引entrypointは`index.bat`、`search.bat`、`web.bat`の3つであり、それ以外の実装詳細はこのディレクトリへ集約している。命名変更は行わない方針である（参照範囲が広く、install済みtreeのpath互換性に影響するため）。

## 成果物境界と配布分類

このリポジトリのfileは、次の3つの成果物境界のいずれかに分類される。詳細は[`_internal/ARCHITECTURE.md`](_internal/ARCHITECTURE.md)の「成果物境界と配布分類」節を参照する。

| 成果物                 | 含まれるもの                                   |
| ---------------------- | ---------------------------------------------- |
| Public source artifact | source、tests、CI定義、community文書           |
| runtime app payload    | 実行に必要なsourceと利用者向け文書             |
| User-built transport   | runtime app payload + installer/model/manifest |

**新しいfileを追加する場合は、`_internal/distribution-policy.json`へ配布分類のentryを追加すること。** 未分類のfileが存在すると、公開候補生成とpackage生成の両方がfail-closedで失敗する（意図的な安全装置であり、bugではない）。

## Pull Request

- 変更したtestの実行結果を記載する。
- 影響範囲（`search.bat`/`web.bat`のいずれか、または両方、あるいはpackage生成のみ等）を明記する。

## 公開前の未完了条件

公開先URL・非公開報告窓口・第三者再配布承認は未確定。`release/preparation-policy.json`のformatter、lint、coverage 90%および要求receiptは公開gateであり、pytest/Pester成功だけでは満たさない。standaloneでこれら全てを再現する検証環境とreceiptは未整備のため、公開準備完了とは扱わない。

`_internal/download-manifest.json`のinstaller固定版・hashReviewの日付は、現在のhash照合の記録を保持する。実利用Ollama版と同梱installer版は別管理であり、版を更新する際はartifactのhash・署名・互換性を再検証する。日付だけを最新化しない。
