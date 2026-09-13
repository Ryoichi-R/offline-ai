# tests/integration

このディレクトリは、package生成からinstallerまでの自動fixture E2Eと、Windows 11 x64 clean環境で行う手動受入の契約を管理します。

## 現状

`TargetSpecPackageE2E.Tests.ps1`はfixture registryとfake Ollamaを使い、外部downloadや実model起動を行わずに次を検証します。

- 対象スペック入力からschemaVersion 2 package生成、実installer script、`.model`、`install-state.json`までの伝播
- 選択したchat modelと必須Embeddingだけがpackageへ収録されること
- 非default model、`KeepExisting`、`OverwriteDefault`、非installable拒否
- RAM不足modelの非対話`-AllowUnsupportedChatModel`と`explicit-switch` receipt
- 拒否時に既存install-stateのhashが変化しないこと
- VRAM基準の推薦、保存先別の不足量、日本語表示、GPU配置観測のreceipt構造

## 未完了の手動受入

fixture E2Eは実model、実GPU、実Ollama runtimeの動作保証ではありません。公開候補に対する正式な手動受入では、`release/preparation-policy.json`のmanual acceptanceに従い、次を別途実施します。

- 固定Ollama版と採用digestを使った実package生成
- Windows 11 x64 clean環境、標準user権限、apostrophe path、別drive `OLLAMA_MODELS`
- install、search、web、restart、uninstallと失敗系
- 非default modelおよびCPU-only、partial offload、full GPUの実測

自動fixture E2EはPesterから実行できます。clean Windows受入の実行結果は、release candidateに紐付くreceiptとして`result/offline-ai/`へ保存します。
