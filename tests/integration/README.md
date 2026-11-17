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

## ソース配布と搬送パッケージの手動受入

ソース ZIP の受入は `release/preparation-policy.json` に従い、Windows 11 x64 でクリーン展開と合成資料を使う索引・CLI・Web のスモークを確認します。実モデル、実 GPU、Ollama 実機を含む搬送パッケージを検証する場合は、次を別途実施します。

- 固定Ollama版と採用digestを使った実package生成
- Windows 11 x64 clean環境、標準user権限、apostrophe path、別drive `OLLAMA_MODELS`
- install、search、web、restart、uninstallと失敗系
- 非default modelおよびCPU-only、partial offload、full GPUの実測

自動fixture E2EはPesterから実行できます。clean Windows受入の実行結果は、release candidateに紐付くreceiptとして`result/offline-ai/`へ保存します。
