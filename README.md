# offline-ai

## 概要

Windows 11 x64上で、ローカルのテキスト資料を検索し、Ollamaモデルで根拠付き回答を生成する完全オフライン向けツールです。

## 配布状態

現在は未公開の開発版です。公開・第三者配布には、`release-metadata.json`の承認と公開前監査が必要です。

## 動作要件

- Windows 11 x64
- RAM 32GB以上（既定の`gpt-oss:20b` balanced候補の保守要件）
- 空き容量20GB以上
- 検索対象: Markdown、テキスト、CSV、JSON、YAML、HTML

PDFは直接検索しません。外部でMarkdown等へ変換し、原本と照合した検索用派生物を`skill-source`へ配置してください。

`_internal/`は利用者が直接触らない実装本体で、OSSの`src/`に相当します。正式なentrypointは`index.bat`（Embedding事前構築）、`search.bat`、`web.bat`です。検索要求はEmbedding構築を自動開始しません。

## 使用方法

Web UIでは「回答を作る」と「資料を探す」を選べます。「資料を探す」は検索計画・回答生成を行わず、検索結果の根拠だけを表示します。根拠に行情報と検索時SHA-256がある場合は「該当箇所を表示」で検索用テキストの該当行と前後を確認できます。正常完了後は、質問・回答または検索結果・根拠抜粋・実行条件を「Markdownで保存」できます。保存はブラウザの明示操作で行われ、サーバーへ履歴は保存しません。

## 設定

「推論: 非表示」は思考欄を表示しない設定です。gpt-ossは内部推論を無効化できないため、この設定では最小の`low`で処理します。無効化に対応した他のモデルでは`think: false`を使用します。

## インストール

1. オンラインPCで`download-package.bat`を実行して搬送用packageを作成します。
   対象オフラインPCで確認した専用VRAM、RAM、app/models/temp保存先の空き容量を入力し、表示された軽量・推奨・品質優先の候補からchatモデルを選びます。オンラインPC自身のGPUは判定に使いません。GPUベンダー未指定でも専用VRAMが入力済みなら容量分類を行い、ベンダー情報は互換性注意にだけ使います。
2. packageをオフラインPCへ搬送します。
3. package内の`install-offline.bat`を実行します。
4. `skill-source`へ資料を配置します。
5. `index.bat build`でEmbeddingインデックスを事前構築し、完了後に`search.bat`または`web.bat`を使用します。通常のbuildはファイル単位の差分更新で、未変更ファイルのEmbeddingを再利用し、追加・変更ファイルだけを再計算します。全件再構築が必要な場合だけ`index.bat build --full`を使います。未構築・更新中でも検索はキーワード検索のみで継続します。

詳しくは[セットアップガイド](offline-ai-setup-guide.md)を参照してください。

## プライバシー

検索・回答生成はローカルで動作します。外部サービスで資料を事前変換する場合は、機密情報、個人情報、組織規程を確認してください。

## ライセンス

自作コードは [MIT License](LICENSE)（Copyright (c) 2026 Ryoichi-Rice and contributors）で提供します。第三者ソフトウェア・モデルにはそれぞれのライセンスが適用されます。

- [アンインストール](UNINSTALL.md)
- [第三者ライセンス](THIRD-PARTY-NOTICES.md)
- [サポート](SUPPORT.md)

## アンインストール

削除対象と保持するデータは[アンインストール手順](UNINSTALL.md)を確認してください。

## 既知の制限

PDFの直接検索には対応していません。モデルの速度やGPU配置は機器構成に依存します。公開前の検証状況は[開発手順](CONTRIBUTING.md)の公開前未完了条件を参照してください。
