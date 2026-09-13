# Security Policy

## 対象バージョン

`VERSION`ファイルに記載された最新版のみをsecurity supportの対象とする。過去版への遡及的な修正は行わない。

## 脆弱性の報告方法

**脆弱性は公開issueへ書き込まないこと。** 公開issueは第三者に悪用の機会を与える可能性がある。

脆弱性を発見した場合は、このリポジトリの **Private Vulnerability Reporting（PVR）** を使って非公開で報告する。

- 報告フォーム: https://github.com/Ryoichi-R/offline-ai/security/advisories/new
- 手順: リポジトリの `Security` タブ → `Report a vulnerability` を選択する。
- メールアドレス等の代替連絡先は設けない。GitHubアカウントを持たない場合は、リポジトリのSUPPORT窓口経由で報告方法を問い合わせること。

報告には次を含めることを推奨する。

- 影響を受けるversion（`VERSION`または`manifest.json`の`productVersion`）
- 再現手順
- 想定される影響範囲

**API key、credential、利用者文書、検索語、環境変数全体は報告に添付しないこと。** これらの情報が問題の再現に不要な場合は含めない。

## 対応方針

- 初回応答の目安: 報告受領から5営業日以内。
- 修正方針・開示時期は、影響範囲の確認後に報告者へ個別に連絡する。
- 修正版がリリースされるまで、詳細の公開を控えるよう協力を依頼する場合がある。

## 対象外

- 完全オフライン環境を前提とした設計上の制約（例: ローカル管理者権限を持つ利用者による意図的な改変）は、本ポリシーの対象外とする。
