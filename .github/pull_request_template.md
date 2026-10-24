## 変更内容

<!-- 何を、なぜ変更したかを記載する -->

## 影響範囲

<!-- 該当するものにチェックする -->

- [ ] `index.bat`（Embedding事前構築）
- [ ] `search.bat`（コンソール検索）
- [ ] `web.bat`（loopback Web検索）
- [ ] package生成（`build-offline-package.ps1` / `download-offline-package.ps1`）
- [ ] install（`install-offline.ps1`）
- [ ] test / CI のみ
- [ ] 文書のみ

## Test実行結果

<!-- 実行したcommandと結果を貼り付ける -->

```text
python -m pytest -q
```

```text
pwsh -NoProfile -File scripts/bootstrap.ps1
```

## 配布分類の確認

新しいfileを追加した場合、`_internal/distribution-policy.json`へ配布分類のentryを追加したか確認する。

- [ ] 新規fileなし、または全て`distribution-policy.json`へ登録済み
