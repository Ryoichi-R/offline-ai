# アンインストール

1. `web.bat`、`search.bat`、`index.bat`とOllamaを終了します。
2. `skill-source`と必要な検索資料を別の場所へバックアップします。
3. install stateでPowerShell 7が`Installed`の場合だけ、他用途で使っていないことを確認してWindowsのアプリ一覧から削除します。
4. `offline-ai`アプリフォルダーを削除します。`_internal\embed_cache.json`、`_internal\embed_cache.checkpoints`、`_internal\embed_index_status.json`、`_internal\embed_cache.lock`、`_internal\embed_cache.json.tmp`、`.search-index`も不要になります。必要ならcache/checkpointを先に別名バックアップしてください。
5. 専用に導入したOllama、Python、モデルを削除する場合は、他用途で使っていないことを確認して個別に削除します。

`skill-source`、共有モデル保存先（例: `E:\ollama-models`）、および明示的に残したcache/checkpointは自動削除しません。削除したアプリフォルダー、Embedding cache、検索indexは、バックアップがなければ復旧できません。
