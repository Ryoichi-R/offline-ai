# offline-ai 内部アーキテクチャ

## 1. 製品境界

offline-aiは、外部で作成・検証済みのテキスト資料を完全オフライン環境で検索し、根拠付き回答を生成する。文書変換、OCR、PDF upload、外部サービス連携は責務に含めない。

正式entrypointは次の3つである。

- `index.bat`: Embedding事前構築・status・cancel・resume
- `search.bat`: コンソール検索
- `web.bat`: loopback Web検索

## 2. 主な構造

```text
offline-ai/
├── index.bat
├── search.bat
├── web.bat
├── install-offline.bat
├── download-package.bat
├── README.md / CONTRIBUTING.md / SECURITY.md / CODE_OF_CONDUCT.md
├── SUPPORT.md / UNINSTALL.md / CHANGELOG.md / THIRD-PARTY-NOTICES.md
├── VERSION / release-metadata.json / pyproject.toml / .gitignore
├── skill-source/            # 利用者資料。非追跡・非配布
├── legal/
├── infographic/
├── .github/                 # CI定義、issue/PRテンプレート
├── scripts/
│   └── bootstrap.ps1        # standalone開発環境準備
├── tests/                   # test一式（unit配下にPython/Pester、conftest.py/support/がroot解決を共通化）
│   └── eval/                # Phase 2 検索品質評価ハーネス（固定corpus・評価仕様・run_eval.py）
└── _internal/
    ├── search.py
    ├── index_cli.py / index_service.py
    ├── web_server.py
    ├── web_services.py
    ├── source_view.py       # 根拠IDから検索用資料の行窓を安全に読出し
    ├── document_schema.py
    ├── rerank.py
    ├── prompt_templates.py
    ├── ARCHITECTURE.md（本file）
    ├── OfflineAi.Common.psm1
    ├── install-offline.ps1
    ├── download-offline-package.ps1
    ├── build-offline-package.ps1
    ├── distribution-policy.json # 配布分類契約（詳細は10.）
    ├── schemas/
    └── scripts/
        ├── collect_skill_source.ps1
        └── Test-DistributionPolicy.ps1
```

## 3. 検索フロー

Embedding indexは検索要求から分離される。`index.bat build`またはWebのindex APIだけがwriterを開始し、検索はcache状態を読むだけである。

```text
search.bat / GET /api/search
  → search.py / web_services.run_search
  → mode=answer: chat model検出 → index state read
  → mode=search: 決定的query → index state read（chat model/検索計画なし）
  → readyならskill-source scan + keyword/Embedding retrieval、未準備ならkeyword retrieval only
  → RRF merge
  → optional loopback reranker
  → mode=answer: Ollama回答生成 → answer + evidence + warnings
  → mode=search: evidence + done（回答生成なし）
```

検索対象は`.md`, `.txt`, `.csv`, `.json`, `.yaml`, `.yml`, `.html`, `.htm`である。対応外の原本は外部で検索用派生物へ変換し、利用者が原本と照合する。

ただし`*.metadata.json`は検索対象から除外する。sidecarは原本の複製であり独立した資料ではないため、索引すると同一内容が二重に根拠提示され根拠precisionを下げる。除外は`EXCLUDED_SOURCE_FILE_PATTERNS`が担う。

### 3.1 根拠ステータス（evidence_status）

`_calculate_confidence`はquery語と根拠本文の一致率（coverage）、snippet充足率、上位RRFスコア、文書数からconfidenceを算出し、`sufficient` / `partial` / `insufficient`を返す。

- coverageは`coverage_terms()`が返す語で測る。日本語のように語間へ区切りを置かない言語では`_split_terms`だけでは query 全体が1語となり coverage が常に0になるため、CJKを含む語は2文字n-gramへ展開する。`must_find_terms`は検索計画が抽出した精密語であり展開せず逐語で評価する。
- Agentic-liteの検索計画は短いJSONを返す補助処理であり、Ollamaの`think`を明示的に`false`へ固定する。thinking対応モデルの既定推論で90秒timeoutを消費せず、回答生成のreasoning設定とは分離する。
- `sufficient`はEmbedding由来の根拠を伴う場合に限る。語の重なりだけでは、質問の限定条件（例:「海外」出張）を満たさない資料を十分な根拠として区別できない。keyword一致のみでsufficientを宣言すると根拠不足の警告が外れ、該当情報なしを維持できなくなる。
- 該当情報なしは2層で担保する。corpusに一切記載がない場合はretrieval層が`insufficient`を返し、関連文書はあるが質問の条件を満たさない近接語の場合は`sufficient`を宣言しないことに留め、最終判定は`prompt_templates.py`の根拠契約が行う。
- retrieval候補は、質問との4文字以上の具体的な連続一致、または0.55以上のEmbedding類似度を根拠支持として要求する。質問条件が同一文中で明示的に否定される場合は候補制限前に矛盾を検出し、`sufficient`を禁止して回答層へ矛盾信号を渡す。
- 回答promptは地域・対象・期間・役職・区分などの限定条件を個別照合し、条件が欠ける・対象が異なる・対象外である場合に、関連語や似た資料の数値を流用せず「該当情報なし」とする。評価ハーネスの`--measure-answer-layer`は回答本文を保存せず、この契約のphrase・禁止fact・transport errorだけをprobeする。

## 4. metadata sidecar

`document_schema.py`は`<file>.md.metadata.json`のschemaとreader/writer helperを提供する。

- `search.py`は`load_metadata_sidecar()`と`metadata_for_line_range()`を利用する。
- sidecarがなくても検索できる。
- sidecar自体は検索対象・根拠対象にしない（3. を参照）。
- sidecarがある場合だけparser、page、heading、layout type、source title等を根拠へ付与する。
- writer helperは外部converterやfixtureが同じschemaを生成できるよう維持する。

## 5. Web設計

`web_server.py`はloopbackへbindし、次のrouteだけを提供する。

- `GET /`
- `GET/POST /bootstrap`
- `GET /api/health`
- `GET /api/search`（SSE）
- `GET /api/evidence/view?evidence_id=...`（検索時に登録した根拠行窓）
- `GET /api/index/status`
- `POST /api/index/start` / `resume` / `cancel`
- `GET /api/index/events?job_id=...`（SSE）

未知routeは共通404へ委ねる。Web層はbootstrap token、HttpOnly cookie、Origin/Host検査、CSP、接続数制限、`JobTable`、`CancellationToken`、slow-client対策を維持する。

Health contractにはindex statusと検索単位timeout契約（`searchTimeoutDefault`/`Min`/`Max`、300〜600秒）を含める。index SSEの切断はjobをcancelせず、statusとbounded event bufferから再接続する。

`/api/search`は任意の`timeout_seconds`クエリパラメータ（300〜600の整数、省略時はサーバー既定）を受理し、`CancellationToken`が確定した総秒数を保持する。SSE購読開始時に`{"type":"budget","timeoutSeconds":N,"remainingSeconds":R}`を送出し、Web UIはこれを起点にサーバーdeadline基準の残り時間を表示する。表示は案内であり、timeout判定は常にサーバー側の`CancellationToken`が行う。

SSEのeventは`status` / `budget` / `result_meta` / `evidence` / `thinking` / `chunk` / `answer_empty` / `done` / `error`である。未知typeはUIが無視する前方互換設計とし、購読ループは`done`と`error`だけを終端として扱う。`result_meta`はrequest単位の開始時刻、mode、実使用model、reasoningを返し、`done`は完了時刻を返す。`mode=search`は`result_meta`、`evidence`、`done`だけを返し、`thinking`、`chunk`、`answer_empty`は発行しない。検索SSEはterminal eventをflushした後にHTTP応答を終了し、Web UIは検索識別子で旧readerの遅い終了を隔離する。`/api/index/events`は別契約であり、terminal後もjob statusとbounded event bufferによる再接続を維持する。

根拠イベントのviewable itemだけが不透明な`evidenceId`を持つ。IDは認証セッションに束縛されたメモリ内registryへ30分・最大512件保持し、閲覧時にsource相対path、拡張子・sidecar除外、reparse point、開いたハンドルの実体、検索時SHA-256を再確認する。閲覧結果は同じUTF-8/errors=replace/splitlines規則で前後各10行、最大200行・64KiBとして生成し、本文をregistryへ保持しない。変更・削除・境界違反時は本文を返さず、内部絶対pathやOS例外を応答しない。

Web UIの「Markdownで保存」は正常完了した検索単位をブラウザBlobとして保存するだけで、サーバー履歴や自動保存は行わない。保存内容は質問、回答または「回答生成なし」、その検索の根拠抜粋、実行条件に限定し、thinking、絶対path、閲覧ID、認証情報を含めない。質問・回答・抜粋は内容中の連続backtickより長いtext fenceへ入れてpreviewで能動化しない。

生成の終端理由は`iter_stream_events`が`("done", done_reason)`として1回だけ返し、`run_search`が本文0文字のときに`answer_empty`（`reason`は`length` / `stop`等）を`done`の直前へ1回送る。prompt・thinking・本文は同じcontextウィンドウを共有するため、`done_reason: length`は「promptとthinkingでcontextを使い切り本文の余地が残らなかった」ことを意味する。timeoutもstallも発生しないままこの状態になり得るため、timeout契約とは独立の失敗経路として扱う。

qwen3.5:9bではthinkingがcontextの残りを使い切るまで伸びた（2026-09-08実測）。thinkレベルはこの長さを制御せず、promptを削っても削った分だけthinkingが伸びるため、空回答を避ける手段はcontextウィンドウの確保だけである。既定`OLLAMA_NUM_CTX`が32768、既定reasoningが`low`なのはこの実測に基づく。

[Ollamaの公式仕様](https://docs.ollama.com/capabilities/thinking)ではgpt-ossの内部推論は無効化できず、`low`/`medium`/`high`を指定する。回答生成のreasoning未指定・空文字・`off`ではgpt-oss familyだけ最小の`low`を明示し、他モデルの`think: false`は維持する。非表示モードの45秒受入を超えた実測を受けた互換性修正であり、明示low/medium/highと補助検索計画のpayloadは変更しない。reasoning `off`の契約は、`web_services`がその文字数を診断値として数える一方、thinking SSEイベントとWeb UI表示へは公開しないことで維持する。reasoning `low`/`medium`/`high`では従来どおりthinkingイベントを表示し、モデル内部の実thinkingを無かったことにはしない。

prompt予算は`OFFLINE_AI_GENERATION_RESERVE_TOKENS`が1以上のとき`OLLAMA_NUM_CTX`から逆算する。`num_ctx - 生成予備 - prompt shellのoverhead`で根拠に使えるtoken予算を求め、文字上限は`OFFLINE_AI_PROMPT_EVIDENCE_CHAR_LIMIT`との小さい方を採る。残量が0以下なら根拠を強制挿入せず`prompt_budget`エラーでfail closedする。既定`0`は逆算を行わない従来挙動であり、既定値はthinkレベルとcontextの組を実測してから確定する。

生成フェーズのsocket timeoutは`code: "stall"`へ写像する。接続不能とretrievalフェーズのtimeoutは対象外とし、原因判別を保つ。

## 6. package・installer

標準packageはPython、Ollama、選択モデルを中心に構成する。PowerShell 7は任意である。

Local rerankerは`activation: transport-only`の任意componentとして搬送できるが、自動起動・自動有効化しない。optional component共通guardはpath境界、reparse point、hash、秘密ファイル名、manifest外fileを検査する。

旧install treeからpackageを作る場合の防御として、package collectorは`.pdftotext`をskip fileに残す。active runtimeやinstallerはこの設定ファイルを生成・参照しない。

## 7. キャッシュ・モデル

- Chat model既定値: `gpt-oss:20b`
- Embedding model既定値: `bge-m3`
- User-built transportでは`download-manifest.json` schemaVersion 2を推薦catalogの正典とし、対象オフラインPCの手入力スペックから`OfflineAi.ModelSelection.psm1`が決定論的に候補を順位付けする。オンラインPCのhardwareは検出しない。
- GPU配置classは専用VRAMの有無・数値から判定する。GPU vendor/nameは数値判定を変更せず、OllamaのOS・driver・対応機種に関するcompatibility noticeへだけ使う。
- app事前容量はcatalogのreview済みruntime inventoryから`max(total + largest file + 512MiB, 1GiB)`で算出する。package generatorとdistribution policyのV-6 gateは、catalogの`appPayloadEstimate`を現行の`runtime`/`both` inventory（total、largest file、file count）と完全一致で照合し、drift時はpackage生成前にfail-closedとする。installerはpackage内appを再計測し、実path単位のvolume preflightを最終判定とする。
- compact/balanced/qualityから選択したchat 1件とrequired embedding 1件だけを取得し、package manifest schemaVersion 2の`selection`と`targetRequirementsInput`へ非機密receiptを記録する。
- 各採用modelはfull manifest、config、license layerのexpected digestをcatalogへ固定する。registry応答が一致しなければblob取得前にfail-closedで停止する。固定Ollamaでの実行、日本語RAG、CPU/partial/full GPU性能は別のmanual adoption receiptで管理し、未完了時はrelease gateを閉じない。
- installerはsmoke test後にOllama `/api/ps`をbest-effortで読み、full/partial/CPU、GPU割合、contextを表示してinstall-stateへ記録する。API取得不能やmodel unloadは導入失敗にしない。
- `download-only-no-models`と`public-source`は`installable: false`であり、installerはinstall-state作成とcopyより前に拒否する。public-source manifestにはselection、GPU/VRAM/RAM/容量入力を投影しない。
- installerはschemaVersion 2でpackage内のselection moduleとcatalog snapshotだけを使用し、selection、catalog、models inventoryの意味的一致を検査する。schemaVersion 1 local-build/download-onlyは`qwen3.5:9b`と`bge-m3`が各1件ある場合だけ互換fallbackする。
- `legacyMigrationModels`はPythonの`LEGACY_MODELS`と集合一致させ、selectable chatとの重複をPowerShell catalog validatorで拒否する。
- Embedding cacheはsource hash、model、chunk設定、versionを検証する。
- metadataだけの変更ではEmbedding全面再計算を要求しない。
- 大規模な初回Embedding構築は、CLI/Web共通の件数進捗（総数・処理済み・新規・再利用・失敗）を表示し、`_internal/embed_cache.checkpoints/` へ一定件数ごとの再開用batchを保存する。checkpointはモデル参照（省略tagは`:latest`へ正規化）、cache version、chunk設定、generationへ束縛する。
- 最終`embed_cache.json`は正常完了時だけatomic replaceし、再読込検証に成功した後でcheckpointを整理する。中断・cancel・timeout・保存失敗では既存の正常cacheを保持し、確定済みcheckpointを削除しない。
- `embed_cache.json`の読込は全文を1つの`str`にしない。UTF-8をincrementalにdecodeしながら`entries`を1件ずつ標準`json`のC scannerでparseし、消費済みbufferを捨てて進む。BMP外の文字を含む大規模cacheで`str`がUCS-4化してpeak working setが数GBへ膨らむのを避けるためであり、cache形式と保存経路は変えない。構造が想定外・破損の場合は全文parse、さらに失敗すれば空cacheへfallbackする。
- Embedding writerはprocess内thread lockとOS file lockで1本に制限する。lock競合時は無期限待機せず、既存cacheの読取または明示的keyword fallbackへ移行する。
- Embedding生成はOllama `/api/embed`の配列入力を使う。入力と返却の件数・順序・次元・有限値を検証し、413/timeout/resource failureのsplitは1回だけとする。旧`/api/embeddings`への暗黙fallbackはしない。
- cache/checkpoint/statusはmodel、chunk契約、source chunk identityから作るgenerationへ束縛する。build失敗、cancel、timeout、generation不一致では既存ready cacheをpromoteしない。
- timeoutは通常検索全体300〜600秒（既定300秒）、検索計画chat既定90秒、retrieval child既定120秒、index batch既定120秒、index全体既定6時間で分離する。各値は設定読込時に有限・範囲内で検証する。通常検索全体の120秒は有効値ではない。
- `embed_index_status.json`はプロセス間lock下でcurrent stateを検証してから、writer別unique temporary fileをatomic replaceする。`job_id`、generation、state遷移を保存前に検査し、cancelling/cancelled/readyを古いworkerのbuilding更新で巻き戻さない。旧固定temporary fileは、正常な本体cacheが確認できる場合だけ復旧対象とする。
- 補助検索計画と回答生成は`num_ctx`、`num_batch`、`num_gpu`、`keep_alive`のchat option builderを共有する。context方針の実機効果（load duration、latency、VRAM、品質）は別のR-3測定で判定し、自動テストだけではPASSにしない。
- source/test/distribution-policyの自動検証完了と、package・導入先での実機受入は別の状態として記録する。実機受入未実施を自動的にPASSへ昇格しない。
- model copyは衝突を検出し、atomic placementと明示的backupを使う。

## 8. 終了コード

| code | 意味         | 主な発生元                        |
| ---: | ------------ | --------------------------------- |
|    0 | 成功         | launcher、search、index status/build |
|    1 | 実行時エラー | search、index失敗、資料収集、Ollama、timeout |
|    2 | 入力・設定不正 | search、index設定不正 |
|  130 | 利用者cancel | `index.bat` Ctrl+C |

削除した機能に対応する番号を詰め直さず、残存契約だけを記載する。

## 9. 復旧

このworkspaceではGitを復旧手段として使用しない。

1. 変更前にworkspace相対path、bytes、SHA-256を持つsnapshot manifestを作る。
2. 変更pathをreceiptと照合する。
3. 復旧先に競合がない場合だけ同じ相対pathへrestoreする。
4. 復旧後のSHA-256をmanifestと比較する。
5. Python、Pester、fast suiteを再実行する。

snapshot不完全、hash不一致、復旧先競合のいずれかがある場合は上書きせず停止する。

## 10. 成果物境界と配布分類

offline-aiを独立public repositoryとして切り出す場合の成果物境界は、次の3つを正規とする。

| 成果物                                                             | 含めるもの                                                                 | 除くもの                                                                                |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Public source artifact                                             | source、tests、CI定義、community文書、公開用metadata schema                | installer、model blob、optional payload、利用者資料、cache、install state、ローカル設定 |
| runtime app payload（User-built transportの`app/offline-ai/`部分） | 実行に必要なsource、利用者向け文書                                         | tests、CI定義、開発者向け文書、toolchain設定、infographic、開発用metadata               |
| User-built transport                                               | runtime app payload、installer、model、manifest、legal inventory、checksum | 利用者資料、秘密情報、PC固有path                                                        |

上記3成果物の公開条件は`release/preparation-policy.json`、`release-metadata.json`および`download-offline-package.ps1`の`Assert-PublicReleaseReady`で管理する。公開承認と検証receiptが揃うまで公開しない。本節はfile単位の分類契約を定義する。

### 配布分類（distribution classification）

`offline-ai/`配下の**全file**は、`_internal/distribution-policy.json`（schema: `_internal/schemas/distribution-policy.schema.json`）へ次のいずれかの分類を持つ。**未分類のfileが1件でも存在すると、candidate検証・package生成の双方がfail-closedで失敗する。** これはbugではなく意図した安全装置である。

| 分類値          | Public source | runtime payload            |
| --------------- | ------------- | -------------------------- |
| `both`          | 含む          | 含む                       |
| `public-source` | 含む          | 除く                       |
| `runtime`       | 除く          | 含む                       |
| `excluded`      | 除く          | 除く                       |
| `release-only`  | 含む          | `-PublicRelease`時のみ含む |

検証は`_internal/scripts/Test-DistributionPolicy.ps1`で行う。V-1（分類網羅）、V-2（audience別allowlist選択）、V-4（path traversal・絶対path・case collision・ADS・予約device名拒否）、V-5（reparse point拒否）を実施し、`candidate_digest`（決定的。policy digest + sorted file listのみで構成）と`generated_at`を含むrun receiptを出力する。**秘密情報の本文scan（V-3相当）は行わない。** `sensitive_scan_performed=false`をreceiptへ記録し、正式な秘密scanはformal audit側のconsent後gateへ委譲する。

新しいfileを追加する場合は、`distribution-policy.json`へentry（`pattern`、`distribution`、`reason`、`boundary_row`）を追加すること。

## 更新ポリシー

- `document_schema.py`、Markdown renderer、JobTable、CancellationToken、optional component共通guardを用途確認なしに削除しない。
- PowerShell module top-levelはfunction/class定義、`Set-StrictMode`、`Export-ModuleMember`、pure constant assignmentだけに限定する。
- package、installer、docs、testは同じ製品契約へ同期する。
- 変更後はdirect file comparison、hash、test、lint、buildを証跡として残す。
- 新しいfileを追加する場合は、`_internal/distribution-policy.json`へ配布分類のentryを同じ変更内で追加する。追加を怠ると、candidate検証・package生成がfail-closedで失敗する。
