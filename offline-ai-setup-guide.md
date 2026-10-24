# オフラインAIツール セットアップガイド

## 機能

| 機能              | 起動方法     | 内容                                               |
| ----------------- | ------------ | -------------------------------------------------- |
| Embedding事前構築 | `index.bat`  | バッチ生成、進捗、キャンセル、checkpointからの再開 |
| 資料検索・回答    | `search.bat` | `skill-source`の資料を検索し、根拠付き回答を生成   |
| Web UI            | `web.bat`    | ブラウザから同じ検索・回答機能を利用               |

PDFは直接検索しません。事前に外部ツールでMarkdown等の対応形式へ変換してください。

## 必要環境

- Windows 11 x64
- RAM 32GB以上（`gpt-oss:20b` balanced候補の保守要件）
- 搬送・導入先に20GB以上の空き容量（モデル保存先はcatalogの見積を確認）
- オンラインPCとオフラインPC
- USBメモリ等の搬送媒体

標準構成はOllama、Python、既定の`gpt-oss:20b` chatモデル、BGE-M3を使用します。PowerShell 7とlocal reranker payloadは任意です。schemaVersion 1の旧形式package互換fallbackだけは`qwen3.5:9b`を使用します。

## 既存qwen環境の扱い

今回の既定chatモデル移行で、受入・動作保証の評価対象とするモデルは`gpt-oss:20b`です。`qwen3.5:9b`の継続動作は今回の保証対象外です。既存設定を残して起動できることと、検索の完走・性能を保証することは別です。

既存の`.model = qwen3.5:9b`は自動変更しません。gpt-ossを使用する場合は、利用者が明示的にモデルを選択・変更してください。qwenのcatalog・legacyへの再追加や、既存モデルの削除も行いません。schemaVersion 1のqwen互換fallbackは保持しますが、qwenの継続動作保証を意味しません。

gpt-oss側はGUIの必須受入を確認済みです。2026-09-13に、現用PCでの実利用によるVRAM安全余裕を受け入れ、自然に再現しないCPU offload時の性能評価を現用PCに限り必須条件から除外しました。他のPCやCPU offload時の性能保証は含みません。配布・導入済み環境の受入は別に管理します。

## オンラインPCで搬送用packageを作る

1. 対象オフラインPCのタスク マネージャーで、GPU名、**専用GPUメモリ**、システムRAMを確認します。共有GPUメモリはVRAMへ加算しません。app、Ollama models、tempの配置先と各空き容量も確認します。
2. オンラインPCで`download-package.bat`を実行します。オンラインPC自身のGPUは自動検出されません。
3. 保存先構成（`SingleVolume` / `ModelsSeparate` / `AllSeparate`）と利用方針（`Speed` / `Balanced` / `Quality`）を入力します。
4. compact、balanced、quality候補のRAM/VRAM条件、搬送容量、配置の見込みを確認し、chatモデルを番号で選びます。Embeddingモデルは`bge-m3`固定です。
5. 内容を確認してダウンロードを開始します。
6. 必要なinstallerと、選択chatモデルおよび固定Embeddingモデルがdownload manifestに従って収集されます。
7. 完了したpackageの`manifest.json`と`checksums.sha256`を保持したまま搬送します。

推薦値は8K context・単一利用を前提にした保守的なプロジェクト推定です。「完全GPU配置」「部分GPU配置」「CPU中心」はすべて見込みであり、GPU、driver、Ollama版、同時利用状況により変わります。性能や完全GPU動作を保証しません。GPUベンダーが未指定でも専用VRAMが入力されていれば容量分類は行います。ベンダー情報はNVIDIA/AMD/Intel等の対応機種・driver注意を表示するために使い、VRAM数値の代用にはしません。

自動化する場合は、`-Yes`と`-ChatModel`に加え、`-TargetRamGiB`、`-TargetStorageLayout`、3保存先の空き容量を明示します。`-DryRun`は質問もファイル作成も行わず、候補順位と見積だけを表示します。RAM最小要件未満を非対話で選ぶ場合に限り、`-ChatModel`と`-AllowUnsupportedChatModel`を併用できます。容量不足やcatalog不正はoverrideできません。

ダウンロード専用処理は、オンラインPCへOllamaやPythonをインストールしません。モデル取得時はcatalogに固定したfull manifest、config、license layerのSHA-256をregistry応答と照合し、tag driftがあればblob取得前に停止します。任意のlocal rerankerを搬送する場合だけ、ライセンス、hash、完全オフライン動作を確認済みのartifactを`optional-components\reranker`へ手動配置し、`-IncludeReranker`を指定します。

## オフラインPCへ導入する

1. packageをローカルdriveへコピーします。
2. package内の`install-offline.bat`を実行します。
3. checksum検証、Python/Ollama確認、モデル配置、設定、smoke testが完了するまで待ちます。smoke test後は`/api/ps`からGPU/CPU配置をbest-effortで表示し、`_internal/install-state.json`の`models.placementObservation`へ記録します。取得不能でも導入失敗にはせず、`ollama ps`での手動確認を案内します。
4. 完了後、`offline-ai\skill-source`へ検索資料を配置します。

installerは既存のPython、PowerShell 7、Ollamaを優先し、不足する必須componentだけを同梱artifactから導入します。Local rerankerは搬送のみで、自動起動・自動有効化しません。

## 検索資料を準備する

検索対象形式は次のとおりです。

- `.md`
- `.txt`
- `.csv`
- `.json`
- `.yaml` / `.yml`
- `.html` / `.htm`

PDFなど対応外の原本は、マルチモーダルLLMまたはlayout対応parser等を使って外部でMarkdownへ変換します。特定ベンダーの利用は必須ではありません。

外部変換時は次を守ってください。

1. 原PDFを検索用Markdownとは別に保持する。
2. 原文を要約、補足、訂正せずMarkdown化する。
3. 原文の見出しだけをMarkdown見出しにする。
4. 表の全セル、図中文字、脚注、日付、金額、条項番号を保持する。
5. 変換後に原PDFと照合する。
6. 非公開資料を外部サービスへ送信できるか組織規程を確認する。

Markdownと同じ場所に`<file>.md.metadata.json`を置くと、ページ、見出し、parser等を検索根拠へ付与できます。metadata sidecarは任意で、なくても検索できます。

## `search.bat`を使う

1. `index.bat build`を起動し、Embeddingインデックスを事前構築します。通常はファイル単位の差分更新で、追加・変更ファイルの全chunkだけを再計算し、未変更ファイルは検証後に再利用します。互換性を再確認したい復旧操作では`index.bat build --full`を使います。Embeddingモデルの識別情報（digest）をOllamaから取得できない場合、同名モデルの差し替えを検知できないため、通常のbuildでも毎回全件を再計算します。資料ファイルを1件でも読み取れない場合は、削除と誤認しないよう更新を中止して`SOURCE_SNAPSHOT_FAILED`を表示するため、ファイルを閉じてから再実行してください。進捗、再計算/再利用件数、生成件数ベースの概算残時間、checkpoint件数だけが表示されます。
2. 長時間処理を中止する場合は`index.bat cancel`、再開する場合は`index.bat resume`を使います。Ctrl+Cの終了コードは130です。
3. `search.bat`を起動して質問を入力します。
4. ready状態ならkeyword、Embedding、RRF等で抽出された資料を基に回答が表示されます。missing/stale/building/cancelled/failed状態では、構築を待たず「キーワード検索のみ」で回答します。
5. 回答と一緒に表示される根拠、route、ページ、見出し、警告を確認します。

Embedding生成はOllamaの`/api/embed`配列入力を使い、件数・順序・次元・有限値を検証します。cacheには資料fileのSHA-256 manifest、モデルdigest、parser/chunk契約、ベクトル次元を保存します。モデルdigestを取得できない場合は既存Embeddingを再利用せず、全件再構築へ倒れます。batch sizeは`OFFLINE_AI_EMBED_BATCH_SIZE`（既定16、許容1～64）、1 batchのrequest timeoutは`OFFLINE_AI_EMBED_REQUEST_TIMEOUT`（既定120秒）、index全体のfail-safe上限は`OFFLINE_AI_INDEX_MAX_SECONDS`（既定6時間）です。checkpointとcacheは利用者資料を含み得る実行時生成物であり、公開packageやreceiptへコピーしないでください。

検索結果は原本の真正性を保証しません。重要な判断では必ず原資料を確認してください。

推論あり（既定「推論: 低」）のまま検索するとモデル生成に数十秒〜4分程度かかりますが、これは想定内の動作です（2026-09-08実測: 中央値100.3秒、最大260.8秒）。`--reasoning off`（環境変数`OFFLINE_AI_REASONING=off`）を指定するとthinkingを無効化し数秒で応答しますが、根拠に基づく検討を経ないため精度は下がります。`--show-thinking`を付けるとCLIでもthinkingの内容を表示できます。

### 新しい環境変数（timeout・性能調整）

| 変数                                  | 既定値      | 用途                                                                                                                                                                                                                       |
| ------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OFFLINE_AI_GENERATION_STALL_TIMEOUT` | `60`（秒）  | 回答生成中にOllamaから何のバイトも届かない時間の上限。thinkingデルタも受信として数える。超過時はWeb UIに「モデルからの応答が途絶えました」と表示される                                                                     |
| `OFFLINE_AI_KEEP_ALIVE`               | `30m`       | chat/embed両モデルをOllama側に保持する時間。`gpt-oss:20b`は初回cold検索でモデルロード約32秒、検索evidence到達約80秒の実測があるため、クエリ間隔が空いても再ロードを避ける。`0`にすると応答直後に従来どおりアンロードされる |
| `OFFLINE_AI_EMBED_CACHE_MEMO`         | `1`（有効） | `embed_cache.json`をmtime/size一致時に再読込しないメモ化。`0`にすると毎回読み直す（トラブルシュート用の退避路）                                                                                                            |
| `OFFLINE_AI_SOURCE_CHUNK_MEMO`        | `1`（有効） | `/api/index/status`が毎回行うskill-source走査（chunk化とgeneration算出）を、相対path・更新時刻・サイズ・実SHA-256集合の一致時に省略するメモ化。同サイズ・同mtimeの変更も見逃さない。`0`で無効化 |

資料の更新を見逃さないため、検索要求ごとにskill-sourceの全ファイルを読み、実SHA-256と更新時刻・サイズで資料世代を確認します（同サイズ・同mtimeの変更も検出）。E版と同規模（479ファイル・約17.9MB）の合成資料で測った目安は、全ファイルのhash走査が中央値約57ms、chunk再構築が約294ms、keyword経路の検索専用処理全体が約413msです（2026-09-14、実資料・実モデルは未使用）。作成直後のファイルでは初回だけ数秒かかる場合がありました（ウイルス対策の走査等によると推測、原因は未確認）。資料規模が大きい環境ではこの分が検索時間に加わります。

### 親子展開（見出しだけの検索候補の配下本文）

見出しだけのチャンク（例: `## 第1章 初動対応` の直下に本文がなく、子見出しだけが続く）が検索に当たったとき、その見出しの配下にある子見出しの本文を根拠へ追加します。検索用チャンクとEmbeddingは変更しないため、有効・無効の切り替えで索引の再構築は不要です。

- Web UI・CLIの根拠一覧とMarkdown保存では「見出し配下の本文を展開」、範囲数・文字数・件数の上限で配下の一部だけを載せた場合は「見出し配下の本文を展開（部分展開）」と表示します。回答用promptにも由来と部分展開を示し、部分展開のときは手順全体を網羅したと述べないよう指示します。
- 引用の行範囲は、実際に渡した本文の行と一致します（途中で切り詰めた場合は最後に渡した行まで）。
- 直接ヒットした根拠と展開した根拠は件数枠を分け合います。直接ヒットが多い質問でも、展開した根拠が上限で黙って落ちないよう、展開元ごとに先に1枠を確保します。
- 検索計画の必須語（`must_find_terms`）を持たない経路（「資料を探す」など）では、展開した根拠だけを理由に根拠ステータスを`sufficient`にしません。

| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `OFFLINE_AI_PARENT_CHILD_EXPANSION` | `true` | 親子展開の有効化。`false`で無効（起動し直すと反映） |
| `OFFLINE_AI_EXPANSION_MAX_PARENTS` | `2` | 1質問あたりの展開元の最大数 |
| `OFFLINE_AI_EXPANSION_MAX_RANGES_PER_PARENT` | `4` | 1展開元あたりに採用する子見出しの最大数 |
| `OFFLINE_AI_EXPANSION_MAX_TOTAL_RANGES` | `4` | 1質問あたりの展開範囲の合計上限 |
| `OFFLINE_AI_EXPANSION_BUDGET_RATIO` | `0.5` | 根拠の文字数上限のうち展開本文に使える割合 |

### prompt予算とcontextウィンドウ（推論あり運用）

prompt・thinking・回答本文は同じcontextウィンドウ（`OLLAMA_NUM_CTX`、既定32768）を共有します。推論ありで根拠が多いと、promptとthinkingでcontextを使い切り、Ollamaが`done_reason: length`で打ち切って**回答本文が0文字**になることがあります。この場合Web UIは黄色の警告領域に理由を表示します（無言で空の回答を返しません）。CLIでも同じ条件で`[警告]`行を表示します。

既定の32768は、2026-09-08の実測（同一10クエリ、`qwen3.5:9b`）で空回答が0/10になる最小のcontextです。8192では3/3、16384では2/10が空回答になります。新しい既定chatは`gpt-oss:20b`です。VRAMが不足する環境で`OLLAMA_NUM_CTX`を下げると、推論ありで回答本文が返らなくなる可能性があります。

回避策は次のとおりです。

- 「推論: 非表示」で再試行する（対応モデルは内部推論を無効化する。gpt-ossは無効化できないため内部推論をlowに抑え、思考欄を表示しない）
- `OLLAMA_NUM_CTX`をさらに大きくする（KVキャッシュのVRAM使用量は増えます）
- 根拠の量を減らす（`OFFLINE_AI_PROMPT_EVIDENCE_CHAR_LIMIT`、`OFFLINE_AI_RETRIEVAL_PROMPT_MATCH_LIMIT`、`OFFLINE_AI_SNIPPET_CHAR_LIMIT`）

**thinkレベル（低・中・高）でthinkingの長さを抑えることはできません。** 実測では`num_ctx=8192`のときlow / medium / highのいずれも同じ長さまでthinkingを生成して打ち切られます。thinkingはcontextの残りを使い切るまで伸びるため、`OFFLINE_AI_GENERATION_RESERVE_TOKENS`でpromptを削っても空回答は解消しません（削った分だけthinkingが伸びます）。既定`low`は、contextに余裕がある状態でthinkingが最も短く速いためです（`num_ctx=16384`で中央値 low 76.7秒 / medium 106.1秒、highは2/3が空回答）。

`OFFLINE_AI_GENERATION_RESERVE_TOKENS`を1以上にすると、根拠snippetの文字上限を`OLLAMA_NUM_CTX`から逆算します（生成用に必ず空けるtoken数を指定する）。既定の`0`では逆算を行わず、`OFFLINE_AI_PROMPT_EVIDENCE_CHAR_LIMIT`だけで決まる従来の挙動です。**既定値はthinkレベル・`OLLAMA_NUM_CTX`との組み合わせを実測してから確定する予定であり、現時点では調査用の設定です。**

| 変数                                     | 既定値 | 用途                                                                  |
| ---------------------------------------- | ------ | --------------------------------------------------------------------- |
| `OFFLINE_AI_GENERATION_RESERVE_TOKENS`   | `0`    | thinkingと本文のために必ず空けるtoken数。`0`で逆算を無効（従来挙動）  |
| `OFFLINE_AI_PROMPT_CHARS_PER_TOKEN`      | `1.4`  | 文字→token換算係数。実測6,180文字→4,361 tokens（1.417）に基づく保守値 |
| `OFFLINE_AI_PROMPT_BUDGET_SAFETY_TOKENS` | `128`  | 換算誤差と固定metadataの余裕                                          |
| `OFFLINE_AI_SNIPPET_CHAR_LIMIT`          | `1200` | 根拠snippet単体の文字上限                                             |

逆算の結果、根拠を1件も載せられない場合は、無言で劣化させず構成エラー（`prompt_budget`）として停止します。`OLLAMA_NUM_CTX`を大きくするか、`OFFLINE_AI_GENERATION_RESERVE_TOKENS`を小さくしてください。

`OFFLINE_AI_EMBED_CACHE_MEMO`が既定（有効）の場合、Web serverの定常メモリ使用量は資料規模に応じて増えます（`embed_cache.json`が数百MB規模なら、常駐RSSが数百MB〜1GB程度増える見込み）。メモリに余裕がない環境では`OFFLINE_AI_EMBED_CACHE_MEMO=0`を設定してください。`OFFLINE_AI_KEEP_ALIVE`を既定の`30m`にすると、`gpt-oss:20b`（約12.90GB）と`bge-m3`（約0.66GB）がVRAMに居座り続けます。先行実測では検索ピーク15,714 / 16,376 MiB、余裕662 MiBでしたが、現行Phase 2の同時常駐再測定はGPU競合で成立していません。他用途とVRAMを共有する環境では、他のGPUアプリを終了したうえで、`0`または短い値（例: `5m`）を検討してください。

## `web.bat`を使う

1. `web.bat`を起動します。
2. 表示された一回限りのbootstrap tokenでブラウザUIを開きます。
3. 「インデックスを準備」から差分更新の事前解析・確認・開始・監視・停止を行います。「全件再構築」は明示的に全chunkを再計算します。確認時の資料generationと開始時のgenerationが変わった場合は開始せず、再確認を求めます。ブラウザをreloadしてもjobは継続します。
4. 検索queryと推論強度、検索タイムアウト（秒）を指定して実行します。モードは既定の「回答を作る」と「資料を探す」から選べます。後者は検索計画と回答生成を行わず、根拠だけを返します。index未準備時やEmbeddingを利用できない場合はキーワード検索へdegradeし、根拠欄にrouteと理由が表示されます。
5. 生成中はthinking（モデルの思考過程）が「思考中…」の折りたたみ欄へ逐次表示されます。回答本文が届き始めると自動的に畳まれます。thinkingの内容はブラウザにのみ表示され、ログや履歴には文字数だけが残ります。
6. 検索実行中は「全体残り R / N秒」がサーバーの確定deadlineを基準に1秒ごと更新表示されます。完了・エラー・利用者中止のいずれでも表示は消え、次回検索で新しい値から始まります。
7. `web.bat`で起動した場合、Web画面のページ（タブ）をすべて閉じるとサーバーは自動で停止します。再読み込みや別タブが開いている間は停止しません。Embeddingインデックスの構築中は完了まで停止を待ちます。閉じた後に使うときは`web.bat`を起動し直してください。組み込みブラウザ等で閉じたことを通知できない場合は、表示中だったページは約1分、背景タブは約5分で停止します。ブラウザの省メモリ機能でタブが休止・破棄された場合も停止するため、画面に「サーバーに接続できません」と出たら`web.bat`を起動し直してください。`OFFLINE_AI_WEB_CONSOLE=1`で起動したコンソール版は従来どおりCtrl+Cで停止します。
8. 回答、根拠、parser警告を確認します。「該当箇所を表示」は検索用テキストの行窓だけを表示し、資料を編集・PDFとして開く操作は行いません。正常完了後の「Markdownで保存」は質問・回答または検索結果、根拠の抜粋、実行条件をブラウザから保存します。保存先や履歴をサーバーへ渡す機能はありません。

Web serverはloopbackへbindし、検索・根拠閲覧・health・bootstrap・Embedding index管理を提供します。全体のtimeout上限はハング防止のための最終防波堤であり、通常はその手前で生成が完了するか、無通信60秒（`OFFLINE_AI_GENERATION_STALL_TIMEOUT`）でstall検知されます。既定はサーバー起動時の`OFFLINEAI_SEARCH_TIMEOUT`（既定300秒）ですが、Web UIの「検索タイムアウト（秒）」入力で検索1回ごとに300〜600秒の範囲から上書きできます（未入力・省略時はサーバー既定）。stallの60秒は利用者設定の対象外で、全体残り時間が残っていてもモデルからの応答が60秒途絶えれば先に打ち切られます。範囲外・小数・非数値をAPIへ直接送った場合はworkerを開始せずHTTP 400 `invalid_timeout`になります。

2026-09-10のruntime改修では、検索計画と回答生成へ同じchat context option（`num_ctx`、`num_batch`、`num_gpu`、`keep_alive`）を明示し、検索SSEのterminal後にHTTP接続を終了するようにしました。UIは`done`/`error`を受け取った時点で検索を終了し、旧検索の遅いreaderが次検索の状態を上書きしないよう検索識別子を使います。status保存はプロセス間lockと遷移検証を伴うため、`embed_index_status.json.lock`とtemporary fileは実行時生成物です。これらは自動テストで検証済みですが、実モデルの再ロード、VRAM、メモリピーク、実ブラウザの1秒以内復帰を保証するものではありません。

source・unit test・distribution policyの自動検証が完了しても、packageを導入先へ反映した実機受入は別工程です。2026-09-07に隔離したEドライブcandidateでは、CLIによる進捗、キャンセル、checkpoint再開、18,284チャンクの正常完了、ready indexを使うhybrid検索まで確認しました。WebのSSE再接続・中止、source変更後のstale判定、server restart、clean Windows package受入は未完了です。

## よくある問題

### 検索できない

- `skill-source`に対応形式の資料があるか確認します。
- Ollamaが起動しているか確認します。
- `.model`と`.model_embed`のモデルが導入済みか確認します。
- `index.bat status`で状態を確認し、必要なら`index.bat resume`を実行します。

### 「処理がタイムアウトしました」/「モデルからの応答が途絶えました」が出る

- 前者（`timeout`）は検索単位の全体timeout（既定300秒、Web UIで300〜600秒の範囲で調整可）を超えた場合、後者（`stall`）は生成中60秒間バイトが届かなかった場合に表示されます。推論ありの通常の生成（数十秒程度）はどちらにも該当しません。
- 資料件数が多い、またはOllamaのモデルがアンロード直後（コールドロード）だとretrievalが長引きます。`gpt-oss:20b`のcold検索では検索計画90秒、retrieval child 120秒の内部期限を使います。`index.bat status`でindexが`ready`か確認してください。
- 頻発する場合は`OFFLINE_AI_KEEP_ALIVE`を延ばしてコールドロードを避けるか、Web UIの検索タイムアウトを600秒まで延ばして原因を切り分けてください。全体残り時間が表示されている間にstallする場合は生成側の無通信が原因なので、`OFFLINE_AI_GENERATION_STALL_TIMEOUT`を一時的に延ばして切り分けてください。

### Web UIを開けない

- `web.bat`を再起動して新しい一回限りtokenを発行します。
- 別PCからではなく同じPCのloopback URLを開きます。
- 使用portが別processに占有されていないか確認します。

### 外部作成Markdownの検索品質が低い

- 見出し、表、箇条書き、ページ境界を原文と照合します。
- OCR誤り、数値、日付、条項番号を修正します。
- 必要ならmetadata sidecarへページ・見出し情報を付けます。

## プライバシー

offline-aiの検索と回答生成はローカルで行います。ただし、資料を外部LLMや外部変換サービスへ送る工程はoffline-aiの管理外です。機密情報、個人情報、契約、組織規程を事前に確認してください。

## 復旧

### 更新前の保全と確認

1. 検索・索引構築を終了し、Webサーバーを通常終了します。構築中の場合は中止完了を待ちます。
2. 現在のアプリ一式を別フォルダへコピーし、相対パス・サイズ・SHA-256の一覧を保存します。利用者資料、`_internal/.model`、`_internal/install-state.json`、索引cacheとcheckpointも旧版と対応づけて保全します。コピーは公開・配布するpackageに混ぜません。
3. 新版を別フォルダで確認し、正常な回答、検索専用、根拠閲覧、Markdown保存、再起動を確認してから通常利用へ切り替えます。既存`.model`を維持する場合は`KeepExisting`を選びます。
4. 資料やEmbeddingモデル、索引形式が変わった場合は、新版側で索引を再構築します。`stale`や互換性エラーを`ready`へ手動で書き換えないでください。

### 旧版へ戻す場合

新版の処理とWebサーバーを通常終了してから、保全した旧版フォルダの起動ファイルへ戻します。更新後に追加した資料・保存結果は別途保全してください。旧版のファイル一覧とhashを確認し、旧版と対応した設定・cacheを使用します。モデルblobを削除したり、Ollama全体を再インストールしたりする必要はありません。旧版の受入が終わるまで新版・旧版のどちらも削除しません。

### 問題の切り分けと再検証

- 資料・索引: `index.bat status`で状態と件数を確認し、資料変更後は再構築します。
- Ollama・モデル: Ollamaの版とモデル名を確認します。モデルが使えない場合も検索専用のkeyword fallbackを確認できます。
- 生成: timeout・中止後に次の検索が可能か確認します。
- Web: サーバー再起動後の新しいloopback URLで確認します。token付きURLを診断記録へ貼り付けないでください。
- package: 搬送前後のchecksumを確認します。破損時は再搬送し、検証を省略して導入しないでください。

診断記録はアプリ・Python・Ollamaの版、モデル名、索引状態/件数、エラーコード、所要時間、候補ファイルのhashを基本とし、資料本文・質問・回答・thinking・認証情報を含めません。アプリ更新時は上記主要操作、モデル/Ollama更新時は加えて検索品質・cold/warm応答時間・VRAM・中止後の復帰を再確認します。

chatモデルを変える場合は、オンラインPCで対象スペックを再入力して新しい搬送packageを作成し、`install-offline.bat`を再実行します。`OverwriteDefault`ではpackage選択値を`.model`へ設定し、`KeepExisting`では既存値を維持してpackage選択値との相違を表示します。`_internal/install-state.json`の`previousChatModel`、`packageSelectedChatModel`、`appliedChatModel`を確認すれば、更新前の値へ手動で戻せます。

検索機能の変更前には、対象ファイル、bytes、SHA-256を記録したsnapshotを作成してください。復旧はworkspace相対pathへno-conflict restoreし、hash一致を確認します。このworkspaceではGitコマンドを復旧手段として使用しません。
