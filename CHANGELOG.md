# Changelog

## 0.1.0 - Unreleased

- 見出しだけのチャンクが検索に当たったとき、その見出し配下の子見出しの本文を根拠へ追加する親子展開を追加した（既定ON、`OFFLINE_AI_PARENT_CHILD_EXPANSION=false`で無効）。検索用チャンクとEmbeddingは変更しない。直接ヒットが多い質問でも展開した根拠が件数上限で黙って落ちないよう展開元ごとに先に枠を確保し、件数枠・根拠予算を適用した後の採用根拠だけで根拠ステータスを再判定する。切り詰めた展開根拠の行範囲は実際に渡した本文の行へ合わせる。範囲数・文字数・件数の上限は子の類似度の高い順に割り当て、配下の一部だけを載せた場合はWeb UI・CLI・Markdown保存・回答promptで「部分展開」と示す。検索計画の必須語を持たない経路（「資料を探す」など）では、展開した根拠だけを理由に`sufficient`にしない。評価harnessに、必要な行範囲の残存・部分展開の期待・候補段階のrecall・無関係な展開の件数・保留質問・回答品質probe（`--measure-answer-quality`）・展開OFF/ON比較（`--compare-expansion`）を追加した。自動試験とkeyword経路の評価のみ実施済みで、実モデル・実画面・元の失敗事例での受入は未実施。

- `web.bat`で起動したWeb UIは、ページをすべて閉じるとサーバーを自動停止するようにした（再読み込み・別タブは猶予内で継続、索引構築中は完了後に停止、コンソール起動は従来どおり）。旧形式（version 4）の索引に対する確認ダイアログが全ファイルを「追加」と表示していた不具合を修正した。

- Embeddingインデックスをファイル単位の差分更新にした。未変更fileは検証済みvectorを再利用し、追加・変更fileは全chunkを再計算、削除・renameされたfileのentryは除外する。モデルdigest・parser/chunk契約・schemaを互換条件とし、不明・不一致時は全件計算する。候補は一時ファイルの再読込検証と昇格直前の資料hash・モデル再確認後だけ原子的に置換し、失敗時は旧cacheを保持する。`index.bat build --full`、Webの「全件再構築」、副作用のない`/api/index/plan`による確認ダイアログ（差分件数・再計算/再利用件数・生成速度ベースの見積もり・全件計算の理由）を追加した。自動試験のみ実施済みで、実モデル性能・実画面・導入版の受入は未実施。

- 配布対象外のダウンロード途中ファイルや実行時cacheをinventory作成時に読み取らず、ファイルロックで配布検証が失敗する不具合を修正。配布対象ファイルのhash検証とpath/reparse検証は維持する。

- 搬送packageのモデル注意書きを現用PCの受入結果へ同期し、更新前の保全、旧版への復帰、本文を含まない診断項目と再検証手順をセットアップガイドへ追加した。

- 索引構築の中断通知で中止状態を先に確定せず、実際の例外に応じた失敗コードまたは中止状態を保持する。

- gpt-ossの思考非表示モードで、無効化できない内部推論へ`think: false`を送っていた互換性問題を修正した。このモデルでは最小の`low`を明示し、thinking SSE/UIの非表示は維持する。他モデルの無効化と明示low/medium/highは維持。UIの「無効・最速」という表現を「非表示」へ修正した。

- 根拠本文とSHA-256を同じ読取りデータから生成し、資料更新による取り違えを防止した。遅延した閲覧応答を検索・閲覧単位で無効化し、根拠IDの期限と上限破棄順を閲覧順に依存させないよう修正した。

- Web UIに根拠行ビューア、正常完了結果のMarkdown保存、「資料を探す」検索専用モードを追加した。検索専用はchat呼出しなしでkeyword/ready Embedding検索を行い、根拠IDは認証セッション・source境界・検索時SHA-256へ束縛する。実ブラウザ・実モデル・導入版の受入は別途確認が必要。

- 生成用60秒stall監視を回答生成開始時まで延期した。検索計画のcold load中に60秒で打ち切られる問題を修正し、検索計画90秒・retrieval120秒・全体300〜600秒のdeadlineは維持する。qwen lowの300秒timeoutと実機受入は別途確認する。

- Chat既定値とcatalogのbalanced枠を`gpt-oss:20b`へ移行した。`qwen3.5:9b`はcatalogから削除したが、`LEGACY_MODELS`には追加せず、既存`.model`のモデル名と内容を変更しない。schemaVersion 1互換fallbackの`$DefaultChatModel = "qwen3.5:9b"`は維持する。Phase 1の12回cold測定はevidence最大34.140秒、本文あり・done 12/12だったため、A-3 cold evidence閾値案を45秒へ改訂した。Phase 2のVRAM同時常駐測定は、GPU開始時使用量3,418 MiBの競合下でchat常駐に至らずNOT_RUNとした。
- 2026-09-12残差再測定: Ollama 0.31.2の`gpt-oss:20b`は、`think: false`を送っても内部responseに`message.thinking`を返す場合があることを直接実測した。reasoning `off`では診断用文字数だけを保持し、thinking SSE/UIイベントを公開しない最小修正を実装した。新candidateの固定10件でA-9はthinking delta 0、本文10/10、done 10/10、stall/error 0となりPASS。A-10は同一runner・固定spec・通常モデル解決でkeyword/hybrid/agentic-lite全経路PASSだったが、旧candidateのhybrid不安定性は残るため、品質受入は最新receiptと再現性を分けて扱う。
- 実Chromeのgpt-oss candidateで空回答UI（根拠0件・資料なし）と、qwen競合ロード時の60秒stall表示「モデルからの応答が60秒途絶えました」を確認した。Phase 2 VRAM再測定は単独成功時に671 MiBの空きが観測された一方、warm/競合条件でstallし、同時常駐と閾値を確定できなかったためNOT_RUN。既存`.model = qwen3.5:9b`は再確認でも固定10件継続FAIL（本文8/10、2 timeout）であり、catalog・legacy・ファイル内容は変更していない。
- Phase 4残存診断で、他GPUアプリ終了後の通常解決gpt-oss検索は検索計画APIが53.797秒、candidateの検索evidenceが79.984〜82.391秒だった。検索計画90秒、retrieval child 120秒へ延長し、同candidateでA-1本文10/10・done 10/10、A-2 evidence最大21.219秒を確認した。A-3 cold evidence閾値は実測最大82.391秒＋10秒余裕の95秒へ更新する。
- 実行安定性の再改修を実装した。index statusはプロセス間lock、writer別unique temporary file、`job_id`/generation/state遷移の検証を組み合わせ、古いworkerの巻戻りを拒否する。旧固定名temporary fileの回収互換性と配布対象外分類も維持した。
- 検索SSEは`done`/`error`のflush後にHTTP応答を終了し、Web UIはterminalイベントで一度だけ操作可能へ戻る。検索識別子で旧readerの遅い終了が次の検索状態を上書きしないようにした。index SSEの再接続契約は維持する。
- 補助検索計画と回答生成で`num_ctx`、`num_batch`、`num_gpu`、`keep_alive`の方針を共通化した。統一前は補助検索が`num_ctx`未指定でモデル既定へ戻り、回答生成の32768と互いをアンロードし合うため、**chat要求のたびに再ロードが発生していた**。固定10クエリの前後比較（reasoning off、両モデル解放から開始）で、1秒超のreloadは20/20→0/20、`load_duration`合計は133.109秒→6.072秒、検索completeの中央値は29.891秒→24.094秒になった。VRAMピークは10,885→10,846 MiBで増加せず、補助検索出力の語集合は10件すべて一致、evidenceのchunk集合も10/10一致で品質低下はない。
- `embed_cache.json`を全文を1つの`str`にせず逐次読込する経路を追加した。441 MBのcacheは内容にBMP外の文字が1つでもあると`str`全体がUCS-4になり、`read_text`だけでpeak working setが3.83 GBに達していた。UTF-8のincremental decoderで1 MBずつ読み、値のparseは標準`json`のC scannerへ渡して`entries`を1件ずつ積む。実データでpeakは3.840 GB→0.862 GB、所要は4.610秒→3.594秒になり、結果は全文parseと完全一致する（key順・entries順を含む）。cache形式と`save_embed_cache`は無変更で、構造が想定外・破損の場合は従来の全文parseへfallbackする。memo、generation一致、writer経路、途中再開の契約も変更していない。
- 2026-09-10時点でPython unit 411件、distribution policy V-1/V-2/V-4/V-5/V-6、TargetSpecPackageE2E 1件、formatter/lintを確認した。実モデル・実ブラウザ・メモリピークを含むruntime受入は未完了で、判定はNO-GOを維持する。

- 推論ありで回答本文が0文字になる事象を解消した。既定 `OLLAMA_NUM_CTX` を8192から**32768**へ、既定 reasoning を `medium` から**`low`**へ変更した。2026-09-08の実測（同一10クエリ、`qwen3.5:9b`）で、8192は3/3・16384は2/10が空回答になるのに対し、32768 + `low` は**空回答0/10**・中央値100.3秒・p95 260.8秒である。VRAM占有は6.74GB（16384比 +0.57GB）。
- 検索単位の全体timeoutの下限を120秒から**300秒**へ引き上げた（範囲300〜600秒、既定300秒は据え置き）。推論ありの生成が中央値100.3秒・最大260.8秒であり、120秒上限は実用にならないため。
- **thinkレベルによるthinking長の抑制は成立しないことを実測で確認した。** `num_ctx=8192` では low / medium / high のいずれも `eval_count` が同一（3011/3000/2992）で全件が `length` 打ち切りになる。thinkingはcontextの残りを使い切るまで伸びるため、prompt予算の逆算（`OFFLINE_AI_GENERATION_RESERVE_TOKENS`）でpromptを1,000 tokens削っても、thinkingが同じだけ伸びて空回答は解消しない。このため予算逆算の既定は`0`（無効）のままとし、調査用の設定として残す。

- 推論ありで回答本文が0文字のまま無言終了する事象を可視化した。`iter_stream_events`がストリーム終端で`("done", done_reason)`を1回yieldする契約へ変更し、`run_search`は本文0文字のとき`{"type": "answer_empty", "reason": ...}`を`done`の直前に1回だけ送出する。Web UIは`length`（context枯渇による打ち切り）とそれ以外を区別した警告を表示し、CLIも同条件で`[警告]`行を出す。`iter_stream_chunks`の返す内容は変えない（後方互換）。**空回答そのものの解消はthinkレベル・`OLLAMA_NUM_CTX`の実測後に行う。**
- stall検知時のerror codeを修正した。生成フェーズのchat streamで発生するsocket timeoutを`internal_error`ではなく`code: "stall"`へ写像する。接続不能（Ollama停止など）とretrievalフェーズのtimeoutは対象外とし、原因判別を保つ。stallメッセージは`run_search`経路とSSE経路の両方で「モデルからの応答が{n}秒途絶えました」に統一した。
- prompt予算を`OLLAMA_NUM_CTX`から逆算するscaffoldを追加した。`OFFLINE_AI_GENERATION_RESERVE_TOKENS`（既定`0`＝逆算無効・従来挙動）を1以上にすると、context全体から生成用の予備とprompt shellのオーバーヘッドを差し引いた残量と`OFFLINE_AI_PROMPT_EVIDENCE_CHAR_LIMIT`の小さい方を根拠snippetの上限に使う。換算係数`OFFLINE_AI_PROMPT_CHARS_PER_TOKEN`（既定1.4）と余裕`OFFLINE_AI_PROMPT_BUDGET_SAFETY_TOKENS`（既定128）で調整でき、残量が0以下なら根拠を強制挿入せず`prompt_budget`エラーでfail closedする。根拠snippet単体の上限は`OFFLINE_AI_SNIPPET_CHAR_LIMIT`（既定1200）として定数化した。既定値は実測後に確定する。
- `/api/index/status`のread経路に、skill-sourceの(相対path, 更新時刻, サイズ)集合をキーとしたmemoを追加した（`OFFLINE_AI_SOURCE_CHUNK_MEMO=0`で無効化可）。毎回の全ファイル走査とgeneration算出を省く。index buildのwriter経路はmemoを通らない。
- `IndexCoordinator.shutdown()`がworker不在時に状態を再評価しないようにした。`cancel()`経由で`status()`を呼ぶため、実データでCtrl+C終了に数秒かかっていた。build実行中のcancel要求は従来どおり行う。

- 検索単位で全体timeoutを120〜600秒の整数から指定できるようにした（Web UI入力、既定300秒）。`/api/search`は`timeout_seconds`クエリパラメータをサーバー側でも検証し、範囲外・小数・非数値・重複指定はworkerを開始せずHTTP 400 `invalid_timeout`で拒否する。省略時は従来どおりサーバー既定を使うため、既存のAPI呼び出しは無変更で動作する。`/api/health`は`searchTimeoutDefault`/`searchTimeoutMin`/`searchTimeoutMax`を公開し、Web UIの入力属性・初期値をハードコードではなくサーバー設定と同期する。
- 検索中はSSEの`budget`イベント（`timeoutSeconds`・サーバーdeadline基準の`remainingSeconds`）を購読開始時に送出し、Web UIが「全体残り R / N秒」を1秒ごとに更新表示する。表示はブラウザのバックグラウンド抑制・スリープ後も経過時間から再計算し、`done`/SSE `error`/fetchエラー/利用者中止/`onEnd`の全終了経路でタイマーを解除して最終値を残さず非表示にする。無通信60秒で打ち切る既存のstall timeoutとは表示上明確に区別する（stallは別途`code: "stall"`のエラーメッセージで通知）。
- 推論あり（UI既定「推論: 中」）のまま検索がtimeoutせず完了するよう、生成フェーズの一律wall-clock上限をstall（無通信）ベースの制御へ置き換えた。全体上限`OFFLINEAI_SEARCH_TIMEOUT`は120秒から300秒（ハング防止のfail-safe）へ引き上げ、新規`OFFLINE_AI_GENERATION_STALL_TIMEOUT`（既定60秒）が生成中の無通信を監視する。CLIの`API_TIMEOUT`も同じstall値へ統一した。
- 生成中のthinkingをWeb UIへ逐次表示するようにした（折りたたみ表示、「思考中… (n文字)」ヘッダ、content到着で自動的に畳む）。CLIには`--show-thinking`フラグを追加。thinking本文はブラウザ表示のみに限定し、ログ・receiptへは文字数のみ記録する。
- `think`パラメータを常に明示送信する契約に変更（`推論: なし`選択時もOllama既定のthinkingへ戻らないよう`think: false`を送る）。thinkパラメータでのエラー時fallback条件も、reasoning値ではなく`think`キーの有無で判定するよう修正。CLIの`--reasoning`に`off`を追加（既定は`medium`のまま）。
- retrieval（検索計画〜候補採点）を短縮。`get_embed_index_status`へキャッシュを渡して内部の二重ロードを廃止、`load_embed_cache`にmtime/sizeベースのmemoを追加（`OFFLINE_AI_EMBED_CACHE_MEMO=0`で無効化可）、複数クエリ変体のEmbeddingを`embedding_search_multi`で1バッチ・1パス採点するよう変更。chat/embed両リクエストへ`keep_alive`（既定30分、`OFFLINE_AI_KEEP_ALIVE`で調整）を付与しコールドロードを回避する。
- `/api/index/status`を軽量化（`validate_entries=False`で全件検証を省略しヘッダ+key集合一致まで）。Web UIのポーリング間隔をindex構築中は15秒、それ以外は60秒へ適応化し、検索実行中は停止する。
- index状態表示が実測と乖離したまま固着する不具合を修正。所有者（worker）不在で実測が`ready`のとき永続`cancelling`/`building`等を上書きしないようにし、`cancel()`も所有者不在時は状態を書き換えず実測へ委ねる。プロセス起動時（web_server / index.bat）に孤児状態を実測へ戻すreconcileと、孤児`embed_cache.json.tmp`の回収（`embed_cache.json`をJSONとして直接parseでき`entries`を持つ場合のみ削除。破損・未存在なら削除せず`error_code` `EMBED_CACHE_UNREADABLE`をindex statusへ残す）を追加した。
- Embedding初回構築を`index.bat`／Webの明示的なbackground jobへ分離。Ollama `/api/embed`配列batch、generation-bound checkpoint、atomic final cache、cancel/resume、Web status/SSEを追加し、検索要求はindex未準備時にkeyword-onlyへdegradeするよう変更。
- 通常検索の全体timeout 120秒を維持し、retrieval child timeout（既定45秒）とindex batch/全体timeoutを分離。query、本文、vector、絶対pathはindex status/eventへ保存しない。
- Eドライブ上の18,284チャンクで、キャンセル／checkpoint再開、60分以内のfull build、ready indexによるhybrid検索を確認。実測区間の安定性と未計測のピークRAM/VRAMを踏まえ、Embedding batchの共通既定値を16に設定（1～64で上書き可能）。

- 製品責務をローカル資料の検索と根拠付き回答生成へ限定。
- 未公開段階のPDF変換CLI、Web upload/API、multipart parser、関連package依存を撤去。
- モデルのatomic copy、保存先driveのpreflight、検索timeoutを維持。
- metadata sidecar読取、Embedding、keyword検索、RRF、reranker transportを維持。
- Phase 2向けに、固定corpus・固定質問で検索品質と根拠品質を測る評価ハーネス（`tests/eval/`）を追加。keyword / hybrid / agentic-liteの3経路を同一条件で比較し、redacted receiptを出力する。
- metadata sidecar（`*.metadata.json`）を検索対象から除外。原本の複製が根拠として二重提示され、根拠precisionを下げていた。
- 日本語クエリでconfidenceのcoverageが常に0になる不具合を修正。CJKを含む語を2文字n-gramへ展開して一致率を測る（`coverage_terms()`）。
- 根拠ステータス`sufficient`の宣言にEmbedding由来の根拠を必須化。keyword一致だけでは質問の限定条件を満たさない資料を区別できず、該当情報なしを維持できなかった。
- 回答生成時に、地域・対象・期間・役職・区分など質問の限定条件を資料へ個別照合し、条件不足・対象外の場合は類似資料の数値を流用せず「該当情報なし」とするprompt契約を追加。
- Phase 2評価ハーネスへ、回答本文をreceiptへ保存せずQ10のanswer層を測る`--measure-answer-layer`と、製品`prompt_templates.py`のSHA-256束縛を追加。Ollama実測でanswer probeのPASS/FAILをretrieval指標から分離して記録する。
- Phase 2の近接語・限定条件検索を改善。弱いEmbedding候補を根拠支持から除外し、明示的な対象外条件で`sufficient`を禁止する。Agentic-lite単独評価でもEmbedding cacheを構築し、検索計画失敗後の重複LLM呼び出しを廃止した。検索計画は短いJSON生成に限定されるため、thinking対応モデルでも`think: false`を明示して不要な推論tokenとtimeoutを防止する。
- Ollamaモデルの可用性判定で、省略されたタグを`:latest`として扱うよう修正。標準設定`bge-m3`がAPI応答`bge-m3:latest`と一致せず、Embedding検索を誤って無効化する問題を解消。
- 大規模なEmbedding初回構築に件数進捗と再開可能なcheckpointを追加。generation・モデル・chunk設定へcheckpointを束縛し、最終cacheの保存検証後だけcheckpointを整理する。CLI/Webのキャンセル・timeout・同時writer競合では既存cacheを保持し、keyword fallbackまたは次回再開へ明示的に移行する。
