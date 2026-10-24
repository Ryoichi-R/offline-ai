# offline-ai 内部アーキテクチャ

## deepモードの検証・終了契約（2026-09-21）

- `deep_research.py`は全初期観点を1巡回として処理し、抽出された参照先・不足点と資料見出しから追加探索する。候補から親の導入本文も別行範囲で保持する。長い単一行の既読キーは資料hash・行・文字位置を含む。
- 抽出JSONは必須型を検証し、破損時は共通期限内で1回だけ再試行する。抽出・最終回答には別の原文支持照合を行う。モデル照合の正しさは実資料・保留群の人手評価で測る。モデルによる自己照合を精度の保証にしない。
- `deep_transport.py`は要求専用の標準ライブラリ子プロセスを使用する。応答ヘッダー待ち、停止したstream、候補検索・候補選別を親の単調時計とキャンセルで監視し、workerにも親期限を渡して選別ループ内で期限切れをfail fastする。中止時はその子のみをkill/communicateして回収する。共有Ollama自体は停止しない。推論が実際に終了した時間は実機試験で別に測る。
- deep入口はモデル設定を読み取るだけで、旧モデル設定の自動移行を行わない。既存互換cacheがあるときはkeywordとembeddingのランキングを再利用し、検索中のcache再構築は行わない。cacheは現在のsnapshotと一致するentryのみ使う。
- CLIとWebは受付時の絶対期限を共通処理へ渡す。探索は統合照合120秒・終了5秒を予約し、モデル入力全体に32,768 context・4,096生成予約・1,024安全予約を適用する。保持根拠12,000字とtoken上限を別々に守る。
- 資料hashを読み取り直前と最終確定前に再検証する。失敗・未確認・上限で落ちた範囲は台帳へ記録する。未検証の解釈はfallbackへ出さず、元の抜粋を表示する。診断dictには本文・質問・回答・自由文エラーを格納しない。
- deepの候補検索は全snapshotを順位付けした後、`deep_candidate_selection.py`で資料別8件・全体40件を選別する。workerから親へはschema version 2の採用候補・除外参照・件数だけを返し、本文・snippet・ベクトルは転送しない。大量の除外参照はpathとsource SHA-256をそれぞれテーブル化し、参照indexで重複転送を避ける。除外参照はcompactなindex形式だけをworker契約として受け付け、親側で展開する。親は同じsnapshotのchunk IDから採用候補を復元する。空pathは旧親処理との互換性のため選別時に即時失敗させず、本文なしで親へ渡して`source_missing`として台帳化する。一方、chunk ID欠落・不正な行範囲など本文復元不能な候補はworker境界でfail closedする。選別前後の件数は保存するが、網羅率・正答率とは解釈しない。
- retrieve応答のUTF-8シリアライズ後サイズは32,000,000 bytes以下を要求する。超過は固定コード`candidate_transfer_limit`として伝達し、初回検索で根拠が無ければ`failed`、既存根拠があれば`partial`とする。通常の候補選別による除外はエラーではなく調査台帳へ記録する。終了理由の優先順位では、cancel・期限・source変更の既存優先を維持したうえで、候補転送超過を通常のevidence/document/unit/round limitより優先し、通信失敗を別の上限理由へ隠さない。不明なschema version・項目・件数不整合は一般障害としてfail closedする。入力JSONの大きさと検索中メモリは別の残存制約であり、本契約はそれらを保証しない。
- deepのreplayは進捗を最新状態へ集約し32イベント／1MB、購読キューは64イベントで制限する。長い実資料で保持上限を早期に消費しないよう、未確認事項・調査台帳・回答本文はSSE投入前に件数・文字列長・UTF-8バイト数を制限する。予期せぬ保持上限超過時は固定エラーで終了する。terminal確定後のイベントは無視し、実行枠は要求ごとに一度だけ解放する。通常モードのstream蓄積仕様は変更しない。
- 自動試験のPASSと、実モデルの精度・Ollama推論終了・30分・通常要求競合・外部切断の受入は別に管理する。後者の未達状態を公開・運用反映の許可に読み替えない。

## deepモードの回答未生成是正（2026-09-23）

2026-09-23の回答未生成調査（実資料保存でpartial/time_budgetのみとなり回答統合が成立しなかった事案）を受けて、以下4点を是正した。合成試験のPASSであり、実資料300秒1回の再受入・E反映・長時間試験は別途行う。

- **本文優先の読取**: `_ranges_with_context`はヒット節の範囲を常に先頭で返し、親導入文脈（祖先見出しの本文）はその後に続く別範囲として処理する。従来は親冒頭側から順に抽出・検証していたため、時間予算切れ時にヒット本文へ到達できないことがあった。親本文は削除せず、単に処理順位を下げるだけで、`親本文＋子節を欠落なく保持する`契約自体は変えていない。
- **除外候補の集約**: `_summarize_unconfirmed_for_prompt`が`candidate_limit`/`document_limit`/`unit_limit`/`viewpoint_limit`/`context_budget`の各理由を件数（`reason: N件`）へ集約し、最終回答prompt（`build_deep_final_prompt`）へはこの集約版だけを渡す。`result.unconfirmed`／`ledger`自体は全件保持のままで、UI・診断向けの「未読を隠さない」契約は変えない。集約対象外の理由（`model_error`・`unresolved`等）は個別のまま残る。回答を確定できず原文抜粋を表示する`_fallback_answer`も同じ集約（`_collapse_bulk_unconfirmed`）を使い、上限理由は件数だけを本文に出す。個別一覧は未確認事項欄・調査台帳に残るため、本文が再接続用保持上限で切り詰められて原文抜粋自体を失うことを防ぐ。
- **回答状態と診断の分離**: `DeepResearchResult.answer_state`（`generated` / `verification_failed` / `not_generated`）を追加し、最終統合の成否と原文一致判定を分離した。最終回答のモデル呼び出し自体が返らなかった場合（timeout・通信失敗・prompt予算超過）は`not_generated`、回答は返ったが根拠ID検査または支持照合に通らなかった場合は`verification_failed`とする。`web_services.build_deep_evidence_event`は固定`confidence`（根拠の有無だけで1.0/0.0を返していた）を`None`にし、`answerState`/`answerStateMessage`へ置き換える。`web/index.html`はdeepモードのときだけ「信頼度」表示を「回答状態」表示に切り替える。
- **時間・失敗原因の記録**: `DeepBudget`にmonotonicなstage別累積時間（`stage_seconds`）と全体経過（`elapsed_seconds`）を追加し、`diagnostics`へ含める。最後の段階（通常は`integrate`）の時間を取りこぼさないよう、`complete`段階への移行を`diagnostics`確定より前に行う。段階は`planning`/`search`/`read`/`integrate`の4区分で、節ごとの抽出と照合は`read`に合算される。Web UIは`deepDiagnosticsFormat`で全体経過・段階別時間・処理件数（数値の許可項目だけ）を整形し、根拠欄の注意と保存Markdownの実行条件の両方へ同じ文言で出す。呼出timeout・stall・URLErrorが混在する`time_budget`終了理由だけでは時間消費の内訳が分からない問題（2026-09-23調査の指摘5）に対し、段階別内訳を独立して残す。UI側は`completedAt`（サーバー確定の完了時刻）と`resultReceivedAt`（クライアント受信時刻）を別フィールドとして保持し、`completedAt`欠落時に受信時刻へ暗黙フォールバックしていた挙動（保存された時刻差の原因を切り分けられなかった要因）を廃止した。保存Markdownは両方を別々の行として明記する。
- **探索中の呼出timeoutは探索の終了として扱う**: 探索中のモデル呼出には「残り時間−統合・終了予約」がtimeoutとして渡る。このため、その呼出のtimeout（worker期限の`TimeoutError`、worker応答の`timeout`/`stall`）は予約境界での探索終了を意味する。調査全体の失敗として外側の例外処理へ流すと、照合済み根拠があっても統合を実行しない（2026-09-23の実資料受入: 300秒中175秒で`not_generated`、未確認事項は詳細なしの`time_budget:`だけ）。探索ブロックで`time_budget`（詳細「モデル呼出が時間内に終わらず探索を終了」）として記録し、統合へ進む。中止（`cancelled`）と全体期限（`DeepBudgetExpired`）は従来どおり外側へ伝播する。
- **読む資料は全観点の集計重要度順**: 1巡回の全観点の選別候補を資料ごとに`1/(60+順位)`で合算し、合計の高い資料から読む（資料内は候補ごとの合算順、見出し由来の候補はその資料の末尾）。従来は候補の登録順に資料上限5件が埋まり、実質的に最初の観点の上位だけが読む資料を決めていた。2026-09-23の実資料では、中心資料が全観点の上位20位内に6チャンクあったのに6番目に現れる資料となり、未読のまま無関係な資料が文字予算を消費した。上限値（資料5・単位20・文字12,000）は変えていない。
- **検索は1巡回1回のworker呼出**: `retrieve_many`で1巡回の全観点をまとめて順位付けし、Embedding cacheの読込と全チャンクの受け渡しを巡回ごとに1回にする。応答は観点ごとのschema 2応答を順に並べた封筒（`schema_version` 1）とする。除外候補の参照からは順位スコアを省く（親は未読記録にしか使わない）。18,254チャンクの実測では、スコア付きの1観点応答は約5.2MB（5観点で25.9MB）だった。封筒が転送上限を超えた場合は、その巡回だけ従来の観点ごとの呼出へ戻す。それ以外のworker失敗は、戻さずに伝播する。
- **最終回答の不合格理由を区別する**: 根拠IDの不足・不正、根拠に支持されないという照合結果、照合の未完了を、未確認事項に別々の文言で残す。停止理由（`model_error`）と回答状態（`verification_failed`）は従来どおり。
- **最終回答の根拠ID検査の前処理と1回の書き直し**: 最終回答の指示文は、未確認事項の欄を書かせない。従来は「確認できない事項は未確認事項に分けて」と指示しながら、見出し以外の全行に根拠IDを要求していたため、その欄を書くと必ず不合格になった（2026-09-23の1800秒実行）。モデルが未確認事項の欄を書いた場合は、見出しが実質「未確認事項」だけの欄を次の見出しまで取り除く。未確認事項は従来どおりプログラム側の一覧で示す。`[E8, E6]`・`【E8】`・全角などの表記は`[E8][E6]`へ機械的に揃えてから検査する。IDの実在検査は変えない。不合格なら、違反した行番号と前回の案を示して、同じ期限内で1回だけ書き直させる（残り時間が30秒未満なら行わない。書き直し呼出のtimeoutは不合格のまま扱い、中止は伝播）。診断`final_answer`には、試行ごとの件数（根拠IDなしの行・不明IDの行・除いた未確認欄の行・表記を直した引用）と固定の結果コードだけを残し、回答文は残さない。Web UIの診断表示にも同じ内容を1行で出す。
- **根拠の限定を落とさない**: 最終回答の指示文で、括弧書きの限定（「〜を除く」「〜を含む」「〜のほか、…の案件」）と適用範囲の限定（「前項第○号の規定に関わらず」）を原文の文言のまま書くよう求める。同一モデルの支持照合は、第3号の限定を落とした回答を通した（2026-09-23の1800秒実行）。このため、機械的な検査も加える。回答の各行を、引用した根拠の中で最も近い原文行（全文を含む、または最長一致12字以上かつ原文の4割以上）に対応付ける。その原文行の限定語付き括弧書き（除く・含む・限る・ほか・以外・のみ）が文字2-gramで6割以上残っているかを確かめる。箇条項目の前置き（「前項第1号又は第2号の規定に関わらず」）は、その項目を使う回答のどこかに残っているかを確かめる。違反は行番号と原文の限定を示して書き直しを1回求め、それでも残れば原文抜粋へ戻す（結果コード`rejected_limitations`）。別名の括弧（「（秘密随意契約）」）は限定として扱わない。
- **読んだが関係なしと判定した節を示す**: 診断`irrelevant_ranges`（パスと行範囲のみ、読取単位の上限まで）に残し、Web UIの診断表示にも出す。公開用の調査台帳は、候補上限などの大量の定型理由を後ろへ回し、表示上限でもこれらの単発の記録が残るようにした。2026-09-23の実行では、第4条（審査時期）は読まれたうえで関係なしと判定されていた。記録がないため、未読と区別できなかった。
- **括弧内の句点で文を区切らない・欠けた限定は原文で補う**: 根拠ID検査は、括弧内（全角・半角、入れ子を含む）の「。！？」では文を区切らない。原文どおりの「（…を含む。）」が根拠IDのない文と誤判定されていた（2026-09-23の再現で確認）。書き直し後も限定が欠ける行は、対応する原文行（箇条記号を除いた原文そのもの）に置き換える。欠けた適用範囲は、前置きの原文行をその行の前に挿入する。いずれも文ごとに根拠IDを付け、置き換え後に根拠ID検査と限定検査をやり直し、その後の支持照合も通常どおり行う。置き換えた内容は、未確認事項（`limitation_repaired`）に原文の限定の文言とともに示す。停止理由の決定には使わない。診断`final_answer.repaired_lines`とWeb UIにも件数を出す。置き換え後も検査に通らない場合だけ原文抜粋へ戻し、欠けた原文の限定（最大3件）を未確認事項に示す。同日の同一条件3回は、3回とも限定の欠け2行で回答全体が破棄されていた。
- **入れ子の括弧の限定と、1行ずつの支持照合**: 限定語は、入れ子の括弧を除いた各括弧自身の階層で探し、最も外側の限定括弧を限定として扱う。第1条の「（…事業（以下「基金事業」という。）等であって、…選定を厚生労働省で行う場合を含む。以下同じ。）」が対象になる。回答行の対応付けは、本文に加えて限定括弧の中身も候補にする（括弧の中身、例えば基金事業の定義を言い換えた行を拾うため）。2026-09-23の3回中2回で、これを見逃した「基金事業は諮問対象とする」が照合を通過した。最終回答は、根拠ID・限定の検査と原文補完の後、各行をその行が引用した根拠だけと並べて1行ずつ支持照合する（原文そのままの行は除く）。支持されない行は、対応する原文行に置き換える。対応する原文行がなければ削除し、未確認事項（`line_unsupported`、停止理由には使わない）に示す。全体照合のために、終了予約とは別に60秒を残した時点で1行ずつの照合を打ち切る。照合の件数（照合・不支持・置換・削除・未照合）は診断`final_answer.line_checks`とWeb UIに出す。その後、従来の全体照合を行う。全行が削除された場合は原文抜粋へ戻す。実資料の根拠14件でローカル再現すると、1行ずつの照合は10行で約97秒だった。
- **長い原文の言い換えも対応付ける・質問が問う事項だけを書かせる**: 回答行と原文行の対応付けに、回答行の文字2-gramのうち原文行に含まれる割合（12個以上で5割以上）を加える。2026-09-23の3回中1回で、92字の第1条を言い換えた行（共通する最長の連続は24字）が、限定の検査から漏れた。最終回答の指示文に、質問が直接問う事項だけを書き、目的・背景・組織・所管範囲の説明を書かないこと、各文を主語と述語のある自己完結した文にすることを加える。同じ回で誤りや要注意の記述が出たのは、いずれも背景説明の行だった。
- **1行ずつの照合を分けた読み取りで判定する**: 1行ずつの照合は、全体照合と共通の真偽1つの指示文ではなく、専用の指示文（`DEEP_LINE_VERIFY_SYSTEM`）で次を別々に答えさせる。その行を述べた原文の1文、その文の種類（質問が問う事項を定めるか、背景・経緯・指示・目的・依頼の説明か）、原文と行それぞれの審査・処理の主体、支持の有無である。判定はプログラムが行う（`_line_verdict`）。背景の説明は行ごと削除する（原文へ置き換えると、背景が回答に残るため）。主体の名称がどちらも示され、一方が他方を含まない場合は不支持とする。原文の文が見つからず種類が空で、支持なしの場合も不支持とする。件数は`line_checks.off_question`とWeb UIに出す。2026-09-24の3回中1回で、設置通知の前文（大臣指示・コスト削減）を付議条件として書いた行が、真偽1つの照合を通過した。同じ根拠でローカル再現すると、この行は除かれ、正しい付議条件・除外の各行は通過した（1行あたり約10秒で、従来の約6秒より長い）。特別会計の勘定元が自ら審査する案件を本委員会の対象とした行は、モデルが原文の主体を「委員会」と読むため、なお検出できない。
- **括弧の外の限定と、列挙の欠けも検査する**: 限定検査は、括弧書きに加えて、括弧の外にある限定の文言（「必要に応じ」「なるべく」「原則として」「できる限り」「やむを得ない」「著しく困難」「支障がないと認め」「〜の場合に限り」「を除き」）も対象にする。直前の読点からその語までの句（40字まで）が、回答行に文字2-gramで6割以上残っているかを確かめる。字数が多いか句点を含む見出し行（40字以上）は、再分割で本文が見出しに載ったものとして本文扱いにする。1行に並んだ「⑴ … ⑵ …」は項目ごとに分ける。連続する箇条・番号付きの行と、1行内の⑴⑵は同じ列挙として扱う（`_source_limits`の`group`）。回答行がその列挙の項目を1つでも述べていれば（2〜10項目の列挙に限る）、残りの項目も回答に必要とする（`_enumeration_findings`）。欠けた項目は、書き直しの指示に原文で示す。書き直し後も欠けていれば、その列挙を使った最後の行の後へ原文の項目を補う。補った件数は`final_answer.added_items`、各試行の欠けは`attempts[].enumeration_gaps`に出し、未確認事項には`enumeration_repaired`（停止理由には使わない）として示す。最終回答の指示文にも、括弧外の限定の文言と、列挙の他の項目を省略しないよう加えた。2026-09-24、別の質問（予定価格の積算省略、通知n-42）の3回中2回で、「必要に応じ」、類型⑴の「不可能又は著しく困難」、記１⑵・記２⑵が欠けた。検査はいずれも見落としていた（n-42の記１・記２は全文が見出し行で、原文行として扱われていなかった）。受領した3回の最終回答に適用すると、欠けた箇所をすべて報告し、合格の1回には何も報告しない。
- **重複した行を除く**: 原文による補完の後と1行ずつの照合の後に、先に出た行と同じ内容の行（根拠IDと区切り記号を除いて比較）を除く（`_drop_duplicate_lines`）。除いた件数は`final_answer.duplicate_lines`とWeb UIに出す。2026-09-24、予定価格の質問の1回で、類型⑴⑵を別々に言い換えた行がそれぞれ同じ原文に置き換わり、原文の行が2回ずつ出た。受領6件に適用すると、この1件の2行だけが除かれる。

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
    ├── deep_candidate_selection.py
    ├── deep_transport.py
    ├── deep_research.py
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

Embedding indexは検索要求から分離される。`index.bat build`またはWebのindex APIだけがwriterを開始し、検索はcache状態を読むだけである。通常のbuildはfile manifestを比較するincremental modeで、同一file hashかつ健全な全entryだけを再利用する。`--full`は完成cacheを再利用しない明示的復旧操作である。

```text
search.bat / GET /api/search
  → search.py / web_services.run_search
  → mode=answer: chat model検出 → index state read
  → mode=search: 決定的query → index state read（chat model/検索計画なし）
  → mode=deep: deep_research.run_deep_research → 候補上限・節読み・原文照合・統合
  → readyならskill-source scan + keyword/Embedding retrieval、未準備ならkeyword retrieval only
  → RRF merge
  → optional loopback reranker
  → mode=answer: Ollama回答生成 → answer + evidence + warnings
  → mode=search: evidence + done（回答生成なし）
  → mode=deep: deep_progress + evidence + answer + done/error（通常の最終根拠上限を再適用しない）
```

検索対象は`.md`, `.txt`, `.csv`, `.json`, `.yaml`, `.yml`, `.html`, `.htm`である。対応外の原本は外部で検索用派生物へ変換し、利用者が原本と照合する。

ただし`*.metadata.json`は検索対象から除外する。sidecarは原本の複製であり独立した資料ではないため、索引すると同一内容が二重に根拠提示され根拠precisionを下げる。除外は`EXCLUDED_SOURCE_FILE_PATTERNS`が担う。

### 3.1 根拠ステータス（evidence_status）

`_calculate_confidence`はquery語と根拠本文の一致率（coverage）、snippet充足率、上位RRFスコア、文書数からconfidenceを算出し、`sufficient` / `partial` / `insufficient`を返す。

- coverageは`coverage_terms()`が返す語で測る。日本語のように語間へ区切りを置かない言語では`_split_terms`だけでは query 全体が1語となり coverage が常に0になるため、CJKを含む語は2文字n-gramへ展開する。`must_find_terms`は検索計画が抽出した精密語であり展開せず逐語で評価する。
- Agentic-liteの検索計画は短いJSONを返す補助処理であり、Ollamaの`think`を明示的に`false`へ固定する。thinking対応モデルの既定推論で90秒timeoutを消費せず、回答生成のreasoning設定とは分離する。
- `sufficient`はEmbedding由来の根拠を伴う場合に限る。語の重なりだけでは、質問の限定条件（例:「海外」出張）を満たさない資料を十分な根拠として区別できない。keyword一致のみでsufficientを宣言すると根拠不足の警告が外れ、該当情報なしを維持できなくなる。
- 質問の内容語（漢字・カタカナ・英数字の2文字以上の連続。「何名」「何年」など「何」で始まる疑問詞と、「必要」「方法」「手続」など質問の型を表す一般語は除く）の半分以上が通常根拠（展開itemを除く）に無く、かつ通常根拠のEmbedding類似度の最大値が0.70未満なら、`sufficient`を`partial`へ下げる（`_lacks_query_content_support`、全経路）。coverage以外の項だけでconfidenceが閾値付近に達するため、資料に無い事項を近接語で問う質問（例: 記載の無い外部委託先への依頼手順）が`sufficient`になっていた。coverage単独・類似度単独の閾値では正例と分離できないため両方を組み合わせる。検索計画の必須語は資料に無い語を推測で含むことが多いため、判定には利用者の質問文の内容語を使う。Embedding類似度が得られない場合は判定しない。
- `sufficient`には原則として独立根拠2件以上を求めるが、Embedding類似度0.70以上かつ質問と4文字以上の具体語が一致する根拠は1件でも認める（`_has_strong_single_evidence`。規程の1条だけで答えが決まる質問のため）。
- この判定は語の有無で近接語を見分けるため、限界がある。質問の語が資料にそろっていても関係が異なる場合（例: 貸与期間の「延長」がある資料への試用期間の延長の質問）や、欠けた語が少数の場合（例: 電話番号だけが無い）は`sufficient`のままになり、最終判定は回答promptの根拠契約に委ねる。汎用語以外の言い換え語が本文に無い正例（例: 「担当人数」）を`partial`にすることもある。閾値と除外語は保留質問で検証した（`tests/eval/eval-spec-sufficiency-holdout.json`、TODO O2-17）。
- 該当情報なしは2層で担保する。corpusに一切記載がない場合はretrieval層が`insufficient`を返し、関連文書はあるが質問の条件を満たさない近接語の場合は`sufficient`を宣言しないことに留め、最終判定は`prompt_templates.py`の根拠契約が行う。
- retrieval候補は、質問との4文字以上の具体的な連続一致、または0.55以上のEmbedding類似度を根拠支持として要求する。質問条件が同一文中で明示的に否定される場合は候補制限前に矛盾を検出し、`sufficient`を禁止して回答層へ矛盾信号を渡す。照合に使う質問側の4文字は漢字・カタカナ・英数字だけの並びに限る（「の日当は」のように助詞をまたぐ並びは、別の条件を否定する「本規程の日当は支給しない」にも一致し、正例を`partial`にして回答が「該当情報なし」と答える原因になったため）。
- 回答promptは地域・対象・期間・役職・区分などの限定条件を個別照合し、条件が欠ける・対象が異なる・対象外である場合に、関連語や似た資料の数値を流用せず「該当情報なし」とする。評価ハーネスの`--measure-answer-layer`は回答本文を保存せず、この契約のphrase・禁止fact・transport errorだけをprobeする。

### 3.2 親子展開（見出しだけの親候補から配下本文への展開）

`source_structure.py`は`search.py`のチャンク分割（`_line_chunks`）とは独立した派生構造である。同一資料snapshotのbytesからATX見出しツリー（レベル、原文見出し、見出し行、親、節の終端、配下本文範囲）を構築し、ディスクへは保存しない。フェンス（バッククォート/チルダ）内は見出しと誤認せず、レベル飛びは直近の上位見出しを親とし、閉じ`#`列は除去する。Setext見出し・インデントコード・表・frontmatterは見出しとして解釈しない。

- 展開資格は、候補の`start_line`が親見出し行に一致し、かつ親見出し行の次行から最初の子見出しの直前までが空行だけの場合に限る（`find_expandable_parent`）。子を持たない空節は展開しない。
- 選択の単位は親の「直接の子見出し」（孫を含む自身の節全体、`direct_child_ranges`）である。行範囲上は連続していても選択候補としては別々に扱う。有効な`QueryContext`と`_EmbedIndex`が利用できれば子節配下チャンクの最大類似度降順（同点は既存keyword_score降順、最後に見出し出現順）、利用できない場合は見出し出現順を選択順位とする（`_order_ranges_for_selection`）。範囲数上限（`max_ranges_per_parent`）・文字予算・後段の件数枠は、いずれもこの選択順位の高い範囲から割り当てる（各itemの`group_priority`）。採用された子節に属する既存チャンクは行範囲が連続するものを1つの実在範囲へまとめ、表示は選択順序に関わらず常に原文順とする（`expand_parent_candidates` / `_expand_single_parent`）。各itemは`group_id`（資料hash・path・親見出し行から決定的に生成）、`expanded_from`（親のchunk_id）、`group_order`を持ち、離れた複数の子節を1つの広いstart/endに偽装しない。文字予算で切り詰める場合は行単位で切り、`end_line`を実際に渡した本文の最終行へ合わせる（1行も収まらない範囲は採用しない、`_truncate_line_aligned`）。最終プロンプトの根拠予算（`_fit_prompt_budget`）でも展開itemは同じ規則で切り詰める。範囲を省いた場合は`group_partial`と理由（`group_partial_reasons`: `range_limit`/`char_budget`/`uncovered_lines`/`match_limit`）を立てる。配下の非空行の連続範囲（`group_required_ranges`）を保持し、採用範囲がそれを覆わない場合（見出し判定の差で境界を跨ぐチャンクが除外された場合など）も部分展開とする。構造を読んだbytesのSHA-256と、親候補・子chunkの`file_sha256`が一致しなければ`source_changed`として展開しない。展開は展開元ごとに`cancel_check`を呼ぶ。
- query context補助API（`embedding_search_multi_with_context`）は`embedding_search_multi`と同じ内部実装（`_embedding_search_multi_impl`）を共有し、戻り値契約を変えず要求ローカルの`QueryContext`（重複排除済みqueries・取得済みvectors・モデル識別）も返す。`_build_chunk_id_lookup`で`_EmbedIndex.chunk_ids`から位置を引き、`_max_similarity_to_context`が対象子chunkのvector/normを閾値なしで採点する（次元不一致・ゼロnorm・欠損は`None`として原文順fallbackへ戻す）。contextは要求終了時に破棄し、ディスクや通常ログへ保存しない。`run_retrieval_pipeline`は各試行でその試行のqueriesから得た`QueryContext`だけを使う（前試行のcontextを持ち越さない）。
- `finalize_ranked_matches`は「支持判定 → 制限前矛盾検出 → file別上限 → 相対スコア足切り → 親子展開（`OFFLINE_AI_PARENT_CHILD_EXPANSION`環境変数、既定ON） → confidence算出」の順で処理する。直接ヒットと重なる展開範囲は重なる行だけを除き、残りを連続範囲ごとの展開itemとして残す。重なった直接ヒットには展開由来を`expanded_overlap_groups`として補足する（`_dedupe_expansion_against_direct_hits`）。`_calculate_confidence`は`group_id`単位で独立根拠を数える（子の増加を独立根拠の増加として数えない）。confidenceは採用本文全体で算出するが、通常経路の`sufficient`は展開itemを除いた通常根拠だけでも`sufficient`が成立する場合に限る（`_evaluate_evidence_status`）。展開groupは独立根拠の件数やcoverageを押し上げ得るため、完全構造根拠の条件を満たさないgroupを`sufficient`の成立根拠に数えない。通常根拠が`sufficient`なら展開groupの追加で`partial`へは落とさない。完全な構造根拠による`sufficient`の追加経路（`_has_complete_structural_evidence`）は、`must_find_terms`があり、groupが部分展開でなく、展開元の親が採用根拠にあって質問と語彙一致（lexical anchor）し、配下の非空行が採用根拠（展開itemと、切り詰められていない同一資料の直接ヒット）の行範囲で覆われ、必須語を採用本文で全て充足し、矛盾がなく、confidence 0.55以上の場合に限る。必須語を持たない経路（検索専用、keyword評価）では親の語彙一致と完全展開だけになるため、この経路を使わない（別資料の同名見出しの展開で`sufficient`を宣言した事例への対処）。
- `run_retrieval_pipeline`は展開前の候補（`ranked_candidates`）と展開後の採用根拠（`evidence_items`）を分離する。次試行のmergeと`_retry_queries`には展開前候補だけを渡し、展開後の根拠を検索候補へ書き戻さない。採用根拠は`select_final_evidence`で確定する。件数上限（`RETRIEVAL_PROMPT_MATCH_LIMIT`と8件の小さい方、`final_evidence_match_limit`）は`_allocate_evidence_slots`が配分し、各展開groupへ先に1枠、残枠を順位順（groupは展開元の親の直後）に割り当てる。通常根拠が残る間は展開範囲を`max(group数, 上限//2)`と`OFFLINE_AI_EXPANSION_MAX_TOTAL_RANGES`の小さい方までに抑え、範囲を省いたgroupは`group_partial`を立てる。confidence/evidence_statusは件数枠の配分後（回答経路では根拠予算の適用後にも）採用根拠だけで再判定し、試行の継続判断もこの状態で行う。制限前に検出した矛盾は維持する。
- 段階追跡: `finalize_ranked_matches` / `select_final_evidence`は`trace`引数に、候補（順位）、落選理由（`relevance_support`/`per_file_limit`/`relative_score_floor`/`overlaps_direct_hit`/`match_limit`/`char_budget`）、展開結果と部分展開の理由、切り詰め、最終採用根拠、最終statusを本文なしで記録する。`RetrievalResult.trace`は`{"attempts": [...], "final": {...}}`を保持し、評価harnessが候補段階のrecallと無関係な展開範囲の算出に使う。通常ログへは出力しない。
- 表示: 回答用prompt（`prompt_templates.build_user_prompt`）は展開itemに由来を、部分展開itemに「部分展開」を付け、部分展開がある場合は全体を網羅したと述べない回答契約を追加する。prompt shell見積り（`_estimate_prompt_shell_chars`）はこの表示分を件数上限ぶん固定で予約する。Web UIの根拠一覧・Markdown保存とCLIの根拠一覧は「見出し配下の本文を展開」「（部分展開）」を表示する。
- 既定ON（2026-09-14）。無効化は`OFFLINE_AI_PARENT_CHILD_EXPANSION=false`。上限は`OFFLINE_AI_EXPANSION_MAX_PARENTS`（既定2）、`OFFLINE_AI_EXPANSION_MAX_RANGES_PER_PARENT`（既定4）、`OFFLINE_AI_EXPANSION_MAX_TOTAL_RANGES`（既定4）、`OFFLINE_AI_EXPANSION_BUDGET_RATIO`（既定0.5、実効根拠予算に対する展開文字量の上限比率）で調整する。

### 3.3 根拠検証（案A）と回答promptの契約

`run_retrieval_pipeline`は、最終根拠予算を適用した後の採用根拠（回答promptに実際に載る根拠）について、検索側の状態（`retrieval_status`）が`sufficient`または`partial`なら、補助chatで「根拠が質問に答えているか」をJSONで判定する（`verify_evidence_support`、`OFFLINE_AI_EVIDENCE_VERIFY`、2026-09-20から既定ON。計画は`plans/offline-ai-evidence-sufficiency-verification-plan.md`）。

- 呼び出しは検索計画と同じ`num_ctx`・`num_batch`・`num_gpu`・`keep_alive`・`think=False`・temperature 0を使い、モデルの再ロードを起こさない。上限は`OFFLINE_AI_EVIDENCE_VERIFY_TIMEOUT`（既定15秒）と、全体の残り時間から回答生成の予約（`OFFLINE_AI_EVIDENCE_VERIFY_GENERATION_RESERVE`、既定30秒）を引いた値の小さい方。残りが`OFFLINE_AI_EVIDENCE_VERIFY_MIN_BUDGET`（既定3秒）未満なら実行せず`skipped_budget`とする。
- 出力契約: `support`（4値）・`reason_code`（5値）・`conditions`（`{condition, supported}`）。`conditions.supported`は「根拠がその条件に明示的に答えているか」であり、「対象に当てはまるか」ではない（「〜は対象外」と明記されていれば`true`）。validatorは不正値と矛盾する組み合わせ（`fully_supported`と`different_subject`、`unsupported`と`answer_found`、不支持条件があるのに`fully_supported`）を拒否する。
- 状態: `verified`・`failed`・`skipped_disabled`・`skipped_not_applicable`・`skipped_budget`。`failed`は理由を`EvidenceVerificationError.reason`（`transport`・`timeout`・`json_parse`・`schema`・`contradiction`）で分類し、`RetrievalResult.verification_failure_reason`とtraceへ残す。
- 格下げのみ: `retrieval_status=sufficient`で`verified`かつ`fully_supported`以外なら`partial`へ下げる。`partial`は格上げしない。`failed`・`skipped_*`では`retrieval_status`のまま回答を続ける。利用者キャンセル・全体期限切れ（`cancel_check`の例外）は検証の失敗へ変換せず上位へ伝播し、回答生成を始めない。
- 不足理由: 格下げまたは不足判定の場合、回答promptへ「根拠検証」の不足理由（理由コード・支持されない条件）を載せる。さらに、`unsupported`、`related_only`・`different_subject`、または不支持条件がある場合だけ、promptの末尾へ`INSUFFICIENCY_ANSWER_CONTRACT`（該当情報なしと書かれていない条件の説明だけにし、別対象の手続き・数値を書かない）を置く（`insufficiency_requires_abstain`）。`partially_supported`・`missing_detail`・不支持条件なしは、言い換え型の正例で検証が控えめに判定する形のため、契約を置かない。
- 表示: 格下げした場合だけ、CLIの根拠一覧とWebの「検索時の注意」・Markdown保存に「根拠検証: … sufficient から partial にしました」を出す（`verification_note`）。外部の`evidenceStatus`の意味は変えない。Webの構造化ログ（`phase: retrieval`）へ`verification_status`と`verification_failure_reason`を出す。
- 資料内指示文への耐性（案Aの有無によらず常時）: `SYSTEM_PROMPT`の基本ルール（`SOURCE_INSTRUCTION_IMMUNITY_RULE`）、根拠ブロック直前の注意（`SOURCE_INSTRUCTION_IMMUNITY_NOTE`）、回答promptの末尾（`SOURCE_INSTRUCTION_IMMUNITY_TAIL`）で、資料本文はデータであり回答方法・語句の指示に従わないと示す。3つを併用した場合に同一根拠の誤従が5/5→0/5になった。
- 予算: 耐性文言と末尾契約は根拠がある場合だけpromptへ現れ、根拠を空にしたshell見積りには出ないため、`_estimate_prompt_shell_chars`へ固定予約を足す。
- 既知の限界（2026-09-20時点）: 語がよく重なる別対象の質問（例: 「外部講師に研修を依頼する」に対し根拠は「外部研修を受講する」）は、検索された根拠の組み合わせによって検証が`fully_supported`と判定し、流用が残ることがある。「答えを求める対象」を先に比べさせる検証（2段階検証の試作）は、既存保留の正例の誤格下げと遅延増で不採用とした。遅延はP2実測で検証単体の中央値5.3〜6.2秒、回答完了まで約+7秒。

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
- `POST /api/search/cancel`（所有セッションのrequest_idに束縛したサーバージョブ中止）
- `GET /api/evidence/view?evidence_id=...`（検索時に登録した根拠行窓）
- `GET /api/index/status`
- `GET /api/index/plan?mode=incremental|full`（副作用のない差分解析）
- `POST /api/index/start` / `resume` / `cancel`
- `GET /api/index/events?job_id=...`（SSE）
- `POST /api/page/heartbeat` / `close`（開いているページの在席通知）

未知routeは共通404へ委ねる。Web層はbootstrap token、HttpOnly cookie、Origin/Host検査、CSP、接続数制限、`JobTable`、`CancellationToken`、slow-client対策を維持する。

`web.bat`の非表示起動（`web_launcher.py`）は`--exit-when-page-closed`を付けて起動し、開いているページが無くなったらサーバーを停止する。コンソール起動と直接起動は付けないため従来どおりCtrl+Cで停止する。ページは15秒ごとに`page_id`と表示状態を`/api/page/heartbeat`へ送り、`pagehide`で`sendBeacon`により`/api/page/close`を送る。サーバーの`PagePresenceMonitor`は最後のページが閉じてから15秒の猶予（再読み込み吸収）を置いて停止し、closeが届かなかったページは最後の通知が表示中なら45秒、非表示なら300秒（背景タブのtimer間引き対策）で失効させる。watchdogは2秒間隔で判定し、tick間隔が30秒を超えた場合はスリープ明けとして在席時刻を寄せる。ページが一度も接続しない場合は600秒で停止する。このプロセスのworkerが索引を構築中、または検索jobが実行中の間は停止を保留する。在席通知もCookie認証・Origin/Host検査の対象で、`page_id`は8〜64文字の英数字・`-`・`_`に限る。

Health contractにはindex statusと検索単位timeout契約（通常は`searchTimeoutDefault`/`Min`/`Max`、300〜600秒、deepは`deepSearchTimeoutDefault`/`Min`/`Max`、300〜1,800秒）、ページ閉鎖時停止の有効状態（`pageCloseStop`）を含める。index SSEの切断はjobをcancelせず、statusとbounded event bufferから再接続する。検索SSEの切断もジョブ中止とは区別し、明示中止だけが`/api/search/cancel`を通じて所有requestへ伝播する。

`/api/index/plan`は追加・変更・削除・未変更file数、生成/完成cache再利用/checkpoint再開のchunk数、mode、対象generation、全件計算になる理由（`full_requested`/`cache_missing`/`model_digest_unavailable`/`cache_incompatible`）、生成chunkだけを分母にした過去速度ベースの見積もりを返す。事前解析はbuildと同じ再利用選別関数を使い、cache・checkpoint・statusを書き換えない。速度は同一モデルのjobでEmbedding API呼出しに要した時間だけから算出し（`rate_basis: "generated"`）、走査・cache読込・JSON保存時間、再利用件数、旧processed基準の速度は混ぜない。32件未満しか生成しなかったjobはモデル起動待ちの影響が大きいため既存実績を上書きしない。生成0件（削除のみを含む）は0秒と断定せず、走査・保存時間がかかる旨を表示する。開始要求は任意の`expected_generation`を受け、確認後に資料が変わっていればjobを開始しない（`index_source_changed`）。start/resumeのmodeはstatusへ保存し、resumeは保存済みmodeを継承し、異なるmode指定は`index_mode_conflict`で拒否する。

cache互換性と資料鮮度は別判定である。互換性にはcache schema、モデルdigest、parser/chunk契約、ベクトル次元を含め、digestを取得できない場合は既存entryを再利用しない（毎回全件計算になり、事前解析は`model_digest_unavailable`を返す）。digest不明で構築したcacheは、現在もdigest不明の同名モデルに限ってreadyとし、digestが取得できるようになったらstaleとして再構築を求める。資料の最新generationが変わっても互換性が満たされれば未変更fileだけを候補へ引き継ぐ。再利用はfile単位のall-or-nothingで、path・file SHA-256・chunk ID列が一致し、そのfileの全entryが本文hash・有限値・次元の検査に合格した場合だけvectorを引き継ぐ（本文・見出し・行位置は現行parserの結果を正とする）。1件でも欠落・不正なら当該file全体を再計算する。追加・変更fileは全chunkを再計算し、削除fileのentryは候補から除外する。候補はmanifest、entry集合、本文hash、引用位置、有限値・次元を保存前に検証し、同一directoryの一時ファイルへ書き込んで再読込検証したうえで、昇格直前に資料の実hash・モデルdigest・cancelを再確認し、すべて成功した場合だけ原子的に置換する。停止・失敗・disk full・置換失敗では直前のcacheを変更せず一時ファイルを削除し、checkpointは保持する。staleならkeyword検索へ退避する。

資料の走査は、index writer（build/plan）では読取り・解析に失敗したfileが1件でもあれば`SOURCE_SNAPSHOT_FAILED`で更新を保留し、削除や空ファイルとして扱わない。検索read経路では読めないfileを飛ばして検索を継続するが、そのsnapshotは不完全としてreadyと判定しない。UTF-8として不正なbyteは検索・index共通でU+FFFDへ置換する。status/検索のsource memo keyは`(相対path, mtime, size, SHA-256)`で、同size・同mtimeの内容変更も検知する。2026-09-13に開発PC上の合成資料（482 file・26 MB、OS file cache warm）で測定したkey算出は中央値約72 ms（stat only約37 ms）であり、毎要求の全hashを維持する。cold cacheや低速diskでは増える可能性があり、実機では未測定である。検索read経路の世代確認（`_source_chunk_memo_key`）は、同サイズ・同mtimeの変更を見逃さないため要求ごとに全fileの実SHA-256を計算し、保証を弱めない。E版と同規模の合成資料（479 file・約17.9 MB、2026-09-14、実資料・実モデル不使用）では、hash走査が中央値約57 ms、要求ごとのchunk再構築（`build_source_chunks`）が約294 ms、keyword経路の検索専用処理全体が約413 msだった。作成直後のfileでは初回のみ約4.9 sの外れ値があった（ウイルス対策の走査等と推測、未確認）。親子展開が有効な検索専用要求では、chunk再構築のbytes読取りとは別にこのhash走査が1回走る。資料規模に比例する制約として扱う。

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

`runtime`/`both`分類のfile（`_internal/search.py`等）の中身を変更した場合、`_internal/download-manifest.json`の`appPayloadEstimate`（V-6ゲート、7.参照）が古いままだと`Test-DistributionPolicy.ps1`が`BLOCKED`になる。`download-manifest.json`自身も`both`分類でinventoryに含まれるため、単純に実測値をそのまま書き込むと自己参照でずれる（totalBytesの桁数が変わらなくても、書き込み時の改行コード混在等でmanifest自身のbyte数が変動し、actualが再びずれる）。更新手順: (1) `-Audience runtime`で`actual.totalBytes`を確認、(2) `other_total = actual.totalBytes - 現在のmanifestファイルのbyte数`を求める、(3) `target = other_total + 書き込み後のmanifestファイルのbyte数`（数値の桁数が変わらなければ現在のbyte数のまま）を`totalBytes`へ書く、(4) 再実行してPASSを確認する。`largestFileBytes`・`largestFile`はinventoryの実測値をそのまま使ってよい（自己参照しない）。`download-manifest.json`はリポジトリ上で改行コードがLFとCRLFの混在（`git ls-files --eol`で`i/mixed`）であり、属性は`-text`（変換なし）なので、テキストとして読み書きせず、バイト列のまま数値だけを置換する（Pythonの`read_text`/`write_text`はWindowsで全行をCRLFへ揃え、byte数と差分を変えてしまう）。

## 更新ポリシー

- `document_schema.py`、Markdown renderer、JobTable、CancellationToken、optional component共通guardを用途確認なしに削除しない。
- PowerShell module top-levelはfunction/class定義、`Set-StrictMode`、`Export-ModuleMember`、pure constant assignmentだけに限定する。
- package、installer、docs、testは同じ製品契約へ同期する。
- 変更後はdirect file comparison、hash、test、lint、buildを証跡として残す。
- 新しいfileを追加する場合は、`_internal/distribution-policy.json`へ配布分類のentryを同じ変更内で追加する。追加を怠ると、candidate検証・package生成がfail-closedで失敗する。

### 2026-09-23: 検索の切断とterminalの分離

- Web UIはSSEのEOF・通信失敗・AbortErrorだけで検索完了とは判断せず、sessionStorageの要求情報を保持する。画面は入力をロックし「再接続」を表示する。自動再試行ループは設けず、再読み込みまたは明示再接続で一度だけ再購読する。
- `resume_only=1` はJobTableのlock内で既存要求だけへ接続する。同一session/query/reasoning/timeout/modeのfingerprintを確認し、未知・TTL失効・再起動後のIDは404 `job_not_found`、不一致は409で拒否する。再接続で新しい要求・期限・モデル生成を作らない。
- done/errorのterminal、確定したHTTP 4xx拒否（408/429を除く）、サーバーの中止受付成功時にだけ復帰情報を破棄する。中止通信が失敗した場合は情報を保持する。旧購読や旧中止応答は次の検索を変更しない。
- storageを利用できない場合は警告を表示する。ページが存続する間はメモリ上の同一要求を再接続できるが、再読み込み復帰は保証しない。
- 自動テストによる制御検証と、通常Chromeでの実再読み込み・Markdownファイル実体の受入は別々に記録する。
