# Phase 2 検索品質・根拠品質 評価ハーネス

検索品質・根拠品質の受入を、固定 corpus と固定質問で再現可能に測るための評価ハーネスである。独立したrepository内で実行でき、親workspaceの計画ファイルには依存しない。

## 何を測るか

回答の流暢さではなく、**根拠の正しさ**を測る。指標は次のとおり分離する。

| 指標                                   | 定義                                                                             |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `retrieval_hit_rate`                   | 期待 source を1件以上返せた質問の割合                                            |
| `hit_at_1_rate`                        | 1位の根拠が期待 source だった質問の割合                                          |
| `evidence_coverage`                    | 期待 source のうち実際に返せた割合（複数文書比較で 0.5 等になる）                |
| `evidence_precision`                   | 返した根拠のうち期待 source だった割合（無関係な根拠の混入量）                   |
| `evidence_line_overlap`                | 返した根拠の行範囲が期待範囲と重なった割合（同一文書内の別条文を掴んでいないか） |
| `expected_lines_retained`              | `require_expected_lines: true` の質問だけ。期待範囲が最終根拠に残り、`required_facts` が重なる抜粋に含まれるか |
| `candidate_line_recall`                | 期待範囲のうち、支持判定・件数制限の前の検索候補（段階追跡）に入っていた割合。候補に入らない失敗と、候補にはあるが選別で落ちる失敗を分ける |
| `expanded_irrelevant_ranges`           | 期待範囲と重ならない展開item（無関係な子）の採用数。受入閾値にはせず報告する |
| `expansion_expectation_met`            | `expected_expansion` の質問だけ。`partial` は部分展開itemを採用し sufficient を宣言しないこと、`complete` は展開itemがどれも部分展開でないこと |
| `holdout_failed`                       | `holdout: true`（上限・重みの調整に使わない保留質問）のうち FAIL した件数 |
| `forbidden_source_hits`                | 根拠として返してはならない file を返した件数                                     |
| `abstain_accuracy`                     | 「該当情報なし」が正解の質問を正しく扱えた割合（判定層は下記）                   |
| `latency_ms_median` / `latency_ms_max` | 検索の応答時間                                                                   |

## 「該当情報なし」の判定層

`answerable: false` の質問は `abstain_layer` で2種類に分ける。混同すると製品側を誤った方向へ変更することになる。

| `abstain_layer`     | 対象                                             | retrieval 層の要件               | 確定判定                                                                                         |
| ------------------- | ------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| `retrieval`（既定） | corpus に一切記載がない                          | `evidence_status = insufficient` | このハーネスで完結                                                                               |
| `answer`            | 関連文書はあるが質問の限定条件を満たさない近接語 | `sufficient` を宣言しないこと    | 生成プロンプト契約（`prompt_templates.py`）側。`--measure-answer-layer`未指定では `NOT_MEASURED` |

近接語（例: 国内出張規程しかない corpus への「海外出張の日当」）は、語の重なりでは正解質問と分離できない。retrieval 層へ `insufficient` を強制すると正解質問まで巻き添えになるため、要件を「`sufficient` を宣言しない」に限定している。

候補の後処理では、質問との4文字以上の具体的な連続一致、または0.55以上のEmbedding類似度を根拠支持として要求する。弱い意味近接だけの候補を除外し、質問条件が同一文中で「対象外」「適用しない」等と明示された場合は`sufficient`を禁止する。矛盾信号は回答プロンプトにも渡し、対象外資料の数値・期限・割合を引用しない。

## 経路（route）

| route          | 内容                                                                    | 前提                                                          |
| -------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| `keyword`      | チャンク keyword 検索のみ                                               | なし（完全に決定的、Ollama不要）                              |
| `hybrid`       | keyword + Embedding + RRF 統合                                          | Embedding model（既定 `bge-m3`）が loopback Ollama で利用可能 |
| `agentic-lite` | 既存 `run_retrieval_pipeline` 相当。search plan と bounded 再検索を含む | chat model が loopback Ollama で利用可能                      |

model が利用できない経路は `skipped` として記録し、**PASS へ数えない**。受入判定も `NOT_MEASURED` になる。

## 実行

```bash
python tests/eval/run_eval.py --no-ollama              # keyword のみ
python tests/eval/run_eval.py --routes all             # 全経路（Ollama必要）
python tests/eval/run_eval.py --routes all --repeat 3  # 非決定性の確認
```

主なオプション:

- `--routes keyword,hybrid,agentic-lite` または `--routes all`
- `--repeat N` — 各 route を N 回実行する。集計は **最良値ではなく中央値**を代表値とし、`stability` に min/median/max を残す
- `--chat-model` / `--embed-model` — `_internal/.model` / `.model_embed` の検出結果を上書きする
- `--out-dir` — receipt 出力先（既定 `<repository>/.test-results/offline-ai-eval/`）
- `--no-write` — 標準出力のみ
- `--measure-answer-layer` — `abstain_layer=answer` の質問を各指定 route で1回生成し、回答本文を保存せず、abstain phrase・禁止 fact・transport errorだけをreceiptへ記録する。生成モデルが必要。評価 corpus で親子展開を行う（製品 `skill-source` を読まない）。
- `--measure-answer-quality` — 回答可能で `required_facts` を持つ質問を各指定 route で1回生成し、回答本文を保存せず、空回答でない・`required_facts` を全て含む・`forbidden_facts` を含まない・期待 source の path を引用する・該当情報なしと答えない・transport error でない、をreceiptへ記録する。1件でも測定済みの不合格があれば route 判定は FAIL、未測定が残れば NOT_MEASURED。機械的な信号であり意味内容の正しさの証明ではない。生成モデルが必要。
- `--compare-expansion` — 親子展開 OFF/ON の両方で評価し（本評価と逆の設定を1回追加実行）、質問ごとの evidence_status 遷移（特に sufficient→partial の低下）、合否の変化、再検索発生率・試行回数を receipt の `expansion_comparison` に残す。本評価の展開設定は `environment.parent_child_expansion` に記録する。
- hybridまたはagentic-liteを指定した場合、評価用Embedding cacheをメモリ内で構築する。製品`embed_cache.json`は読み書きしない。

終了コードは受入判定が `PASS` のとき 0、それ以外は 1（仕様不備は 2）。

## corpus

`corpus/` は合成した非機密 fixture であり、利用者資料を一切含まない。マスタープランが要求する構造をひととおり含む。

| file                                          | 目的                                                     |
| --------------------------------------------- | -------------------------------------------------------- |
| `regulations/travel-expense-rules.md`         | 規程型、同一文書内に2つの表（日当 / 宿泊費）             |
| `regulations/equipment-loan-rules.md`         | 規程型、質問語「貸出」に対し本文表記は「貸与」（類似語） |
| `regulations/equipment-return-rules.md`       | 貸与規程との複数文書比較の相手                           |
| `guides/onboarding-handbook.md`               | 長文（15見出し）、metadata sidecar あり                  |
| `guides/onboarding-handbook.md.metadata.json` | sidecar。根拠として返ってはならない                      |
| `notes/support-meeting-notes.txt`             | sidecar なしのプレーンテキスト                           |
| `data/office-equipment-inventory.csv`         | CSV の表。拠点列を無視すると誤答する                     |
| `guides/incident-response-procedure.md`       | 親子展開シナリオ。見出しだけの親チャンクがヒットしても配下本文がkeyword候補に入らない節（Q11）と、質問語彙が本文と直接重なり単独ヒットする対照節（Q12/Q13）を両方含む |
| `guides/data-restore-procedure.md`            | 親子展開の保留シナリオ。障害対応手順書と同名の見出し「第2章 復旧対応」（Q14）、子節が5つあり範囲上限で部分展開になる親（Q15）、記載のない外部委託の近接語（Q16） |

## 評価仕様

`eval-spec.json` が質問と受入基準を保持する。仕様は fail-closed で検証され、次を満たさない場合は実行しない。

- `answerable: true` の質問は `expected_sources` 必須
- `answerable: false`（該当情報なし）の質問へ `expected_sources` は指定不可
- 質問 ID の重複、行範囲の逆転、未知の `acceptance` キーは拒否
- `require_expected_lines`（任意、真偽値）は行範囲付きの `expected_sources` を持つ質問にだけ指定可
- `expected_expansion`（任意、`complete` / `partial`）は answerable な質問にだけ指定可。不一致は質問単位で FAIL、route 判定の `expansion_expectation_met` も FAIL
- `holdout`（任意、真偽値）。保留質問の FAIL は `holdout_failed` として別集計する（repeat 中に1回でも FAIL すれば代表値に残す）

`required_facts` / `forbidden_facts` は回答生成を伴う評価のために保持している項目であり、現行の retrieval 評価では `require_expected_lines: true` の質問で最終抜粋への残存確認に `required_facts` を使う以外は採点に使用しない。回答本文の意味単位の採点は、Phase 2 の手動受入で判定する。

## receipt の秘密情報の扱い

receipt には次を含めない。

- 資料本文・snippet（`redact_match` が `path` / `heading` / 行範囲 / スコアだけを残す）
- 生成した回答本文
- 絶対 path、ユーザー名

corpus は fixture なので file 単位の SHA-256 は記録する。これにより receipt を corpus と製品 digest（`search.py` の SHA-256、`VERSION`、retrieval 設定値）へ束縛できる。

## 親子展開（`OFFLINE_AI_PARENT_CHILD_EXPANSION`）

既定ON（2026-09-14、E:実機でのP2比較受入結果を踏まえ利用者判断で切替。`result/offline-ai/parent-child-expansion-p2-20260913/`）。`OFFLINE_AI_PARENT_CHILD_EXPANSION=false`で無効化できる。有効時は`keyword`/`hybrid`/`agentic-lite`いずれのrouteも`finalize_ranked_matches`へcorpusのchunksと`corpus_dir`（`source_root`）を渡し、見出しだけの親candidateから配下本文への展開を行う。評価は製品`skill-source`を一切読まない契約を保つため、展開有効時は必ず固定`corpus_dir`を`source_root`として渡す（`_expansion_kwargs`）。`agentic-lite`routeでは、展開が資料の実bytesを読み直す都合上、`search.SKILL_SOURCE_DIR`自体を一時的に`corpus_dir`へ差し替える（`_isolated_agentic_inputs`）。

Q11（親のみ検索に当たり、配下本文はkeyword候補にすら入らない）はOFF時に`evidence_line_overlap = 0`（既知の取得漏れの再現）、ON時に`1.0`（展開による解消）を示す。Q11は`require_expected_lines: true`を持ち、1.1の行範囲と`required_facts`が最終根拠（件数枠の配分後の抜粋）に残らなければ質問単位でFAILとし、route受入判定にも`expected_lines_retained`（欠落0件、repeat中に1回でも欠落すればFAIL）として反映する。平均の`evidence_line_overlap`が閾値を上回っても、この欠落は相殺しない。`keyword`/`hybrid` routeは製品pipelineと同じ`select_final_evidence`で件数枠の配分と状態の再判定を行う。

Q14〜Q16は、上限・重みの調整に使っていない保留質問（`holdout: true`）である。Q14は別資料の同名見出しで、障害対応手順書側の「第2章 復旧対応」の展開が`expanded_irrelevant_ranges`に1件として現れる（keyword route、2026-09-14）。必須語を持たない経路では、この展開を理由に sufficient を宣言しない。Q15は`expected_expansion: partial`で、部分展開の範囲（1.1）と必要事項が最終根拠に残り、sufficient を宣言しないことを確認する。Q16は該当なしの近接語で、関連資料が当たっても sufficient を宣言しないことを確認する。keyword route の`candidate_line_recall`はQ11・Q15で0（候補に入らず、親子展開でだけ最終根拠に届く）である。Q12/Q13は質問語彙が本文と直接重なりkeywordで単独ヒットする対照ケースで、ON/OFFいずれも`1.0`を維持する（展開が既存の直接ヒット経路を壊さないことの確認）。

## ハーネス自体の検証

`tests/unit/test_eval_harness.py` が、仕様検証の fail-closed、採点、集計、redaction、skip の扱い、keyword route の決定性を固定する。ハーネスを変更する場合はこのテストを先に更新すること。
