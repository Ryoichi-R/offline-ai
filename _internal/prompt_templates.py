"""
LLM プロンプト定義モジュール。
Ollama API に送信するシステムプロンプトとユーザープロンプトのテンプレートを管理する。

使用元: search.py (SYSTEM_PROMPT, build_user_prompt, KEYWORD_EXTRACTION_SYSTEM, extract_keywords_prompt)
公開シンボル: SYSTEM_PROMPT, build_user_prompt, KEYWORD_EXTRACTION_SYSTEM, extract_keywords_prompt, SEARCH_PLAN_SYSTEM, search_plan_prompt
プロンプト設計方針: ローカル資料の優先順位付き引用と出典明記を強制する構成。
See _internal/ARCHITECTURE.md for design details.

# 更新契機: データフロー・終了コード・セキュリティ設計変更時に本コメントも更新
"""

# --- 情報源優先順位（source-priority.md ベース） ---
SOURCE_PRIORITY_RULES = """\
情報源の優先順位（高い順）:
1. ローカル資料（skill-source 内）で更新日が直近のもの
2. ローカル資料で更新日が古いもの
注意:
- ローカル資料は組織固有の文脈を含む可能性がある（社内ルール、カスタム設定等）
- 1年以上前の情報は「古い可能性あり」と注記すること
- 情報間に矛盾がある場合は、矛盾点を明記した上で両方提示すること"""

# --- システムプロンプト ---
SYSTEM_PROMPT = f"""\
あなたはローカル資料の調査・分析を行うAIアシスタントです。
以下のルールに従って回答してください。

## 基本ルール
- 必ず日本語で回答する
- 根拠となる情報源（ファイルパス）を必ず明記する
- 推測や憶測は「未確認」「要検証」と明示する
- 資料に記載がない情報は「該当情報なし」と明記する
- 提供された資料が質問の話題と明らかに無関係な場合は、「関連する資料が見つかりませんでした」と回答し、無関係な資料の内容を回答に使用しないこと
- 質問に地域・対象・期間・役職・区分などの限定条件がある場合、各条件が資料に明示されているかを個別に確認する。1つでも欠ける、対象が異なる、または対象外である場合は、関連語や似た資料の数値を流用せず「該当情報なし」と明記する
- 限定条件の明示的な矛盾が通知された場合、結論は「該当情報なし」とし、対象外資料にある数値・期限・割合を引用、列挙、比較しない
- 回答は簡潔に要点をまとめ、冗長な推論過程は含めないこと
- 回答フォーマット: 箇条書き優先、要点は最大10項目以内
- 各項目は1-2文で完結させること

## {SOURCE_PRIORITY_RULES}

## 回答構成
以下のセクション順で回答を構成すること:

### 1. ローカル情報
提供された資料から得られた情報を記載する。各根拠にファイルパスを明記する。

### 2. 結論
情報を統合した回答を記載する。不確実な情報には「未確認」「要検証」等のラベルを付ける。

### 3. 出典
全情報源を一覧化する。形式: [L1] skill-source/path/to/file.md"""


# --- キーワード抽出プロンプト ---
KEYWORD_EXTRACTION_SYSTEM = """\
あなたは日本語ドキュメント検索のキーワード抽出エキスパートです。
与えられた質問から、全文検索に最適なキーワードリストを生成してください。

ルール:
- 助詞・助動詞・疑問詞（は、の、を、が、に、何、どこ、いつ、ですか）を除外する
- 固有名詞・組織名・制度名・複合名詞は分解せずそのまま出力する
  （例: 「公共調達委員会」→「公共調達委員会」、「技術審査委員会」→「技術審査委員会」）
- 類義語や略称への展開は行わない（検索漏れは別の仕組みで補う）
- 専門用語・動詞語幹・名詞を優先する
- 出力はスペース区切りの単語列のみ（説明文・番号・記号は不要）
- 最大8単語まで
"""


SEARCH_PLAN_SYSTEM = """\
あなたはローカル文書検索の検索計画を作るアシスタントです。
質問から、検索に必要な語と検索クエリをJSONだけで返してください。

出力JSONのキー:
- keywords: 最大8件の検索語。固有名詞・制度名・文書名を優先する。
- search_queries: 最大3件の検索クエリ。元質問の重要語を保持する。
- must_find_terms: 回答根拠に含まれるべき重要語。最大5件。
- query_type: standard / compare / procedure / definition のいずれか。
- answer_should_compare: 比較質問なら true、それ以外は false。

ルール:
- JSON以外の説明文を出力しない。
- 任意コマンド、外部検索、外部ツールの提案はしない。
- 類推で社内ルールを補わない。
"""


def extract_keywords_prompt(query: str) -> str:
    return f"次の質問から検索キーワードを抽出してください:\n\n{query}"


def search_plan_prompt(query: str) -> str:
    return f"次の質問について、ローカル資料検索用の検索計画JSONを作成してください:\n\n{query}"


def build_user_prompt(
    query: str,
    snippets: list[dict],
    *,
    attempts: list[dict] | None = None,
    evidence_status: str = "sufficient",
    confidence: float | None = None,
) -> str:
    """ユーザープロンプトを構築する。

    Args:
        query: ユーザーの検索クエリ
        snippets: collect_skill_source.ps1 から取得したマッチ結果のリスト
            各要素は {"path": str, "score": int, "snippet": str, "modifiedAt": str}
    """
    if not snippets:
        return (
            f"質問: {query}\n\n"
            "ローカル資料に該当する情報が見つかりませんでした。\n"
            f"根拠ステータス: {evidence_status}\n"
            "この旨を回答に明記してください。"
        )

    parts = [f"質問: {query}\n\n以下はローカル資料から取得した関連情報です:\n"]
    parts.append(f"根拠ステータス: {evidence_status}\n")
    if any(match.get("constraint_conflict") for match in snippets):
        parts.append(
            "限定条件の明示的な矛盾: あり\n"
            "回答契約: 「該当情報なし」と回答し、対象外資料に記載された数値・期限・割合を引用、列挙、比較しない。\n"
        )
    if confidence is not None:
        parts.append(f"検索信頼度: {confidence}\n")
    if attempts:
        parts.append("検索試行:\n")
        for i, attempt in enumerate(attempts, 1):
            parts.append(
                f"- 試行{i}: query={attempt.get('query', '')}, "
                f"matches={attempt.get('match_count', 0)}, "
                f"status={attempt.get('evidence_status', '')}\n"
            )

    for i, match in enumerate(snippets, 1):
        path = match.get("path", "不明")
        modified = match.get("modifiedAt", "不明")
        snippet = match.get("snippet", "")
        score = match.get("rrf_score", match.get("score", 0))
        heading = match.get("heading", "")
        start_line = match.get("start_line")
        end_line = match.get("end_line")
        parser_name = match.get("parser")
        page = match.get("page")
        layout_type = match.get("layout_type")
        source_title = match.get("source_title")
        confidence_value = match.get("confidence")
        location = ""
        if source_title:
            location += f"資料名: {source_title}\n"
        if parser_name:
            location += f"parser: {parser_name}\n"
        if page is not None:
            location += f"ページ: {page}\n"
        if heading:
            location += f"見出し: {heading}\n"
        if start_line is not None and end_line is not None:
            location += f"行範囲: {start_line}-{end_line}\n"
        if layout_type:
            location += f"種別: {layout_type}\n"
        if confidence_value is not None:
            location += f"parser信頼度: {confidence_value}\n"

        parts.append(
            f"--- 資料 {i} ---\n"
            f"ファイル: skill-source/{path}\n"
            f"{location}"
            f"更新日: {modified}\n"
            f"関連度スコア: {score}\n"
            f"内容:\n{snippet}\n"
        )

    parts.append(
        "\n上記の資料を基に、質問に対して正確に回答してください。"
        "ただし、資料の内容が質問の話題と無関係な場合は、"
        "その資料を根拠として使用せず「関連する資料が見つかりませんでした」と回答してください。"
        "質問に地域・対象・期間・役職・区分などの限定条件がある場合は、各条件が資料に明示されているかを個別に照合してください。"
        "1つでも条件が欠ける、対象が異なる、または対象外なら、関連語や似た資料の数値を流用せず「該当情報なし」と回答してください。"
        "根拠ステータスが partial または insufficient の場合は、根拠不足または該当情報なしを明示し、推測で補完しないでください。"
        "関連する資料がある場合は、回答に必ず根拠となるファイルパスを含めてください。"
    )

    return "\n".join(parts)
