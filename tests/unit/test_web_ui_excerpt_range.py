"""根拠の抜粋が範囲の一部のとき、画面と保存Markdownに抜粋自体の行範囲と省略を示す。"""

import json

from test_web_ui_answer_quality import INDEX_HTML, run_node


def test_saved_markdown_and_screen_label_partial_excerpts():
    items = [
        {"path": "notice.md", "sourceTitle": "notice.md", "startLine": 6, "endLine": 9,
         "snippet": "抜粋", "sourceSha256": "a" * 64,
         "snippetStartLine": 8, "snippetEndLine": 8, "snippetTruncated": True},
        {"path": "notice.md", "sourceTitle": "notice.md", "startLine": 1, "endLine": 3,
         "snippet": "全体", "sourceSha256": "a" * 64,
         "snippetStartLine": 1, "snippetEndLine": 3, "snippetTruncated": False},
        {"path": "other.md", "sourceTitle": "other.md", "startLine": 1, "endLine": 9,
         "snippet": "検索時", "sourceSha256": "b" * 64, "snippetTruncated": True},
    ]
    result = run_node(
        f"""
const items = {json.dumps(items, ensure_ascii=False)};
state.resultMeta = {{mode: 'deep', startedAt: '2026-11-08T00:00:00.000Z', requestId: 'r1'}};
state.resultEvidence = {{items, evidenceStatus: 'partial', route: 'deep', indexState: 'deep', warnings: [],
    answerState: 'generated', deepLedger: []}};
state.resultReceivedAt = '2026-11-08T00:05:00.000Z';
state.activeQuery = '質問';
state.rawSearchText = '回答';
ui.saveMarkdown();
out({{labels: items.map((item) => ui._excerptRangeLabel(item)), markdown: saved[0]}});
"""
    )
    assert result["labels"] == [
        "抜粋 行 8-8（根拠範囲の一部。省略あり）",
        "",
        "抜粋は根拠範囲の一部（省略あり）",
    ]
    markdown = result["markdown"].replace("\\", "")
    assert "- 抜粋の範囲: 抜粋 行 8-8（根拠範囲の一部。省略あり）" in markdown
    assert markdown.count("- 抜粋の範囲:") == 2


def test_evidence_list_shows_the_excerpt_label():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "const excerptLabel = this._excerptRangeLabel(item);" in html


def test_segment_ranges_are_listed_in_the_label():
    item = {"snippetTruncated": True, "snippetStartLine": 30, "snippetEndLine": 58,
            "snippetSegments": [{"startLine": 30, "endLine": 38, "text": "a"}, {"startLine": 53, "endLine": 58, "text": "b"}]}
    result = run_node(f"out(ui._excerptRangeLabel({json.dumps(item)}));")
    assert result == "抜粋 行 30-38、53-58（根拠範囲の一部。区間の間と前後を省略）"
