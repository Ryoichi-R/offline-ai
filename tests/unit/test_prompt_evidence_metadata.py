from prompt_templates import SYSTEM_PROMPT, build_user_prompt
from search import build_evidence_summary


def test_build_user_prompt_includes_evidence_metadata():
    prompt = build_user_prompt(
        "申請手順は?",
        [
            {
                "path": "docs/procedure.md",
                "snippet": "申請します。",
                "modifiedAt": "2026-07-06T00:00:00Z",
                "parser": "structured-fixture",
                "page": 4,
                "layout_type": "table",
                "source_title": "手順書",
                "start_line": 10,
                "end_line": 12,
            }
        ],
    )

    assert "parser: structured-fixture" in prompt
    assert "ページ: 4" in prompt
    assert "種別: table" in prompt
    assert "資料名: 手順書" in prompt


def test_cli_evidence_summary_includes_location_source_and_warning():
    summary = build_evidence_summary(
        [
            {
                "path": "docs/sample.md",
                "source_title": "Sample",
                "page": 4,
                "heading": "手順",
                "start_line": 10,
                "end_line": 14,
                "layout_type": "table",
                "parser": "structured",
                "source": "keyword+embedding+rerank",
                "parser_warnings": ["OCR confidence is low"],
            }
        ],
        evidence_status="sufficient",
        confidence=0.87,
    )

    assert "Sample" in summary
    assert "ページ 4" in summary
    assert "行 10-14" in summary
    assert "検索 keyword+embedding+rerank" in summary
    assert "OCR confidence is low" in summary
    assert "見出し配下の本文を展開" not in summary


def test_cli_evidence_summary_marks_expanded_and_partial_evidence():
    base = {"path": "docs/sample.md", "heading": "手順", "start_line": 3, "end_line": 9}
    summary = build_evidence_summary(
        [
            {**base, "source": "expanded", "group_id": "g1", "group_partial": False},
            {**base, "source": "expanded", "group_id": "g2", "group_partial": True},
        ],
        evidence_status="partial",
        confidence=0.5,
    )

    lines = summary.splitlines()
    assert lines[1].endswith("見出し配下の本文を展開")
    assert lines[2].endswith("見出し配下の本文を展開（部分展開）")


def test_prompt_contract_checks_all_query_constraints_before_reusing_values():
    prompt = build_user_prompt(
        "海外出張の日当はいくらですか",
        [{"path": "regulations/travel.md", "snippet": "国内出張の日当は2,500円。"}],
        evidence_status="sufficient",
        confidence=0.7,
    )

    assert "限定条件" in SYSTEM_PROMPT
    assert "数値を流用せず「該当情報なし」" in prompt


def test_prompt_forbids_values_from_explicitly_out_of_scope_evidence():
    prompt = build_user_prompt(
        "海外出張の日当はいくらですか",
        [
            {
                "path": "rules/travel.md",
                "snippet": "海外出張は対象外。国内出張の日当は2,500円。",
                "constraint_conflict": True,
            }
        ],
        evidence_status="partial",
    )

    assert "限定条件の明示的な矛盾: あり" in prompt
    assert "数値・期限・割合を引用、列挙、比較しない" in prompt
