"""Conditions inside source tables must survive final-answer paraphrases."""

import pytest
import deep_research as deep


def evidence(text):
    return [{"evidence_id": "E1", "excerpt": text}]


@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("| 審査 | 必要に応じ、設備の検査を実施する。 |", "設備の検査を実施する。[E1]"),
        ("| 対象 | 申請書（臨時申請を除く）を保管する。 |", "申請書を保管する。[E1]"),
        (
            "| 方法 | 原則、担当者会議は対面形式で開催する。 |",
            "担当者会議は対面形式で開催する。[E1]",
        ),
        (
            "| 選任 | 安全性に配慮の上、検査担当者を第三者から選任する。 |",
            "検査担当者を第三者から選任する。[E1]",
        ),
    ],
)
def test_table_condition_omission_is_detected_and_repaired(source, answer):
    rows = evidence("| 項目 | 条件 |\n| --- | --- |\n" + source)
    findings = deep._limitation_findings(answer, rows)
    assert findings
    record = {"attempts": []}
    prepared, issues = deep._check_final_candidate(answer, rows, record)
    assert issues["limitation_lines"]
    assert record["attempts"][0]["limitation_lines"] == 1
    assert "原文:" in deep._final_retry_prompt("質問", prepared, issues)
    repaired, notes = deep._repair_limitations(answer, findings)
    assert notes
    assert deep._limitation_issues(repaired, rows) == []
    assert deep._validate_final_answer(repaired, {"E1"})
    assert source.split("|")[2].strip() in repaired


def test_faithful_table_condition_does_not_require_repair():
    source = "| 選任 | 安全性に配慮の上、検査担当者を第三者から選任する。 |"
    answer = "安全性に配慮の上、検査担当者を第三者から選任する。[E1]"
    assert deep._limitation_issues(answer, evidence(source)) == []


def test_condition_from_another_table_row_is_not_borrowed():
    source = (
        "| 検査 | 必要に応じ、設備の検査を実施する。 |\n| 保存 | 書類の保管期間は五年間とする。 |"
    )
    assert deep._limitation_issues("書類の保管期間は五年間とする。[E1]", evidence(source)) == []


def test_table_separator_is_not_a_source_claim():
    assert deep._source_limits(evidence("| --- | :---: |")) == []


def test_conditions_are_not_borrowed_from_neighbouring_cells():
    source = "| 必要に応じ、設備の検査を実施する。 | 書類の保管期間は五年間とする。 |"
    assert deep._limitation_issues("書類の保管期間は五年間とする。[E1]", evidence(source)) == []


def test_escaped_and_code_pipes_remain_in_their_source_cell():
    source = r"| 項目 | `a|b` と x\|y を必要に応じ処理する。 |"
    cells = deep._source_segments(source)
    assert cells == [("項目", False), (r"`a|b` と x\|y を必要に応じ処理する。", False)]
