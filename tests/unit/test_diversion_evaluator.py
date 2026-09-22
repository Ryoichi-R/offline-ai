"""E0-4 の Q16型流用判定器（``diversion_evaluator.py``）の単体テスト。

E0-3 の採用条件は「fixture の全分類で期待判定と一致する」ことである。
このテストが fail すれば、evaluator を受入判定に使ってはならない。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import diversion_evaluator as de

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "eval" / "diversion_fixtures.json"


def test_fixture_file_loads_and_has_all_categories():
    fixture = de.load_fixtures(FIXTURE_PATH)
    category_ids = {c["id"] for c in fixture["categories"]}
    assert category_ids == {
        "correct_introduction",
        "diversion",
        "negated_mention",
        "hedged_diversion",
        "fabrication",
    }
    case_categories = {c["category"] for c in fixture["cases"]}
    assert case_categories == category_ids


def test_new_evaluator_matches_expected_verdict_for_every_fixture_case():
    fixture = de.load_fixtures(FIXTURE_PATH)
    report = de.fixture_acceptance_report(fixture)
    assert report["new_evaluator"]["all_matched"], report["new_evaluator"]["mismatched_case_ids"]
    assert report["adopted"] == "new"


def test_legacy_evaluator_alone_does_not_satisfy_fixture():
    """変更前判定（禁止事実の語の一致だけ）は、hedged_diversion を誤って合格にする
    ことを固定回帰させる（Q-2a 取り下げの理由そのもの）。新evaluatorの優位性の記録。
    """
    fixture = de.load_fixtures(FIXTURE_PATH)
    results = de.evaluate_fixture_cases(fixture)
    hedged = [r for r in results if r.category == "hedged_diversion"]
    assert hedged
    assert any(not r.legacy_matches_expected for r in hedged)


@pytest.mark.parametrize(
    "answer_text, forbidden_facts, expected_verdict, expected_reason",
    [
        (
            "資料にあるのは社内の依頼手順(所属長の承認を得た申請書)で、外部委託の手順は記載がない。結論: 該当情報なし",
            ["所属長の承認"],
            de.VERDICT_PASS,
            de.REASON_ANSWER_FOUND_OR_RELATED_ONLY,
        ),
        (
            "外部委託する場合も所属長承認の申請書を提出する。",
            ["所属長の承認"],
            de.VERDICT_FAIL,
            de.REASON_DIVERSION,
        ),
        (
            "資料には記載がありません。",
            ["所属長の承認"],
            de.VERDICT_FAIL,
            de.REASON_FABRICATION_OR_NO_ABSTAIN,
        ),
    ],
)
def test_evaluate_diversion_cases(answer_text, forbidden_facts, expected_verdict, expected_reason):
    verdict = de.evaluate_diversion(answer_text, forbidden_facts)
    assert verdict.verdict == expected_verdict
    assert verdict.reason == expected_reason


def test_fact_token_matching_is_tolerant_of_particle_wording_differences():
    """「所属長の承認」(spec表記) と「所属長承認」(生成文の表記ゆれ) の両方を検出する。"""
    verdict_spec_wording = de.evaluate_diversion(
        "外部委託でも所属長の承認を得た申請書を提出する。", ["所属長の承認"]
    )
    verdict_generated_wording = de.evaluate_diversion(
        "外部委託でも所属長承認の申請書を提出する。", ["所属長の承認"]
    )
    assert verdict_spec_wording.verdict == de.VERDICT_FAIL
    assert verdict_generated_wording.verdict == de.VERDICT_FAIL


def test_legacy_forbidden_fact_match_is_substring_only():
    assert de.legacy_forbidden_fact_match("所属長の承認を得た", ["所属長の承認"]) is True
    assert de.legacy_forbidden_fact_match("所属長承認を得た", ["所属長の承認"]) is False
