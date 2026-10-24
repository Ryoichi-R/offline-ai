"""deep coverage evaluatorのfail-closed・根拠範囲・複数回集計テスト。"""

from pathlib import Path

import coverage_evaluator as evaluator


SPEC = Path(__file__).resolve().parent.parent / "eval" / "eval-spec-coverage-synthetic.json"


def _result(*, complete=True, bad_citation=False):
    evidence = [
        {
            "evidence_id": "E1",
            "path": "regulations/training-program-rules.md",
            "source_sha256": "47a04b63672b09d5a84f25ce4470bf3185bab0ca0806f1ebf24ab11bfb9a4156",
            "start_line": 5,
            "end_line": 7,
            "excerpt": "本制度は正社員を対象とする。非正規雇用者は対象外とする。",
            "conditions": [],
            "exceptions": [],
        },
        {
            "evidence_id": "E2",
            "path": "regulations/training-program-rules.md",
            "source_sha256": "47a04b63672b09d5a84f25ce4470bf3185bab0ca0806f1ebf24ab11bfb9a4156",
            "start_line": 17,
            "end_line": 19,
            "excerpt": "外部研修は事前に所属長の承認を得て受講申込書を提出する。会社負担を原則とするが、資格試験は個人負担。",
            "conditions": [],
            "exceptions": [],
        },
        {
            "evidence_id": "E3",
            "path": "regulations/training-program-rules.md",
            "source_sha256": "47a04b63672b09d5a84f25ce4470bf3185bab0ca0806f1ebf24ab11bfb9a4156",
            "start_line": 25,
            "end_line": 27,
            "excerpt": "人材開発課が窓口。指定様式を研修開始日の5営業日前までに提出する。",
            "conditions": [],
            "exceptions": [],
        },
    ]
    answer = (
        "正社員が対象で、非正規雇用者は対象外です [E1]。外部研修は会社負担を原則としますが、"
        "資格試験は個人負担で、所属長の承認と受講申込書が必要です [E2]。"
        "窓口は人材開発課で、指定様式を研修開始日の5営業日前までに提出します [E3]。"
    )
    if not complete:
        answer = answer.replace("5営業日前", "期限は未確認")
    if bad_citation:
        answer += " [E99]"
    for entry in evidence:
        lines = (SPEC.parent / "corpus" / entry["path"]).read_text(encoding="utf-8").splitlines()
        entry["excerpt"] = "\n".join(lines[entry["start_line"] - 1 : entry["end_line"]])
    return {"status": "completed", "answer": answer, "evidence": evidence}


def test_coverage_spec_and_complete_result_pass():
    report = evaluator.evaluate_result(evaluator.load_spec(SPEC), _result())

    assert report["status"] == "NEEDS_REVIEW"
    assert report["item_coverage"] == 1.0
    assert report["critical_misses"] == []


def test_missing_condition_fails_without_average_masking():
    report = evaluator.evaluate_result(evaluator.load_spec(SPEC), _result(complete=False))

    assert report["status"] == "FAIL"
    assert report["critical_misses"] == ["C03"]
    assert report["item_coverage"] < 1.0


def test_unknown_citation_is_reported_as_failure():
    report = evaluator.evaluate_result(evaluator.load_spec(SPEC), _result(bad_citation=True))

    assert report["status"] == "FAIL"
    assert report["unsupported_citations"] == ["E99"]


def test_aggregate_requires_every_run_to_pass():
    spec = evaluator.load_spec(SPEC)
    reports = [
        evaluator.evaluate_result(spec, _result()),
        evaluator.evaluate_result(spec, _result(complete=False)),
        evaluator.evaluate_result(spec, _result()),
    ]

    aggregate = evaluator.aggregate_reports(reports)

    assert aggregate["runs"] == 3
    assert aggregate["all_runs_passed"] is False
    assert aggregate["critical_misses"] == ["C03"]
    assert aggregate["item_covered_runs"]["C03"] == 2
