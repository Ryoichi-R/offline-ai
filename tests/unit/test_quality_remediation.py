from dataclasses import replace

import pytest

from test_eval_harness import (
    SPEC_PATH,
    _answerable_question,
    _minimal_spec,
    _outcome,
    harness,
)


@pytest.mark.parametrize(
    "answer,expected",
    [("1回のみ", True), ("一度だけ", True), ("2回のみ", False)],
)
def test_fact_aliases_are_explicit_and_preserve_numbers(answer, expected):
    assert harness._fact_present(answer, "一度だけ", ("1回のみ",)) is expected
    if answer == "1回のみ":
        assert not harness._fact_present(answer, "一度だけ")


def test_numeric_typography_is_not_missing_information():
    assert harness._fact_present("月額 1,800 円、12 か月", "1,800円")
    assert harness._fact_present("月額 1,800 円、12 か月", "12か月")
    assert not harness._fact_present("月額 1,900 円", "1,800円")
    assert not harness._fact_present("112か月", "12か月")
    assert not harness._fact_present("1回のみではありません", "一度だけ", ("1回のみ",))


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("料金は3,000円。a/b.md\n派遣職員の例外については該当情報なし。", True),
        ("料金は3,000円。a/b.md\n質問全体として該当情報なし。", False),
        ("3,000円 a/b.md\n該当情報なし。", False),
        ("派遣職員については該当情報なし。a/b.md", False),
        ("料金は2,500円または3,000円。a/b.md\n派遣職員については該当情報なし。", False),
    ],
)
def test_scoped_abstention_keeps_facts_and_forbidden_fact_gate(
    monkeypatch, answer, expected
):
    question = replace(_answerable_question(), allow_scoped_abstention=True)
    monkeypatch.setattr(harness.search, "stream_ollama_chat", lambda *a, **k: answer)
    result = harness.measure_answer_quality_probe(
        question, _outcome([]), chat_model="fake"
    )
    assert result["passed"] is expected


@pytest.mark.parametrize("aliases", [{"unknown": ["x"]}, {"x": []}, {"x": "y"}, []])
def test_invalid_alias_spec_fails_closed(aliases):
    data = _minimal_spec()
    data["questions"][0]["required_facts"] = ["x"]
    data["questions"][0]["required_fact_aliases"] = aliases
    with pytest.raises(harness.SpecError, match="required_fact_aliases"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_alias_and_scope_options_round_trip():
    data = _minimal_spec()
    data["questions"][0].update(
        required_facts=["一度だけ"],
        required_fact_aliases={"一度だけ": ["1回のみ"]},
        allow_scoped_abstention=True,
    )
    question = harness.parse_spec(data, spec_path=SPEC_PATH).questions[0]
    assert question.required_fact_aliases == (("一度だけ", ("1回のみ",)),)
    assert question.allow_scoped_abstention


def test_complete_irrelevant_scope_can_abstain_without_fabricated_answer(
    tmp_path, monkeypatch
):
    from test_macos_acceptance_fixes import _independent_backend
    from test_deep_remediation import research

    _independent_backend(monkeypatch, "irrelevant")
    result = research(tmp_path, "# 資料\n沿革の説明のみ。\n")
    assert result.status == "completed"
    assert result.answer_state == "not_generated"
    assert "根拠を確認できません" in result.answer
    assert result.evidence == []
    import web_services

    event = web_services.build_deep_evidence_event(result)
    assert event["answerState"] == "not_generated"
    assert (
        event["answerStateMessage"]
        == "資料の確認を完了しましたが、質問に答える根拠はありません"
    )
    assert (
        result.diagnostics["final_answer"]["result"] == "abstained_no_relevant_evidence"
    )


def test_unread_long_line_slice_cannot_be_declared_fully_reviewed():
    import deep_research as deep

    text = "文" * (deep.DEEP_MAX_UNIT_CHARS + 2)
    sources = {"source.md": (None, "hash", text, [text])}
    units = deep._split_line_range([text], 1, 1)
    rows = [dict(path="source.md", status="irrelevant", **unit) for unit in units]
    assert not deep._all_source_units_irrelevant(sources, rows[:1])
    assert deep._all_source_units_irrelevant(sources, rows)


def test_unread_document_cannot_be_declared_fully_reviewed():
    import deep_research as deep

    sources = {name: (None, "hash", "本文", ["本文"]) for name in ("a.md", "b.md")}
    unit = deep._split_line_range(["本文"], 1, 1)[0]
    rows = [dict(path="a.md", status="irrelevant", **unit)]
    assert not deep._all_source_units_irrelevant(sources, rows)


def test_changed_irrelevant_source_cannot_complete(tmp_path, monkeypatch):
    import deep_research as deep
    from test_macos_acceptance_fixes import _independent_backend
    from test_deep_remediation import research

    _independent_backend(monkeypatch, "irrelevant")
    original = deep._all_source_units_irrelevant

    def change_after_review(sources, ledger):
        complete = original(sources, ledger)
        if complete:
            (tmp_path / "source.md").write_text("変更後の本文", encoding="utf-8")
        return complete

    monkeypatch.setattr(deep, "_all_source_units_irrelevant", change_after_review)
    result = research(tmp_path, "# 資料\n沿革の説明のみ。\n")
    assert result.status == "failed"
    assert result.stop_reason == "source_changed"
    assert (
        result.diagnostics["final_answer"].get("result")
        != "abstained_no_relevant_evidence"
    )


@pytest.mark.parametrize(
    "gap,uncited,expected",
    [(False, False, 1.0), (True, False, 0.0), (False, True, 0.0)],
)
def test_split_cited_evidence_requires_complete_validated_union(gap, uncited, expected):
    import coverage_evaluator as evaluator
    from test_coverage_evaluator import SPEC, _result

    spec = evaluator.load_spec(SPEC)
    spec["items"] = spec["items"][:1]
    result = _result()
    original = result["evidence"][0]
    lines = (SPEC.parent / "corpus" / original["path"]).read_text().splitlines()
    first = dict(original, end_line=5, excerpt=lines[4])
    second_start = 7 if gap else 6
    second = dict(
        original,
        evidence_id="E4",
        start_line=second_start,
        excerpt="\n".join(lines[second_start - 1 : 7]),
    )
    result["evidence"] = [first, second]
    result["answer"] = "正社員が対象で非正規雇用者は対象外です [E1]。" + (
        "" if uncited else " [E4]"
    )
    report = evaluator.evaluate_result(spec, result)
    assert report["item_coverage"] == expected
    assert report["status"] != "PASS"  # human review remains required


def test_required_second_document_cannot_be_replaced_by_first_document():
    import coverage_evaluator as evaluator

    item = {
        "expected_sources": [
            {"path": "a.md", "line_start": 1, "line_end": 2},
            {"path": "b.md", "line_start": 1, "line_end": 2},
        ]
    }
    assert (
        evaluator._matching_evidence(
            item, [{"path": "a.md", "start_line": 1, "end_line": 2}]
        )
        == []
    )
