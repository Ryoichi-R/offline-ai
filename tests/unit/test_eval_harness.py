"""Phase 2 評価ハーネス自体の単体テスト。

ハーネスが誤った指標を出すと Phase 2 の受入判断そのものが壊れるため、
仕様検証（fail-closed）、採点、集計、redaction、skip の扱いを固定する。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import eval_harness as harness

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
SPEC_PATH = EVAL_DIR / "eval-spec.json"


# ---------------------------------------------------------------------------
# 仕様の検証
# ---------------------------------------------------------------------------


def _minimal_spec() -> dict:
    return {
        "schema_version": "1.0",
        "corpus_dir": "corpus",
        "acceptance": {"min_retrieval_hit_rate": 0.9},
        "questions": [
            {
                "id": "Q01",
                "category": "regulation",
                "query": "日当はいくらか",
                "answerable": True,
                "expected_sources": [{"path": "a/b.md", "line_start": 1, "line_end": 5}],
            }
        ],
    }


def test_parse_spec_accepts_minimal_spec():
    spec = harness.parse_spec(_minimal_spec(), spec_path=SPEC_PATH)
    assert spec.schema_version == "1.0"
    assert len(spec.questions) == 1
    assert spec.questions[0].expected_sources[0].path == "a/b.md"


@pytest.mark.parametrize(
    "mutate, message_fragment",
    [
        (lambda d: d.update(schema_version="9.9"), "schema_version"),
        (lambda d: d.update(corpus_dir=""), "corpus_dir"),
        (lambda d: d.update(acceptance={}), "acceptance"),
        (lambda d: d.update(acceptance={"unknown_key": 1}), "未知のキー"),
        (lambda d: d.update(acceptance={"min_retrieval_hit_rate": "high"}), "数値"),
        (lambda d: d.update(questions=[]), "questions"),
    ],
)
def test_parse_spec_rejects_invalid_spec(mutate, message_fragment):
    data = _minimal_spec()
    mutate(data)
    with pytest.raises(harness.SpecError) as excinfo:
        harness.parse_spec(data, spec_path=SPEC_PATH)
    assert message_fragment in str(excinfo.value)


def test_parse_spec_rejects_duplicate_question_id():
    data = _minimal_spec()
    data["questions"].append(dict(data["questions"][0]))
    with pytest.raises(harness.SpecError, match="重複"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_requires_expected_sources_for_answerable_question():
    data = _minimal_spec()
    data["questions"][0].pop("expected_sources")
    with pytest.raises(harness.SpecError, match="expected_sources"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_rejects_expected_sources_for_no_answer_question():
    data = _minimal_spec()
    data["questions"][0]["answerable"] = False
    with pytest.raises(harness.SpecError, match="該当情報なし"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_rejects_reversed_line_range():
    data = _minimal_spec()
    data["questions"][0]["expected_sources"][0] = {
        "path": "a/b.md",
        "line_start": 9,
        "line_end": 2,
    }
    with pytest.raises(harness.SpecError, match="line 範囲"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_rejects_non_bool_require_expected_lines():
    data = _minimal_spec()
    data["questions"][0]["require_expected_lines"] = "yes"
    with pytest.raises(harness.SpecError, match="require_expected_lines"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_requires_line_range_for_require_expected_lines():
    data = _minimal_spec()
    data["questions"][0]["expected_sources"][0] = {"path": "a/b.md"}
    data["questions"][0]["require_expected_lines"] = True
    with pytest.raises(harness.SpecError, match="行範囲付き"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


# ---------------------------------------------------------------------------
# 同梱の実仕様と corpus
# ---------------------------------------------------------------------------


def test_shipped_spec_is_valid_and_covers_required_categories():
    spec = harness.load_spec(SPEC_PATH)
    categories = {q.category for q in spec.questions}
    for required in (
        "regulation-table",
        "long-document",
        "multi-document",
        "synonym",
        "csv-table",
        "no-answer",
    ):
        assert required in categories, f"必須の評価区分が欠けている: {required}"
    assert any(not q.answerable for q in spec.questions), "該当情報なしの質問が必要"


def test_shipped_spec_expected_paths_exist_in_corpus():
    spec = harness.load_spec(SPEC_PATH)
    for question in spec.questions:
        for expected in question.expected_sources:
            assert (spec.corpus_dir / expected.path).is_file(), (
                f"{question.id}: {expected.path} が corpus にない"
            )


def test_shipped_spec_expected_line_ranges_are_within_file():
    spec = harness.load_spec(SPEC_PATH)
    for question in spec.questions:
        for expected in question.expected_sources:
            if expected.line_end is None:
                continue
            line_count = len(
                (spec.corpus_dir / expected.path).read_text(encoding="utf-8").splitlines()
            )
            assert expected.line_end <= line_count, (
                f"{question.id}: {expected.path} の行範囲が末尾を超えている"
            )


def test_corpus_chunks_are_built():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    assert chunks
    assert {c["path"] for c in chunks} >= {
        "regulations/travel-expense-rules.md",
        "guides/onboarding-handbook.md",
        "data/office-equipment-inventory.csv",
        "notes/support-meeting-notes.txt",
    }


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------


def _question(**overrides) -> harness.Question:
    base = dict(
        id="Q01",
        category="regulation",
        query="日当はいくらか",
        answerable=True,
        expected_sources=(harness.ExpectedSource(path="a/b.md", line_start=10, line_end=20),),
    )
    base.update(overrides)
    return harness.Question(**base)


def _outcome(matches, **overrides) -> harness.RouteOutcome:
    base = dict(
        route=harness.ROUTE_KEYWORD,
        status=harness.STATUS_COMPLETED,
        matches=matches,
        confidence=0.6,
        evidence_status="sufficient",
        latency_ms=5.0,
    )
    base.update(overrides)
    return harness.RouteOutcome(**base)


def test_score_question_perfect_retrieval():
    score = harness.score_question(
        _question(),
        _outcome([{"path": "a/b.md", "start_line": 12, "end_line": 18, "snippet": "本文"}]),
    )
    assert score.passed is True
    assert score.retrieval_hit is True
    assert score.hit_at_1 is True
    assert score.evidence_coverage == 1.0
    assert score.evidence_precision == 1.0
    assert score.evidence_line_overlap == 1.0


def test_score_question_counts_precision_over_all_returned_matches():
    score = harness.score_question(
        _question(),
        _outcome(
            [
                {"path": "a/b.md", "start_line": 12, "end_line": 18},
                {"path": "other/c.md", "start_line": 1, "end_line": 3},
            ]
        ),
    )
    assert score.evidence_precision == 0.5
    assert score.evidence_coverage == 1.0


def test_score_question_line_overlap_is_zero_when_range_misses():
    score = harness.score_question(
        _question(),
        _outcome([{"path": "a/b.md", "start_line": 1, "end_line": 5}]),
    )
    assert score.retrieval_hit is True
    assert score.evidence_line_overlap == 0.0


def test_score_question_require_expected_lines_needs_facts_in_overlapping_excerpt():
    """行範囲が重なっていても、必要事項が最終抜粋から消えていればFAILにする。"""
    question = _question(require_expected_lines=True, required_facts=("3,000円",))
    truncated = _outcome([{"path": "a/b.md", "start_line": 12, "end_line": 13, "snippet": "日当は"}])
    retained = _outcome(
        [{"path": "a/b.md", "start_line": 12, "end_line": 18, "snippet": "日当は3,000円"}]
    )

    missing_score = harness.score_question(question, truncated)
    retained_score = harness.score_question(question, retained)

    assert missing_score.evidence_line_overlap == 1.0
    assert missing_score.expected_lines_retained is False
    assert missing_score.passed is False
    assert retained_score.expected_lines_retained is True
    assert retained_score.passed is True


def test_score_question_without_require_expected_lines_keeps_previous_verdict():
    score = harness.score_question(
        _question(required_facts=("3,000円",)),
        _outcome([{"path": "a/b.md", "start_line": 40, "end_line": 41, "snippet": "別"}]),
    )
    assert score.expected_lines_retained is None
    assert score.passed is True


def test_evaluate_acceptance_fails_when_required_lines_missing_in_any_run():
    missing = _completed_score(expected_lines_retained=False, passed=False)
    retained = _completed_score(expected_lines_retained=True)
    summaries = [
        harness.aggregate_route([retained]),
        harness.aggregate_route([missing]),
        harness.aggregate_route([retained]),
    ]
    representative = harness._representative_summary(summaries)
    result = harness.evaluate_acceptance(representative, {"min_retrieval_hit_rate": 0.9})
    assert representative["expected_lines_missing"] == 1
    assert result["verdict"] == "FAIL"


def test_score_question_multi_document_coverage_is_partial():
    question = _question(
        expected_sources=(
            harness.ExpectedSource(path="a/b.md", line_start=1, line_end=5),
            harness.ExpectedSource(path="a/c.md", line_start=1, line_end=5),
        )
    )
    score = harness.score_question(
        question, _outcome([{"path": "a/b.md", "start_line": 1, "end_line": 5}])
    )
    assert score.evidence_coverage == 0.5
    assert score.passed is True


def test_score_question_fails_on_forbidden_source():
    question = _question(forbidden_sources=("a/b.md.metadata.json",))
    score = harness.score_question(
        question,
        _outcome(
            [
                {"path": "a/b.md", "start_line": 12, "end_line": 18},
                {"path": "a/b.md.metadata.json", "start_line": 1, "end_line": 4},
            ]
        ),
    )
    assert score.forbidden_source_hits == 1
    assert score.passed is False


def test_score_question_no_answer_requires_insufficient_status():
    question = _question(answerable=False, expected_sources=())
    passing = harness.score_question(question, _outcome([], evidence_status="insufficient"))
    failing = harness.score_question(
        question,
        _outcome(
            [{"path": "a/b.md", "start_line": 1, "end_line": 5}],
            evidence_status="partial",
        ),
    )
    assert passing.abstain_correct is True and passing.passed is True
    assert failing.abstain_correct is False and failing.passed is False


def test_score_question_skipped_route_is_neither_pass_nor_fail():
    score = harness.score_question(
        _question(),
        harness.RouteOutcome(
            route=harness.ROUTE_HYBRID,
            status=harness.STATUS_SKIPPED,
            reason="Embedding model が利用できないため未実行（PASSへ数えない）",
        ),
    )
    assert score.passed is None
    assert score.status == harness.STATUS_SKIPPED
    assert score.retrieval_hit is None


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


def test_redact_match_drops_document_text():
    redacted = harness.redact_match(
        {
            "path": "a/b.md",
            "chunk_id": "a/b.md#0001",
            "heading": "第2条",
            "start_line": 10,
            "end_line": 20,
            "rrf_score": 0.031,
            "snippet": "秘密の本文",
            "text": "秘密の本文",
        }
    )
    assert "snippet" not in redacted
    assert "text" not in redacted
    assert redacted["path"] == "a/b.md"
    assert redacted["start_line"] == 10


def test_score_question_evidence_is_redacted():
    score = harness.score_question(
        _question(),
        _outcome(
            [
                {
                    "path": "a/b.md",
                    "start_line": 12,
                    "end_line": 18,
                    "snippet": "秘密の本文",
                }
            ]
        ),
    )
    serialized = json.dumps(score.matches, ensure_ascii=False)
    assert "秘密の本文" not in serialized


# ---------------------------------------------------------------------------
# 集計と受入判定
# ---------------------------------------------------------------------------


def _completed_score(**overrides) -> harness.QuestionScore:
    base = dict(
        question_id="Q01",
        category="regulation",
        route=harness.ROUTE_KEYWORD,
        status=harness.STATUS_COMPLETED,
        passed=True,
        retrieval_hit=True,
        hit_at_1=True,
        evidence_coverage=1.0,
        evidence_precision=1.0,
        evidence_line_overlap=1.0,
        latency_ms=4.0,
    )
    base.update(overrides)
    return harness.QuestionScore(**base)


def test_aggregate_route_excludes_skipped_from_metric_denominator():
    scores = [
        _completed_score(),
        _completed_score(
            question_id="Q02", retrieval_hit=False, passed=False, evidence_coverage=0.0
        ),
        harness.QuestionScore(
            question_id="Q03",
            category="regulation",
            route=harness.ROUTE_KEYWORD,
            status=harness.STATUS_SKIPPED,
            passed=None,
        ),
    ]
    summary = harness.aggregate_route(scores)
    assert summary["questions_total"] == 3
    assert summary["completed"] == 2
    assert summary["skipped"] == 1
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["passed"] == 1
    assert summary["failed"] == 1


def test_score_and_aggregate_preserve_bounded_attempt_count():
    score = harness.score_question(_question(), _outcome([], attempts=2))
    summary = harness.aggregate_route([score])
    assert score.attempts == 2
    assert summary["attempts_mean"] == 2.0
    assert summary["attempts_median"] == 2
    assert summary["attempts_max"] == 2


def test_evaluate_acceptance_reports_not_measured_for_missing_metric():
    summary = harness.aggregate_route(
        [
            harness.QuestionScore(
                question_id="Q01",
                category="regulation",
                route=harness.ROUTE_HYBRID,
                status=harness.STATUS_SKIPPED,
                passed=None,
            )
        ]
    )
    result = harness.evaluate_acceptance(summary, {"min_retrieval_hit_rate": 0.9})
    assert result["verdict"] == "NOT_MEASURED"
    assert all(check["result"] != "PASS" for check in result["checks"])


def test_evaluate_acceptance_fails_when_threshold_missed():
    summary = harness.aggregate_route([_completed_score(retrieval_hit=False, passed=False)])
    result = harness.evaluate_acceptance(summary, {"min_retrieval_hit_rate": 0.9})
    assert result["verdict"] == "FAIL"


def test_evaluate_acceptance_enforces_forbidden_source_maximum():
    summary = harness.aggregate_route([_completed_score(forbidden_source_hits=2, passed=False)])
    result = harness.evaluate_acceptance(summary, {"max_forbidden_source_hits": 0})
    assert result["verdict"] == "FAIL"


def test_overall_verdict_is_fail_when_any_route_fails():
    reports = {
        "keyword": {"acceptance": {"verdict": "PASS"}},
        "hybrid": {"acceptance": {"verdict": "FAIL"}},
    }
    assert harness.overall_verdict(reports) == "FAIL"


def test_overall_verdict_is_not_measured_when_route_skipped():
    reports = {
        "keyword": {"acceptance": {"verdict": "PASS"}},
        "hybrid": {"acceptance": {"verdict": "NOT_MEASURED"}},
    }
    assert harness.overall_verdict(reports) == "NOT_MEASURED"


# ---------------------------------------------------------------------------
# route 実行（Ollama不要の経路のみ）
# ---------------------------------------------------------------------------


def test_keyword_route_is_deterministic():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    question = spec.questions[0]
    first = harness.run_keyword_route(question, chunks)
    second = harness.run_keyword_route(question, chunks)
    assert first.status == harness.STATUS_COMPLETED
    assert [m["chunk_id"] for m in first.matches] == [m["chunk_id"] for m in second.matches]
    assert first.evidence_status == second.evidence_status


@pytest.mark.parametrize("expansion, expected_pass", [("true", True), ("false", False)])
def test_shipped_q11_fails_per_question_when_expanded_lines_are_missing(
    monkeypatch, expansion, expected_pass
):
    """Q11は必要な行範囲が最終根拠に無ければ、質問単位とroute判定の両方でFAILになる。"""
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", expansion)
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    question = next(q for q in spec.questions if q.id == "Q11")
    assert question.require_expected_lines is True

    outcome = harness.run_keyword_route(question, chunks, corpus_dir=spec.corpus_dir)
    score = harness.score_question(question, outcome)

    assert score.expected_lines_retained is expected_pass
    assert score.passed is expected_pass
    summary = harness.aggregate_route([score])
    check = next(
        c
        for c in harness.evaluate_acceptance(summary, spec.acceptance)["checks"]
        if c["criterion"] == "expected_lines_retained"
    )
    assert check["result"] == ("PASS" if expected_pass else "FAIL")


def test_hybrid_and_agentic_routes_skip_without_models():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    question = spec.questions[0]
    hybrid = harness.run_hybrid_route(question, chunks, embed_model=None, embed_cache=None)
    agentic = harness.run_agentic_lite_route(question, chat_model=None)
    assert hybrid.status == harness.STATUS_SKIPPED
    assert agentic.status == harness.STATUS_SKIPPED
    assert "PASSへ数えない" in hybrid.reason


def test_agentic_route_uses_eval_corpus_and_isolated_embedding_inputs():
    spec = harness.load_spec(SPEC_PATH)
    question = spec.questions[0]
    chunks = [{"chunk_id": "eval#1"}]
    embed_cache = {"entries": {"eval#1": {"embedding": [1.0]}}}
    observed = {}

    def fake_pipeline(query, *, model):
        observed["query"] = query
        observed["model"] = model
        observed["chunks"] = harness.search.build_source_chunks(harness.search.SKILL_SOURCE_DIR)
        observed["embed_model"] = harness.search.detect_embed_model()
        observed["embed_cache"] = harness.search.build_or_update_embed_index("embed:tag", chunks)
        return SimpleNamespace(
            matches=[], confidence=0.0, evidence_status="insufficient", attempts=[]
        )

    with patch.object(harness.search, "run_retrieval_pipeline", side_effect=fake_pipeline):
        outcome = harness.run_agentic_lite_route(
            question,
            chat_model="chat:tag",
            chunks=chunks,
            embed_model="embed:tag",
            embed_cache=embed_cache,
        )

    assert outcome.status == harness.STATUS_COMPLETED
    assert observed == {
        "query": question.query,
        "model": "chat:tag",
        "chunks": chunks,
        "embed_model": "embed:tag",
        "embed_cache": embed_cache,
    }


def test_run_evaluation_keyword_route_produces_full_report():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    reports = harness.run_evaluation(spec, routes=[harness.ROUTE_KEYWORD], chunks=chunks, repeat=2)
    report = reports[harness.ROUTE_KEYWORD]
    assert len(report["runs"]) == 2
    assert report["summary"]["completed"] == len(spec.questions)
    assert report["stability"]["runs"] == 2
    assert report["acceptance"]["verdict"] in {"PASS", "FAIL", "NOT_MEASURED"}


def test_run_evaluation_rejects_unknown_route():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    with pytest.raises(harness.SpecError, match="未知の route"):
        harness.run_evaluation(spec, routes=["bm25"], chunks=chunks)


def test_run_evaluation_rejects_non_positive_repeat():
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    with pytest.raises(harness.SpecError, match="repeat"):
        harness.run_evaluation(spec, routes=[harness.ROUTE_KEYWORD], chunks=chunks, repeat=0)


# ---------------------------------------------------------------------------
# 該当情報なしの判定層（retrieval / answer）
# ---------------------------------------------------------------------------


def test_parse_spec_rejects_unknown_abstain_layer():
    data = _minimal_spec()
    data["questions"][0]["answerable"] = False
    data["questions"][0].pop("expected_sources")
    data["questions"][0]["abstain_layer"] = "prompt"
    with pytest.raises(harness.SpecError, match="abstain_layer"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_parse_spec_rejects_abstain_layer_on_answerable_question():
    data = _minimal_spec()
    data["questions"][0]["abstain_layer"] = "retrieval"
    with pytest.raises(harness.SpecError, match="abstain_layer"):
        harness.parse_spec(data, spec_path=SPEC_PATH)


def test_answer_layer_abstain_only_requires_not_sufficient():
    question = _question(
        answerable=False,
        expected_sources=(),
        abstain_layer=harness.ABSTAIN_LAYER_ANSWER,
    )
    partial = harness.score_question(
        question,
        _outcome(
            [{"path": "a/b.md", "start_line": 1, "end_line": 5}],
            evidence_status="partial",
        ),
    )
    confident = harness.score_question(
        question,
        _outcome(
            [{"path": "a/b.md", "start_line": 1, "end_line": 5}],
            evidence_status="sufficient",
        ),
    )
    assert partial.abstain_correct is True, "近接語ケースは根拠が返ること自体を defect としない"
    assert confident.abstain_correct is False, "近接語ケースで sufficient を宣言してはならない"
    assert partial.abstain_layer == harness.ABSTAIN_LAYER_ANSWER


def test_answer_layer_abstain_is_reported_as_not_measured_without_generation():
    score = _completed_score(
        question_id="Q10",
        retrieval_hit=None,
        hit_at_1=None,
        evidence_coverage=None,
        evidence_precision=None,
        evidence_line_overlap=None,
        abstain_correct=True,
        abstain_layer=harness.ABSTAIN_LAYER_ANSWER,
    )
    summary = harness.aggregate_route([score])
    assert summary["answer_layer_abstain_pending"] == 1
    result = harness.evaluate_acceptance(summary, {"min_abstain_accuracy": 1.0})
    assert result["verdict"] == "NOT_MEASURED"
    assert any(c["criterion"] == "answer_layer_abstain_verified" for c in result["checks"])


def test_measure_answer_probe_records_flags_without_answer_body():
    question = _question(
        id="Q10",
        category="no-answer-near-miss",
        query="海外出張の日当はいくらですか",
        answerable=False,
        expected_sources=(),
        abstain_layer=harness.ABSTAIN_LAYER_ANSWER,
        forbidden_facts=("2,500円",),
    )
    outcome = _outcome(
        [{"path": "regulations/travel.md", "snippet": "国内出張の日当"}],
        route=harness.ROUTE_HYBRID,
        evidence_status="sufficient",
    )
    with patch.object(
        harness.search,
        "stream_ollama_chat",
        return_value="該当情報なし。",
    ):
        probe = harness.measure_answer_probe(question, outcome, chat_model="chat")

    assert probe["status"] == "measured"
    assert probe["passed"] is True
    assert probe["abstain_phrase"] is True
    assert "answer" not in probe
    assert "snippet" not in probe


def test_shipped_spec_separates_abstain_layers():
    spec = harness.load_spec(SPEC_PATH)
    layers = {q.abstain_layer for q in spec.questions if not q.answerable}
    assert layers == {harness.ABSTAIN_LAYER_RETRIEVAL, harness.ABSTAIN_LAYER_ANSWER}


# ---------------------------------------------------------------------------
# 親子展開の追加指標・回答品質・OFF/ON比較
# ---------------------------------------------------------------------------


def test_parse_spec_validates_expected_expansion_and_holdout():
    data = _minimal_spec()
    data["questions"][0]["expected_expansion"] = "sometimes"
    with pytest.raises(harness.SpecError, match="expected_expansion"):
        harness.parse_spec(data, spec_path=SPEC_PATH)
    data = _minimal_spec()
    data["questions"][0]["holdout"] = "yes"
    with pytest.raises(harness.SpecError, match="holdout"):
        harness.parse_spec(data, spec_path=SPEC_PATH)
    data = _minimal_spec()
    data["questions"][0].update(expected_expansion="partial", holdout=True)
    question = harness.parse_spec(data, spec_path=SPEC_PATH).questions[0]
    assert question.expected_expansion == "partial" and question.holdout is True


def test_score_question_reports_candidate_recall_and_irrelevant_expansion():
    trace = [{"candidates": [{"path": "a/b.md", "start_line": 1, "end_line": 2}]}]
    matches = [
        {"path": "a/b.md", "start_line": 12, "end_line": 18, "snippet": "本文", "source": "expanded"},
        {"path": "a/b.md", "start_line": 40, "end_line": 45, "snippet": "別節", "source": "expanded"},
        {"path": "c.md", "start_line": 1, "end_line": 2, "snippet": "別資料", "source": "expanded"},
    ]
    score = harness.score_question(_question(), _outcome(matches, trace=trace))

    assert score.candidate_line_recall == 0.0, "候補段階では期待範囲に届いていない"
    assert score.expanded_irrelevant_ranges == 2
    summary = harness.aggregate_route([score])
    assert summary["expanded_irrelevant_ranges"] == 2
    assert summary["candidate_line_recall_mean"] == 0.0


def test_candidate_recall_is_none_without_trace():
    score = harness.score_question(
        _question(), _outcome([{"path": "a/b.md", "start_line": 12, "end_line": 18}])
    )
    assert score.candidate_line_recall is None


@pytest.mark.parametrize(
    "expected, matches, status, met",
    [
        ("partial", [{"source": "expanded", "group_partial": True}], "partial", True),
        ("partial", [{"source": "expanded", "group_partial": True}], "sufficient", False),
        ("partial", [{"source": "expanded", "group_partial": False}], "partial", False),
        ("complete", [{"source": "expanded", "group_partial": False}], "sufficient", True),
        ("complete", [{"source": "keyword"}], "sufficient", False),
    ],
)
def test_expansion_expectation_is_scored_and_fails_acceptance(expected, matches, status, met):
    base = {"path": "a/b.md", "start_line": 12, "end_line": 18, "snippet": "本文"}
    outcome = _outcome([{**base, **m} for m in matches], evidence_status=status)
    score = harness.score_question(_question(expected_expansion=expected), outcome)

    assert score.expansion_expectation_met is met
    assert score.passed is met
    result = harness.evaluate_acceptance(
        harness.aggregate_route([score]), {"min_retrieval_hit_rate": 0.9}
    )
    check = next(c for c in result["checks"] if c["criterion"] == "expansion_expectation_met")
    assert check["result"] == ("PASS" if met else "FAIL")


def test_holdout_failures_are_counted_separately():
    scores = [
        _completed_score(question_id="Q01", holdout=True, passed=False, retrieval_hit=False),
        _completed_score(question_id="Q02", holdout=True),
        _completed_score(question_id="Q03"),
    ]
    summary = harness.aggregate_route(scores)
    assert (summary["holdout_total"], summary["holdout_failed"]) == (2, 1)


def _answerable_question(**overrides):
    base = dict(
        required_facts=("3,000円",),
        forbidden_facts=("2,500円",),
        expected_sources=(harness.ExpectedSource(path="a/b.md", line_start=10, line_end=20),),
    )
    base.update(overrides)
    return _question(**base)


@pytest.mark.parametrize(
    "answer, passed",
    [
        ("日当は3,000円です。出典: skill-source/a/b.md", True),
        ("日当は3,000円です。", False),
        ("該当情報なし。出典: a/b.md 3,000円", False),
        ("日当は2,500円または3,000円です。a/b.md", False),
        ("出典: a/b.md", False),
    ],
)
def test_answer_quality_probe_checks_facts_citation_and_abstain(answer, passed):
    outcome = _outcome([{"path": "a/b.md", "start_line": 12, "end_line": 18, "snippet": "本文"}])
    with patch.object(harness.search, "stream_ollama_chat", return_value=answer):
        probe = harness.measure_answer_quality_probe(
            _answerable_question(), outcome, chat_model="chat"
        )

    assert probe["status"] == "measured"
    assert probe["passed"] is passed
    assert "answer" not in probe and "snippet" not in probe


def test_answer_quality_probe_is_not_measured_without_model_or_facts():
    outcome = _outcome([])
    no_model = harness.measure_answer_quality_probe(_answerable_question(), outcome, chat_model=None)
    no_facts = harness.measure_answer_quality_probe(
        _answerable_question(required_facts=()), outcome, chat_model="chat"
    )
    assert no_model["status"] == "not_measured"
    assert no_facts["status"] == "not_measured"


def test_apply_answer_quality_to_acceptance_fails_on_measured_failure():
    report = {"acceptance": {"verdict": "PASS", "checks": [{"criterion": "x", "result": "PASS"}]}}
    harness.apply_answer_quality_to_acceptance(
        report, [{"status": "measured", "passed": True}, {"status": "measured", "passed": False}]
    )
    assert report["acceptance"]["verdict"] == "FAIL"

    report = {"acceptance": {"verdict": "PASS", "checks": [{"criterion": "x", "result": "PASS"}]}}
    harness.apply_answer_quality_to_acceptance(
        report, [{"status": "measured", "passed": True}, {"status": "not_measured", "passed": None}]
    )
    assert report["acceptance"]["verdict"] == "NOT_MEASURED"


def test_measure_answer_layer_and_quality_use_eval_corpus_dir():
    """回答層・回答品質の probe も評価 corpus で親子展開を行う（製品資料を読まない）。"""
    spec = harness.load_spec(SPEC_PATH)
    seen = []

    def fake_run_route(route, question, **kwargs):
        seen.append(kwargs.get("corpus_dir"))
        return _outcome([], route=route)

    with patch.object(harness, "run_route", side_effect=fake_run_route):
        harness.measure_answer_layer(spec, routes=[harness.ROUTE_KEYWORD], chunks=[])
        harness.measure_answer_quality(spec, routes=[harness.ROUTE_KEYWORD], chunks=[])

    assert seen and all(value == spec.corpus_dir for value in seen)


def test_compare_expansion_reports_status_transitions_and_retry_rate():
    def report(*scores):
        return {"keyword": {"runs": [{"scores": list(scores)}]}}

    off = report(
        _completed_score(question_id="Q01", evidence_status="sufficient", attempts=1),
        _completed_score(question_id="Q02", evidence_status="partial", passed=False, attempts=2),
    )
    on = report(
        _completed_score(question_id="Q01", evidence_status="partial", passed=False, attempts=2),
        _completed_score(question_id="Q02", evidence_status="sufficient", attempts=1),
    )

    comparison = harness.compare_expansion_reports(off, on)["keyword"]

    assert comparison["status_down"] == ["Q01"]
    assert comparison["status_up"] == ["Q02"]
    assert comparison["newly_failed"] == ["Q01"]
    assert comparison["newly_passed"] == ["Q02"]
    assert comparison["off_retry_rate"] == 0.5 and comparison["on_retry_rate"] == 0.5


def test_shipped_spec_has_expansion_scenarios_and_holdout_questions():
    spec = harness.load_spec(SPEC_PATH)
    categories = {q.category for q in spec.questions}
    for required in (
        "parent-child-expansion",
        "parent-child-same-heading-other-document",
        "parent-child-partial-expansion",
        "parent-child-no-answer-near-miss",
    ):
        assert required in categories
    assert any(q.expected_expansion == "partial" for q in spec.questions)
    assert any(q.holdout for q in spec.questions)


def test_shipped_spec_keyword_route_meets_expansion_scenarios(monkeypatch):
    """同梱の親子展開シナリオが既定（展開ON）の keyword route で全て合格する。"""
    monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)
    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    targets = [q for q in spec.questions if q.category.startswith("parent-child")]
    scores = [
        harness.score_question(q, harness.run_keyword_route(q, chunks, corpus_dir=spec.corpus_dir))
        for q in targets
    ]
    assert all(s.passed for s in scores), [s.question_id for s in scores if not s.passed]
    assert all(s.evidence_status != "sufficient" for s in scores if s.question_id in {"Q14", "Q15", "Q16"})


def test_run_eval_markdown_includes_quality_and_comparison_sections():
    import run_eval

    spec = harness.load_spec(SPEC_PATH)
    chunks = harness.build_corpus_chunks(spec.corpus_dir)
    reports = harness.run_evaluation(spec, routes=[harness.ROUTE_KEYWORD], chunks=chunks)
    receipt = {
        "harness_version": run_eval.HARNESS_VERSION,
        "started_at": "s",
        "generated_at": "g",
        "verdict": harness.overall_verdict(reports),
        "spec": {"name": "eval-spec.json", "sha256": "0", "question_count": 0, "acceptance": {}},
        "corpus": {"dir": "corpus", "chunk_count": 0, "files": []},
        "product": run_eval._product_baseline(),
        "environment": {
            "python": "3",
            "platform": "p",
            "chat_model": "",
            "embed_model": "",
            "model_diagnostics": {"chat": "-", "embed": "-"},
            "repeat": 1,
            "routes_requested": [harness.ROUTE_KEYWORD],
            "parent_child_expansion": "on",
        },
        "routes": run_eval._routes_to_dict(reports),
        "answer_quality_probes": {
            "keyword": [{"question_id": "Q11", "status": "measured", "passed": True}]
        },
        "expansion_comparison": harness.compare_expansion_reports(reports, reports),
    }

    markdown = run_eval.build_markdown(receipt)

    assert "回答品質 probe" in markdown
    assert "親子展開 OFF/ON 比較" in markdown
    assert "保留質問:" in markdown
