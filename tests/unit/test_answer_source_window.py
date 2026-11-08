"""条項の文脈を保持する原文窓と、条件付き一覧の有限再生成の回帰。独立合成資料のみ。"""

from copy import deepcopy

import pytest

import answer_pipeline as pipeline
import answer_policy as ap
from answer_fixtures import FakeModel, claim
from test_answer_pipeline import make_env


def long_rule(article="第７条", paragraph="４"):
    lines = [f"{article}　研修審査の対象を定める。", ""]
    lines += [f"- 項目{i}　研修申請の説明を確認する。" for i in range(12)]
    lines += [f"{paragraph}　前項第１号又は第２号の規定に関わらず、次の研修費は審査対象としない。", ""]
    lines += ["- 一　毎月反復する研修費", "- 二　更新する教材の賃借費"]
    lines += ["その他の独立した注意事項です。" * 10] * 15
    return {"evidence_id": "E4", "path": "synthetic/training.md", "source_sha256": "fixed",
            "excerpt": "\n".join(lines), "start_line": 41, "end_line": 40 + len(lines)}, lines


def verify(text, *, article="第７条", paragraph="４", source_lines_for=None):
    evidence, lines = long_rule(article, paragraph)
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(text, "E4", "毎月反復する研修費")]}]})
    by_id = {"E4": evidence}
    assert ap.check_g1(draft, by_id, source_lines_for=source_lines_for)["c1"].status == ap.PASSED
    model = FakeModel()
    results = pipeline.verify_claims("研修費の除外条件は？", draft, ["c1"], by_id,
                                     make_env(model, source_lines_for=source_lines_for))
    return results["c1"], model, draft, evidence


@pytest.mark.parametrize("article,paragraph,text", [
    ("第７条", "４", "第7条第4項の研修費は、前項第1号又は第2号の規定にかかわらず審査対象外です。"),
    ("第七条", "４", "第七条第4項の研修費は、前項第一号又は第二号の規定にかかわらず審査対象外です。"),
    ("第７条の２", "４", "第7条の2第4項の研修費は、前項第1号又は第2号の規定にかかわらず審査対象外です。"),
])
def test_enclosing_article_and_paragraph_reach_semantic_verification(article, paragraph, text):
    result, model, _, _ = verify(text, article=article, paragraph=paragraph)
    assert result.status == ap.PASSED
    assert model.count("g2") == 1
    wire = next(payload for kind, payload in model.calls if kind == "g2")
    source = "\n".join(item["text"] for item in wire["source_catalog"].values())
    assert article in source and f"{paragraph}　前項" in source
    assert "毎月反復する研修費" in source
    assert "…" in source


@pytest.mark.parametrize("text,code", [
    ("第8条第4項で研修費を対象外としています。", "number_changed"),
    ("第7条第9項で研修費を対象外としています。", "number_changed"),
    ("研修費は金額や契約形態にかかわらず審査対象外です。", "scope_overreach"),
    ("研修費は前項第1号の規定にかかわらず審査対象外です。", "scope_overreach"),
])
def test_context_preservation_does_not_accept_wrong_numbers_or_generalization(text, code):
    result, model, _, _ = verify(text)
    assert result.status == ap.FAILED and result.codes == (code,)
    assert model.count("g2") == 0


def test_source_change_is_rejected_before_context_or_semantic_checks():
    evidence, _ = long_rule()
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim("第7条第4項の研修費は対象外です。", "E4", "毎月反復する研修費")]}]})
    checked = ap.check_g1(draft, {"E4": evidence}, source_lines_for=lambda path, sha: None)
    assert checked["c1"].status == ap.FAILED and "source_unavailable" in checked["c1"].codes


@pytest.mark.parametrize("boundary", ["第９条　別の規定である。", "## 別の規程"])
def test_context_does_not_cross_an_article_or_markdown_boundary(boundary):
    evidence, lines = long_rule()
    lines = ["第８条　別資料の条項。", "３　別資料の項。", boundary, *lines[14:]]
    evidence.update(excerpt="\n".join(lines), start_line=1, end_line=len(lines))
    span = {"start_line": 5, "end_line": 5}
    text = pipeline._source_window(evidence, span)
    assert "毎月反復する研修費" in text
    assert "第８条" not in text and "３　別資料" not in text
    assert len(text) <= pipeline._WINDOW_CHARS


def test_long_neighbor_cannot_displace_a_short_verified_citation():
    lines = ["前方の長い説明。" * 300, "手当は880円を支給する。", "後方の長い説明。" * 300]
    evidence = {"excerpt": "\n".join(lines), "start_line": 1}
    text = pipeline._source_window(evidence, {"start_line": 2, "end_line": 2})
    assert "手当は880円を支給する。" in text and len(text) <= pipeline._WINDOW_CHARS


def test_forward_context_stops_before_the_next_article():
    evidence, lines = long_rule()
    lines[18] = "第８条　別の給付は900円とする。"
    evidence["excerpt"] = "\n".join(lines)
    text = pipeline._source_window(evidence, {"start_line": 57, "end_line": 57})
    assert "第８条" not in text and "900円" not in text


def test_list_only_answer_is_rewritten_into_independent_explanation_and_keeps_items():
    evidence, lines = long_rule()
    by_id = {"E4": evidence}
    group = ap.evidence_catalog(evidence)["lists"][1]
    listing = claim("対象外となる研修費は次のとおりです。", "E4", "毎月反復する研修費",
                    source_list={"id": "E4", "group_id": group["group_id"]}, aspect="除外")
    first = {"sections": [{"heading": "除外", "claims": [listing]}]}
    explanation = claim("第7条第4項の研修費は、前項第1号又は第2号の規定にかかわらず審査対象外です。",
                        "E4", lines[14], aspect="条件")
    repaired = {"sections": [{"heading": "条件と除外", "claims": [explanation, deepcopy(listing)]}]}
    model = FakeModel(drafts=[first, repaired])
    result = pipeline.run_answer_pipeline("研修費が除外される条件と案件は？",
        ap.AnswerIntent(target="研修費", predicate_terms=("対象",), required_aspects=("除外",)),
        list(by_id.values()), make_env(model, source_lines_for=None))
    assert result.kind == "full"
    assert result.answer.startswith(explanation["text"])
    assert result.answer.count(lines[16]) == 1 and result.answer.count(lines[17]) == 1
    assert model.count("draft") == 2
    assert result.diagnostics["attempts"][0]["gates"]["G4"]["codes"] == ["no_conclusion"]
    assert result.diagnostics["attempts"][1]["conclusion_from_verified_claim"]
    prompt = [payload for kind, payload in model.calls if kind == "draft"][1]
    assert "scopeやsource_listを付けず" in prompt and "conclusionは出力しません" in prompt


def test_list_only_answer_cannot_be_promoted_or_silently_accepted():
    evidence, _ = long_rule()
    group = ap.evidence_catalog(evidence)["lists"][1]
    listing = claim("対象外の研修費は次のとおりです。", "E4", "毎月反復する研修費",
                    source_list={"id": "E4", "group_id": group["group_id"]})
    raw = {"sections": [{"claims": [listing]}]}
    model = FakeModel(drafts=[raw, raw])
    result = pipeline.run_answer_pipeline("対象外の研修費は？", ap.AnswerIntent(), [evidence],
                                          make_env(model, source_lines_for=None))
    assert result.quality["state"] != "passed" and model.count("draft") == 2
    assert all(not a["conclusion_from_verified_claim"] for a in result.diagnostics["attempts"])
    assert "no_conclusion" in result.quality["gates"]["G4"]["codes"]


@pytest.mark.parametrize("text,expected", [
    ("第7条の2第4項に基づき7開庁日前までに提出します。", []),
    ("第7条の2第4項に基づき7営業日前までに提出します。", ["7営業日前"]),
    ("第7条の2第4項に基づき700万件です。", ["700万件"]),
])
def test_clause_references_do_not_mask_changed_quantity_units(text, expected):
    source = "第７条の２　７開庁日前までに提出する。\n４　前項の金額は700万円とする。"
    assert ap.number_units_changed(text, source) == expected
