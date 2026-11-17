"""P4調整の条件欠落・観点欠落・表示/保持を固定する独立合成回帰。"""

from copy import deepcopy

import pytest

import answer_pipeline as pipeline
import answer_policy as ap
from answer_fixtures import FakeModel, claim, make_env


def evidence(text, heading="研修申請", eid="E1"):
    return {
        "evidence_id": eid,
        "path": "synthetic/training.md",
        "source_sha256": "fixed",
        "heading": heading,
        "excerpt": text,
        "start_line": 1,
        "end_line": len(text.splitlines()),
    }


def checked(text, item, quote, **extra):
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(text, "E1", quote, **extra)]}]})
    by_id = {"E1": item}
    assert ap.check_g1(draft, by_id)["c1"].status == ap.PASSED
    model = FakeModel()
    result = pipeline.verify_claims("研修の準備は？", draft, ["c1"], by_id, make_env(model))
    return result["c1"], model, draft


def test_explicit_checklist_request_survives_llm_omission():
    query = "研修申請の提出書類と提出期限、チェックリストで確認される項目を教えて。"
    explicit = ap.extract_explicit_intent(query)
    assert "チェックリストの確認項目" in explicit.required_aspects
    data = {"granularity": "practical", "required_aspects": ["提出書類", "期限"]}
    planned = ap.parse_intent_json(data, query, explicit)
    assert "チェックリストの確認項目" in planned.required_aspects


def test_explicit_checklist_exclusion_does_not_add_required_aspect():
    intent = ap.extract_explicit_intent("研修申請の提出期限を教えて。チェックリストは不要です。")
    assert "チェックリストの確認項目" not in intent.required_aspects


@pytest.mark.parametrize(
    "text,expected",
    [
        ("必要書類は申請書と評価基準です。", ap.FAILED),
        ("申請書を提出し、点数方式の場合のみ評価基準も提出します。", ap.PASSED),
        ("次の一覧が、申請方式別に必要な書類を示します。", ap.PASSED),
    ],
)
def test_list_conditions_are_checked_even_when_only_intro_is_cited(text, expected):
    source = "申請書類は次のとおり。\n- 申請書\n- 評価基準（点数方式の場合のみ）"
    item = evidence(source, "提出書類")
    group = ap.evidence_catalog(item)["lists"][0]
    result, model, _ = checked(
        text,
        item,
        "申請書類は次のとおり。",
        source_list={"id": "E1", "group_id": group["group_id"]},
    )
    assert result.status == expected
    assert model.count("g2") == (expected == ap.PASSED)
    if expected == ap.FAILED:
        assert result.codes == ("condition_dropped",)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("提案方式の研修案件が審査対象です。", ap.FAILED),
        ("提案方式の新規案件及び前回単独応募の案件が審査対象です。", ap.PASSED),
        ("次の一覧は審査対象を示しています。", ap.PASSED),
    ],
)
def test_parenthetical_eligibility_without_only_marker_is_not_generalized(text, expected):
    source = "- 三　提案方式（前号に該当するもののほか、新規案件及び前回単独応募の案件。）"
    item = evidence(source)
    extra = {} if "次の一覧" not in text else {"source_list": {"id": "E1", "group_id": "L1"}}
    result, _, _ = checked(text, item, source, **extra)
    assert result.status == expected


@pytest.mark.parametrize(
    "text,changed",
    [
        ("変更契約で金額が基準を超えるものを審査します。", True),
        ("変更契約で金額が基準以上となるものを審査します。", False),
        ("第1号の金額以上となる変更契約を審査します。", False),
    ],
)
def test_cross_referenced_amount_keeps_inclusive_boundary(text, changed):
    source = "変更契約であって、変更後の契約金額が第１号の金額以上となるもの"
    assert bool(ap.number_bounds_changed(text, source)) is changed


@pytest.mark.parametrize(
    "text,missing",
    [
        ("除外は、①毎月の研修、②会場借料、③教材の再利用、④遠隔配信、⑤受講契約です。", []),
        ("除外は、①毎月の研修、②研修費800円の契約です。", ["800"]),
        ("研修費は⑤円です。", ["5"]),
        ("研修は⑤日以内です。", ["5"]),
    ],
)
def test_inline_circled_enumeration_never_masks_a_real_quantity(text, missing):
    source = (
        "毎月の研修、会場借料、教材の再利用、遠隔配信、受講契約を除外する。研修費は700円である。"
    )
    assert ap.claim_numbers_missing(text, source) == missing
    if not missing:
        assert ap.number_units_changed(text, source) == []


def test_duplicate_background_repair_does_not_restore_the_old_wording():
    ref = [ap.EvidenceRef("E1", "", "S1"), ap.EvidenceRef("E1", "", "S2")]
    old = ap.DraftClaim(
        "old",
        "理事長は設備更新や保守方法の見直しで費用を2割削減するよう指示し、費用削減を目的に設備委員会を設置しました。",
        refs=deepcopy(ref),
    )
    new = ap.DraftClaim(
        "new",
        "設備更新・保守方法の見直しによる2割削減の理事長指示を受け、費用削減を目的として設備委員会を設置することとしました。",
        refs=deepcopy(ref),
    )
    assert pipeline._same_fact_rephrased(old, new)
    new.text += "報告書は毎月提出します。"
    assert not pipeline._same_fact_rephrased(old, new)


def test_near_duplicate_with_the_same_citations_cannot_pass_g4():
    refs = [{"id": "E1", "span_id": "S1"}, {"id": "E1", "span_id": "S2"}]
    draft = ap.parse_answer_draft(
        {
            "conclusion": {"text": "計画の妥当性を審査します。", "evidence": refs},
            "sections": [
                {
                    "claims": [
                        {
                            "text": "理事長は設備更新や保守方法の見直しで費用を2割削減するよう指示し、費用削減を目的に設備委員会を設置しました。",
                            "evidence": refs,
                        },
                        {
                            "text": "設備更新・保守方法の見直しによる2割削減の理事長指示を受け、費用削減を目的として設備委員会を設置することとしました。",
                            "evidence": refs,
                        },
                    ]
                }
            ],
        }
    )
    assert "duplicate_claim" in ap.check_g4_structural(draft, set(draft.claims)).codes


def test_shared_original_items_are_referenced_with_their_earlier_group():
    source = (
        "## 筆記方式の提出書類\n- 申請書\n- 評価基準\n## 面接方式の提出書類\n- 申請書\n- 面接要領"
    )
    item = evidence(source, "提出書類")
    groups = ap.evidence_catalog(item)["lists"]
    draft = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "heading": "筆記方式",
                    "claims": [
                        claim(
                            "筆記方式の必要書類です。",
                            "E1",
                            "評価基準",
                            source_list={"id": "E1", "group_id": groups[0]["group_id"]},
                        )
                    ],
                },
                {
                    "heading": "面接方式",
                    "claims": [
                        claim(
                            "面接方式の必要書類です。",
                            "E1",
                            "面接要領",
                            source_list={"id": "E1", "group_id": groups[1]["group_id"]},
                        )
                    ],
                },
            ]
        }
    )
    ap.check_g1(draft, {"E1": item})
    body = ap.render_answer(draft, set(draft.claims))
    assert body.count("> - 申請書") == 1
    assert "前掲「筆記方式」の項目1と共通" in body
    assert "> - 面接要領" in body


def test_checklist_name_and_form_submission_are_not_criteria_coverage():
    intent = ap.AnswerIntent(
        required_aspects=("チェックリストの確認項目",), granularity="practical"
    )
    fake = ap.DraftClaim(
        "c1", "チェックリストを申請書と一緒に提出します。", aspect="チェックリストの確認項目"
    )
    assert not ap._aspect_covered(intent.required_aspects[0], [fake])


def test_criteria_categories_require_substantive_verified_claims():
    source = "## チェックリスト\n| 項目 | 確認内容 | 結果 |\n|---|---|---|\n| 1. 資格 | 受講資格を不当に制限していないか。 | |\n| 2. 期間 | 準備期間を十分に確保しているか。 | |"
    item = evidence(source, "チェックリスト")
    intent = ap.AnswerIntent(
        required_aspects=("チェックリストの確認項目",), granularity="practical"
    )
    only = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "claims": [
                        claim(
                            "受講資格を不当に制限しないことを確認します。",
                            "E1",
                            "受講資格を不当に制限していないか。",
                            aspect="チェックリストの確認項目",
                        )
                    ]
                }
            ]
        }
    )
    assert pipeline._missing_checklist_topics(only, intent, {"c1"}, [item]) == [
        "チェックリストの確認項目"
    ]
    both = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "claims": [
                        claim(
                            "受講資格を不当に制限しないことを確認します。",
                            "E1",
                            "受講資格を不当に制限していないか。",
                            aspect="チェックリストの確認項目",
                        ),
                        claim(
                            "準備期間を十分に確保することを確認します。",
                            "E1",
                            "準備期間を十分に確保しているか。",
                            aspect="チェックリストの確認項目",
                        ),
                    ]
                }
            ]
        }
    )
    assert pipeline._missing_checklist_topics(both, intent, set(both.claims), [item]) == []
    assert pipeline._missing_checklist_topics(both, intent, {"c1"}, [item])


def test_g3_reports_missing_criteria_even_when_llm_claims_all_aspects_pass(monkeypatch):
    # Exercise the completeness gate when original-criteria completion cannot run.
    monkeypatch.setattr(pipeline, "_seed_checklist_topics", lambda *args: 0)
    source = "## チェックリスト\n| 項目 | 確認内容 |\n|---|---|\n| 資格 | 受講資格を不当に制限していないか。 |"
    item = evidence(source, "チェックリスト")
    data = {
        "sections": [
            {
                "claims": [
                    claim(
                        "チェックリストの確認項目を記載した申請書を提出します。",
                        "E1",
                        "受講資格を不当に制限していないか。",
                        aspect="チェックリストの確認項目",
                    )
                ]
            }
        ]
    }
    model = FakeModel(
        drafts=[data, data],
        quality={
            "answers_question": "yes",
            "granularity_fit": "fit",
            "missing_aspects": [],
            "structure_issues": [],
        },
    )
    intent = ap.AnswerIntent(
        required_aspects=("チェックリストの確認項目",), granularity="practical"
    )
    result = pipeline.run_answer_pipeline(
        "確認項目は？", intent, [item], make_env(model, source_lines_for=None)
    )
    assert result.quality["state"] != "passed"
    assert "checklist_topics_missing" in result.quality["gates"]["G3"]["codes"]


@pytest.mark.parametrize(
    "text,missing",
    [
        ("点数方式の理由書、評価基準と委員名簿を提出します。", True),
        ("点数方式の場合は、理由書、評価基準と委員名簿を提出します。", False),
    ],
)
def test_a_condition_name_in_the_document_title_is_not_the_applicability_condition(text, missing):
    rows = [
        "- 点数方式の理由書（点数方式の場合のみ）",
        "- 評価基準（点数方式の場合のみ）",
        "- 委員名簿（点数方式の場合のみ）",
    ]
    assert bool(ap.dropped_qualifiers(text, rows)) is missing


def test_literal_compound_condition_is_not_rejected():
    source = "- 交通費（受講者が県外の場合のみ）"
    assert not ap.dropped_qualifiers("受講者が県外の場合のみ交通費を支給します。", [source])
    assert not ap.dropped_qualifiers("受講者が県外であれば交通費を支給します。", [source])
    assert ap.dropped_qualifiers("受講者が県外に出かける交通費を支給します。", [source])


def test_establishment_background_is_not_direct_from_an_ancestor_caption():
    intent = ap.AnswerIntent(
        target_terms=("設備委員会",),
        predicate_terms=("審議", "審査"),
        required_aspects=("設置の背景との区別",),
    )
    source = "今般、理事長の指示があった。ついては、費用削減を目的として設備委員会を設置する。"
    result = ap.classify_role(source, "設置通知", heading_path=["審査事項"], intent=intent)
    assert result.role == ap.ROLE_BACKGROUND
    assert "establishment_relation" in result.reasons
    requested = ap.extract_explicit_intent("設備委員会を設置した趣旨を教えて")
    assert ap.classify_role(source, "設置通知", intent=requested).role == ap.ROLE_DIRECT


@pytest.mark.parametrize(
    "text",
    [
        "研修費は金額要件に該当しても対象外です。",
        "研修費は契約形態の要件を満たしても対象外です。",
        "研修費は金額要件を満たす場合であっても対象外です。",
    ],
)
def test_even_if_wording_cannot_widen_specific_rule_precedence(text):
    source = "前項第１号又は第２号の規定に関わらず、研修費は対象外とする。"
    result, model, _ = checked(text, evidence(source), source)
    assert result.status == ap.FAILED and result.codes == ("scope_overreach",)
    assert model.count("g2") == 0


def test_list_identity_ignores_fields_resolved_by_g1():
    refs = [ap.EvidenceRef("E1", "", "S1")]
    old = ap.DraftClaim(
        "old",
        "研修に必要な書類は次の一覧のとおりです。",
        refs=deepcopy(refs),
        source_list={"id": "E1", "group_id": "L1", "from": "申請書", "to": "理由書"},
    )
    new = ap.DraftClaim(
        "new", old.text, refs=deepcopy(refs), source_list={"id": "E1", "group_id": "L1"}
    )
    assert pipeline._same_fact_rephrased(old, new)
    new.source_list["group_id"] = "L2"
    assert not pipeline._same_fact_rephrased(old, new)


def test_retention_does_not_merge_changed_quantity_units_or_added_facts():
    refs = [ap.EvidenceRef("E1", "", "S1")]
    old = ap.DraftClaim(
        "old",
        "申請書は研修開始の7開庁日前までに事務局へ提出しなければなりません。",
        refs=deepcopy(refs),
    )
    new = ap.DraftClaim("new", old.text.replace("開庁", "営業"), refs=deepcopy(refs))
    assert not pipeline._same_fact_rephrased(old, new)
    new.text = old.text[:-1] + "が、申請先は地区本部です。"
    assert not pipeline._same_fact_rephrased(old, new)


def test_source_binding_discards_role_based_on_overlapping_prefix():
    original = "今般、理事長の指示があった。ついては、費用削減を目的として設備委員会を設置する。"
    item = evidence("審査する内容は計画の妥当性である。" * 12, "設置通知")
    item.update(start_line=1, end_line=1, role=ap.ROLE_DIRECT)
    env = make_env(FakeModel(), source_lines_for=lambda path, sha: [original])
    assert pipeline._bind_source_rows([item], env) == 1
    assert "role" not in item
    assert (
        ap.classify_role(
            item["excerpt"],
            item["heading"],
            intent=ap.AnswerIntent(target_terms=("設備委員会",), predicate_terms=("審査",)),
        ).role
        == ap.ROLE_BACKGROUND
    )


def test_raw_relation_is_an_independent_claim_and_passes_the_same_gates():
    source = "第７条　研修費の対象を定める。\n４　前項第１号又は第２号の規定に関わらず、次の研修費は対象としない。\n- 一　毎月支払う受講料\n- 二　会場借料"
    item = evidence(source, "対象と除外")
    data = {
        "sections": [
            {
                "claims": [
                    claim(
                        "対象外の研修費は次のとおりです。",
                        "E1",
                        "毎月支払う受講料",
                        source_list={"id": "E1", "group_id": "L1"},
                    )
                ]
            }
        ]
    }
    model = FakeModel(drafts=[data])
    intent = ap.AnswerIntent(required_aspects=("除外", "条件"), predicate_terms=("対象",))
    result = pipeline.run_answer_pipeline(
        "対象外の研修費と条件は？", intent, [item], make_env(model, source_lines_for=None)
    )
    assert result.kind == "full"
    assert "第１号又は第２号" in result.answer
    assert result.answer.count("毎月支払う受講料") == 1
    assert model.count("draft") == 1 and model.count("g2") == 1
    assert all(g["status"] == ap.PASSED for g in result.quality["gates"].values())


def test_relation_seed_cannot_bypass_g2_or_source_change():
    source = "前項第１号又は第２号の規定に関わらず、研修費は対象としない。\n- 一　毎月支払う受講料"
    item = evidence(source, "除外")
    data = {
        "sections": [
            {
                "claims": [
                    claim(
                        "対象外の研修費は次のとおりです。",
                        "E1",
                        "毎月支払う受講料",
                        source_list={"id": "E1", "group_id": "L1"},
                    )
                ]
            }
        ]
    }
    intent = ap.AnswerIntent(required_aspects=("除外", "条件"))
    rejecting = FakeModel(
        drafts=[data, data], g2=lambda cs: {c["claim_id"]: ("unsupported", "other") for c in cs}
    )
    result = pipeline.run_answer_pipeline(
        "除外条件は？", intent, [item], make_env(rejecting, source_lines_for=None)
    )
    assert rejecting.count("g2") > 0
    assert any(
        c["claim_id"] == "original_relation"
        for purpose, payload in rejecting.calls
        if purpose == "g2"
        for c in payload["claims"]
    )
    assert result.kind == "fallback" and "原文では" not in result.answer.split("## 参考")[0]
    changed = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline(
        "除外条件は？", intent, [item], make_env(changed, source_lines_for=lambda p, s: None)
    )
    assert result.kind == "fallback" and changed.count("g2") == 0


@pytest.mark.parametrize("limit", ["claims", "sections", "quote", "already-explained"])
def test_original_relation_respects_existing_limits_and_does_not_duplicate(limit):
    source = "前項第１号又は第２号の規定に関わらず、研修費は対象としない。"
    intent = ap.AnswerIntent(required_aspects=("条件", "除外"))
    draft = ap.parse_answer_draft(
        {"sections": [{"claims": [claim("研修の説明です。", "E1", source)]}]}
    )
    if limit == "claims":
        for n in range(ap.MAX_CLAIMS_TOTAL - len(draft.claims)):
            draft.claims[f"extra{n}"] = ap.DraftClaim(f"extra{n}", "別の事項です。")
    elif limit == "sections":
        draft.sections.extend(
            ap.DraftSection("別の観点", None, [])
            for _ in range(ap.MAX_SECTIONS - len(draft.sections))
        )
    elif limit == "quote":
        source = source[:-1] + "。" + "長い導入規定" * ap.MAX_CLAIM_CHARS
    else:
        draft.claims["c1"].text = source
    before = (len(draft.claims), len(draft.sections))
    assert not pipeline._seed_exception_relation(draft, intent, [evidence(source)])
    assert (len(draft.claims), len(draft.sections)) == before
    assert "original_relation" not in draft.claims


@pytest.mark.parametrize("aspects", [("条件",), ("除外",)])
def test_relation_seed_requires_both_explicit_requested_aspects(aspects):
    source = "前項第１号又は第２号の規定に関わらず、研修費は対象としない。"
    draft = ap.parse_answer_draft(
        {"sections": [{"claims": [claim("研修の説明です。", "E1", source)]}]}
    )
    assert not pipeline._seed_exception_relation(
        draft, ap.AnswerIntent(required_aspects=aspects), [evidence(source)]
    )
    assert "original_relation" not in draft.claims


@pytest.mark.parametrize("kind", ["normal", "deep"])
def test_original_parent_scope_reaches_draft_and_verification_without_crossing_sibling(kind):
    lines = [
        "# 研修規程",
        "## 対面研修",
        "### 確認表",
        "以前の表",
        "## 通信研修",
        "### 確認表",
        "- 教材の配送に3営業日以上を確保するか。",
    ]
    item = evidence(lines[5] + "\n" + lines[6], "確認表")
    item.update(start_line=6, end_line=7)
    if kind == "deep":
        item["context"] = [{"path": item["path"], "start_line": 2, "text": "## 偽の研修方式"}]
    env = make_env(FakeModel(), source_lines_for=lambda p, s: lines)
    assert pipeline._bind_source_rows([item], env) == 1
    assert item["verified_parent_headings"] == ["## 通信研修", "# 研修規程"]
    intent = ap.AnswerIntent(granularity="practical", required_aspects=("確認項目",))
    prompt = pipeline.build_draft_prompt(
        "通信研修の確認項目は？", intent, [item], {"E1": ap.ROLE_DIRECT}
    )
    assert "## 通信研修" in prompt and "対面研修" not in prompt and "偽の研修方式" not in prompt
    draft = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "claims": [
                        claim(
                            "通信研修では教材の配送に3営業日以上を確保することを確認します。",
                            "E1",
                            lines[6],
                        )
                    ]
                }
            ]
        }
    )
    assert (
        ap.check_g1(draft, {"E1": item}, source_lines_for=lambda p, s: lines)["c1"].status
        == ap.PASSED
    )
    payload = pipeline._claim_payload(draft.claims["c1"], "", {"E1": item})
    assert (
        "## 通信研修" in payload["source"][0]["text"]
        and "対面研修" not in payload["source"][0]["text"]
    )


def test_parent_scope_requires_the_same_original_hash_and_preserves_heading_budget():
    lines = ["# " + "原本" * 200, "## 通信研修", "### 確認表", "確認は3営業日前に行う。"]
    item = evidence(lines[2] + "\n" + lines[3], "確認表")
    item.update(start_line=3, end_line=4)
    changed = deepcopy(item)
    assert (
        pipeline._bind_source_rows(
            [changed], make_env(FakeModel(), source_lines_for=lambda p, s: None)
        )
        == 0
    )
    assert "verified_parent_headings" not in changed
    assert (
        pipeline._bind_source_rows(
            [item], make_env(FakeModel(), source_lines_for=lambda p, s: lines)
        )
        == 1
    )
    assert item["verified_parent_headings"] == ["## 通信研修"]
    assert len("\n".join(item["verified_parent_headings"])) <= 300


def test_parent_scope_ignores_fenced_examples_and_keeps_valid_original_context():
    lines = [
        "# 研修規程",
        "## 通信研修",
        "```md",
        "## 対面研修",
        "```",
        "導入規定",
        "### 確認表",
        "確認内容",
    ]
    item = evidence("### 確認表\n確認内容", "確認表")
    item.update(
        start_line=7,
        end_line=8,
        context=[{"path": item["path"], "start_line": 6, "text": "導入規定"}],
    )
    assert (
        pipeline._bind_source_rows(
            [item], make_env(FakeModel(), source_lines_for=lambda p, s: lines)
        )
        == 1
    )
    assert item["verified_parent_headings"] == ["## 通信研修", "# 研修規程"]
    assert any(row["text"] == "導入規定" for row in item["context"])


@pytest.mark.parametrize("comparison", ["区別", "比較", "違い", "相違", "切り分け"])
def test_background_comparison_alias_does_not_promote_background(comparison):
    source = "今般、理事長の指示があった。ついては、費用削減を目的として設備委員会を設置する。"
    intent = ap.AnswerIntent(
        target_terms=("設備委員会",),
        predicate_terms=("審議",),
        required_aspects=(f"各会議体の設置背景の{comparison}",),
    )
    role = ap.classify_role(source, "設置通知", intent=intent).role
    assert role == ap.ROLE_BACKGROUND
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(source, "E1", source)]}]})
    assert not pipeline._place_verified_conclusion(draft, {"c1"}, intent, {"E1": role})
    assert draft.conclusion_id is None


@pytest.mark.parametrize(
    "intent",
    [
        ap.AnswerIntent(required_aspects=("設置の背景",)),
        ap.AnswerIntent(predicate_terms=("趣旨",), required_aspects=("設置背景の違い",)),
    ],
)
def test_background_as_main_question_remains_direct(intent):
    source = "今般、理事長の指示があった。ついては、費用削減を目的として設備委員会を設置する。"
    assert ap.classify_role(source, "設置通知", intent=intent).role == ap.ROLE_DIRECT


EXCLUSION_SOURCE = "４　前項第１号又は第２号の規定に関わらず、次の研修費は対象としない。\n- 一　毎月支払う受講料\n- 二　会場借料"


def overbroad_list_draft(source=EXCLUSION_SOURCE):
    return {
        "sections": [
            {
                "claims": [
                    claim(
                        "金額要件を満たしても次の研修費は対象外です。",
                        "E1",
                        source.splitlines()[0],
                        source_list={"id": "E1", "group_id": "L1"},
                        aspect="除外",
                    )
                ]
            }
        ]
    }


def test_unsupported_list_explanation_preserves_independently_checked_original_list():
    model = FakeModel(drafts=[overbroad_list_draft(), overbroad_list_draft()])
    intent = ap.AnswerIntent(required_aspects=("除外", "条件"))
    result = pipeline.run_answer_pipeline(
        "除外条件は？", intent, [evidence(EXCLUSION_SOURCE)], make_env(model, source_lines_for=None)
    )
    body = result.answer.split("## 参考")[0]
    assert "金額要件を満たしても" not in body
    assert "第１号又は第２号" in body
    assert body.count("> - 一　毎月支払う受講料") == 1
    assert body.count("> - 二　会場借料") == 1
    assert any(row["original_lists_split"] == 1 for row in result.diagnostics["attempts"])
    assert any(
        c["claim_id"].startswith("original_list_") and c.get("listed_source_items")
        for purpose, payload in model.calls
        if purpose == "g2"
        for c in payload["claims"]
    )


@pytest.mark.parametrize("failure", ["g2", "changed-source"])
def test_split_list_does_not_bypass_original_or_semantic_verification(failure):
    data = overbroad_list_draft()
    model = FakeModel(
        drafts=[data, data],
        g2=(lambda cs: {c["claim_id"]: ("unsupported", "other") for c in cs})
        if failure == "g2"
        else None,
    )
    env = make_env(
        model, source_lines_for=(lambda p, s: None) if failure == "changed-source" else None
    )
    result = pipeline.run_answer_pipeline(
        "除外条件は？",
        ap.AnswerIntent(required_aspects=("除外", "条件")),
        [evidence(EXCLUSION_SOURCE)],
        env,
    )
    assert result.kind == "fallback"
    assert "> - 一　毎月支払う受講料" not in result.answer.split("## 参考")[0]
    assert model.count("g2") > 0 if failure == "g2" else model.count("g2") == 0


@pytest.mark.parametrize(
    "failure",
    ["limit", "missing-group", "scope", "intervening-line", "long-intro", "already-correct"],
)
def test_list_split_requires_bounded_independent_immediate_original_context(failure):
    source = EXCLUSION_SOURCE
    if failure == "intervening-line":
        source = source.replace("\n- 一", "\n別の案件は別途審査する。\n- 一")
    if failure == "long-intro":
        source = source.replace("次の研修費", "詳しい研修費の説明" * 80 + "次の研修費")
    draft = ap.parse_answer_draft(overbroad_list_draft(source))
    if failure == "limit":
        for n in range(ap.MAX_CLAIMS_TOTAL - len(draft.claims)):
            draft.claims[f"extra{n}"] = ap.DraftClaim(f"extra{n}", "別の事項です。")
    elif failure == "missing-group":
        draft.claims["c1"].source_list["group_id"] = "L99"
    elif failure == "scope":
        draft.sections[0].scope_claim_id = "unverified_scope"
    elif failure == "already-correct":
        draft.claims["c1"].text = "前項第１号又は第２号の規定に関わらず、次の研修費は対象としない。"
    assert pipeline._split_exception_list(draft, [evidence(source)]) == 0
    assert draft.claims["c1"].source_list is not None


def retained_draft(old_text, new_text):
    refs = [ap.EvidenceRef("E1", "", "S1")]
    draft = ap.AnswerDraft()
    draft.claims = {
        "kept_c1": ap.DraftClaim("kept_c1", old_text, refs=deepcopy(refs)),
        "c1": ap.DraftClaim("c1", new_text, refs=deepcopy(refs)),
    }
    draft.sections = [
        ap.DraftSection("以前", None, ["kept_c1"]),
        ap.DraftSection("修復", None, ["c1"]),
    ]
    return draft


@pytest.mark.parametrize("reverse", [False, True])
def test_verified_collective_and_separate_sentences_do_not_restore_duplicate(reverse):
    sentences = [
        "筆記方式は公告前に審査を受けます。",
        "面接方式は案内前に審査を受けます。",
        "推薦方式は採用前に審査を受けます。",
    ]
    draft = retained_draft("".join(sentences), sentences[0])
    for n, text in enumerate(sentences[1:], 2):
        draft.claims[f"c{n}"] = ap.DraftClaim(f"c{n}", text, refs=deepcopy(draft.claims["c1"].refs))
        draft.sections[1].claim_ids.append(f"c{n}")
    if reverse:
        draft = retained_draft(sentences[0], "".join(sentences))
    verified = set(draft.claims)
    assert pipeline._prune_retained_sentences(draft, verified) == 1
    assert "kept_c1" not in draft.claims and "kept_c1" not in verified
    assert all(section.claim_ids for section in draft.sections)


@pytest.mark.parametrize(
    "change",
    [
        "unverified",
        "quantity-unit",
        "condition",
        "negation",
        "extra-fact",
        "different-citation",
        "list",
        "scope",
    ],
)
def test_retained_sentences_preserve_changed_or_unverified_facts(change):
    text = "書類は7開庁日前に提出します。点数方式の場合のみ理由書が必要です。"
    draft = retained_draft(text, text)
    new = draft.claims["c1"]
    verified = set(draft.claims)
    if change == "unverified":
        verified.remove("c1")
    elif change == "quantity-unit":
        new.text = text.replace("開庁", "営業")
    elif change == "condition":
        new.text = text.replace("の場合のみ", "では")
    elif change == "negation":
        new.text = text.replace("必要です", "必要ではない")
    elif change == "extra-fact":
        draft.claims["kept_c1"].text += "提出先は地区本部です。"
    elif change == "different-citation":
        new.refs[0].evidence_id = "E2"
    elif change == "list":
        draft.claims["kept_c1"].source_list = {"id": "E1", "group_id": "L1"}
    else:
        draft.claims["kept_c1"].kind = "scope"
    assert pipeline._prune_retained_sentences(draft, verified) == 0
    assert "kept_c1" in draft.claims and "kept_c1" in verified


@pytest.mark.parametrize("reject_new", [False, True])
def test_pipeline_prunes_only_after_rewritten_sentences_pass_g2(reject_new):
    a = "筆記方式の公告を掲載する前に審査を受けます。"
    b = "面接方式では応募者の選考と通知の準備を終えた後に審査します。"
    source = a + b
    first = {"sections": [{"claims": [claim(a, "E1", source), claim(b, "E1", source)]}]}
    second = {"sections": [{"claims": [claim(source, "E1", source)]}]}
    model = FakeModel(
        drafts=[first, second],
        g2=lambda cs: {
            c["claim_id"]: ("unsupported", "other")
            if reject_new and c["claim"] == source
            else ("supported", "ok")
            for c in cs
        },
    )
    result = pipeline.run_answer_pipeline(
        "審査時期と提出先は？",
        ap.AnswerIntent(required_aspects=("時期", "提出先")),
        [evidence(source)],
        make_env(model, source_lines_for=None),
    )
    assert model.count("draft") == 2
    assert result.diagnostics["attempts"][1]["retained_duplicates_removed"] == (
        0 if reject_new else 2
    )
    body = result.answer.split("## 参考")[0]
    assert body.count(a) == 1 and body.count(b) == 1


@pytest.mark.parametrize("same_scope", [False, True])
def test_retained_duplicate_sentences_keep_their_applicable_section_scope(same_scope):
    draft = retained_draft("書類の審査を受けます。", "書類の審査を受けます。")
    draft.claims["kept_scope"] = ap.DraftClaim("kept_scope", "県内研修の手続です。", kind="scope")
    draft.claims["scope"] = ap.DraftClaim(
        "scope", "県内研修の手続です。" if same_scope else "県外研修の手続です。", kind="scope"
    )
    draft.sections[0].scope_claim_id = "kept_scope"
    draft.sections[1].scope_claim_id = "scope"
    verified = set(draft.claims)
    assert pipeline._prune_retained_sentences(draft, verified) == int(same_scope)
    assert ("kept_c1" in draft.claims) is not same_scope
    assert ("kept_scope" in verified) is not same_scope


@pytest.mark.parametrize("aspect", ["例外", "除外", "例外・除外"])
def test_original_exclusion_wording_counts_only_when_verified(aspect):
    source = EXCLUSION_SOURCE.replace("対象としない", "審査案件としない")
    data = overbroad_list_draft(source)
    model = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline(
        "例外の条件は？",
        ap.AnswerIntent(required_aspects=(aspect, "条件")),
        [evidence(source)],
        make_env(model, source_lines_for=None),
    )
    assert aspect not in result.quality["unconfirmedAspects"]
    assert result.quality["gates"]["G3"]["status"] == ap.PASSED
    assert "> - 一　毎月支払う受講料" in result.answer
    assert result.quality["gates"]["G2"]["status"] == ap.FAILED
    assert result.quality["state"] != "passed"
    unrelated = ap.DraftClaim("unrelated", "参考となる研修案件を示します。", aspect=aspect)
    assert not ap._aspect_covered(aspect, [unrelated])


@pytest.mark.parametrize("aspect", ["例外", "除外", "例外・除外"])
def test_source_wording_of_exclusion_does_not_require_a_generated_aspect_label(aspect):
    supported = ap.DraftClaim("c1", "原文ではこれらを審査案件としないと定めています。")
    assert ap._aspect_covered(aspect, [supported])
    unrelated = ap.DraftClaim("c2", "原文の審査案件を示します。", aspect=aspect)
    assert not ap._aspect_covered(aspect, [unrelated])
