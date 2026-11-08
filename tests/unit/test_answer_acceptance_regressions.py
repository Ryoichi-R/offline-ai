"""Acceptance regressions using independent synthetic application rules."""

from copy import deepcopy

import pytest

import answer_pipeline as pipeline
import answer_policy as ap
from answer_fixtures import FakeModel, claim, make_env


def evidence(text, heading="提出書類"):
    return {"evidence_id": "E1", "path": "notice/application.md", "source_sha256": "b" * 64,
            "heading": heading, "start_line": 1, "end_line": len(text.splitlines()), "excerpt": text}


DOCUMENTS = """## 提出書類
① 通常申請
- 申請書
- 確認票（新規申請の場合のみ）
② 更新申請
- 更新申請書
- 前回の確認記録
③ 特別申請
- 理由書
- 特別確認票
"""
INTENT = ap.AnswerIntent(required_aspects=("提出書類",))


def partial_draft():
    return {"sections": [{"heading": "書類", "claims": [
        claim("通常申請の書類は次のとおりです。", "E1", "① 通常申請\n- 申請書",
              source_list={"id": "E1", "group_id": "L1"}, aspect="提出書類")]}]}


def test_sibling_document_list_scopes_do_not_inherit_other_methods():
    groups = pipeline._required_document_groups(INTENT, evidence(DOCUMENTS))
    assert len(groups) == 3
    assert "通常申請" not in groups[1]["heading"]
    assert "通常申請" not in groups[2]["heading"]
    assert "更新申請" not in groups[2]["heading"]


def test_missing_document_groups_are_completed_and_semantically_checked():
    item = evidence(DOCUMENTS)
    data = partial_draft()
    model = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline("提出書類を教えてください", INTENT, [item],
                                          make_env(model, source_lines_for=None))
    assert result.diagnostics["attempts"][0]["document_lists_from_original"] == 2
    assert result.quality["gates"]["G1"]["status"] == ap.PASSED
    assert result.quality["gates"]["G2"]["status"] == ap.PASSED
    assert "required_list_missing" not in result.quality["gates"]["G3"]["codes"]
    for row in ("申請書", "確認票（新規申請の場合のみ）", "更新申請書", "前回の確認記録", "理由書", "特別確認票"):
        assert row in result.answer
    checked = [c for stage, payload in model.calls if stage == "g2" for c in payload["claims"]]
    assert any(c["claim_id"].startswith("original_documents_") and c.get("listed_source_items") for c in checked)


@pytest.mark.parametrize("failure", ["source_changed", "unsupported"])
def test_completed_lists_cannot_bypass_original_or_semantic_rejection(failure):
    data = partial_draft()
    model = FakeModel(drafts=[data, data],
                      g2=(lambda cs: {c["claim_id"]: ("unsupported", "other") for c in cs})
                      if failure == "unsupported" else None)
    env = make_env(model, source_lines_for=(lambda path, sha: None) if failure == "source_changed" else None)
    result = pipeline.run_answer_pipeline("提出書類を教えてください", INTENT, [evidence(DOCUMENTS)], env)
    assert result.kind == "fallback"
    assert result.quality["state"] != ap.QUALITY_PASSED
    assert model.count("g2") == 0 if failure == "source_changed" else model.count("g2") > 0


@pytest.mark.parametrize("limit", ["sections", "claims", "lists"])
def test_completion_respects_existing_limits(limit):
    draft = ap.parse_answer_draft(partial_draft())
    if limit == "sections":
        draft.sections.extend(ap.DraftSection("別の節", None, [])
                              for _ in range(ap.MAX_SECTIONS - len(draft.sections)))
    elif limit == "claims":
        draft.claims.update({f"extra{i}": ap.DraftClaim(f"extra{i}", "別の事項")
                             for i in range(ap.MAX_CLAIMS_TOTAL - len(draft.claims))})
    else:
        for i in range(ap.MAX_LIST_BLOCKS - 1):
            draft.claims[f"extra{i}"] = ap.DraftClaim(f"extra{i}", "別の一覧", source_list={"id": "E9", "group_id": f"L{i}"})
    before = deepcopy(draft)
    assert pipeline._seed_required_document_lists(draft, INTENT, [evidence(DOCUMENTS)]) == 0
    assert draft == before


def test_no_requested_documents_or_no_original_heading_means_no_completion():
    draft = ap.parse_answer_draft(partial_draft())
    assert pipeline._seed_required_document_lists(draft, ap.AnswerIntent(), [evidence(DOCUMENTS)]) == 0
    item = evidence("- 書類甲\n- 書類乙")
    assert pipeline._seed_required_document_lists(draft, INTENT, [item]) == 0


@pytest.mark.parametrize("text,scope,expected", [
    ("申請は提出前に審査します。", "", ["承認のない申請"]),
    ("通常の手続は登録前、申請は提出前に審査します。", "", ["承認のない申請"]),
    ("承認のない申請は提出前に審査します。", "", []),
    ("申請は提出前に審査します。", "承認のない申請について", []),
    ("承認を受けていない申請は提出前に審査します。", "", []),
])
def test_subject_limit_guard_preserves_conditions_and_leaves_other_paraphrases_to_g2(text, scope, expected):
    assert ap.dropped_subject_qualifiers(text, ["承認のない申請については、提出前に審査する。"], scope) == expected


def test_a_bare_subtype_is_rejected_even_when_the_fake_semantic_model_accepts_it():
    source = "承認のない申請については、提出前に審査する。"
    data = {"sections": [{"claims": [claim("申請は提出前に審査します。", "E1", source),
                                     claim("承認のない申請は提出前に審査します。", "E1", source)]}]}
    model = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline("審査時期は？", ap.AnswerIntent(), [evidence(source, "審査時期")],
                                          make_env(model, source_lines_for=None))
    assert result.quality["gates"]["G2"]["status"] == ap.FAILED
    assert "condition_dropped" in result.quality["gates"]["G2"]["codes"]
    checked = [c for stage, payload in model.calls if stage == "g2" for c in payload["claims"]]
    assert not any(c["claim"] == "申請は提出前に審査します。" for c in checked)


def test_optional_no_does_not_turn_the_original_vague_frequency_into_a_numeric_limit():
    source = ["一定の頻度で実施する検査で、直近に複数者から有効な提案があったもの。",
              "別の検査では10件以上を対象とする。"]
    assert ap.rounded_limits("一定頻度で実施する検査で、直近に複数者から有効な提案があったものです。", source) == []
    assert ap.rounded_limits("一定規模の検査を対象とします。", ["10件以上の検査を対象とする。"])


def test_repair_prompt_identifies_the_rejected_claim_and_its_specific_fix():
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim("全方式が例外です。", "E1", "原文の条件")]}]})
    draft.claims["c1"].required_fixes = ["第3号だけへの例外を全方式へ広げない"]
    prompt = pipeline.build_draft_prompt("例外は？", ap.AnswerIntent(), [], {},
                                         previous=draft, verified_ids=set(), issues=["範囲を直す"])
    assert "検査で却下された主張" in prompt
    assert "全方式が例外です。" in prompt and "第3号だけへの例外を全方式へ広げない" in prompt


EXCEPTIONS = """前項第１号又は第２号の規定に関わらず、次の各号に該当する案件は対象としない。
- 一 同じ仕様による部品の交換
- 二 保守契約の終了時の再契約
"""


def test_an_exclusion_label_without_actual_cases_cannot_satisfy_the_question():
    item = evidence(EXCEPTIONS, "除外")
    intent = ap.AnswerIntent(required_aspects=("例外・除外", "条件"))
    data = {"sections": [{"claims": [claim("次の案件は対象外です。", "E1", "同じ仕様による部品の交換", aspect="除外")]}]}
    draft = ap.parse_answer_draft(data)
    assert pipeline._missing_exception_items(draft, intent, set(draft.claims), [item]) == ["例外・除外"]
    model = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline("除外案件と条件は？", intent, [item], make_env(model, source_lines_for=None))
    assert "同じ仕様による部品の交換" in result.answer and "保守契約の終了時の再契約" in result.answer
    assert "第１号又は第２号" in result.answer
    assert "exception_items_missing" not in result.quality["gates"]["G3"]["codes"]
    checked = [c for stage, payload in model.calls if stage == "g2" for c in payload["claims"]]
    assert any(c["claim_id"].startswith("original_exceptions_") for c in checked)


def test_an_intervening_rule_cannot_be_attached_to_an_exclusion_list():
    item = evidence(EXCEPTIONS.replace("\n- 一", "\n追加の対象案件を示す。\n- 一"), "除外")
    assert pipeline._required_exception_groups(ap.AnswerIntent(required_aspects=("除外",)), item) == []


def test_exclusion_completion_fails_closed_when_semantic_model_rejects_the_original_list():
    data = {"sections": [{"claims": [claim("原文の対象外案件です。", "E1", "同じ仕様による部品の交換")]}]}
    model = FakeModel(drafts=[data, data], g2=lambda cs: {c["claim_id"]: ("unsupported", "other") for c in cs})
    result = pipeline.run_answer_pipeline("除外と条件は？", ap.AnswerIntent(required_aspects=("除外", "条件")),
                                          [evidence(EXCEPTIONS)], make_env(model, source_lines_for=None))
    assert result.kind == "fallback" and result.quality["state"] == ap.QUALITY_FAILED


def test_declared_unanswered_aspects_require_a_repair_even_if_the_lexical_gates_pass():
    data = {"sections": [{"claims": [claim("原文の対象外案件です。", "E1", "同じ仕様による部品の交換")]}],
            "unanswered_aspects": ["個別の申請方法"]}
    model = FakeModel(drafts=[data, data])
    result = pipeline.run_answer_pipeline("除外案件は？", ap.AnswerIntent(required_aspects=("除外",)),
                                          [evidence(EXCEPTIONS)], make_env(model, source_lines_for=None))
    assert model.count("draft") == 2
    assert "個別の申請方法" in result.quality["unconfirmedAspects"]
    assert result.quality["state"] != ap.QUALITY_PASSED


def test_a_completed_but_failed_semantic_check_is_not_reported_as_an_unfinished_check():
    gates = {gid: ap.GateResult(ap.PASSED) for gid in ap.GATE_IDS}
    gates["G2"] = ap.GateResult(ap.FAILED, ("condition_dropped",))
    quality = ap.build_quality(gates, kind="full", removed_claims=1)
    assert "意味の支持" in quality["message"] and "検査で問題" in quality["message"]
    assert "完了できません" not in quality["message"]


CHECKLIST = """## 更新申請
### 申請チェックリスト
| 項目 | 確認内容 |
|---|---|
| 期間 | 受付期間を5営業日以上確保しているか。 |
| 資格 | 新規申請の場合に限り、資格証明書を添付しているか。 |
"""


def test_original_checklist_completion_preserves_method_number_and_condition():
    item = evidence(CHECKLIST, "申請チェックリスト")
    data = {"sections": [{"claims": [claim("更新申請の確認項目を準備します。", "E1", "## 更新申請\n### 申請チェックリスト")]}]}
    model = FakeModel(drafts=[data, data])
    intent = ap.AnswerIntent(granularity="practical", required_aspects=("チェックリストの確認項目",))
    result = pipeline.run_answer_pipeline("確認項目は？", intent, [item], make_env(model, source_lines_for=None))
    assert "5営業日以上" in result.answer and "新規申請の場合に限り" in result.answer
    assert "更新申請」のチェックリスト" in result.answer
    checked = [c for stage, payload in model.calls if stage == "g2" for c in payload["claims"]]
    assert any(c["claim_id"].startswith("original_check_") for c in checked)


@pytest.mark.parametrize("failure", ["source_changed", "unsupported"])
def test_original_checklist_completion_cannot_bypass_verification(failure):
    data = {"sections": [{"claims": [claim("更新申請の確認項目を準備します。", "E1", "## 更新申請\n### 申請チェックリスト")]}]}
    model = FakeModel(drafts=[data, data], g2=(lambda cs: {c["claim_id"]: ("unsupported", "other") for c in cs})
                      if failure == "unsupported" else None)
    env = make_env(model, source_lines_for=(lambda path, sha: None) if failure == "source_changed" else None)
    result = pipeline.run_answer_pipeline("確認項目は？", ap.AnswerIntent(required_aspects=("確認項目",)),
                                          [evidence(CHECKLIST)], env)
    assert result.kind == "fallback"
    assert result.quality["state"] != ap.QUALITY_PASSED


@pytest.mark.parametrize(('source', 'answer', 'changed'), [
    ('委員の２分の１以上を外部委員とする。', '外部委員は過半数以上とする。', True),
    ('外部委員は過半数とする。', '外部委員は半数以上とする。', True),
    ('外部委員は２分の１以上とする。', '外部委員は半数以上とする。', False),
    ('外部委員は５０％以上とする。', '外部委員は過半数とする。', True),
    ('外部委員は過半数とする。', '外部委員は過半数以上とする。', False),
])
def test_half_and_strict_majority_have_different_boundaries(source, answer, changed):
    assert bool(ap.number_bounds_changed(answer, source)) is changed


def test_checklist_method_is_recovered_from_verified_original_parent_outside_chunk():
    source = CHECKLIST.splitlines()
    item = evidence('\n'.join(source[1:]), '申請チェックリスト')
    item.update(start_line=2, end_line=len(source))
    env = make_env(FakeModel(), source_lines_for=lambda path, sha: source)
    assert pipeline._bind_source_rows([item], env) == 1
    topics = ap.checklist_topics(item)
    assert all(t['scope_quote'] == '## 更新申請' for t in topics)
    draft = ap.AnswerDraft()
    pipeline._seed_checklist_topics(draft, ap.AnswerIntent(required_aspects=('確認項目',)), [item])
    scopes = [c for c in draft.claims.values() if c.kind == 'scope']
    assert len(scopes) == 1 and '更新申請' in scopes[0].text
    g1 = ap.check_g1(draft, {item['evidence_id']: item}, source_lines_for=env.source_lines_for)
    assert all(result.status == ap.PASSED for result in g1.values())


def test_sibling_checklists_keep_distinct_method_scopes():
    item = evidence(CHECKLIST + CHECKLIST.replace('更新申請', '特別申請'), '申請チェックリスト')
    topics = ap.checklist_topics(item)
    assert [t['scope_quote'] for t in topics] == ['## 更新申請'] * 2 + ['## 特別申請'] * 2
    draft = ap.AnswerDraft()
    assert pipeline._seed_checklist_topics(draft, ap.AnswerIntent(required_aspects=('確認項目',)), [item]) == 4
    assert len(draft.sections) == 2
    assert '更新申請' in draft.claims[draft.sections[0].scope_claim_id].text
    assert '特別申請' in draft.claims[draft.sections[1].scope_claim_id].text


def test_populated_section_does_not_truncate_original_checklist_completion():
    data = {'sections': [{'heading': 'チェックリスト', 'claims': [
        claim('準備します。', 'E1', '受付期間を5営業日以上確保しているか。') for _ in range(7)]}]}
    draft = ap.parse_answer_draft(data)
    assert pipeline._seed_checklist_topics(draft, ap.AnswerIntent(required_aspects=('確認項目',)),
                                          [evidence(CHECKLIST)]) == 2
    assert [len(s.claim_ids) for s in draft.sections] == [7, 2]
    assert draft.sections[0].scope_claim_id == draft.sections[1].scope_claim_id


def test_nested_original_quotes_do_not_hide_real_question_fragments():
    text = '配分では、「「実施体制」の点数は適切か。」を確認します。'
    assert not ap._QUESTION_FRAGMENT_RE.search(ap._without_quoted_spans(text))
    assert ap._QUESTION_FRAGMENT_RE.search(ap._without_quoted_spans(text + '他に必要か。'))
    unclosed = '「確認が必要か。'
    assert ap._without_quoted_spans(unclosed) == unclosed


def test_verified_originals_remove_only_summaries_of_the_same_source_ranges_and_scope():
    original = ap.DraftClaim('original_check_E1_0_0', '原文の完全な確認内容', refs=[ap.EvidenceRef('E1', '確認内容')])
    row = {'evidence_id': 'E1', 'path': 'notice/application.md', 'source_sha256': 'b' * 64,
           'start_line': 4, 'end_line': 6}
    original.ranges = [row]
    summary = ap.DraftClaim('c1', '確認内容の要約', ranges=[dict(row, start_line=5, end_line=5)])
    different = ap.DraftClaim('c2', '別の説明', ranges=[dict(row, start_line=7, end_line=7)])
    rejected = ap.DraftClaim('c3', '不支持の説明', ranges=[row])
    draft = ap.AnswerDraft(claims={c.claim_id: c for c in (original, summary, different, rejected)},
                           sections=[ap.DraftSection('確認項目', None, [c.claim_id for c in (original, summary, different, rejected)])])
    verified = {'original_check_E1_0_0', 'c1', 'c2'}
    assert pipeline._prune_original_duplicates(draft, verified) == 1
    assert 'c1' not in draft.claims and 'c1' not in verified
    assert 'c2' in draft.claims and 'c3' in draft.claims
    # Unverified originals must not remove a supported model claim.
    draft.claims['c1'] = summary
    verified = {'c1'}
    assert pipeline._prune_original_duplicates(draft, verified) == 0


def test_identical_verified_lists_are_shown_once_but_different_conditions_are_kept():
    row = {'evidence_id': 'E1', 'path': 'notice/application.md', 'source_sha256': 'b' * 64,
           'start_line': 4, 'end_line': 6}
    def listed(cid, lines):
        return ap.DraftClaim(cid, '確認書類の一覧', source_list={'id':'E1','group_id':'L1'},
                             list_lines=lines, ranges=[row])
    claims = [listed('c1',['- 申請書（更新のみ）']), listed('c2',['- 申請書（更新のみ）']),
              listed('c3',['- 申請書（新規のみ）'])]
    draft = ap.AnswerDraft(claims={c.claim_id:c for c in claims},
                           sections=[ap.DraftSection('書類',None,[c.claim_id for c in claims])])
    verified = set(draft.claims)
    assert pipeline._prune_original_duplicates(draft,verified) == 1
    assert draft.sections[0].claim_ids == ['c1','c3']


@pytest.mark.parametrize(('source','answer','changed'), [
    ('コストを２割削減する。','コストを約2割削減する。',True),
    ('コストを約２割削減する。','コストを約2割削減する。',False),
    ('概算所要見込額が2000万円である。','金額は約2000万円である。',False),
    ('概算所要見込額が2000万円以上の場合。','金額は約2000万円以上の場合。',True),
    ('人数は10人前後である。','人数は約10人である。',False),
    ('受付期間は7営業日とする。','受付期間は7営業日とする。',False),
])
def test_exact_quantities_must_not_become_approximate(source,answer,changed):
    assert bool(ap.approximate_numbers_added(answer,source)) is changed


def test_deliberation_question_prompt_does_not_add_unrequested_case_thresholds():
    prompt = pipeline.build_draft_prompt('委員会が審議する事項と設置背景は？',ap.AnswerIntent(),[],{})
    assert '対象案件の金額基準や契約種別は列挙せず' in prompt
    requested = pipeline.build_draft_prompt('委員会が審議する事項と対象案件の金額は？',ap.AnswerIntent(),[],{})
    assert '対象案件の金額基準や契約種別は列挙せず' not in requested


def test_verified_before_after_role_comparison_does_not_remain_declared_unanswered():
    text = '受付委員会は受付前に審査し、監視委員会は受付後に審議します。'
    aspect = '別の会議体の役割との区別'
    data = {'conclusion':claim(text,'E1',text,aspect=aspect), 'sections':[], 'unanswered_aspects':[aspect]}
    intent = ap.AnswerIntent(target='受付委員会',required_aspects=(aspect,))
    model = FakeModel(drafts=[data,data])
    result = pipeline.run_answer_pipeline('二つの委員会の役割は？',intent,[evidence(text,'役割')],
                                          make_env(model,source_lines_for=None))
    assert result.quality['state'] == ap.QUALITY_PASSED
    assert not result.quality['unconfirmedAspects']
    assert model.count('draft') == 1
    # Merely naming two organizations is not an account of their respective roles.
    named = ap.DraftClaim('c1','受付委員会と監視委員会を記載します。')
    assert not ap._aspect_covered(aspect,[named])


def test_rejected_comparison_cannot_clear_declared_unanswered_aspect():
    text = '受付委員会は受付前に審査し、監視委員会は受付後に審議します。'
    aspect = '別の会議体の役割との区別'
    data = {'sections':[{'claims':[claim(text,'E1',text,aspect=aspect)]}], 'unanswered_aspects':[aspect]}
    model = FakeModel(drafts=[data,data],g2=lambda cs:{c['claim_id']:('unsupported','other') for c in cs})
    result = pipeline.run_answer_pipeline('役割の区別は？',ap.AnswerIntent(required_aspects=(aspect,)),
                                          [evidence(text,'役割')],make_env(model,source_lines_for=None))
    assert result.quality['state'] != ap.QUALITY_PASSED
    assert aspect in result.quality['unconfirmedAspects']
