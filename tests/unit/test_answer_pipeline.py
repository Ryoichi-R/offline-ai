"""回答方針v2の有限な流れ: 草稿→G1→G2→G3/G4→再生成/再探索→表示/fallback。偽モデルで検証する。"""

import copy

import pytest

import answer_pipeline as pipeline
import answer_policy as ap
from answer_fixtures import (
    QUERY,
    SOURCE_SHA,
    FakeModel,
    claim,
    good_draft,
    make_env,
    make_evidence,
)

INTENT = ap.extract_explicit_intent(QUERY)


def complete_draft():
    """必須観点（背景・別の会議体）の両方に答える自然な草稿。"""
    data = good_draft()
    data["sections"][2]["claims"].append(
        claim(
            "技術評価会の開催方法は、別紙のチェック項目で確認する対象であり、委員会自身の審査事項ではありません。",
            "E3",
            "技術評価会は、原則、会議形式で開催することになっているか。",
            aspect="別の会議体の役割との区別",
        )
    )
    return data


def run(model, **kwargs):
    env = kwargs.pop("env", None) or make_env(model)
    return pipeline.run_answer_pipeline(QUERY, INTENT, make_evidence(), env, **kwargs), env


def draft_calls(model):
    return model.count("draft")


# --- 正常系 ---------------------------------------------------------------------


def test_clean_first_draft_passes_with_one_draft_call_and_one_check_call():
    model = FakeModel(drafts=[complete_draft()])
    result, env = run(model)
    assert result.kind == "full"
    assert result.quality["state"] == "passed"
    assert [g["status"] for g in result.quality["gates"].values()] == ["passed"] * 4
    assert (model.count("draft"), model.count("g2"), model.count("g34")) == (1, 1, 0)
    lines = result.answer.splitlines()
    assert lines[0].startswith("委員会は、設備の更新前に計画を審査し")
    assert "## 審査する内容" in result.answer and "## 設置の背景との区別" in result.answer
    # 直接回答（E2）を中心に使い、チェック表(E3)は区別の説明だけに使う。
    assert set(result.evidence_used) == {"E1", "E2", "E3"}
    assert result.diagnostics["calls"]["check_calls"] == 1
    assert set(env.clock.summary()) >= {"generate", "verify", "finish"}


def test_practical_hint_is_added_for_overview_only_when_detail_evidence_was_not_expanded():
    data = good_draft()
    data["sections"][2]["claims"].append(
        claim(
            "別の会議体（技術評価会）の役割は、この委員会の審査事項とは別です。",
            "E2",
            "委員会において審査する案件は、次のとおりとする。",
            aspect="別の会議体の役割との区別",
        )
    )
    result, _ = run(FakeModel(drafts=[data]))
    assert result.answer.rstrip().endswith(pipeline.PRACTICAL_HINT)
    result2, _ = run(FakeModel(drafts=[complete_draft()]))
    assert pipeline.PRACTICAL_HINT not in result2.answer


# --- 原文・意味の検査 ---------------------------------------------------------------


def wrong_actor_g2(claims):
    verdicts = {}
    for item in claims:
        if "技術評価会が" in item["claim"]:
            verdicts[item["claim_id"]] = ("unsupported", "subject_mismatch")
        else:
            verdicts[item["claim_id"]] = ("supported", "ok")
    return verdicts


def test_wrong_subject_claim_is_removed_and_the_corrected_draft_is_adopted():
    bad = complete_draft()
    bad["sections"][0]["claims"].append(
        claim("技術評価会が費用削減の可否を審査します。", "E2", "費用削減の可否")
    )
    model = FakeModel(drafts=[bad, complete_draft()], g2=wrong_actor_g2)
    result, _ = run(model)
    assert draft_calls(model) == 2
    assert "技術評価会が費用削減" not in result.answer
    assert result.quality["state"] == "passed"
    second_prompt = [u for name, u in model.calls if name == "draft"][1]
    assert "主体（誰が行うか）を原文どおりにしてください" in second_prompt


def test_paraphrase_is_supported_but_the_same_words_with_another_subject_are_not():
    paraphrase = complete_draft()
    paraphrase["sections"][0]["claims"][0]["text"] = "費用を下げられるかどうかを確認します。"
    result, _ = run(FakeModel(drafts=[paraphrase], g2=wrong_actor_g2))
    assert "費用を下げられるかどうかを確認します。" in result.answer
    assert result.quality["state"] == "passed"


def test_invented_quote_fails_g1_without_a_semantic_call_for_that_claim_and_is_regenerated():
    bad = complete_draft()
    bad["sections"][0]["claims"].append(
        claim("存在しない規定があります。", "E2", "原文に存在しない引用の一節です")
    )
    model = FakeModel(drafts=[bad, complete_draft()])
    result, _ = run(model)
    first_g2 = [p for name, p in model.calls if name == "g2"][0]
    assert all("存在しない規定" not in c["claim"] for c in first_g2["claims"])
    assert draft_calls(model) == 2
    assert result.quality["state"] == "passed"
    assert "一字一句そのまま写した一節" in [u for n, u in model.calls if n == "draft"][1]


def test_uncertain_claims_are_never_adopted_and_the_answer_is_marked_for_review():
    def uncertain(claims):
        return {
            c["claim_id"]: (
                ("uncertain", "other") if "仕様が妥当" in c["claim"] else ("supported", "ok")
            )
            for c in claims
        }

    model = FakeModel(drafts=[complete_draft(), complete_draft()], g2=uncertain)
    result, _ = run(model)
    assert "仕様が妥当かどうか" not in result.answer
    assert result.quality["state"] == "needs_review"
    assert result.quality["label"] == "要確認"
    assert result.quality["removedClaims"] == 1
    assert result.quality["gates"]["G2"]["status"] == "uncertain"


def test_answer_with_all_claims_unsupported_becomes_the_common_fallback():
    model = FakeModel(
        drafts=[complete_draft(), complete_draft()],
        g2=lambda claims: {c["claim_id"]: ("unsupported", "not_in_source") for c in claims},
    )
    result, _ = run(model)
    assert result.kind == "fallback"
    assert result.quality["state"] == "failed"
    assert result.answer.startswith("回答を作成できませんでした。")
    assert "## 参考の原文" in result.answer
    assert result.answer.count("[E2]") == 1


# --- 有限性・不正JSON・時間 ------------------------------------------------------------


def test_two_unusable_drafts_end_in_fallback_after_exactly_two_generation_calls():
    model = FakeModel(drafts=[ValueError("bad json"), {"conclusion": "x"}])
    result, _ = run(model)
    assert result.kind == "fallback"
    assert draft_calls(model) == 2
    assert model.count("g2") == 0
    assert [a["draft"] for a in result.diagnostics["attempts"]] == [
        "call_failed",
        "schema:sections_missing",
    ]


def test_a_schema_error_consumes_a_call_and_the_retry_can_still_succeed():
    model = FakeModel(drafts=[{"sections": "no"}, complete_draft()])
    result, _ = run(model)
    assert draft_calls(model) == 2
    assert result.quality["state"] == "passed"


def test_no_time_means_not_run_and_never_a_pass():
    model = FakeModel(drafts=[complete_draft()])
    env = make_env(model)
    env.clock = ap.StageClock(300, "normal", remaining_fn=lambda: 0.0)
    result, _ = run(model, env=env)
    assert result.kind == "fallback" and result.quality["state"] == "failed"
    assert model.calls == []
    assert result.diagnostics["calls"]["counts"]["draft"] == {"not_run": 1}


def test_semantic_check_skipped_for_lack_of_time_is_not_run_and_the_draft_is_not_adopted():
    state = {"draft_done": False}

    class Model(FakeModel):
        def __call__(self, system, user, *, timeout):
            value = super().__call__(system, user, timeout=timeout)
            state["draft_done"] = True
            return value

    model = Model(drafts=[complete_draft()])
    env = make_env(model)
    env.clock = ap.StageClock(
        300, "normal", remaining_fn=lambda: 50.0 if not state["draft_done"] else 0.0
    )
    result, _ = run(model, env=env)
    assert result.kind == "fallback"
    assert model.count("g2") == 0
    assert result.diagnostics["calls"]["counts"]["g2"] == {"not_run": 1}


def big_evidence(count=1, size=1400):
    text = "\n".join(f"第{i}項　" + "あ" * 40 for i in range(size // 50))
    return [
        {
            "evidence_id": f"E{i + 1}",
            "path": "notice/sample-equipment.md",
            "source_sha256": SOURCE_SHA,
            "heading": "大きな節",
            "start_line": 1,
            "end_line": size // 50,
            "excerpt": text,
            "conditions": [],
            "exceptions": [],
            "references": [],
        }
        for i in range(count)
    ]


def test_semantic_batches_are_capped_at_two_per_draft_and_unchecked_claims_are_not_adopted(
    monkeypatch,
):
    evidence = big_evidence()
    quote = "第3項　" + "あ" * 40
    many = {
        "conclusion": claim("大きな節の内容です。", "E1", quote),
        "sections": [
            {
                "heading": f"節{s}",
                "claims": [claim(f"主張{s}-{i}です。", "E1", quote) for i in range(5)],
            }
            for s in range(5)
        ],
    }
    model = FakeModel(drafts=[many, many])
    env = make_env(model)
    env.source_lines_for = None
    result = pipeline.run_answer_pipeline(QUERY, INTENT, evidence, env)
    per_draft = [p for n, p in model.calls if n == "g2"]
    assert len(per_draft) <= 2 * 2  # 1案あたり最大2batch、最大2案
    assert env.ledger.check_count() <= ap.MAX_CHECK_CALLS
    checked = {c["claim_id"] for batch in per_draft for c in batch["claims"]}
    assert len(checked) < 26  # 上限を超えた主張は意味照合されない
    # 検査していない主張は表示文に出ない。
    shown = [line for line in result.answer.splitlines() if line.startswith("- 主張")]
    assert shown and len(shown) <= len(checked)
    assert result.quality["state"] != "passed"


def test_a_control_exception_is_never_swallowed():
    class Cancelled(Exception):
        pass

    def boom(system, user, *, timeout):
        raise Cancelled()

    env = make_env(boom)
    env.is_control_error = lambda exc: isinstance(exc, Cancelled)
    with pytest.raises(Cancelled):
        pipeline.run_answer_pipeline(QUERY, INTENT, make_evidence(), env)


# --- 再探索（最大1回） --------------------------------------------------------------


def test_missing_aspect_triggers_exactly_one_reexploration_then_passes():
    asked = []

    def reexplore(missing):
        asked.append(list(missing))
        return [
            {
                "evidence_id": "E4",
                "path": "notice/sample-equipment.md",
                "source_sha256": SOURCE_SHA,
                "heading": "別紙１　設備更新　設備更新委員会チェックリスト",
                "start_line": make_evidence()[2]["start_line"],
                "end_line": make_evidence()[2]["end_line"],
                "excerpt": make_evidence()[2]["excerpt"],
                "conditions": [],
                "exceptions": [],
                "references": [],
            }
        ]

    model = FakeModel(drafts=[good_draft(), complete_draft()])
    env = make_env(model, reexplore=reexplore)
    result = pipeline.run_answer_pipeline(QUERY, INTENT, make_evidence(), env)
    assert asked == [["別の会議体の役割との区別"]]
    assert result.diagnostics["reexplored"] is True and result.diagnostics["reexplored_items"] == 1
    assert result.quality["state"] == "passed"
    second = [u for n, u in model.calls if n == "draft"][1]
    assert "[E4]" in second  # 追加の根拠が次の草稿へ渡る


def test_without_reexploration_the_unanswered_aspect_is_shown_as_unconfirmed():
    model = FakeModel(drafts=[good_draft(), good_draft()])
    result, _ = run(model)
    assert result.kind == "partial"
    assert result.quality["state"] == "partial"
    assert result.quality["unconfirmedAspects"] == ["別の会議体の役割との区別"]
    assert "## 確認できなかった観点" in result.answer
    assert result.answer.count("- 別の会議体の役割との区別") == 1


# --- 文章品質の再生成 ------------------------------------------------------------------


def test_repetitive_draft_is_rewritten_once_with_the_problem_named_then_flagged_if_unchanged():
    from test_answer_policy import repetitive_draft

    model = FakeModel(drafts=[repetitive_draft(), repetitive_draft()])
    result, _ = run(model)
    assert draft_calls(model) == 2
    assert "同じ書き出しの文が繰り返されています" in [u for n, u in model.calls if n == "draft"][1]
    assert result.quality["gates"]["G4"]["status"] == "failed"
    assert "repeated_prefix" in result.quality["gates"]["G4"]["codes"]
    assert result.quality["state"] != "passed"


def test_overview_answer_built_only_from_the_checklist_is_rewritten_toward_the_direct_rule():
    from_table = {
        "conclusion": claim("図面の添付などを確認します。", "E3", "図面が添付されているか。"),
        "sections": [
            {
                "heading": "確認項目",
                "claims": [
                    claim(f"項目{i}を確認します。{'あ' * i}", "E3", "図面が添付されているか。")
                    for i in range(5)
                ],
            }
        ],
    }
    model = FakeModel(drafts=[from_table, complete_draft()])
    result, _ = run(model)
    assert draft_calls(model) == 2
    assert (
        "規定の根拠（role=direct_answer）を使っていません"
        in [u for n, u in model.calls if n == "draft"][1]
    )
    assert result.quality["state"] == "passed"


def test_the_better_attempt_is_chosen_when_the_rewrite_is_worse():
    worse = {
        "conclusion": claim("概要です。", "E2", "費用削減の可否"),
        "sections": [{"claims": [claim("確認します。", "E9", "x")]}],
    }
    model = FakeModel(drafts=[good_draft(), worse])
    result, _ = run(model)
    assert result.kind in {"partial", "full"}
    assert "費用削減が可能かどうか" in result.answer


def test_an_attempt_with_displayable_claims_beats_one_that_passes_more_gates_but_shows_nothing():
    # P4実資料(deep): 初稿は多数の主張が検査を通ったが、一部が不支持(G2不合格)で結論もなかった。
    # 再生成案は範囲の主張だけが通りG2は合格。旧採点は通過ゲート数を優先し、表示できる説明の
    # ない再生成案を選んで原文表示(fallback)にした。
    first = (
        good_draft()
    )  # 別の会議体の観点が欠ける(G3不合格)。支持主張なしの案は欠落観点0件と数えられる
    first["conclusion"] = claim(
        "結論です。", "E9", "存在しない引用"
    )  # G1で落ち、結論なし(G4不合格)
    first["sections"][0]["claims"].append(
        claim("技術評価会が費用削減の可否を審査します。", "E2", "費用削減の可否")
    )
    scope_only = {
        "conclusion": claim("結論です。", "E9", "存在しない引用"),
        "sections": [
            {
                "heading": "範囲",
                "scope": {
                    "text": "設備更新委員会の審査について",
                    "evidence": [{"id": "E2", "quote": "委員会において審査する内容"}],
                },
                "claims": [claim("審査します。", "E9", "存在しない引用")],
            }
        ],
    }
    model = FakeModel(drafts=[first, scope_only], g2=wrong_actor_g2)
    result, _ = run(model)
    assert draft_calls(model) == 2
    assert result.kind != "fallback"
    assert "費用削減が可能かどうか" in result.answer
    assert "技術評価会が費用削減" not in result.answer
    assert result.quality["state"] == "needs_review"


def test_an_answer_whose_quotes_were_corrected_to_the_source_is_never_marked_as_checked():
    data = complete_draft()
    # 括弧書きを省いた引用（原文の一部を省いただけ）。原文の範囲へ補正して照合する。
    data["sections"][1]["claims"][1]["evidence"][0]["quote"] = (
        "概算費用３００万円以上の設備改修（緊急修繕は除く）"
    )
    model = FakeModel(drafts=[data, data])
    result, _ = run(model)
    assert "設備改修が対象ですが、緊急修繕は除きます" in result.answer
    assert result.quality["correctedQuotes"] == 1
    assert result.quality["state"] == "needs_review"
    # 意味照合には補正後の実際の原文を渡す。
    quotes = [
        q
        for name, payload in model.calls
        if name == "g2"
        for c in payload["claims"]
        for q in c["cited_quotes"]
    ]
    assert "- 二　概算費用３００万円以上の設備改修（緊急修繕を除く。）" in quotes


def test_a_claim_with_a_number_absent_from_its_source_is_rejected_before_the_semantic_check():
    # P4実資料: G2が数値・語の改変を見逃した。原文に無い数値は意味照合を待たず不支持にする。
    data = complete_draft()
    data["sections"][1]["claims"][0]["text"] = "概算費用が５００万円以上の設備更新が対象です。"
    model = FakeModel(drafts=[data, complete_draft()])
    result, _ = run(model)
    first_g2 = [payload for name, payload in model.calls if name == "g2"][0]
    assert all("５００万円" not in c["claim"] for c in first_g2["claims"])
    assert "number_changed" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]
    assert "５００万円" not in result.answer


def test_terms_absent_from_the_source_are_given_to_the_semantic_check():
    data = complete_draft()
    # 曖昧な基準は前段で拒否するため、ここは原文外の語をG2へ渡す役割だけ検証する。
    data["sections"][1]["claims"][0]["text"] = "公共インフラの更新が対象です。"
    model = FakeModel(drafts=[data, complete_draft()])
    run(model)
    payload = [p for name, p in model.calls if name == "g2"][0]
    flagged = next(c for c in payload["claims"] if "インフラ" in c["claim"])
    assert "インフラ" in flagged["terms_not_in_source"]
    assert "terms_not_in_source" in pipeline.CLAIM_VERIFY_SYSTEM


def test_a_split_statute_name_is_rejected_and_scope_elements_are_marked_for_the_semantic_check(
    monkeypatch,
):
    data = complete_draft()
    model = FakeModel(drafts=[data, complete_draft()])
    run(model)
    payload = [p for name, p in model.calls if name == "g2"][0]
    kinds = {c["kind"] for c in payload["claims"]}
    assert {"scope", "claim", "conclusion"} <= kinds
    assert "kind が scope" in pipeline.CLAIM_VERIFY_SYSTEM
    monkeypatch.setattr(
        ap,
        "compound_names_split",
        lambda claim, source: ["X及びY令"] if "１，０００万円" in claim else [],
    )
    model = FakeModel(drafts=[complete_draft(), complete_draft()])
    result, _ = run(model)
    assert "name_changed" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]
    assert all(
        "１，０００万円" not in c["claim"] for _, p in model.calls if _ == "g2" for c in p["claims"]
    )
    # 再生成では名称を原文どおりに書くよう具体的に指示する。
    second = [u for name, u in model.calls if name == "draft"][1]
    assert "「A及びB令」を「AやB令」のように分けない" in second
    assert "名称は原文どおり「X及びY令」と書く" in second


def reject_scope(code):
    def judge(claims):
        return {
            c["claim_id"]: (("unsupported", code) if c["kind"] == "scope" else ("supported", "ok"))
            for c in claims
        }

    return judge


def test_a_scope_rejected_as_too_broad_is_hidden_but_its_supported_claims_stay():
    # P4実資料: 「〜の比較」というscopeが範囲の拡大と判定され、検査済みの区別の主張が節ごと消えた。
    model = FakeModel(
        drafts=[complete_draft(), complete_draft()], g2=reject_scope("scope_overreach")
    )
    result, _ = run(model)
    assert "１，０００万円以上の設備更新が対象です" in result.answer
    assert "審査する案件は、金額などで次のとおり定められています。" not in result.answer
    assert result.diagnostics["attempts"][0]["detached_scopes"] == 1
    assert result.quality["state"] != "passed"


def test_a_scope_is_not_detached_for_other_errors_or_when_a_claim_of_its_section_fails():
    model = FakeModel(
        drafts=[complete_draft(), complete_draft()], g2=reject_scope("subject_mismatch")
    )
    result, _ = run(model)
    assert "１，０００万円以上の設備更新が対象です" not in result.answer
    assert result.diagnostics["attempts"][0]["detached_scopes"] == 0

    def scope_and_one_claim(claims):
        return {
            c["claim_id"]: (
                ("unsupported", "scope_overreach")
                if c["kind"] == "scope" or "３００万円" in c["claim"]
                else ("supported", "ok")
            )
            for c in claims
        }

    result, _ = run(FakeModel(drafts=[complete_draft(), complete_draft()], g2=scope_and_one_claim))
    assert "１，０００万円以上の設備更新が対象です" not in result.answer


def test_a_claim_that_drops_the_qualifier_of_its_cited_item_is_rejected_before_the_semantic_check():
    data = complete_draft()
    data["sections"][1]["claims"][1]["text"] = "概算費用が３００万円以上の設備改修が対象です。"
    model = FakeModel(drafts=[data, complete_draft()])
    result, _ = run(model)
    first_g2 = [payload for name, payload in model.calls if name == "g2"][0]
    assert all(
        c["claim"] != "概算費用が３００万円以上の設備改修が対象です。" for c in first_g2["claims"]
    )
    assert "condition_dropped" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]
    second = [u for name, u in model.calls if name == "draft"][1]
    assert "原文の条件を落とさず" in second
    # どの限定を落としたかを具体的に示す（一般的な指示だけでは同じ誤りを繰り返した）。
    assert "原文の限定「(緊急修繕を除く。)」を落とさない" in second


def test_extraction_notes_are_marked_as_not_quotable_in_the_draft_prompt():
    evidence = make_evidence()
    evidence[1]["conditions"] = ["緊急修繕を除く"]
    prompt = pipeline.build_draft_prompt(QUERY, INTENT, evidence, {})
    assert "条件（抽出メモ。quoteには使わない）: 緊急修繕を除く" in prompt


def test_a_listed_claim_keeps_the_qualifiers_of_its_items_and_the_semantic_check_sees_them():
    data = complete_draft()
    data["sections"][1]["claims"] = [
        {
            "text": "審査する案件は、金額で次のとおり定められています。",
            "source_list": {
                "id": "E2",
                "from": "概算費用１，０００万円以上の設備更新",
                "to": "概算費用３００万円以上の設備改修（緊急修繕を除く。）",
            },
        },
    ]
    model = FakeModel(drafts=[data, data])
    result, _ = run(model)
    assert "> - 二　概算費用３００万円以上の設備改修（緊急修繕を除く。）" in result.answer
    listed = [
        c
        for name, p in model.calls
        if name == "g2"
        for c in p["claims"]
        if c.get("listed_source_items")
    ]
    assert (
        listed and "- 一　概算費用１，０００万円以上の設備更新" in listed[0]["listed_source_items"]
    )
    assert (
        "source_list" in pipeline.DRAFT_SYSTEM
        and "listed_source_items" in pipeline.CLAIM_VERIFY_SYSTEM
    )
    # 原文一覧が正しくても説明文による範囲拡大は拒否する。
    data["sections"][1]["claims"][0]["text"] = (
        "概算費用３００万円以上の設備改修などが、次のとおり対象です。"
    )
    result, _ = run(FakeModel(drafts=[data, data]))
    assert "概算費用３００万円以上の設備改修などが、次のとおり対象です。" not in result.answer
    assert "condition_dropped" in result.quality["gates"]["G2"]["codes"]


def test_a_changed_unit_word_is_rejected_with_a_specific_fix():
    data = complete_draft()
    data["sections"][1]["claims"][0]["text"] = "概算費用が１，０００万件以上の設備更新が対象です。"
    model = FakeModel(drafts=[data, complete_draft()])
    result, _ = run(model)
    assert "number_changed" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]
    assert (
        "数値に付く単位を原文どおりにする（1000万件以上）"
        in [u for n, u in model.calls if n == "draft"][1]
    )


def lexical_miss_draft():
    """別の会議体の観点に答えているが、観点の語の一致が足りない主張を持つ草稿。"""
    data = good_draft()
    data["sections"][2]["claims"].append(
        claim(
            "技術評価会は、委員会とは役割が異なります。",
            "E3",
            "技術評価会は、原則、会議形式で開催することになっているか。",
        )
    )
    return data


GOOD_QUALITY = {
    "answers_question": "yes",
    "granularity_fit": "fit",
    "missing_aspects": [],
    "structure_issues": [],
}


def test_an_aspect_missed_only_by_wording_is_confirmed_by_one_evaluation_call():
    # P4実資料: 結論で答えた観点が「確認できなかった観点」に表示された（語の一致だけの判定）。
    model = FakeModel(drafts=[lexical_miss_draft(), lexical_miss_draft()], quality=GOOD_QUALITY)
    result, _ = run(model)
    assert model.count("g34") == 1
    assert result.quality["unconfirmedAspects"] == []
    assert result.diagnostics["attempts"][0]["aspects_reconciled"] == 1


def test_the_evaluation_cannot_clear_an_aspect_it_lists_or_that_no_claim_mentions():
    listed = dict(GOOD_QUALITY, missing_aspects=["別の会議体との役割の区別"])
    model = FakeModel(drafts=[lexical_miss_draft(), lexical_miss_draft()], quality=listed)
    result, _ = run(model)
    assert "別の会議体の役割との区別" in result.quality["unconfirmedAspects"]
    # 観点の語がどの検査済み主張にも無ければ、評価が挙げなくても答えたとはみなさない。
    model = FakeModel(drafts=[good_draft(), good_draft()], quality=GOOD_QUALITY)
    result, _ = run(model)
    assert "別の会議体の役割との区別" in result.quality["unconfirmedAspects"]


def scope_section(data):
    return next(s for s in data["sections"] if s.get("scope"))


def test_a_scope_whose_only_problem_is_its_quote_is_carried_by_the_supported_claims_of_its_section():
    # P4実資料(deep): 節のscopeの引用ずれだけで、検査を通った審査内容の主張が節ごと消えた。
    # G2は主張をscope付きで照合するので、節の主張がすべて支持ならscopeも検査済みとして表示する。
    data = complete_draft()
    scope_section(data)["scope"]["evidence"][0]["quote"] = (
        "委員会において審査する案件は次のとおりとする等の規定"
    )
    model = FakeModel(drafts=[data, data])
    result, _ = run(model)
    assert "１，０００万円以上の設備更新が対象です" in result.answer
    assert "審査する案件は、金額などで次のとおり定められています。 [E2]" in result.answer
    applicable = [
        c["applicable_scope"] for _, payload in model.calls if _ == "g2" for c in payload["claims"]
    ]
    assert "審査する案件は、金額などで次のとおり定められています。" in applicable
    # G1の不合格は記録に残り、全ゲート合格とはしない。
    assert result.quality["state"] != "passed"
    assert result.diagnostics["attempts"][0]["inherited_scopes"] == 1


def test_a_scope_is_not_carried_when_a_claim_of_its_section_is_unsupported_or_it_adopted_an_instruction():
    data = complete_draft()
    scope_section(data)["scope"]["evidence"][0]["quote"] = (
        "委員会において審査する案件は次のとおりとする等の規定"
    )

    def reject_amounts(claims):
        return {
            c["claim_id"]: (
                ("unsupported", "scope_overreach")
                if "１，０００万円" in c["claim"]
                else ("supported", "ok")
            )
            for c in claims
        }

    result, _ = run(FakeModel(drafts=[data, data], g2=reject_amounts))
    assert (
        "設備改修が対象ですが" not in result.answer
    )  # 節の他の主張も範囲が成り立たないため出さない
    assert result.diagnostics["attempts"][0]["inherited_scopes"] == 0
    draft = ap.parse_answer_draft(data)
    sid = next(s.scope_claim_id for s in draft.sections if s.scope_claim_id)
    g1 = {
        sid: ap.GateResult(
            ap.FAILED, ("quote_not_in_evidence", "source_instruction_adopted"), "claim"
        )
    }
    g2 = {cid: ap.GateResult(ap.PASSED, (), "claim") for cid in draft.claims if cid != sid}
    assert pipeline._inherit_scopes(draft, g1, g2) == set()


# --- G3/G4評価（構造で判定できないときだけ） -----------------------------------------------


def undetermined_draft():
    return {
        "conclusion": claim(
            "図面の添付などを確認する項目があります。", "E3", "図面が添付されているか。"
        ),
        "sections": [
            {
                "heading": "確認項目",
                "claims": [
                    claim("保守契約の期間が適切かを確認します。", "E3", "保守契約の期間は適切か。")
                ],
            }
        ],
    }


def test_llm_quality_evaluation_runs_once_only_when_structure_cannot_decide():
    intent = ap.AnswerIntent(target_terms=("設備更新委員会",))
    evidence = [e for e in make_evidence() if e["evidence_id"] == "E3"]
    quality = {
        "answers_question": "yes",
        "granularity_fit": "fit",
        "missing_aspects": [],
        "structure_issues": [],
    }
    model = FakeModel(drafts=[undetermined_draft()], quality=quality)
    env = make_env(model)
    result = pipeline.run_answer_pipeline("確認項目を教えて", intent, evidence, env)
    assert model.count("g34") == 1
    assert result.quality["gates"]["G3"]["status"] == "passed"

    failing = FakeModel(
        drafts=[undetermined_draft(), undetermined_draft()],
        quality={
            "answers_question": "no",
            "granularity_fit": "fit",
            "missing_aspects": [],
            "structure_issues": [],
        },
    )
    result2 = pipeline.run_answer_pipeline("確認項目を教えて", intent, evidence, make_env(failing))
    assert result2.quality["gates"]["G3"]["status"] == "failed"
    assert "llm_answers_no" in result2.quality["gates"]["G3"]["codes"]

    broken = FakeModel(drafts=[undetermined_draft()])  # 評価器が壊れたJSON/例外を返す
    result3 = pipeline.run_answer_pipeline("確認項目を教えて", intent, evidence, make_env(broken))
    assert result3.quality["gates"]["G3"]["status"] == "uncertain"
    assert result3.quality["state"] == "needs_review"


# --- 意図計画（最大1呼出し） ---------------------------------------------------------------


def test_intent_plan_uses_one_call_and_falls_back_to_explicit_intent_on_failure():
    explicit = ap.extract_explicit_intent(QUERY)
    ok = FakeModel(
        intent={"granularity": "overview", "target": "設備更新委員会", "required_aspects": []}
    )
    planned = pipeline.plan_intent(QUERY, explicit, make_env(ok))
    assert planned.source == "llm" and ok.count("intent") == 1

    broken = FakeModel(intent=None)
    env = make_env(broken)
    failed = pipeline.plan_intent(QUERY, explicit, env)
    assert failed.source == "plan_failed"
    assert failed.granularity == "overview" and failed.required_aspects == explicit.required_aspects
    assert env.ledger.count("intent") == 1  # 失敗も1回として数え、追加retryしない

    disabled = pipeline.plan_intent(QUERY, explicit, make_env(broken), enabled=False)
    assert disabled is explicit

    no_time = make_env(ok)
    no_time.clock = ap.StageClock(300, "normal", remaining_fn=lambda: 0.0)
    assert pipeline.plan_intent(QUERY, explicit, no_time) is explicit


# --- 通常回答の根拠形式 ---------------------------------------------------------------------


def test_evidence_from_matches_builds_the_shared_evidence_shape():
    items = pipeline.evidence_from_matches(
        [
            {
                "path": "a\\b.md",
                "snippet": "本文",
                "file_sha256": "f" * 64,
                "heading": "見出し",
                "start_line": 3,
                "end_line": 5,
            },
            {"path": "c.md", "text": "別本文"},
        ]
    )
    assert [i["evidence_id"] for i in items] == ["E1", "E2"]
    assert (
        items[0]["path"] == "a/b.md"
        and items[0]["excerpt"] == "本文"
        and items[0]["source_sha256"] == "f" * 64
    )
    assert items[1]["start_line"] == 1 and items[1]["excerpt"] == "別本文"


def test_pipeline_does_not_mutate_the_callers_evidence():
    evidence = make_evidence()
    snapshot = copy.deepcopy(evidence)
    model = FakeModel(drafts=[complete_draft()])
    pipeline.run_answer_pipeline(QUERY, INTENT, evidence, make_env(model))
    assert evidence == snapshot


def test_a_claim_whose_source_cannot_fit_a_batch_is_not_run_and_not_adopted():
    huge = "x" * 5000
    evidence = make_evidence()
    evidence[1]["conditions"] = [huge[:1000]] * 6
    evidence[1]["exceptions"] = [huge[:1000]] * 6
    data = complete_draft()
    model = FakeModel(drafts=[data, data])
    env = make_env(model)
    result = pipeline.run_answer_pipeline(QUERY, INTENT, evidence, env)
    sent = [
        source["evidence_id"]
        for n, p in model.calls
        if n == "g2"
        for c in p["claims"]
        for source in c["source"]
    ]
    # E2を引用する主張はbatchに収まらず、意味照合へ送られず、表示文にも出ない。
    assert sent and "E2" not in sent
    assert "[E2]" not in result.answer
    assert result.quality["state"] != "passed"
    assert result.quality["gates"]["G2"]["status"] == "not_run"
    assert "input_too_large" in result.quality["gates"]["G2"]["codes"]


@pytest.mark.parametrize(("mode", "total"), [("normal", 300.0), ("deep", 300.0), ("deep", 1800.0)])
def test_the_first_drafts_checks_run_even_in_the_shortest_window(mode, total):
    """検査枠が2回分に満たない最短設定でも、初稿の意味照合が「時間なし」で止まらない。"""
    model = FakeModel(drafts=[complete_draft()])
    result = pipeline.run_answer_pipeline(
        QUERY, INTENT, make_evidence(), make_env(model, mode=mode, total=total)
    )
    assert model.count("g2") == 1
    assert result.quality["gates"]["G2"]["status"] == "passed"
    assert result.quality["state"] == "passed"


@pytest.mark.parametrize("kind", ["conclusion", "claim"])
def test_rounded_limit_is_rejected_before_model_and_repaired(kind):
    bad = complete_draft()
    text = "概算費用が一定基準を超える設備更新が対象です。"
    if kind == "conclusion":
        bad["conclusion"] = claim(text, "E2", "委員会において審査する案件は、次のとおりとする。")
    else:
        bad["sections"][1]["claims"][0]["text"] = text
    model = FakeModel(drafts=[bad, complete_draft()])
    result, _ = run(model)
    assert "condition_rounded" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]
    first_g2 = next(payload for name, payload in model.calls if name == "g2")
    assert all(c["claim"] != text for c in first_g2["claims"])
    second_prompt = [p for name, p in model.calls if name == "draft"][1]
    assert "概算費用１，０００万円以上の設備更新" in second_prompt
    assert "condition_rounded" not in result.quality["gates"]["G2"]["codes"]
    assert text not in result.answer
    assert result.quality["state"] == "passed"


def test_source_list_does_not_exempt_its_description_from_rounding_check():
    bad = complete_draft()
    bad["sections"][1]["claims"][0] = claim(
        "概算費用が一定基準を超える設備更新が対象です。",
        "E2",
        "概算費用１，０００万円以上の設備更新",
        source_list={
            "id": "E2",
            "from": "- 一　概算費用１，０００万円以上の設備更新",
            "to": "- 二　概算費用３００万円以上の設備改修（緊急修繕を除く。）",
        },
    )
    result, _ = run(FakeModel(drafts=[bad, complete_draft()]))
    assert "condition_rounded" in result.diagnostics["attempts"][0]["gates"]["G2"]["codes"]


def test_draft_prompt_points_requested_aspects_to_available_evidence_without_guaranteeing_support():
    evidence = make_evidence()
    intent = ap.AnswerIntent(
        required_aspects=["設置の背景", "提出期限・時期", "未知の比較"],
        target_terms=["設備更新委員会"],
    )
    evidence.append(
        {
            "evidence_id": "E4",
            "heading": "提出期限",
            "excerpt": "提出期限は開催の５日前。",
            "start_line": 1,
            "end_line": 1,
        }
    )
    evidence.append(
        {
            "evidence_id": "E5",
            "heading": "審査時期",
            "excerpt": "公告前に審査する。",
            "start_line": 1,
            "end_line": 1,
        }
    )
    roles = {
        "E1": ap.ROLE_BACKGROUND,
        "E2": ap.ROLE_DIRECT,
        "E3": ap.ROLE_PRACTICAL,
        "E4": ap.ROLE_PRACTICAL,
        "E5": ap.ROLE_DIRECT,
    }
    prompt = pipeline.build_draft_prompt("背景と提出期限・時期", intent, evidence, roles)
    assert "観点「設置の背景」: E1。" in prompt
    assert "観点「提出期限」: E4。" in prompt
    assert "観点「時期」: E5。" in prompt
    assert "観点「未知の比較」: 対応未確定。" in prompt
    assert "支持を保証する対応ではない" in prompt
    assert "節の主張で説明し、aspectに観点名を書く" in prompt


def test_prompt_includes_background_embedded_in_direct_evidence_without_reclassifying_it():
    intent = ap.AnswerIntent(required_aspects=["設置の背景"], target_terms=["設備更新委員会"])
    evidence = [
        {
            "evidence_id": "E9",
            "heading": "審査内容",
            "excerpt": "費用削減を目的として設備更新委員会を設置した。計画を審査する。",
        }
    ]
    roles = {"E9": ap.ROLE_DIRECT}
    prompt = pipeline.build_draft_prompt("設置の背景", intent, evidence, roles)
    assert "観点「設置の背景」: E9。" in prompt
    assert "[E9] role=direct_answer" in prompt


def document_evidence():
    rows = ["提出書類は次のとおりです。"]
    for kind, count in [("通常", 7), ("特別", 7), ("更新", 4)]:
        rows += ["① " + kind] + [f"- {kind}申請書類{i}" for i in range(count)]
    return {
        "evidence_id": "E4",
        "heading": "提出書類",
        "excerpt": "\n".join(rows),
        "start_line": 1,
        "path": "synthetic-documents.md",
    }


def test_requested_document_items_must_be_present_despite_a_matching_label():
    evidence = document_evidence()
    draft = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "claims": [
                        claim(
                            "書類は7日前までに提出します。",
                            "E4",
                            "提出書類は次のとおりです。",
                            aspect="提出書類",
                        )
                    ]
                }
            ]
        }
    )
    intent = ap.AnswerIntent(required_aspects=("提出書類",))
    assert pipeline._missing_required_lists(draft, intent, set(draft.claims), [evidence]) == [
        "提出書類"
    ]
    assert (
        pipeline._missing_required_lists(draft, ap.AnswerIntent(), set(draft.claims), [evidence])
        == []
    )


def test_all_eighteen_document_items_survive_the_pipeline_as_three_lists():
    evidence = document_evidence()
    groups = ap.evidence_catalog(evidence)["lists"]
    raw = {
        "conclusion": {
            "text": "方式別の書類を提出します。",
            "evidence": [{"id": "E4", "span_id": "S1"}],
        },
        "sections": [
            {
                "heading": "必要書類",
                "claims": [
                    {
                        "text": g["heading"] + "の場合に必要な書類です。",
                        "aspect": "提出書類",
                        "source_list": {"id": "E4", "group_id": g["group_id"]},
                    }
                    for g in groups
                ],
            }
        ],
    }
    model = FakeModel(drafts=[raw])
    env = make_env(model, source_lines_for=None)
    result = pipeline.run_answer_pipeline(
        "提出書類を教えてください。",
        ap.AnswerIntent(required_aspects=("提出書類",)),
        [evidence],
        env,
    )
    assert result.quality["gates"]["G1"]["status"] == "passed"
    assert result.quality["gates"]["G3"]["status"] == "passed"
    for group in groups:
        for row in group["items"]:
            assert result.answer.count(row) == 1


def test_repair_retains_verified_body_that_the_model_omits():
    first = good_draft()
    second = {"conclusion": first["conclusion"], "sections": [complete_draft()["sections"][2]]}
    model = FakeModel(drafts=[first, second])
    result, _ = run(model)
    assert result.diagnostics["attempts"][1]["retained_claims"] >= 3
    assert "費用削減が可能かどうか" in result.answer
    assert "仕様が妥当かどうか" in result.answer
    assert result.quality["state"] == "passed"


def test_an_accurate_list_does_not_protect_a_description_with_a_missing_condition():
    data = complete_draft()
    data["sections"][1]["claims"] = [
        {
            "text": "概算費用３００万円以上の設備改修が対象です。",
            "source_list": {
                "id": "E2",
                "from": "概算費用１，０００万円以上の設備更新",
                "to": "概算費用３００万円以上の設備改修（緊急修繕を除く。）",
            },
        }
    ]
    model = FakeModel(drafts=[data, data])
    result, _ = run(model)
    assert "condition_dropped" in result.quality["gates"]["G2"]["codes"]
    for name, payload in model.calls:
        if name == "g2":
            assert not any(
                c["claim"] == "概算費用３００万円以上の設備改修が対象です。"
                for c in payload["claims"]
            )


def test_search_overlap_is_bound_to_original_rows_before_emitting_span_ids():
    source = [
        "古い節です。",
        "前の節の本文です。",
        "# 設置背景",
        "",
        "費用削減を目的として研修委員会を設置します。",
    ]
    evidence = [
        {
            "evidence_id": "E4",
            "path": "synthetic-overlap.md",
            "source_sha256": "fixed",
            "start_line": 3,
            "end_line": 5,
            "excerpt": "前の節の本文です。\n# 設置背景\n\n費用削減を目的として研修委員会を設置します。",
        }
    ]
    env = make_env(
        FakeModel(), source_lines_for=lambda path, sha: source if sha == "fixed" else None
    )
    assert pipeline._bind_source_rows(evidence, env) == 1
    assert evidence[0]["excerpt"] == "\n".join(source[2:5])
    assert ap.evidence_catalog(evidence[0])["spans"][-1]["line"] == 5
    original = dict(evidence[0], source_sha256="changed", excerpt="元の候補です。")
    assert (
        pipeline._bind_source_rows([original], env) == 0 and original["excerpt"] == "元の候補です。"
    )


def test_g2_keeps_distant_cited_windows_within_the_same_evidence():
    lines = (
        ["- 序盤に研修手当は100円を支給します。"]
        + ["その他の独立した規定です。" * 9] * 30
        + ["- 終盤に教材手当は200円を支給します。"]
    )
    evidence = {"E4": {"excerpt": "\n".join(lines), "start_line": 1}}
    c = ap.DraftClaim(
        "c1",
        "研修手当は100円、教材手当は200円です。",
        refs=[ap.EvidenceRef("E4", lines[0]), ap.EvidenceRef("E4", lines[-1])],
    )
    c.ranges = [{"evidence_id": "E4", "start_line": n, "end_line": n} for n in (1, len(lines))]
    payload = pipeline._claim_payload(c, "", evidence)
    texts = "\n".join(s["text"] for s in payload["source"])
    assert "100円" in texts and "200円" in texts


def test_document_groups_exclude_procedural_submission_deadline_lists():
    e = document_evidence()
    e["excerpt"] += (
        "\n提出期限に留意すること。\n① 書類は前もって提出すること。\n② 契約変更の時間を確保すること。"
    )
    groups = pipeline._required_document_groups(ap.AnswerIntent(required_aspects=("提出書類",)), e)
    assert [len(g["items"]) for g in groups] == [7, 7, 4]


def test_retained_claims_reuse_only_identical_g2_payloads_after_g1():
    first = good_draft()
    second = {"conclusion": first["conclusion"], "sections": [complete_draft()["sections"][2]]}
    model = FakeModel(drafts=[first, second])
    result, _ = run(model)
    assert result.diagnostics["attempts"][1]["reused_g2_claims"] >= 3
    checks = [p for name, p in model.calls if name == "g2"]
    assert len(checks) == 2
    assert all("費用削減が可能" not in c["claim"] for c in checks[1]["claims"])
    draft = ap.parse_answer_draft(first)
    before = pipeline._verification_keys(
        draft, list(draft.claims), {e["evidence_id"]: e for e in make_evidence()}
    )
    scope_section(first)["scope"]["text"] = "研修まで含む新しい範囲です。"
    changed = ap.parse_answer_draft(first)
    after = pipeline._verification_keys(
        changed, list(changed.claims), {e["evidence_id"]: e for e in make_evidence()}
    )
    assert before[changed.sections[1].claim_ids[0]] != after[changed.sections[1].claim_ids[0]]


def test_semantic_batch_deduplicates_source_text_without_mixing_claim_sources():
    source_a = {"text": "研修手当を100円支給します。" * 100, "evidence_id": "E1"}
    source_b = {"text": "教材手当を200円支給します。", "evidence_id": "E2"}
    payloads = [
        {"claim_id": f"c{i}", "claim": "研修手当は100円です。", "source": [source_a]}
        for i in range(8)
    ]
    payloads.append({"claim_id": "c9", "claim": "教材手当は200円です。", "source": [source_b]})
    batches, oversize = pipeline._pack_batches(payloads)
    assert len(batches) == 1 and not oversize
    wire = pipeline._batch_payload(batches[0])
    assert len(wire["source_catalog"]) == 2
    assert wire["claims"][0]["source_ids"] != wire["claims"][-1]["source_ids"]
    assert all(
        wire["source_catalog"][row["source_ids"][0]] == source_a for row in wire["claims"][:8]
    )


def test_draft_example_does_not_require_optional_list_or_scope_on_every_claim():
    example = pipeline.DRAFT_SYSTEM.split("## 出力形式")[1].split("## 任意項目")[0]
    assert '"source_list"' not in example and '"scope"' not in example
    assert '"span_id"' in example


def test_retention_does_not_add_the_same_cited_deadline_rephrased():
    old = ap.DraftClaim(
        "c1",
        "審査書類は委員会開催の7開庁日前までに事務局へ提出しなければなりません。",
        refs=[ap.EvidenceRef("E1", "", "S1")],
    )
    new = ap.DraftClaim(
        "c2",
        "書類は委員会開催の7開庁日前までに事務局へ提出しなければなりません。",
        refs=[ap.EvidenceRef("E1", "", "S1")],
    )
    assert pipeline._same_fact_rephrased(old, new)
    new.text = new.text.replace("7", "8")
    assert not pipeline._same_fact_rephrased(old, new)


def test_exception_answer_must_preserve_the_source_precedence_relation():
    evidence = {e["evidence_id"]: e for e in make_evidence()}
    raw = {
        "sections": [
            {
                "claims": [
                    claim(
                        "同一仕様による部品の交換は対象外です。", "E2", "同一仕様による部品の交換"
                    )
                ]
            }
        ]
    }
    draft = ap.parse_answer_draft(raw)
    ap.check_g1(draft, evidence)
    intent = ap.AnswerIntent(required_aspects=("例外・除外", "条件"))
    assert pipeline._missing_condition_relation(draft, intent, set(draft.claims), evidence)
    draft.claims["c1"].text = "同一仕様による部品の交換は、前項の規定にかかわらず対象外です。"
    assert not pipeline._missing_condition_relation(draft, intent, set(draft.claims), evidence)


def test_conclusion_is_moved_from_verified_independent_body_without_changing_words_or_quotes():
    raw = complete_draft()
    raw.pop("conclusion")
    model = FakeModel(drafts=[raw])
    result, _ = run(model)
    assert result.quality["state"] == "passed"
    assert result.diagnostics["attempts"][0]["conclusion_from_verified_claim"]
    assert result.answer.count("費用削減が可能かどうかを審査します。") == 1
    assert result.answer.startswith("費用削減が可能かどうかを審査します。 [E2]")
    assert model.count("draft") == 1 and model.count("g2") == 1


def test_conclusion_promotion_does_not_remove_a_required_scope_or_an_unverified_claim():
    raw = {
        "sections": [
            {
                "scope": claim("特別申請の場合", "E2", "費用削減の可否"),
                "claims": [claim("費用削減の可否を審査します。", "E2", "費用削減の可否")],
            }
        ]
    }
    draft = ap.parse_answer_draft(raw)
    assert not pipeline._place_verified_conclusion(
        draft, set(draft.claims), INTENT, {"E2": ap.ROLE_DIRECT}
    )
    draft.sections[0].scope_claim_id = None
    assert not pipeline._place_verified_conclusion(draft, set(), INTENT, {"E2": ap.ROLE_DIRECT})


def test_identical_source_window_is_shared_but_each_claim_keeps_its_citation_lines():
    payloads = [
        {
            "claim_id": f"c{i}",
            "source": [{"evidence_id": "E1", "text": "研修費は100円です。", "lines": f"{i}-{i}"}],
        }
        for i in range(1, 4)
    ]
    wire = pipeline._batch_payload(payloads)
    assert len(wire["source_catalog"]) == 1
    assert [c["source_ranges"][0]["lines"] for c in wire["claims"]] == ["1-1", "2-2", "3-3"]
    assert all(c["source_ids"] == ["T1"] for c in wire["claims"])


def test_a_verified_quote_at_the_end_of_a_long_physical_line_is_not_rejected_as_an_invented_number():
    quote = "研修費は800円です。"
    evidence = [
        {
            "evidence_id": "E4",
            "heading": "研修費",
            "start_line": 1,
            "end_line": 1,
            "excerpt": "文脈の独立した説明です。" * 150 + quote,
        }
    ]
    model = FakeModel()
    env = make_env(model, source_lines_for=None)
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(quote, "E4", quote)]}]})
    by_id = {e["evidence_id"]: e for e in evidence}
    assert ap.check_g1(draft, by_id)["c1"].status == ap.PASSED
    assert pipeline.verify_claims("研修費は？", draft, ["c1"], by_id, env)["c1"].status == ap.PASSED


def test_declarative_supported_body_can_be_used_as_conclusion():
    draft = ap.parse_answer_draft(
        {
            "sections": [
                {
                    "claims": [
                        claim("設備更新会は数量と費用の妥当性を審査する。", "E2", "費用削減の可否")
                    ]
                }
            ]
        }
    )
    assert (
        pipeline._place_verified_conclusion(
            draft, set(draft.claims), INTENT, {"E2": ap.ROLE_DIRECT}
        )
        == "c1"
    )


def test_retention_does_not_duplicate_a_promoted_conclusion_already_in_repaired_body():
    text = "設備更新会は数量と費用の妥当性を審査します。"
    raw = {"sections": [{"claims": [claim(text, "E2", "費用削減の可否")]}]}
    previous = ap.parse_answer_draft(raw)
    verified = set(previous.claims)
    pipeline._place_verified_conclusion(previous, verified, INTENT, {"E2": ap.ROLE_DIRECT})
    gates = {name: ap.GateResult(ap.PASSED, (), "answer") for name in ("G1", "G2", "G3", "G4")}
    attempt = pipeline._Attempt(previous, verified, gates, [], 0, {})
    repaired = ap.parse_answer_draft(raw)
    assert pipeline._retain_verified(repaired, attempt) == 0
    assert len(repaired.claims) == 1
    assert (
        pipeline._place_verified_conclusion(
            repaired, set(repaired.claims), INTENT, {"E2": ap.ROLE_DIRECT}
        )
        == "c1"
    )


@pytest.mark.parametrize(
    "text,missing",
    [
        ("反復する研修費は、金額や契約形態にかかわらず対象外です。", True),
        ("反復する研修費は、前項第1号の規定にかかわらず対象外です。", True),
        ("反復する研修費は、前項第1号又は第2号の規定にかかわらず対象外です。", False),
        ("反復する研修費は、前項第一号又は第二号の規定にかかわらず対象外です。", False),
    ],
)
def test_precedence_relation_retains_the_specific_numbered_rules(text, missing):
    source = "前項第１号又は第２号の規定に関わらず、反復する研修費は対象外である。"
    evidence = {"E4": {"evidence_id": "E4", "excerpt": source, "start_line": 1, "end_line": 1}}
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(text, "E4", source)]}]})
    assert ap.check_g1(draft, evidence)["c1"].status == ap.PASSED
    intent = ap.AnswerIntent(required_aspects=("条件", "例外"))
    assert (
        bool(pipeline._missing_condition_relation(draft, intent, set(draft.claims), evidence))
        is missing
    )


@pytest.mark.parametrize(
    "text,status",
    [
        ("研修費は金額にかかわらず対象外です。", ap.FAILED),
        ("研修費は前項第1号又は第2号の規定にかかわらず対象外です。", ap.PASSED),
        ("研修費は対象外です。", ap.PASSED),
    ],
)
def test_g2_rejects_widened_numbered_precedence_before_the_model_can_support_it(text, status):
    source = "前項第１号又は第２号の規定に関わらず、研修費は対象外である。"
    evidence = {"E4": {"evidence_id": "E4", "excerpt": source, "start_line": 1, "end_line": 1}}
    draft = ap.parse_answer_draft({"sections": [{"claims": [claim(text, "E4", source)]}]})
    assert ap.check_g1(draft, evidence)["c1"].status == ap.PASSED
    model = FakeModel()
    result = pipeline.verify_claims(
        "対象外となる研修費は？", draft, ["c1"], evidence, make_env(model, source_lines_for=None)
    )
    assert result["c1"].status == status
    if status == ap.FAILED:
        assert result["c1"].codes == ("scope_overreach",)
        assert model.count("g2") == 0


@pytest.mark.parametrize(
    ("source", "status"),
    [
        (
            "| 様式 | 内容 |\n| --- | --- |\n| A | 申込書 |\nこの通知は4月1日から適用する。",
            ap.FAILED,
        ),
        ("## 実施時期\n| 様式 | 実施時期 |\n| --- | --- |\n| 申込書の様式 |  |", ap.FAILED),
        ("| 様式 | 実施時期 |\n| --- | --- |\n| A | 4月1日から |", ap.PASSED),
        ("各様式の実施時期は別表のとおりとする。", ap.PASSED),
    ],
)
def test_g2_checks_for_source_support_for_timing_by_form(source, status):
    evidence = {
        "E4": {
            "evidence_id": "E4",
            "excerpt": source,
            "start_line": 1,
            "end_line": len(source.splitlines()),
        }
    }
    timing_claim = claim("実施時期は様式ごとに定められています。", "E4", source.splitlines()[-1])
    draft = ap.parse_answer_draft({"sections": [{"claims": [timing_claim]}]})
    assert ap.check_g1(draft, evidence)["c1"].status == ap.PASSED
    model = FakeModel()

    result = pipeline.verify_claims(
        "手続の詳細は？", draft, ["c1"], evidence, make_env(model, source_lines_for=None)
    )

    assert result["c1"].status == status
    assert model.count("g2") == (1 if status == ap.PASSED else 0)
    if status == ap.FAILED:
        assert result["c1"].codes == ("not_in_source",)
