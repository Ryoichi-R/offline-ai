"""deepの回答方針v2: 役割別の読取順・枠配分・品質状態・fallback・再探索。実モデル・実資料は使わない。"""

import json
import re

import pytest

import answer_pipeline as apipe
import answer_policy as ap
import deep_candidate_selection
import deep_research as deep
from answer_fixtures import QUERY, SOURCE_TEXT
from deep_fakes import patch_retrieval

VERDICT = {
    "supported": True,
    "contradictions": [],
    "missing_conditions": [],
    "unsupported_claims": [],
}


def evidence_blocks(user):
    """プロンプトの根拠ブロックを {id: (role, 本文)} にする。"""
    blocks = {}
    parts = re.split(r"(?m)^\[(E\d+)\] role=(\w+) ", user)
    for index in range(1, len(parts), 3):
        blocks[parts[index]] = (parts[index + 1], parts[index + 2])
    return blocks


def find_id(blocks, quote, prefer_role=None):
    normalized = ap.normalize_text(quote)
    hits = [eid for eid, (role, text) in blocks.items() if normalized in ap.normalize_text(text)]
    if prefer_role:
        hits.sort(key=lambda eid: blocks[eid][0] != prefer_role)
    return hits[0] if hits else None


def claim_for(blocks, text, quote, **extra):
    eid = find_id(blocks, quote)
    return {"text": text, "evidence": [{"id": eid or "E99", "quote": quote}], **extra}


def complete_draft_for(user):
    blocks = evidence_blocks(user)
    rule = "委員会において審査する内容は、前項の案件について、次のとおりとする。"
    return {
        "conclusion": claim_for(
            blocks,
            "委員会は、設備の更新前に計画を審査し、費用削減の可否・仕様の妥当性・競争性の阻害要因の有無を確認します。",
            rule,
            aspect="審査内容",
        ),
        "sections": [
            {
                "heading": "審査する内容",
                "claims": [
                    claim_for(blocks, "費用削減が可能かどうかを審査します。", "費用削減の可否"),
                    claim_for(blocks, "仕様が妥当かどうかを審査します。", "仕様の妥当性"),
                ],
            },
            {
                "heading": "区別",
                "claims": [
                    claim_for(
                        blocks,
                        "理事長の指示（更新費用の２割削減）は設置の背景であり、審査内容そのものではありません。",
                        "設備の更新費用を２割削減する。",
                        aspect="設置の背景との区別",
                    ),
                    claim_for(
                        blocks,
                        "技術評価会の開催方法は、別紙のチェック項目で確認する対象であり、委員会自身の審査事項ではありません。",
                        "技術評価会は、原則、会議形式で開催することになっているか。",
                        aspect="別の会議体の役割との区別",
                    ),
                ],
            },
        ],
        "unanswered_aspects": [],
    }


class Backend:
    def __init__(self, monkeypatch, *, drafts=None, g2=None, rank="checklist_first", agentic=True):
        self.prompts = []
        self.retrieval_queries = []
        self.intents = []
        self.drafts = list(drafts) if drafts else None
        self.g2 = g2 or (lambda claims: {c["claim_id"]: ("supported", "ok") for c in claims})
        self.g2_calls = 0
        self.draft_calls = 0
        monkeypatch.setenv("OFFLINE_AI_AGENTIC_LITE", "true" if agentic else "false")
        monkeypatch.setattr(deep, "_call_ollama_json", self.model)
        patch_retrieval(monkeypatch, deep, self.retrieve)
        self.rank = rank

    def retrieve(self, query, chunks, **kwargs):
        kwargs["cancel_check"]()
        self.retrieval_queries.append(query)
        ordered = sorted(
            chunks,
            key=lambda c: (
                ("チェックリスト" not in (c.get("heading") or ""))
                if self.rank == "checklist_first"
                else 0,
                c["start_line"],
            ),
        )
        ranked = [{**c, "source_sha256": c["file_sha256"]} for c in ordered]
        role_of = None
        if "intent" in kwargs:
            intent = ap.AnswerIntent.from_dict(kwargs["intent"])
            self.intents.append(intent)

            def role_of(chunk):
                return ap.classify_role(
                    chunk.get("text", ""), chunk.get("heading", ""), intent=intent
                ).role

        return deep_candidate_selection.select_candidates(
            ranked, source_chunks=chunks, role_of=role_of
        )

    def model(self, name, system, user, **kwargs):
        if system == deep.DEEP_RELEVANCE_SYSTEM:
            return {"relevance": "irrelevant"}
        if system == deep.DEEP_VERIFY_SYSTEM:
            return dict(VERDICT)
        if system == deep.DEEP_GAPS_SYSTEM:
            return {"queries": [], "unresolved": []}
        if system == deep.DEEP_SECTION_SYSTEM:
            return {
                "subject": "対象",
                "scope": "原文の範囲",
                "conditions": ["条件"],
                "exceptions": [],
                "references": [],
            }
        if system.startswith("あなたは質問の意図"):
            return {"granularity": "overview", "target": "設備更新委員会", "required_aspects": []}
        if system.startswith("あなたはローカル資料の調査結果を"):
            self.prompts.append(user)
            self.draft_calls += 1
            if self.drafts:
                draft = self.drafts.pop(0)
                return draft(user) if callable(draft) else draft
            return complete_draft_for(user)
        if system.startswith("あなたは回答の主張と原文の照合者"):
            self.g2_calls += 1
            payload = json.loads(user)
            return {
                "results": [
                    {"claim_id": cid, "verdict": v, "reason_code": r}
                    for cid, (v, r) in self.g2(payload["claims"]).items()
                ]
            }
        raise AssertionError("unexpected prompt")


def run(
    tmp_path,
    monkeypatch,
    *,
    text=SOURCE_TEXT,
    policy="v2",
    timeout=300,
    query=QUERY,
    **backend_kwargs,
):
    backend = Backend(monkeypatch, **backend_kwargs)
    (tmp_path / "sample.md").write_text(text, encoding="utf-8")
    result = deep.run_deep_research(
        query, model="fake", source_root=tmp_path, timeout_seconds=timeout, answer_policy=policy
    )
    return result, backend


# --- 方針の固定 ---------------------------------------------------------------------


@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_high_reasoning_draft_output_limit_uses_one_source_only_retry(
    tmp_path, monkeypatch, retry_succeeds
):
    backend = Backend(monkeypatch)
    prompts = []

    def model(name, system, user, **kwargs):
        if system in {apipe.DRAFT_SYSTEM, apipe.DRAFT_RETRY_SYSTEM}:
            prompts.append((system, user, kwargs))
            if system == apipe.DRAFT_SYSTEM or not retry_succeeds:
                raise deep.DeepWorkerError(code=deep.MODEL_OUTPUT_INCOMPLETE_CODE)
            return complete_draft_for(user)
        assert "num_predict" not in kwargs
        if (
            system == apipe.CLAIM_VERIFY_SYSTEM
            and kwargs["timeout"] <= ap.CHECK_CALL_CAP_SECONDS["deep"]
        ):
            raise TimeoutError("first meaning check requires more than the usual 60 seconds")
        return backend.model(name, system, user, **kwargs)

    monkeypatch.setattr(deep, "_call_ollama_json", model)
    (tmp_path / "sample.md").write_text(SOURCE_TEXT, encoding="utf-8")
    result = deep.run_deep_research(
        QUERY,
        model="gpt-oss:120b",
        reasoning="high",
        source_root=tmp_path,
        timeout_seconds=1800,
        answer_policy="v2",
    )

    assert [system for system, _, _ in prompts] == [apipe.DRAFT_SYSTEM, apipe.DRAFT_RETRY_SYSTEM]
    first, retry = prompts
    assert "抽出メモ" in first[1] and "抽出メモ" not in retry[1]
    assert "原文:" in retry[1] and "引用・一覧の選択表" in retry[1]
    assert "num_predict" not in first[2]
    assert retry[2]["num_predict"] == deep.DEEP_FINAL_RETRY_NUM_PREDICT
    assert retry[2]["reasoning"] == "high"
    assert (backend.g2_calls > 0) is retry_succeeds
    assert (result.answer_state == deep.ANSWER_STATE_GENERATED) is retry_succeeds


def test_default_is_legacy_and_the_request_fixes_its_policy(tmp_path, monkeypatch):
    monkeypatch.delenv(ap.POLICY_ENV, raising=False)
    result, backend = run(tmp_path, monkeypatch, policy=None)
    assert result.diagnostics["answer_policy"] == "legacy"
    assert result.answer_quality == {}
    assert backend.intents == []
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    result2, _ = run(tmp_path, monkeypatch, policy=None)
    assert result2.diagnostics["answer_policy"] == "v2"
    assert result2.answer_quality["state"] in {"passed", "partial", "needs_review", "failed"}
    # 明示指定は環境より優先される（要求内では全経路が一つに固定される）。
    result3, _ = run(tmp_path, monkeypatch, policy="legacy")
    assert result3.diagnostics["answer_policy"] == "legacy"
    with pytest.raises(ValueError):
        run(tmp_path, monkeypatch, policy="v3")


# --- 読取順と枠配分 -------------------------------------------------------------------


def test_direct_rule_is_read_before_the_checklist_even_when_the_checklist_ranks_first(
    tmp_path, monkeypatch
):
    legacy, _ = run(tmp_path, monkeypatch, policy="legacy")
    v2, backend = run(tmp_path, monkeypatch, policy="v2")
    assert "チェックリスト" in legacy.evidence[0]["heading"]
    assert v2.evidence[0]["role"] == ap.ROLE_DIRECT
    assert "審査案件及び審査内容" in v2.evidence[0]["heading"]
    assert v2.diagnostics["answer_policy_v2"]["role_selection"].get(ap.ROLE_DIRECT, 0) >= 1
    assert backend.intents and backend.intents[0].granularity == "overview"


def long_checklist_source():
    rows = "\n".join(f"| ①{i}　設備{i}の図面が添付されているか。 | |" for i in range(260))
    return (
        SOURCE_TEXT.split("## 別紙１")[0]
        + "## 別紙１　設備更新　設備更新委員会チェックリスト\n\n| 確認項目 | 結果 |\n|---|---|\n"
        + rows
        + "\n"
    )


def test_overview_defers_long_detail_tables_without_calling_them_unconfirmed(tmp_path, monkeypatch):
    result, _ = run(tmp_path, monkeypatch, text=long_checklist_source())
    diag = result.diagnostics["answer_policy_v2"]
    practical_chars = diag["role_chars"].get(ap.ROLE_PRACTICAL, 0)
    assert (
        practical_chars
        <= ap.role_budget_caps("overview", deep.DEEP_MAX_EVIDENCE_CHARS)[ap.ROLE_PRACTICAL]
    )
    assert diag["role_deferred"].get(ap.ROLE_PRACTICAL, 0) >= 1
    deferred = [e for e in result.ledger if e["status"] == "deferred"]
    assert deferred and all(e["reason"] == "role_deferred" for e in deferred)
    assert not any(item.startswith("role_deferred") for item in result.unconfirmed)
    assert result.stop_reason != "role_deferred"
    # 直接規定と条件は枠に残っている。
    assert any(e["role"] == ap.ROLE_DIRECT for e in result.evidence)


def test_a_detail_question_does_not_defer_the_table(tmp_path, monkeypatch):
    query = "設備更新委員会の実務の準備として具体的なチェック項目を教えて"
    result, _ = run(tmp_path, monkeypatch, text=long_checklist_source(), query=query)
    assert result.diagnostics["answer_policy_v2"]["intent"]["granularity"] == "practical"
    assert result.diagnostics["answer_policy_v2"]["role_deferred"].get(ap.ROLE_PRACTICAL, 0) == 0


# --- 品質状態 -------------------------------------------------------------------------


def test_v2_builds_a_checked_answer_and_reports_quality_separately_from_generated(
    tmp_path, monkeypatch
):
    result, backend = run(tmp_path, monkeypatch)
    assert result.answer_state == "generated"
    assert result.answer_quality["state"] == "passed"
    assert result.answer_quality["label"] == "検査済み"
    assert result.answer.splitlines()[0].startswith("委員会は、設備の更新前に計画を審査し")
    assert "[E" in result.answer and "\\" not in result.answer
    final = result.diagnostics["final_answer"]
    assert final["result"] == "v2_passed" and final["attempts"] == []
    calls = final["v2"]["calls"]["counts"]
    assert calls["draft"] == {"ok": 1} and calls["g2"] == {"ok": 1}
    assert result.to_public_dict()["answer_quality"]["state"] == "passed"
    assert backend.draft_calls == 1
    assert result.status == "completed"


def test_unsupported_claims_never_reach_the_answer_and_quality_is_needs_review(
    tmp_path, monkeypatch
):
    def reject_one(claims):
        return {
            c["claim_id"]: (
                ("unsupported", "subject_mismatch")
                if "仕様が妥当" in c["claim"]
                else ("supported", "ok")
            )
            for c in claims
        }

    result, _ = run(tmp_path, monkeypatch, g2=reject_one)
    assert "仕様が妥当かどうか" not in result.answer
    assert result.answer_quality["state"] == "needs_review"
    assert result.status == "partial"
    assert any(e["reason"] == "answer_quality" for e in result.ledger)
    # 回答の状態を説明する注記は、探索を止めた理由にはならない。
    assert result.stop_reason == "scope_processed"


def test_when_nothing_can_be_verified_the_common_fallback_is_shown(tmp_path, monkeypatch):
    result, _ = run(
        tmp_path,
        monkeypatch,
        g2=lambda claims: {c["claim_id"]: ("unsupported", "not_in_source") for c in claims},
    )
    assert result.answer_state == "verification_failed"
    assert result.answer.startswith("回答を作成できませんでした。")
    assert "## 参考の原文" in result.answer
    assert result.answer_quality["state"] == "failed"
    assert result.status == "partial"
    # 同じ原文を二重に表示しない。
    locations = re.findall(r"`skill-source/[^`]+` L\d+-\d+", result.answer)
    assert len(locations) == len(set(locations))


def test_invalid_json_twice_ends_in_fallback_with_exactly_two_generation_calls(
    tmp_path, monkeypatch
):
    def broken(user):
        raise ValueError("bad json")

    result, backend = run(tmp_path, monkeypatch, drafts=[broken, broken])
    assert backend.draft_calls == 2 and backend.g2_calls == 0
    assert result.answer_state == "verification_failed"
    assert result.answer_quality["state"] == "failed"


def test_no_model_output_is_ever_written_to_the_diagnostics(tmp_path, monkeypatch):
    result, _ = run(tmp_path, monkeypatch)
    dumped = json.dumps(result.diagnostics, ensure_ascii=False)
    for secret in ("審査します", "設備更新委員会", "技術評価会", "費用削減"):
        assert secret not in dumped


# --- 時間配分 -------------------------------------------------------------------------


def test_exploration_keeps_the_planned_downstream_share(tmp_path, monkeypatch):
    result, _ = run(tmp_path, monkeypatch, timeout=300)
    assert result.diagnostics["answer_policy_v2"]["downstream_reserve_seconds"] == pytest.approx(
        141.0
    )
    assert (
        result.diagnostics["reserved_seconds"]["integration"]
        == deep.DEEP_INTEGRATION_RESERVE_SECONDS
    )


# --- 再探索（最大1回） -------------------------------------------------------------------


def test_missing_aspect_triggers_one_bounded_reexploration(tmp_path, monkeypatch):
    def without_separation(user):
        data = complete_draft_for(user)
        data["sections"][1]["claims"].pop()  # 「別の会議体の役割との区別」に答える主張を欠く
        return data

    monkeypatch.setattr(deep, "DEEP_FINAL_RETRY_MIN_SECONDS", 0)
    result, backend = run(tmp_path, monkeypatch, drafts=[without_separation, complete_draft_for])
    final = result.diagnostics["final_answer"]["v2"]
    assert final["reexplored"] is True
    assert backend.draft_calls == 2
    extra_views = [q for q in backend.retrieval_queries if "別の会議体の役割との区別" in q]
    assert 1 <= len(extra_views) <= 2
    assert result.answer_quality["state"] == "passed"


def test_reexploration_is_skipped_when_no_time_remains_beyond_the_reserve(tmp_path, monkeypatch):
    def without_separation(user):
        data = complete_draft_for(user)
        data["sections"][1]["claims"].pop()
        return data

    monkeypatch.setattr(deep, "DEEP_FINAL_RETRY_MIN_SECONDS", 10**6)  # 余裕が必要量を超えない
    result, backend = run(tmp_path, monkeypatch, drafts=[without_separation, without_separation])
    assert all("別の会議体の役割との区別" not in q for q in backend.retrieval_queries[5:])
    assert result.answer_quality["state"] == "partial"
    assert result.answer_quality["unconfirmedAspects"] == ["別の会議体の役割との区別"]


# --- 意図計画（agentic無効時） --------------------------------------------------------------


def test_intent_planning_is_skipped_when_agentic_lite_is_disabled(tmp_path, monkeypatch):
    seen = []
    original = apipe.plan_intent

    def spy(query, explicit, env, *, enabled=True):
        seen.append(enabled)
        return original(query, explicit, env, enabled=enabled)

    monkeypatch.setattr(apipe, "plan_intent", spy)
    result, _ = run(tmp_path, monkeypatch, agentic=False)
    assert seen == [False]
    assert result.diagnostics["answer_policy_v2"]["llm_calls"]["counts"].get("intent") is None
    assert result.diagnostics["answer_policy_v2"]["intent"]["source"] == "deterministic"


def test_intent_plan_makes_at_most_one_call(tmp_path, monkeypatch):
    result, _ = run(tmp_path, monkeypatch)
    assert result.diagnostics["answer_policy_v2"]["llm_calls"]["counts"]["intent"] == {"ok": 1}
    assert result.diagnostics["answer_policy_v2"]["intent"]["source"] == "llm"


# --- 中止（既存の終端契約を保つ） -----------------------------------------------------------


def test_cancel_during_v2_answer_construction_ends_as_cancelled_without_a_partial_answer(
    tmp_path, monkeypatch
):
    import web_services

    state = {"cancelled": False}

    def cancel_after_draft(user):
        state["cancelled"] = True  # 草稿生成が返った直後から中止が要求される
        return complete_draft_for(user)

    def cancel_check():
        if state["cancelled"]:
            raise web_services.CancelledError("cancelled", code="cancelled")

    backend = Backend(monkeypatch, drafts=[cancel_after_draft])
    (tmp_path / "sample.md").write_text(SOURCE_TEXT, encoding="utf-8")
    result = deep.run_deep_research(
        QUERY,
        model="fake",
        source_root=tmp_path,
        timeout_seconds=300,
        answer_policy="v2",
        cancel_check=cancel_check,
    )
    assert result.status == "cancelled" and result.stop_reason == "cancelled"
    assert result.answer_state != "generated"
    assert backend.g2_calls == 0  # 中止後は意味照合へ進まない
    assert result.answer_quality["state"] == "failed"
