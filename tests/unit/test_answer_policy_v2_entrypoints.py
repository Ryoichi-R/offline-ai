"""回答方針v2の入口: 通常回答の検索・構成、Web SSE、deepイベント、CLI。実モデルは使わない。"""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import answer_pipeline as apipe
import answer_policy as ap
import deep_research
import search as cli_search
import web_services
from answer_fixtures import QUERY, SOURCE_TEXT, FakeModel
from test_deep_answer_policy_v2 import complete_draft_for

SAMPLE = "notice/sample-equipment.md"


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    (tmp_path / "notice").mkdir()
    (tmp_path / SAMPLE).write_text(SOURCE_TEXT, encoding="utf-8")
    for index in range(4):
        rows = "\n".join(f"| ①{n}　設備更新委員会の図面{n}が添付されているか。 | |" for n in range(12))
        (tmp_path / "notice" / f"checklist-{index}.md").write_text(
            f"## 別紙{index}　設備更新　設備更新委員会チェックリスト\n\n| 確認項目 | 結果 |\n|---|---|\n{rows}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(cli_search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(cli_search, "_chunk_retrieval_enabled", lambda: True)
    monkeypatch.setattr(cli_search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(cli_search, "RERANK_CONFIG", cli_search.RerankConfig("off", "", 8, 8, 1, False))
    monkeypatch.setattr(cli_search, "_evidence_verify_enabled", lambda: False)
    plan_calls = []

    def plan(query, model, **kwargs):
        plan_calls.append(kwargs)
        result = {
            "keywords": cli_search._split_terms(query),
            "search_queries": [query],
            "must_find_terms": [],
            "query_type": "standard",
            "answer_should_compare": False,
        }
        if kwargs.get("with_intent"):
            result["answer_intent"] = ap.extract_explicit_intent(query).to_dict()
        return result

    monkeypatch.setattr(cli_search, "create_search_plan", plan)
    return tmp_path, plan_calls


# --- 通常回答の検索 ---------------------------------------------------------------------


def crowded_candidates(tmp_path):
    """チェック表が上位を占め、直接規定が下位にある検索結果（旧経路の失敗の形）。"""
    chunks = cli_search.build_source_chunks(tmp_path)
    tables = [c for c in chunks if "checklist-" in c["path"] and c["start_line"] == 1]
    rule = [c for c in chunks if "（審査案件及び審査内容）" in c.get("heading", "")]
    assert len(tables) == 4 and rule
    ranked = [*tables, *rule]
    return [
        {**c, "snippet": c["text"], "rrf_score": 1.0 - 0.01 * index, "source": "keyword"}
        for index, c in enumerate(ranked)
    ]


def test_v2_retrieval_carries_the_intent_adds_the_rule_view_and_keeps_the_rule_in_the_frame(corpus, monkeypatch):
    tmp_path, plan_calls = corpus
    candidates = crowded_candidates(tmp_path)
    queries_seen = []

    def fixed_retrieval(search_query, *args, **kwargs):
        queries_seen.append(search_query)
        return [dict(c) for c in candidates]

    monkeypatch.setattr(cli_search, "_run_single_retrieval", fixed_retrieval)
    monkeypatch.setattr(cli_search, "final_evidence_match_limit", lambda: 4)
    legacy = cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="legacy")
    legacy_queries = list(queries_seen)
    queries_seen.clear()
    v2 = cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="v2")
    assert plan_calls == [{}, {"with_intent": True}]  # 旧経路の呼出しは不変
    assert "answer_intent" not in legacy.plan
    assert v2.plan["answer_intent"]["granularity"] == "overview"
    assert "設備更新委員会 審議 審査 規定 事項" in queries_seen
    assert "設備更新委員会 審議 審査 規定 事項" not in legacy_queries
    assert v2.trace["final"]["answer_policy"]["version"] == "v2"
    # 旧経路では上位のチェック表が件数枠を占め、直接規定が落ちる。v2では直接規定が先頭に残る。
    assert all("審査案件及び審査内容" not in m.get("snippet", "") for m in legacy.matches)
    assert "審査案件及び審査内容" in v2.matches[0]["snippet"]


def test_v2_takes_a_wider_pool_before_the_candidate_limit_so_a_low_ranked_rule_survives(corpus, monkeypatch):
    # P4実資料: 直接規定は質問viewで31位・根拠規定viewで26位。検索viewごとの上位8件で打ち切る旧経路では、
    # 後段の役割並べ替えでは救えない（並べ替えるのは既に残った候補だけ）。
    tmp_path, _ = corpus
    for index in range(4, 14):
        rows = "\n".join(f"| ①{n}　設備更新委員会の配線{n}が確認されているか。 | |" for n in range(12))
        (tmp_path / "notice" / f"checklist-{index}.md").write_text(
            f"## 別紙{index}　設備更新委員会チェックリスト\n\n| 確認項目 | 結果 |\n|---|---|\n{rows}\n",
            encoding="utf-8",
        )
    chunks = cli_search.build_source_chunks(tmp_path)
    tables = [c for c in chunks if "checklist-" in c["path"] and c["start_line"] == 1]
    rule = [c for c in chunks if "（審査案件及び審査内容）" in c.get("heading", "")]
    assert len(tables) == 14 and rule
    ranked = [
        {**c, "snippet": c["text"], "rrf_score": 1.0 - 0.01 * index, "source": "keyword"}
        for index, c in enumerate([*tables, *rule])
    ]
    limits = []

    def ranked_retrieval(search_query, *args, candidate_limit=None, **kwargs):
        limit = candidate_limit or cli_search.RETRIEVAL_CANDIDATE_LIMIT
        limits.append(limit)
        return [dict(c) for c in ranked[:limit]]

    monkeypatch.setattr(cli_search, "_run_single_retrieval", ranked_retrieval)
    monkeypatch.setattr(cli_search, "final_evidence_match_limit", lambda: 4)
    legacy = cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="legacy")
    assert set(limits) == {cli_search.RETRIEVAL_CANDIDATE_LIMIT}  # 旧経路の呼出しは不変
    assert all("審査案件及び審査内容" not in m.get("snippet", "") for m in legacy.matches)
    limits.clear()
    v2 = cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="v2")
    assert set(limits) == {cli_search.ANSWER_V2_ROLE_POOL}
    assert "審査案件及び審査内容" in v2.matches[0]["snippet"]


def test_role_selection_keeps_original_rank_slots_for_candidates_it_cannot_classify():
    intent = ap.extract_explicit_intent(QUERY)
    rules = [
        {"snippet": f"第{n}条　委員会において審査する内容は、次のとおりとする。", "heading": "（審査案件及び審査内容）"}
        for n in range(5)
    ]
    vague = [{"snippet": f"関連する記述{n}。", "heading": ""} for n in range(10)]
    chosen = cli_search.select_matches_by_role([*vague, *rules], intent, 5)
    # 役割で埋めるのは枠の6割(5件中3件)まで。残りは元の順位で埋め、判定できない上位候補を失わない。
    assert chosen[:3] == rules[:3]
    assert chosen[3:] == vague[:2]
    assert cli_search.select_matches_by_role(vague, intent, 0) == []


def evidence_match(item):
    return {"snippet": item["excerpt"], "heading": item["heading"], "path": item["path"]}


def test_aspect_evidence_is_placed_right_after_the_top_direct_answers():
    # P4実資料: 背景（通知本文）の根拠が最終の件数枠に入らず、背景との区別に答えられなかった。
    from answer_fixtures import make_evidence

    intent = ap.extract_explicit_intent(QUERY)
    background, rule, checklist = (evidence_match(e) for e in make_evidence())
    rules = [dict(rule, path=f"r{n}.md") for n in range(4)]
    decoy = {"snippet": "委員会を設置する。", "heading": "別の委員会の設置", "path": "decoy.md"}
    ranked = cli_search.prioritize_matches_by_role([*rules, decoy, checklist, background], intent)
    assert ranked[:2] == rules[:2]
    assert ranked[2] is background  # 見出しに「設置」があるだけの候補は背景の枠に入らない
    chosen = cli_search.select_matches_by_role([decoy, checklist, *rules, background], intent, 6)
    assert background in chosen


def test_aspect_slots_split_listed_aspects_and_skip_practical_detail_for_an_overview():
    from answer_fixtures import make_evidence

    assert ap.aspect_units("期限・時期") == ["期限", "時期"]
    assert ap.aspect_units("別の会議体の役割との区別") == ["別の会議体の役割との区別"]
    _, rule, checklist = (evidence_match(e) for e in make_evidence())
    overview = ap.AnswerIntent(target_terms=("設備更新委員会",), predicate_terms=("審査",), required_aspects=("図面",))
    practical = ap.AnswerIntent(
        granularity="practical", target_terms=("設備更新委員会",), predicate_terms=("審査",), required_aspects=("図面",)
    )
    rules = [dict(rule, path=f"r{n}.md") for n in range(3)]
    assert checklist not in cli_search.prioritize_matches_by_role([*rules, checklist], overview)[:3]
    assert cli_search.prioritize_matches_by_role([*rules, checklist], practical)[:3].count(checklist) == 1


def test_policy_is_fixed_per_request_and_an_invalid_value_is_rejected(corpus, monkeypatch):
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    assert "answer_intent" in cli_search.run_retrieval_pipeline(QUERY, model="m").plan
    # 明示引数は環境より優先される。
    assert "answer_intent" not in cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="legacy").plan
    with pytest.raises(ValueError):
        cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="v9")
    # 検索専用は回答方針と無関係で、LLMを呼ばない。
    assert "answer_intent" not in cli_search.run_retrieval_pipeline(QUERY, model="", mode="search").plan


def test_plan_intent_failure_keeps_a_conservative_intent(monkeypatch):
    monkeypatch.setenv("OFFLINE_AI_AGENTIC_LITE", "true")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"message": {"content": "{\\"keywords\\": [\\"a\\"], \\"search_queries\\": [\\"a\\"], \\"answer_intent\\": {\\"granularity\\": \\"huge\\"}}"}}'

    requested = []

    def urlopen(request, *args, **kwargs):
        requested.append(json.loads(request.data)["messages"][0]["content"])
        return Response()

    monkeypatch.setattr(cli_search.urllib.request, "urlopen", urlopen)
    plan = cli_search.create_search_plan(QUERY, "m", with_intent=True)
    assert plan["answer_intent"]["source"] == "plan_failed"
    assert plan["answer_intent"]["granularity"] == "overview"
    assert plan["answer_intent"]["required_aspects"]  # 質問に明示された観点は失われない
    cli_search.create_search_plan(QUERY, "m")
    # v2だけが意図付きの計画プロンプトを使い、旧経路のプロンプトは変わらない（計画は1回の要求）。
    assert requested == [cli_search.SEARCH_PLAN_SYSTEM_V2, cli_search.SEARCH_PLAN_SYSTEM]


# --- 通常回答の構成 ---------------------------------------------------------------------


def patch_chat(monkeypatch, model):
    captured = {}

    def chat(model_name, system, user, **kwargs):
        captured.update(kwargs)
        return model(system, user, timeout=kwargs["timeout"])

    monkeypatch.setattr(cli_search, "chat_json_stream", chat)
    return captured


SIMPLE_QUERY = "提供資料において、設備更新委員会が審議する事項を教えてください"


def simple_draft_for(user):
    """根拠に実在する直接規定だけで答える草稿（観点の指定がない質問用）。"""
    from test_deep_answer_policy_v2 import claim_for, evidence_blocks

    blocks = evidence_blocks(user)
    rule = "委員会において審査する内容は、前項の案件について、次のとおりとする。"
    return {
        "conclusion": claim_for(blocks, "委員会は、費用削減の可否と仕様の妥当性を審査します。", rule),
        "sections": [
            {
                "heading": "審査する内容",
                "claims": [
                    claim_for(blocks, "費用削減が可能かどうかを審査します。", "費用削減の可否"),
                    claim_for(blocks, "仕様が妥当かどうかを審査します。", "仕様の妥当性"),
                ],
            }
        ],
    }


def test_run_answer_v2_builds_a_checked_answer_from_the_retrieved_evidence(corpus, monkeypatch):
    v2 = cli_search.run_retrieval_pipeline(SIMPLE_QUERY, model="m", answer_policy="v2")
    model = FakeModel(drafts=[simple_draft_for])
    captured = patch_chat(monkeypatch, model)
    outcome = cli_search.run_answer_v2(
        SIMPLE_QUERY, v2, model="m", total_seconds=300, cancel_check=lambda: None, touch=lambda: None
    )
    assert outcome.kind == "full" and outcome.quality["state"] == "passed"
    assert outcome.answer.splitlines()[0].startswith("委員会は、費用削減の可否と仕様の妥当性を審査します。")
    assert (model.count("draft"), model.count("g2")) == (1, 1)
    assert callable(captured["cancel_check"]) and callable(captured["touch"])


def test_an_aspect_without_evidence_is_reported_as_unconfirmed_not_invented(corpus, monkeypatch):
    """検索した根拠に背景の通知本文が無ければ、背景の観点は確認できなかった観点として示される。"""
    v2 = cli_search.run_retrieval_pipeline(QUERY, model="m", answer_policy="v2")
    assert not any("２割" in m["snippet"] for m in v2.matches)
    model = FakeModel(drafts=[complete_draft_for, complete_draft_for])
    patch_chat(monkeypatch, model)
    outcome = cli_search.run_answer_v2(QUERY, v2, model="m", total_seconds=300)
    assert outcome.kind == "partial" and outcome.quality["state"] == "partial"
    assert outcome.quality["unconfirmedAspects"] == ["設置の背景との区別"]
    assert "２割" not in outcome.answer and "## 確認できなかった観点" in outcome.answer


def test_a_changed_source_fails_g1_and_is_never_shown_as_a_checked_answer(corpus, monkeypatch):
    tmp_path, _ = corpus
    v2 = cli_search.run_retrieval_pipeline(SIMPLE_QUERY, model="m", answer_policy="v2")
    (tmp_path / SAMPLE).write_text(SOURCE_TEXT + "\n追記。\n", encoding="utf-8")
    model = FakeModel(drafts=[simple_draft_for, simple_draft_for])
    patch_chat(monkeypatch, model)
    outcome = cli_search.run_answer_v2(SIMPLE_QUERY, v2, model="m", total_seconds=300)
    assert outcome.kind == "fallback" and outcome.quality["state"] == "failed"
    assert model.count("g2") == 0  # G1で止まり、意味照合へ進まない
    assert outcome.answer.startswith("回答を作成できませんでした。")


def test_no_evidence_is_reported_as_not_found_not_as_a_generation_failure(corpus, monkeypatch):
    model = FakeModel()
    patch_chat(monkeypatch, model)
    empty = SimpleNamespace(plan={}, matches=[], evidence_status="insufficient")
    outcome = cli_search.run_answer_v2(QUERY, empty, model="m", total_seconds=300)
    assert outcome.answer == cli_search.ANSWER_V2_NO_EVIDENCE_TEXT
    assert outcome.quality["message"] == "質問に該当する資料を確認できませんでした。"
    assert model.calls == []


def test_retrieval_signals_reach_the_draft_prompt_as_notes():
    retrieval = SimpleNamespace(
        evidence_status="partial",
        matches=[{"constraint_conflict": True}, {"source": "expanded", "group_partial": True}],
    )
    notes = "\n".join(cli_search.answer_v2_note_lines(retrieval))
    assert "根拠ステータス: partial" in notes
    assert "限定条件の明示的な矛盾" in notes and "部分展開" in notes
    assert cli_search.answer_v2_note_lines(SimpleNamespace(evidence_status="sufficient", matches=[])) == []


# --- Web SSE --------------------------------------------------------------------------


def run_web(monkeypatch, *, mode="answer", query=QUERY):
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "fake-model")
    token = web_services.CancellationToken(timeout=300)
    entry = web_services.JobEntry(token)
    subscriber = entry.add_subscriber()
    web_services.run_search(query, "off", web_services._BroadcastQueue(entry), token, "rid", mode=mode)
    events = []
    while not subscriber.empty():
        events.append(subscriber.get_nowait())
    return events


def test_web_answer_v2_emits_one_checked_answer_and_the_quality_on_done(corpus, monkeypatch):
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    model = FakeModel(drafts=[simple_draft_for])
    patch_chat(monkeypatch, model)
    events = run_web(monkeypatch, query=SIMPLE_QUERY)
    types = [e["type"] for e in events]
    assert "thinking" not in types and "answer_empty" not in types and types[-1] == "done"
    meta = next(e for e in events if e["type"] == "result_meta")
    assert meta["answerPolicy"] == "v2"
    chunks = [e["text"] for e in events if e["type"] == "chunk"]
    assert len(chunks) == 1 and chunks[0].splitlines()[0].startswith("委員会は、費用削減の可否と仕様の妥当性を審査します。")
    done = events[-1]
    assert done["answerQuality"]["state"] == "passed"
    assert done["answerQuality"]["label"] == "検査済み"
    # 検索結果の根拠一覧は従来どおり先に届く。
    assert types.index("evidence") < types.index("chunk")


def test_web_answer_v2_marks_needs_review_when_a_check_cannot_run(corpus, monkeypatch):
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    model = FakeModel(
        drafts=[simple_draft_for, simple_draft_for],
        g2=lambda claims: {c["claim_id"]: ("uncertain", "other") for c in claims},
    )
    patch_chat(monkeypatch, model)
    events = run_web(monkeypatch, query=SIMPLE_QUERY)
    done = events[-1]
    assert done["answerQuality"]["state"] == "failed"
    assert done["answerQuality"]["label"] == "要確認"
    body = "".join(e["text"] for e in events if e["type"] == "chunk")
    assert body.startswith("回答を作成できませんでした。") and "## 参考の原文" in body


def test_web_invalid_policy_is_a_fixed_error_before_any_work(monkeypatch):
    monkeypatch.setenv(ap.POLICY_ENV, "v9")
    events = run_web(monkeypatch)
    assert events[-1]["type"] == "error" and events[-1]["code"] == "invalid_answer_policy"


def deep_result(state, quality):
    return deep_research.DeepResearchResult(
        status="partial", stop_reason="scope_processed", answer="回答", answer_state=state,
        answer_quality=quality,
    )


def test_deep_event_never_calls_an_unchecked_answer_verified():
    legacy = web_services.build_deep_evidence_event(deep_result("generated", {}))
    assert legacy["answerStateMessage"] == "回答生成済み（原文照合を通過）" and "answerQuality" not in legacy
    review = ap.build_quality({gid: ap.GateResult(ap.PASSED) for gid in ap.GATE_IDS}, kind="full", removed_claims=1)
    event = web_services.build_deep_evidence_event(deep_result("generated", review))
    assert event["answerQuality"]["state"] == "needs_review"
    assert "要確認" in event["answerStateMessage"] and "原文照合を通過" not in event["answerStateMessage"]
    passed = ap.build_quality({gid: ap.GateResult(ap.PASSED) for gid in ap.GATE_IDS}, kind="full")
    message = web_services.build_deep_evidence_event(deep_result("generated", passed))["answerStateMessage"]
    assert "保証ではありません" in message
    failed = web_services.build_deep_evidence_event(
        deep_result("verification_failed", ap.build_quality({}, kind="fallback"))
    )
    assert "要確認" in failed["answerStateMessage"]


def test_web_deep_done_event_carries_the_quality(monkeypatch):
    quality = ap.build_quality({gid: ap.GateResult(ap.PASSED) for gid in ap.GATE_IDS}, kind="full")
    result = deep_research.DeepResearchResult(
        status="completed", stop_reason="scope_processed", answer="回答", answer_state="generated",
        answer_quality=quality,
    )
    seen = {}

    def fake_deep(*args, **kwargs):
        seen.update(kwargs)
        return result

    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    monkeypatch.setattr(web_services, "detect_deep_model", lambda: "fake")
    monkeypatch.setattr(web_services, "run_deep_research", fake_deep)
    events = run_web(monkeypatch, mode="deep")
    assert seen["answer_policy"] == "v2"
    assert events[-1]["type"] == "done" and events[-1]["answerQuality"]["state"] == "passed"
    assert next(e for e in events if e["type"] == "result_meta")["answerPolicy"] == "v2"


# --- CLI ------------------------------------------------------------------------------


def patch_cli(monkeypatch, outcome, *, platform=None):
    retrieval = SimpleNamespace(
        attempts=[], matches=[], evidence_status="sufficient", confidence=0.5, plan={},
        user_prompt="", verification_status="skipped", retrieval_status="sufficient",
    )
    captured = {}
    if platform:
        monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(cli_search, "_is_model_available", lambda *a: cli_search.ModelStatus.AVAILABLE)
    monkeypatch.setattr(cli_search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(cli_search, "run_retrieval_pipeline", lambda *a, **k: captured.update(k) or retrieval)

    def fake_v2(query, retrieved, **kwargs):
        captured["v2"] = kwargs
        return outcome

    monkeypatch.setattr(cli_search, "run_answer_v2", fake_v2)
    return captured


def review_outcome():
    quality = ap.build_quality({gid: ap.GateResult(ap.PASSED) for gid in ap.GATE_IDS}, kind="partial", unconfirmed_aspects=["観点A"])
    return apipe.PipelineResult("検査済みの本文。[E1]", "partial", quality, {}, ["E1"])


def test_cli_v2_without_a_limit_gets_an_explicit_300_second_quality_budget(monkeypatch, capsys):
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    captured = patch_cli(monkeypatch, review_outcome(), platform="linux")
    monkeypatch.setattr(sys, "argv", ["search.py", "質問", "--model", "some-model"])
    cli_search.main()
    out = capsys.readouterr().out
    assert "時間上限 300 秒" in out and "時間上限: 300 秒" in out
    assert captured["answer_policy"] == "v2"
    assert captured["v2"]["total_seconds"] == 300
    assert "検査済みの本文。[E1]" in out
    assert "品質状態: 要確認" in out and "未確認の範囲: 観点A" in out


def test_cli_legacy_keeps_its_unlimited_default(monkeypatch, capsys):
    monkeypatch.delenv(ap.POLICY_ENV, raising=False)
    captured = patch_cli(monkeypatch, review_outcome(), platform="linux")
    streamed = []
    monkeypatch.setattr(cli_search, "stream_ollama_chat", lambda *a, **k: streamed.append(k))
    monkeypatch.setattr(sys, "argv", ["search.py", "質問", "--model", "some-model"])
    cli_search.main()
    out = capsys.readouterr().out
    assert "時間上限" not in out and "v2" not in captured and streamed
    assert captured["answer_policy"] == "legacy"


def test_cli_invalid_policy_exits_with_code_2(monkeypatch, capsys):
    monkeypatch.setenv(ap.POLICY_ENV, "nope")
    monkeypatch.setattr(sys, "argv", ["search.py", "質問", "--model", "some-model"])
    with pytest.raises(SystemExit) as exc:
        cli_search.main()
    assert exc.value.code == 2
    assert "OFFLINE_AI_ANSWER_POLICY_VERSION" in capsys.readouterr().err


def test_source_reader_rejects_changed_and_out_of_root_files(tmp_path):
    (tmp_path / "a.md").write_text("一行目\n二行目", encoding="utf-8")
    sha = hashlib.sha256((tmp_path / "a.md").read_bytes()).hexdigest()
    reader = cli_search.SkillSourceLineReader(tmp_path)
    assert reader("a.md", sha) == ["一行目", "二行目"]
    assert reader("a.md", "0" * 64) is None
    assert reader("../outside.md", "") is None
    assert reader("missing.md", "") is None


# --- 中止・期限（v2でも既存の終端契約を保つ） -------------------------------------------


def test_web_cancel_during_the_semantic_check_ends_with_cancelled_and_no_answer(corpus, monkeypatch):
    monkeypatch.setenv(ap.POLICY_ENV, "v2")
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "fake-model")
    token = web_services.CancellationToken(timeout=300)
    model = FakeModel(drafts=[simple_draft_for])
    original = model.__call__

    def cancelling(system, user, *, timeout):
        value = original(system, user, timeout=timeout)
        if system.startswith("あなたはローカル資料の調査結果を"):
            token.cancel()  # 草稿生成の直後に利用者が中止する
        return value

    patch_chat(monkeypatch, cancelling)
    entry = web_services.JobEntry(token)
    subscriber = entry.add_subscriber()
    web_services.run_search(SIMPLE_QUERY, "off", web_services._BroadcastQueue(entry), token, "rid")
    events = []
    while not subscriber.empty():
        events.append(subscriber.get_nowait())
    assert events[-1]["type"] == "error" and events[-1]["code"] == "cancelled"
    assert not any(e["type"] in {"chunk", "done"} for e in events)


def test_a_deadline_during_v2_generation_is_not_swallowed_as_a_failed_call():
    import answer_pipeline as pipeline_module

    class Expired(Exception):
        code = "timeout"

    def call(system, user, *, timeout):
        raise Expired()

    def check():
        raise Expired()

    from answer_fixtures import make_env, make_evidence

    env = make_env(call)
    env.check = check
    env.is_control_error = lambda exc: isinstance(exc, Expired)
    with pytest.raises(Expired):
        pipeline_module.run_answer_pipeline(QUERY, ap.extract_explicit_intent(QUERY), make_evidence(), env)
