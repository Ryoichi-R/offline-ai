"""確認案内と追質問（guided-follow-up 段階2）の判定・抽出・状態管理。"""

import guided_follow_up as gf
import pytest
import search
import web_services

NOTICE_LINES = (
    "# 会計事務における押印の取扱いについて（通知）",  # 1
    "各部局会計課長 殿",  # 2
    "令和6年4月1日",  # 3
    "",  # 4
    "## 1 趣旨",  # 5
    "請求書等の押印を原則として求めないこととする。",  # 6
    "## 2 提出書類",  # 7
    "請求書と領収書を提出する。",  # 8
    "- 請求書の提出期限は翌月10日とする。",  # 9
    "- ただし外国法人の請求書は押印を求めることがある。",  # 10
    "## 3 領収書の保存",  # 11
    "領収書は5年間保存する。保存の期限は会計年度末から起算する。",  # 12
    "## 4 適用日",  # 13
    "令和6年4月1日から適用する。",  # 14
    "## 5 照会先",  # 15
    "会計課 契約係（内線1234）",  # 16
    "",
)
NOTICE = "\n".join(NOTICE_LINES)
TITLE = "会計事務における押印の取扱いについて（通知）"


def _root(tmp_path, **extra):
    root = tmp_path / "skill-source"
    root.mkdir()
    (root / "notice.md").write_text(NOTICE, encoding="utf-8")
    for name, text in extra.items():
        (root / f"{name}.md").write_text(text, encoding="utf-8")
    return root


def _decide(tmp_path, query, **extra):
    root = _root(tmp_path, **extra)
    return gf.decide(query, search.build_source_chunks(root), root), root


def _document():
    return {"path": "notice.md", "title": TITLE, "sha256": "x", "text": NOTICE}


# --- 振り分け -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("会計事務における押印の取扱いについての内容詳細を教えて", "topics"),
        ("会計事務における押印の取扱いのポイント", "topics"),
        ("会計事務における押印の取扱いについての適用日はいつですか", "direct"),
        ("会計事務における押印の取扱いについての例外を詳しく", "direct"),
        ("会計事務における押印の取扱いについての全体をまとめて", "direct"),
    ],
)
def test_confirmed_document_routes_by_question_breadth(tmp_path, query, kind):
    decision, _ = _decide(tmp_path, query)
    assert decision.kind == kind


def test_broad_question_without_formal_title_asks_for_hints(tmp_path):
    decision, _ = _decide(tmp_path, "会計課の押印の取扱いについて詳しく")
    assert decision.kind == "hints"
    assert "発出者" in decision.message


def test_specific_question_without_document_goes_direct(tmp_path):
    decision, _ = _decide(tmp_path, "旅費の上限はいくらですか")
    assert decision.kind == "direct"


def test_same_title_documents_are_offered_as_candidates(tmp_path):
    decision, _ = _decide(
        tmp_path,
        "押印の見直しについての内容",
        body="# 押印の見直しについて（通知）\n令和5年4月1日\n本体\n",
        annex="# 押印の見直しについて（別添）\n別添の本文\n",
    )
    assert decision.kind == "documents"
    assert {card["path"] for card in decision.candidates} == {"body.md", "annex.md"}
    assert any(card.get("kind") == "別添" for card in decision.candidates)
    assert any(card.get("date") == "令和5年4月1日" for card in decision.candidates)


def test_topics_overview_has_main_items_not_only_date_and_contact(tmp_path):
    decision, _ = _decide(tmp_path, "会計事務における押印の取扱いについての内容詳細")
    texts = [p.get("text", "") for p in decision.overview["points"]]
    joined = "\n".join(texts)
    assert "押印を原則として求めない" in joined
    assert "令和6年4月1日から適用する" not in joined and "内線1234" not in joined
    assert 2 <= len(decision.overview["points"]) <= 4
    # 原則の直後の箇条書きにある例外を一緒に抜き出す。
    assert any("外国法人" in text and "請求書と領収書を提出する" in text for text in texts)
    labels = [o["label"] for o in decision.overview["options"]]
    assert labels[:3] == ["1 趣旨", "2 提出書類", "3 領収書の保存"]
    assert labels[-1] == "全体をまとめて"
    assert "4 適用日" not in labels


def test_long_paragraph_keeps_leading_sentences_unless_a_qualifier_would_be_dropped():
    long_text = "# 表題\n## 本文\n" + "長い本文。" * 200 + "\n"
    point = gf.build_overview(long_text)["points"][0]
    assert point["kind"] == "excerpt" and point["truncated"] is True
    assert point["text"].endswith("。") and len(point["text"]) <= gf.OVERVIEW_POINT_CHARS * 2
    qualified = (
        "# 表題\n## 本文\n" + "原則として押印は不要とする。" * 30 + "ただし外国法人は除く。\n"
    )
    assert gf.build_overview(qualified)["points"][0]["kind"] == "guide"
    one_sentence = "# 表題\n## 本文\n" + "あ" * 500 + "。\n"
    assert gf.build_overview(one_sentence)["points"][0]["kind"] == "guide"


def test_headingless_document_has_points():
    plain = gf.build_overview("第一段落。\n\n第二段落。\n\n第三段落。\n")
    assert [p["text"] for p in plain["points"]][:2] == ["第一段落。", "第二段落。"]
    assert [o["kind"] for o in plain["options"]] == ["overview"]


# --- 追質問の対象と論点 ------------------------------------------------------------


def _options():
    return gf.build_overview(NOTICE)["options"]


def test_targets_carry_over_to_aspect_only_follow_up():
    doc = _document()
    topic = gf.Topic()
    first = gf.resolve_follow_up(
        doc, topic, _options(), action=gf.ACTION_FOLLOW_UP, text="提出書類"
    )
    assert first.ask is None and first.topic.targets == ()
    second = gf.resolve_follow_up(
        doc, first.topic, _options(), action=gf.ACTION_FOLLOW_UP, text="請求書だけ"
    )
    assert second.topic.targets == ("請求書",)
    third = gf.resolve_follow_up(
        doc, second.topic, _options(), action=gf.ACTION_FOLLOW_UP, text="期限は？"
    )
    assert third.ask is None
    assert third.topic.targets == ("請求書",)
    # 請求書の期限がある「提出書類」の節に絞る（領収書の保存期限の節ではない）。
    assert third.focus_ranges == [(7, 10, "notice.md")]
    assert "請求書" in third.query and "期限は？" in third.query


def test_replace_add_and_drop_targets():
    text = "# 規程\n## 書類\n請求書と領収書と見積書を扱う。期限は翌月10日。\n"
    doc = {"path": "a.md", "title": "規程", "sha256": "x", "text": text}
    options = gf.build_overview(text)["options"]
    base = gf.Topic(None, ("請求書",))
    added = gf.resolve_follow_up(doc, base, options, action=gf.ACTION_FOLLOW_UP, text="領収書も")
    assert added.topic.targets == ("請求書", "領収書")
    replaced = gf.resolve_follow_up(
        doc, added.topic, options, action=gf.ACTION_FOLLOW_UP, text="領収書のみ"
    )
    assert replaced.topic.targets == ("領収書",)
    dropped = gf.resolve_follow_up(
        doc,
        added.topic,
        options,
        action=gf.ACTION_FOLLOW_UP,
        text="期限は？",
        drop_targets=("請求書",),
    )
    assert dropped.topic.targets == ("領収書",)


def test_selecting_a_section_clears_targets_and_numbers_map_to_options():
    doc, options = _document(), _options()
    topic = gf.Topic(None, ("請求書",))
    chosen = gf.resolve_follow_up(doc, topic, options, action=gf.ACTION_SELECT_TOPIC, choice=2)
    assert chosen.topic.section["label"] == "3 領収書の保存" and chosen.topic.targets == ()
    assert chosen.focus_ranges == [(11, 12, "notice.md")]
    by_number = gf.resolve_follow_up(doc, topic, options, action=gf.ACTION_FOLLOW_UP, text="3")
    assert by_number.topic.section == chosen.topic.section
    overview = gf.resolve_follow_up(
        doc, topic, options, action=gf.ACTION_SELECT_TOPIC, choice=len(options) - 1
    )
    assert overview.overview is True
    with pytest.raises(gf.ConfirmationError) as error:
        gf.resolve_follow_up(doc, topic, options, action=gf.ACTION_SELECT_TOPIC, choice=99)
    assert error.value.code == "confirmation_invalid"


def test_target_in_several_places_is_asked_back_and_unknown_terms_are_reported():
    doc, options = _document(), _options()
    asked = gf.resolve_follow_up(
        doc, gf.Topic(), options, action=gf.ACTION_FOLLOW_UP, text="領収書について"
    )
    assert asked.ask is not None and len(asked.ask["options"]) == 2
    missing = gf.resolve_follow_up(
        doc, gf.Topic(), options, action=gf.ACTION_FOLLOW_UP, text="旅費規程"
    )
    assert missing.ask is not None and "旅費規程" in missing.ask["message"]
    assert missing.topic.targets == ()


def test_follow_up_text_limits():
    assert gf.normalize_follow_up_text("あ" * 1000) == "あ" * 1000
    for value in ("", "あ" * 1001, None):
        with pytest.raises(gf.ConfirmationError):
            gf.normalize_follow_up_text(value)


# --- 確認情報の状態管理 ---------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _fixed_decision():
    return gf.Decision(
        "topics",
        document={
            "path": "notice.md",
            "title": TITLE,
            "sha256": "x",
            "sourceRoot": "/s",
            "text": NOTICE,
        },
        overview=gf.build_overview(NOTICE),
    )


def test_expiry_is_extended_only_by_completion_and_view_does_not_extend():
    clock = Clock()
    store = gf.ConfirmationStore(clock=clock)
    record = store.create("s", "q", _fixed_decision())
    cid = record.confirmation_id
    clock.now += 29 * 60
    store.view(cid, "s")
    store.begin(cid, "s", action=gf.ACTION_FOLLOW_UP, options_version=None, request_id="r1")
    assert store.finish(cid, "r1", completed=True) is not None  # 正常完了で延長
    clock.now += 29 * 60
    store.view(cid, "s")  # 閲覧は延長しない
    clock.now += 2 * 60
    with pytest.raises(gf.ConfirmationError) as error:
        store.view(cid, "s")
    assert error.value.code == "confirmation_expired"


def test_max_age_and_limit_and_no_resurrection():
    clock = Clock()
    store = gf.ConfirmationStore(clock=clock, limit=2)
    first = store.create("s", "q1", _fixed_decision())
    store.create("s", "q2", _fixed_decision())
    clock.now += 1
    store.create("s", "q3", _fixed_decision())
    with pytest.raises(gf.ConfirmationError) as error:
        store.view(first.confirmation_id, "s")
    assert error.value.code == "confirmation_expired"

    store = gf.ConfirmationStore(clock=clock)
    record = store.create("s", "q", _fixed_decision())
    cid = record.confirmation_id
    for _ in range(4):
        store.begin(cid, "s", action=gf.ACTION_FOLLOW_UP, options_version=None, request_id="r")
        clock.now += 25 * 60
        store.finish(cid, "r", completed=True)
    store.begin(cid, "s", action=gf.ACTION_FOLLOW_UP, options_version=None, request_id="r")
    clock.now += 25 * 60  # 作成から2時間を超える（最終完了からは30分以内）
    assert store.finish(cid, "r", completed=True) is None
    with pytest.raises(gf.ConfirmationError):
        store.view(cid, "s")


def test_busy_session_state_and_stale_options_are_rejected():
    store = gf.ConfirmationStore()
    record = store.create("s", "q", _fixed_decision())
    cid, version = record.confirmation_id, record.options_version
    with pytest.raises(gf.ConfirmationError) as error:
        store.begin(cid, "other", action=gf.ACTION_FOLLOW_UP, options_version=None, request_id="r")
    assert error.value.code == "confirmation_invalid"
    with pytest.raises(gf.ConfirmationError) as error:
        store.begin(
            cid, "s", action=gf.ACTION_SELECT_DOCUMENT, options_version=version, request_id="r"
        )
    assert error.value.code == "confirmation_invalid"
    with pytest.raises(gf.ConfirmationError) as error:
        store.begin(
            cid, "s", action=gf.ACTION_SELECT_TOPIC, options_version=version - 1, request_id="r"
        )
    assert error.value.code == "confirmation_invalid"
    store.begin(cid, "s", action=gf.ACTION_SELECT_TOPIC, options_version=version, request_id="r1")
    with pytest.raises(gf.ConfirmationError) as error:
        store.begin(cid, "s", action=gf.ACTION_FOLLOW_UP, options_version=None, request_id="r2")
    assert error.value.code == "confirmation_busy"
    # 失敗・中止では論点を更新しない。
    assert store.finish(cid, "r1", topic=gf.Topic(None, ("請求書",)), completed=False) is None
    assert store.view(cid, "s")["topic"]["targets"] == []
    with pytest.raises(gf.ConfirmationError) as error:
        store.view("bad id!", "s")
    assert error.value.code == "confirmation_invalid"


def test_public_view_never_contains_document_text_or_hashes():
    store = gf.ConfirmationStore()
    public = store.create("s", "q", _fixed_decision()).public()
    assert "text" not in public["document"] and "sha256" not in public["document"]
    pending = store.create(
        "s",
        "q",
        gf.Decision("documents", candidates=[{"path": "a.md", "title": "A", "fileSha256": "x"}]),
    )
    assert pending.public()["candidates"] == [{"path": "a.md", "title": "A"}]


# --- 深掘り枠 -------------------------------------------------------------------------


def test_deferred_deep_jobs_do_not_hold_the_slot_until_research_starts():
    table = web_services.JobTable(max_concurrent=3)
    try:

        def token():
            return web_services.CancellationToken(300)

        clarify_a, _ = table.submit(
            "a", token(), fingerprint=("s", "q", "", 300, "deep"), mode="deep", defer_deep_slot=True
        )
        clarify_b, _ = table.submit(
            "b",
            token(),
            fingerprint=("s", "q2", "", 300, "deep"),
            mode="deep",
            defer_deep_slot=True,
        )
        assert table.acquire_deep_slot("a") is True
        assert table.acquire_deep_slot("a") is True  # 同じジョブの再取得
        assert table.acquire_deep_slot("b") is False
        with pytest.raises(RuntimeError):
            table.submit("c", token(), fingerprint=("s", "q3", "", 300, "deep"), mode="deep")
        clarify_a.broadcast({"type": "done"})  # 正常終了で枠が空く
        assert table.acquire_deep_slot("b") is True
        clarify_b.broadcast({"type": "error", "code": "cancelled"})
        assert table.acquire_deep_slot("missing") is False
        table.submit("d", token(), fingerprint=("s", "q4", "", 300, "deep"), mode="deep")
    finally:
        table.shutdown()


def test_deep_research_starts_only_after_taking_the_slot(monkeypatch):
    import queue

    import deep_research

    calls = []
    monkeypatch.setattr(web_services, "detect_deep_model", lambda: "test-model")
    monkeypatch.setattr(
        web_services,
        "run_deep_research",
        lambda query, **kwargs: (
            calls.append(kwargs.get("fixed_document"))
            or deep_research.DeepResearchResult(status="completed", stop_reason="scope_processed")
        ),
    )
    busy = queue.Queue()
    web_services._run_search_core(
        "q",
        "off",
        busy,
        web_services.CancellationToken(300),
        "r1",
        mode="deep",
        acquire_deep_slot=lambda: False,
    )
    events = [busy.get_nowait() for _ in range(busy.qsize())]
    assert [e.get("code") for e in events if e["type"] == "error"] == ["deep_busy"]
    assert calls == []
    fixed = {"path": "notice.md", "sha256": "x", "focus_ranges": [[7, 10]], "primary": False}
    done = queue.Queue()
    web_services._run_search_core(
        "q",
        "off",
        done,
        web_services.CancellationToken(300),
        "r2",
        mode="deep",
        acquire_deep_slot=lambda: True,
        fixed_document=fixed,
    )
    assert calls == [fixed]
    assert any(e["type"] == "done" for e in [done.get_nowait() for _ in range(done.qsize())])


def test_disabled_run_search_calls_the_core_unchanged(monkeypatch):
    seen = []
    monkeypatch.delenv("OFFLINE_AI_GUIDED_FOLLOW_UP", raising=False)
    monkeypatch.setattr(
        web_services,
        "_run_search_core",
        lambda *a, **k: seen.append((a[0], k.get("fixed_document"))),
    )
    web_services.run_search(
        "質問", "off", None, None, "r", mode="deep", confirmation_store=gf.ConfirmationStore()
    )
    assert seen == [("質問", None)]
