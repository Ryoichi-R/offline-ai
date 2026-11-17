"""確認案内と追質問の HTTP/SSE 統合検証（偽の回答処理、モデル呼び出しなし）。"""

import http.client
import json
import threading
from urllib.parse import quote

import pytest
import source_folder
import source_view
import web_server
import web_services

NOTICE_LINES = (
    "# 会計事務における押印の取扱いについて（通知）",
    "各部局会計課長 殿",
    "令和6年4月1日",
    "",
    "## 1 趣旨",
    "請求書等の押印を原則として求めないこととする。",
    "## 2 提出書類",
    "請求書と領収書を提出する。",
    "- 請求書の提出期限は翌月10日とする。",
    "- ただし外国法人の請求書は押印を求めることがある。",
    "## 3 領収書の保存",
    "領収書は5年間保存する。保存の期限は会計年度末から起算する。",
    "## 4 適用日",
    "令和6年4月1日から適用する。",
    "",
)
NOTICE = "\n".join(NOTICE_LINES)
BROAD = "会計事務における押印の取扱いについての内容詳細を教えて"
SESSION = "guided-session"


@pytest.fixture
def guided_server(monkeypatch, tmp_path):
    root = tmp_path / "skill-source"
    root.mkdir()
    (root / "notice.md").write_text(NOTICE, encoding="utf-8")
    monkeypatch.setenv("OFFLINE_AI_GUIDED_FOLLOW_UP", "1")
    monkeypatch.setattr(web_services, "SKILL_SOURCE_DIR", root)
    monkeypatch.setattr(source_folder, "load_source_root", lambda project_root: root)
    calls = []

    def fake_core(query, reasoning, event_queue, cancel_token, request_id, **kwargs):
        calls.append(
            {"query": query, "fixed": kwargs.get("fixed_document"), "mode": kwargs.get("mode")}
        )
        event_queue.put(
            {"type": "result_meta", "requestId": request_id, "mode": kwargs.get("mode")}
        )
        event_queue.put({"type": "chunk", "text": "回答"})
        event_queue.put({"type": "done"})

    monkeypatch.setattr(web_services, "_run_search_core", fake_core)
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token=SESSION,
        search_timeout=300,
        evidence_registry=source_view.EvidenceRegistry(root),
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, calls, root
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(server, method, path, body=None, session=SESSION):
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    headers = {"Host": f"127.0.0.1:{server.bind_port}", "Cookie": f"offlineai_session={session}"}
    data = None
    if body is not None:
        data = (
            body
            if isinstance(body, bytes)
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        headers["Content-Type"] = "application/json"
        headers["Origin"] = f"http://127.0.0.1:{server.bind_port}"
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()
    payload = response.read().decode("utf-8")
    conn.close()
    events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: ")]
    return response.status, events, payload


def _types(events):
    return [event["type"] for event in events if event["type"] not in {"budget", "status"}]


def _start(server):
    status, events, _ = _request(
        server, "GET", f"/api/search?q={quote(BROAD)}&mode=answer&request_id=g1"
    )
    assert status == 200
    return next(event for event in events if event["type"] == "clarification")


def _continue(server, clarification, request_id, **fields):
    body = {
        "confirmationId": clarification["confirmationId"],
        "mode": "answer",
        "requestId": request_id,
        **fields,
    }
    return _request(server, "POST", "/api/search", body)


def test_broad_question_returns_clarification_without_answer_processing(guided_server):
    server, calls, _ = guided_server
    status, events, _ = _request(
        server, "GET", f"/api/search?q={quote(BROAD)}&mode=answer&request_id=g1"
    )
    assert status == 200
    assert _types(events) == ["clarification", "done"]
    done = events[-1]
    assert done["stage"] == "clarify"
    clarification = events[-2]
    assert clarification["kind"] == "topics"
    assert clarification["document"]["title"].startswith("会計事務における押印の取扱い")
    assert [o["label"] for o in clarification["options"]][:3] == [
        "1 趣旨",
        "2 提出書類",
        "3 領収書の保存",
    ]
    assert calls == []  # 確認段階ではモデル検出・回答処理へ進まない


def test_specific_question_goes_straight_to_the_existing_path(guided_server):
    server, calls, _ = guided_server
    query = "会計事務における押印の取扱いについての適用日はいつですか"
    status, events, _ = _request(
        server, "GET", f"/api/search?q={quote(query)}&mode=answer&request_id=g2"
    )
    assert status == 200 and _types(events) == ["result_meta", "chunk", "done"]
    assert calls == [{"query": query, "fixed": None, "mode": "answer"}]


def test_guided_overview_links_to_the_original_lines_and_restores_access(guided_server):
    server, _, root = guided_server
    initial = _start(server)
    status, _, payload = _request(
        server, "GET", f"/api/confirmation?id={initial['confirmationId']}"
    )
    assert status == 200
    restored = json.loads(payload)["confirmation"]
    for panel in (initial, restored):
        point = panel["points"][0]
        status, _, body = _request(
            server, "GET", f"/api/evidence/view?evidence_id={point['evidenceId']}"
        )
        assert status == 200
        evidence = json.loads(body)
        assert (evidence["highlightStartLine"], evidence["highlightEndLine"]) == (
            point["startLine"],
            point["endLine"],
        )
        assert any(
            line["highlighted"] and line["text"] == NOTICE_LINES[5] for line in evidence["lines"]
        )
        status, _, _ = _request(
            server, "GET", f"/api/evidence/view?evidence_id={point['evidenceId']}", session="other"
        )
        assert status == 403

    (root / "notice.md").write_text(NOTICE + "追記\n", encoding="utf-8")
    status, _, body = _request(
        server, "GET", f"/api/evidence/view?evidence_id={restored['points'][0]['evidenceId']}"
    )
    assert status == 409 and json.loads(body)["error"]["code"] == "source_changed"


def test_topic_free_input_and_short_follow_up_keep_the_same_document(guided_server):
    server, calls, _ = guided_server
    clarification = _start(server)
    status, events, _ = _continue(
        server,
        clarification,
        "g3",
        action="select_topic",
        choice=1,
        optionsVersion=clarification["optionsVersion"],
    )
    assert status == 200
    assert _types(events) == ["result_meta", "chunk", "confirmation_state", "done"], events
    assert calls[-1]["fixed"]["path"] == "notice.md"
    assert calls[-1]["fixed"]["focus_ranges"] == [[7, 10, "notice.md"]]
    assert calls[-1]["fixed"]["focus_only"] is True
    state = next(e for e in events if e["type"] == "confirmation_state")
    assert state["topic"]["section"]["label"] == "2 提出書類"

    status, events, _ = _continue(
        server, clarification, "g4", action="follow_up", text="請求書だけ"
    )
    assert "focus_only" not in calls[-1]["fixed"]
    state = next(e for e in events if e["type"] == "confirmation_state")
    assert state["topic"]["targets"] == ["請求書"]
    status, events, _ = _continue(server, clarification, "g5", action="follow_up", text="期限は？")
    state = next(e for e in events if e["type"] == "confirmation_state")
    assert state["topic"]["targets"] == ["請求書"]
    assert calls[-1]["fixed"]["focus_ranges"] == [[7, 10, "notice.md"]]
    assert all(call["fixed"]["path"] == "notice.md" for call in calls)

    # 同じ入力で再購読しても処理を重複実行しない。
    count = len(calls)
    status, events, _ = _continue(
        server, clarification, "g5", action="follow_up", text="期限は？", resume_only=True
    )
    assert status == 200 and len(calls) == count
    assert "done" in _types(events)


def test_panel_restore_and_errors(guided_server):
    server, _, root = guided_server
    clarification = _start(server)
    cid = clarification["confirmationId"]
    status, _, payload = _request(server, "GET", f"/api/confirmation?id={cid}")
    assert status == 200 and json.loads(payload)["confirmation"]["confirmationId"] == cid
    status, _, _ = _request(server, "GET", f"/api/confirmation?id={cid}", session="other")
    assert status == 403

    status, _, payload = _continue(
        server,
        clarification,
        "e1",
        action="select_topic",
        choice=0,
        optionsVersion=clarification["optionsVersion"] - 1,
    )
    assert status == 409 and json.loads(payload)["error"]["code"] == "confirmation_invalid"

    status, _, payload = _continue(
        server, clarification, "e2", action="follow_up", text="あ" * 1001
    )
    assert status == 400 and json.loads(payload)["error"]["code"] == "invalid_query"
    status, _, _ = _continue(server, clarification, "e3", action="follow_up", text="あ" * 1000)
    assert status == 200  # 日本語1000文字はPOSTで送信できる
    oversized = json.dumps(
        {"confirmationId": cid, "action": "follow_up", "text": "x" * 9000}
    ).encode()
    status, _, payload = _request(server, "POST", "/api/search", oversized)
    assert status == 400

    (root / "notice.md").write_text(NOTICE + "追記\n", encoding="utf-8")
    status, events, _ = _continue(server, clarification, "e4", action="follow_up", text="期限は？")
    assert [e.get("code") for e in events if e["type"] == "error"] == ["source_changed"]
    status, _, payload = _request(server, "GET", f"/api/confirmation?id={cid}")
    assert status == 410 and json.loads(payload)["error"]["code"] == "confirmation_expired"


def test_guided_endpoints_are_absent_when_disabled(guided_server, monkeypatch):
    server, calls, _ = guided_server
    monkeypatch.delenv("OFFLINE_AI_GUIDED_FOLLOW_UP")
    status, _, _ = _request(
        server, "POST", "/api/search", {"confirmationId": "x" * 24, "action": "follow_up"}
    )
    assert status == 404
    status, events, _ = _request(
        server, "GET", f"/api/search?q={quote(BROAD)}&mode=answer&request_id=d1"
    )
    assert _types(events) == ["result_meta", "chunk", "done"]
    assert calls[-1]["fixed"] is None
