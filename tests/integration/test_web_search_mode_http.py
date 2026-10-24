"""Fake backend を用いた search-only HTTP/SSE の統合検証。"""

import http.client
import json
import threading
from urllib.parse import quote

import source_view
import web_server


def test_search_only_http_stream_has_no_chat_generation(monkeypatch, tmp_path):
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="integration-session",
        search_timeout=300,
        evidence_registry=source_view.EvidenceRegistry(tmp_path),
    )
    server.bind_port = server.server_address[1]
    calls = []

    def fake_run_search(
        query,
        reasoning,
        event_queue,
        cancel_token,
        request_id,
        *,
        mode="answer",
        started_at=None,
        session_id="",
        evidence_registry=None,
    ):
        calls.append((query, reasoning, mode))
        event_queue.put(
            {
                "type": "result_meta",
                "requestId": request_id,
                "mode": mode,
                "startedAt": started_at,
                "chatModel": None,
                "embeddingModel": None,
                "reasoning": "off",
            }
        )
        event_queue.put(
            {
                "type": "evidence",
                "evidenceStatus": "insufficient",
                "confidence": 0.0,
                "route": "keyword",
                "indexState": "unknown",
                "items": [],
                "warnings": [],
            }
        )
        event_queue.put({"type": "done", "completedAt": started_at})

    monkeypatch.setattr(web_server, "run_search", fake_run_search)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
        conn.request(
            "GET",
            f"/api/search?q={quote('本文')}&mode=search&request_id=integration",
            headers={
                "Host": f"127.0.0.1:{server.bind_port}",
                "Cookie": "offlineai_session=integration-session",
            },
        )
        response = conn.getresponse()
        payload = response.read().decode("utf-8")
        conn.close()
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: ")]
    assert response.status == 200
    assert calls == [("本文", "", "search")]
    assert [event["type"] for event in events if event["type"] != "budget"] == [
        "result_meta",
        "evidence",
        "done",
    ]
