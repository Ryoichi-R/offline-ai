import hashlib
import http.client
import json
import threading

import pytest

import source_view
import web_server
import web_services


@pytest.fixture
def running_server(tmp_path):
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="test-session-token",
        search_timeout=300,
        evidence_registry=source_view.EvidenceRegistry(tmp_path),
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, tmp_path
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(server, path):
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request(
        "GET",
        path,
        headers={
            "Host": f"127.0.0.1:{server.bind_port}",
            "Cookie": "offlineai_session=test-session-token",
        },
    )
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, json.loads(body.decode("utf-8"))


def test_empty_duplicate_and_invalid_mode_are_rejected_before_worker(running_server):
    server, _tmp_path = running_server
    status, body = _request(server, "/api/search?q=test&mode=")
    assert status == 400
    assert body["error"]["code"] == "invalid_mode"

    status, body = _request(server, "/api/search?q=test&mode=answer&mode=search")
    assert status == 400
    assert body["error"]["code"] == "invalid_mode"
    assert server.job_table.running_count == 0


def test_request_id_cannot_switch_mode_or_query(running_server, monkeypatch):
    server, _tmp_path = running_server

    def emit_done(query, reasoning, event_queue, cancel_token, request_id):
        event_queue.put({"type": "done", "completedAt": "2026-09-12T08:00:00.000Z"})

    monkeypatch.setattr(web_server, "run_search", emit_done)
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request(
        "GET",
        "/api/search?q=first&mode=answer&request_id=fixed",
        headers={
            "Host": f"127.0.0.1:{server.bind_port}",
            "Cookie": "offlineai_session=test-session-token",
        },
    )
    response = conn.getresponse()
    assert response.status == 200
    assert '"type": "done"' in response.read().decode("utf-8")
    conn.close()

    status, body = _request(server, "/api/search?q=second&mode=search&request_id=fixed")
    assert status == 409
    assert body["error"]["code"] == "request_conflict"


def test_evidence_view_endpoint_requires_auth_and_returns_line_window(running_server):
    server, tmp_path = running_server
    source = tmp_path / "guide.md"
    source.write_text("先頭\n該当\n末尾", encoding="utf-8")
    evidence_id = server.evidence_registry.register(
        session_id="test-session-token",
        relative_path="guide.md",
        start_line=2,
        end_line=2,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    status, body = _request(server, f"/api/evidence/view?evidence_id={evidence_id}")
    assert status == 200
    assert body["relativePath"] == "guide.md"
    assert body["lines"][1]["highlighted"] is True

    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request(
        "GET",
        f"/api/evidence/view?evidence_id={evidence_id}",
        headers={"Host": f"127.0.0.1:{server.bind_port}"},
    )
    response = conn.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    conn.close()
    assert response.status == 403
    assert body["error"]["code"] == "unauthorized"


def test_search_cancel_requires_owner_and_sets_server_token(running_server):
    server, _tmp_path = running_server
    token = web_services.CancellationToken(300)
    server.job_table.submit(
        "cancel-me",
        token,
        fingerprint=("test-session-token", "質問", "off", 300, "deep"),
        mode="deep",
    )
    payload = json.dumps({"request_id": "cancel-me"}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request(
        "POST",
        "/api/search/cancel",
        body=payload,
        headers={
            "Host": f"127.0.0.1:{server.bind_port}",
            "Cookie": "offlineai_session=test-session-token",
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    response = conn.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    conn.close()
    assert response.status == 200
    assert body["cancelled"] is True
    assert token.is_cancelled

    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request(
        "POST",
        "/api/search/cancel",
        body=payload,
        headers={
            "Host": f"127.0.0.1:{server.bind_port}",
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    response = conn.getresponse()
    body = json.loads(response.read().decode("utf-8"))
    conn.close()
    assert response.status == 403
    assert body["error"]["code"] == "unauthorized"



@pytest.mark.parametrize("suffix", [
    "&resume_only=", "&resume_only=0", "&resume_only=1&resume_only=1",
    "&resume_only=1&request_id=",
])
def test_invalid_resume_never_starts_worker(running_server, suffix):
    server, _ = running_server
    status, _body = _request(server, "/api/search?q=test" + suffix)
    assert status == 400
    assert server.job_table.running_count == 0


def test_missing_resume_after_restart_or_expiry_never_creates_job(running_server, monkeypatch):
    server, _ = running_server
    calls = []
    monkeypatch.setattr(web_server, "run_search", lambda *a, **k: calls.append(1))
    status, body = _request(server, "/api/search?q=test&request_id=lost&resume_only=1")
    assert status == 404
    assert body["error"]["code"] == "job_not_found"
    assert server.job_table.get("lost") is None
    assert server.job_table.running_count == 0
    assert calls == []


def test_resume_preserves_token_deadline_and_replays_terminal_without_worker(running_server, monkeypatch):
    server, _ = running_server
    calls = []
    monkeypatch.setattr(web_server, "run_search", lambda *a, **k: calls.append(1))
    token = web_services.CancellationToken(300)
    deadline = token.deadline
    fingerprint = ("test-session-token", "test", "off", 300, "deep")
    entry, _ = server.job_table.submit("existing", token, fingerprint=fingerprint, mode="deep")
    same, is_new = server.job_table.submit(
        "existing", web_services.CancellationToken(300),
        fingerprint=fingerprint, mode="deep", resume_only=True,
    )
    assert same is entry and not is_new
    assert same.cancel_token.deadline == deadline
    # Terminal replay must work without creating or starting any new generation.
    entry.broadcast({"type": "done", "status": "partial", "stopReason": "time_budget"})
    server.job_table.finish("existing")
    conn = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)
    conn.request("GET", "/api/search?q=test&mode=deep&reasoning=off&timeout_seconds=300&request_id=existing&resume_only=1",
                 headers={"Host": f"127.0.0.1:{server.bind_port}", "Cookie": "offlineai_session=test-session-token"})
    response = conn.getresponse()
    assert response.status == 200
    assert '"type": "done"' in response.read().decode("utf-8")
    conn.close()
    assert calls == []
    assert server.job_table.get("existing").cancel_token is token
    assert token.deadline == deadline
    status, _ = _request(server, "/api/search?q=changed&mode=deep&reasoning=off&request_id=existing&resume_only=1")
    assert status == 409
    # Expired completed entries are never resurrected by resume.
    entry.completed_at -= server.job_table.JOB_TTL + 1
    server.job_table._cleanup_expired()
    status, _ = _request(server, "/api/search?q=test&mode=deep&reasoning=off&request_id=existing&resume_only=1")
    assert status == 404
    assert calls == []


def test_resume_rejects_another_owner_atomically():
    table = web_services.JobTable()
    try:
        token = web_services.CancellationToken(300)
        table.submit("owner", token, fingerprint=("one", "q", "off", 300, "deep"))
        with pytest.raises(web_services.RequestConflictError):
            table.submit("owner", web_services.CancellationToken(300),
                         fingerprint=("two", "q", "off", 300, "deep"), resume_only=True)
        assert table.running_count == 1
        assert table.get("owner").cancel_token is token
    finally:
        table.shutdown()



def test_running_search_disconnect_resume_uses_one_worker(running_server, monkeypatch):
    server, _ = running_server
    release = threading.Event()
    calls = []

    def worker(query, reasoning, event_queue, cancel_token, request_id):
        calls.append(cancel_token)
        event_queue.put({"type": "budget", "timeoutSeconds": 300, "remainingSeconds": 240})
        if release.wait(5):
            event_queue.put({"type": "done", "status": "partial"})

    monkeypatch.setattr(web_server, "run_search", worker)
    headers = {"Host": f"127.0.0.1:{server.bind_port}", "Cookie": "offlineai_session=test-session-token"}
    path = "/api/search?q=test&mode=deep&reasoning=off&timeout_seconds=300&request_id=running"
    first = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=5)
    second = http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=5)
    try:
        first.request("GET", path, headers=headers)
        response = first.getresponse()
        assert response.status == 200
        line = response.readline()
        while line and not line.startswith(b"data: "):
            line = response.readline()
        first_remaining = json.loads(line[6:])["remainingSeconds"]
        calls[0].deadline -= 60
        deadline = calls[0].deadline
        response.close()
        first.close()
        assert not calls[0].is_cancelled
        second.request("GET", path + "&resume_only=1", headers=headers)
        resumed = second.getresponse()
        assert resumed.status == 200
        line = resumed.readline()
        while line and not line.startswith(b"data: "):
            line = resumed.readline()
        assert 0 < json.loads(line[6:])["remainingSeconds"] <= first_remaining - 59
        assert len(calls) == 1
        assert server.job_table.get("running").cancel_token.deadline == deadline
        release.set()
        assert b'"type": "done"' in resumed.read()
        assert len(calls) == 1
    finally:
        release.set()
        first.close()
        second.close()
