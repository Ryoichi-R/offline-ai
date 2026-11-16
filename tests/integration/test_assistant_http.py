"""Authenticated assistant HTTP, SSE, cancellation and replay with fake model replies."""

import hashlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import assistant
import pytest
import web_server
from assistant_sessions import failure_result
from assistant_tools import SourceTools, verify_source
from source_view import EvidenceRegistry


@pytest.fixture
def service(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "notice.md"
    source.write_text("# 試験通知\n期限は30日です。\n", encoding="utf-8")
    registry = EvidenceRegistry(root, max_entries=1)
    calls = []
    release = threading.Event()
    entered = threading.Event()
    monkeypatch.setattr(
        web_server.OfflineAIHandler,
        "_get_auth_session_id",
        lambda self: self.headers.get("X-Test-Session", ""),
    )

    def model(
        root,
        generation,
        turn,
        question,
        model,
        reasoning,
        timeout,
        turns,
        prior,
        key,
        cancelled,
        emit,
    ):
        calls.append((turn, question, turns, prior))
        emit("assistant_thinking", {"text": "思考の一時表示"})
        emit("assistant_progress", {"text": "途中経過"})
        emit(
            "assistant_tool",
            {"tool": "search_sources", "query": "試験通知", "results": 1},
        )
        if question == "待機":
            entered.set()
            while not release.wait(0.01):
                if cancelled():
                    return failure_result(turn, question, "cancelled")
        tools = SourceTools(
            root,
            generation,
            turn,
            prior,
            retrieve=None,
            list_paths=lambda: ["notice.md"],
            key=key,
        )
        evidence = tools.execute(
            "read_source",
            {"path": "notice.md", "start_line": 2, "end_line": 2},
            lambda: None,
            10,
        )["untrusted_source_data"]
        outcome = failure_result(turn, question, "completed")
        outcome.update(
            answer=f"期限は30日です。{evidence['citation']}",
            sources=[verify_source(root, tools.sources[evidence["id"]])],
            trace=tools.trace,
        )
        return outcome

    monkeypatch.setattr(assistant, "live_turn", model)
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="unused",
        evidence_registry=registry,
        source_root=root,
        chat_model="gpt-oss:120b",
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, source, calls, entered
    finally:
        release.set()
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def connect(server, method, path, body=None, *, owner="owner"):
    port = server.bind_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    headers = {"Host": f"127.0.0.1:{port}", "X-Test-Session": owner}
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body, ensure_ascii=False).encode()
    conn.request(method, path, body, headers)
    return conn, conn.getresponse()


def request(server, method, path, body=None, **kwargs):
    conn, response = connect(server, method, path, body, **kwargs)
    try:
        content = response.read().decode()
        value = (
            [json.loads(line[6:]) for line in content.splitlines() if line.startswith("data: ")]
            if response.getheader("Content-Type").startswith("text/event-stream")
            else json.loads(content)
        )
        return response.status, value
    finally:
        conn.close()


def start(server):
    status, data = request(server, "POST", "/api/conversations", {})
    assert status == 201
    return data["conversationId"]


def payload(id_, *, turn=1, request_id="req-1", **extra):
    return {
        "mode": "assistant",
        "conversationId": id_,
        "requestId": request_id,
        "turn": turn,
        **extra,
    }


def test_assistant_sse_history_save_replay_and_regenerated_evidence(service):
    server, file, calls, _ = service
    id_ = start(server)
    path = f"/api/conversations/{id_}"
    sent = payload(id_, text="書類の期限は")
    status, events = request(server, "POST", "/api/search", sent)
    assert status == 200
    assert [event["type"] for event in events] == [
        "budget",
        "assistant_thinking",
        "assistant_progress",
        "assistant_tool",
        "assistant_result",
        "done",
    ]
    result = events[-2]["result"]
    assert "途中経過" not in result["answer"]
    assert request(server, "POST", "/api/search", sent)[1][-2]["result"] == result
    assert len(calls) == 1
    assert (
        request(
            server,
            "POST",
            "/api/search",
            payload(id_, request_id="req-other", text="重複"),
        )[0]
        == 409
    )
    status, resumed = request(server, "POST", "/api/search", payload(id_, resume_only=True))
    assert status == 200 and resumed[-2]["result"] == result
    entry = server.job_table.get("req-1")
    entry.completed_at = time.monotonic() - 65
    server.job_table._cleanup_expired()
    assert request(server, "POST", "/api/search", payload(id_, resume_only=True))[0] == 404
    record = request(server, "GET", path)[1]
    assert record["turns"] == [result] and record["active"] is None
    assert "思考の一時表示" not in json.dumps(record, ensure_ascii=False)
    assert request(server, "GET", path, owner="other")[0] == 404
    assert request(server, "GET", path + "/sources/S1", owner="other")[0] == 404
    assert request(server, "GET", path + "/turns/1/markdown", owner="other")[0] == 404
    markdown = request(server, "GET", path + "/turns/1/markdown")[1]["markdown"]
    assert result["answer"] in markdown and "思考の一時表示" not in markdown
    first = request(server, "GET", path + "/sources/S1")[1]["evidence"][0]["id"]
    second = request(server, "GET", path + "/sources/S1")[1]["evidence"][0]["id"]
    assert first != second
    loaded = request(server, "POST", "/api/conversations/import", {"markdown": markdown})[1][
        "result"
    ]
    assert loaded["sources"][0]["verification"] == "matched"
    assert loaded["answer"] == result["answer"]
    assert (
        request(
            server,
            "POST",
            "/api/search",
            payload(id_, turn=2, request_id="req-2", text="期限は"),
        )[0]
        == 200
    )
    assert calls[-1][2] == [result] and "S1" in calls[-1][3]
    file.write_text("# 改版\n期限は60日です。\n", encoding="utf-8")
    assert request(server, "GET", path + "/sources/S1")[1]["verification"] == "changed"
    assert (
        request(server, "POST", "/api/conversations/import", {"markdown": markdown})[1]["result"][
            "sources"
        ][0]["verification"]
        == "changed"
    )


def test_disconnected_job_resumes_cancels_and_releases_once(service):
    server, _, calls, entered = service
    id_ = start(server)
    conn, response = connect(server, "POST", "/api/search", payload(id_, text="待機"))
    try:
        assert response.status == 200 and entered.wait(3)
    finally:
        response.close()
        conn.close()
    assert request(server, "GET", f"/api/conversations/{id_}")[1]["active"]["requestId"] == "req-1"
    assert (
        request(server, "POST", "/api/search/cancel", {"request_id": "req-1"}, owner="other")[0]
        == 403
    )
    assert request(server, "POST", "/api/search/cancel", {"request_id": "req-1"})[0] == 200
    status, replay = request(server, "POST", "/api/search", payload(id_, resume_only=True))
    assert status == 200 and replay[-2]["result"]["status"] == "cancelled"
    assert len(calls) == 1
    assert server.job_table.running_count == 0
    server.job_table.finish("req-1")
    assert server.job_table.running_count == 0


def test_invalid_conversation_payloads_and_index_invalidation(service, monkeypatch):
    server, _, _, _ = service
    id_ = start(server)
    assert request(server, "POST", "/api/search", payload(id_, text="x" * 1001))[0] == 400
    assert (
        request(server, "POST", "/api/search", payload(id_, text="x", timeout_seconds=299))[0]
        == 400
    )
    assert request(server, "POST", "/api/search", payload(id_, text="x", reason="other"))[0] == 400
    assert request(server, "POST", "/api/conversations/import", {"markdown": "bad"})[0] == 400
    monkeypatch.setattr(web_server, "_index_marker", lambda: ["new-index-generation"])
    assert request(server, "GET", f"/api/conversations/{id_}")[0] == 404


def test_disposable_workers_use_local_fake_ollama_and_no_search_chat(tmp_path, monkeypatch):
    source = tmp_path / "notice.md"
    source.write_text("# 倉庫利用規程\n## 貸出\n計測器を貸し出します。\n", encoding="utf-8")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    requests = []

    class FakeOllama(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if len(requests) == 1:
                message = {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_sources",
                                "arguments": {"query": "倉庫利用規程 計測器"},
                            }
                        }
                    ]
                }
            elif len(requests) == 2:
                message = {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_source",
                                "arguments": {
                                    "path": source.name,
                                    "start_line": 3,
                                    "end_line": 3,
                                },
                            }
                        }
                    ]
                }
            else:
                citation = json.loads(body["messages"][-1]["content"])["citation"]
                message = {"content": "計測器を貸し出します。" + citation}
            self.wfile.write(json.dumps({"done": False, "message": message}).encode() + b"\n")
            self.wfile.write(b'{"done":true,"done_reason":"stop","message":{}}\n')

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    host = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("OLLAMA_HOST", host)
    monkeypatch.setenv("OFFLINE_AI_RERANK_MODE", "off")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = assistant.live_turn(
            tmp_path,
            assistant.source_generation(tmp_path),
            1,
            "貸出は何ですか",
            "gpt-oss:120b",
            "off",
            300,
            [],
            {},
            b"key",
            lambda: False,
            lambda *_: None,
            host=host,
        )
        assert [path for path, _ in requests] == ["/api/chat"] * 3
        assert result["status"] == "completed"
        assert result["sources"][0]["state"] == "search_excerpt"
        read = [item for item in result["sources"] if item["state"] == "read"]
        assert len(read) == 1 and read[0]["sha256"] == sha
        assert result["answer"].endswith(read[0]["citation"])
        assert result["trace"][0]["embedding_calls"] == 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
