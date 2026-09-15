"""検索単位timeout設定・全体残り時間表示のテスト（Phase 7）。

対象:
- config.validate_search_timeout_seconds の300〜600秒契約（test_config.py側でも純粋契約を検証）
- web_server._parse_search_timeout_param のクエリパラメータ解釈
- web_server._build_budget_event が返す SSE `budget` イベントの形
- web_server.ERROR_CODES への invalid_timeout 追加
- 実サーバー経由での /api/health のsearchTimeoutDefault/Min/Max公開
- 実サーバー経由での /api/search の timeout_seconds 受理・拒否とSSE budgetイベント
"""

import http.client
import json
import os
import socket
import threading
import time

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import config  # noqa: E402
import search  # noqa: E402
import web_server  # noqa: E402
import web_services  # noqa: E402

# 実サーバー経由の契約確認は /api/health と /api/search を叩くため、既定のまま
# だと `OLLAMA_HOST` が localhost:11434 を指し、実 Ollama が稼働している開発機で
# 実モデルへ要求を送ってしまう。接続失敗を許容する経路なので合否は変わらないが、
# 性能計測環境との分離が壊れるため、全接続先を到達不能な fixture へ固定する。
UNREACHABLE_OLLAMA_HOST = "http://127.0.0.1:1"


def test_gpt_oss_cold_timeout_contract_has_measured_headroom():
    """gpt-oss cold検索の実測（evidence最大82.391秒）を下回らない契約。"""
    assert search.KEYWORD_EXTRACT_TIMEOUT == 90
    assert search.RETRIEVAL_TIMEOUT == 120


def test_retrieval_does_not_consume_generation_stall_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(web_services.time, "monotonic", lambda: clock[0])
    token = web_services.CancellationToken(300, stall_timeout=60, defer_stall=True)
    clock[0] = 85.0
    token.check()
    assert token.stall_remaining() == float("inf")
    token.start_stall_monitoring()
    assert token.remaining() == 215
    clock[0] = 144.0
    token.check()
    clock[0] = 145.0
    with pytest.raises(web_services.CancelledError) as exc:
        token.check()
    assert exc.value.code == "stall"


def test_deferred_stall_preserves_overall_deadline_and_cancel(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(web_services.time, "monotonic", lambda: clock[0])
    token = web_services.CancellationToken(300, stall_timeout=60, defer_stall=True)
    clock[0] = 300.0
    with pytest.raises(web_services.CancelledError) as exc:
        token.check()
    assert exc.value.code == "timeout"
    token.cancel()
    with pytest.raises(web_services.CancelledError) as exc:
        token.check()
    assert exc.value.code == "cancelled"


# --- _parse_search_timeout_param -------------------------------------------


def test_parse_search_timeout_param_uses_default_when_omitted():
    assert web_server._parse_search_timeout_param([], 300) == 300


@pytest.mark.parametrize("raw, expected", [(["300"], 300), (["450"], 450), (["600"], 600)])
def test_parse_search_timeout_param_accepts_contract_values(raw, expected):
    assert web_server._parse_search_timeout_param(raw, 300) == expected


@pytest.mark.parametrize("raw", [["119"], ["601"], ["0"], ["abc"], ["300.5"]])
def test_parse_search_timeout_param_rejects_out_of_contract_values(raw):
    with pytest.raises(ValueError):
        web_server._parse_search_timeout_param(raw, 300)


def test_parse_search_timeout_param_rejects_duplicate_values():
    with pytest.raises(ValueError):
        web_server._parse_search_timeout_param(["300", "450"], 300)


# --- _build_budget_event -----------------------------------------------------


def test_build_budget_event_reports_confirmed_total_and_remaining():
    token = web_services.CancellationToken(300, stall_timeout=60)
    event = web_server._build_budget_event(token)

    assert event["type"] == "budget"
    assert event["timeoutSeconds"] == 300
    assert isinstance(event["remainingSeconds"], int)
    assert 0 <= event["remainingSeconds"] <= 300


def test_build_budget_event_remaining_decreases_on_resubscribe():
    token = web_services.CancellationToken(2, stall_timeout=60)
    first = web_server._build_budget_event(token)
    time.sleep(1.1)
    second = web_server._build_budget_event(token)

    assert second["timeoutSeconds"] == first["timeoutSeconds"]
    assert second["remainingSeconds"] < first["remainingSeconds"]


def test_build_budget_event_does_not_leak_between_tokens():
    token_a = web_services.CancellationToken(300)
    token_b = web_services.CancellationToken(600)

    assert web_server._build_budget_event(token_a)["timeoutSeconds"] == 300
    assert web_server._build_budget_event(token_b)["timeoutSeconds"] == 600


def test_invalid_timeout_error_code_is_http_400():
    assert web_services.ERROR_CODES["invalid_timeout"] == 400


# --- 実サーバー経由の契約確認 -------------------------------------------------


@pytest.fixture(autouse=True)
def pinned_ollama_host(monkeypatch):
    """search と web_services の両方の接続先を fixture へ固定する。

    片方だけ差し替えると実 Ollama へ漏れる（2026-09-09 の誤送信事例）ため、
    両モジュールを必ず同時に固定する。
    """
    monkeypatch.setattr(search, "OLLAMA_HOST", UNREACHABLE_OLLAMA_HOST)
    monkeypatch.setattr(web_services, "OLLAMA_HOST", UNREACHABLE_OLLAMA_HOST)


@pytest.fixture
def running_server():
    server = web_server.LimitedThreadingServer(
        ("127.0.0.1", 0),
        web_server.OfflineAIHandler,
        session_token="test-session-token",
        search_timeout=300,
    )
    server.bind_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown_jobs_once()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _connection(server):
    return http.client.HTTPConnection("127.0.0.1", server.bind_port, timeout=10)


def test_health_endpoint_publishes_search_timeout_contract(running_server):
    conn = _connection(running_server)
    try:
        conn.request(
            "GET",
            "/api/health",
            headers={"Host": f"127.0.0.1:{running_server.bind_port}"},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()

    assert resp.status == 200
    assert body["searchTimeoutDefault"] == running_server.search_timeout
    assert body["searchTimeoutMin"] == config.SEARCH_TIMEOUT_MIN
    assert body["searchTimeoutMax"] == config.SEARCH_TIMEOUT_MAX


def test_search_endpoint_rejects_out_of_contract_timeout_without_starting_worker(
    running_server,
):
    conn = _connection(running_server)
    try:
        conn.request(
            "GET",
            "/api/search?q=test&timeout_seconds=601",
            headers={
                "Host": f"127.0.0.1:{running_server.bind_port}",
                "Cookie": "offlineai_session=test-session-token",
            },
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()

    assert resp.status == 400
    assert body["error"]["code"] == "invalid_timeout"
    assert running_server.job_table.running_count == 0


def _read_first_sse_data_event(server, path):
    """`/api/search` の最初の `data: ...` イベントだけを生ソケットで読む。

    SSEレスポンスは Content-Length も chunked encoding も付与しないため
    （O2-6として別途追跡している既知の接続終了問題）、http.client の
    ``read()`` は接続がcloseされるまでブロックする。ここでは最初の1
    イベントぶんだけを読み切ったら接続を切る。
    """
    sock = socket.create_connection(("127.0.0.1", server.bind_port), timeout=10)
    sock.settimeout(10)
    try:
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.bind_port}\r\n"
            "Cookie: offlineai_session=test-session-token\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))

        buf = b""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            marker = buf.find(b"data: ")
            if marker != -1 and b"\n\n" in buf[marker:]:
                break
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        else:
            pytest.fail("timed out waiting for first SSE data event")

        head, _, body = buf.partition(b"\r\n\r\n")
        marker = body.find(b"data: ")
        assert marker != -1, f"no SSE data line received: {buf!r}"
        after = body[marker + len(b"data: ") :]
        event_line = after.split(b"\n\n", 1)[0]
        status_line = head.split(b"\r\n", 1)[0].decode("ascii")
        return status_line, json.loads(event_line.decode("utf-8"))
    finally:
        sock.close()


def _wait_until_cancelled(query, reasoning, event_queue, cancel_token, request_id, **_kwargs):
    """budget イベントだけを読む試験用の worker。

    実 pipeline を走らせると、到達不能な Ollama への接続待ちの後に製品資料の
    chunk 読み込みへ進む daemon thread が試験終了後も残り、後続の試験の
    ``build_source_chunks`` 呼び出し回数を汚す。fixture 終了時の cancel で抜ける。
    """
    deadline = time.monotonic() + 10
    while not cancel_token.is_cancelled and time.monotonic() < deadline:
        time.sleep(0.01)


def test_search_endpoint_sse_budget_reflects_requested_timeout(running_server, monkeypatch):
    monkeypatch.setattr(web_server, "run_search", _wait_until_cancelled)
    status_line, first_event = _read_first_sse_data_event(
        running_server, "/api/search?q=test&timeout_seconds=480"
    )

    assert "200" in status_line
    assert first_event["type"] == "budget"
    assert first_event["timeoutSeconds"] == 480
    assert 0 <= first_event["remainingSeconds"] <= 480


def test_search_endpoint_sse_budget_uses_server_default_when_timeout_omitted(
    running_server, monkeypatch
):
    monkeypatch.setattr(web_server, "run_search", _wait_until_cancelled)
    _status_line, first_event = _read_first_sse_data_event(running_server, "/api/search?q=test")

    assert first_event["type"] == "budget"
    assert first_event["timeoutSeconds"] == running_server.search_timeout


def test_search_sse_closes_http_response_after_terminal_event(running_server, monkeypatch):
    """doneをflushした検索SSEは、ReadableStreamのEOFまで到達させる。"""

    def emit_done(query, reasoning, event_queue, cancel_token, request_id):
        event_queue.put({"type": "done"})

    monkeypatch.setattr(web_server, "run_search", emit_done)
    conn = _connection(running_server)
    try:
        conn.request(
            "GET",
            "/api/search?q=terminal-test&timeout_seconds=300",
            headers={
                "Host": f"127.0.0.1:{running_server.bind_port}",
                "Cookie": "offlineai_session=test-session-token",
            },
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
    finally:
        conn.close()

    assert resp.status == 200
    assert '"type": "done"' in body
    assert resp.will_close is True
