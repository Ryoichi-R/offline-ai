"""thinking可視化とstallベースtimeoutの回帰テスト。

対象:
- search.iter_stream_events / iter_stream_chunks の thinking/content 分離
- search.build_chat_payload の think 常時送信契約（F-7）
- search.stream_ollama_chat / web_services.run_search の think フォールバック（F-8）
- web_services.CancellationToken の touch() / stall_remaining()
- web_services.run_search の thinking イベント送出とログの文字数のみ記録
"""

import json
from io import BytesIO
from unittest.mock import MagicMock

import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402
import web_services  # noqa: E402


# --- iter_stream_events / iter_stream_chunks -------------------------------


def _stream_line(*, thinking=None, content=None, done=False, done_reason=None):
    message = {}
    if thinking is not None:
        message["thinking"] = thinking
    if content is not None:
        message["content"] = content
    payload = {"message": message, "done": done}
    if done_reason is not None:
        payload["done_reason"] = done_reason
    return json.dumps(payload).encode("utf-8") + b"\n"


def test_iter_stream_events_yields_thinking_then_content_in_order():
    lines = [
        _stream_line(thinking="考え中"),
        _stream_line(thinking="続き", content="回"),
        _stream_line(content="答"),
        _stream_line(done=True),
    ]
    events = list(search.iter_stream_events(BytesIO(b"".join(lines))))
    assert events == [
        ("thinking", "考え中"),
        ("thinking", "続き"),
        ("content", "回"),
        ("content", "答"),
        ("done", "stop"),
    ]


def test_iter_stream_events_handles_thinking_only_line():
    lines = [_stream_line(thinking="独り言"), _stream_line(done=True)]
    events = list(search.iter_stream_events(BytesIO(b"".join(lines))))
    assert events == [("thinking", "独り言"), ("done", "stop")]


def test_iter_stream_events_handles_content_only_line():
    lines = [_stream_line(content="本文"), _stream_line(done=True)]
    events = list(search.iter_stream_events(BytesIO(b"".join(lines))))
    assert events == [("content", "本文"), ("done", "stop")]


def test_iter_stream_events_reassembles_split_json_line():
    full = json.dumps({"message": {"content": "分割された行"}, "done": False})
    half = len(full) // 2
    lines = [
        full[:half].encode("utf-8"),
        full[half:].encode("utf-8") + b"\n",
        _stream_line(done=True),
    ]
    events = list(search.iter_stream_events(BytesIO(b"".join(lines))))
    assert events == [("content", "分割された行"), ("done", "stop")]


def test_iter_stream_events_stops_at_done():
    lines = [
        _stream_line(content="前"),
        _stream_line(done=True),
        _stream_line(content="後"),  # done後は読まれない
    ]
    events = list(search.iter_stream_events(BytesIO(b"".join(lines))))
    assert events == [("content", "前"), ("done", "stop")]


def test_iter_stream_chunks_returns_content_only_for_backward_compat():
    lines = [
        _stream_line(thinking="考え中", content=None),
        _stream_line(content="回答"),
        _stream_line(done=True),
    ]
    chunks = list(search.iter_stream_chunks(BytesIO(b"".join(lines))))
    assert chunks == ["回答"]


# --- build_chat_payload の think 契約（F-7） ---------------------------------


class TestThinkContract:
    @pytest.mark.parametrize(
        "reasoning, expected",
        [
            ("high", "high"),
            ("medium", "medium"),
            ("low", "low"),
            (None, False),
            ("", False),
            ("off", False),
        ],
    )
    def test_think_value_follows_reasoning(self, reasoning, expected):
        body = search.build_chat_payload("m", "sys", "user", reasoning=reasoning)
        assert body["think"] == expected

    def test_resolve_reasoning_off_is_accepted(self):
        assert search._resolve_reasoning("off") == "off"

    def test_resolve_reasoning_default_is_low(self):
        assert search._resolve_reasoning(None) in {"medium", "high", "low", "off"}
        # 環境変数未設定時の既定は low（2026-09-08 の実測で medium から変更）。
        import os as _os

        env_backup = _os.environ.pop("OFFLINE_AI_REASONING", None)
        try:
            assert search._resolve_reasoning(None) == "low"
        finally:
            if env_backup is not None:
                _os.environ["OFFLINE_AI_REASONING"] = env_backup


def _make_stream_response(chunks):
    lines = []
    for chunk in chunks:
        lines.append(_stream_line(content=chunk))
    lines.append(_stream_line(done=True))
    return BytesIO(b"".join(lines))


class TestCliThinkFallback:
    """CLI stream_ollama_chat: think キー存在ベースの fallback（F-8）。"""

    def test_think_false_still_falls_back_on_http_error(self, monkeypatch):
        from urllib.error import HTTPError

        http_error = HTTPError(
            url="http://localhost:11434/api/chat", code=400, msg="Bad Request", hdrs={}, fp=None
        )
        success_resp = _make_stream_response(["OK"])
        success_resp.__enter__ = lambda s: s
        success_resp.__exit__ = MagicMock(return_value=False)

        calls = {"n": 0}

        def fake_urlopen(request, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise http_error
            return success_resp

        monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)

        # reasoning=None でも think=False が明示送信されるため fallback が発動する。
        result = search.stream_ollama_chat("m", "sys", "q", reasoning=None)

        assert result == "OK"
        assert calls["n"] == 2

    def test_fallback_retries_only_once(self, monkeypatch):
        from urllib.error import HTTPError

        http_error = HTTPError(
            url="http://localhost:11434/api/chat", code=500, msg="Server Error", hdrs={}, fp=None
        )
        monkeypatch.setattr(
            search.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(http_error)
        )

        with pytest.raises(HTTPError):
            search.stream_ollama_chat("m", "sys", "q", reasoning="high")


# --- web_services.CancellationToken の stall 拡張 ---------------------------


class TestCancellationTokenStall:
    def test_touch_extends_stall_but_not_overall_deadline(self):
        token = web_services.CancellationToken(10.0, stall_timeout=5.0)
        overall_before = token.deadline
        token.touch()
        assert token.deadline == overall_before  # 全体deadlineは touch() で延びない
        assert token.stall_remaining() > 4.9

    def test_check_raises_timeout_code_on_overall_deadline(self):
        token = web_services.CancellationToken(0.0, stall_timeout=60.0)
        with pytest.raises(web_services.CancelledError) as exc_info:
            token.check()
        assert exc_info.value.code == "timeout"

    def test_check_raises_stall_code_when_stall_exceeded(self):
        token = web_services.CancellationToken(60.0, stall_timeout=0.0)
        with pytest.raises(web_services.CancelledError) as exc_info:
            token.check()
        assert exc_info.value.code == "stall"

    def test_check_raises_cancelled_code_when_explicitly_cancelled(self):
        token = web_services.CancellationToken(60.0)
        token.cancel()
        with pytest.raises(web_services.CancelledError) as exc_info:
            token.check()
        assert exc_info.value.code == "cancelled"

    def test_stall_remaining_is_infinite_without_stall_timeout(self):
        token = web_services.CancellationToken(60.0)
        assert token.stall_remaining() == float("inf")

    def test_repeated_touch_keeps_stall_alive(self):
        """thinkingデルタのみが流れ続ける間は stall しない。"""
        token = web_services.CancellationToken(60.0, stall_timeout=0.05)
        for _ in range(5):
            token.touch()
            assert token.stall_remaining() > 0
        # touchが止まればstallする
        import time

        time.sleep(0.1)
        assert token.stall_remaining() <= 0

    def test_timeout_seconds_is_retained_from_constructor(self):
        """検索単位で確定した全体秒数をtokenが保持する（Phase 7 budgetイベントの根拠）。"""
        token = web_services.CancellationToken(300)
        assert token.timeout_seconds == 300

    def test_timeout_seconds_does_not_leak_between_tokens(self):
        """同じサーバー上の別検索へ値が漏れない（token毎に独立）。"""
        token_a = web_services.CancellationToken(120)
        token_b = web_services.CancellationToken(600)
        assert token_a.timeout_seconds == 120
        assert token_b.timeout_seconds == 600


# --- web_services.run_search の thinking イベント送出 -----------------------


def _prepare_worker(monkeypatch, pipeline):
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "m")
    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)


class _FakeRetrieval:
    matches = []
    evidence_status = "sufficient"
    confidence = 0.9
    user_prompt = "prompt"
    route = "keyword"
    index_state = "ready"


def _default_pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
    return _FakeRetrieval()


def test_worker_arms_stall_only_after_retrieval(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(web_services.time, "monotonic", lambda: clock[0])
    token = web_services.CancellationToken(300, stall_timeout=60, defer_stall=True)

    def pipeline(query, **kwargs):
        clock[0] = 85.0
        kwargs["cancel_check"]()
        assert token.stall_remaining() == float("inf")
        return _FakeRetrieval()

    _prepare_worker(monkeypatch, pipeline)
    resp = _make_stream_response_with_thinking(["思考"], ["回答"])

    def open_response(*args, **kwargs):
        assert token.stall_remaining() == 60
        assert token.remaining() == 215
        return resp

    monkeypatch.setattr(web_services.urllib.request, "urlopen", open_response)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "low", web_services._BroadcastQueue(entry), token, "rid")
    assert token.stall_remaining() == 60


def test_run_search_emits_thinking_events_distinct_from_chunk(monkeypatch):
    _prepare_worker(monkeypatch, _default_pipeline)
    resp = _make_stream_response_with_thinking(["思考A", "思考B"], ["回答"])
    monkeypatch.setattr(web_services.urllib.request, "urlopen", _single_response_urlopen(resp))

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "medium", web_services._BroadcastQueue(entry), token, "rid")

    events = []
    replay = entry.add_subscriber()
    while not replay.empty():
        events.append(replay.get_nowait())

    thinking_events = [e for e in events if e["type"] == "thinking"]
    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert [e["text"] for e in thinking_events] == ["思考A", "思考B"]
    assert [e["text"] for e in chunk_events] == ["回答"]
    assert events[-1]["type"] == "done"


def test_run_search_hides_thinking_events_when_reasoning_is_off(monkeypatch):
    _prepare_worker(monkeypatch, _default_pipeline)
    resp = _make_stream_response_with_thinking(["内部思考"], ["回答"])
    monkeypatch.setattr(web_services.urllib.request, "urlopen", _single_response_urlopen(resp))

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "off", web_services._BroadcastQueue(entry), token, "rid")

    events = []
    replay = entry.add_subscriber()
    while not replay.empty():
        events.append(replay.get_nowait())

    assert [e for e in events if e["type"] == "thinking"] == []
    assert [e["text"] for e in events if e["type"] == "chunk"] == ["回答"]
    assert events[-1]["type"] == "done"


def test_run_search_logs_thinking_char_count_only(monkeypatch):
    _prepare_worker(monkeypatch, _default_pipeline)
    resp = _make_stream_response_with_thinking(["秘密の思考内容"], ["回答"])
    monkeypatch.setattr(web_services.urllib.request, "urlopen", _single_response_urlopen(resp))
    logged = []
    monkeypatch.setattr(
        web_services, "log_structured", lambda request_id, **fields: logged.append(fields)
    )

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "medium", web_services._BroadcastQueue(entry), token, "rid")

    complete_logs = [f for f in logged if f.get("phase") == "complete"]
    assert complete_logs
    assert complete_logs[0]["thinking_chars"] == len("秘密の思考内容")
    rendered = json.dumps(complete_logs[0], ensure_ascii=False)
    assert "秘密の思考内容" not in rendered


def _make_stream_response_with_thinking(thinking_chunks, content_chunks):
    lines = []
    for chunk in thinking_chunks:
        lines.append(_stream_line(thinking=chunk))
    for chunk in content_chunks:
        lines.append(_stream_line(content=chunk))
    lines.append(_stream_line(done=True))
    resp = BytesIO(b"".join(lines))
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _single_response_urlopen(resp):
    def _urlopen(request, timeout=None):
        return resp

    return _urlopen


def test_run_search_think_fallback_removes_think_and_retries_once(monkeypatch):
    from urllib.error import HTTPError

    _prepare_worker(monkeypatch, _default_pipeline)
    http_error = HTTPError(
        url="http://localhost:11434/api/chat", code=400, msg="Bad Request", hdrs={}, fp=None
    )
    success_resp = _make_stream_response_with_thinking([], ["OK"])

    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        body = json.loads(request.data.decode("utf-8"))
        if calls["n"] == 1:
            assert "think" in body
            raise http_error
        assert "think" not in body
        return success_resp

    monkeypatch.setattr(web_services.urllib.request, "urlopen", fake_urlopen)

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "high", web_services._BroadcastQueue(entry), token, "rid")

    assert calls["n"] == 2
    assert entry.state == "completed"


def test_run_search_think_fallback_respects_parent_remaining_time(monkeypatch):
    """親cancel_tokenの残時間が尽きていれば再送しない。"""
    from urllib.error import HTTPError

    clock = [0.0]
    monkeypatch.setattr(web_services.time, "monotonic", lambda: clock[0])
    _prepare_worker(monkeypatch, _default_pipeline)
    http_error = HTTPError(
        url="http://localhost:11434/api/chat", code=400, msg="Bad Request", hdrs={}, fp=None
    )
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        # 1回目のPOST中にdeadlineを使い切らせ、fallback再送が発生しないことを確認する。
        clock[0] += 0.15
        raise http_error

    monkeypatch.setattr(web_services.urllib.request, "urlopen", fake_urlopen)

    token = web_services.CancellationToken(timeout=0.1, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "high", web_services._BroadcastQueue(entry), token, "rid")

    assert calls["n"] == 1  # fallback再送が行われていない
    assert entry.state in {"cancelled", "failed"}
