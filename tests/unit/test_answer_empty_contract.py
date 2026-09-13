"""本文0文字終了の明示（answer_empty）と stall error code 写像のテスト。

対象:
- search.iter_stream_events が ``("done", done_reason)`` を最後に 1 回だけ yield する
- search.iter_stream_chunks が done イベントを content として漏らさない（後方互換）
- web_services.run_search が本文 0 文字のとき reason 付き `answer_empty` を
  `done` の直前に 1 回だけ送る
- 生成フェーズの socket timeout が `internal_error` ではなく `stall` になる
- 接続不能と retrieval フェーズの timeout は `stall` へ写像されない

検証対象: _internal/search.py の空回答と回答生成契約
"""

import json
import os
import socket
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402
import web_services  # noqa: E402


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


def _make_response(lines):
    resp = BytesIO(b"".join(lines))
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _single_response_urlopen(resp):
    def _urlopen(request, timeout=None):
        return resp

    return _urlopen


def _raising_urlopen(exc):
    def _urlopen(request, timeout=None):
        raise exc

    return _urlopen


class _FakeRetrieval:
    matches = []
    evidence_status = "sufficient"
    confidence = 0.9
    user_prompt = "prompt"
    route = "keyword"
    index_state = "ready"


def _default_pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
    return _FakeRetrieval()


def _prepare_worker(monkeypatch, pipeline=_default_pipeline):
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "m")
    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)


def _drain(entry):
    events = []
    replay = entry.add_subscriber()
    while not replay.empty():
        events.append(replay.get_nowait())
    return events


def _run(monkeypatch, lines=None, *, urlopen=None, reasoning="medium", timeout=5.0):
    if urlopen is None:
        urlopen = _single_response_urlopen(_make_response(lines))
    monkeypatch.setattr(web_services.urllib.request, "urlopen", urlopen)
    token = web_services.CancellationToken(timeout=timeout, stall_timeout=timeout)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", reasoning, web_services._BroadcastQueue(entry), token, "rid")
    return _drain(entry)


# --- iter_stream_events / iter_stream_chunks の done 契約 --------------------


def test_iter_stream_events_yields_done_reason_length():
    lines = [_stream_line(thinking="長い思考"), _stream_line(done=True, done_reason="length")]
    events = list(search.iter_stream_events(_make_response(lines)))
    assert events == [("thinking", "長い思考"), ("done", "length")]


def test_iter_stream_events_defaults_missing_done_reason_to_stop():
    lines = [_stream_line(content="本文"), _stream_line(done=True)]
    events = list(search.iter_stream_events(_make_response(lines)))
    assert events[-1] == ("done", "stop")


def test_iter_stream_events_yields_done_only_once():
    lines = [
        _stream_line(content="本文"),
        _stream_line(done=True, done_reason="stop"),
        _stream_line(done=True, done_reason="length"),  # done後は読まれない
    ]
    events = list(search.iter_stream_events(_make_response(lines)))
    assert [kind for kind, _ in events].count("done") == 1


def test_iter_stream_chunks_does_not_leak_done_event():
    """done イベントを content として漏らさない（既存 CLI / テスト互換）。"""
    lines = [_stream_line(content="回答"), _stream_line(done=True, done_reason="length")]
    assert list(search.iter_stream_chunks(_make_response(lines))) == ["回答"]


# --- CLI 経路の空回答明示 ---------------------------------------------------


def test_answer_empty_message_for_length_suggests_concrete_remedy():
    message = search.answer_empty_message("length")
    assert "OLLAMA_NUM_CTX" in message
    assert "推論: なし" in message


def test_answer_empty_message_for_other_reasons_does_not_assert_cause():
    message = search.answer_empty_message("stop")
    assert "OLLAMA_NUM_CTX" not in message
    assert "再試行" in message


# --- run_search の answer_empty 送出 ----------------------------------------


def test_run_search_emits_answer_empty_with_length_reason_before_done(monkeypatch):
    _prepare_worker(monkeypatch)
    events = _run(
        monkeypatch,
        [_stream_line(thinking="思考のみ"), _stream_line(done=True, done_reason="length")],
    )
    types = [e["type"] for e in events]
    assert types[-2:] == ["answer_empty", "done"]
    assert [e for e in events if e["type"] == "answer_empty"][0]["reason"] == "length"


def test_run_search_emits_answer_empty_for_stop_reason_too(monkeypatch):
    """reason が length 以外でも、本文 0 文字なら無言終了させない。"""
    _prepare_worker(monkeypatch)
    events = _run(monkeypatch, [_stream_line(done=True, done_reason="stop")])
    empties = [e for e in events if e["type"] == "answer_empty"]
    assert len(empties) == 1
    assert empties[0]["reason"] == "stop"


def test_run_search_emits_answer_empty_only_once(monkeypatch):
    _prepare_worker(monkeypatch)
    events = _run(
        monkeypatch,
        [_stream_line(thinking="a"), _stream_line(thinking="b"), _stream_line(done=True)],
    )
    assert [e["type"] for e in events].count("answer_empty") == 1


def test_run_search_does_not_emit_answer_empty_when_content_exists(monkeypatch):
    _prepare_worker(monkeypatch)
    events = _run(
        monkeypatch,
        [_stream_line(content="回答"), _stream_line(done=True, done_reason="stop")],
    )
    assert not [e for e in events if e["type"] == "answer_empty"]


def test_run_search_does_not_emit_answer_empty_when_length_but_content_exists(monkeypatch):
    """打ち切られていても本文があるなら警告は出さない。"""
    _prepare_worker(monkeypatch)
    events = _run(
        monkeypatch,
        [_stream_line(content="途中まで"), _stream_line(done=True, done_reason="length")],
    )
    assert not [e for e in events if e["type"] == "answer_empty"]


def test_run_search_logs_done_reason_and_content_chars(monkeypatch):
    _prepare_worker(monkeypatch)
    logged = []
    monkeypatch.setattr(
        web_services, "log_structured", lambda request_id, **fields: logged.append(fields)
    )
    _run(monkeypatch, [_stream_line(done=True, done_reason="length")])
    complete = [f for f in logged if f.get("phase") == "complete"]
    assert complete and complete[0]["done_reason"] == "length"
    assert complete[0]["content_chars"] == 0


# --- stall error code 写像 --------------------------------------------------


def _error_event(events):
    errors = [e for e in events if e["type"] == "error"]
    assert errors, "error イベントが送出されていない"
    return errors[0]


def test_generation_socket_timeout_maps_to_stall(monkeypatch):
    _prepare_worker(monkeypatch)
    events = _run(monkeypatch, urlopen=_raising_urlopen(TimeoutError("timed out")))
    error = _error_event(events)
    assert error["code"] == "stall"


def test_generation_urlerror_with_timeout_reason_maps_to_stall(monkeypatch):
    _prepare_worker(monkeypatch)
    exc = urllib.error.URLError(socket.timeout("timed out"))
    events = _run(monkeypatch, urlopen=_raising_urlopen(exc))
    assert _error_event(events)["code"] == "stall"


def test_stall_message_includes_seconds_matching_check_path(monkeypatch):
    """run_search 経路と CancellationToken.check 経路で文言を揃える。"""
    _prepare_worker(monkeypatch)
    events = _run(monkeypatch, urlopen=_raising_urlopen(TimeoutError("timed out")))
    message = _error_event(events)["message"]
    expected_seconds = int(web_services.GENERATION_STALL_TIMEOUT)
    assert message == f"モデルからの応答が{expected_seconds}秒途絶えました"

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=expected_seconds)
    token._last_activity -= expected_seconds + 1
    try:
        token.check()
    except web_services.CancelledError as exc:
        assert str(exc) == message
        assert exc.code == "stall"
    else:  # pragma: no cover - 到達したら契約違反
        raise AssertionError("stall で CancelledError が送出されなかった")


def test_connection_refused_is_not_mapped_to_stall(monkeypatch):
    """Ollama 停止による接続不能まで stall と表示しない（原因判別を保つ）。"""
    _prepare_worker(monkeypatch)
    exc = urllib.error.URLError(ConnectionRefusedError("refused"))
    events = _run(monkeypatch, urlopen=_raising_urlopen(exc))
    assert _error_event(events)["code"] != "stall"


def test_retrieval_phase_timeout_is_not_mapped_to_stall(monkeypatch):
    """写像は生成フェーズの chat stream に限定する。"""

    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        raise TimeoutError("embed request timed out")

    _prepare_worker(monkeypatch, pipeline)
    events = _run(monkeypatch, [_stream_line(done=True)])
    assert _error_event(events)["code"] != "stall"


def test_cancellation_token_exposes_stall_timeout():
    token = web_services.CancellationToken(timeout=5.0, stall_timeout=42.0)
    assert token.stall_timeout == 42.0
    assert web_services.CancellationToken(timeout=5.0).stall_timeout is None
