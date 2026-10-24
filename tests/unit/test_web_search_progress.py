"""Web検索workerのEmbedding進捗配信・キャンセル終端のテスト。

対象:
- run_search が retrieval pipeline の status を SSE `status` event として配信する
- Webキャンセル / timeout で job が cancelled または failed へ終端し、worker が続行しない
- 再接続 buffer が進捗を replay し、資料本文・path を含まない
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402
import web_services  # noqa: E402


def _progress_status(processed: int, total: int) -> str:
    return search._format_embed_progress(
        {
            "phase": "progress",
            "processed": processed,
            "total": total,
            "generated": processed,
            "reused": 0,
            "failed": 0,
            "checkpointed": processed,
        }
    )


def _prepare_worker(monkeypatch, pipeline):
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "m")
    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)


def test_embedding_progress_is_published_as_sse_status_events(monkeypatch):
    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        emit_status("Embeddingインデックスを更新しています...")
        emit_status(_progress_status(5, 10))
        emit_status(_progress_status(10, 10))
        # 進捗配信後にキャンセルされたケースを再現する。
        raise web_services.CancelledError("cancelled")

    _prepare_worker(monkeypatch, pipeline)
    token = web_services.CancellationToken(timeout=5.0)
    entry = web_services.JobEntry(token)
    subscriber = entry.add_subscriber()

    web_services.run_search("q", "", web_services._BroadcastQueue(entry), token, "rid")

    events = []
    while not subscriber.empty():
        events.append(subscriber.get_nowait())
    statuses = [event["text"] for event in events if event["type"] == "status"]
    assert any("5 / 10 (50.0%)" in text for text in statuses)
    assert any("10 / 10 (100.0%)" in text for text in statuses)
    assert events[-1]["type"] == "error"


def test_cancelled_job_terminates_without_continuing_worker(monkeypatch):
    calls = {"n": 0}

    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        calls["n"] += 1
        cancel_check()
        pytest.fail("キャンセル後にworkerが処理を続けてはならない")

    _prepare_worker(monkeypatch, pipeline)
    token = web_services.CancellationToken(timeout=5.0)
    token.cancel()
    entry = web_services.JobEntry(token)

    web_services.run_search("q", "", web_services._BroadcastQueue(entry), token, "rid")

    assert calls["n"] == 0  # pipeline 到達前の cancel_token.check() で終端する
    assert entry.state in {"cancelled", "failed"}
    assert entry.completed_at is not None


def test_timeout_terminates_job_after_progress(monkeypatch):
    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        emit_status(_progress_status(3, 10))
        raise web_services.CancelledError("タイムアウト")

    _prepare_worker(monkeypatch, pipeline)
    token = web_services.CancellationToken(timeout=5.0)
    entry = web_services.JobEntry(token)

    web_services.run_search("q", "", web_services._BroadcastQueue(entry), token, "rid")

    assert entry.state in {"cancelled", "failed"}
    assert not entry.subscriber_queues


def test_reconnect_buffer_replays_progress_without_source_content(monkeypatch):
    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        emit_status(_progress_status(7, 20))
        raise web_services.CancelledError("cancelled")

    _prepare_worker(monkeypatch, pipeline)
    token = web_services.CancellationToken(timeout=5.0)
    entry = web_services.JobEntry(token)

    web_services.run_search("社外秘の資料名", "", web_services._BroadcastQueue(entry), token, "rid")

    replay = entry.add_subscriber()
    replayed = []
    while not replay.empty():
        replayed.append(replay.get_nowait())
    statuses = [event["text"] for event in replayed if event["type"] == "status"]
    assert any("7 / 20 (35.0%)" in text for text in statuses)
    for text in statuses:
        assert "社外秘の資料名" not in text
        assert ".md" not in text
