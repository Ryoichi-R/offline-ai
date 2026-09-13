import hashlib

import search
import web_services


def _events(entry):
    queue = entry.add_subscriber()
    result = []
    while not queue.empty():
        result.append(queue.get_nowait())
    return result


def test_evidence_event_registers_opaque_id_and_hash(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# 見出し\n本文", encoding="utf-8")
    registry = web_services.create_evidence_registry(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    retrieval = search.RetrievalResult(
        query="本文",
        matches=[
            {
                "path": "guide.md",
                "source_title": "ガイド",
                "start_line": 2,
                "end_line": 2,
                "snippet": "本文",
                "source_sha256": digest,
            }
        ],
    )

    event = web_services.build_evidence_event(
        retrieval, evidence_registry=registry, session_id="session-a"
    )

    item = event["items"][0]
    assert item["sourceSha256"] == digest
    assert item["viewable"] is True
    assert item["evidenceId"]
    assert "/" not in item["evidenceId"]
    assert registry.view(item["evidenceId"], session_id="session-a")["relativePath"] == "guide.md"


def test_search_worker_publishes_meta_evidence_done_without_answer_events(monkeypatch):
    calls = []

    def pipeline(
        query, *, model, reasoning=None, emit_status=None, cancel_check=None, mode="answer"
    ):
        calls.append((model, reasoning, mode))
        return search.RetrievalResult(query=query, matches=[])

    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)
    monkeypatch.setattr(
        web_services,
        "detect_model",
        lambda: (_ for _ in ()).throw(AssertionError("chat model must not be detected")),
    )
    token = web_services.CancellationToken(timeout=5.0)
    entry = web_services.JobEntry(token)

    web_services.run_search(
        "見つける",
        "high",
        web_services._BroadcastQueue(entry),
        token,
        "rid",
        mode="search",
        started_at="2026-09-12T08:00:00.000Z",
    )

    events = _events(entry)
    assert calls == [("", "off", "search")]
    assert [event["type"] for event in events] == ["result_meta", "evidence", "done"]
    assert events[0]["chatModel"] is None
    assert events[0]["embeddingModel"] is None
    assert events[0]["startedAt"] == "2026-09-12T08:00:00.000Z"
    assert events[-1]["completedAt"]
