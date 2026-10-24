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


def test_evidence_event_carries_expansion_group_fields(tmp_path):
    """親子展開item（group_id/expanded_from/group_order/group_partial）が
    Web公開用イベントへ引き継がれることを確認する。"""
    source = tmp_path / "guide.md"
    source.write_text("# 親\n\n## 子\n配下本文\n", encoding="utf-8")
    registry = web_services.create_evidence_registry(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    retrieval = search.RetrievalResult(
        query="配下",
        matches=[
            {
                "path": "guide.md",
                "chunk_id": "abcd1234#r01",
                "start_line": 3,
                "end_line": 4,
                "snippet": "配下本文",
                "source_sha256": digest,
                "source": "expanded",
                "group_id": "abcd1234",
                "expanded_from": "guide.md#0001",
                "group_order": 1,
                "group_partial": False,
            }
        ],
    )

    event = web_services.build_evidence_event(
        retrieval, evidence_registry=registry, session_id="session-a"
    )

    item = event["items"][0]
    assert item["groupId"] == "abcd1234"
    assert item["expandedFrom"] == "guide.md#0001"
    assert item["groupOrder"] == 1
    assert item["groupPartial"] is False


def test_evidence_event_omits_group_fields_for_direct_hits():
    retrieval = search.RetrievalResult(
        query="通常",
        matches=[
            {
                "path": "guide.md",
                "chunk_id": "guide.md#0001",
                "start_line": 1,
                "end_line": 1,
                "snippet": "本文",
                "source": "keyword",
            }
        ],
    )

    event = web_services.build_evidence_event(retrieval)

    item = event["items"][0]
    assert "groupId" not in item
    assert "expandedFrom" not in item


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
