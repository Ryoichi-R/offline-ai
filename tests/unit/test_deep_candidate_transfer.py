"""Candidate transfer contract tests; no live model or user corpus."""

import json
from pathlib import Path
import sys
import time

import pytest

import deep_candidate_selection as selection
import deep_research
import deep_transport
import search as cli_search
import web_services
from deep_fakes import patch_retrieval


def _candidate(path: str, index: int, *, source_sha256: str = "a" * 64) -> dict:
    return {
        "path": path,
        "chunk_id": f"{path}#{index:04d}",
        "start_line": index + 1,
        "end_line": index + 1,
        "source_sha256": source_sha256,
        "keyword_score": index / 10,
        "rrf_score": 1 / (index + 1),
        "snippet": "本文を転送してはいけない" * 100,
        "embedding": [0.1] * 64,
    }


def test_worker_selection_matches_previous_parent_selection_order_and_exclusions():
    ranked = [
        _candidate("a\\doc.md", index)
        for index in range(12)
    ] + [_candidate("b.md", index) for index in range(12)] + [
        _candidate("c.md", index) for index in range(30)
    ]

    actual = selection.select_candidates(ranked)

    assert [item["chunk_id"] for item in actual["selected"]] == [
        *(f"a/doc.md#{index:04d}" for index in range(8)),
        *(f"b.md#{index:04d}" for index in range(8)),
        *(f"c.md#{index:04d}" for index in range(8)),
    ]
    assert [item["chunk_id"] for item in actual["excluded"]] == [
        *(f"a/doc.md#{index:04d}" for index in range(8, 12)),
        *(f"b.md#{index:04d}" for index in range(8, 12)),
        *(f"c.md#{index:04d}" for index in range(8, 30)),
    ]
    assert actual["counts"] == {"ranked": 54, "selected": 24, "excluded": 30}
    assert all("path" not in item for item in actual["excluded"])
    assert all("path_index" in item and "source_hash_index" in item for item in actual["excluded"])
    normalized = selection.expand_response(selection.validate_response(actual))
    assert [item["path"] for item in normalized["excluded"]] == [
        *(["a/doc.md"] * 4),
        *("b.md" for _ in range(4)),
        *("c.md" for _ in range(22)),
    ]


def test_selection_edge_cases_are_deterministic_without_a_reference_copy():
    assert selection.select_candidates([])["counts"] == {
        "ranked": 0,
        "selected": 0,
        "excluded": 0,
    }

    same_path = [_candidate("same\\doc.md", index) for index in range(9)]
    same_path.extend(_candidate("same/doc.md", index) for index in range(9, 10))
    result = selection.select_candidates(same_path)
    assert result["counts"] == {"ranked": 10, "selected": 8, "excluded": 2}
    assert [item["chunk_id"] for item in result["selected"]] == [
        f"same/doc.md#{index:04d}" for index in range(8)
    ]
    assert [item["chunk_id"] for item in result["excluded"]] == [
        "same/doc.md#0008",
        "same/doc.md#0009",
    ]

    duplicate = [_candidate("duplicate.md", 0) for _ in range(3)]
    duplicate_result = selection.select_candidates(duplicate)
    assert duplicate_result["counts"] == {"ranked": 3, "selected": 3, "excluded": 0}


def test_candidate_selection_honors_cancellation_and_deadline_inside_selection():
    ranked = [_candidate("a.md", index) for index in range(3)]

    def cancel():
        raise web_services.CancelledError("cancel", code="cancelled")

    with pytest.raises(web_services.CancelledError):
        selection.select_candidates(ranked, cancel_check=cancel)
    with pytest.raises(TimeoutError):
        selection.select_candidates(ranked, deadline=time.monotonic() - 1)


def test_worker_selection_honors_expired_parent_deadline():
    chunk = {
        "path": "a.md",
        "chunk_id": "a.md#0001",
        "start_line": 1,
        "end_line": 1,
        "text": "対象",
        "file_sha256": "a" * 64,
    }
    with pytest.raises(TimeoutError, match="candidate selection deadline"):
        deep_transport._execute(
            {
                "operation": "retrieve",
                "query": "対象",
                "chunks": [chunk],
                "_worker_deadline_monotonic": time.monotonic() - 1,
            }
        )


def test_parent_hydrates_only_selected_candidates_from_its_snapshot():
    chunks = [
        {
            "path": "資料.md",
            "chunk_id": "資料.md#0001",
            "start_line": 1,
            "end_line": 2,
            "text": "親が保持する原文",
            "file_sha256": "b" * 64,
        }
    ]
    response = selection.select_candidates(
        [{**chunks[0], "source_sha256": chunks[0]["file_sha256"]}],
        source_chunks=chunks,
    )

    hydrated = deep_research._prepare_retrieve_response(response, chunks)

    assert hydrated["selected"][0]["text"] == "親が保持する原文"
    assert "text" not in response["selected"][0]
    assert "embedding" not in response["selected"][0]
    assert hydrated["counts"] == {"ranked": 1, "selected": 1, "excluded": 0}


def test_real_worker_retrieval_returns_compact_versioned_response():
    chunks = [
        {
            "path": "source.md",
            "chunk_id": "source.md#0001",
            "start_line": 1,
            "end_line": 2,
            "text": "公共調達委員会の対象条件",
            "file_sha256": "c" * 64,
            "text_sha256": "d" * 64,
        }
    ]

    response = deep_transport.run_worker(
        {"operation": "retrieve", "query": "公共調達委員会", "chunks": chunks},
        timeout=10,
    )
    normalized = selection.validate_response(response)

    assert normalized["schema_version"] == selection.CANDIDATE_RESPONSE_SCHEMA_VERSION
    assert normalized["counts"] == {"ranked": 1, "selected": 1, "excluded": 0}
    assert "text" not in normalized["selected"][0]
    assert "source_sha256" in normalized["selected"][0]


def test_multiple_queries_and_rounds_keep_each_retrieval_budget(tmp_path, monkeypatch):
    for index in range(5):
        (tmp_path / f"source-{index}.md").write_text("対象\n" * 20_000, encoding="utf-8")

    calls = []
    gap_calls = 0

    def retrieve(query, chunks, **kwargs):
        calls.append(query)
        kwargs["cancel_check"]()
        ranked = [{**chunk, "source_sha256": chunk["file_sha256"]} for chunk in chunks]
        return selection.select_candidates(ranked, source_chunks=chunks)

    def model(_model, system, _user, **_kwargs):
        nonlocal gap_calls
        if system == deep_research.DEEP_GAPS_SYSTEM:
            gap_calls += 1
            return {"queries": ["q3"] if gap_calls == 1 else [], "unresolved": []}
        if system == deep_research.DEEP_VERIFY_SYSTEM:
            return {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
        return {
            "subject": "対象",
            "scope": "範囲",
            "conditions": ["対象"],
            "exceptions": [],
            "references": [],
        }

    patch_retrieval(monkeypatch, deep_research, retrieve)
    monkeypatch.setattr(deep_research, "_initial_viewpoints", lambda query: ["q1", "q2"])
    monkeypatch.setattr(deep_research, "_call_ollama_json", model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", lambda *args, **kwargs: "対象。[E1]")

    result = deep_research.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300
    )

    assert calls == ["q1", "q2", "q3"]
    assert len(result.diagnostics["retrieval_counts"]) == 3
    assert all(
        counts["ranked"] > 40
        and counts["selected"] == 40
        and counts["excluded"] == counts["ranked"] - 40
        for counts in result.diagnostics["retrieval_counts"]
    )


def test_empty_path_keeps_legacy_nonfatal_source_missing_behavior():
    ranked = [_candidate("", 0)]
    response = selection.select_candidates(ranked)

    assert selection.validate_response(response)["selected"][0]["path"] == ""
    hydrated = selection.hydrate_selected(
        response,
        [
            {
                "path": "actual.md",
                "chunk_id": ranked[0]["chunk_id"],
                "start_line": 1,
                "end_line": 1,
                "text": "本文",
                "file_sha256": "b" * 64,
            }
        ],
    )
    assert "text" not in hydrated["selected"][0]


def test_worker_must_return_compact_excluded_references_only():
    response = selection.select_candidates([_candidate("a.md", index) for index in range(9)])
    expanded = selection.expand_response(response)

    with pytest.raises(ValueError, match="expanded excluded"):
        selection.validate_response(expanded)


@pytest.mark.parametrize(
    ("field", "value"),
    [("chunk_id", ""), ("start_line", 0), ("end_line", 0), ("end_line", -1)],
)
def test_missing_chunk_or_invalid_line_is_rejected_at_worker_boundary(field, value):
    candidate = _candidate("a.md", 0)
    candidate[field] = value

    with pytest.raises(ValueError):
        selection.select_candidates([candidate])


def test_empty_path_reaches_parent_source_missing_instead_of_model_error(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    source.write_text("# 規程\n対象\n", encoding="utf-8")
    invalid_path_response = selection.select_candidates([_candidate("", 0)])
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda *args, **kwargs: invalid_path_response,
    )
    monkeypatch.setattr(deep_research, "_initial_viewpoints", lambda query: ["対象"])

    result = deep_research.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300
    )

    assert result.stop_reason == "unresolved_scope"
    assert any(entry["reason"] == "source_missing" for entry in result.ledger)
    assert not any(entry["reason"] == "model_error" for entry in result.ledger)


def test_large_synthetic_retrieval_is_body_free_and_under_worker_limit():
    ranked = [
        _candidate(f"資料-{index % 937:03d}-委員会.md", index)
        for index in range(18_241)
    ]

    response = selection.select_candidates(ranked)
    encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    assert response["counts"] == {
        "ranked": 18_241,
        "selected": 40,
        "excluded": 18_201,
    }
    legacy_encoded = json.dumps(
        {"ok": True, "value": ranked}, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert len(response["selected"]) <= 40
    assert len(encoded) < deep_transport.WORKER_OUTPUT_MAX_BYTES
    assert len(legacy_encoded) > deep_transport.WORKER_OUTPUT_MAX_BYTES
    assert all("snippet" not in item and "embedding" not in item for item in response["selected"])
    assert all("snippet" not in item and "embedding" not in item for item in response["excluded"])


def test_worker_output_limit_uses_fixed_candidate_error():
    payload = {"operation": "retrieve"}
    with pytest.raises(deep_transport.CandidateTransferLimitError) as exc_info:
        deep_transport._encode_success(payload, {"value": "x" * 100}, max_bytes=64)
    assert exc_info.value.code == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert json.loads(deep_transport._error_payload(exc_info.value.code)) == {
        "ok": False,
        "code": selection.CANDIDATE_TRANSFER_LIMIT_CODE,
    }

    with pytest.raises(ValueError):
        deep_transport._encode_success({"operation": "chat"}, {"value": "x" * 100}, max_bytes=64)


def test_worker_output_limit_accepts_exact_boundary_and_rejects_one_byte_over():
    payload = {"operation": "retrieve"}
    value = {"value": "候補" * 30}
    encoded = json.dumps(
        {"ok": True, "value": value}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

    assert deep_transport._encode_success(payload, value, max_bytes=len(encoded)) == encoded
    with pytest.raises(deep_transport.CandidateTransferLimitError):
        deep_transport._encode_success(payload, value, max_bytes=len(encoded) - 1)


class _FakeWorkerProcess:
    def __init__(self, output):
        self.output = output
        self.returncode = 0

    def communicate(self, input=None, timeout=None):
        return (self.output, b"")

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class _PendingWorkerProcess:
    def __init__(self):
        self.returncode = None

    def communicate(self, input=None, timeout=None):
        if timeout is not None:
            raise deep_transport.subprocess.TimeoutExpired("fake-worker", timeout)
        return (b"", b"")

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_retrieve_worker_cancel_or_deadline_reaps_process(monkeypatch, mode):
    process = _PendingWorkerProcess()
    monkeypatch.setattr(deep_transport.subprocess, "Popen", lambda *args, **kwargs: process)
    payload = {"operation": "retrieve", "query": "対象", "chunks": []}

    if mode == "cancel":
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise web_services.CancelledError("cancel", code="cancelled")

        with pytest.raises(web_services.CancelledError):
            deep_transport.run_worker(payload, timeout=1, cancel_check=cancel)
    else:
        with pytest.raises(TimeoutError):
            deep_transport.run_worker(payload, timeout=0.01)
    assert process.poll() is not None


def test_parent_receive_overflow_is_classified_as_candidate_transfer(monkeypatch):
    process = _FakeWorkerProcess(b"x" * (deep_transport.WORKER_OUTPUT_MAX_BYTES + 1))
    monkeypatch.setattr(deep_transport.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(deep_transport.DeepWorkerError) as exc_info:
        deep_transport.run_worker(
            {"operation": "retrieve", "query": "対象", "chunks": []}, timeout=1
        )
    assert exc_info.value.code == selection.CANDIDATE_TRANSFER_LIMIT_CODE


def test_unknown_worker_error_code_fails_closed_to_worker_error(monkeypatch):
    process = _FakeWorkerProcess(b'{"ok":false,"code":"secret_marker"}')
    monkeypatch.setattr(deep_transport.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(deep_transport.DeepWorkerError) as exc_info:
        deep_transport.run_worker(
            {"operation": "retrieve", "query": "対象", "chunks": []}, timeout=1
        )
    assert exc_info.value.code == "worker_error"


@pytest.mark.parametrize(
    "response",
    [
        {
            "schema_version": selection.CANDIDATE_RESPONSE_SCHEMA_VERSION + 1,
            "paths": [],
            "source_hashes": [],
            "selected": [],
            "excluded": [],
            "counts": {"ranked": 0, "selected": 0, "excluded": 0},
        },
        {
            "schema_version": selection.CANDIDATE_RESPONSE_SCHEMA_VERSION,
            "paths": [],
            "source_hashes": [],
            "selected": [],
            "excluded": [],
            "counts": {"ranked": 1, "selected": 0, "excluded": 0},
        },
        {
            "schema_version": selection.CANDIDATE_RESPONSE_SCHEMA_VERSION,
            "paths": [],
            "source_hashes": [],
            "selected": [],
            "excluded": [],
            "counts": {"ranked": 0, "selected": 0, "excluded": 0},
            "extra": True,
        },
    ],
)
def test_invalid_or_unknown_retrieve_response_fails_closed(response):
    with pytest.raises(ValueError):
        selection.validate_response(response)


def test_initial_candidate_transfer_failure_is_failed_with_fixed_reason(tmp_path, monkeypatch):
    (tmp_path / "source.md").write_text("# 規程\n対象\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise deep_transport.DeepWorkerError(code=selection.CANDIDATE_TRANSFER_LIMIT_CODE)

    patch_retrieval(monkeypatch, deep_research, fail)
    result = deep_research.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300
    )

    assert result.status == "failed"
    assert result.stop_reason == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE in result.answer


def test_later_candidate_transfer_failure_is_partial_when_evidence_exists(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    source.write_text("# 規程\n対象\n", encoding="utf-8")
    first = {"done": False}

    def retrieve(query, chunks, **kwargs):
        if first["done"]:
            raise deep_transport.DeepWorkerError(code=selection.CANDIDATE_TRANSFER_LIMIT_CODE)
        first["done"] = True
        chunk = chunks[0]
        return selection.select_candidates(
            [{**chunk, "source_sha256": chunk["file_sha256"]}], source_chunks=chunks
        )

    patch_retrieval(monkeypatch, deep_research, retrieve)
    monkeypatch.setattr(deep_research, "_initial_viewpoints", lambda query: ["対象"])
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_json",
        lambda model, system, user, **kwargs: (
            {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
            if system == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": ["追加"], "unresolved": []}
            if system == deep_research.DEEP_GAPS_SYSTEM
            else {
                "subject": "対象",
                "scope": "対象の範囲",
                "conditions": ["対象"],
                "exceptions": [],
                "references": ["追加"],
            }
        ),
    )
    monkeypatch.setattr(
        deep_research, "_call_ollama_text", lambda *args, **kwargs: "対象。[E1]"
    )
    result = deep_research.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300
    )

    assert result.status == "partial"
    assert result.stop_reason == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert result.evidence


@pytest.mark.parametrize(
    ("initial", "reasons", "expected"),
    [
        (
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
            {"document_limit", "unit_limit", "round_limit"},
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
        ),
        (
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
            {"evidence_budget"},
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
        ),
        (
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
            {"time_budget"},
            "time_budget",
        ),
        (
            "model_error",
            {selection.CANDIDATE_TRANSFER_LIMIT_CODE},
            selection.CANDIDATE_TRANSFER_LIMIT_CODE,
        ),
    ],
)
def test_candidate_transfer_reason_is_not_hidden_by_other_limits(initial, reasons, expected):
    assert deep_research._select_stop_reason(initial, reasons) == expected


def test_transfer_failure_wins_after_a_later_document_limit(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text(
        "# A1\n"
        + ("対象1\n" * 3_000)
        + "# A2\n"
        + ("対象2\n" * 3_000),
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("# B\n別対象\n", encoding="utf-8")
    calls = []
    gap_calls = 0

    def retrieve(query, chunks, **kwargs):
        calls.append(query)
        kwargs["cancel_check"]()
        by_path = {}
        for chunk in chunks:
            by_path.setdefault(chunk["path"], []).append(chunk)
        if query == "q1":
            ranked = [
                {
                    **by_path["a.md"][0],
                    "source_sha256": by_path["a.md"][0]["file_sha256"],
                }
            ]
        elif query == "q2":
            ranked = [
                {
                    **by_path["a.md"][-1],
                    "source_sha256": by_path["a.md"][-1]["file_sha256"],
                },
                {
                    **by_path["b.md"][0],
                    "source_sha256": by_path["b.md"][0]["file_sha256"],
                },
            ]
        else:
            raise deep_transport.DeepWorkerError(
                code=selection.CANDIDATE_TRANSFER_LIMIT_CODE
            )
        return selection.select_candidates(ranked, source_chunks=chunks)

    def model(_model, system, _user, **_kwargs):
        nonlocal gap_calls
        if system == deep_research.DEEP_GAPS_SYSTEM:
            gap_calls += 1
            return {"queries": ["q2"] if gap_calls == 1 else ["q3"], "unresolved": []}
        if system == deep_research.DEEP_VERIFY_SYSTEM:
            return {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
        return {
            "subject": "対象",
            "scope": "範囲",
            "conditions": ["対象"],
            "exceptions": [],
            "references": [],
        }

    patch_retrieval(monkeypatch, deep_research, retrieve)
    monkeypatch.setattr(deep_research, "_initial_viewpoints", lambda query: ["q1"])
    monkeypatch.setattr(deep_research, "_call_ollama_json", model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", lambda *args, **kwargs: "対象。[E1]")

    result = deep_research.run_deep_research(
        "対象",
        model="fake",
        source_root=tmp_path,
        timeout_seconds=300,
        max_documents=1,
    )

    assert calls == ["q1", "q2", "q3"]
    assert result.status == "partial"
    assert result.stop_reason == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert any(entry["reason"] == "document_limit" for entry in result.ledger)


def test_web_exposes_candidate_transfer_reason_in_japanese():
    result = deep_research.DeepResearchResult(
        status="failed",
        stop_reason=selection.CANDIDATE_TRANSFER_LIMIT_CODE,
    )

    event = web_services.build_deep_evidence_event(result)

    assert event["routeReason"] == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert event["routeReasonMessage"] == selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE
    assert selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE in event["warnings"]


def test_web_failed_candidate_transfer_uses_fixed_sse_code(monkeypatch):
    result = deep_research.DeepResearchResult(
        status="failed",
        stop_reason=selection.CANDIDATE_TRANSFER_LIMIT_CODE,
    )
    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_deep_model", lambda: "fake")
    monkeypatch.setattr(web_services, "run_deep_research", lambda *args, **kwargs: result)
    token = web_services.CancellationToken(timeout=300)
    entry = web_services.JobEntry(token)

    web_services.run_search(
        "対象", "off", web_services._BroadcastQueue(entry), token, "rid", mode="deep"
    )

    events = []
    subscriber = entry.add_subscriber()
    while not subscriber.empty():
        events.append(subscriber.get_nowait())
    error = next(event for event in events if event["type"] == "error")
    assert error["code"] == selection.CANDIDATE_TRANSFER_LIMIT_CODE
    assert error["message"] == selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE


def test_cli_markdown_and_replay_preserve_the_same_human_reason(monkeypatch, capsys):
    result = deep_research.DeepResearchResult(
        status="failed",
        stop_reason=selection.CANDIDATE_TRANSFER_LIMIT_CODE,
    )
    monkeypatch.setattr(deep_research, "detect_deep_model", lambda: "fake")
    monkeypatch.setattr(deep_research, "run_deep_research", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        ["search.py", "対象", "--mode", "deep", "--timeout-seconds", "300"],
    )

    cli_search.main()
    assert selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE in capsys.readouterr().out

    evidence_event = web_services.build_deep_evidence_event(result)
    entry = web_services.JobEntry(
        web_services.CancellationToken(300),
        fingerprint=("s", "対象", "off", 300, "deep"),
    )
    entry.broadcast(evidence_event)
    entry.broadcast(
        {
            "type": "done",
            "status": result.status,
            "stopReason": result.stop_reason,
            "stopReasonMessage": selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE,
        }
    )
    subscriber = entry.add_subscriber()
    replay = []
    while not subscriber.empty():
        replay.append(subscriber.get_nowait())
    assert replay[0]["routeReasonMessage"] == selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE
    assert replay[1]["stopReasonMessage"] == selection.CANDIDATE_TRANSFER_LIMIT_MESSAGE

    html = Path(web_services.__file__).resolve().parent / "web" / "index.html"
    html_text = html.read_text(encoding="utf-8")
    assert "meta.stopReasonMessage" in html_text
    assert "evidence.routeReasonMessage" in html_text
