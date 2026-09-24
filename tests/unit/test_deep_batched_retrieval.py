"""Batched round retrieval and document-relevance reading order; no live model or user corpus.

2026-09-23 real-corpus acceptance: the document limit (5) was filled by the
first five documents to appear in the first viewpoint's ranking, so the core
document (6 selected chunks in the top 20 of every viewpoint) was never read.
Each viewpoint also paid a full worker start, 442MB embedding-cache load and
35.7MB chunk transfer.
"""

import json

import pytest

import deep_candidate_selection as selection
import deep_research
import deep_transport
import search
from deep_fakes import patch_retrieval


VERDICT = {"supported": True, "contradictions": [], "missing_conditions": [], "unsupported_claims": []}


def _model(name, system, user, **kwargs):
    if system == deep_research.DEEP_VERIFY_SYSTEM:
        return dict(VERDICT)
    if system == deep_research.DEEP_GAPS_SYSTEM:
        return {"queries": [], "unresolved": []}
    return {"subject": "対象", "scope": "範囲", "conditions": ["対象である"], "exceptions": [], "references": []}


def _synthetic_chunks(tmp_path):
    root = tmp_path / "skill-source"
    root.mkdir()
    (root / "notice.md").write_text(
        "# 公共調達委員会\n導入。\n## 審査案件\n委員会の審査案件は次のとおり。\n## 除外\n審査案件としない案件。\n",
        encoding="utf-8",
    )
    (root / "other.md").write_text("# 別制度\n公共調達の一般的な説明。\n", encoding="utf-8")
    return root, search.build_source_chunks(root)


# --- worker: retrieve_many -------------------------------------------------------


def test_real_worker_retrieve_many_matches_single_retrieve_per_query(tmp_path):
    _, chunks = _synthetic_chunks(tmp_path)
    queries = ["公共調達委員会 審査案件", "除外 案件"]

    many = selection.validate_many_response(
        deep_transport.run_worker(
            {"operation": "retrieve_many", "queries": queries, "chunks": chunks}, timeout=20
        ),
        len(queries),
    )
    singles = [
        selection.validate_response(
            deep_transport.run_worker(
                {"operation": "retrieve", "query": query, "chunks": chunks}, timeout=20
            )
        )
        for query in queries
    ]

    for batched, single in zip(many, singles):
        assert batched["selected"] == single["selected"]
        assert batched["counts"] == single["counts"]
        assert [e["chunk_id"] for e in batched["excluded"]] == [e["chunk_id"] for e in single["excluded"]]


def test_batched_excluded_references_carry_no_scores_but_stay_valid():
    ranked = [
        {"path": "a.md", "chunk_id": f"a.md#{i}", "start_line": i + 1, "end_line": i + 1,
         "source_sha256": "a" * 64, "rrf_score": 1 / (i + 1), "keyword_score": 0.5}
        for i in range(12)
    ]
    response = selection.select_candidates(ranked, max_per_file=8)
    stripped = selection.strip_excluded_scores(response)

    assert len(stripped["excluded"]) == 4
    assert all(not set(ref) & {"score", "keyword_score", "embedding_score", "rrf_score"} for ref in stripped["excluded"])
    assert stripped["selected"] == response["selected"]
    assert selection.validate_response(stripped) is stripped
    assert len(json.dumps(stripped)) < len(json.dumps(response))


@pytest.mark.parametrize("queries", [[], "対象", [1], None])
def test_real_worker_retrieve_many_rejects_invalid_queries(queries):
    with pytest.raises(deep_transport.DeepWorkerError) as exc_info:
        deep_transport.run_worker(
            {"operation": "retrieve_many", "queries": queries, "chunks": []}, timeout=20
        )
    assert exc_info.value.code == "worker_error"


def test_retrieve_many_oversize_output_is_a_candidate_transfer_limit():
    with pytest.raises(deep_transport.CandidateTransferLimitError):
        deep_transport._encode_success({"operation": "retrieve_many"}, {"value": "x" * 100}, max_bytes=64)


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": 1, "responses": []},
        {"schema_version": 2, "responses": [selection.select_candidates([])]},
        {"schema_version": 1, "responses": [selection.select_candidates([])], "extra": 1},
        [selection.select_candidates([])],
        {"schema_version": 1, "responses": [{"schema_version": 2}]},
    ],
)
def test_validate_many_response_fails_closed(value):
    with pytest.raises(ValueError):
        selection.validate_many_response(value, 1)


# --- parent: one worker call per round, per-viewpoint fallback --------------------


class _Budget:
    check = staticmethod(lambda: None)

    def remaining(self):
        return 1000.0


def _recorders(monkeypatch, *, many_error=None):
    calls = {"many": [], "single": []}

    def many(queries, chunks, **kwargs):
        calls["many"].append(list(queries))
        if many_error is not None:
            raise many_error
        return [selection.select_candidates([]) for _ in queries]

    def single(query, chunks, **kwargs):
        calls["single"].append(query)
        return selection.select_candidates([])

    monkeypatch.setattr(deep_research, "_retrieve_candidates_many", many)
    monkeypatch.setattr(deep_research, "_retrieve_candidates", single)
    return calls


def test_round_uses_one_batched_call_for_several_viewpoints(monkeypatch):
    calls = _recorders(monkeypatch)
    responses = deep_research._retrieve_round(["a", "b", "c"], [], _Budget(), 125)
    assert calls == {"many": [["a", "b", "c"]], "single": []}
    assert len(responses) == 3


def test_round_with_one_viewpoint_uses_the_single_call(monkeypatch):
    calls = _recorders(monkeypatch)
    deep_research._retrieve_round(["a"], [], _Budget(), 125)
    assert calls == {"many": [], "single": ["a"]}


def test_round_falls_back_per_viewpoint_when_the_batch_exceeds_the_transfer_limit(monkeypatch):
    calls = _recorders(
        monkeypatch,
        many_error=deep_transport.DeepWorkerError(code=selection.CANDIDATE_TRANSFER_LIMIT_CODE),
    )
    responses = deep_research._retrieve_round(["a", "b"], [], _Budget(), 125)
    assert calls == {"many": [["a", "b"]], "single": ["a", "b"]}
    assert len(responses) == 2


def test_round_does_not_hide_other_worker_failures_behind_the_fallback(monkeypatch):
    calls = _recorders(monkeypatch, many_error=deep_transport.DeepWorkerError(code="worker_error"))
    with pytest.raises(deep_transport.DeepWorkerError):
        deep_research._retrieve_round(["a", "b"], [], _Budget(), 125)
    assert calls["single"] == []


def test_run_deep_research_with_the_real_batched_worker(tmp_path, monkeypatch):
    root, _ = _synthetic_chunks(tmp_path)
    monkeypatch.setattr(deep_research, "_call_ollama_json", _model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", lambda *a, **k: "委員会の審査案件は次のとおり。[E1]")

    result = deep_research.run_deep_research(
        "公共調達委員会の審査案件", model="fake", source_root=root, timeout_seconds=300
    )

    assert result.evidence
    assert len(result.diagnostics["retrieval_counts"]) == len(deep_research._initial_viewpoints("公共調達委員会の審査案件"))


# --- reading order: documents by summed relevance ---------------------------------


def _ref(path, line):
    return {"path": path, "start_line": line, "end_line": line, "chunk_id": f"{path}#{line}"}


def test_documents_are_ordered_by_summed_relevance_not_first_appearance():
    first = [_ref("noise-1.md", 1), _ref("core.md", 10), _ref("core.md", 20)]
    second = [_ref("core.md", 20), _ref("core.md", 10), _ref("noise-2.md", 1)]
    heading = {"path": "core.md", "start_line": 30, "end_line": 31}

    ordered = list(deep_research._order_by_document_relevance([first, second], {"core.md": [heading]}))

    assert [key[0] for key in ordered] == ["core.md", "core.md", "core.md", "noise-1.md", "noise-2.md"]
    # Inside a document: summed candidate score (line 20: 1/63+1/61 > line 10: 1/62+1/63);
    # heading-derived candidates follow the ranked ones.
    assert ordered[:3] == [("core.md", 20, 20), ("core.md", 10, 10), ("core.md", 30, 31)]


def test_equal_scores_keep_first_seen_order():
    ordered = list(
        deep_research._order_by_document_relevance(
            [[_ref("b.md", 1), _ref("a.md", 1)], [_ref("a.md", 1), _ref("b.md", 1)]]
        )
    )
    assert [key[0] for key in ordered] == ["b.md", "a.md"]


def test_core_document_is_read_even_if_five_other_documents_rank_first_in_one_viewpoint(tmp_path, monkeypatch):
    root = tmp_path / "skill-source"
    root.mkdir()
    for index in range(1, 6):
        (root / f"noise-{index}.md").write_text(f"# 無関係{index}\n公共調達の一般論。\n", encoding="utf-8")
    (root / "core.md").write_text(
        "# 設置要綱\n## 審査案件\n対象である。\n## 除外\n対象としない。\n## 審査時期\n事前に審査する。\n",
        encoding="utf-8",
    )
    chunks = search.build_source_chunks(root)
    by_path = {}
    for chunk in chunks:
        by_path.setdefault(chunk["path"], []).append({**chunk, "source_sha256": chunk["file_sha256"]})
    noise = [by_path[f"noise-{i}.md"][0] for i in range(1, 6)]
    core = [c for c in by_path["core.md"] if c.get("heading") in {"審査案件", "除外", "審査時期"}]
    assert len(core) == 3

    def retrieve(query, chunks_arg, **kwargs):
        # The first viewpoint ranks five unrelated documents above the core
        # document; every other viewpoint ranks the core document first.
        ranked = [*noise, *core] if query == "v1" else [*core, *noise]
        return selection.select_candidates(ranked, source_chunks=chunks_arg)

    patch_retrieval(monkeypatch, deep_research, retrieve)
    monkeypatch.setattr(deep_research, "_initial_viewpoints", lambda query: ["v1", "v2", "v3"])
    monkeypatch.setattr(deep_research, "_call_ollama_json", _model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", lambda *a, **k: "対象である。[E1]")

    result = deep_research.run_deep_research("審査案件", model="fake", source_root=root, timeout_seconds=300)

    read_paths = {e["path"] for e in result.evidence}
    assert "core.md" in read_paths
    assert result.diagnostics["documents"] <= deep_research.DEEP_MAX_DOCUMENTS
    assert any(e["reason"] == "document_limit" for e in result.ledger)
