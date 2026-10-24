"""親子展開P1b: query context補助API（子のEmbeddingスコアによる予算超過時の
優先順位付け）の決定的テスト。

対象: embedding_search_multi_with_context / _max_similarity_to_context /
_order_ranges_for_selection / _expand_single_parent への統合。
既存の embedding_search_multi の戻り値契約は変えず、query_context /
embed_index を渡さない既定経路は常に原文順であることを確認する。
"""

import math
import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def _write(tmp_path, rel_path, text):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


# --- embedding_search_multi_with_context -------------------------------------


def test_with_context_returns_same_results_as_plain_multi(monkeypatch):
    cache = {
        "entries": {
            "a.md#0001": {
                "path": "a.md",
                "chunk_id": "a.md#0001",
                "text": "本文A",
                "embedding": [1.0, 0.0],
                "start_line": 1,
                "end_line": 1,
            },
        }
    }
    monkeypatch.setattr(
        search, "_get_embeddings", lambda texts, embed_model, **k: [[1.0, 0.0] for _ in texts]
    )
    index = search._build_embed_index(cache)

    plain = search.embedding_search_multi(["q"], "m", index, top_k=8)
    with_context, context = search.embedding_search_multi_with_context(["q"], "m", index, top_k=8)

    assert plain == with_context
    assert context is not None
    assert context.queries == ["q"]
    assert context.vectors["q"] == [1.0, 0.0]
    assert context.embed_model == "m"


def test_with_context_returns_none_context_on_embedding_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise search.EmbeddingBatchError("EMBED_TRANSPORT_ERROR", "failed")

    monkeypatch.setattr(search, "_get_embeddings", fail)
    index = search._build_embed_index(
        {"entries": {"a.md#0001": {"path": "a.md", "embedding": [1.0]}}}
    )

    results, context = search.embedding_search_multi_with_context(["q"], "m", index, top_k=8)

    assert results == {"q": []}
    assert context is None


def test_with_context_returns_none_when_no_queries_or_empty_index():
    index = search._build_embed_index({"entries": {}})
    results, context = search.embedding_search_multi_with_context(["q"], "m", index, top_k=8)
    assert results == {"q": []}
    assert context is None


# --- _max_similarity_to_context -----------------------------------------------


def _context(queries_vectors, embed_model="m"):
    return search.QueryContext(
        queries=[q for q, _v in queries_vectors],
        vectors=dict(queries_vectors),
        embed_model=embed_model,
    )


def test_max_similarity_picks_best_among_queries():
    index = search._EmbedIndex(
        chunk_ids=["c1"], entries=[{}], vectors=[[1.0, 0.0]], norms=[1.0]
    )
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q1", [0.0, 1.0]), ("q2", [1.0, 0.0])])  # q2は完全一致(cos=1.0)

    similarity = search._max_similarity_to_context("c1", lookup, index, ctx)

    assert similarity is not None
    assert math.isclose(similarity, 1.0, rel_tol=1e-6)


def test_max_similarity_returns_none_for_unknown_chunk():
    index = search._EmbedIndex(chunk_ids=["c1"], entries=[{}], vectors=[[1.0]], norms=[1.0])
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q1", [1.0])])

    assert search._max_similarity_to_context("missing", lookup, index, ctx) is None


def test_max_similarity_returns_none_for_zero_norm_chunk():
    index = search._EmbedIndex(chunk_ids=["c1"], entries=[{}], vectors=[[0.0, 0.0]], norms=[0.0])
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q1", [1.0, 0.0])])

    assert search._max_similarity_to_context("c1", lookup, index, ctx) is None


def test_max_similarity_returns_none_for_dimension_mismatch():
    index = search._EmbedIndex(chunk_ids=["c1"], entries=[{}], vectors=[[1.0, 0.0]], norms=[1.0])
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q1", [1.0, 0.0, 0.0])])  # 次元不一致

    assert search._max_similarity_to_context("c1", lookup, index, ctx) is None


def test_max_similarity_returns_none_when_all_query_vectors_invalid():
    index = search._EmbedIndex(chunk_ids=["c1"], entries=[{}], vectors=[[1.0, 0.0]], norms=[1.0])
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q1", [0.0, 0.0])])  # ゼロnormのquery vector

    assert search._max_similarity_to_context("c1", lookup, index, ctx) is None


# --- _order_ranges_for_selection -----------------------------------------------


def test_order_ranges_falls_back_to_document_order_without_context():
    ranges = [(10, 12), (1, 3), (5, 7)]
    ordered, reason = search._order_ranges_for_selection(
        [], ranges, query_context=None, embed_index=None, chunk_lookup=None
    )
    assert ordered == ranges
    assert reason == "embedding_unavailable"


def test_order_ranges_prefers_higher_similarity_chunk():
    child_chunks = [
        {"chunk_id": "low", "start_line": 1, "end_line": 2},
        {"chunk_id": "high", "start_line": 5, "end_line": 6},
    ]
    ranges = [(1, 2), (5, 6)]
    index = search._EmbedIndex(
        chunk_ids=["low", "high"],
        entries=[{}, {}],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        norms=[1.0, 1.0],
    )
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q", [0.0, 1.0])])  # "high" と完全一致

    ordered, reason = search._order_ranges_for_selection(
        child_chunks, ranges, query_context=ctx, embed_index=index, chunk_lookup=lookup
    )

    assert ordered[0] == (5, 6)
    assert reason == ""


def test_order_ranges_uses_max_similarity_for_multi_chunk_range():
    """複数チャンクからなる範囲は配下チャンクの最大類似度を選択用に使う。"""
    child_chunks = [
        {"chunk_id": "weak", "start_line": 1, "end_line": 2},
        {"chunk_id": "strong", "start_line": 3, "end_line": 4},
        {"chunk_id": "other", "start_line": 10, "end_line": 11},
    ]
    ranges = [(1, 4), (10, 11)]  # 最初の範囲は weak+strongの2チャンクからなる
    index = search._EmbedIndex(
        chunk_ids=["weak", "strong", "other"],
        entries=[{}, {}, {}],
        vectors=[[0.1, 0.99], [0.0, 1.0], [1.0, 0.0]],
        norms=[math.hypot(0.1, 0.99), 1.0, 1.0],
    )
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q", [0.0, 1.0])])

    ordered, _reason = search._order_ranges_for_selection(
        child_chunks, ranges, query_context=ctx, embed_index=index, chunk_lookup=lookup
    )

    assert ordered[0] == (1, 4)


def test_order_ranges_tiebreaks_by_keyword_score_then_document_order():
    child_chunks = [
        {"chunk_id": "a", "start_line": 1, "end_line": 1, "keyword_score": 1.0},
        {"chunk_id": "b", "start_line": 5, "end_line": 5, "keyword_score": 5.0},
    ]
    ranges = [(1, 1), (5, 5)]
    index = search._EmbedIndex(
        chunk_ids=["a", "b"], entries=[{}, {}], vectors=[[1.0], [1.0]], norms=[1.0, 1.0]
    )
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q", [1.0])])  # 両方とも類似度1.0で同点

    ordered, _reason = search._order_ranges_for_selection(
        child_chunks, ranges, query_context=ctx, embed_index=index, chunk_lookup=lookup
    )

    assert ordered[0] == (5, 5)  # keyword_scoreが高い方を優先


def test_order_ranges_falls_back_when_no_vectors_available():
    child_chunks = [{"chunk_id": "missing", "start_line": 1, "end_line": 1}]
    ranges = [(1, 1)]
    index = search._EmbedIndex(chunk_ids=[], entries=[], vectors=[], norms=[])
    lookup = search._build_chunk_id_lookup(index)
    ctx = _context([("q", [1.0])])

    ordered, reason = search._order_ranges_for_selection(
        child_chunks, ranges, query_context=ctx, embed_index=index, chunk_lookup=lookup
    )

    assert ordered == ranges
    assert reason == "embedding_unavailable"


# --- _expand_single_parent / expand_parent_candidates 統合 --------------------


def test_expand_prioritizes_similar_child_under_budget(tmp_path):
    """予算超過（max_ranges制約）時、類似度の高い子節が優先採用される。"""
    text = "# 親\n\n## 無関係な子\n無関係な内容の本文\n## 関連する子\n関連する内容の本文\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")
    unrelated_chunk = next(c for c in chunks if c["heading"] == "無関係な子")
    relevant_chunk = next(c for c in chunks if c["heading"] == "関連する子")

    index = search._EmbedIndex(
        chunk_ids=[unrelated_chunk["chunk_id"], relevant_chunk["chunk_id"]],
        entries=[{}, {}],
        vectors=[[0.0, 1.0], [1.0, 0.0]],
        norms=[1.0, 1.0],
    )
    ctx = search.QueryContext(queries=["q"], vectors={"q": [1.0, 0.0]}, embed_model="m")

    items, meta = search.expand_parent_candidates(
        [parent_chunk],
        chunks,
        source_root=tmp_path,
        max_ranges_per_parent=1,
        query_context=ctx,
        embed_index=index,
    )

    assert len(items) == 1
    assert "関連する内容" in items[0]["snippet"]
    assert meta[0]["partial"] is True


def test_expand_display_order_is_document_order_even_when_selection_differs(tmp_path):
    """選択順序（類似度降順）と表示順序（原文順）は独立している。

    子1・子2は行範囲上連続しているため両方採用されると1つの表示範囲へ
    まとまるが、その中身のテキストは選択順位（子2が先）ではなく常に
    原文順（子1が先）になる。
    """
    text = "# 親\n\n## 子1\n本文1\n## 子2\n本文2\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")
    child1 = next(c for c in chunks if c["heading"] == "子1")
    child2 = next(c for c in chunks if c["heading"] == "子2")

    index = search._EmbedIndex(
        chunk_ids=[child1["chunk_id"], child2["chunk_id"]],
        entries=[{}, {}],
        vectors=[[0.0, 1.0], [1.0, 0.0]],  # 子2の方が類似度が高い想定
        norms=[1.0, 1.0],
    )
    ctx = search.QueryContext(queries=["q"], vectors={"q": [1.0, 0.0]}, embed_model="m")

    items, _meta = search.expand_parent_candidates(
        [parent_chunk],
        chunks,
        source_root=tmp_path,
        max_ranges_per_parent=2,
        query_context=ctx,
        embed_index=index,
    )

    assert len(items) == 1
    assert items[0]["snippet"].index("本文1") < items[0]["snippet"].index("本文2")
    assert items[0]["group_order"] == 1


def test_expand_selection_order_differs_from_document_order_for_non_contiguous_sections(
    tmp_path,
):
    """子節候補が予算(max_ranges_per_parent)を超える場合、選択順位の高い
    （類似度の高い）子節が優先採用される（行範囲上は連続していても
    選択の単位としては別候補として扱う）。"""
    text = "# 親\n\n## 子1\n本文1\n## 子2\n本文2\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")
    child1 = next(c for c in chunks if c["heading"] == "子1")
    child2 = next(c for c in chunks if c["heading"] == "子2")

    index = search._EmbedIndex(
        chunk_ids=[child1["chunk_id"], child2["chunk_id"]],
        entries=[{}, {}],
        vectors=[[0.0, 1.0], [1.0, 0.0]],  # 子2の方が類似度が高い
        norms=[1.0, 1.0],
    )
    ctx = search.QueryContext(queries=["q"], vectors={"q": [1.0, 0.0]}, embed_model="m")

    items, meta = search.expand_parent_candidates(
        [parent_chunk],
        chunks,
        source_root=tmp_path,
        max_ranges_per_parent=1,
        query_context=ctx,
        embed_index=index,
    )

    assert len(items) == 1
    assert "本文2" in items[0]["snippet"]
    assert "本文1" not in items[0]["snippet"]
    assert meta[0]["partial"] is True


def test_expand_falls_back_to_document_order_without_vectors(tmp_path):
    """embed_index/query_contextを渡さない既定経路は常に原文順（P1a互換）。"""
    text = "# 親\n\n## 子1\n本文1\n## 子2\n本文2\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")

    items, _meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path, max_ranges_per_parent=1
    )

    assert len(items) == 1
    assert "本文1" in items[0]["snippet"]  # 原文順で最初の子が採用される
