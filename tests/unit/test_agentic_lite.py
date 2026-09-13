import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402


def test_search_plan_parse_json_object():
    parsed = search._parse_json_object(
        '```json\n{"keywords":["休暇"],"search_queries":["年次休暇"]}\n```'
    )
    assert parsed["keywords"] == ["休暇"]


def test_search_plan_request_failure_does_not_repeat_llm_call(monkeypatch):
    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)
    calls = {"n": 0}

    def fail_once(request, timeout):
        calls["n"] += 1
        body = search.json.loads(request.data.decode("utf-8"))
        assert body["think"] is False
        assert body["options"]["num_ctx"] == search.NUM_CTX
        assert body["options"]["num_batch"] == search.NUM_BATCH
        assert body["options"]["num_gpu"] == search.NUM_GPU
        assert body["keep_alive"] == search.OLLAMA_KEEP_ALIVE
        raise TimeoutError

    monkeypatch.setattr(search.urllib.request, "urlopen", fail_once)
    monkeypatch.setattr(
        search,
        "extract_keywords_via_llm",
        lambda query, model: (_ for _ in ()).throw(AssertionError("再送してはならない")),
    )

    plan = search.create_search_plan("海外出張の日当", "m")

    assert calls["n"] == 1
    assert plan["keywords"] == ["海外出張の日当"]
    assert plan["fallback"] == "plan_request_failed"


def test_retrieval_pipeline_retries_once_on_low_confidence(monkeypatch):
    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)
    monkeypatch.setattr(search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(
        search,
        "create_search_plan",
        lambda query, model: {
            "keywords": ["休暇"],
            "search_queries": [query],
            "must_find_terms": ["申請期限"],
        },
    )
    calls = {"n": 0}
    chunks = [{"chunk_id": "fixture"}]
    build_calls = {"n": 0}

    def fake_build_source_chunks(source_root):
        build_calls["n"] += 1
        return chunks

    monkeypatch.setattr(search, "build_source_chunks", fake_build_source_chunks)

    def fake_single(query, keywords, embed_model, embed_cache=None, source_chunks=None):
        assert source_chunks is chunks
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                {
                    "path": "rules/leave.md",
                    "chunk_id": "rules/leave.md#0001",
                    "snippet": "休暇",
                    "rrf_score": 0.02,
                }
            ]
        return [
            {
                "path": "rules/leave.md",
                "chunk_id": "rules/leave.md#0002",
                "snippet": "申請期限は3日前",
                "rrf_score": 0.04,
            }
        ]

    monkeypatch.setattr(search, "_run_single_retrieval", fake_single)

    result = search.run_retrieval_pipeline("休暇の申請期限", model="m")

    assert calls["n"] == 2
    assert build_calls["n"] == 1
    assert len(result.attempts) == 2
    assert result.evidence_status in {"partial", "sufficient"}
    assert "根拠ステータス" in result.user_prompt


def test_coverage_terms_expands_japanese_query_into_ngrams():
    """日本語クエリは空白で分割されないため、n-gram へ展開しないと coverage が常に0になる。"""
    terms = search.coverage_terms("海外出張の日当")
    assert "海外" in terms
    assert "日当" in terms
    assert all(len(t) == search.COVERAGE_NGRAM_SIZE for t in terms)


def test_coverage_terms_keeps_whitespace_separated_terms_intact():
    assert search.coverage_terms("travel expense report") == [
        "travel",
        "expense",
        "report",
    ]


def test_coverage_terms_keeps_must_find_terms_verbatim():
    """must_find_terms は検索計画が抽出した精密語であり、展開せず逐語で評価する。"""
    terms = search.coverage_terms("日当", ["宿泊費上限", "14,000円"])
    assert "宿泊費上限" in terms
    assert "14,000円" in terms


def _confidence_matches(source):
    return [
        {
            "path": "regulations/travel.md",
            "heading": "第2条 日当",
            "snippet": "出張1日あたりの日当は、一般社員で2,500円とする。",
            "rrf_score": 0.04,
            "source": source,
        },
        {
            "path": "regulations/travel-2.md",
            "heading": "第3条 宿泊費",
            "snippet": "宿泊費は実費精算とし、1泊あたりの上限額を定める。",
            "rrf_score": 0.03,
            "source": source,
        },
    ]


def test_confidence_coverage_is_non_zero_for_japanese_query():
    """回帰防止: 日本語クエリで coverage が死んでいると sufficient に到達できない。"""
    confidence, status = search._calculate_confidence(
        "一般社員の日当はいくらですか", _confidence_matches("keyword+embedding")
    )
    assert confidence > 0.5
    assert status == "sufficient"


def test_confidence_requires_embedding_evidence_for_sufficient():
    """keyword一致だけでは意味的十分性を示せないため sufficient を宣言しない。"""
    confidence, status = search._calculate_confidence(
        "一般社員の日当はいくらですか", _confidence_matches("keyword")
    )
    assert confidence > 0.5, "confidence値自体は算出される"
    assert status == "partial"


def test_confidence_is_insufficient_without_matches():
    assert search._calculate_confidence("該当のない質問", []) == (0.0, "insufficient")


def test_relevance_support_rejects_weak_embedding_only_near_miss():
    matches = [
        {
            "path": "rules/travel.md",
            "heading": "第2条 日当",
            "snippet": "一般社員の日当は2,500円とする。",
            "score": 0.43,
            "embedding_score": 0.43,
            "source": "embedding",
        }
    ]

    assert search.filter_by_relevance_support("育児休業給付金の給付率", matches) == []


def test_relevance_support_keeps_strong_semantic_candidate_without_exact_anchor():
    match = {
        "path": "rules/leave.md",
        "heading": "休暇申請",
        "snippet": "有給の申請は所属長へ提出する。",
        "embedding_score": 0.78,
        "source": "embedding",
    }

    assert search.filter_by_relevance_support("年次休暇の手続", [match]) == [match]


def test_finalize_rejects_candidates_without_relevance_support():
    matches = [
        {
            "path": "rules/travel.md",
            "heading": "第2条 日当",
            "snippet": "一般社員の日当は2,500円とする。",
            "rrf_score": 0.04,
            "embedding_score": 0.43,
            "source": "embedding",
        }
    ]

    filtered, confidence, status = search.finalize_ranked_matches(
        "育児休業給付金の給付率は何パーセントですか", matches
    )

    assert filtered == []
    assert confidence == 0.0
    assert status == "insufficient"


def test_explicit_scope_conflict_cannot_be_sufficient():
    matches = [
        {
            "path": "regulations/travel-expense-rules.md",
            "heading": "第2条 日当",
            "snippet": "海外出張の日当について参照する。",
            "rrf_score": 0.05,
            "keyword_score": 1.0,
            "embedding_score": 0.7,
            "source": "embedding+keyword",
        },
        {
            "path": "regulations/travel-expense-rules.md",
            "heading": "第3条 宿泊費",
            "snippet": "海外出張の宿泊費について参照する。",
            "rrf_score": 0.045,
            "keyword_score": 1.0,
            "embedding_score": 0.7,
            "source": "embedding+keyword",
        },
        {
            "path": "regulations/travel-expense-rules.md",
            "heading": "国内出張旅費規程",
            "snippet": "海外出張は本規程の対象外とし、別途定める規程による。",
            "rrf_score": 0.04,
            "keyword_score": 1.2,
            "embedding_score": 0.8,
            "source": "embedding+keyword",
        },
    ]

    filtered, confidence, status = search.finalize_ranked_matches(
        "海外出張の日当はいくらですか", matches
    )

    assert len(filtered) == 2
    assert filtered[0]["constraint_conflict"] is True
    assert confidence >= 0.55
    assert status == "partial"


# --- Embeddingインデックス構築とpipelineの接続 ------------------------------


class _FakeWebCancelled(Exception):
    """Web の CancelledError 相当。pipeline が握りつぶさないことを確認する。"""


def _prepare_embed_pipeline(monkeypatch, run_single=None):
    monkeypatch.setattr(
        search,
        "create_search_plan",
        lambda query, model: {"keywords": ["k"], "search_queries": [query]},
    )
    monkeypatch.setattr(search, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(
        search,
        "_is_model_available",
        lambda model, timeout=2.0: search.ModelStatus.AVAILABLE,
    )
    monkeypatch.setattr(search, "_chunk_retrieval_enabled", lambda: True)
    monkeypatch.setattr(search, "build_source_chunks", lambda source_root: [])
    monkeypatch.setattr(
        search,
        "get_embed_index_status",
        lambda embed_model, chunks=None, **_kwargs: {
            "state": "missing",
            "total": len(chunks or []),
        },
    )
    monkeypatch.setattr(
        search,
        "_run_single_retrieval",
        run_single
        or (lambda query, keywords, embed_model, embed_cache=None, source_chunks=None: []),
    )


def test_retrieval_pipeline_never_builds_a_missing_index(monkeypatch):
    """検索要求は明示的なindex jobを開始せずkeyword-onlyへ移る。"""

    def unexpected_build(*_args, **_kwargs):
        pytest.fail("search must not start an index build")

    monkeypatch.setattr(search, "build_or_update_embed_index", unexpected_build)
    _prepare_embed_pipeline(monkeypatch)
    statuses = []
    result = search.run_retrieval_pipeline(
        "休暇の申請期限",
        model="m",
        emit_status=statuses.append,
    )
    assert result.route == "keyword"
    assert result.index_state == "missing"
    assert any("キーワード検索のみ" in text for text in statuses)


def test_retrieval_pipeline_marks_stale_index_as_keyword_only(monkeypatch):
    """stale indexは旧vectorを使わずkeyword routeへ移る。"""
    used_models = []

    def record_single(query, keywords, embed_model, embed_cache=None, source_chunks=None):
        used_models.append(embed_model)
        return []

    _prepare_embed_pipeline(monkeypatch, record_single)
    monkeypatch.setattr(
        search,
        "get_embed_index_status",
        lambda embed_model, chunks=None, **_kwargs: {"state": "stale", "total": 0},
    )
    statuses = []

    search.run_retrieval_pipeline("休暇の申請期限", model="m", emit_status=statuses.append)

    assert used_models and all(model is None for model in used_models)
    assert any("キーワード検索のみ" in t for t in statuses)


def test_retrieval_pipeline_cancel_still_stops_search(monkeypatch):
    """明示cancelはkeyword fallbackとして握りつぶさない。"""
    used_models = []

    def record_single(query, keywords, embed_model, embed_cache=None, source_chunks=None):
        used_models.append(embed_model)
        return []

    _prepare_embed_pipeline(monkeypatch, record_single)
    monkeypatch.setattr(
        search,
        "get_embed_index_status",
        lambda embed_model, chunks=None, **_kwargs: {"state": "building", "total": 0},
    )

    def cancel_check():
        raise _FakeWebCancelled("cancelled")

    with pytest.raises(_FakeWebCancelled):
        search.run_retrieval_pipeline("休暇の申請期限", model="m", cancel_check=cancel_check)
    assert not used_models
