import search


def test_search_mode_uses_deterministic_keyword_path_without_chat(monkeypatch, tmp_path):
    source = tmp_path / "rules.md"
    source.write_text("# 申請\n申請には本人確認が必要です。", encoding="utf-8")
    calls = []

    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "_chunk_retrieval_enabled", lambda: True)
    monkeypatch.setattr(search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(search, "create_search_plan", lambda *_args: calls.append("plan") or {})
    monkeypatch.setattr(
        search, "extract_keywords_via_llm", lambda *_args: calls.append("keywords") or ""
    )
    monkeypatch.setattr(
        search, "build_user_prompt", lambda *_args, **_kwargs: calls.append("prompt") or ""
    )
    monkeypatch.setattr(search, "RERANK_CONFIG", search.RerankConfig("off", "", 8, 8, 1, False))

    result = search.run_retrieval_pipeline("申請 本人確認", model="unused", mode="search")

    assert result.user_prompt == ""
    assert result.plan["query_type"] == "search_only"
    assert len(result.attempts) == 1
    assert any("本人確認" in item["snippet"] for item in result.matches)
    assert calls == []


def test_search_mode_unknown_embedding_keeps_local_index_state_and_falls_back(
    monkeypatch, tmp_path
):
    source = tmp_path / "rules.md"
    source.write_text("本人確認の規則", encoding="utf-8")
    status_calls = []

    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "_chunk_retrieval_enabled", lambda: True)
    monkeypatch.setattr(search, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(search, "load_embed_cache", lambda: {"entries": {}})
    monkeypatch.setattr(
        search,
        "get_embed_index_status",
        lambda *args, **kwargs: status_calls.append(True) or {"state": "ready"},
    )
    monkeypatch.setattr(search, "_is_model_available", lambda *_args: search.ModelStatus.UNKNOWN)
    monkeypatch.setattr(search, "RERANK_CONFIG", search.RerankConfig("off", "", 8, 8, 1, False))

    result = search.run_retrieval_pipeline("本人確認", model="unused", mode="search")

    assert status_calls == [True]
    assert result.route == "keyword"
    assert result.embedding_model is None
    assert result.index_state == "ready"
    assert "確認できない" in result.route_reason
    assert result.user_prompt == ""


def test_search_mode_embedding_failure_does_not_retry(monkeypatch, tmp_path):
    source = tmp_path / "rules.md"
    source.write_text("本人確認の規則", encoding="utf-8")
    calls = {"embed": 0, "build": 0}
    cache = {
        "entries": {
            "rules.md#0001": {"path": "rules.md", "text": "本人確認の規則", "embedding": [1.0]}
        }
    }

    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "_chunk_retrieval_enabled", lambda: True)
    monkeypatch.setattr(search, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(search, "load_embed_cache", lambda: cache)
    monkeypatch.setattr(
        search, "get_embed_index_status", lambda *args, **kwargs: {"state": "ready"}
    )
    monkeypatch.setattr(search, "_is_model_available", lambda *_args: search.ModelStatus.AVAILABLE)

    def fail_embedding(*_args, **_kwargs):
        calls["embed"] += 1
        raise search.EmbeddingBatchError("EMBED_TRANSPORT_ERROR", "failed")

    monkeypatch.setattr(search, "embedding_search_multi_with_context", fail_embedding)
    monkeypatch.setattr(
        search,
        "build_or_update_embed_index",
        lambda *_args: calls.__setitem__("build", calls["build"] + 1),
    )
    monkeypatch.setattr(search, "RERANK_CONFIG", search.RerankConfig("off", "", 8, 8, 1, False))

    result = search.run_retrieval_pipeline("本人確認", model="unused", mode="search")

    assert calls == {"embed": 1, "build": 0}
    assert result.route == "keyword"
    assert "失敗" in result.route_reason
    assert result.embedding_model is None
