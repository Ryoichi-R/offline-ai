"""load_embed_cache memo・多クエリ一括採点・keep_alive の回帰テスト（Phase 2）。

対象:
- get_embed_index_status(cache=...) が load_embed_cache を呼ばない
- run_retrieval_pipeline 1回で実ファイル読み込みが1回以下
- load_embed_cache の memo（mtime/size キー、invalidate、writer経路のバイパス）
- OFFLINE_AI_EMBED_CACHE_MEMO=0 での無効化
- embedding_search_multi が embedding_search と round(sim, 4) の桁で一致する
- embedding_search（単一クエリ、既存シグネチャ）が引き続き動作する
- クエリ埋め込みが _get_embeddings へ1回のバッチで渡される
- keep_alive が chat / embed の両ボディに入る
"""

import json
import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_embed_cache_memo():
    search._invalidate_embed_cache_memo()
    yield
    search._invalidate_embed_cache_memo()


def _write_cache(path, *, embed_model="m", generation="g1", entries=None):
    payload = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": embed_model,
        "generation": generation,
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "entries": entries or {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


# --- load_embed_cache memo ---------------------------------------------------


def test_load_embed_cache_memo_skips_io_on_second_call(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    _write_cache(path)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)

    # 逐次読込では read_text を使わないため、disk 読込の入口そのものを数える。
    calls = {"n": 0}
    real_load_from_disk = search._load_embed_cache_from_disk

    def counting_load_from_disk():
        calls["n"] += 1
        return real_load_from_disk()

    monkeypatch.setattr(search, "_load_embed_cache_from_disk", counting_load_from_disk)

    first = search.load_embed_cache()
    second = search.load_embed_cache()

    assert calls["n"] == 1
    assert first is second  # memoヒット時は共有参照


def test_load_embed_cache_memo_invalidates_on_mtime_change(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    _write_cache(path, generation="g1")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)

    first = search.load_embed_cache()
    assert first["generation"] == "g1"

    # mtimeを確実に進めてから内容を書き換える。
    _write_cache(path, generation="g2")
    new_mtime = os.stat(path).st_mtime_ns + 10_000_000
    os.utime(path, ns=(new_mtime, new_mtime))

    second = search.load_embed_cache()
    assert second["generation"] == "g2"


def test_load_embed_cache_memo_invalidates_on_size_change(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    _write_cache(path, generation="g1", entries={})
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    stat1 = os.stat(path)

    first = search.load_embed_cache()
    assert first["entries"] == {}

    entries = {"a.md#0001": {"chunk_id": "a.md#0001", "embedding": [1.0, 0.0]}}
    _write_cache(path, generation="g1", entries=entries)
    stat2 = os.stat(path)
    # サイズが変わっていることを前提とする（内容量が増えている）。
    assert stat2.st_size != stat1.st_size
    # mtimeが変わらない可能性があるファイルシステムに備え、明示的に進める。
    new_mtime = stat2.st_mtime_ns + 10_000_000
    os.utime(path, ns=(new_mtime, new_mtime))

    second = search.load_embed_cache()
    assert "a.md#0001" in second["entries"]


def test_load_embed_cache_memo_invalidated_by_path_switch(tmp_path, monkeypatch):
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    _write_cache(path_a, generation="from-a")
    _write_cache(path_b, generation="from-b")

    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path_a)
    first = search.load_embed_cache()
    assert first["generation"] == "from-a"

    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path_b)
    second = search.load_embed_cache()
    assert second["generation"] == "from-b"


def test_save_embed_cache_discards_memo(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    _write_cache(path, generation="g1")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)

    search.load_embed_cache()  # memoへ載せる
    search.save_embed_cache({**_write_cache_dict("g2")})

    reloaded = search.load_embed_cache()
    assert reloaded["generation"] == "g2"


def _write_cache_dict(generation):
    return {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "generation": generation,
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "entries": {},
    }


def test_writer_paths_bypass_memo_and_see_fresh_content(tmp_path, monkeypatch):
    """build_or_update_embed_index 内の writer 経路は memo を経由しない。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("hello world", encoding="utf-8")
    cache_path = tmp_path / "embed_cache.json"
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [0.1, 0.2])

    # 1回目のbuildの後、read専用memoに旧内容を故意に載せておく。
    search.build_or_update_embed_index("m1")
    stale = search.load_embed_cache()
    assert stale["embed_model"] == "m1"

    # writer経路（モデル変更で再構築）はmemoを無視して最新を読み書きする。
    result = search.build_or_update_embed_index("m2")
    assert result["embed_model"] == "m2"
    # save成功でmemoが破棄されているため、read専用呼び出しも最新を見る。
    assert search.load_embed_cache()["embed_model"] == "m2"


def test_embed_cache_memo_can_be_disabled(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    _write_cache(path, generation="g1")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    monkeypatch.setattr(search, "EMBED_CACHE_MEMO_ENABLED", False)

    # 逐次読込では read_text を使わないため、disk 読込の入口そのものを数える。
    calls = {"n": 0}
    real_load_from_disk = search._load_embed_cache_from_disk

    def counting_load_from_disk():
        calls["n"] += 1
        return real_load_from_disk()

    monkeypatch.setattr(search, "_load_embed_cache_from_disk", counting_load_from_disk)

    search.load_embed_cache()
    search.load_embed_cache()

    assert calls["n"] == 2  # memo無効時は毎回I/Oする


# --- get_embed_index_status(cache=...) ---------------------------------------


def test_get_embed_index_status_with_cache_does_not_call_load_embed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", tmp_path / "status.json")
    chunks = [{"chunk_id": "a.md#0001"}]
    identity = {"name": "m:latest", "digest": "fixture-digest"}
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "generation": search.compute_embed_generation("m", chunks),
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "entries": {
            "a.md#0001": {
                "chunk_id": "a.md#0001",
                "file_sha256": "f",
                "text_sha256": "t",
                "text": "本文",
                "embedding": [1.0, 0.0],
            }
        },
        "compatibility": search._new_embed_compatibility("m", identity, 2),
        "files": search._source_manifest_from_chunks(chunks),
    }
    monkeypatch.setattr(search, "get_embed_model_identity", lambda *_args, **_kwargs: identity)

    def fail_load(*_args, **_kwargs):
        pytest.fail("cache指定時は load_embed_cache を呼んではならない")

    monkeypatch.setattr(search, "load_embed_cache", fail_load)

    status = search.get_embed_index_status("m", chunks, cache=cache)
    assert status["state"] == "ready"


def test_get_embed_index_status_validate_entries_false_skips_full_scan(tmp_path, monkeypatch):
    """validate_entries=False は key集合一致のみで ready 判定し、不正entryを検出しない。"""
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", tmp_path / "status.json")
    chunks = [{"chunk_id": "a.md#0001"}]
    identity = {"name": "m:latest", "digest": "fixture-digest"}
    generation = search.compute_embed_generation("m", chunks)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "generation": generation,
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        # embeddingが空 = 本来は不正entry
        "entries": {"a.md#0001": {"chunk_id": "a.md#0001", "embedding": []}},
        "compatibility": search._new_embed_compatibility("m", identity, None),
        "files": search._source_manifest_from_chunks(chunks),
    }
    monkeypatch.setattr(search, "get_embed_model_identity", lambda *_args, **_kwargs: identity)

    lenient = search.get_embed_index_status("m", chunks, cache=cache, validate_entries=False)
    strict = search.get_embed_index_status("m", chunks, cache=cache, validate_entries=True)

    assert lenient["state"] == "ready"
    assert strict["state"] != "ready"


def test_run_retrieval_pipeline_loads_cache_at_most_once(monkeypatch):
    """1-a: run_retrieval_pipeline は1回のretrievalでload_embed_cacheを高々1回呼ぶ。"""
    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)
    monkeypatch.setattr(
        search,
        "create_search_plan",
        lambda query, model: {"keywords": ["k"], "search_queries": [query]},
    )
    monkeypatch.setattr(search, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(
        search, "_is_model_available", lambda model, timeout=2.0: search.ModelStatus.AVAILABLE
    )
    monkeypatch.setattr(search, "_chunk_retrieval_enabled", lambda: True)
    chunks = [{"chunk_id": "a.md#0001"}]
    monkeypatch.setattr(search, "build_source_chunks", lambda source_root: chunks)
    generation = search.compute_embed_generation("bge-m3", chunks)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "bge-m3",
        "generation": generation,
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "entries": {},
    }
    calls = {"n": 0}

    def counting_load():
        calls["n"] += 1
        return cache

    monkeypatch.setattr(search, "load_embed_cache", counting_load)
    monkeypatch.setattr(search, "keyword_search_chunks", lambda *a, **k: [])

    search.run_retrieval_pipeline("q", model="m")

    assert calls["n"] <= 1


# --- embedding_search_multi 一致性 -------------------------------------------


def _dim_mismatch_and_normal_cache():
    return {
        "entries": {
            "good.md#0001": {
                "chunk_id": "good.md#0001",
                "path": "good.md",
                "text": "一致エントリ",
                "embedding": [1.0, 0.5, 0.2],
            },
            "good.md#0002": {
                "chunk_id": "good.md#0002",
                "path": "good.md",
                "text": "別エントリ",
                "embedding": [0.2, 0.9, 0.1],
            },
            "bad.md#0001": {
                "chunk_id": "bad.md#0001",
                "path": "bad.md",
                "text": "次元不一致",
                "embedding": [1.0, 0.0],  # 2次元（クエリは3次元）
            },
            "borderline.md#0001": {
                "chunk_id": "borderline.md#0001",
                "path": "borderline.md",
                "text": "閾値境界",
                "embedding": [0.3, 0.3, 0.3],
            },
        }
    }


def test_embedding_search_multi_matches_single_query_scores(monkeypatch):
    cache = _dim_mismatch_and_normal_cache()
    query_vecs = {
        "query one": [1.0, 0.4, 0.3],
        "query two": [0.0, 1.0, 0.0],
    }

    def fake_get_embeddings(texts, embed_model, *, timeout=None):
        return [query_vecs[t] for t in texts]

    def fake_get_embedding(text, embed_model):
        return query_vecs[text]

    monkeypatch.setattr(search, "_get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(search, "_get_embedding", fake_get_embedding)

    index = search._build_embed_index(cache)
    multi_result = search.embedding_search_multi(list(query_vecs), "m", index, top_k=8)

    for query in query_vecs:
        single_result = search.embedding_search(query, "m", cache, top_k=8)
        multi_scores = {m["chunk_id"]: m["score"] for m in multi_result[query]}
        single_scores = {m["chunk_id"]: m["score"] for m in single_result}
        assert multi_scores == single_scores
        # 次元不一致entryはどちらの経路でも除外される。
        assert "bad.md#0001" not in multi_scores
        assert "bad.md#0001" not in single_scores


def test_embedding_search_multi_respects_top_k_truncation(monkeypatch):
    cache = {
        "entries": {
            f"c{i}.md#0001": {
                "chunk_id": f"c{i}.md#0001",
                "path": f"c{i}.md",
                "text": "t",
                "embedding": [1.0, float(i) * 0.01],
            }
            for i in range(10)
        }
    }
    monkeypatch.setattr(
        search, "_get_embeddings", lambda texts, embed_model, **k: [[1.0, 0.0] for _ in texts]
    )
    index = search._build_embed_index(cache)
    result = search.embedding_search_multi(["q"], "m", index, top_k=3)
    assert len(result["q"]) == 3


def test_embedding_search_single_query_signature_still_works(monkeypatch):
    """embedding_search（単一クエリ、既存シグネチャ）が引き続き動作する。"""
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0, 0.0])
    cache = {
        "entries": {
            "a.md#0001": {
                "chunk_id": "a.md#0001",
                "path": "a.md",
                "text": "t",
                "embedding": [1.0, 0.0, 0.0],
            }
        }
    }
    result = search.embedding_search("q", "m", cache, top_k=5)
    assert result[0]["chunk_id"] == "a.md#0001"


def test_query_embeddings_are_fetched_in_a_single_batch(monkeypatch):
    """クエリ埋め込みが _get_embeddings へ1回のバッチで渡される（呼び出し回数と順序）。"""
    cache = _dim_mismatch_and_normal_cache()
    index = search._build_embed_index(cache)
    calls = []

    def fake_get_embeddings(texts, embed_model, *, timeout=None):
        calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(search, "_get_embeddings", fake_get_embeddings)

    search.embedding_search_multi(["q1", "q2", "q3"], "m", index, top_k=8)

    assert calls == [["q1", "q2", "q3"]]


# --- keep_alive ---------------------------------------------------------------


def test_chat_payload_keep_alive_default_and_zero(monkeypatch):
    monkeypatch.setattr(search, "OLLAMA_KEEP_ALIVE", "30m")
    body = search.build_chat_payload("m", "sys", "user")
    assert body["keep_alive"] == "30m"

    monkeypatch.setattr(search, "OLLAMA_KEEP_ALIVE", "0")
    body_zero = search.build_chat_payload("m", "sys", "user")
    assert body_zero["keep_alive"] == "0"


def test_embed_request_includes_keep_alive(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"embeddings": [[1.0, 0.0]]}).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(search, "OLLAMA_KEEP_ALIVE", "45m")

    search._get_embeddings(["text"], "bge-m3")

    assert captured["body"]["keep_alive"] == "45m"
