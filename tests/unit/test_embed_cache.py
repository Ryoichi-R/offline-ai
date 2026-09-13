"""Embedding キャッシュのモデル変更耐性・次元不整合ガードのテスト。

対象修正:
- load_embed_cache: version 1（embed_model 無し）の後方互換と embed_model 補完
- _cosine_similarity: 次元不一致時に 0.0 を返す二重防御
- embedding_search: 次元不一致・空/None ベクトルのスキップ
- build_or_update_embed_index: embed_model 変更時の自動再構築
- save_embed_cache: アトミック書き込み
"""

import json
import os

import pytest

# OLLAMA_HOST が remote 値で設定された環境でも `import search` が
# モジュールロード時（search.py トップレベルの検証）に失敗しないようにする。
os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402
import document_schema  # noqa: E402


# --- _cosine_similarity -----------------------------------------------------


def test_cosine_similarity_dimension_mismatch_returns_zero():
    # 768次元クエリ vs 1024次元キャッシュのような不一致を 0.0 で弾く
    assert search._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert abs(search._cosine_similarity(v, v) - 1.0) < 1e-9


# --- load_embed_cache -------------------------------------------------------


def test_load_embed_cache_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", tmp_path / "absent.json")
    cache = search.load_embed_cache()
    assert cache["version"] == search.EMBED_CACHE_VERSION
    assert cache["embed_model"] is None
    assert cache["entries"] == {}


def test_load_embed_cache_v1_migration(tmp_path, monkeypatch):
    # version 1（embed_model キー無し）の旧キャッシュ
    path = tmp_path / "embed_cache.json"
    path.write_text(
        json.dumps({"version": 1, "entries": {"a.md": {"sha256": "x", "embedding": [0.1]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    cache = search.load_embed_cache()
    # embed_model が None 補完され、エントリは保持される（後段でversion不一致再構築）
    assert cache["embed_model"] is None
    assert "a.md" in cache["entries"]


def test_load_embed_cache_corrupt_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    cache = search.load_embed_cache()
    assert cache["entries"] == {}


# --- 逐次読込（peak working set 抑制） --------------------------------------

# BMP 外の文字（𠮷）を含めることで、全文 str が UCS-4 化する条件を再現する。
_STREAM_SAMPLE = {
    "version": 4,
    "embed_model": "bge-m3:latest",
    "generation": "g" * 64,
    "chunking": {"max_chars": 2000, "overlap_chars": 200},
    "entries": {
        "README.txt#0001": {
            "path": "README.txt",
            "text": '日本語と改行\nと引用符"と波括弧{}と𠮷',
            "start_line": 1,
            "embedding": [0.125, -0.5, 1.0],
        },
        "docs/a.md#0002": {
            "path": "docs/a.md",
            "text": "",
            "start_line": 2000,
            "embedding": [],
            "nested": {"deep": {"deeper": [1, 2, {"x": None}]}},
        },
        "docs/b.md#0003": {"path": "docs/b.md", "embedding": [0.0] * 32},
    },
}


@pytest.mark.parametrize("indent", [2, None])
def test_load_embed_cache_streaming_matches_full_parse(tmp_path, indent):
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps(_STREAM_SAMPLE, ensure_ascii=False, indent=indent), encoding="utf-8")

    streamed = search._load_embed_cache_streaming(path)
    full = json.loads(path.read_text(encoding="utf-8"))

    assert streamed == full
    # トップレベル・entries とも並び順まで一致させる（既存 cache 形式を変えない）。
    assert list(streamed.keys()) == list(full.keys())
    assert list(streamed["entries"].keys()) == list(full["entries"].keys())


def test_load_embed_cache_streaming_refills_buffer_mid_value(tmp_path, monkeypatch):
    """chunk 境界が key・数値・文字列の途中に来ても値を取り違えない。"""
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps(_STREAM_SAMPLE, ensure_ascii=False, indent=2), encoding="utf-8")
    full = json.loads(path.read_text(encoding="utf-8"))

    for chunk_bytes in (1, 2, 3, 7, 64):
        monkeypatch.setattr(search, "_EMBED_CACHE_STREAM_CHUNK_BYTES", chunk_bytes)
        assert search._load_embed_cache_streaming(path) == full


def test_load_embed_cache_streaming_handles_empty_entries(tmp_path):
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps({"version": 4, "entries": {}}), encoding="utf-8")
    assert search._load_embed_cache_streaming(path) == {"version": 4, "entries": {}}


@pytest.mark.parametrize(
    "raw",
    [
        '{"version": 4, "entries": {"a": {"embedding": [0.1]',  # 途中で切れた entries
        '{"version": 4, "entries": ',  # 値が無い
        '["not", "an", "object"]',  # object でない
        "{ this is not valid json",  # 破損
        '{"version": 4, "entries": {"a": {}} extra',  # 末尾の余剰
    ],
)
def test_load_embed_cache_streaming_returns_none_for_broken_input(tmp_path, raw):
    path = tmp_path / "embed_cache.json"
    path.write_text(raw, encoding="utf-8")
    assert search._load_embed_cache_streaming(path) is None


def test_load_embed_cache_uses_streaming_without_reading_whole_document(tmp_path, monkeypatch):
    """成功経路では全文 str を作らない（read_text へ落ちない）。"""
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps(_STREAM_SAMPLE, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    search._invalidate_embed_cache_memo()

    def fail_read_text(*_args, **_kwargs):
        raise AssertionError("全文 read_text は逐次読込の成功時に呼ばれてはならない")

    monkeypatch.setattr(type(path), "read_text", fail_read_text)
    cache = search.load_embed_cache()
    assert list(cache["entries"].keys()) == list(_STREAM_SAMPLE["entries"].keys())


def test_load_embed_cache_falls_back_to_full_parse_when_streaming_fails(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps(_STREAM_SAMPLE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    search._invalidate_embed_cache_memo()
    monkeypatch.setattr(search, "_load_embed_cache_streaming", lambda _path: None)

    cache = search.load_embed_cache()
    assert list(cache["entries"].keys()) == list(_STREAM_SAMPLE["entries"].keys())
    assert cache["embed_model"] == "bge-m3:latest"


# --- save_embed_cache (atomic) ---------------------------------------------


def test_save_embed_cache_atomic_no_tmp_left(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)
    payload = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {"a.md#0001": {"sha256": "x", "embedding": [0.1]}},
    }
    search.save_embed_cache(payload)
    # 本体が書かれ、一時ファイルは残らない
    assert path.exists()
    assert not (tmp_path / "embed_cache.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["embed_model"] == "m"


def test_save_embed_cache_replace_failure_keeps_existing_cache(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.json"
    path.write_text(json.dumps({"version": 3, "entries": {"old": {}}}), encoding="utf-8")
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", path)

    def fail_replace(_source, _destination):
        raise PermissionError("simulated Windows reader handle")

    monkeypatch.setattr(search.os, "replace", fail_replace)
    assert search.save_embed_cache({"version": 3, "entries": {"new": {}}}) is False
    assert json.loads(path.read_text(encoding="utf-8"))["entries"] == {"old": {}}
    assert not (tmp_path / "embed_cache.json.tmp").exists()


# --- embedding_search -------------------------------------------------------


def test_embedding_search_skips_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0, 0.0])
    monkeypatch.setattr(search, "EMBED_SIM_THRESHOLD", 0.0)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "good.md#0001": {
                "path": "good.md",
                "chunk_id": "good.md#0001",
                "text": "本文",
                "embedding": [1.0, 0.0, 0.0],
            },
            "bad.md#0001": {
                "path": "bad.md",
                "chunk_id": "bad.md#0001",
                "text": "本文",
                "embedding": [1.0, 0.0],
            },
        },
    }
    results = search.embedding_search("q", "m", cache)
    paths = [r["path"] for r in results]
    assert "good.md" in paths
    assert "bad.md" not in paths


def test_embedding_search_returns_empty_when_query_vec_none(monkeypatch):
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: None)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "a.md#0001": {
                "path": "a.md",
                "chunk_id": "a.md#0001",
                "text": "本文",
                "embedding": [1.0, 0.0],
            }
        },
    }
    assert search.embedding_search("q", "m", cache) == []


def test_embedding_search_skips_empty_embedding(monkeypatch):
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0, 0.0])
    monkeypatch.setattr(search, "EMBED_SIM_THRESHOLD", 0.0)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "empty.md#0001": {
                "path": "empty.md",
                "chunk_id": "empty.md#0001",
                "text": "本文",
                "embedding": [],
            },
            "good.md#0001": {
                "path": "good.md",
                "chunk_id": "good.md#0001",
                "text": "本文",
                "embedding": [1.0, 0.0, 0.0],
            },
        },
    }
    paths = [r["path"] for r in search.embedding_search("q", "m", cache)]
    assert "empty.md" not in paths
    assert "good.md" in paths


def test_embedding_search_uses_current_sidecar_not_cached_metadata(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    source = src / "procedure.md"
    source.write_text("# 申請手順\n本文", encoding="utf-8")
    metadata = document_schema.build_document_metadata(
        source_path=src / "procedure.pdf",
        markdown_path=source,
        markdown_text=source.read_text(encoding="utf-8"),
        parser="current-parser",
    )
    metadata["pages"][0]["blocks"][0]["page"] = 5
    metadata["pages"][0]["blocks"][0]["layoutType"] = "table"
    document_schema.write_metadata_sidecar(source, metadata)
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0, 0.0])
    monkeypatch.setattr(search, "EMBED_SIM_THRESHOLD", 0.0)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "procedure.md#0001": {
                "chunk_id": "procedure.md#0001",
                "heading": "申請手順",
                "start_line": 1,
                "end_line": 2,
                "text": "申請手順",
                "embedding": [1.0, 0.0, 0.0],
                "parser": "stale-parser",
                "page": 99,
            }
        },
    }

    result = search.embedding_search("申請", "m", cache)[0]

    assert result["parser"] == "current-parser"
    assert result["page"] == 5
    assert result["layout_type"] == "table"


def test_embedding_search_memoizes_sidecar_load_per_path(monkeypatch):
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0, 0.0])
    monkeypatch.setattr(search, "EMBED_SIM_THRESHOLD", 0.0)
    calls = {"n": 0}

    def fake_load_metadata_sidecar(path):
        calls["n"] += 1
        return None

    monkeypatch.setattr(search, "load_metadata_sidecar", fake_load_metadata_sidecar)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "same.md#0001": {
                "path": "same.md",
                "chunk_id": "same.md#0001",
                "start_line": 1,
                "end_line": 2,
                "text": "本文1",
                "embedding": [1.0, 0.0, 0.0],
            },
            "same.md#0002": {
                "path": "same.md",
                "chunk_id": "same.md#0002",
                "start_line": 3,
                "end_line": 4,
                "text": "本文2",
                "embedding": [1.0, 0.0, 0.0],
            },
        },
    }

    assert len(search.embedding_search("q", "m", cache, top_k=2)) == 2
    assert calls["n"] == 1


def test_embedding_search_zero_vectors_excluded(monkeypatch):
    # 全0ベクトル同士は len 一致だが cosine=0.0 → 閾値(>0)未満で除外される
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [0.0, 0.0, 0.0])
    monkeypatch.setattr(search, "EMBED_SIM_THRESHOLD", 0.3)
    cache = {
        "version": search.EMBED_CACHE_VERSION,
        "embed_model": "m",
        "entries": {
            "zero.md#0001": {
                "path": "zero.md",
                "chunk_id": "zero.md#0001",
                "text": "本文",
                "embedding": [0.0, 0.0, 0.0],
            }
        },
    }
    assert search.embedding_search("q", "m", cache) == []


# --- build_or_update_embed_index -------------------------------------------


def test_build_index_rebuilds_on_model_change(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("hello world", encoding="utf-8")

    cache_path = tmp_path / "embed_cache.json"
    # 旧モデルの古い次元のキャッシュ
    cache_path.write_text(
        json.dumps(
            {
                "version": 2,
                "embed_model": "old-model",
                "entries": {"a.md": {"sha256": "stale", "embedding": [9.9]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [0.1, 0.2, 0.3])

    cache = search.build_or_update_embed_index("new-model")

    # モデル名が更新され、旧次元エントリは破棄→新モデルで再計算される
    assert cache["embed_model"] == "new-model"
    assert cache["entries"]["a.md#0001"]["embedding"] == [0.1, 0.2, 0.3]
    assert cache["entries"]["a.md#0001"]["text"] == "hello world"


def test_build_index_rebuilds_from_v1_cache(tmp_path, monkeypatch):
    # version 1 旧キャッシュ（embed_model 無し）→ モデル指定で再構築される統合経路
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("hello world", encoding="utf-8")

    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {"a.md": {"sha256": "stale", "embedding": [9.9]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [0.1, 0.2, 0.3])

    cache = search.build_or_update_embed_index("bge-m3")

    assert cache["version"] == search.EMBED_CACHE_VERSION
    assert cache["embed_model"] == "bge-m3"
    assert cache["entries"]["a.md#0001"]["embedding"] == [0.1, 0.2, 0.3]


def test_build_index_keeps_cache_on_same_model(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("hello world", encoding="utf-8")

    cache_path = tmp_path / "embed_cache.json"
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)

    calls = {"n": 0}

    def fake_embed(text, model):
        calls["n"] += 1
        return [0.5, 0.5]

    monkeypatch.setattr(search, "_get_embedding", fake_embed)

    # 1回目: 新規計算
    search.build_or_update_embed_index("same-model")
    first = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]["a.md#0001"]["embedding"]
    # 2回目: 同一モデル・同一内容なら SHA-256 一致で再計算スキップ
    search.build_or_update_embed_index("same-model")
    second = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]["a.md#0001"]["embedding"]

    assert calls["n"] == 1  # 呼び出し回数で再計算抑止を担保
    assert first == second  # 内容面でも再計算スキップを担保


def test_build_index_does_not_store_sidecar_metadata(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    source = src / "a.md"
    source.write_text("# 見出し\n本文", encoding="utf-8")
    metadata = document_schema.build_document_metadata(
        source_path=src / "a.pdf",
        markdown_path=source,
        markdown_text=source.read_text(encoding="utf-8"),
        parser="pdftotext",
    )
    document_schema.write_metadata_sidecar(source, metadata)
    cache_path = tmp_path / "embed_cache.json"
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [0.1, 0.2, 0.3])

    cache = search.build_or_update_embed_index("m")
    entry = cache["entries"]["a.md#0001"]

    assert "embedding" in entry
    assert "parser" not in entry
    assert "metadata" not in entry
    assert "page" not in entry


def test_build_index_removes_stale_entry_when_changed_chunk_embedding_fails(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    source = src / "a.md"
    source.write_text("new content", encoding="utf-8")
    cache_path = tmp_path / "embed_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": search.EMBED_CACHE_VERSION,
                "embed_model": "m",
                "chunking": {
                    "max_chars": search.CHUNK_MAX_CHARS,
                    "overlap_chars": search.CHUNK_OVERLAP_CHARS,
                },
                "entries": {
                    "a.md#0001": {
                        "path": "a.md",
                        "chunk_id": "a.md#0001",
                        "file_sha256": "old-file",
                        "text_sha256": "old-text",
                        "text": "old content",
                        "embedding": [0.9, 0.1],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", src)
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: None)

    with pytest.raises(search.EmbedBuildError, match="failed"):
        search.build_or_update_embed_index("m")

    # A failed generation must not replace the existing cache.
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert persisted["entries"]["a.md#0001"]["file_sha256"] == "old-file"


def _embed_test_chunks(count: int) -> list[dict]:
    return [
        {
            "path": "fixture.md",
            "chunk_id": f"fixture.md#{index:04d}",
            "text": f"text {index}",
            "file_sha256": f"file-{index}",
            "text_sha256": f"text-{index}",
            "modifiedAt": "",
        }
        for index in range(1, count + 1)
    ]


def test_build_index_emits_progress_and_resumes_checkpoint(tmp_path, monkeypatch):
    cache_path = tmp_path / "embed_cache.json"
    checkpoint_path = tmp_path / "embed_cache.checkpoints"
    lock_path = tmp_path / "embed_cache.lock"
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(search, "EMBED_LOCK_PATH", lock_path)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_INTERVAL", 2)
    monkeypatch.setattr(search, "EMBED_PROGRESS_INTERVAL", 1)
    chunks = _embed_test_chunks(3)
    calls = {"n": 0}
    events = []

    def first_embed(text, model):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("stop after two successful vectors")
        return [float(calls["n"]), 0.0]

    monkeypatch.setattr(search, "_get_embedding", first_embed)
    with pytest.raises(RuntimeError, match="stop"):
        search.build_or_update_embed_index("bge-m3", chunks, emit_progress=events.append)

    assert (checkpoint_path / "batch-00000001.json").exists()
    assert events[0]["phase"] == "start"
    assert events[-1]["phase"] == "interrupted"
    assert all(
        event["processed"] == event["generated"] + event["reused"] + event["failed"]
        for event in events
    )
    state, checkpoint_entries = search._load_checkpoint("bge-m3:latest")
    assert state["generation"]
    assert len(checkpoint_entries) == 2

    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [9.0, 0.0])
    resumed_events = []
    result = search.build_or_update_embed_index(
        "bge-m3:latest", chunks, emit_progress=resumed_events.append
    )
    assert len(result["entries"]) == 3
    assert resumed_events[0]["phase"] == "resumed"
    assert resumed_events[0]["reused"] == 2
    assert resumed_events[-1]["phase"] == "completed"
    assert not checkpoint_path.exists() or not list(checkpoint_path.glob("batch-*.json"))


def test_checkpoint_generation_header_mismatch_starts_new_generation(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "embed_cache.checkpoints"
    checkpoint_path.mkdir()
    old_state = {
        "schema_version": search.EMBED_CHECKPOINT_SCHEMA_VERSION,
        "cache_version": search.EMBED_CACHE_VERSION,
        "embed_model": "old-model:latest",
        "chunking": {
            "max_chars": search.CHUNK_MAX_CHARS,
            "overlap_chars": search.CHUNK_OVERLAP_CHARS,
        },
        "generation": "old-generation",
        "next_batch": 2,
        "created_at": "old",
        "updated_at": "old",
    }
    (checkpoint_path / "state.json").write_text(json.dumps(old_state), encoding="utf-8")
    (checkpoint_path / "batch-00000001.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_PATH", checkpoint_path)
    state, entries = search._load_checkpoint("new-model")
    assert entries == {}
    assert state["generation"] != "old-generation"
    assert not (checkpoint_path / "batch-00000001.json").exists()


def test_keyword_search_uses_supplied_chunks_without_rescan(monkeypatch):
    supplied = [
        {
            "path": "rules.md",
            "chunk_id": "rules.md#0001",
            "heading": "休暇",
            "text": "年次休暇の申請期限は3日前",
            "modifiedAt": "",
        }
    ]
    monkeypatch.setattr(
        search,
        "build_source_chunks",
        lambda source_root: (_ for _ in ()).throw(AssertionError("unexpected rescan")),
    )

    results = search.keyword_search_chunks("申請期限", top_k=2, chunks=supplied)

    assert results[0]["chunk_id"] == "rules.md#0001"


# --- checkpoint / progress の境界と中断・失敗時の契約 -----------------------


def _isolate_embed_runtime_paths(tmp_path, monkeypatch):
    """製品の cache / checkpoint / lock を tmp へ隔離する。"""
    cache_path = tmp_path / "embed_cache.json"
    checkpoint_path = tmp_path / "embed_cache.checkpoints"
    lock_path = tmp_path / "embed_cache.lock"
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", cache_path)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(search, "EMBED_LOCK_PATH", lock_path)
    # 経過時間による進捗emitを止め、件数境界だけを決定的に検証する。
    monkeypatch.setattr(search, "EMBED_PROGRESS_SECONDS", 3600)
    return cache_path, checkpoint_path, lock_path


def test_build_index_zero_chunks_completes_without_zero_division(tmp_path, monkeypatch):
    """total == 0 を0除算せず、即completeとして表示する。"""
    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        search,
        "_get_embedding",
        lambda text, model: pytest.fail("空chunkでEmbedding APIを呼んではならない"),
    )
    events = []

    cache = search.build_or_update_embed_index("bge-m3", [], emit_progress=events.append)

    assert cache["entries"] == {}
    assert [event["phase"] for event in events] == ["start", "completed"]
    assert "0 / 0 (100.0%)" in search._format_embed_progress(events[-1])


@pytest.mark.parametrize(
    "count, expected_progress, expected_checkpoints",
    [(1, 0, 1), (24, 0, 1), (25, 1, 1), (99, 3, 1), (100, 4, 1), (101, 4, 2)],
)
def test_progress_and_checkpoint_event_boundaries(
    tmp_path, monkeypatch, count, expected_progress, expected_checkpoints
):
    """25件ごとの進捗emitと100件ごとのcheckpointを境界値で固定する。"""
    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(search, "EMBED_PROGRESS_INTERVAL", 25)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_INTERVAL", 100)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0])
    events = []

    search.build_or_update_embed_index(
        "bge-m3", _embed_test_chunks(count), emit_progress=events.append
    )

    phases = [event["phase"] for event in events]
    assert phases[0] == "start"
    assert phases[-1] == "completed"
    assert phases.count("progress") == expected_progress
    assert phases.count("checkpoint") == expected_checkpoints
    assert events[-1]["processed"] == count
    assert events[-1]["generated"] == count


def test_progress_events_expose_counters_only(tmp_path, monkeypatch):
    """進捗へ本文・資料path・絶対pathを載せない。"""
    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(search, "EMBED_PROGRESS_INTERVAL", 1)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0])
    chunks = [
        {
            "path": "confidential-source.md",
            "chunk_id": "confidential-source.md#0001",
            "text": "SECRET-BODY-TEXT",
            "file_sha256": "f1",
            "text_sha256": "t1",
            "modifiedAt": "",
        }
    ]
    events = []

    search.build_or_update_embed_index("bge-m3", chunks, emit_progress=events.append)

    allowed = {
        "phase",
        "processed",
        "total",
        "generated",
        "reused",
        "failed",
        "checkpointed",
    }
    for event in events:
        assert set(event) == allowed
        assert event["processed"] == event["generated"] + event["reused"] + event["failed"]
    rendered = " ".join(search._format_embed_progress(event) for event in events)
    assert "SECRET-BODY-TEXT" not in rendered
    assert "confidential-source" not in rendered
    assert str(tmp_path) not in rendered


def test_resume_after_interrupt_regenerates_only_remaining_chunks(tmp_path, monkeypatch):
    """130/250で中断し、定期100件と中断時30件のflush後に120件だけ再生成する。"""
    _, checkpoint_path, _ = _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_INTERVAL", 100)
    chunks = _embed_test_chunks(250)
    first_calls = {"n": 0}

    def stop_at_130(text, model):
        first_calls["n"] += 1
        if first_calls["n"] > 130:
            raise RuntimeError("interrupted at 130")
        return [float(first_calls["n"]), 0.0]

    monkeypatch.setattr(search, "_get_embedding", stop_at_130)
    events = []
    with pytest.raises(RuntimeError, match="interrupted at 130"):
        search.build_or_update_embed_index("bge-m3", chunks, emit_progress=events.append)

    checkpointed = [event["checkpointed"] for event in events if event["phase"] == "checkpoint"]
    assert checkpointed == [100, 130]
    assert events[-1]["phase"] == "interrupted"
    _, saved = search._load_checkpoint("bge-m3")
    assert len(saved) == 130

    second_calls = {"n": 0}

    def count_regenerations(text, model):
        second_calls["n"] += 1
        return [0.5, 0.0]

    monkeypatch.setattr(search, "_get_embedding", count_regenerations)
    resumed = []

    cache = search.build_or_update_embed_index("bge-m3", chunks, emit_progress=resumed.append)

    assert second_calls["n"] == 120
    assert resumed[0]["phase"] == "resumed"
    assert resumed[0]["reused"] == 130
    assert len(cache["entries"]) == 250
    assert not list(checkpoint_path.glob("batch-*.json"))


def test_keyboard_interrupt_flushes_partial_batch_and_reraises(tmp_path, monkeypatch):
    """Ctrl+C でも端数batchを確定してから元の例外を再送出する。"""
    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_INTERVAL", 100)
    calls = {"n": 0}

    def interrupt_after_three(text, model):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return [1.0, 0.0]

    monkeypatch.setattr(search, "_get_embedding", interrupt_after_three)
    events = []

    with pytest.raises(KeyboardInterrupt):
        search.build_or_update_embed_index(
            "bge-m3", _embed_test_chunks(5), emit_progress=events.append
        )

    assert events[-1]["phase"] == "interrupted"
    assert events[-1]["checkpointed"] == 3
    _, saved = search._load_checkpoint("bge-m3")
    assert len(saved) == 3


def test_cancel_exception_flushes_and_propagates(tmp_path, monkeypatch):
    """Webキャンセル相当の例外を握りつぶさず、成功分だけ確定する。"""

    class _FakeCancelled(Exception):
        pass

    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_INTERVAL", 100)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0])
    checks = {"n": 0}

    def cancel_after_three_chunks():
        checks["n"] += 1
        # lock取得直後に1回、以降は1 chunkにつき前後2回呼ばれる。
        # 3件確定後・4件目の開始前にキャンセルする。
        if checks["n"] > 7:
            raise _FakeCancelled("cancelled")

    with pytest.raises(_FakeCancelled):
        search.build_or_update_embed_index(
            "bge-m3",
            _embed_test_chunks(10),
            cancel_check=cancel_after_three_chunks,
        )

    _, saved = search._load_checkpoint("bge-m3")
    assert len(saved) == 3


def test_final_cache_save_failure_keeps_checkpoint(tmp_path, monkeypatch):
    """最終cacheの保存に失敗したらcheckpointを削除しない。"""
    cache_path, checkpoint_path, _ = _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    real_write = search._atomic_write_json

    def fail_final_cache(path, payload):
        if path == cache_path:
            return False
        return real_write(path, payload)

    monkeypatch.setattr(search, "_atomic_write_json", fail_final_cache)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0])

    with pytest.raises(search.EmbedCachePersistenceError):
        search.build_or_update_embed_index("bge-m3", _embed_test_chunks(3))

    assert list(checkpoint_path.glob("batch-*.json"))
    assert not cache_path.exists()


def test_second_writer_reuses_existing_cache_without_new_embeddings(tmp_path, monkeypatch):
    """lock競合時は既存の有効cacheを読取専用で使い、同じbatchへ書かない。"""
    _, checkpoint_path, _ = _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    chunks = _embed_test_chunks(2)
    monkeypatch.setattr(search, "_get_embedding", lambda text, model: [1.0, 0.0])
    search.build_or_update_embed_index("bge-m3", chunks)

    monkeypatch.setattr(
        search,
        "_get_embedding",
        lambda text, model: pytest.fail("2本目のwriterはEmbedding APIを呼ばない"),
    )
    assert search._EMBED_PROCESS_LOCK.acquire(blocking=False)
    try:
        result = search.build_or_update_embed_index("bge-m3", chunks)
    finally:
        search._EMBED_PROCESS_LOCK.release()

    assert len(result["entries"]) == 2
    assert not list(checkpoint_path.glob("batch-*.json"))


def test_second_writer_without_usable_cache_reports_busy(tmp_path, monkeypatch):
    """有効な最終cacheが無い場合は待たずにbusyを通知する。"""
    _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        search,
        "_get_embedding",
        lambda text, model: pytest.fail("2本目のwriterはEmbedding APIを呼ばない"),
    )
    assert search._EMBED_PROCESS_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(search.EmbedIndexBusyError):
            search.build_or_update_embed_index("bge-m3", _embed_test_chunks(1))
    finally:
        search._EMBED_PROCESS_LOCK.release()


def test_load_checkpoint_ignores_corrupt_foreign_and_invalid_entries(tmp_path, monkeypatch):
    """破損JSON・欠番・別generation・不正vectorをfail-safeに読み飛ばす。"""
    checkpoint_path = tmp_path / "embed_cache.checkpoints"
    checkpoint_path.mkdir()
    monkeypatch.setattr(search, "EMBED_CHECKPOINT_PATH", checkpoint_path)
    state = search._new_checkpoint_state("bge-m3")
    (checkpoint_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def batch(sequence, entries, generation=None):
        return json.dumps(
            {
                "schema_version": search.EMBED_CHECKPOINT_SCHEMA_VERSION,
                "generation": generation or state["generation"],
                "sequence": sequence,
                "entries": entries,
            }
        )

    def entry(chunk_id, embedding):
        return {
            "chunk_id": chunk_id,
            "path": "fixture.md",
            "text": "t",
            "file_sha256": "f",
            "text_sha256": "t",
            "embedding": embedding,
        }

    (checkpoint_path / "batch-00000001.json").write_text(
        batch(1, [entry("a.md#0001", [1.0, 0.0])]), encoding="utf-8"
    )
    (checkpoint_path / "batch-00000002.json").write_text("{broken", encoding="utf-8")
    # 3 は欠番（batch確定前に中断したケース）
    (checkpoint_path / "batch-00000004.json").write_text(
        batch(4, [entry("d.md#0001", [1.0, 0.0])], generation="other-generation"),
        encoding="utf-8",
    )
    (checkpoint_path / "batch-00000005.json").write_text(
        batch(
            5,
            [
                entry("e.md#0001", []),
                entry("e.md#0002", [float("nan")]),
                {"chunk_id": "e.md#0003"},
            ],
        ),
        encoding="utf-8",
    )
    (checkpoint_path / "batch-00000006.json.tmp").write_text("{}", encoding="utf-8")

    loaded_state, entries = search._load_checkpoint("bge-m3")

    assert set(entries) == {"a.md#0001"}
    assert loaded_state["next_batch"] == 6
    assert not list(checkpoint_path.glob("*.tmp"))


def test_checkpoint_header_treats_omitted_tag_as_latest():
    """省略タグは:latestと同一identity、明示した別tagは不一致とする。"""
    state = search._new_checkpoint_state("bge-m3")

    assert search._checkpoint_header_matches(state, "bge-m3")
    assert search._checkpoint_header_matches(state, "bge-m3:latest")
    assert not search._checkpoint_header_matches(state, "bge-m3:q4_k_m")


def test_checkpoint_entry_takes_precedence_over_stale_final_cache(tmp_path, monkeypatch):
    """同一chunkは、hash一致するcheckpoint entryを最終cacheより優先する。"""
    cache_path, checkpoint_path, _ = _isolate_embed_runtime_paths(tmp_path, monkeypatch)
    chunks = _embed_test_chunks(1)
    chunk = chunks[0]
    cache_path.write_text(
        json.dumps(
            {
                "version": search.EMBED_CACHE_VERSION,
                "embed_model": "bge-m3:latest",
                "chunking": {
                    "max_chars": search.CHUNK_MAX_CHARS,
                    "overlap_chars": search.CHUNK_OVERLAP_CHARS,
                },
                "entries": {
                    chunk["chunk_id"]: {
                        "chunk_id": chunk["chunk_id"],
                        "path": chunk["path"],
                        "text": "stale body",
                        "file_sha256": "stale",
                        "text_sha256": "stale",
                        "embedding": [0.1, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path.mkdir()
    state = search._new_checkpoint_state(
        "bge-m3", search.compute_embed_generation("bge-m3", chunks)
    )
    (checkpoint_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (checkpoint_path / "batch-00000001.json").write_text(
        json.dumps(
            {
                "schema_version": search.EMBED_CHECKPOINT_SCHEMA_VERSION,
                "generation": state["generation"],
                "sequence": 1,
                "entries": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "path": chunk["path"],
                        "text": chunk["text"],
                        "file_sha256": chunk["file_sha256"],
                        "text_sha256": chunk["text_sha256"],
                        "embedding": [0.9, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        search,
        "_get_embedding",
        lambda text, model: pytest.fail("hash一致entryを再生成してはならない"),
    )
    events = []

    cache = search.build_or_update_embed_index("bge-m3", chunks, emit_progress=events.append)

    assert cache["entries"][chunk["chunk_id"]]["embedding"] == [0.9, 0.0]
    assert events[0]["phase"] == "resumed"
    assert events[0]["reused"] == 1
