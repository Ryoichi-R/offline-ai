"""Tests for the explicit Ollama /api/embed batch contract."""

import json

import pytest

import search


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_get_embeddings_posts_array_and_preserves_order(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"embeddings": [[1, 0], [0, 1]]})

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)

    result = search._get_embeddings(["first", "second"], "bge-m3")

    assert captured["url"].endswith("/api/embed")
    assert captured["body"]["input"] == ["first", "second"]
    assert captured["body"]["truncate"] is True
    assert result == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "payload, code",
    [
        ({"embeddings": [[1, 0]]}, "EMBED_COUNT_MISMATCH"),
        ({"embeddings": [[]]}, "EMBED_EMPTY_VECTOR"),
        ({"embeddings": [[1, float("nan")]]}, "EMBED_INVALID_VECTOR"),
    ],
)
def test_get_embeddings_fails_closed_on_invalid_response(monkeypatch, payload, code):
    monkeypatch.setattr(
        search.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )

    inputs = ["text", "other"] if code == "EMBED_COUNT_MISMATCH" else ["text"]
    with pytest.raises(search.EmbeddingBatchError) as exc_info:
        search._get_embeddings(inputs, "bge-m3")
    assert exc_info.value.code == code


def test_index_batch_splits_once_on_retryable_batch_error(monkeypatch):
    calls = []

    def fake_get(texts, model):
        calls.append(len(texts))
        if len(texts) > 2:
            raise search.EmbeddingBatchError("EMBED_HTTP_413", "too large", retryable=True)
        return [[float(index), 0.0] for index, _ in enumerate(texts)]

    monkeypatch.setattr(search, "_get_embeddings", fake_get)

    result = search._get_index_embedding_batch(["a", "b", "c", "d"], "bge-m3")

    assert calls == [4, 2, 2]
    assert len(result) == 4


def test_generation_changes_when_source_identity_changes():
    chunks = [
        {
            "chunk_id": "a.md#0001",
            "file_sha256": "file-a",
            "text_sha256": "text-a",
        }
    ]

    first = search.compute_embed_generation("bge-m3", chunks)
    same = search.compute_embed_generation("bge-m3", list(chunks))
    changed = search.compute_embed_generation("bge-m3", [{**chunks[0], "text_sha256": "text-b"}])

    assert first == same
    assert first != changed


def test_index_status_does_not_start_build(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "EMBED_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(search, "EMBED_STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(search, "detect_embed_model", lambda: "bge-m3")
    monkeypatch.setattr(search, "build_source_chunks", lambda _root: [])

    def unexpected(*_args, **_kwargs):
        pytest.fail("status inspection must not build an index")

    monkeypatch.setattr(search, "build_or_update_embed_index", unexpected)

    status = search.get_embed_index_status("bge-m3", [])

    assert status["state"] == "missing"
