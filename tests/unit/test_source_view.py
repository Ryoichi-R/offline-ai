import hashlib

import pytest

from source_view import EvidenceRegistry, EvidenceViewError


def test_search_snapshot_hash_rejects_change_during_chunking(tmp_path, monkeypatch):
    import search

    source = tmp_path / "race.md"
    original_bytes = "検索時の本文\r\n末尾\xff".encode("utf-8") + b"\xff"
    source.write_bytes(original_bytes)
    original_chunker = search._line_chunks

    def change_after_text_read(*args, **kwargs):
        source.write_bytes("変更後の本文\n末尾".encode("utf-8"))
        return original_chunker(*args, **kwargs)

    monkeypatch.setattr(search, "_line_chunks", change_after_text_read)
    chunk = search.chunk_source_file(source, source.name)[0]
    assert "検索時の本文" in chunk["text"]
    assert "\ufffd" in chunk["text"]
    assert chunk["file_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    registry = EvidenceRegistry(tmp_path)
    evidence_id = registry.register(
        session_id="session-a",
        relative_path=source.name,
        start_line=chunk["start_line"],
        end_line=chunk["end_line"],
        source_sha256=chunk["file_sha256"],
    )
    assert evidence_id
    with pytest.raises(EvidenceViewError) as excinfo:
        registry.view(evidence_id, session_id="session-a")
    assert excinfo.value.code == "source_changed"


@pytest.mark.parametrize("expires_at", [10.0, 11.0])
def test_view_does_not_extend_ttl_or_hide_expired_entries(tmp_path, expires_at):
    source = tmp_path / "guide.md"
    source.write_text("本文", encoding="utf-8")
    now = [0.0]
    registry = EvidenceRegistry(tmp_path, ttl_seconds=10, clock=lambda: now[0])
    first = _register(registry, source)
    now[0] = 5.0
    second = _register(registry, source)
    now[0] = 6.0
    registry.view(first, session_id="session-a")
    now[0] = expires_at
    with pytest.raises(EvidenceViewError) as excinfo:
        registry.view(first, session_id="session-a")
    assert excinfo.value.status == 404
    assert registry.view(second, session_id="session-a")["relativePath"] == source.name


def test_view_does_not_change_oldest_registration_eviction(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("本文", encoding="utf-8")
    registry = EvidenceRegistry(tmp_path, max_entries=2)
    first = _register(registry, source)
    second = _register(registry, source)
    registry.view(first, session_id="session-a")
    third = _register(registry, source)
    with pytest.raises(EvidenceViewError):
        registry.view(first, session_id="session-a")
    for evidence_id in (second, third):
        assert registry.view(evidence_id, session_id="session-a")["relativePath"] == source.name


def _register(registry, source, *, start=2, end=2, session="session-a"):
    return registry.register(
        session_id=session,
        relative_path=source.name,
        start_line=start,
        end_line=end,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def test_view_reads_utf8_lines_and_marks_highlight(tmp_path):
    source = tmp_path / "guide.md"
    source.write_bytes("先頭\r\n強調行\r\n末尾\n".encode("utf-8"))
    registry = EvidenceRegistry(tmp_path)
    evidence_id = _register(registry, source)

    result = registry.view(evidence_id, session_id="session-a")

    assert result["relativePath"] == "guide.md"
    assert result["startLine"] == 2
    assert [line["text"] for line in result["lines"]] == ["先頭", "強調行", "末尾"]
    assert [line["lineNumber"] for line in result["lines"] if line["highlighted"]] == [2]
    assert result["truncated"] is False


def test_view_is_bound_to_session_and_hash(tmp_path):
    source = tmp_path / "guide.txt"
    source.write_text("本文", encoding="utf-8")
    registry = EvidenceRegistry(tmp_path)
    evidence_id = _register(registry, source, start=1, end=1)

    with pytest.raises(EvidenceViewError) as excinfo:
        registry.view(evidence_id, session_id="other-session")
    assert excinfo.value.code == "evidence_not_found"

    source.write_text("変更後", encoding="utf-8")
    with pytest.raises(EvidenceViewError) as excinfo:
        registry.view(evidence_id, session_id="session-a")
    assert excinfo.value.code == "source_changed"
    assert str(source) not in str(excinfo.value)


def test_view_rejects_deleted_source_without_returning_stale_body(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("検索時の本文", encoding="utf-8")
    registry = EvidenceRegistry(tmp_path)
    evidence_id = _register(registry, source, start=1, end=1)
    source.unlink()

    with pytest.raises(EvidenceViewError) as excinfo:
        registry.view(evidence_id, session_id="session-a")
    assert excinfo.value.code == "source_changed"
    assert "再検索" in str(excinfo.value)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../guide.md",
        r"C:\guide.md",
        r"\\server\share\guide.md",
        "guide.md:secret",
        ".hidden/guide.md",
    ],
)
def test_register_rejects_unsafe_paths(tmp_path, relative_path):
    source = tmp_path / "guide.md"
    source.write_text("本文", encoding="utf-8")
    registry = EvidenceRegistry(tmp_path)

    assert (
        registry.register(
            session_id="session-a",
            relative_path=relative_path,
            start_line=1,
            end_line=1,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        is None
    )


def test_registry_ttl_and_limit_evict_old_entries(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("本文", encoding="utf-8")
    now = [0.0]
    registry = EvidenceRegistry(tmp_path, ttl_seconds=10, max_entries=1, clock=lambda: now[0])
    first = _register(registry, source)
    second = _register(registry, source)
    assert first != second
    with pytest.raises(EvidenceViewError):
        registry.view(first, session_id="session-a")
    assert registry.view(second, session_id="session-a")["relativePath"] == "guide.md"
    now[0] = 11.0
    with pytest.raises(EvidenceViewError):
        registry.view(second, session_id="session-a")
