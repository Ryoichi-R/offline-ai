import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def test_markdown_heading_chunks_keep_heading_and_lines(tmp_path):
    source = tmp_path / "handbook.md"
    source.write_text("# 年次休暇\n本文1\n本文2\n\n# 病気休暇\n本文3", encoding="utf-8")

    chunks = search.chunk_source_file(source, "rules/leave.md", max_chars=50)

    assert chunks[0]["chunk_id"] == "rules/leave.md#0001"
    assert chunks[0]["heading"] == "年次休暇"
    assert chunks[0]["start_line"] == 1
    assert "本文1" in chunks[0]["text"]
    assert chunks[1]["heading"] == "病気休暇"


def test_long_text_splits_to_multiple_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "CHUNK_OVERLAP_CHARS", 0)
    source = tmp_path / "long.txt"
    source.write_text("\n".join(["abcdef"] * 20), encoding="utf-8")

    chunks = search.chunk_source_file(source, "long.txt", max_chars=40)

    assert len(chunks) > 1
    assert all(chunk["path"] == "long.txt" for chunk in chunks)
