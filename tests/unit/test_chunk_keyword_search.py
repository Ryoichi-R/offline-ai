import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def test_chunk_keyword_search_ranks_heading_match(tmp_path, monkeypatch):
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "leave.md").write_text(
        "# 年次休暇\n申請期限は3日前です。", encoding="utf-8"
    )
    (tmp_path / "rules" / "other.md").write_text(
        "# その他\n年次休暇という単語だけあります。", encoding="utf-8"
    )
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "CHUNK_OVERLAP_CHARS", 0)

    results = search.keyword_search_chunks("年次休暇の申請期限", "年次休暇 申請期限", top_k=2)

    assert results[0]["path"] == "rules/leave.md"
    assert results[0]["snippet"]
