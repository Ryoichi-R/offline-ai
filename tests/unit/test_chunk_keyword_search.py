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


def _chunk(path, text, heading=""):
    return {
        "path": path,
        "chunk_id": path + "#1",
        "heading": heading,
        "text": text,
        "start_line": 1,
        "end_line": 2,
    }


def test_japanese_question_preserves_original_and_extracts_subject_terms():
    terms = search._split_terms("安全審査会が審議する事項を教えてください。サーバーの取扱いは？")
    assert "安全審査会が審議する事項を教えてください" in terms
    assert {"安全審査会", "審議", "事項", "サーバー", "取扱"} <= set(terms)
    assert len(terms) == len(set(terms))


def test_unspaced_subject_beats_repeated_generic_terms():
    query = "資料において、安全審査会が審議する事項を教えてください。条件と例外を示してください。"
    chunks = [
        _chunk("background.md", "条件と例外を示してください。" * 100),
        _chunk("core.md", "安全審査会は設備の新設について審議する。", "安全審査会"),
    ]
    matches = search.keyword_search_chunks(query, top_k=2, chunks=chunks)
    assert matches[0]["path"] == "core.md"
    assert "安全審査会" in matches[0]["hitKeywords"]
    assert len(matches) == 2  # Ranking does not silently discard other sources.


def test_katakana_subject_beats_generic_request_text():
    chunks = [
        _chunk("generic.md", "方法を教えてください。例外の条件。" * 80),
        _chunk("network.md", "ルーターの更新方法。", "ルーター更新"),
    ]
    matches = search.keyword_search_chunks(
        "ルーターを更新する方法を教えてください。", chunks=chunks
    )
    assert matches[0]["path"] == "network.md"


def test_mixed_script_and_english_exact_terms_are_retained():
    assert "azure-openai" in search._split_terms("Azure-OpenAI connection")
    assert "取扱い" in search._split_terms("取扱い")
    chunks = [_chunk("a.md", "Different product."), _chunk("b.md", "Azure-OpenAI connection")]
    assert (
        search.keyword_search_chunks("Azure-OpenAI connection", chunks=chunks)[0]["path"] == "b.md"
    )


def test_empty_and_unmatched_questions_remain_empty():
    chunks = [_chunk("a.md", "全く無関係な文書")]
    assert search.keyword_search_chunks("", chunks=chunks) == []
    assert search.keyword_search_chunks("zzzzzzzz", chunks=chunks) == []
