"""親子展開（P1a: 構造解析・原文順展開・単一/複数連続範囲出力）の決定的テスト。

対象: expand_parent_candidates / _expand_single_parent。
run_retrieval_pipeline / finalize_ranked_matches への統合はP1bで行うため、
本ファイルは keyword 検索専用の隔離テストとして展開関数を直接検証する。
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def _write(tmp_path, rel_path, text):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _parent_match(path, start_line, end_line, *, chunk_id="doc.md#0001", heading="親"):
    return {
        "path": path,
        "chunk_id": chunk_id,
        "heading": heading,
        "start_line": start_line,
        "end_line": end_line,
        "snippet": heading,
        "source": "keyword",
    }


def test_expand_recovers_child_body_not_hit_directly(tmp_path):
    """見出しだけの親candidateから、単独ヒットしなかった子本文が展開される。"""
    text = "# 親見出し\n\n## 子見出し\n子の本文行1\n子の本文行2\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    # 親（見出しだけ）を検索candidateとして扱う。
    parent_chunk = next(c for c in chunks if c["heading"] == "親見出し")

    items, meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path
    )

    assert len(items) == 1
    assert "子の本文行1" in items[0]["snippet"]
    assert "子の本文行2" in items[0]["snippet"]
    assert items[0]["expanded_from"] == parent_chunk["chunk_id"]
    assert items[0]["group_order"] == 1
    assert meta[0]["expanded"] is True


def test_expand_does_not_mix_similar_sibling_sections(tmp_path):
    """似た語を含む別の親配下の子を混ぜない。"""
    text = (
        "# 親A\n\n## 設定手順\n手順A-1\n手順A-2\n"
        "# 親B\n\n## 設定手順\n手順B-1\n手順B-2\n"
    )
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_a = next(c for c in chunks if c["heading"] == "親A")

    items, _meta = search.expand_parent_candidates(
        [parent_a], chunks, source_root=tmp_path
    )

    joined = "\n".join(item["snippet"] for item in items)
    assert "手順A-1" in joined
    assert "手順B-1" not in joined


def test_leaf_parent_without_children_is_not_expanded(tmp_path):
    text = "# 見出し\n本文のみで子見出しなし\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = chunks[0]

    items, meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path
    )

    assert items == []
    assert meta == []


def test_parent_with_own_body_is_not_expanded_non_blank_gap(tmp_path):
    """親見出し行の直後に本文がある（空行gapを満たさない）場合は展開しない。"""
    text = "# 親\n親自身の本文がある\n## 子\n子の本文\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["start_line"] == 1)

    items, meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path
    )

    assert items == []


def test_quoted_ranges_match_actual_source_lines(tmp_path):
    """全引用本文が固定snapshotの実際の行範囲と一致する。"""
    text = "# 親\n\n## 子\n行4\n行5\n行6\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")

    items, _meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path
    )

    lines = text.splitlines()
    for item in items:
        expected = "\n".join(lines[item["start_line"] - 1 : item["end_line"]])
        assert item["snippet"] == expected


def test_budget_truncation_marks_partial_and_preserves_prefix(tmp_path):
    """予算超過時は部分展開としてマークし、先頭からの内容は保持する。"""
    long_child = "\n".join(f"手順{i}" for i in range(50))
    text = f"# 親\n\n## 子\n{long_child}\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")

    items, meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path, char_budget=20
    )

    assert items, "予算内で採用できる先頭分は残るべき"
    assert meta[0]["partial"] is True
    assert "手順0" in items[0]["snippet"]


def test_source_changed_hash_mismatch_skips_expansion(tmp_path):
    """検索時のhashと現在資料のhashが一致しない場合は展開を省略する。"""
    text = "# 親\n\n## 子\n本文\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = dict(next(c for c in chunks if c["heading"] == "親"))
    parent_chunk["source_sha256"] = "0" * 64  # 資料のhashと一致しない

    items, meta = search.expand_parent_candidates(
        [parent_chunk], chunks, source_root=tmp_path
    )

    assert items == []
    assert meta[0]["reason"] == "source_changed"


def test_missing_source_file_skips_without_adding_whole_document(tmp_path):
    """資料が取得できない場合、展開を省略し資料全体を代わりに追加しない。"""
    parent_chunk = _parent_match("missing.md", 1, 1)

    items, meta = search.expand_parent_candidates(
        [parent_chunk], [], source_root=tmp_path
    )

    assert items == []
    assert meta[0]["reason"] == "structure_unavailable"


def test_max_parents_and_total_ranges_are_enforced(tmp_path):
    text_a = "# 親A\n\n## 子A1\n本文A1\n## 子A2\n本文A2\n"
    text_b = "# 親B\n\n## 子B1\n本文B1\n"
    _write(tmp_path, "a.md", text_a)
    _write(tmp_path, "b.md", text_b)
    chunks_a = search.chunk_source_file(tmp_path / "a.md", "a.md")
    chunks_b = search.chunk_source_file(tmp_path / "b.md", "b.md")
    all_chunks = chunks_a + chunks_b
    parent_a = next(c for c in chunks_a if c["heading"] == "親A")
    parent_b = next(c for c in chunks_b if c["heading"] == "親B")

    items, meta = search.expand_parent_candidates(
        [parent_a, parent_b],
        all_chunks,
        source_root=tmp_path,
        max_parents=1,
        max_total_ranges=4,
    )

    assert len(meta) == 1
    assert meta[0]["path"] == "a.md"
    assert all(item["path"] == "a.md" for item in items)


def test_group_id_is_deterministic_for_same_snapshot(tmp_path):
    text = "# 親\n\n## 子\n本文\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")

    items1, _ = search.expand_parent_candidates([parent_chunk], chunks, source_root=tmp_path)
    items2, _ = search.expand_parent_candidates([parent_chunk], chunks, source_root=tmp_path)

    assert items1[0]["group_id"] == items2[0]["group_id"]


def test_child_without_rrf_score_still_survives_expansion(tmp_path):
    """語の一致・rrf_scoreを持たない子も、合格した親の配下として残る。"""
    text = "# 親\n\n## 子\nまったく異なる語彙のみを含む本文\n"
    _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
    parent_chunk = next(c for c in chunks if c["heading"] == "親")
    assert "rrf_score" not in parent_chunk

    items, _meta = search.expand_parent_candidates([parent_chunk], chunks, source_root=tmp_path)

    assert len(items) == 1
    assert "rrf_score" not in items[0]
