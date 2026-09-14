"""親子展開のP1b: run_retrieval_pipeline / finalize_ranked_matches 統合試験。

OFFLINE_AI_PARENT_CHILD_EXPANSION による既定ON（明示的にfalseでOFF切り替え
可能）、agentic-lite分離（展開前候補だけを次試行のmerge/_retry_queriesへ
渡す）、8件以下の出力上限、sufficient追加判定経路を、keyword専用の決定的
経路で検証する。Ollama を必要としない（chat/embedding を一切呼ばない）。
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def _write(tmp_path, rel_path, text):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def _prepare_keyword_only_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(
        search,
        "create_search_plan",
        lambda query, model: {
            "keywords": [],
            "search_queries": [query],
            "must_find_terms": [],
            "fallback": "",
        },
    )


def test_expansion_enabled_by_default(monkeypatch, tmp_path):
    """既定（環境変数未設定）ONでは、見出しだけの親候補が展開される。

    2026-09-14、E:実機でのP2比較受入（result/offline-ai/
    parent-child-expansion-p2-20260913/）を経て利用者判断により既定ONへ
    切り替えた。
    """
    monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)
    _write(
        tmp_path,
        "doc.md",
        "# 手順概要\n\n## 手順の詳細\n手順の詳細本文をここに書く\n",
    )
    _prepare_keyword_only_pipeline(monkeypatch, tmp_path)

    result = search.run_retrieval_pipeline("手順概要", model="m", mode="search")

    expanded = [m for m in result.matches if m.get("source") == "expanded"]
    assert expanded, "既定ONでは展開item が根拠に含まれるべき"
    assert any("手順の詳細本文" in m.get("snippet", "") for m in expanded)


def test_expansion_can_be_disabled_via_env_var(monkeypatch, tmp_path):
    """OFFLINE_AI_PARENT_CHILD_EXPANSION=false を明示すれば展開を無効化できる。"""
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", "false")
    try:
        _write(
            tmp_path,
            "doc.md",
            "# 手順概要\n\n## 手順の詳細\n手順の詳細本文をここに書く\n",
        )
        _prepare_keyword_only_pipeline(monkeypatch, tmp_path)

        result = search.run_retrieval_pipeline("手順概要", model="m", mode="search")

        assert all(m.get("source") != "expanded" for m in result.matches)
    finally:
        monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)


def test_expansion_enabled_recovers_child_body(monkeypatch, tmp_path):
    """ONにすると、見出しだけの親ヒットから配下本文が展開されて根拠に入る。"""
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", "true")
    try:
        _write(
            tmp_path,
            "doc.md",
            "# 手順概要\n\n## 手順の詳細\n手順の詳細本文をここに書く\n",
        )
        _prepare_keyword_only_pipeline(monkeypatch, tmp_path)

        result = search.run_retrieval_pipeline("手順概要", model="m", mode="search")

        expanded = [m for m in result.matches if m.get("source") == "expanded"]
        assert expanded, "展開item が根拠に含まれるべき"
        assert any("手順の詳細本文" in m.get("snippet", "") for m in expanded)
    finally:
        monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)


def test_expansion_output_never_exceeds_eight_items(monkeypatch, tmp_path):
    """展開元2件+通常根拠を同時に持つケースでも出力は8件以下。"""
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", "true")
    try:
        sections = []
        for doc_index in range(3):
            heading = f"# 資料{doc_index}手順概要\n\n"
            children = "".join(
                f"## 手順{doc_index}-{child_index}\n本文{doc_index}-{child_index}詳細\n"
                for child_index in range(4)
            )
            sections.append((f"doc{doc_index}.md", heading + children))
        for rel_path, text in sections:
            _write(tmp_path, rel_path, text)
        _prepare_keyword_only_pipeline(monkeypatch, tmp_path)

        result = search.run_retrieval_pipeline("手順概要", model="m", mode="search")

        assert len(result.matches) <= 8
    finally:
        monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)


def test_expansion_group_split_by_limit_marks_retained_ranges_partial():
    """件数上限で一部の範囲を省いたgroupは、残った範囲を部分展開として記録する。"""
    items = [
        {"path": "a.md", "start_line": 1, "end_line": 1, "chunk_id": "a#1"},
        {"path": "b.md", "start_line": 1, "end_line": 1, "group_id": "g1", "group_order": 1},
        {"path": "b.md", "start_line": 3, "end_line": 3, "group_id": "g1", "group_order": 2},
        {"path": "b.md", "start_line": 5, "end_line": 5, "group_id": "g1", "group_order": 3},
    ]
    limited = search._allocate_evidence_slots(items, 2)
    group_items = [m for m in limited if m.get("group_id") == "g1"]
    assert [m["start_line"] for m in group_items] == [1]
    assert all(m["group_partial"] is True for m in group_items)
    assert "group_partial" not in items[1], "入力dictを変更しない"


def test_retry_queries_use_pre_expansion_candidates(monkeypatch, tmp_path):
    """次試行のmerge/_retry_queriesは展開前候補（ranked_candidates）だけを使う。

    展開groupのsnippet（親配下の全本文）に must_find_terms の語が含まれて
    いても、_retry_queries の判定対象は展開前候補でなければならない。
    """
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", "true")
    try:
        _write(
            tmp_path,
            "doc.md",
            "# 概要\n\n## 詳細\n必須キーワードXYZを含む本文\n",
        )
        _prepare_keyword_only_pipeline(monkeypatch, tmp_path)
        monkeypatch.setattr(
            search,
            "create_search_plan",
            lambda query, model: {
                "keywords": [],
                "search_queries": [query],
                "must_find_terms": ["必須キーワードXYZ"],
                "fallback": "",
            },
        )
        seen_retry_inputs = []
        original_retry = search._retry_queries

        def spy_retry_queries(query, plan, matches):
            seen_retry_inputs.append(matches)
            return original_retry(query, plan, matches)

        monkeypatch.setattr(search, "_retry_queries", spy_retry_queries)
        monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)

        search.run_retrieval_pipeline("概要", model="m", mode="answer")

        for matches in seen_retry_inputs:
            assert all(m.get("source") != "expanded" for m in matches)
    finally:
        monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)


def test_sufficient_upgrade_requires_complete_expansion_and_terms():
    """完全展開1件・must_find_terms全充足・矛盾なしでsufficientを許容する。"""
    matches = [
        {
            "path": "doc.md",
            "chunk_id": "g1#r01",
            "heading": "詳細",
            "start_line": 3,
            "end_line": 3,
            "snippet": "必須語Aと必須語Bを含む本文です。関連する説明も十分にあります。",
            "source": "expanded",
            "group_id": "g1",
            "group_partial": False,
        }
    ]
    assert search._has_complete_structural_evidence(matches, ["必須語A", "必須語B"]) is True


def test_sufficient_upgrade_rejected_when_terms_missing():
    matches = [
        {
            "path": "doc.md",
            "chunk_id": "g1#r01",
            "heading": "詳細",
            "start_line": 3,
            "end_line": 3,
            "snippet": "必須語Aだけを含む本文です。",
            "source": "expanded",
            "group_id": "g1",
            "group_partial": False,
        }
    ]
    assert search._has_complete_structural_evidence(matches, ["必須語A", "必須語B"]) is False


def test_sufficient_upgrade_rejected_when_partial():
    matches = [
        {
            "path": "doc.md",
            "chunk_id": "g1#r01",
            "heading": "詳細",
            "start_line": 3,
            "end_line": 3,
            "snippet": "必須語Aと必須語Bを含む本文。",
            "source": "expanded",
            "group_id": "g1",
            "group_partial": True,
        }
    ]
    assert search._has_complete_structural_evidence(matches, ["必須語A", "必須語B"]) is False


def test_direct_hit_overlapping_expansion_range_is_deduped():
    direct = [
        {"path": "doc.md", "start_line": 3, "end_line": 4, "chunk_id": "doc.md#0002"},
    ]
    expanded = [
        {"path": "doc.md", "start_line": 3, "end_line": 4, "chunk_id": "g1#r01", "group_id": "g1"},
        {"path": "doc.md", "start_line": 6, "end_line": 7, "chunk_id": "g1#r02", "group_id": "g1"},
    ]
    kept = search._dedupe_expansion_against_direct_hits(expanded, direct)
    assert [item["chunk_id"] for item in kept] == ["g1#r02"]


def test_parent_without_relevance_support_is_not_expanded(monkeypatch, tmp_path):
    """親自身が支持判定（filter_by_relevance_support）を通らない場合は展開しない。"""
    monkeypatch.setenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", "true")
    try:
        _write(tmp_path, "doc.md", "# 無関係な見出し\n\n## 子\n配下本文\n")
        chunks = search.chunk_source_file(tmp_path / "doc.md", "doc.md")
        weak_parent = dict(next(c for c in chunks if c["heading"] == "無関係な見出し"))
        # heading/snippet が質問語と全く重ならず embedding score も無いため
        # filter_by_relevance_support を通らない想定。
        weak_parent["snippet"] = weak_parent["heading"]

        matches, confidence, status = search.finalize_ranked_matches(
            "まったく別の質問文言",
            [weak_parent],
            source_chunks=chunks,
            source_root=tmp_path,
        )

        assert matches == []
    finally:
        monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)


def test_sufficient_normal_evidence_not_downgraded_by_weak_expansion_group():
    """通常根拠がsufficientの時、弱い（不完全・矛盾なしだが条件不足の）展開groupを
    追加してもpartialへ落ちない。"""
    normal_matches = [
        {
            "path": "a.md",
            "chunk_id": "a.md#0001",
            "heading": "第2条 日当",
            "snippet": "出張1日あたりの日当は、一般社員で2,500円とする。",
            "rrf_score": 0.04,
            "source": "keyword+embedding",
            "embedding_score": 0.8,
        },
        {
            "path": "a.md",
            "chunk_id": "a.md#0002",
            "heading": "第3条 宿泊費",
            "snippet": "宿泊費は実費精算とし、1泊あたりの上限額を定める。",
            "rrf_score": 0.03,
            "source": "keyword+embedding",
            "embedding_score": 0.75,
        },
    ]
    # sufficient単体での確認（展開なし）
    confidence, status = search._calculate_confidence(
        "一般社員の日当はいくらですか", normal_matches
    )
    assert status == "sufficient"

    # 不完全（partial）な展開groupを混ぜても状態は変わらない想定。
    with_weak_expansion = normal_matches + [
        {
            "path": "b.md",
            "chunk_id": "g1#r01",
            "heading": "無関係節",
            "snippet": "無関係な断片",
            "source": "expanded",
            "group_id": "g1",
            "group_partial": True,
        }
    ]
    confidence2, status2 = search._calculate_confidence(
        "一般社員の日当はいくらですか", with_weak_expansion
    )
    assert status2 == "sufficient"


def test_pre_limit_conflict_is_preserved_after_final_budget_truncation(monkeypatch, tmp_path):
    """最終出力時の予算切詰めが発生しても、制限前に検出済みの矛盾状態は維持する。"""
    monkeypatch.setattr(search, "SKILL_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(search, "detect_embed_model", lambda: None)
    monkeypatch.setattr(
        search,
        "create_search_plan",
        lambda query, model: {
            "keywords": [],
            "search_queries": [query],
            "must_find_terms": [],
            "fallback": "",
        },
    )
    _write(
        tmp_path,
        "travel.md",
        "# 国内出張旅費規程\n\n海外出張は本規程の対象外とし、別途定める規程による。\n\n"
        "## 第2条 日当\n\n海外出張の日当について参照する。一般社員は2,500円とする。\n",
    )

    result = search.run_retrieval_pipeline("海外出張の日当はいくらですか", model="m", mode="search")

    if result.matches:
        assert result.evidence_status != "sufficient"


def test_confidence_does_not_count_expanded_children_as_independent_evidence():
    """子の増加だけでは独立根拠数が増えず、sufficientへ昇格しない。"""
    matches = [
        {
            "path": "doc.md",
            "chunk_id": "doc.md#0001",
            "heading": "見出し",
            "snippet": "見出し",
            "rrf_score": 0.02,
            "source": "keyword",
        },
        {
            "path": "doc.md",
            "chunk_id": "g1#r01",
            "heading": "見出し",
            "snippet": "配下の本文A",
            "source": "expanded",
            "group_id": "g1",
        },
        {
            "path": "doc.md",
            "chunk_id": "g1#r02",
            "heading": "見出し",
            "snippet": "配下の本文B",
            "source": "expanded",
            "group_id": "g1",
        },
    ]
    confidence, status = search._calculate_confidence("見出し", matches)
    assert status != "sufficient"
