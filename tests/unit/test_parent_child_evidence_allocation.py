"""親子展開の採用根拠の件数枠配分・切り詰め・状態再判定の回帰試験。

- 直接ヒットが多い質問でも、展開本文が件数上限で黙って落ちないこと
- 切り詰めた展開itemの end_line が実際に渡した本文の行範囲と一致すること
- 件数枠・根拠予算を適用した後の採用根拠だけで evidence_status を再判定すること

Ollama を必要としない（chat/embedding を一切呼ばない）。
"""

import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402


def _write(tmp_path, rel_path, text):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _direct(index, **extra):
    item = {
        "path": f"d{index}.md",
        "chunk_id": f"d{index}.md#0001",
        "start_line": 1,
        "end_line": 2,
        "snippet": f"直接{index}",
    }
    item.update(extra)
    return item


def _range(group_id, order, parent_chunk_id, **extra):
    item = {
        "path": "doc.md",
        "chunk_id": f"{group_id}#r{order:02d}",
        "start_line": order * 10,
        "end_line": order * 10 + 2,
        "snippet": f"展開{group_id}-{order}",
        "source": "expanded",
        "group_id": group_id,
        "expanded_from": parent_chunk_id,
        "group_order": order,
        "group_partial": False,
    }
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# 件数枠の配分
# ---------------------------------------------------------------------------


def test_each_group_gets_a_slot_even_when_direct_hits_fill_the_limit():
    """直接ヒットだけで上限を超えても、各展開groupへ先に1枠を割り当てる。"""
    direct = [_direct(i) for i in range(6)]
    groups = [_range("g0", 1, "d0.md#0001"), _range("g0", 2, "d0.md#0001")]
    groups += [_range("g1", 1, "d3.md#0001"), _range("g1", 2, "d3.md#0001")]

    result = search._allocate_evidence_slots(direct + groups, 5)

    assert len(result) == 5
    assert {m["group_id"] for m in result if m.get("group_id")} == {"g0", "g1"}
    normal = [m for m in result if not m.get("group_id")]
    assert len(normal) == 3, "通常根拠の枠も確保する"
    assert all(m["group_partial"] is True for m in result if m.get("group_id"))


def test_group_is_placed_right_after_its_parent():
    direct = [_direct(i) for i in range(3)]
    expanded = [_range("g", 1, "d1.md#0001"), _range("g", 2, "d1.md#0001")]

    result = search._allocate_evidence_slots(direct + expanded, 8)

    assert [m["chunk_id"] for m in result] == [
        "d0.md#0001",
        "d1.md#0001",
        "g#r01",
        "g#r02",
        "d2.md#0001",
    ]
    assert all(m["group_partial"] is False for m in result if m.get("group_id"))


def test_leftover_slots_go_to_expansion_when_direct_hits_are_exhausted():
    direct = [_direct(0)]
    expanded = [_range("g", order, "d0.md#0001") for order in range(1, 5)]

    result = search._allocate_evidence_slots(direct + expanded, 5)

    assert len(result) == 5
    assert all(m["group_partial"] is False for m in result if m.get("group_id"))


def test_allocation_is_idempotent_on_its_own_output():
    direct = [_direct(i) for i in range(6)]
    expanded = [_range("g", order, "d2.md#0001") for order in range(1, 4)]

    first = search._allocate_evidence_slots(direct + expanded, 5)
    second = search._allocate_evidence_slots(first, 5)

    assert second == first


@pytest.mark.parametrize("mode", ["search", "answer"])
def test_pipeline_keeps_expanded_body_when_many_direct_hits_exist(monkeypatch, tmp_path, mode):
    """直接ヒットする他資料が多くても、見出しだけの親の配下本文が採用根拠に残る。

    修正前は展開itemが通常根拠の後ろへ並び、先頭5件で切られて黙って落ちていた。
    """
    monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)
    _write(
        tmp_path,
        "doc.md",
        "# 障害対応手順\n\n## 初動\n初動では連絡網で責任者へ通報する\n\n"
        "## 復旧\n復旧では切替手順を実施する\n",
    )
    for index in range(6):
        _write(tmp_path, f"other{index}.md", f"# 別資料{index}\n障害対応手順の概要{index}\n")
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

    result = search.run_retrieval_pipeline("障害対応手順", model="m", mode=mode)

    assert len(result.matches) <= search.final_evidence_match_limit()
    expanded = [m for m in result.matches if m.get("source") == "expanded"]
    assert expanded, "展開本文が件数上限で落ちてはならない"
    assert any("連絡網" in m.get("snippet", "") for m in expanded)
    assert any(m.get("source") != "expanded" for m in result.matches)


# ---------------------------------------------------------------------------
# 状態の再判定
# ---------------------------------------------------------------------------


def _structural_items():
    parent = {
        "path": "doc.md",
        "chunk_id": "doc.md#0001",
        "heading": "障害対応手順",
        "start_line": 1,
        "end_line": 1,
        "snippet": "障害対応手順",
        "source": "keyword",
        "rrf_score": 0.05,
    }
    required = {"group_required_ranges": [[10, 12], [20, 22]]}
    first = _range("g", 1, "doc.md#0001", snippet="障害対応手順の初動", **required)
    second = _range("g", 2, "doc.md#0001", snippet="連絡網で通報する", **required)
    return [parent, first, second]


def test_structural_sufficient_holds_when_whole_group_is_adopted():
    matches, _confidence, status = search.select_final_evidence(
        "障害対応手順", _structural_items(), limit=5, must_find_terms=["連絡網"]
    )
    assert len(matches) == 3
    assert status == "sufficient"


def test_status_is_reevaluated_after_slot_allocation_drops_ranges():
    """件数枠で必要語を含む範囲が落ちたら、完全構造根拠として sufficient にしない。"""
    matches, _confidence, status = search.select_final_evidence(
        "障害対応手順", _structural_items(), limit=2, must_find_terms=["連絡網"]
    )
    assert all("連絡網" not in m.get("snippet", "") for m in matches)
    assert status != "sufficient"


def test_status_is_reevaluated_after_char_budget_truncation():
    items = _structural_items()
    budget = len(items[0]["snippet"]) + len(items[1]["snippet"]) + 2
    matches, _confidence, status = search.select_final_evidence(
        "障害対応手順", items, limit=5, must_find_terms=["連絡網"], char_limit=budget
    )
    assert all(m["group_partial"] is True for m in matches if m.get("group_id"))
    assert status != "sufficient"


def test_pre_limit_conflict_survives_final_selection():
    items = _structural_items()
    items[0] = {**items[0], "constraint_conflict": True}
    items.insert(0, _direct(9, rrf_score=0.06))
    matches, _confidence, status = search.select_final_evidence(
        "障害対応手順", items, limit=2, must_find_terms=["連絡網"]
    )
    assert matches[0].get("constraint_conflict") is True
    assert status != "sufficient"


# ---------------------------------------------------------------------------
# 切り詰め後の引用範囲
# ---------------------------------------------------------------------------


def _assert_quote_matches_lines(item, lines):
    expected = "\n".join(lines[item["start_line"] - 1 : item["end_line"]])
    assert item["snippet"] == expected


def test_expansion_char_budget_truncation_keeps_end_line_aligned(tmp_path):
    body = "\n".join(f"手順{i}の本文" for i in range(50))
    text = f"# 親\n\n## 子\n{body}\n"
    path = _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(path, "doc.md")
    parent = next(c for c in chunks if c["heading"] == "親")

    items, meta = search.expand_parent_candidates(
        [parent], chunks, source_root=tmp_path, char_budget=30
    )

    assert meta[0]["partial"] is True
    lines = text.splitlines()
    for item in items:
        _assert_quote_matches_lines(item, lines)
        assert len(item["snippet"]) <= 30
    assert items[-1]["end_line"] < len(lines)


def test_prompt_budget_truncation_realigns_expanded_end_line_and_marks_group(tmp_path):
    lines = [f"行{i}の本文です" for i in range(1, 21)]
    expanded = [
        _range("g", 1, "p#1", start_line=1, end_line=10, snippet="\n".join(lines[0:10])),
        _range("g", 2, "p#1", start_line=12, end_line=20, snippet="\n".join(lines[11:20])),
    ]
    normal = {"path": "n.md", "start_line": 1, "end_line": 9, "snippet": "あ" * 100}

    fitted = search._fit_prompt_budget(expanded + [normal], 30)

    fitted_expanded = [m for m in fitted if m.get("source") == "expanded"]
    assert [m["chunk_id"] for m in fitted_expanded] == ["g#r01"], "1行も収まらない範囲は採用しない"
    first = fitted_expanded[0]
    assert first["end_line"] < 10
    _assert_quote_matches_lines(first, lines)
    assert first["group_partial"] is True
    assert sum(len(m["snippet"]) for m in fitted) <= 30
    assert expanded[0]["end_line"] == 10, "入力dictを変更しない"


def test_prompt_budget_keeps_normal_item_range_contract_unchanged():
    normal = {"path": "n.md", "start_line": 1, "end_line": 9, "snippet": "あ" * 100}
    fitted = search._fit_prompt_budget([normal], 30)
    assert fitted[0]["snippet"] == "あ" * 30
    assert fitted[0]["end_line"] == 9
