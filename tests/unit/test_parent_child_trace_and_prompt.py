"""親子展開の残差分に対する回帰試験。

- 文字予算・件数枠の配分で、原文順ではなく選択順位（子の類似度）を優先する
- 配下の非空行を渡しきれない展開を完全扱いしない
- 構造を読んだ資料と検索chunkの世代不一致を展開しない、展開中のキャンセル
- 候補→落選理由→展開→最終採用の段階追跡（本文を含まない）
- 部分展開を回答プロンプトへ伝え、prompt shell 見積りで表示分を予約する

Ollama を必要としない（chat/embedding を一切呼ばない）。
"""

import json
import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import prompt_templates  # noqa: E402
import search  # noqa: E402


def _write(tmp_path, rel_path, text):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _chunks_and_parent(tmp_path, text, parent_heading="親"):
    path = _write(tmp_path, "doc.md", text)
    chunks = search.chunk_source_file(path, "doc.md")
    parent = next(c for c in chunks if c["heading"] == parent_heading)
    return chunks, parent


# ---------------------------------------------------------------------------
# 選択順位
# ---------------------------------------------------------------------------


def test_char_budget_is_assigned_by_selection_priority_not_document_order(tmp_path):
    """離れた2範囲が文字予算に収まらない場合、類似度の高い範囲を先に採用する。"""
    text = (
        "# 親\n\n"
        "## 子1\n子1の本文は低い類似度\n"
        "## 子2\n子2の本文は最も低い類似度\n"
        "## 子3\n子3の本文は高い類似度\n"
    )
    chunks, parent = _chunks_and_parent(tmp_path, text)
    by_heading = {c["heading"]: c for c in chunks}
    index = search._EmbedIndex(
        chunk_ids=[by_heading[h]["chunk_id"] for h in ("子1", "子2", "子3")],
        entries=[{}, {}, {}],
        vectors=[[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]],
        norms=[1.0, 1.0, 1.0],
    )
    ctx = search.QueryContext(queries=["q"], vectors={"q": [1.0, 0.0]}, embed_model="m")

    items, meta = search.expand_parent_candidates(
        [parent],
        chunks,
        source_root=tmp_path,
        max_parents=1,
        max_ranges_per_parent=2,
        char_budget=len("## 子3\n子3の本文は高い類似度") + 1,
        query_context=ctx,
        embed_index=index,
    )

    assert [item["snippet"].splitlines()[0] for item in items] == ["## 子3"]
    assert meta[0]["partial"] is True
    assert "char_budget" in meta[0]["partial_reasons"]


def test_slot_allocation_keeps_highest_priority_range_and_document_order():
    parent = {"path": "a.md", "chunk_id": "a.md#0001", "start_line": 1, "end_line": 1}
    ranges = [
        {
            "path": "doc.md",
            "chunk_id": f"g#r0{order}",
            "start_line": order * 10,
            "end_line": order * 10,
            "source": "expanded",
            "group_id": "g",
            "expanded_from": "a.md#0001",
            "group_order": order,
            "group_priority": priority,
        }
        for order, priority in ((1, 3), (2, 1), (3, 2))
    ]

    result = search._allocate_evidence_slots([parent] + ranges, 3)

    group_items = [m for m in result if m.get("group_id")]
    assert [m["group_order"] for m in group_items] == [2, 3]
    assert all("match_limit" in m["group_partial_reasons"] for m in group_items)


# ---------------------------------------------------------------------------
# 完全性・世代・キャンセル
# ---------------------------------------------------------------------------


def test_expansion_with_uncovered_body_lines_is_partial(tmp_path):
    """見出し判定の差で境界を跨ぐチャンクが除外され、配下の行を渡せない場合は部分展開。"""
    text = "# 親\n\n## 子A\n本文A\n## 子B\n本文B\n#\n後続の本文\n"
    chunks, parent = _chunks_and_parent(tmp_path, text)

    items, meta = search.expand_parent_candidates([parent], chunks, source_root=tmp_path)

    assert items, "取り込める子Aは展開する"
    assert all("本文B" not in item["snippet"] for item in items)
    assert meta[0]["partial"] is True
    assert "uncovered_lines" in meta[0]["partial_reasons"]


def test_complete_expansion_records_required_body_ranges(tmp_path):
    text = "# 親\n\n## 子A\n本文A\n\n## 子B\n本文B\n"
    chunks, parent = _chunks_and_parent(tmp_path, text)

    items, meta = search.expand_parent_candidates([parent], chunks, source_root=tmp_path)

    assert meta[0]["partial"] is False
    assert items[0]["group_required_ranges"] == [[3, 4], [6, 7]]


def test_child_chunks_from_other_source_generation_are_not_expanded(tmp_path):
    """親候補にhashが無くても、子chunkのhashが構造を読んだ資料と違えば展開しない。"""
    text = "# 親\n\n## 子\n本文\n"
    chunks, parent = _chunks_and_parent(tmp_path, text)
    parent = {k: v for k, v in parent.items() if k not in {"file_sha256", "source_sha256"}}
    stale_chunks = [{**c, "file_sha256": "0" * 64} for c in chunks]

    items, meta = search.expand_parent_candidates([parent], stale_chunks, source_root=tmp_path)

    assert items == []
    assert meta[0]["reason"] == "source_changed"


def test_expansion_calls_cancel_check_before_each_parent(tmp_path):
    text = "# 親\n\n## 子\n本文\n"
    chunks, parent = _chunks_and_parent(tmp_path, text)

    class Cancelled(Exception):
        pass

    def cancel():
        raise Cancelled()

    with pytest.raises(Cancelled):
        search.expand_parent_candidates(
            [parent], chunks, source_root=tmp_path, cancel_check=cancel
        )


# ---------------------------------------------------------------------------
# 段階追跡
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["search", "answer"])
def test_pipeline_trace_records_stages_without_body_text(monkeypatch, tmp_path, mode):
    monkeypatch.delenv("OFFLINE_AI_PARENT_CHILD_EXPANSION", raising=False)
    _write(
        tmp_path,
        "doc.md",
        "# 障害対応手順\n\n## 初動\n初動では秘匿本文マーカーで通報する\n",
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

    attempt = result.trace["attempts"][0]
    assert attempt["candidates"], "候補段階を記録する"
    assert any(meta.get("expanded") for meta in attempt["expansion"])
    assert attempt["expanded"], "展開段階を記録する"
    assert any(
        d["reason"] in {"per_file_limit", "relative_score_floor", "match_limit"}
        for d in attempt.get("dropped", [])
    ), "落選理由を記録する"
    final = result.trace["final"]["final"]
    assert [ref.get("chunk_id") for ref in final] == [m.get("chunk_id") for m in result.matches]
    serialized = json.dumps(result.trace, ensure_ascii=False)
    assert "秘匿本文マーカー" not in serialized
    assert '"snippet"' not in serialized and '"text"' not in serialized


def test_final_selection_trace_records_budget_truncation():
    items = [
        {"path": "a.md", "chunk_id": "a#1", "start_line": 1, "end_line": 1, "snippet": "あ" * 50},
        {"path": "b.md", "chunk_id": "b#1", "start_line": 1, "end_line": 1, "snippet": "い" * 50},
    ]
    trace: dict = {}

    search.select_final_evidence("質問", items, limit=5, char_limit=60, trace=trace)

    assert [d["chunk_id"] for d in trace["truncated"]] == ["b#1"]
    assert [ref["chunk_id"] for ref in trace["final"]] == ["a#1", "b#1"]


# ---------------------------------------------------------------------------
# 回答プロンプト
# ---------------------------------------------------------------------------


def _expanded_item(partial):
    return {
        "path": "doc.md",
        "heading": "手順",
        "start_line": 3,
        "end_line": 5,
        "snippet": "手順の一部",
        "source": "expanded",
        "group_id": "g",
        "group_partial": partial,
    }


def test_prompt_marks_partial_expansion_and_forbids_claiming_completeness():
    prompt = prompt_templates.build_user_prompt("手順の全体は", [_expanded_item(True)])

    assert prompt_templates.EXPANDED_EVIDENCE_NOTE in prompt
    assert prompt_templates.PARTIAL_EXPANSION_NOTE in prompt
    assert prompt_templates.PARTIAL_EXPANSION_CONTRACT in prompt


def test_prompt_does_not_add_partial_contract_for_complete_or_direct_evidence():
    direct = {"path": "a.md", "snippet": "本文", "source": "keyword"}
    prompt = prompt_templates.build_user_prompt("質問", [_expanded_item(False), direct])

    assert prompt.count(prompt_templates.EXPANDED_EVIDENCE_NOTE) == 1
    assert prompt_templates.PARTIAL_EXPANSION_NOTE not in prompt
    assert prompt_templates.PARTIAL_EXPANSION_CONTRACT not in prompt


def test_prompt_shell_estimate_reserves_expansion_notices():
    base = len(prompt_templates.SYSTEM_PROMPT) + len(
        prompt_templates.build_user_prompt(
            "質問", [], attempts=[], evidence_status="partial", confidence=0.5
        )
    )
    estimated = search._estimate_prompt_shell_chars(
        "質問", attempts=[], evidence_status="partial", confidence=0.5
    )
    assert estimated - base == prompt_templates.expansion_prompt_reserve_chars(
        search.EVIDENCE_OUTPUT_MAX_ITEMS
    )
    assert estimated - base >= len(prompt_templates.PARTIAL_EXPANSION_CONTRACT)
