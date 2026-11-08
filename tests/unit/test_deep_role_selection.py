"""役割を考慮した候補選別（回答方針v2）。旧経路は完全に不変であること、検索の打切りで直接規定が落ちないこと。"""

import pytest

import answer_policy as ap
import deep_candidate_selection as selection
import deep_transport
from answer_fixtures import QUERY


def chunk(index, path="doc.md", heading="", text="本文"):
    return {
        "path": path,
        "chunk_id": f"{path}#{index}",
        "start_line": index * 10 + 1,
        "end_line": index * 10 + 9,
        "file_sha256": "f" * 64,
        "heading": heading,
        "text": text,
    }


RULE = "第３条　委員会において審査する案件は、次のとおりとする。\n- 一　概算費用１，０００万円以上の設備更新"
TABLE = "| ①　図面が添付されているか。 | |\n| ②　保守契約の期間は適切か。 | |"
INTENT = ap.extract_explicit_intent(QUERY)


def role_of(chunk):
    return ap.classify_role(chunk["text"], chunk["heading"], intent=INTENT).role


def ranked_from(chunks):
    return [{**c, "source_sha256": c["file_sha256"], "score": 1.0 / (i + 1)} for i, c in enumerate(chunks)]


def tables_then_rule(count=30, rule_rank=None):
    """チェック表が上位を占め、直接規定が下位にある検索結果。"""
    tables = [chunk(i, path=f"checklist-{i % 6}.md", heading="別紙　設備更新委員会チェックリスト", text=TABLE) for i in range(count)]
    rule = chunk(900, path="rule.md", heading="（審査案件及び審査内容）", text=RULE)
    position = len(tables) if rule_rank is None else rule_rank
    return [*tables[:position], rule, *tables[position:]]


def test_legacy_selection_is_unchanged_without_a_role_function():
    chunks = tables_then_rule(count=60)
    legacy = selection.select_candidates(ranked_from(chunks), source_chunks=chunks)
    assert legacy["counts"] == {"ranked": 61, "selected": 40, "excluded": 21}
    assert all(ref["path"] != "rule.md" for ref in legacy["selected"])  # 下位の直接規定は旧順位では落ちる
    assert "role_selected" not in legacy["counts"]


def test_role_aware_selection_keeps_a_low_ranked_direct_rule():
    chunks = tables_then_rule(count=60)
    result = selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=role_of)
    assert any(ref["path"] == "rule.md" for ref in result["selected"])
    assert result["counts"]["selected"] == 40
    assert result["counts"]["role_selected"]["direct_answer"] == 1
    assert selection.validate_response(result) is result


def test_selected_candidates_stay_in_rank_order_and_limits_hold():
    chunks = tables_then_rule(count=60, rule_rank=45)
    result = selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=role_of)
    ids = [ref["chunk_id"] for ref in result["selected"]]
    expected = [c["chunk_id"] for c in chunks if c["chunk_id"] in set(ids)]
    assert ids == expected
    per_file = {}
    for ref in result["selected"]:
        per_file[ref["path"]] = per_file.get(ref["path"], 0) + 1
    assert max(per_file.values()) <= selection.DEEP_MAX_CANDIDATES_PER_FILE
    assert len(ids) <= selection.DEEP_MAX_CANDIDATES_TOTAL


def test_the_priority_share_does_not_starve_the_original_rank_path():
    rules = [chunk(i, path=f"rule-{i}.md", heading="（審査案件及び審査内容）", text=RULE) for i in range(40)]
    tables = [chunk(100 + i, path=f"checklist-{i}.md", heading="別紙チェックリスト", text=TABLE) for i in range(40)]
    chunks = [*tables[:10], *rules, *tables[10:]]
    result = selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=role_of)
    selected = [ref["path"] for ref in result["selected"]]
    quota = int(selection.DEEP_MAX_CANDIDATES_TOTAL * selection.ROLE_PRIORITY_SHARE)
    assert sum(p.startswith("rule-") for p in selected) >= quota
    # 残りの枠は元の順位の経路で埋まる。上位10件のチェック表は優先枠に押し出されず残る。
    top_tables = {f"checklist-{i}.md" for i in range(10)}
    assert top_tables <= set(selected)
    assert len(selected) == selection.DEEP_MAX_CANDIDATES_TOTAL


def test_only_the_top_ranked_candidates_are_classified():
    classified = []

    def counting(chunk):
        classified.append(chunk["chunk_id"])
        return role_of(chunk)

    chunks = tables_then_rule(count=selection.ROLE_SCAN_LIMIT + 50)
    selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=counting)
    assert len(classified) == selection.ROLE_SCAN_LIMIT


def test_response_validation_accepts_role_counts_and_rejects_bad_ones():
    chunks = tables_then_rule(count=5)
    result = selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=role_of)
    bad = {**result, "counts": {**result["counts"], "role_selected": {"direct_answer": -1}}}
    with pytest.raises(ValueError):
        selection.validate_response(bad)
    bad2 = {**result, "counts": {**result["counts"], "surprise": 1}}
    with pytest.raises(ValueError):
        selection.validate_response(bad2)


def test_excluded_references_still_account_for_every_candidate():
    chunks = tables_then_rule(count=60)
    result = selection.select_candidates(ranked_from(chunks), source_chunks=chunks, role_of=role_of)
    counts = result["counts"]
    assert counts["ranked"] == counts["selected"] + counts["excluded"] == len(chunks)
    assert sum(counts["role_selected"].values()) == counts["selected"]
    assert sum(counts["role_excluded"].values()) == counts["excluded"]


def test_worker_payload_intent_is_validated_and_the_real_worker_reports_roles(tmp_path, monkeypatch):
    import search

    text = "# 制度\n\n### （審査案件及び審査内容）\n\n" + RULE + "\n\n## 別紙　チェックリスト\n\n" + TABLE + "\n"
    source = tmp_path / "sample.md"
    source.write_text(text, encoding="utf-8")
    chunks = search._chunk_source_bytes(text.encode("utf-8"), source, "sample.md")
    monkeypatch.setenv("OFFLINE_AI_EMBED_MODEL", "")
    payload = {"operation": "retrieve", "query": QUERY, "chunks": chunks, "isolated_embedding": {"model": None, "cache_path": None, "cache_sha256": None}}
    legacy = deep_transport._execute(dict(payload))
    assert "role_selected" not in legacy["counts"]
    with_intent = deep_transport._execute({**payload, "intent": INTENT.to_dict()})
    assert with_intent["counts"]["role_selected"]
    assert selection.validate_response(with_intent)
    with pytest.raises(ValueError):
        deep_transport._execute({**payload, "intent": {"granularity": "huge"}})
