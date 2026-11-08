"""段階0で切り分けた欠落の回帰: 取得段階と、SSE・保存の抜粋段階（guided-follow-up 段階1）。

合成の通知と偽モデルで、主要事項が (1) 取得 (2) モデル入力 (3) SSE・保存の抜粋
のどこで欠けるかを確認する。草稿・最終回答の意味の反映は実モデルが必要なため、
ここでは扱わない。
"""

import hashlib

import deep_candidate_selection
import deep_research
import search
import source_view
import web_services
from deep_fakes import patch_retrieval

FILLER = "この取扱いは各部局の会計事務において統一的に運用するものとする。" * 15
KEY_X = "契約金額が100万円以上の場合は担当者の記名を必須とする。"
KEY_Y = "電子署名がある場合は記名も省略できる。ただし外国法人は除く。"
NOTICE = "\n".join(
    [
        "# 会計事務における押印の取扱いについて（通知）",  # 1
        "各部局会計課長 殿",  # 2
        "令和6年4月1日 会計課長",  # 3
        "## 1 趣旨",  # 4
        "請求書等の押印を求めないこととする。",  # 5
        "## 2 取扱い",  # 6
        "(1) 請求書は押印を省略できる。" + FILLER,  # 7
        "(2) " + KEY_X + FILLER,  # 8
        "(3) " + KEY_Y,  # 9
        "## 3 適用日",  # 10
        "令和6年4月1日から適用する。",  # 11
        "## 4 照会先",  # 12
        "会計課 契約係（内線1234）",  # 13
        "",
    ]
)


def _corpus(tmp_path, **extra):
    root = tmp_path / "skill-source"
    root.mkdir()
    (root / "notice.md").write_text(NOTICE, encoding="utf-8")
    for name, text in extra.items():
        (root / f"{name}.md").write_text(text, encoding="utf-8")
    return root


def _run_deep(monkeypatch, root, query, hit_headings, answer):
    chunks = search.build_source_chunks(root)
    by_heading = {c["heading"]: c for c in chunks if c.get("heading")}
    candidates = [
        {
            "path": "notice.md",
            "chunk_id": by_heading[h]["chunk_id"],
            "heading": h,
            "start_line": by_heading[h]["start_line"],
            "end_line": by_heading[h]["end_line"],
            "score": 2 - i * 0.1,
            "rrf_score": 0.5 - i * 0.01,
            "source_sha256": by_heading[h]["file_sha256"],
        }
        for i, h in enumerate(hit_headings)
    ]
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_json",
        lambda *a, **k: (
            {"supported": True, "contradictions": [], "missing_conditions": [], "unsupported_claims": []}
            if a[1] == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": [], "unresolved": []}
            if a[1] == deep_research.DEEP_GAPS_SYSTEM
            else {"subject": "押印", "scope": "会計事務", "conditions": [], "exceptions": [], "references": []}
        ),
    )
    prompts = []

    def fake_text(*args, **kwargs):
        prompts.append(args[2])
        return answer(prompts)

    monkeypatch.setattr(deep_research, "_call_ollama_text", fake_text)
    result = deep_research.run_deep_research(query, model="m", source_root=root, timeout_seconds=300)
    return result, prompts


def test_broad_question_reads_main_sections_even_when_hits_are_ancillary(tmp_path, monkeypatch):
    """段階0の欠落(取得): 検索ヒットが適用日だけでも、確定した通知の主要節を読む。"""
    root = _corpus(tmp_path)
    result, prompts = _run_deep(
        monkeypatch, root, "会計事務における押印の取扱いについての内容詳細", ["3 適用日"],
        lambda prompts: "押印を求めない。[E1]",
    )
    ranges = {(e["start_line"], e["end_line"]) for e in result.evidence}
    assert {(4, 5), (6, 9)} <= ranges
    # 主要節を先に読み、検索ヒット（適用日）はその後に読む。
    assert [(e["start_line"], e["end_line"]) for e in result.evidence][:2] == [(4, 5), (6, 9)]
    # 予算内の選別後にモデルへ送った入力に主要事項が含まれる。
    assert any(KEY_X in prompt and KEY_Y in prompt for prompt in prompts)


def test_specific_question_keeps_the_existing_selection(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    result, _ = _run_deep(
        monkeypatch, root, "会計事務における押印の取扱いについての適用日はいつですか", ["3 適用日"],
        lambda prompts: "令和6年4月1日から適用する。[E1]",
    )
    ranges = {(e["start_line"], e["end_line"]) for e in result.evidence}
    assert (6, 9) not in ranges and (10, 11) in ranges


def test_multiple_documents_keep_the_existing_selection(tmp_path, monkeypatch):
    root = _corpus(tmp_path, other="# 押印の見直しについて\n本文。\n")
    result, _ = _run_deep(
        monkeypatch, root, "押印の内容詳細", ["3 適用日"], lambda prompts: "適用する。[E1]",
    )
    assert (6, 9) not in {(e["start_line"], e["end_line"]) for e in result.evidence}


def test_unread_primary_range_is_reported(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    monkeypatch.setattr(deep_research, "DEEP_MAX_UNIT_CHARS", 600)
    monkeypatch.setattr(
        deep_research, "_split_line_range",
        lambda lines, a, b, max_chars=600, _orig=deep_research._split_line_range: _orig(lines, a, b, max_chars=max_chars),
    )
    result, _ = _run_deep(
        monkeypatch, root, "会計事務における押印の取扱いについての内容詳細", ["3 適用日"],
        lambda prompts: "押印を求めない。[E1]",
    )
    assert any(item.startswith("primary_unread: notice.md") and "2 取扱い" in item for item in result.unconfirmed)


def test_deep_snippet_prefers_the_cited_part_and_records_its_own_lines(tmp_path, monkeypatch):
    """段階0の欠落(SSE・保存): 4000文字の単位の先頭500文字だけを保存していた。"""
    root = _corpus(tmp_path)

    result, _ = _run_deep(
        monkeypatch, root, "会計事務における押印の取扱いについての内容詳細", ["2 取扱い"],
        lambda prompts: "押印を求めない。[E1]",
    )
    record = next(e for e in result.evidence if (e["start_line"], e["end_line"]) == (6, 9))
    result.answer = f"{KEY_X}[{record['evidence_id']}]"
    event = web_services.build_deep_evidence_event(result, query="押印の取扱い")
    item = next(i for i in event["items"] if (i["startLine"], i["endLine"]) == (6, 9))
    assert KEY_X in item["snippet"]
    assert len(item["snippet"]) <= web_services.EVIDENCE_EXCERPT_MAX_CHARS
    assert item["snippetTruncated"] is True
    lines = NOTICE.splitlines()
    excerpt_lines = lines[item["snippetStartLine"] - 1 : item["snippetEndLine"]]
    assert item["snippet"] == "\n".join(excerpt_lines)[: web_services.EVIDENCE_EXCERPT_MAX_CHARS]


def test_select_excerpt_keeps_short_ranges_whole_and_falls_back_to_the_head():
    whole = web_services.select_excerpt("一行目\n二行目", 10)
    assert whole == {"snippet": "一行目\n二行目", "startLine": 10, "endLine": 11, "truncated": False}
    text = "\n".join(["先頭" + "あ" * 300, "中間" + "い" * 300, "末尾の重要事項です"])
    head = web_services.select_excerpt(text, 1)
    assert head["startLine"] == 1 and head["truncated"] is True
    focused = web_services.select_excerpt(text, 1, focus=("重要事項",))
    assert focused["endLine"] == 3 and "末尾の重要事項です" in focused["snippet"]
    long_line = web_services.select_excerpt("う" * 900, 5)
    assert long_line["snippet"] == "う" * 500 and (long_line["startLine"], long_line["endLine"]) == (5, 5)


def test_normal_evidence_excerpt_comes_from_the_hashed_source_range(tmp_path):
    root = _corpus(tmp_path)
    data = (root / "notice.md").read_bytes()
    retrieval = search.RetrievalResult(
        query="100万円以上の記名",
        attempts=[],
        matches=[
            {
                "path": "notice.md",
                "start_line": 6,
                "end_line": 9,
                "snippet": "検索時チャンクの本文",
                "source_sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        confidence=0.5,
        evidence_status="partial",
        user_prompt="",
    )
    registry = source_view.EvidenceRegistry(root)
    item = web_services.build_evidence_event(retrieval, evidence_registry=registry, session_id="s")["items"][0]
    assert "100万円以上" in item["snippet"]
    assert item["snippetTruncated"] is True
    assert item["snippet"] in "\n".join(NOTICE.splitlines()[item["snippetStartLine"] - 1 : item["snippetEndLine"]])
    # ハッシュが一致しなければ検索時の抜粋を使い、抜粋の行範囲は記録しない。
    retrieval.matches[0]["source_sha256"] = "0" * 64
    item = web_services.build_evidence_event(retrieval, evidence_registry=registry, session_id="s")["items"][0]
    assert item["snippet"] == "検索時チャンクの本文" and "snippetStartLine" not in item


def test_normal_path_adds_primary_ranges_only_for_a_confirmed_broad_question(tmp_path):
    root = _corpus(tmp_path)
    chunks = search.build_source_chunks(root)
    hit = [{"path": "notice.md", "start_line": 10, "end_line": 11, "snippet": "令和6年4月1日から適用する。", "score": 1.0}]
    items, warnings = search._with_primary_ranges(
        "会計事務における押印の取扱いについての内容詳細", hit,
        source_root=root, chunks=chunks, char_budget=8000, item_limit=5,
    )
    primary = [i for i in items if i.get("primary_range")]
    assert primary and primary[0]["start_line"] == 4
    lines = NOTICE.splitlines()
    for item in primary:
        assert item["snippet"] == "\n".join(lines[item["start_line"] - 1 : item["end_line"]])
        assert item["source"] == "expanded"
    assert any(KEY_X in i["snippet"] for i in primary)
    assert items[-1] == hit[0] and warnings == []
    # 具体的な質問、チャンク検索無効、ハッシュ不一致では変えない。
    assert search._with_primary_ranges(
        "会計事務における押印の取扱いについての適用日はいつ", hit,
        source_root=root, chunks=chunks, char_budget=8000, item_limit=5,
    ) == (hit, [])
    assert search._with_primary_ranges(
        "会計事務における押印の取扱いについての内容詳細", hit,
        source_root=root, chunks=None, char_budget=8000, item_limit=5,
    ) == (hit, [])
    changed = [dict(c, file_sha256="0" * 64) for c in chunks]
    assert search._with_primary_ranges(
        "会計事務における押印の取扱いについての内容詳細", hit,
        source_root=root, chunks=changed, char_budget=8000, item_limit=5,
    ) == (hit, [])


def test_normal_path_warns_about_unread_primary_ranges(tmp_path):
    root = _corpus(tmp_path)
    items, warnings = search._with_primary_ranges(
        "会計事務における押印の取扱いについての内容詳細", [],
        source_root=root, chunks=search.build_source_chunks(root), char_budget=300, item_limit=5,
    )
    assert any("2 取扱い" in w for w in warnings)
    assert all(i.get("primary_range") for i in items)


def test_fixed_document_restricts_deep_research_to_the_focus_range(tmp_path, monkeypatch):
    """段階2: 固定した資料の焦点範囲を先に読み、他の資料は調査対象にしない。"""
    root = _corpus(tmp_path, other="# 別の通知\n押印の取扱いの別資料。\n")
    sha = hashlib.sha256((root / "notice.md").read_bytes()).hexdigest()
    chunks = search.build_source_chunks(root)
    other = next(c for c in chunks if c["path"] == "other.md")
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(
            [{"path": c["path"], "chunk_id": c["chunk_id"], "heading": c.get("heading", ""),
              "start_line": c["start_line"], "end_line": c["end_line"], "score": 1, "rrf_score": 0.5,
              "source_sha256": c["file_sha256"]} for c in ch],
            source_chunks=ch,
        ),
    )
    monkeypatch.setattr(
        deep_research, "_call_ollama_json",
        lambda *a, **k: (
            {"supported": True, "contradictions": [], "missing_conditions": [], "unsupported_claims": []}
            if a[1] == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": [], "unresolved": []} if a[1] == deep_research.DEEP_GAPS_SYSTEM
            else {"subject": "押印", "scope": "会計事務", "conditions": [], "exceptions": [], "references": []}
        ),
    )
    monkeypatch.setattr(deep_research, "_call_ollama_text", lambda *a, **k: "記名が必要。[E1]")
    result = deep_research.run_deep_research(
        "請求書の記名", model="m", source_root=root, timeout_seconds=300,
        fixed_document={"path": "notice.md", "sha256": sha, "focus_ranges": [[6, 9]], "primary": False},
    )
    seen_paths = {e["path"] for e in result.evidence}
    assert seen_paths == {"notice.md"} and other["path"] not in seen_paths
    assert (result.evidence[0]["start_line"], result.evidence[0]["end_line"]) == (6, 9)
    changed = deep_research.run_deep_research(
        "請求書の記名", model="m", source_root=root, timeout_seconds=300,
        fixed_document={"path": "notice.md", "sha256": "0" * 64},
    )
    assert changed.evidence == [] and any(u.startswith("source_changed") for u in changed.unconfirmed)


def test_fixed_document_limits_normal_evidence_to_the_focus_range(tmp_path):
    root = _corpus(tmp_path)
    sha = hashlib.sha256((root / "notice.md").read_bytes()).hexdigest()
    items, _ = search._with_primary_ranges(
        "請求書の記名", [], source_root=root, chunks=None, char_budget=8000, item_limit=5,
        forced={"path": "notice.md", "title": "通知", "sha256": sha, "focus_ranges": [[10, 11]], "primary": False},
    )
    assert [(i["start_line"], i["end_line"]) for i in items] == [(10, 11)]
    assert search._with_primary_ranges(
        "q", [], source_root=root, chunks=None, char_budget=8000, item_limit=5,
        forced={"path": "notice.md", "title": "通知", "sha256": sha, "focus_ranges": [], "primary": False},
    ) == ([], [])
    assert search._with_primary_ranges(
        "q", [], source_root=root, chunks=None, char_budget=8000, item_limit=5,
        forced={"path": "notice.md", "title": "通知", "sha256": "0" * 64, "focus_ranges": [[10, 11]]},
    ) == ([], [])
