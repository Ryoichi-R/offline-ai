"""段階3（実資料照合後の改修 R1〜R6）の回帰。plans/offline-ai-guided-follow-up-plan.md 7章。

発端事例（会計課長通知）と同じ構造の合成資料を使う。正解は原文から先に定めた
K1（押印は原則不要）・K2（真正性の確保）・別紙一覧・除外の注記。
"""

import hashlib
import json

import pytest

import answer_pipeline as pipeline
import deep_candidate_selection
import deep_research
import guided_follow_up as gf
import search
import source_structure as ss
import web_services
from deep_fakes import patch_retrieval

K1 = "会計手続における押印は、法令に基づかないものも含め原則不要とする。"
K2 = "押印を求めていた書類は、本人確認や連絡内容の記録等により真正性を確認する。"
NOTICE_LINES = (
    "> 会発1225第２号",  # 1
    "> 令和２年12月25日",  # 2
    "",  # 3
    "本省内部部局長",  # 4
    "",  # 5
    "殿",  # 6
    "",  # 7
    "**大臣官房会計課長（公印省略）**",  # 8
    "",  # 9
    "# 会計手続における押印の見直しについて（通知）",  # 10
    "",  # 11
    "政府方針を踏まえ、会計手続における押印の見直しについて下記のとおり通知する。",  # 12
    "",  # 13
    "記",  # 14
    "",  # 15
    "## １ 押印の見直し",  # 16
    K1,  # 17
    "## ２ 真正性の確保",  # 18
    K2,  # 19
    "## ３ 実施時期",  # 20
    "本通知は、令和３年１月１日から適用する。",  # 21
    "",  # 22
    "（別紙）",  # 23
    "",  # 24
    "| 押印欄のある様式等 | 通知等の名称 |",  # 25
    "| --- | --- |",  # 26
    "| 指名停止通知書 | 指名停止等の措置要領 |",  # 27
    "| 再苦情申立書 | 公共調達審査会の設置 |",  # 28
    "| 業務完了報告書 | 検査調書等の取扱い |",  # 29
    "",  # 30
    "※ 本通知とは別に通知の改正を行い、押印の見直しを行っている取組については除く。",  # 31
    "",  # 32
    "上記以外で押印の取扱いについて疑義がある場合には、監査指導室あて照会されたい。",  # 33
)
NOTICE = "\n".join(NOTICE_LINES) + "\n"
MEMO_LINES = (
    "【機密性２：省内限り】",  # 1
    "",  # 2
    "> 事務連絡",  # 3
    "> 令和２年12月25日",  # 4
    "",  # 5
    "会計事務担当課長",  # 6
    "",  # 7
    "殿",  # 8
    "",  # 9
    "**大臣官房会計課監査指導室長**",  # 10
    "",  # 11
    "# 押印を前提としている契約関係書類の取扱いについて",  # 12
    "",  # 13
    "令和２年12月25日付け会発 1225第２号をもって通知されたところですが、真正性の確認が重要となります。",  # 14
    "",  # 15
    "## 押印省略確認　チェックリスト（例）",  # 16
    "",  # 17
    "契約（調達）案件名：",  # 18
    "",  # 19
    "| | 確認事項 | チェック欄 | 備考 |",  # 20
    "| --- | --- | --- | --- |",  # 21
    "| 1 | 押印省略の意思表示 |  |  |",  # 22
    "| 2 | 名刺等の受領 |  |  |",  # 23
    "| 3 | 連絡先の記載の有無 |  | （確認先）○○ ○○ |",  # 24
)
MEMO = "\n".join(MEMO_LINES) + "\n"
NOTICE_PATH = "notice/押印見直し/会計課長通知（会計手続における押印の見直しについて）.md"
MEMO_PATH = (
    "notice/押印見直し/【監査指導室長事務連絡】押印を前提としている契約関係書類の取扱いについて.md"
)


def _root(tmp_path, extra=None):
    root = tmp_path / "skill-source"
    for path, text in {NOTICE_PATH: NOTICE, MEMO_PATH: MEMO, **(extra or {})}.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


# --- R1 付随範囲の共通判定 ----------------------------------------------------------


def test_classify_range_marks_headers_frontmatter_blank_forms_and_contacts():
    assert ss.classify_range(NOTICE, 1, 9) is True  # 宛先・発出日・発出者・文書番号
    assert ss.classify_range(MEMO, 1, 11) is True  # 機密区分を含む冒頭部
    assert ss.classify_range(MEMO, 16, 24) is True  # 空欄のチェックリスト
    frontmatter = "---\ndoc_type: official_source\ntitles:\n  - 通知\n---\n\n# 本文\n本文です。\n"
    assert ss.classify_range(frontmatter, 1, 6) is True
    assert ss.classify_range(NOTICE, 16, 17) is False and ss.classify_range(NOTICE, 18, 19) is False
    assert ss.classify_range(NOTICE, 20, 33) is False  # 除外の注記を含む節は主要範囲
    checklist = "## 確認\n| 確認項目 | 結果 |\n|---|---|\n| 図面が添付されているか。 | |\n| 押印があるか。 | |\n"
    assert ss.classify_range(checklist, 1, 5) is False  # 文で書いた確認項目は実務の本文


def _fake_models(monkeypatch, prompts, answer="押印は原則不要。[E1]"):
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_json",
        lambda *a, **k: (
            {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
            if a[1] == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": [], "unresolved": []}
            if a[1] == deep_research.DEEP_GAPS_SYSTEM
            else {
                "subject": "押印",
                "scope": "会計手続",
                "conditions": [],
                "exceptions": [],
                "references": [],
            }
        ),
    )

    def text(*args, **kwargs):
        prompts.append(args[2])
        return answer

    monkeypatch.setattr(deep_research, "_call_ollama_text", text)


def _candidates(root, wanted):
    chunks = search.build_source_chunks(root)
    selected = []
    for i, (path, start) in enumerate(wanted):
        chunk = next(
            c for c in chunks if c["path"] == path and c["start_line"] <= start <= c["end_line"]
        )
        selected.append(
            {
                "path": path,
                "chunk_id": chunk["chunk_id"],
                "heading": chunk.get("heading", ""),
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "score": 2 - i * 0.1,
                "rrf_score": 0.5 - i * 0.01,
                "source_sha256": chunk["file_sha256"],
            }
        )
    return selected


def test_deep_keeps_main_items_when_ancillary_ranges_rank_first(tmp_path, monkeypatch):
    """発端の欠落（読取後に根拠枠を付随範囲に取られた）が再現しない。"""
    root = _root(tmp_path)
    # 付随範囲（事務連絡の冒頭・空欄チェックリスト、通知の冒頭）が先にヒットする。
    candidates = _candidates(
        root,
        [(MEMO_PATH, 3), (MEMO_PATH, 20), (NOTICE_PATH, 4), (NOTICE_PATH, 17), (NOTICE_PATH, 19)],
    )
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    # 主要範囲（約2,000字）は入るが、付随範囲を先に入れると溢れる枠にする。
    monkeypatch.setattr(deep_research, "DEEP_MAX_EVIDENCE_CHARS", 2100)
    prompts = []
    _fake_models(monkeypatch, prompts)
    result = deep_research.run_deep_research(
        "押印を求めない事に関する事務の内容詳細を教えてください",
        model="m",
        source_root=root,
        timeout_seconds=300,
    )
    assert any(K1 in p and K2 in p for p in prompts)
    main = [e for e in result.evidence if not e.get("ancillary")]
    assert main and all(e["verification_status"] != "raw_unverified" or e["path"] for e in main)
    ancillary_rows = [row for row in result.ledger if row.get("reason") == "ancillary_deferred"]
    raw = result.diagnostics["ancillary_ranges"]
    assert raw["raw"] + raw["deferred"] >= 1
    assert all(row["status"] == "deferred" for row in ancillary_rows)
    assert not any(item.startswith("ancillary") for item in result.unconfirmed)
    # 付随範囲は抽出・照合のモデル呼び出しを使わない。
    assert all(
        e["verification_status"] == "raw_unverified" for e in result.evidence if e.get("ancillary")
    )


def test_deep_reads_ancillary_ranges_when_the_question_asks_for_them(tmp_path, monkeypatch):
    root = _root(tmp_path)
    candidates = _candidates(root, [(NOTICE_PATH, 21)])
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    prompts = []
    _fake_models(monkeypatch, prompts, answer="令和３年１月１日から適用する。[E1]")
    result = deep_research.run_deep_research(
        "会計課長通知の適用日はいつですか",
        model="m",
        source_root=root,
        timeout_seconds=300,
    )
    assert result.diagnostics["ancillary_ranges"] == {"deferred": 0, "raw": 0}


def test_normal_answer_moves_ancillary_evidence_to_the_back(tmp_path):
    root = _root(tmp_path)
    sha = hashlib.sha256((root / NOTICE_PATH).read_bytes()).hexdigest()
    items = [
        {"path": NOTICE_PATH, "start_line": 1, "end_line": 9, "source_sha256": sha},
        {"path": NOTICE_PATH, "start_line": 16, "end_line": 17, "source_sha256": sha},
    ]
    ordered = search._demote_ancillary("押印の内容", items, source_root=root)
    assert [(i["start_line"], i.get("ancillary", False)) for i in ordered] == [
        (16, False),
        (1, True),
    ]
    assert search._demote_ancillary("照会先はどこですか", items, source_root=root) == items


# --- R2 発出者＋主題の単独候補 ------------------------------------------------------


def test_single_issuer_candidate_adds_primary_ranges_only_when_it_ranks_first(tmp_path):
    root = _root(tmp_path)
    chunks = search.build_source_chunks(root)
    query = "押印を求めない事に関する会計課長通知の内容詳細を教えてください"
    sha = hashlib.sha256((root / NOTICE_PATH).read_bytes()).hexdigest()
    top = [
        {"path": NOTICE_PATH, "start_line": 1, "end_line": 2, "source_sha256": sha, "score": 1.0}
    ]
    items, _ = search._with_primary_ranges(
        query, top, source_root=root, chunks=chunks, char_budget=8000, item_limit=5
    )
    primary = [i for i in items if i.get("primary_range")]
    assert primary and {i["primary_source"] for i in primary} == {"single_candidate"}
    assert any(K1 in i["snippet"] for i in primary) and items[-1] == top[0]
    other = [{"path": MEMO_PATH, "start_line": 14, "end_line": 14, "score": 1.0}]
    assert search._with_primary_ranges(
        query, other, source_root=root, chunks=chunks, char_budget=8000, item_limit=5
    ) == (other, [])
    specific = "押印を求めない事に関する会計課長通知の適用日はいつですか"
    assert search._with_primary_ranges(
        specific, top, source_root=root, chunks=chunks, char_budget=8000, item_limit=5
    ) == (top, [])


def test_overview_keeps_an_operative_date_when_primary_ranges_leave_space(tmp_path):
    root = _root(tmp_path)
    chunks = search.build_source_chunks(root)
    query = "会計手続における押印の見直しについての内容詳細"
    hit = {"path": NOTICE_PATH, "start_line": 1, "end_line": 2, "score": 1.0}
    full, _ = search._with_primary_ranges(
        query,
        [hit],
        source_root=root,
        chunks=chunks,
        char_budget=8000,
        item_limit=5,
    )
    full = search._demote_ancillary(query, full, source_root=root)
    kept = [item for item in full if item.get("primary_range")]
    assert any(K1 in item["snippet"] and K2 in item["snippet"] for item in kept)
    assert any(NOTICE_LINES[20] in item["snippet"] for item in kept)
    assert not any("令和２年12月25日" in item["snippet"] for item in kept)
    tight, _ = search._with_primary_ranges(
        query,
        [hit],
        source_root=root,
        chunks=chunks,
        char_budget=8000,
        item_limit=3,
    )
    assert any(K1 in item.get("snippet", "") and K2 in item["snippet"] for item in tight)
    assert all(NOTICE_LINES[20] not in item.get("snippet", "") for item in tight)


def test_deep_single_candidate_reads_main_sections_when_it_ranks_first(tmp_path, monkeypatch):
    root = _root(tmp_path)
    candidates = _candidates(root, [(NOTICE_PATH, 21)])
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    prompts = []
    _fake_models(monkeypatch, prompts)
    result = deep_research.run_deep_research(
        "押印を求めない事に関する会計課長通知の内容詳細を教えてください",
        model="m",
        source_root=root,
        timeout_seconds=300,
    )
    assert result.diagnostics["primary_source"] == "single_candidate"
    assert any(K1 in p and K2 in p for p in prompts)
    assert any(
        e.get("primary_operative_date") and e["excerpt"] == NOTICE_LINES[20]
        for e in result.evidence
    )


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("truncated", [True, False])
def test_v2_retains_operative_date_and_required_follow_through_from_original(
    tmp_path, monkeypatch, supported, truncated
):
    root = _root(tmp_path)
    original = (
        "書類の真正性が保たれていることを確認した上で、迅速かつ適切に手続を進めるよう努めること。"
    )
    notice = root / NOTICE_PATH
    notice.write_text(NOTICE.replace(K2, original), encoding="utf-8")
    candidates = _candidates(root, [(NOTICE_PATH, 21)])
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    _fake_models(monkeypatch, [])
    fake = deep_research._call_ollama_json
    checked = []

    def model(name, system, user, **kwargs):
        if system == pipeline.DRAFT_SYSTEM:
            if truncated:
                raise deep_research.DeepWorkerError(code=deep_research.MODEL_OUTPUT_INCOMPLETE_CODE)
            import re

            eid = next(
                match[1]
                for match in re.finditer(r"(?m)^\[(E\d+)\].*\n", user)
                if K1 in user[match.end() :].split("\n[", 1)[0]
            )
            date_id = next(
                match[1]
                for match in re.finditer(r"(?m)^\[(E\d+)\].*\n", user)
                if NOTICE_LINES[20] in user[match.end() :].split("\n[", 1)[0]
            )
            return {
                "sections": [
                    {
                        "heading": "押印",
                        "claims": [
                            {"text": K1, "evidence": [{"id": eid, "quote": K1}]},
                            {
                                "text": "この通知は令和3年1月1日から適用されます。",
                                "evidence": [{"id": date_id, "quote": NOTICE_LINES[20]}],
                            },
                        ],
                    }
                ]
            }
        if system == pipeline.CLAIM_VERIFY_SYSTEM:
            claims = json.loads(user)["claims"]
            checked.extend(claim["claim"] for claim in claims)
            return {
                "results": [
                    {
                        "claim_id": claim["claim_id"],
                        "verdict": "supported"
                        if supported or claim["claim"] == K1
                        else "unsupported",
                        "reason_code": "ok"
                        if supported or claim["claim"] == K1
                        else "not_in_source",
                    }
                    for claim in claims
                ]
            }
        assert system != pipeline.DRAFT_RETRY_SYSTEM
        return fake(name, system, user, **kwargs)

    monkeypatch.setattr(deep_research, "_call_ollama_json", model)
    result = deep_research.run_deep_research(
        "押印を求めない事に関する会計課長通知の内容詳細を教えてください",
        model="gpt-oss:120b",
        reasoning="high",
        source_root=root,
        timeout_seconds=1800,
        answer_policy="v2",
    )
    assert result.answer_state == deep_research.ANSWER_STATE_GENERATED
    assert original in checked and NOTICE_LINES[20] in checked
    assert (original in result.answer) is supported
    assert (NOTICE_LINES[20] in result.answer) is supported
    assert "適用されます" not in result.answer
    assert result.answer.count(K1) == 1
    assert ("原文に基づく回答" in result.answer) is truncated
    if truncated:
        assert (result.answer_quality["state"] == "passed") is supported


def test_multiple_operative_dates_are_not_automatically_selected():
    assert ss.unique_operative_date_range(NOTICE) == (21, "３ 実施時期")
    assert (
        ss.unique_operative_date_range(
            NOTICE + "\n## 施行日\n本通知は、令和３年２月１日から施行する。\n"
        )
        is None
    )


@pytest.mark.parametrize("supported", [True, False])
def test_deep_explicit_section_uses_only_selected_lines_and_still_verifies(
    tmp_path, monkeypatch, supported
):
    root = _root(tmp_path)
    prompts = []
    _fake_models(monkeypatch, prompts, answer=f"{K2}[E1]")
    original = deep_research._call_ollama_json
    verification = []

    def model(name, system, user, **kwargs):
        verdict = original(name, system, user, **kwargs)
        if (
            system == deep_research.DEEP_VERIFY_SYSTEM
            and json.loads(user)["verification_scope"] == "final_answer"
        ):
            verification.append(user)
            verdict["supported"] = supported
        return verdict

    monkeypatch.setattr(deep_research, "_call_ollama_json", model)

    def unexpected(*args, **kwargs):
        raise AssertionError("A selected section must not start a wider search")

    patch_retrieval(monkeypatch, deep_research, unexpected)
    result = deep_research.run_deep_research(
        "真正性の確保について説明してください",
        model="m",
        source_root=root,
        timeout_seconds=300,
        fixed_document={
            "path": NOTICE_PATH,
            "sha256": hashlib.sha256((root / NOTICE_PATH).read_bytes()).hexdigest(),
            "focus_ranges": [[18, 19, NOTICE_PATH]],
            "focus_only": True,
        },
    )
    expected = (
        deep_research.ANSWER_STATE_GENERATED
        if supported
        else deep_research.ANSWER_STATE_VERIFICATION_FAILED
    )
    assert result.answer_state == expected
    assert [(item["start_line"], item["end_line"]) for item in result.evidence] == [(18, 19)]
    assert K2 in prompts[0] and K1 not in prompts[0]
    assert len(verification) == 1 and K2 in verification[0] and K1 not in verification[0]


# --- R3 別紙・注記 -------------------------------------------------------------------


def test_appendix_and_exception_note_appear_in_overview_and_options():
    units = ss.primary_units(NOTICE)
    kinds = [
        (u["start_line"], u["end_line"], u["kind"], u["ancillary"])
        for u in units
        if u["start_line"] >= 20
    ]
    assert kinds == [
        (20, 22, "body", True),
        (23, 30, "appendix", False),
        (31, 31, "note", False),
        (33, 33, "note", True),
    ]
    overview = gf.build_overview(NOTICE)
    appendix = next(p for p in overview["points"] if "別紙" in p["heading"])
    assert appendix["text"].startswith("| 押印欄のある様式等 | 通知等の名称 |")
    assert appendix["notes"][0]["startLine"] == 31 and "除く" in appendix["notes"][0]["text"]
    labels = [o["label"] for o in overview["options"]]
    assert "別紙: 押印欄のある様式等 / 通知等の名称" in labels
    assert not any("照会" in (p.get("text") or "") for p in overview["points"])


# --- R4 抜粋の複数区間 ---------------------------------------------------------------


def test_excerpt_shows_separate_cited_lines_with_their_own_ranges():
    unit_lines = NOTICE.splitlines()[19:33] + ["| 追加 | " + "あ" * 300 + " |"] * 2
    unit = "\n".join(unit_lines)
    lines = [""] * 19 + unit_lines  # 行番号20から始まる根拠範囲
    answer = [
        "本通知は令和３年１月１日から適用する。[E5]",
        "本通知とは別の取組は除外され、疑義は監査指導室へ照会する。[E5]",
    ]
    selected = web_services.select_excerpt(unit, 20, focus=answer)
    assert len(selected["snippet"]) <= web_services.EVIDENCE_EXCERPT_MAX_CHARS
    segments = selected["segments"]
    assert len(segments) >= 2
    for segment in segments:
        expected = "\n".join(lines[segment["startLine"] - 1 : segment["endLine"]])
        assert segment["text"] == expected
    assert "適用する" in selected["snippet"] and "除く" in selected["snippet"]
    item = {}
    web_services._apply_excerpt(item, selected)
    assert item["snippetSegments"] == segments


# --- R5 分割ファイル ------------------------------------------------------------------


def test_split_parts_become_one_candidate_and_one_fixed_document(tmp_path):
    parts = {
        "tebiki/契約事務研修講義資料-resegmented-01.md": "# 契約事務研修講義資料\n## 総則\n契約の総則。\n",
        "tebiki/契約事務研修講義資料-resegmented-02.md": "## 随意契約\n随意契約の要件。\n",
    }
    root = _root(tmp_path, parts)
    chunks = search.build_source_chunks(root)
    documents = [d for d in ss.documents_from_chunks(chunks) if d["path"].startswith("tebiki/")]
    assert len(documents) == 1 and [m["path"] for m in documents[0]["paths"]] == sorted(parts)
    decision = gf.decide("契約事務研修講義資料の内容詳細", chunks, root)
    assert decision.kind == "topics"
    assert {p.get("path") for p in decision.overview["points"]} == set(parts)  # 部分ごとの概要
    assert all(
        o["label"].startswith("［1/2］")
        for o in decision.overview["options"]
        if o["kind"] == "section"
    )
    fixed = decision.document
    plan = gf.resolve_follow_up(
        fixed,
        gf.Topic(),
        decision.overview["options"],
        action=gf.ACTION_FOLLOW_UP,
        text="随意契約の要件",
    )
    assert (
        plan.ask is None
        and plan.focus_ranges[0][2] == "tebiki/契約事務研修講義資料-resegmented-02.md"
    )
    sha = {m["path"]: m["sha256"] for m in fixed["paths"]}
    forced = {
        "path": fixed["path"],
        "title": fixed["title"],
        "sha256": fixed["sha256"],
        "paths": [{"path": p, "sha256": h} for p, h in sha.items()],
        "focus_ranges": [list(r) for r in plan.focus_ranges],
        "primary": False,
    }
    items, _ = search._with_primary_ranges(
        "q", [], source_root=root, chunks=None, char_budget=4000, item_limit=5, forced=forced
    )
    assert [i["path"] for i in items] == ["tebiki/契約事務研修講義資料-resegmented-02.md"]
    # 1部分でも変わると固定資料として読めない。
    (root / "tebiki/契約事務研修講義資料-resegmented-02.md").write_text("変更\n", encoding="utf-8")
    assert search._with_primary_ranges(
        "q", [], source_root=root, chunks=None, char_budget=4000, item_limit=5, forced=forced
    ) == ([], [])


def test_deep_restricts_to_all_parts_of_a_fixed_split_document(tmp_path):
    parts = {
        "tebiki/資料-resegmented-01.md": "# 資料\n本文1。\n",
        "tebiki/資料-resegmented-02.md": "# 資料2\n本文2。\n",
    }
    root = _root(tmp_path, parts)
    sources, chunks = deep_research._snapshot(
        root.resolve(), deep_research.DeepBudget(300), lambda *a, **k: None
    )
    fixed = {
        "path": "tebiki/資料-resegmented-01.md",
        "sha256": "",
        "paths": [{"path": p, "sha256": sources[p][1]} for p in parts],
    }
    restricted, kept = deep_research._restrict_to_document(
        sources, chunks, fixed, lambda *a, **k: None
    )
    assert set(restricted) == set(parts) and {c["path"] for c in kept} == set(parts)
    batch = deep_research._fixed_candidates(restricted, {**fixed, "primary": True})
    assert {c["path"] for c in batch} == set(parts)


# --- R6 関連事務連絡 -----------------------------------------------------------------


def test_related_memo_citing_the_notice_number_is_offered_and_kept(tmp_path):
    root = _root(tmp_path)
    chunks = search.build_source_chunks(root)
    decision = gf.decide("会計手続における押印の見直しについての内容詳細", chunks, root)
    related = [o for o in decision.overview["options"] if o["kind"] == "related"]
    assert len(related) == 1 and related[0]["paths"][0]["path"] == MEMO_PATH
    assert related[0]["label"].startswith("関連資料（事務連絡")
    index = decision.overview["options"].index(related[0])
    plan = gf.resolve_follow_up(
        decision.document,
        gf.Topic(),
        decision.overview["options"],
        action=gf.ACTION_SELECT_TOPIC,
        choice=index,
    )
    assert plan.related["paths"][0]["path"] == MEMO_PATH and plan.overview is True
    store = gf.ConfirmationStore()
    decision.document["indexMarker"] = None
    record = store.create("s", "q", decision)
    store.begin(
        record.confirmation_id,
        "s",
        action=gf.ACTION_SELECT_TOPIC,
        options_version=record.options_version,
        request_id="r",
    )
    public = store.finish(
        record.confirmation_id, "r", add_paths=plan.related["paths"], completed=True
    )
    assert public["document"]["parts"] == 2


# --- 受入で見つかった問題の回帰（2026-11-08 R7） -------------------------------------


def test_nested_title_matches_confirm_the_longest_and_generic_cores_do_not():
    docs = ss.documents_from_texts(
        {
            "laws/毒物及び劇物取締法.md": "# 毒物及び劇物取締法\n本文\n",
            "laws/毒物及び劇物取締法施行令.md": "# 毒物及び劇物取締法施行令\n本文\n",
            "notice/一部改正.md": "# 一部改正について\n本文\n",
        }
    )
    assert (
        ss.match_documents("毒物及び劇物取締法施行令について詳しく", docs)["confirmed"]
        == "laws/毒物及び劇物取締法施行令.md"
    )
    assert (
        ss.match_documents("毒物及び劇物取締法について詳しく", docs)["confirmed"]
        == "laws/毒物及び劇物取締法.md"
    )
    assert ss.match_documents("一部改正について教えて", docs)["confirmed"] is None


def test_title_words_of_unconfirmed_candidates_are_not_asked_items(tmp_path):
    root = tmp_path / "skill-source"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a/銃砲の範囲を定める命令.md").write_text(
        "# 銃砲の範囲を定める命令\n本文。\n", encoding="utf-8"
    )
    (root / "b/銃砲の範囲を定める命令.md").write_text(
        "# 銃砲の範囲を定める命令\n別版の本文。\n", encoding="utf-8"
    )
    decision = gf.decide(
        "銃砲の範囲を定める命令について教えてください", search.build_source_chunks(root), root
    )
    assert decision.kind == "documents"  # 「範囲」は表題の語であり、尋ねている項目ではない


def test_deep_does_not_read_a_hit_range_already_read_as_primary(tmp_path, monkeypatch):
    root = _root(tmp_path)
    candidates = _candidates(root, [(NOTICE_PATH, 31)])
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda q, ch, **k: deep_candidate_selection.select_candidates(candidates, source_chunks=ch),
    )
    prompts = []
    _fake_models(monkeypatch, prompts)
    result = deep_research.run_deep_research(
        "会計手続における押印の見直しについての内容詳細",
        model="m",
        source_root=root,
        timeout_seconds=300,
    )
    ranges = [(e["start_line"], e["end_line"]) for e in result.evidence if e["path"] == NOTICE_PATH]
    assert (31, 31) in ranges and (20, 33) not in ranges
    # 読み残しの適用日・照会先は付随範囲として後回しになる（主要範囲と重ねて読まない）。
    assert not any(a <= 31 <= b for a, b in ranges if (a, b) != (31, 31))


def test_follow_up_leaves_the_selected_section_when_the_asked_item_is_elsewhere():
    """受入（R7）: 「１ 押印の見直し」選択中に「別紙の様式のうち除かれるものは？」で除外の注記を拾う。"""
    document = {
        "path": NOTICE_PATH,
        "title": "会計手続における押印の見直しについて（通知）",
        "sha256": "x",
        "text": NOTICE,
    }
    options = gf.build_overview(NOTICE)["options"]
    topic = gf.resolve_follow_up(
        document, gf.Topic(), options, action=gf.ACTION_SELECT_TOPIC, choice=0
    ).topic
    plan = gf.resolve_follow_up(
        document,
        topic,
        options,
        action=gf.ACTION_FOLLOW_UP,
        text="別紙の様式のうち除かれるものは？",
    )
    assert plan.ask is None
    assert any(start <= 31 <= end for start, end, _ in plan.focus_ranges)
