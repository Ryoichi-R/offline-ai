"""確定した資料の主要範囲・資料の照合・質問範囲の判定（guided-follow-up 段階1）。

発端事例（深掘りで通知の主要事項が回答と保存結果から抜けた）を合成資料で
段階ごとに切り分けた結果を回帰テストに固定する。正解の行範囲は、取得規則を
使わずに合成資料の原文から先に定める。
"""

import answer_policy as ap
import source_structure as ss

NOTICE_LINES = (
    "# 会計事務における押印の取扱いについて（通知）",  # 1
    "各部局会計課長 殿",  # 2
    "令和6年4月1日",  # 3
    "",  # 4
    "## 1 趣旨",  # 5
    "請求書等の押印を原則として求めないこととする。",  # 6
    "## 2 取扱い",  # 7
    "(1) 請求書は押印を省略できる。",  # 8
    "(2) 契約金額が100万円以上の場合は担当者の記名を必須とする。",  # 9
    "## 3 適用日",  # 10
    "令和6年4月1日から適用する。",  # 11
    "## 4 照会先",  # 12
    "会計課 契約係（内線1234）",  # 13
)
NOTICE = "\n".join(NOTICE_LINES)
# 原文から先に定めた正解: 主要事項は趣旨（5-6）と取扱い（7-9）。
EXPECTED_MAIN = {(5, 6), (7, 9)}


def test_title_wrapped_sections_are_units_and_ancillary_parts_are_demoted():
    units = ss.primary_units(NOTICE)
    main = {(u["start_line"], u["end_line"]) for u in units if not u["ancillary"]}
    ancillary = {(u["start_line"], u["end_line"]) for u in units if u["ancillary"]}
    assert main == EXPECTED_MAIN
    # 宛先・発出日の段落、単独の適用日、照会先は付随範囲。
    assert ancillary == {(2, 3), (10, 11), (12, 13)}


def test_effective_date_section_with_transitional_measure_stays_primary():
    text = NOTICE.replace(
        "令和6年4月1日から適用する。",
        "令和6年4月1日から適用する。ただし経過措置として従前の様式も使用できる。",
    )
    main = {(u["start_line"], u["end_line"]) for u in ss.primary_units(text) if not u["ancillary"]}
    assert (10, 11) in main


def test_budget_is_shared_per_section_and_unread_ranges_are_reported():
    long_body = "\n".join(f"本文{i:02d}" + "あ" * 40 for i in range(20))
    text = "# 表題\n## 前半\n" + long_body + "\n## 後半\n後半の主要事項。\n"
    selection = ss.primary_section_ranges(text, budget_chars=200)
    headings = [item["heading"] for item in selection["ranges"]]
    # 前半が長くても後半の先頭を読む（1つの節が予算を占有しない）。
    assert headings == ["前半", "後半"]
    assert selection["ranges"][1]["complete"] is True
    assert selection["ranges"][0]["complete"] is False
    assert selection["unread"] and selection["unread"][0]["heading"] == "前半"


def test_headingless_document_uses_paragraphs():
    text = "第一段落の本文。\n続き。\n\n第二段落の本文。\n"
    units = ss.primary_units(text)
    assert [(u["start_line"], u["end_line"]) for u in units] == [(1, 2), (4, 4)]
    assert all(not u["ancillary"] for u in units)


def test_nested_sections_read_each_body_once():
    text = "# 表題\n## 取扱い\n### 原則\n原則本文。\n### 例外\n例外本文。\n"
    unit = ss.primary_units(text)[0]
    assert unit["read_ranges"] == [(3, 4, "原則"), (5, 6, "例外")]


def _documents(**texts):
    return ss.documents_from_texts({f"{name}.md": text for name, text in texts.items()})


def test_title_variants_confirm_the_document():
    docs = _documents(
        a="# 会計事務における押印の取扱いについて（通知）\n本文", b="# 旅費の支給について\n本文"
    )
    result = ss.match_documents("会計事務における押印の取扱い の内容詳細", docs)
    assert result["confirmed"] == "a.md"
    # 全角・半角や括弧の表記ゆれを吸収する。
    assert (
        ss.match_documents("「会計事務における押印の取扱い」について", docs)["confirmed"] == "a.md"
    )


def test_same_issuer_or_same_title_does_not_confirm():
    docs = _documents(
        a="# 押印の取扱いについて（通知）\n会計課長\n",
        b="# 旅費の支給について（通知）\n会計課長\n",
    )
    # 発出者だけの一致では確定しない。
    assert ss.match_documents("会計課長の通知を教えて", docs)["confirmed"] is None
    # 本体と別添が同じ表題を持つ場合は、どちらにも確定しない。
    same = _documents(body="# 押印の取扱いについて\n本体", annex="# 押印の取扱いについて\n別添")
    result = ss.match_documents("押印の取扱いについて", same)
    assert result["confirmed"] is None
    assert len(result["candidates"]) == 2


def test_document_number_confirms():
    docs = _documents(a="# 通知\n会計第１２号\n本文", b="# 通知\n会計第13号\n本文")
    assert ss.match_documents("第12号の通知の内容", docs)["confirmed"] == "a.md"


def test_documents_from_chunks_uses_metadata_title_then_first_heading():
    chunks = [
        {"path": "a.md", "start_line": 1, "text": "# 見出し表題\n本文", "file_sha256": "x"},
        {"path": "b.md", "start_line": 1, "text": "本文のみ", "source_title": "メタデータ表題"},
        {"path": "c/d.md", "start_line": 1, "text": "本文のみ"},
    ]
    titles = {d["path"]: d["title"] for d in ss.documents_from_chunks(chunks)}
    assert titles == {"a.md": "見出し表題", "b.md": "メタデータ表題", "c/d.md": "d"}


def test_guided_scope_separates_breadth_from_granularity():
    broad = (
        "この通知の内容詳細を教えて",
        "会計課の押印の取扱いについて詳しく",
        "この通知のポイント",
    )
    specific = ("この通知の適用日はいつですか", "この通知の例外を詳しく", "期限は？", "適用日")
    for query in broad:
        assert ap.classify_guided_scope(query).scope == ap.GUIDED_SCOPE_BROAD, query
    for query in specific:
        assert ap.classify_guided_scope(query).scope == ap.GUIDED_SCOPE_SPECIFIC, query
    assert ap.classify_guided_scope("この通知の全体をまとめて").scope == ap.GUIDED_SCOPE_OVERVIEW


def test_title_words_are_not_counted_as_asked_items():
    title = "適用除外の手続について"
    assert (
        ap.classify_guided_scope(f"{title}の内容", exclude_texts=(title,)).scope
        == ap.GUIDED_SCOPE_BROAD
    )
    assert ap.classify_guided_scope(f"{title}の内容").scope == ap.GUIDED_SCOPE_SPECIFIC


def test_guided_cues_do_not_change_v2_aspect_cues():
    # 振り分け用の表は v2 の明示論点の表と独立している（v2 の判定・必須観点を変えない）。
    assert ap._ASPECT_CUES == (
        ("例外・除外", ("例外", "除外", "適用除外")),
        ("条件", ("条件", "要件")),
        ("比較", ("比較", "違い", "相違", "差異")),
        ("期限・時期", ("期限", "いつまで", "時期")),
        ("チェックリストの確認項目", ("チェックリスト", "チェック項目", "確認項目")),
    )
    intent = ap.extract_explicit_intent("この通知の適用日はいつですか")
    assert intent.required_aspects == ()


# --- 実資料（発端事例）と同じ構造の照合（合成資料） -------------------------------------

HEADER_NOTICE_LINES = (
    "> 会発1225第２号",  # 1
    "> 令和２年12月25日",  # 2
    "",  # 3
    "本省内部部局長",  # 4
    "",  # 5
    "施設等機関の長",  # 6
    "",  # 7
    "殿",  # 8
    "",  # 9
    "**大臣官房会計課長（公印省略）**",  # 10
    "",  # 11
    "# 会計手続における押印の見直しについて（通知）",  # 12
    "",  # 13
    "政府方針を踏まえ、会計手続における押印の見直しについて通知する。",  # 14
    "",  # 15
    "記",  # 16
    "",  # 17
    "## １ 押印の見直し",  # 18
    "会計手続における押印は原則不要とする。",  # 19
    "## ２ 真正性の確保",  # 20
    "本人確認や連絡内容の記録等により真正性を確認する。",  # 21
    "## ３ 実施時期",  # 22
    "本通知は、令和３年１月１日から適用する。",  # 23
)
HEADER_NOTICE = "\n".join(HEADER_NOTICE_LINES)
CITING_MEMO_LINES = (
    "事務連絡",
    "大臣官房会計課監査指導室長",
    "# 押印を前提としている契約関係書類の取扱いについて",
    "令和２年12月25日付け会発 1225第２号をもって通知されたところです。",
)
CITING_MEMO = "\n".join(CITING_MEMO_LINES)


def _corpus_documents():
    return ss.documents_from_texts(
        {
            "notice/会計課長通知（会計手続における押印の見直しについて）.md": HEADER_NOTICE,
            "notice/【監査指導室長事務連絡】押印を前提としている契約関係書類の取扱いについて.md": CITING_MEMO,
            "laws/会計法-resegmented-01.md": "# 会計法\n本文\n",
            "notice/n-37-再委託の適正化（周知）.md": "# 再委託の適正化について\n会計課長\n本文\n",
        },
        {
            "notice/会計課長通知（会計手続における押印の見直しについて）.md": "会計課長通知（会計手続における押印の見直しについて）"
        },
    )


def test_header_block_and_record_marker_are_ancillary():
    units = ss.primary_units(HEADER_NOTICE)
    main = [(u["start_line"], u["end_line"]) for u in units if not u["ancillary"]]
    assert main == [(14, 14), (18, 19), (20, 21)]
    # 「３ 実施時期」は単独の適用日なので付随範囲。
    assert (22, 23) in [(u["start_line"], u["end_line"]) for u in units if u["ancillary"]]


def test_bracketed_formal_title_and_issuer_phrase():
    variants = ss.title_variants(["会計課長通知（会計手続における押印の見直しについて）"])
    assert "会計手続における押印の見直し" in variants
    assert "会計課長通知" not in variants  # 発出者と種別だけの句は表題の中核にしない
    docs = _corpus_documents()
    confirmed = ss.match_documents("会計手続における押印の見直しについての内容詳細", docs)[
        "confirmed"
    ]
    assert confirmed.startswith("notice/会計課長通知")


def test_issuer_and_subject_find_a_candidate_without_confirming():
    result = ss.match_documents(
        "押印を求めない事に関する会計課長通知の内容詳細を教えてください", _corpus_documents()
    )
    assert result["confirmed"] is None
    assert [c["path"] for c in result["candidates"]] == [
        "notice/会計課長通知（会計手続における押印の見直しについて）.md"
    ]
    assert result["candidates"][0]["reasons"] == ["issuer_subject"]
    # 発出者だけでは候補にも確定にもしない（「会計法」の表題の部分一致も拾わない）。
    assert ss.match_documents("会計課長通知の内容を教えて", _corpus_documents())["candidates"] == []


def test_subject_terms_alone_list_candidates():
    result = ss.match_documents("押印の見直しについて詳しく", _corpus_documents())
    assert result["confirmed"] is None
    assert "notice/会計課長通知（会計手続における押印の見直しについて）.md" in [
        c["path"] for c in result["candidates"]
    ]


def test_full_document_number_uses_the_header_not_citations():
    docs = _corpus_documents()
    result = ss.match_documents("会発1225第２号の内容詳細", docs)
    assert result["confirmed"] == "notice/会計課長通知（会計手続における押印の見直しについて）.md"
    assert ss.document_numbers(CITING_MEMO.split("# ")[0]) == set()
    assert ("会発1225", "2") in ss.document_numbers("会発 1225 第２号")


def test_split_file_suffix_is_not_part_of_the_title():
    assert ss.title_variants(["会計法-resegmented-01"]) == ["会計法"]
    assert ss.title_variants(["n-37-再委託の適正化（周知）"]) == ["再委託の適正化"]
