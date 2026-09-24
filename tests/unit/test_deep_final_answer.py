"""Final-answer citation checks, notation normalization, and one rewrite; no live model or corpus.

2026-09-23 real-corpus run (1800s): the core document was read, but the final
answer failed the citation check. The final prompt told the model to put
unverifiable points into a "未確認事項" section, while the check requires a
citation on every non-heading line, so such a section could never pass.
"""

import json

import pytest

import deep_research as deep
import prompt_templates
from test_deep_remediation import backend, research


def _text_calls(monkeypatch, *answers):
    calls = []

    def text(model, system, prompt, **kwargs):
        calls.append(prompt)
        return answers[min(len(calls), len(answers)) - 1]

    monkeypatch.setattr(deep, "_call_ollama_text", text)
    return calls


# --- notation and section handling ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("条件A。[E8, E6]", "条件A。[E8][E6]"),
        ("条件A。[E8、E6]", "条件A。[E8][E6]"),
        ("条件A。[E8・E6]", "条件A。[E8][E6]"),
        ("条件A。【E2】", "条件A。[E2]"),
        ("条件A。［Ｅ３］", "条件A。[E3]"),
        ("条件A。[E1]", "条件A。[E1]"),
    ],
)
def test_citation_notation_variants_are_normalized(raw, expected):
    assert deep._normalize_citations(raw)[0] == expected


def test_normalization_never_invents_or_drops_ids():
    normalized, count = deep._normalize_citations("A。[E1, E2] B。[E3]")
    assert normalized == "A。[E1][E2] B。[E3]"
    assert count == 1
    # Non-citation brackets are left alone.
    assert deep._normalize_citations("別紙[様式]を使う。[E1]")[0] == "別紙[様式]を使う。[E1]"


@pytest.mark.parametrize(
    "answer",
    [
        "## 付議条件\n条件A。[E1]\n## 未確認事項\n- 審査時期は不明\n- 別紙は未読\n## 除外\n除外B。[E2]",
        "## 付議条件\n条件A。[E1]\n## 除外\n除外B。[E2]\n**未確認事項**：\n- 審査時期は不明",
        "## 付議条件\n条件A。[E1]\n## 除外\n除外B。[E2]\n### 未確認事項（原文で確認できなかった点）\n- 不明",
        "## 付議条件\n条件A。[E1]\n## 除外\n除外B。[E2]\n未確認事項:\n- 不明",
    ],
)
def test_model_written_unconfirmed_section_is_removed_and_the_rest_kept(answer):
    stripped, dropped = deep._strip_unconfirmed_section(answer)
    assert "不明" not in stripped and "未読" not in stripped
    assert "条件A。[E1]" in stripped and "除外B。[E2]" in stripped
    assert dropped >= 2
    assert deep._validate_final_answer(stripped, {"E1", "E2"})


def test_a_cited_sentence_that_mentions_unconfirmed_items_is_not_a_section():
    answer = "未確認事項のうち審査時期は第4条にある。[E1]"
    assert deep._strip_unconfirmed_section(answer) == (answer, 0)


def test_final_prompt_no_longer_asks_for_an_unconfirmed_section():
    assert "「未確認事項」に分けて" not in prompt_templates.DEEP_FINAL_SYSTEM
    assert "未確認事項の欄は書かないでください" in prompt_templates.DEEP_FINAL_SYSTEM


def test_issue_lines_distinguish_missing_and_unknown_ids():
    issues = deep._final_answer_issues("条件A。[E1]\n条件B。\n条件C。[E9]", {"E1"})
    assert issues == {"uncited_lines": [2], "unknown_id_lines": [3], "has_citation": True}


# --- end to end with fake model ---------------------------------------------------


def test_answer_with_unconfirmed_section_is_accepted_without_a_rewrite(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = _text_calls(monkeypatch, "対象である。[E1]\n## 未確認事項\n- 別紙は読めていない")
    result = research(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert result.answer == "対象である。[E1]"
    assert len(calls) == 1
    final = result.diagnostics["final_answer"]
    assert final["result"] == "accepted"
    assert final["attempts"] == [
        {
            "normalized_citations": 0,
            "dropped_unconfirmed_lines": 2,
            "uncited_lines": 0,
            "unknown_id_lines": 0,
            "limitation_lines": 0,
            "enumeration_gaps": 0,
        }
    ]


def test_one_rewrite_with_failing_line_numbers_can_pass(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = _text_calls(monkeypatch, "対象である。[E1]\n根拠のない補足。", "対象である。[E1]")
    result = research(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert len(calls) == 2
    assert "2行目" in calls[1] and "前回の回答案" in calls[1] and "根拠のない補足。" in calls[1]
    final = result.diagnostics["final_answer"]
    assert final["result"] == "accepted"
    assert [a["uncited_lines"] for a in final["attempts"]] == [1, 0]


def test_rewrite_that_still_fails_is_rejected_for_citations(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = _text_calls(monkeypatch, "根拠のない断定。", "また根拠のない断定。")
    result = research(tmp_path)

    assert len(calls) == 2
    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert result.diagnostics["final_answer"]["result"] == "rejected_citations"
    assert any("根拠IDが不足または不正" in item for item in result.unconfirmed)


def test_no_rewrite_when_too_little_time_remains(tmp_path, monkeypatch):
    backend(monkeypatch)
    monkeypatch.setattr(deep, "DEEP_FINAL_RETRY_MIN_SECONDS", 10_000)
    calls = _text_calls(monkeypatch, "根拠のない断定。", "対象である。[E1]")
    result = research(tmp_path)

    assert len(calls) == 1
    assert result.diagnostics["final_answer"]["result"] == "rejected_citations"


def test_rewrite_timeout_keeps_the_citation_failure_instead_of_not_generated(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = []

    def text(model, system, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise TimeoutError("deep worker deadline")
        return "根拠のない断定。"

    monkeypatch.setattr(deep, "_call_ollama_text", text)
    result = research(tmp_path)

    assert len(calls) == 2
    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert result.diagnostics["final_answer"]["result"] == "rejected_citations"


def test_rewrite_cancellation_still_cancels(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = []

    class Cancelled(Exception):
        code = "cancelled"

    def text(model, system, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise Cancelled()
        return "根拠のない断定。"

    monkeypatch.setattr(deep, "_call_ollama_text", text)
    result = research(tmp_path)

    assert result.status == "cancelled"


def test_final_answer_diagnostics_hold_numbers_only(tmp_path, monkeypatch):
    backend(monkeypatch)
    _text_calls(monkeypatch, "対象である。[E1]\n秘密の補足文。", "対象である。[E1]")
    result = research(tmp_path)

    serialized = json.dumps(result.diagnostics["final_answer"], ensure_ascii=False)
    assert "秘密の補足文" not in serialized and "対象である" not in serialized
    for attempt in result.diagnostics["final_answer"]["attempts"]:
        assert all(isinstance(value, int) for value in attempt.values())


# --- limits (parenthetical and scope) must survive -----------------------------------
# Synthetic article modelled on the structure that failed on 2026-09-23: a list of
# referral items with parenthetical limits, and an exclusion list whose lead limits
# it to items 1 and 2.

ARTICLE = "\n".join(
    [
        "第３条　委員会において審査する案件は、次のとおりとする。",
        "",
        "- 一　概算所要見込額２，０００万円以上の競争入札",
        "- 二　概算所要見込額５００万円以上の随意契約（公募によるものを含み、不調・不落による随意契約を除く。この項において同じ。）",
        "- 三　企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件。）",
        "- 四　随意契約で当初契約を行った案件を増額する変更契約であって、変更後の契約金額が第２号の金額以上となるもの",
        "",
        "２　前項第１号又は第２号の規定に関わらず、次の各号に該当する案件は審査案件としない。",
        "",
        "- 一　予算決算及び会計令第９９条第１号による随意契約（秘密随意契約）",
        "- 二　庁舎で使用する電力の供給及びガスの供給に関する一般競争入札",
    ]
)
EVIDENCE = [{"evidence_id": "E1", "excerpt": ARTICLE}]
FAITHFUL = "\n".join(
    [
        "概算所要見込額2,000万円以上の競争入札は審査する。[E1]",
        "概算所要見込額500万円以上の随意契約（公募によるものを含み、不調・不落による随意契約を除く）は審査する。[E1]",
        "企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）は審査する。[E1]",
        "随意契約で当初契約を行った案件を増額する変更契約で、変更後の契約金額が第2号の金額以上となるものは審査する。[E1]",
        "前項第1号又は第2号の規定に関わらず、予算決算及び会計令第99条第1号による随意契約（秘密随意契約）は審査しない。[E1]",
        "庁舎で使用する電力の供給及びガスの供給に関する一般競争入札は審査しない。[E1]",
    ]
)


def test_faithful_answer_keeps_every_limit():
    assert deep._limitation_issues(FAITHFUL, EVIDENCE) == []


def test_dropped_parenthetical_limit_is_reported_on_its_line():
    answer = FAITHFUL.replace("企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）", "企画競争")
    issues = deep._limitation_issues(answer, EVIDENCE)
    assert [line for line, _ in issues] == [3]
    assert "新規案件及び前回一者応募の案件" in issues[0][1][0]


def test_dropped_exclusion_scope_is_reported_once():
    answer = FAITHFUL.replace("前項第1号又は第2号の規定に関わらず、", "")
    issues = deep._limitation_issues(answer, EVIDENCE)
    assert issues == [(5, ["前項第1号又は第2号の規定に関わらず"])]


def test_neighbouring_unlimited_item_is_not_blamed_for_a_limited_one():
    # The fourth item has no limit; it must not be matched to the limited second item.
    answer = "随意契約で当初契約を行った案件を増額する変更契約で、変更後の契約金額が第2号の金額以上となるものは審査する。[E1]"
    assert deep._limitation_issues(answer, EVIDENCE) == []


def test_alias_parenthesis_and_uncited_lines_are_not_limits():
    answer = "予算決算及び会計令第99条第1号による随意契約は、前項第1号又は第2号の規定に関わらず審査しない。[E1]"
    assert deep._limitation_issues(answer, EVIDENCE) == []
    # Only lines citing the evidence are matched against it.
    assert deep._limitation_issues("企画競争は審査する。[E2]", EVIDENCE) == []


def _article_backend(monkeypatch, tmp_path):
    backend(monkeypatch)
    (tmp_path / "source.md").write_text(ARTICLE, encoding="utf-8")
    monkeypatch.setattr(deep, "_split_line_range", lambda lines, start, end, **kw: [
        {"start_line": 1, "end_line": len(ARTICLE.splitlines()), "char_start": None, "char_end": None,
         "text": ARTICLE}
    ] if start == 1 else [])


def _run_article(tmp_path):
    return deep.run_deep_research("対象", model="fake", source_root=tmp_path, timeout_seconds=300)


def test_limit_dropped_then_restored_by_the_rewrite(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    dropped = FAITHFUL.replace("企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）", "企画競争")
    calls = _text_calls(monkeypatch, dropped, FAITHFUL)
    result = _run_article(tmp_path)

    assert len(calls) == 2
    assert "3行目で根拠の限定が抜けています" in calls[1]
    assert "新規案件及び前回一者応募の案件" in calls[1]
    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert [a["limitation_lines"] for a in result.diagnostics["final_answer"]["attempts"]] == [1, 0]


def test_limit_still_dropped_after_rewrite_is_repaired_with_source_text(tmp_path, monkeypatch):
    """2026-09-23 three runs: every run was discarded for two dropped limits. The
    remaining lines are now replaced by the cited source text instead."""
    _article_backend(monkeypatch, tmp_path)
    dropped = FAITHFUL.replace("前項第1号又は第2号の規定に関わらず、", "").replace(
        "企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）", "企画競争"
    )
    _text_calls(monkeypatch, dropped, dropped)
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert "三　企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件。）[E1]" in result.answer
    assert "２　前項第１号又は第２号の規定に関わらず、次の各号に該当する案件は審査案件としない。[E1]" in result.answer
    final = result.diagnostics["final_answer"]
    assert final["result"] == "accepted"
    assert final["repaired_lines"] == 2
    repaired_notes = [item for item in result.unconfirmed if item.startswith("limitation_repaired")]
    assert any("3行目は原文の限定（前号に該当するもののほか" in item for item in repaired_notes)
    assert any("前項第1号又は第2号の規定に関わらず" in item for item in repaired_notes)
    # A repair note does not change why exploration stopped.
    assert result.stop_reason != "limitation_repaired"


def test_unrepairable_limit_is_rejected_and_names_the_source_limit(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    dropped = FAITHFUL.replace("企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）", "企画競争")
    _text_calls(monkeypatch, dropped, dropped)
    monkeypatch.setattr(deep, "_repair_limitations", lambda answer, findings: (answer, []))
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert result.diagnostics["final_answer"]["result"] == "rejected_limitations"
    detail = next(item for item in result.unconfirmed if "根拠の限定（括弧書き・限定の文言・適用範囲）が欠けた" in item)
    assert "欠けた原文の限定: 前号に該当するもののほか、新規案件及び前回一者応募の案件。" in detail
    assert "原文抜粋" in result.answer


@pytest.mark.parametrize(
    "answer",
    [
        "競争入札（不調・不落による随意契約を含む。）で当初契約を行った案件は審査する。[E1]",
        "随意契約（公募によるものを含み、不調・不落による随意契約を除く。この項において同じ。）は審査する。[E1]",
        "契約（別紙（様式。）を含む。）は審査する。[E1]",
    ],
)
def test_sentence_stop_inside_parentheses_does_not_split_a_claim(answer):
    assert deep._final_answer_issues(answer, {"E1"})["uncited_lines"] == []


def test_uncited_sentence_after_parentheses_is_still_caught():
    answer = "競争入札（不調・不落による随意契約を含む。）は審査する。[E1] 企画競争も審査する。"
    assert deep._final_answer_issues(answer, {"E1"})["uncited_lines"] == [1]


def test_verbatim_source_text_is_cited_after_every_sentence():
    text = "２　前項の規定に関わらず、次の案件は審査しない（公募を含む。）。なお書類は別途提出する。"
    cited = deep._cite_verbatim(text, "E3")
    assert cited == "２　前項の規定に関わらず、次の案件は審査しない（公募を含む。）。[E3]なお書類は別途提出する。[E3]"
    assert deep._validate_final_answer(cited, {"E3"})


def test_final_prompt_asks_to_keep_limits_verbatim():
    assert "原文の文言のまま書いてください" in prompt_templates.DEEP_FINAL_SYSTEM
    assert "前項第○号の規定に関わらず" in prompt_templates.DEEP_FINAL_SYSTEM


# --- units read but judged irrelevant are visible --------------------------------------


def test_irrelevant_units_are_listed_in_diagnostics(tmp_path, monkeypatch):
    backend(
        monkeypatch,
        extract=lambda user: {
            "subject": "無関係",
            "scope": "",
            "relevance": "irrelevant",
            "conditions": [],
            "exceptions": [],
            "references": [],
        },
    )
    result = research(tmp_path)

    assert result.evidence == []
    assert result.diagnostics["irrelevant_ranges"] == [{"path": "source.md", "start_line": 1, "end_line": 1}]


def test_public_ledger_keeps_irrelevant_entries_ahead_of_bulk_exclusions():
    import web_services

    ledger = [
        {"path": f"doc-{i}.md", "reason": "candidate_limit", "status": "unconfirmed", "start_line": 1, "end_line": 2}
        for i in range(web_services.DEEP_PUBLIC_LEDGER_MAX + 50)
    ] + [{"path": "core.md", "reason": "irrelevant", "status": "irrelevant", "start_line": 110, "end_line": 117}]
    result = deep.DeepResearchResult(status="partial", stop_reason="evidence_budget", ledger=ledger)

    event = web_services.build_deep_evidence_event(result)

    assert event["deepLedger"][0]["reason"] == "irrelevant"
    assert event["deepLedger"][0]["start_line"] == 110
    assert event["deepLedgerTruncated"] is True


# --- nested parentheses: a limit outside an inner definition -------------------------
# Modelled on 第1条 (2026-09-23 runs 2 and 3 overgeneralized 基金事業).

ARTICLE_ONE = (
    "第１条　厚生労働省における調達（法人等に国庫補助金等を交付して設置造成された基金により実施する事業"
    "（以下「基金事業」という。）等であって、当該事業等の委託先の選定を厚生労働省で行う場合を含む。以下同じ。）"
    "について、調達実施前に案件の審査を行う。"
)
ARTICLE_ONE_EVIDENCE = [{"evidence_id": "E5", "excerpt": ARTICLE_ONE}]


def test_limit_outside_a_nested_definition_is_required():
    answer = "基金事業（法人等に国庫補助金等を交付して設置造成された基金により実施する事業）は諮問対象とする。[E5]"
    issues = deep._limitation_issues(answer, ARTICLE_ONE_EVIDENCE)
    assert [line for line, _ in issues] == [1]
    assert "委託先の選定を厚生労働省で行う場合を含む" in issues[0][1][0]


def test_answer_keeping_the_nested_limit_passes():
    answer = (
        "調達には、法人等に国庫補助金等を交付して設置造成された基金により実施する事業等であって、"
        "当該事業等の委託先の選定を厚生労働省で行う場合を含む。[E5]"
    )
    assert deep._limitation_issues(answer, ARTICLE_ONE_EVIDENCE) == []


def test_nested_limit_line_is_repaired_with_the_whole_source_line():
    answer = "基金事業（法人等に国庫補助金等を交付して設置造成された基金により実施する事業）は諮問対象とする。[E5]"
    repaired, notes = deep._repair_limitations(answer, deep._limitation_findings(answer, ARTICLE_ONE_EVIDENCE))
    assert repaired.startswith("第１条　厚生労働省における調達（")
    assert deep._validate_final_answer(repaired, {"E5"})
    assert deep._limitation_issues(repaired, ARTICLE_ONE_EVIDENCE) == []
    assert "委託先の選定を厚生労働省で行う場合を含む" in notes[0]


def test_only_the_outermost_limiting_parenthesis_is_taken():
    text = "契約（公募を含む（別紙の様式のみ）。）"
    spans = deep._limiting_spans(text)
    assert [(text[start : end + 1]) for start, end, _ in spans] == ["（公募を含む（別紙の様式のみ）。）"]
    # An alias-only nested parenthesis is not a limit.
    assert deep._limiting_spans("随意契約（秘密随意契約（予決令））") == []


# --- per-line support check -----------------------------------------------------------


def _line_verdicts(monkeypatch, unsupported_lines, *, background=(), verdict=None):
    """Fake verifier: per-line claims listed in `unsupported_lines` fail and those in
    `background` restate background; all else passes. `verdict` overrides the reply."""
    seen = []

    def model(name, system, user, **kwargs):
        if system == deep.DEEP_LINE_VERIFY_SYSTEM:
            claims = json.loads(user)["claims"]
            seen.append(claims)
            if verdict is not None:
                return verdict
            return {
                "source_sentence": "",
                "source_kind": "background" if any(fragment in claims for fragment in background) else "answer",
                "source_actor": "委員会",
                "claim_actor": "委員会",
                "supported": not any(fragment in claims for fragment in unsupported_lines),
            }
        if system == deep.DEEP_VERIFY_SYSTEM:
            return {"supported": True, "contradictions": [], "missing_conditions": [], "unsupported_claims": []}
        if system == deep.DEEP_GAPS_SYSTEM:
            return {"queries": [], "unresolved": []}
        return {"subject": "対象", "scope": "範囲", "conditions": ["対象である"], "exceptions": [], "references": []}

    monkeypatch.setattr(deep, "_call_ollama_json", model)
    return seen


def test_unsupported_line_is_replaced_by_the_source_line_it_restates(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    overstated = FAITHFUL.replace(
        "庁舎で使用する電力の供給及びガスの供給に関する一般競争入札は審査しない。[E1]",
        "庁舎で使用する電力の供給及びガスの供給に関する一般競争入札は常に審査しない。[E1]",
    )
    _text_calls(monkeypatch, overstated)
    seen = _line_verdicts(monkeypatch, ["常に審査しない"])
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert "常に審査しない" not in result.answer
    assert "二　庁舎で使用する電力の供給及びガスの供給に関する一般競争入札[E1]" in result.answer
    checks = result.diagnostics["final_answer"]["line_checks"]
    assert checks == {"checked": 6, "unsupported": 1, "replaced": 1, "dropped": 0, "off_question": 0, "unchecked": 0}
    assert len(seen) == 6
    assert any(item.startswith("line_unsupported") and "6行目" in item for item in result.unconfirmed)
    assert result.stop_reason != "line_unsupported"


def test_unsupported_line_without_a_matching_source_line_is_dropped(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    _text_calls(monkeypatch, FAITHFUL + "\n審査は迅速に行われる。[E1]")
    _line_verdicts(monkeypatch, ["迅速に"])
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert "迅速に" not in result.answer
    assert result.diagnostics["final_answer"]["line_checks"]["dropped"] == 1


def test_verbatim_source_lines_are_not_rechecked(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    verbatim = "二　庁舎で使用する電力の供給及びガスの供給に関する一般競争入札[E1]"
    _text_calls(monkeypatch, FAITHFUL + "\n" + verbatim)
    seen = _line_verdicts(monkeypatch, [])
    result = _run_article(tmp_path)

    assert all("二　庁舎" not in claim for claim in seen)
    assert result.diagnostics["final_answer"]["line_checks"]["checked"] == 6


def test_line_checks_stop_to_keep_time_for_the_whole_answer_check(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(deep, "DEEP_LINE_CHECK_RESERVE_SECONDS", 10_000)
    _text_calls(monkeypatch, FAITHFUL)
    seen = _line_verdicts(monkeypatch, [])
    result = _run_article(tmp_path)

    assert seen == []
    assert result.diagnostics["final_answer"]["line_checks"]["unchecked"] == 6
    assert result.answer_state == deep.ANSWER_STATE_GENERATED


def test_answer_with_every_line_dropped_is_rejected(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    _text_calls(monkeypatch, "審査は迅速に行われる。[E1]")
    _line_verdicts(monkeypatch, ["迅速に"])
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert result.diagnostics["final_answer"]["result"] == "rejected_support"
    assert "原文抜粋" in result.answer


def test_line_check_diagnostics_hold_numbers_only(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    _text_calls(monkeypatch, FAITHFUL)
    _line_verdicts(monkeypatch, ["企画競争"])
    result = _run_article(tmp_path)

    checks = result.diagnostics["final_answer"]["line_checks"]
    assert set(checks) == {"checked", "unsupported", "replaced", "dropped", "off_question", "unchecked"}
    assert all(isinstance(value, int) for value in checks.values())


def test_line_restating_background_is_dropped_even_when_a_source_line_matches(tmp_path, monkeypatch):
    """2026-09-24 run 1: a line turned the notice's background into a condition."""
    _article_backend(monkeypatch, tmp_path)
    background = "二　庁舎で使用する電力の供給及びガスの供給に関する一般競争入札が審査の目的である。[E1]"
    _text_calls(monkeypatch, FAITHFUL + "\n" + background)
    _line_verdicts(monkeypatch, [], background=["審査の目的である"])
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert "審査の目的である" not in result.answer
    checks = result.diagnostics["final_answer"]["line_checks"]
    assert (checks["off_question"], checks["dropped"], checks["replaced"]) == (1, 1, 0)
    assert any(item.startswith("line_unsupported") and "背景" in item for item in result.unconfirmed)


def test_line_naming_another_actor_is_unsupported(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    _text_calls(monkeypatch, FAITHFUL)
    verdict = {"source_sentence": "", "source_kind": "answer", "source_actor": "特別会計の勘定元",
               "claim_actor": "委員会", "supported": True}
    _line_verdicts(monkeypatch, [], verdict=verdict)
    result = _run_article(tmp_path)

    checks = result.diagnostics["final_answer"]["line_checks"]
    assert checks["unsupported"] == checks["checked"] == 6
    assert checks["off_question"] == 0


def test_line_verdict_reads_kind_actors_and_support():
    base = {"source_sentence": "", "source_kind": "answer", "source_actor": "委員会",
            "claim_actor": "厚生労働省の公共調達委員会", "supported": True}
    assert deep._line_verdict(base) == deep.LINE_SUPPORTED
    assert deep._line_verdict({**base, "claim_actor": ""}) == deep.LINE_SUPPORTED
    assert deep._line_verdict({**base, "source_actor": "大臣"}) == deep.LINE_UNSUPPORTED
    assert deep._line_verdict({**base, "supported": False}) == deep.LINE_UNSUPPORTED
    assert deep._line_verdict({**base, "source_kind": "background"}) == deep.LINE_OFF_QUESTION
    # No source sentence found: the kind is left empty.
    assert deep._line_verdict({**base, "source_kind": "", "supported": False}) == deep.LINE_UNSUPPORTED
    broken_verdicts = (
        {**base, "source_kind": "rule"},
        {**base, "source_kind": ""},
        {**base, "supported": "true"},
        {**base, "source_actor": None},
    )
    for broken in broken_verdicts:
        with pytest.raises(ValueError):
            deep._line_verdict(broken)


def test_malformed_line_verdict_leaves_the_line_unchecked(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    _text_calls(monkeypatch, FAITHFUL)
    _line_verdicts(monkeypatch, [], verdict={"supported": True})
    result = _run_article(tmp_path)

    checks = result.diagnostics["final_answer"]["line_checks"]
    assert (checks["checked"], checks["unchecked"]) == (0, 6)
    assert result.answer_state == deep.ANSWER_STATE_GENERATED


def test_line_check_prompt_asks_for_separate_readings():
    for key in ("source_sentence", "source_kind", "source_actor", "claim_actor", "supported"):
        assert key in prompt_templates.DEEP_LINE_VERIFY_SYSTEM

# --- limits outside parentheses; enumerations used in part ---------------------------

# n-42 as resegmented: 記１ and 記２ are whole paragraphs on heading lines, items inline.
NOTICE_42 = "\n".join(
    [
        "###### １次に掲げる随意契約については、予定価格調書その他の書面による予定価格の積算を省略し、"
        "又は見積書の徴取を省略してさしつかえないこととする。 ⑴ 法令に基づいて取引価格（料金）が定められて"
        "いることその他特別の事由があることにより、特定の取引価格（料金）によらなければ契約をすることが不可能"
        "又は著しく困難であると認められるものに係る随意契約 ⑵ 予定価格が２５０万円をこえない随意契約で、"
        "各省各庁における契約事務の実情を勘案し、各省各庁の長において契約担当官等が予定価格調書その他の書面"
        "による予定価格の積算を省略し、又は見積書の徴取を省略しても支障がないと認めるもの",
        "",
        "###### ２上記１により処理することとした場合においても、次に掲げる措置を講じ、契約事務の適正化を図る"
        "ものとする。 ⑴ 契約担当官等は、予定価格調書その他の書面による予定価格の積算を省略することとした場合"
        "においても、必要に応じ、補助職員をしてあらかじめ書面による予定価格の積算を行なわせ、その積算資料を"
        "当該契約に係る決議書に添付させるよう措置するものとする。 ⑵ 契約担当官等は、見積書の徴取を省略する"
        "こととした場合においても、必要に応じ、補助職員をして口頭照会による見積り合せ、又は市場価格調査等を"
        "行なわせ、その結果を記載した資料を当該契約に係る決議書に添付させるよう措置するものとする。",
    ]
)
NOTICE_42_EVIDENCE = [{"evidence_id": "E8", "excerpt": NOTICE_42}]
KIND_ONE = (
    "法令に基づき取引価格（料金）が定められていることやその他特別な事由があり、特定の取引価格（料金）に"
    "よらなければ契約が不可能又は著しく困難であると認められる随意契約では、予定価格調書または見積書徴取を省略できる。[E8]"
)
KIND_TWO = (
    "予定価格が250万円以下の随意契約においては、各省各庁の長において契約担当官等が予定価格の積算を省略し、"
    "又は見積書の徴取を省略しても支障がないと認めた場合、省略できる。[E8]"
)
MEASURE_ONE = (
    "予定価格調書の作成を省略する場合には、必要に応じ補助職員をしてあらかじめ書面による予定価格の積算を行わせ、"
    "その積算資料を当該契約に係る決議書に添付させるよう措置する。[E8]"
)
MEASURE_TWO = (
    "見積書徴取を省略する場合には、必要に応じ補助職員をして口頭照会による見積り合せ又は市場価格調査等を行わせ、"
    "その結果を記載した資料を当該契約に係る決議書に添付させるよう措置する。[E8]"
)
COMPLETE_42 = "\n".join([KIND_ONE, KIND_TWO, MEASURE_ONE, MEASURE_TWO])


def test_paragraph_on_a_heading_line_is_split_into_its_inline_items():
    rows = deep._source_limits(NOTICE_42_EVIDENCE)
    items = [row for row in rows if row["group"] is not None]
    assert [row["raw"][:1] for row in items] == ["⑴", "⑵", "⑴", "⑵"]
    assert len({row["group"] for row in items}) == 2
    assert rows[0]["group"] is None and rows[0]["raw"].startswith("１次に掲げる随意契約")


def test_complete_answer_keeps_limits_outside_parentheses_and_every_item():
    assert deep._limitation_issues(COMPLETE_42, NOTICE_42_EVIDENCE) == []
    assert deep._enumeration_issues(COMPLETE_42, NOTICE_42_EVIDENCE) == []


def test_dropped_qualifier_outside_parentheses_is_reported():
    """2026-09-24 runs 2 and 3: the measures lost "必要に応じ" and kind ⑴ its 困難 clause."""
    answer = "\n".join(
        [
            "法令に基づき取引価格（料金）が定められていることやその他特別な事由がある場合、随意契約では予定価格調書の作成を省略できる。[E8]",
            KIND_TWO,
            MEASURE_ONE.replace("必要に応じ", ""),
            MEASURE_TWO,
        ]
    )
    issues = dict(deep._limitation_issues(answer, NOTICE_42_EVIDENCE))
    assert set(issues) == {1, 3}
    assert any("不可能又は著しく困難" in text for text in issues[1])
    assert issues[3] == ["必要に応じ"]


def test_item_left_out_of_a_list_the_answer_uses_is_reported_and_added():
    answer = "\n".join([KIND_ONE, MEASURE_ONE])
    gaps = deep._enumeration_issues(answer, NOTICE_42_EVIDENCE)
    assert [evidence_id for evidence_id, _ in gaps] == ["E8", "E8"]
    assert [texts[0][:1] for _, texts in gaps] == ["⑵", "⑵"]
    repaired, notes = deep._repair_enumerations(answer, deep._enumeration_findings(answer, NOTICE_42_EVIDENCE))
    lines = repaired.splitlines()
    assert lines[0] == KIND_ONE and lines[1].startswith("⑵ 予定価格が２５０万円をこえない随意契約で")
    assert lines[2] == MEASURE_ONE and lines[3].startswith("⑵ 契約担当官等は、見積書の徴取を省略する")
    assert deep._enumeration_issues(repaired, NOTICE_42_EVIDENCE) == []
    assert deep._validate_final_answer(repaired, {"E8"})
    assert len(notes) == 2 and all("列挙の項目" in text for text in notes)


def test_list_not_used_by_the_answer_is_not_required():
    # 第３条 has two lists; an answer about the first list alone leaves the second alone.
    answer = "\n".join(FAITHFUL.splitlines()[:4])
    assert deep._enumeration_issues(answer, EVIDENCE) == []
    assert deep._enumeration_issues(FAITHFUL, EVIDENCE) == []


def test_item_left_out_is_named_in_the_rewrite_and_added_when_still_missing(tmp_path, monkeypatch):
    _article_backend(monkeypatch, tmp_path)
    partial = "\n".join(line for line in FAITHFUL.splitlines() if "企画競争" not in line)
    calls = _text_calls(monkeypatch, partial, partial)
    result = _run_article(tmp_path)

    assert len(calls) == 2
    assert "根拠E1の列挙のうち、次の項目が回答にありません" in calls[1] and "企画競争" in calls[1]
    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert "三　企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件。）[E1]" in result.answer
    final = result.diagnostics["final_answer"]
    assert final["added_items"] == 1
    assert [attempt["enumeration_gaps"] for attempt in final["attempts"]] == [1, 1]
    assert any(item.startswith("enumeration_repaired") for item in result.unconfirmed)
    assert result.stop_reason != "enumeration_repaired"


def test_repeated_line_is_dropped_citations_aside():
    item = "⑵ 予定価格が２５０万円をこえない随意契約[E11]"
    answer = "\n".join([KIND_ONE, item, "", MEASURE_ONE, item.replace("[E11]", "[E12]"), item])
    deduped, dropped = deep._drop_duplicate_lines(answer)
    assert dropped == 2
    assert deduped.splitlines() == [KIND_ONE, item, "", MEASURE_ONE]


def test_two_paraphrases_repaired_into_one_source_line_appear_once(tmp_path, monkeypatch):
    """2026-09-24: 類型⑴⑵ were repaired into the same source lines twice."""
    _article_backend(monkeypatch, tmp_path)
    dropped_limit = "企画競争は審査する。[E1]"
    answer = FAITHFUL.replace(
        "企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件）は審査する。[E1]",
        dropped_limit + "\n企画競争の案件は審査する。[E1]",
    )
    _text_calls(monkeypatch, answer, answer)
    result = _run_article(tmp_path)

    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert result.answer.count("三　企画競争（前号に該当するもののほか、新規案件及び前回一者応募の案件。）[E1]") == 1
    assert result.diagnostics["final_answer"]["duplicate_lines"] == 1


def test_final_prompt_asks_for_limits_outside_parentheses_and_whole_lists():
    assert "括弧の外にある限定の文言" in prompt_templates.DEEP_FINAL_SYSTEM
    assert "同じ列挙の他の項目も省略せずに書いてください" in prompt_templates.DEEP_FINAL_SYSTEM

# --- paraphrase of a long source line; answer scope -----------------------------------


def test_paraphrase_of_a_long_source_line_is_matched_and_its_limit_required():
    """2026-09-23 run 2: a paraphrase of the whole 第1条 shared only a 24-character run
    with it, so the line escaped the limit check and dropped the 基金事業 limit."""
    answer = (
        "基金事業に係る調達案件については、調達実施前に案件の審査を行い、一括購入による削減の可否や"
        "契約方法・調達数量等の妥当性・適正性を確保するため、厚生労働省公共調達委員会が設置される。 [E5]"
    )
    article = ARTICLE_ONE.replace(
        "について、調達実施前に案件の審査を行う。",
        "について、調達実施前に案件の審査を行うことにより、一括購入による削減の可否、契約方法及び調達数量等の"
        "妥当性、適正性を確保するため、厚生労働省に公共調達委員会を設置する。",
    )
    issues = deep._limitation_issues(answer, [{"evidence_id": "E5", "excerpt": article}])
    assert [line for line, _ in issues] == [1]


def test_short_generic_line_is_not_attributed_by_shared_words_alone():
    # Shares "公共調達委員会" with the source line but is not a restatement of it.
    answer = "公共調達委員会は毎月開催される。[E5]"
    assert deep._limitation_issues(answer, ARTICLE_ONE_EVIDENCE) == []


def test_final_prompt_limits_the_answer_to_what_the_question_asks():
    assert "質問が直接問う事項" in prompt_templates.DEEP_FINAL_SYSTEM
    assert "所管範囲の説明は書かないでください" in prompt_templates.DEEP_FINAL_SYSTEM
    assert "主語と述語のある文にしてください" in prompt_templates.DEEP_FINAL_SYSTEM
