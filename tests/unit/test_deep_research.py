"""deep 調査モードの予算・原文範囲・世代固定契約。"""

import json

import pytest

import deep_candidate_selection
import deep_research
import search
from deep_fakes import patch_retrieval


def test_validate_deep_timeout_keeps_normal_range_separate():
    assert deep_research.validate_deep_timeout_seconds("1800") == 1800
    assert deep_research.validate_deep_timeout_seconds(300) == 300
    with pytest.raises(ValueError):
        deep_research.validate_deep_timeout_seconds(299)
    with pytest.raises(ValueError):
        deep_research.validate_deep_timeout_seconds(1801)


def test_deep_json_verification_has_a_separate_output_budget(monkeypatch):
    payloads = []

    def fake_worker(payload, **kwargs):
        payloads.append(payload)
        return '{"supported": true}'

    monkeypatch.setattr(deep_research, "run_worker", fake_worker)

    assert deep_research._call_ollama_json("model", "system", "user", timeout=1) == {
        "supported": True
    }
    assert payloads[0]["body"]["options"]["num_ctx"] == deep_research.DEEP_NUM_CTX
    assert (
        payloads[0]["body"]["options"]["num_predict"]
        == deep_research.DEEP_JSON_NUM_PREDICT
    )

    payloads.clear()
    monkeypatch.setattr(
        deep_research,
        "run_worker",
        lambda payload, **kwargs: payloads.append(payload) or "answer",
    )
    assert deep_research._call_ollama_text("model", "system", "user", timeout=1) == "answer"
    assert payloads[0]["body"]["options"]["num_predict"] == deep_research.DEEP_TEXT_NUM_PREDICT


def test_split_line_range_keeps_long_line_character_offsets():
    line = "あ" * 8500
    units = deep_research._split_line_range([line], 1, 1, max_chars=4000)
    assert [unit["text"] for unit in units] == [
        line[:4000],
        line[4000:8000],
        line[8000:],
    ]
    assert [(unit["char_start"], unit["char_end"]) for unit in units] == [
        (0, 3999),
        (4000, 7999),
        (8000, 8499),
    ]


def test_section_range_selects_inner_section_without_losing_heading_context():
    text = "# 規程\n導入\n## 条件\n適用条件\n## 例外\n除外条件\n"
    assert deep_research._section_range(text, 4, 4) == (3, 4, "条件")
    assert deep_research._section_range(text, 6, 6) == (5, 6, "例外")


def test_ranges_with_context_processes_hit_section_before_parent_intro():
    """2026-09-23調査: ヒット本文より親冒頭が先に処理され、時間予算切れ時に
    本文まで到達できなかった。ヒット本文のrangeが常に先頭に来ることを固定する。"""
    text = "# 制度\n親の適用除外。\n## 個別\n対象である。\n"
    ranges = deep_research._ranges_with_context(text, 4, 4)
    assert ranges[0] == (3, 4, "個別")
    assert ranges[1:] == [(1, 2, "制度")]


def test_ranges_with_context_orders_hit_first_across_multiple_ancestors():
    text = "# 大分類\n大分類の説明。\n## 中分類\n中分類の説明。\n### 個別\n対象である。\n"
    ranges = deep_research._ranges_with_context(text, 6, 6)
    assert ranges[0] == (5, 6, "個別")
    assert set(ranges[1:]) == {(3, 4, "中分類"), (1, 2, "大分類")}


def test_run_deep_research_reads_multiple_sections_and_validates_citations(tmp_path, monkeypatch):
    root = tmp_path / "skill-source"
    root.mkdir()
    source = root / "notice.md"
    source.write_text(
        "# 公共調達委員会\n導入文\n## 付議条件\n条件Aを満たす場合。\n## 除外\n例外Bは対象外。\n",
        encoding="utf-8",
    )
    chunks = search.build_source_chunks(root)
    by_heading = {chunk["heading"]: chunk for chunk in chunks if chunk.get("heading")}
    candidates = [
        {
            "path": "notice.md",
            "chunk_id": by_heading["付議条件"]["chunk_id"],
            "heading": "付議条件",
            "start_line": by_heading["付議条件"]["start_line"],
            "end_line": by_heading["付議条件"]["end_line"],
            "score": 2,
            "rrf_score": 0.5,
            "source_sha256": by_heading["付議条件"]["file_sha256"],
        },
        {
            "path": "notice.md",
            "chunk_id": by_heading["除外"]["chunk_id"],
            "heading": "除外",
            "start_line": by_heading["除外"]["start_line"],
            "end_line": by_heading["除外"]["end_line"],
            "score": 1,
            "rrf_score": 0.4,
            "source_sha256": by_heading["除外"]["file_sha256"],
        },
    ]

    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda query, chunks, **kwargs: deep_candidate_selection.select_candidates(
            candidates, source_chunks=chunks
        ),
    )
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_json",
        lambda *args, **kwargs: (
            {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
            if args[1] == deep_research.DEEP_VERIFY_SYSTEM
            else {"queries": [], "unresolved": []}
            if args[1] == deep_research.DEEP_GAPS_SYSTEM
            else {
                "subject": "公共調達委員会",
                "scope": "通知の適用範囲",
                "conditions": ["本文に記載された条件"],
                "exceptions": ["本文に記載された例外"],
                "references": [],
            }
        ),
    )
    monkeypatch.setattr(
        deep_research,
        "_call_ollama_text",
        lambda *args, **kwargs: "条件Aを確認した。[E1]\n例外Bを確認した。[E2]",
    )

    result = deep_research.run_deep_research(
        "公共調達委員会の条件",
        model="test-model",
        source_root=root,
        timeout_seconds=300,
    )

    assert result.status in {"completed", "partial"}
    assert len(result.evidence) == 3
    assert {(item["start_line"], item["end_line"]) for item in result.evidence} == {
        (1, 2),
        (3, 4),
        (5, 6),
    }
    assert result.answer.startswith("条件Aを確認した")
    public = result.to_public_dict()
    assert "excerpt" in public["evidence"][0]
    assert "条件Aを確認した" not in json.dumps(public["diagnostics"], ensure_ascii=False)


def test_run_deep_research_does_not_mix_changed_source_generation(tmp_path, monkeypatch):
    root = tmp_path / "skill-source"
    root.mkdir()
    source = root / "notice.md"
    source.write_text("# 規程\n本文\n", encoding="utf-8")
    chunk = search.build_source_chunks(root)[0]
    candidate = {
        "path": "notice.md",
        "chunk_id": chunk["chunk_id"],
        "heading": chunk["heading"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "source_sha256": "different-generation",
    }
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda query, chunks, **kwargs: deep_candidate_selection.select_candidates(
            [candidate], source_chunks=chunks
        ),
    )
    result = deep_research.run_deep_research(
        "規程",
        model="",
        source_root=root,
        timeout_seconds=300,
    )
    assert result.status == "failed"
    assert result.stop_reason == "source_changed"
    assert result.evidence == []
    assert any("source_changed" in item for item in result.unconfirmed)


def test_run_deep_research_returns_cancelled_terminal_state(tmp_path):
    class Cancelled(Exception):
        code = "cancelled"

    (tmp_path / "notice.md").write_text("# 規程\n本文\n", encoding="utf-8")

    def cancel():
        raise Cancelled()

    result = deep_research.run_deep_research(
        "規程",
        model="",
        source_root=tmp_path,
        timeout_seconds=300,
        cancel_check=cancel,
    )
    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled"


def test_summarize_unconfirmed_collapses_bulk_reasons_for_prompt():
    """2026-09-23調査(指摘2): 除外候補全件(387,778文字相当)をそのまま最終
    promptへ渡すとevidence追加前にprompt予算を超える。件数集約後は
    小さく収まり、未読であることが明示され続けることを検証する。"""
    ledger = [
        {"reason": "candidate_limit", "path": f"doc-{i}.md", "status": "unconfirmed"}
        for i in range(500)
    ] + [{"reason": "model_error", "path": "important.md", "status": "unconfirmed"}]
    unconfirmed = [
        f"candidate_limit: doc-{i}.md L1-2000000000" for i in range(500)
    ] + ["model_error: important.md — 追加観点の確認に失敗"]

    summary = deep_research._summarize_unconfirmed_for_prompt(unconfirmed, ledger)

    joined = "\n".join(summary)
    assert len(joined) < deep_research.DEEP_UNCONFIRMED_PROMPT_CHAR_LIMIT + 200
    assert "candidate_limit: 500件" in joined
    assert "未読" in joined
    assert any("model_error" in line for line in summary)
    # 個別の除外候補500件分の生テキストは集約されて消え、件数だけが残る。
    assert "doc-0.md" not in joined


def test_summarize_unconfirmed_truncates_when_detail_lines_overflow_budget():
    ledger = [{"reason": "model_error", "path": f"doc-{i}.md", "status": "unconfirmed"} for i in range(200)]
    unconfirmed = [f"model_error: doc-{i}.md — 詳細理由がここに入る想定の長めの文言" for i in range(200)]

    summary = deep_research._summarize_unconfirmed_for_prompt(unconfirmed, ledger, max_chars=500)

    assert len("\n".join(summary)) < 700
    assert summary[-1].startswith("...他")


def test_deep_budget_tracks_stage_seconds_and_elapsed_time():
    budget = deep_research.DeepBudget(timeout_seconds=300)
    budget.enter_stage("search")
    budget.enter_stage("read")
    budget.enter_stage("complete")
    assert set(budget.stage_seconds) == {"planning", "search", "read"}
    assert all(v >= 0 for v in budget.stage_seconds.values())
    assert budget.elapsed_seconds() >= 0


def test_diagnostics_stage_seconds_include_final_integration_time(tmp_path, monkeypatch):
    """最後の段階(統合・最終照合)の時間は、診断の確定前に計上されること。"""
    (tmp_path / "notice.md").write_text("# 規程\n対象である。\n", encoding="utf-8")
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda query, chunks, **kwargs: deep_candidate_selection.select_candidates(
            [{**c, "source_sha256": c["file_sha256"]} for c in chunks], source_chunks=chunks
        ),
    )

    def model(name, system, user, **kwargs):
        if system == deep_research.DEEP_VERIFY_SYSTEM:
            return {"supported": True, "contradictions": [], "missing_conditions": [], "unsupported_claims": []}
        if system == deep_research.DEEP_GAPS_SYSTEM:
            return {"queries": [], "unresolved": []}
        return {"subject": "対象", "scope": "s", "conditions": ["対象である"], "exceptions": [], "references": []}

    clock = {"now": 1000.0}
    monkeypatch.setattr(deep_research.time, "monotonic", lambda: clock["now"])

    def slow_final(*args, **kwargs):
        clock["now"] += 7.0
        return "対象である。[E1]"

    monkeypatch.setattr(deep_research, "_call_ollama_json", model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", slow_final)

    result = deep_research.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300
    )

    assert result.answer_state == deep_research.ANSWER_STATE_GENERATED
    stages = result.diagnostics["stage_seconds"]
    assert stages["integrate"] == pytest.approx(7.0)
    assert sum(stages.values()) == pytest.approx(result.diagnostics["elapsed_seconds"])


def test_fallback_answer_collapses_bulk_exclusions_but_keeps_important_entries():
    """原文抜粋表示でも上限による大量除外は件数へ集約し、重要事項は個別に残す。"""
    ledger = [
        {"reason": "candidate_limit", "path": f"doc-{i}.md", "status": "unconfirmed"}
        for i in range(300)
    ] + [{"reason": "model_error", "path": "", "status": "unconfirmed"}]
    unconfirmed = [f"candidate_limit: doc-{i}.md L1-2" for i in range(300)] + [
        "model_error:  — 最終回答を生成できず原文抜粋を表示"
    ]
    evidence = [
        {"evidence_id": "E1", "path": "a.md", "start_line": 1, "end_line": 1, "excerpt": "対象である。"}
    ]

    answer = deep_research._fallback_answer(evidence, unconfirmed, ledger, "time_budget")

    assert "candidate_limit: 300件が上限のため未読のまま" in answer
    assert "doc-0.md" not in answer
    assert "model_error:  — 最終回答を生成できず原文抜粋を表示" in answer
    assert "[E1]" in answer and "> 対象である。" in answer
    assert "調査終了理由: `time_budget`" in answer


def test_run_deep_research_diagnostics_separate_elapsed_from_call_timeout(tmp_path, monkeypatch):
    """2026-09-23調査(指摘5): 全体経過時間(time_budget)だけでは呼出timeout・
    stall・URLErrorのどれで時間を消費したか説明できない。monotonicな
    段階別内訳と全体経過が診断に残ることを確認する。"""
    (tmp_path / "notice.md").write_text("# 規程\n本文\n", encoding="utf-8")
    patch_retrieval(
        monkeypatch,
        deep_research,
        lambda query, chunks, **kwargs: deep_candidate_selection.select_candidates(
            [], source_chunks=chunks
        ),
    )
    result = deep_research.run_deep_research(
        "規程",
        model="",
        source_root=tmp_path,
        timeout_seconds=300,
    )
    assert "elapsed_seconds" in result.diagnostics
    assert "stage_seconds" in result.diagnostics
    assert result.diagnostics["reserved_seconds"] == {
        "integration": deep_research.DEEP_INTEGRATION_RESERVE_SECONDS,
        "finalization": deep_research.DEEP_FINALIZATION_RESERVE_SECONDS,
    }


def test_deep_answer_reaches_generation_with_deep_headings_and_bulk_exclusions(tmp_path, monkeypatch):
    """回答未生成の是正（2026-09-23計画・最優先節）の受入条件そのもの:
    深い見出し階層と、候補・資料・単位の各上限を超える大量の除外候補が
    同時にあっても、本文読取から回答生成・検証まで到達することを確認する。
    是正前は除外候補全件が最終promptへ連結され、実資料規模で
    prompt予算超過(387,778文字相当)を起こしていた。
    """
    root = tmp_path
    # 実資料と同様に長い資料名を使い、除外候補の自由文が最終prompt予算を
    # 単独で超える規模にする(是正前はここでprompt予算超過となり回答未生成)。
    long_name = "長い資料名" * 16
    for doc_index in range(40):
        lines = [f"# 大分類{doc_index}", "大分類の説明。"]
        for section_index in range(12):
            lines.append(f"## 中分類{doc_index}-{section_index}")
            lines.append("中分類の説明。")
            lines.append(f"### 個別{doc_index}-{section_index}")
            lines.append("対象である。")
        (root / f"{long_name}-{doc_index}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    chunks = search.build_source_chunks(root)
    heading_chunks = [c for c in chunks if str(c.get("heading", "")).startswith("個別")]
    # DEEP_MAX_CANDIDATES_TOTAL(40)・DEEP_MAX_DOCUMENTS(5)・DEEP_MAX_UNITS(20)の
    # いずれも超える候補数であることを前提として保証する。
    assert len(heading_chunks) > deep_research.DEEP_MAX_CANDIDATES_TOTAL
    final_prompts = []

    def final_text(model_name, system, prompt, **kwargs):
        final_prompts.append((system, prompt))
        return "対象である。[E1]"

    def retrieve(query, chunks_arg, **kwargs):
        ranked = [{**c, "source_sha256": c["file_sha256"]} for c in heading_chunks]
        return deep_candidate_selection.select_candidates(ranked, source_chunks=chunks_arg)

    def model(name, system, user, **kwargs):
        if system == deep_research.DEEP_VERIFY_SYSTEM:
            return {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
        if system == deep_research.DEEP_GAPS_SYSTEM:
            return {"queries": [], "unresolved": []}
        return {
            "subject": "対象",
            "scope": "適用範囲",
            "conditions": ["対象である"],
            "exceptions": [],
            "references": [],
        }

    patch_retrieval(monkeypatch, deep_research, retrieve)
    monkeypatch.setattr(deep_research, "_call_ollama_json", model)
    monkeypatch.setattr(deep_research, "_call_ollama_text", final_text)

    result = deep_research.run_deep_research(
        "対象の条件",
        model="fake",
        source_root=root,
        timeout_seconds=300,
    )

    assert result.answer_state == deep_research.ANSWER_STATE_GENERATED
    assert result.status in {"completed", "partial"}
    assert result.answer.startswith("対象である")
    # 候補・資料・単位いずれかの上限で大量の除外が発生し、未読のまま記録される。
    assert len(result.unconfirmed) > deep_research.DEEP_MAX_CANDIDATES_TOTAL
    assert result.diagnostics["documents"] <= deep_research.DEEP_MAX_DOCUMENTS
    assert result.diagnostics["units"] <= deep_research.DEEP_MAX_UNITS
    # 是正前の方式(未確認事項を全件promptへ連結)ならprompt予算を超える規模であること。
    system, prompt = final_prompts[-1]
    unabridged = deep_research.build_deep_final_prompt(
        "対象の条件", result.evidence, result.unconfirmed, "scope_processed"
    )
    with pytest.raises(search.PromptBudgetError):
        deep_research._check_prompt(system, unabridged)
    deep_research._check_prompt(system, prompt)
    assert "candidate_limit:" in prompt and "件が上限のため未読のまま" in prompt
