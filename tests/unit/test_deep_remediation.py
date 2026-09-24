"""Regression cases from the 2026-09-21 audit; no live model or user corpus."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import deep_research as deep
import deep_candidate_selection
import deep_transport
import coverage_evaluator as evaluator
import web_services
from test_coverage_evaluator import _result, SPEC
from deep_fakes import patch_retrieval


VERDICT = {
    "supported": True,
    "contradictions": [],
    "missing_conditions": [],
    "unsupported_claims": [],
}


def backend(monkeypatch, *, extract=None, answer="対象である。[E1]", verdict=None, queries=None):
    calls = []

    def retrieve(query, chunks, **kwargs):
        kwargs["cancel_check"]()
        calls.append(query)
        ranked = [{**c, "source_sha256": c["file_sha256"]} for c in chunks]
        return deep_candidate_selection.select_candidates(
            ranked, source_chunks=chunks
        )

    def model(name, system, user, **kwargs):
        if system == deep.DEEP_VERIFY_SYSTEM:
            return verdict(user) if verdict else dict(VERDICT)
        if system == deep.DEEP_GAPS_SYSTEM:
            return {"queries": queries or [], "unresolved": []}
        if extract:
            return extract(user)
        return {
            "subject": "対象",
            "scope": "原文の範囲",
            "conditions": ["対象である"],
            "exceptions": [],
            "references": [],
        }

    patch_retrieval(monkeypatch, deep, retrieve)
    monkeypatch.setattr(deep, "_call_ollama_json", model)
    monkeypatch.setattr(deep, "_call_ollama_text", lambda *a, **k: answer)
    return calls


def research(tmp_path, text="対象である。", **kwargs):
    (tmp_path / "source.md").write_text(text, encoding="utf-8")
    return deep.run_deep_research(
        "対象", model="fake", source_root=tmp_path, timeout_seconds=300, **kwargs
    )


def test_long_line_survives_pipeline_and_metadata_budget(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path, "あ" * 8500)
    slices = [e for e in result.evidence if e["start_line"] == 1]
    assert "".join(e["excerpt"] for e in slices) == "あ" * 8500
    assert {e["char_start"] for e in slices} == {0, 4000, 8000}
    assert result.diagnostics["evidence_chars"] <= 12000
    assert result.diagnostics["evidence_chars"] > 9000


def test_parent_scope_preserved_separately(tmp_path, monkeypatch):
    backend(monkeypatch)
    def retrieve_parent(q, chunks, **kwargs):
        candidate = dict(chunks[0])
        candidate["start_line"] = 4
        candidate["end_line"] = 4
        return deep_candidate_selection.select_candidates(
            [{**candidate, "source_sha256": candidate["file_sha256"]}],
            source_chunks=chunks,
        )

    patch_retrieval(monkeypatch, deep, retrieve_parent)
    result = research(tmp_path, "# 制度\n親の適用除外。\n## 個別\n対象である。")
    assert {(e["start_line"], e["end_line"]) for e in result.evidence} == {
        (1, 2),
        (3, 4),
    }


def test_evidence_overflow_is_partial_and_recorded(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path, "あ" * 15000)
    assert result.status == "partial"
    assert result.stop_reason == "evidence_budget"
    assert result.diagnostics["evidence_chars"] <= 12000
    assert any(e["reason"] == "evidence_budget" for e in result.ledger)


def test_real_retrieval_worker_uses_requested_synthetic_snapshot(tmp_path, monkeypatch):
    retrieval = deep._retrieve_candidates
    backend(monkeypatch)
    patch_retrieval(monkeypatch, deep, retrieval)
    result = research(tmp_path)
    assert result.evidence
    assert {e["path"] for e in result.evidence} == {"source.md"}


def test_missing_fields_retry_cannot_create_verified_empty_record(tmp_path, monkeypatch):
    backend(monkeypatch, extract=lambda user: {"conditions": []})
    result = research(tmp_path)
    assert all(e["verification_status"] == "raw_unverified" for e in result.evidence)


def test_final_answer_rejected_for_reversed_meaning(tmp_path, monkeypatch):
    def verdict(user):
        data = json.loads(user)
        if isinstance(data["claims"], str):
            return {**VERDICT, "supported": False, "contradictions": ["否定が逆"]}
        return dict(VERDICT)

    backend(monkeypatch, answer="対象外である。[E1]", verdict=verdict)
    result = research(tmp_path)
    assert result.status == "partial"
    assert "対象外である" not in result.answer
    assert "対象である。" in result.answer


def test_real_model_citation_spacing_is_accepted():
    answer = "対象条件：正社員で所属長の事前承認が必要です。\u202f[E3]  \n除外項目：資格試験は認定研修に含めません。 [E2]  "
    assert deep._validate_final_answer(answer, {"E2", "E3"})


@pytest.mark.parametrize(
    "answer",
    [
        "対象である。[E1]\n999万円が必要。",
        "対象である。[E99]",
        "条件A。条件B。[E1]",
        # A table row and a blockquote aside still need their own citation;
        # structural exemptions must not become a way to smuggle a bare claim.
        "| 項目 | 内容 |\n|---|---|\n| 対象 | 正社員のみ |",
        "| 項目 | 内容 |\n|---|---|\n| 対象 | 正社員のみ [E99] |",
        "> 正社員のみが対象である。",
    ],
)
def test_uncited_or_invented_claims_rejected(answer):
    assert not deep._validate_final_answer(answer, {"E1"})


@pytest.mark.parametrize(
    "answer",
    [
        # Table header + rule row carry no claim of their own; each data row
        # cites once for the whole row rather than once per cell.
        "| 項目 | 内容 | 根拠 |\n|---|---|---|\n| 対象 | 正社員のみ | [E1] |",
        "**制度概要**\n\n| 項目 | 内容 |\n|---|---|\n| 対象 | 正社員のみ [E1] |",
        # A blockquote aside cites once for the whole line.
        "> 正社員のみが対象である。[E1]",
    ],
)
def test_structural_lines_with_citations_are_accepted(answer):
    assert deep._validate_final_answer(answer, {"E1"})


def test_all_initial_viewpoints_then_additional_search(tmp_path, monkeypatch):
    calls = backend(monkeypatch, queries=["追加の資料"])
    result = research(tmp_path)
    assert all(q in calls for q in deep._initial_viewpoints("対象"))
    assert "追加の資料" in calls
    assert result.diagnostics["rounds"] == 2
    assert result.stop_reason != "round_limit"


def test_unresolved_reference_never_completed(tmp_path, monkeypatch):
    backend(
        monkeypatch,
        extract=lambda user: {
            "subject": "対象",
            "scope": "",
            "conditions": ["対象"],
            "exceptions": [],
            "references": ["存在しない別紙99"],
        },
    )
    result = research(tmp_path)
    assert result.status == "partial"
    assert any("unresolved_reference" in x for x in result.unconfirmed)


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_source_revalidated_after_model_call(tmp_path, monkeypatch, operation):
    def extract(user):
        if operation == "change":
            (tmp_path / "source.md").write_text("新しい資料", encoding="utf-8")
        else:
            (tmp_path / "source.md").unlink(missing_ok=True)
        return {
            "subject": "対象",
            "scope": "",
            "conditions": ["対象"],
            "exceptions": [],
            "references": [],
        }

    backend(monkeypatch, extract=extract)
    result = research(tmp_path)
    assert result.evidence == []
    assert result.stop_reason == "source_changed"
    assert result.status != "completed"


def test_final_cancel_is_cancelled_and_no_later_model_call(tmp_path, monkeypatch):
    backend(monkeypatch)

    def cancel(*a, **k):
        raise web_services.CancelledError("cancel", code="cancelled")

    monkeypatch.setattr(deep, "_call_ollama_text", cancel)
    result = research(tmp_path)
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")


def test_empty_extraction_retried_once_then_unconfirmed(tmp_path, monkeypatch):
    attempts = []
    backend(monkeypatch, extract=lambda user: attempts.append(user) or {})
    result = research(tmp_path)
    assert len(attempts) == 2
    assert result.status == "partial"
    assert result.evidence[0]["verification_status"] == "raw_unverified"


def test_no_silent_structured_text_truncation():
    original = {
        "subject": "x" * 600,
        "scope": "",
        "conditions": ["x" * 600] * 13,
        "exceptions": [],
        "references": [],
    }
    checked = deep._validate_section_json(original)
    assert len(checked["conditions"]) == 13
    assert checked["conditions"][0] == "x" * 600
    with pytest.raises(ValueError):
        deep._validate_section_json({})


def test_shared_absolute_deadline_not_reset(tmp_path, monkeypatch):
    calls = backend(monkeypatch)
    result = research(tmp_path, absolute_deadline=time.monotonic() - 1)
    assert result.stop_reason == "time_budget"
    assert calls == []


def test_integration_reservation_used(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path, absolute_deadline=time.monotonic() + 120)
    assert result.stop_reason == "time_budget"
    assert result.diagnostics["units"] == 0


def test_prompt_token_reservation_is_enforced():
    from search import PromptBudgetError

    with pytest.raises(PromptBudgetError):
        deep._check_prompt("system", "あ" * 100000)


def test_per_file_four_candidates_and_overflow_ledger(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path, "\n".join(f"## 節{i}\n対象{i}。" for i in range(10)))
    assert len(result.evidence) >= 4
    assert any(e["reason"] == "candidate_limit" for e in result.ledger)
    assert result.status == "partial"


def test_diagnostics_do_not_contain_source_or_question(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path, "SECRET_SYNTHETIC_MARKER")
    assert "SECRET_SYNTHETIC_MARKER" not in json.dumps(result.diagnostics)


def test_keyword_only_score_requires_human_review():
    spec = evaluator.load_spec(SPEC)
    result = _result()
    assert evaluator.evaluate_result(spec, result)["status"] == "NEEDS_REVIEW"
    result["semantic_review"] = {
        "kind": "human",
        "reviewer": "synthetic-test-review",
        "approved": True,
        "spec_sha256": spec["spec_sha256"],
        "result_sha256": evaluator.result_digest(result),
        "unsupported_claims": 0,
        "supported_item_ids": [i["id"] for i in spec["items"]],
    }
    assert evaluator.evaluate_result(spec, result)["status"] == "PASS"
    result["answer"] += " 別の主張。"
    assert evaluator.evaluate_result(spec, result)["status"] != "PASS"


def test_reverse_answer_fails_evaluator():
    result = _result()
    result["answer"] = (
        result["answer"]
        .replace("正社員が対象", "正社員は対象外")
        .replace("所属長の承認と受講申込書が必要", "所属長の承認と受講申込書は不要")
    )
    assert evaluator.evaluate_result(evaluator.load_spec(SPEC), result)["status"] == "FAIL"


def test_forged_quote_is_rejected():
    result = _result()
    result["evidence"][0]["excerpt"] = "原文に存在しない"
    assert evaluator.evaluate_result(evaluator.load_spec(SPEC), result)["invalid_evidence"] == [
        "E1"
    ]


def test_terminal_and_slot_release_exactly_once():
    table = web_services.JobTable()
    try:
        entry, _ = table.submit(
            "x",
            web_services.CancellationToken(300),
            fingerprint=("s", "q", "off", 300, "deep"),
            mode="deep",
        )
        entry.broadcast({"type": "done"})
        entry.broadcast({"type": "error", "code": "cancelled"})
        assert entry.state == "completed"
        assert len(entry._event_buffer) == 1
        table.finish("x")
        table.submit("y", web_services.CancellationToken(300))
        table.finish("x")
        assert table.running_count == 1
    finally:
        table.shutdown()


def test_progress_and_slow_subscriber_buffers_bounded():
    entry = web_services.JobEntry(
        web_services.CancellationToken(300), fingerprint=("s", "q", "off", 300, "deep")
    )
    subscriber = entry.add_subscriber()
    for i in range(2000):
        entry.broadcast({"type": "deep_progress", "units": i})
    assert len(entry._event_buffer) == 1
    assert subscriber.qsize() <= 64
    assert entry.add_subscriber().get_nowait()["units"] == 1999


def test_deep_public_replay_payload_compacts_large_ledger_and_warnings():
    result = deep.DeepResearchResult(
        status="partial",
        stop_reason="time_budget",
        unconfirmed=[f"unconfirmed-{i}" for i in range(1000)],
        ledger=[
            {"path": f"skill-source/doc-{i}.md", "reason": "time_budget", "status": "unconfirmed"}
            for i in range(1000)
        ],
        diagnostics={"ledger_counts": {"time_budget": 1000}},
    )

    event = web_services.build_deep_evidence_event(result)

    assert len(json.dumps(event, ensure_ascii=False).encode("utf-8")) < 1_000_000
    assert event["deepLedgerTruncated"] is True
    assert event["deepWarningsTruncated"] is True
    assert event["deepLedger"][-1]["reason"] == "replay_truncated"
    assert event["warnings"][-1].endswith("件省略しました")


def test_deep_public_answer_is_utf8_bounded_for_replay():
    bounded = web_services._truncate_deep_text("あ" * 1000, max_bytes=128)

    assert len(bounded.encode("utf-8")) <= 128
    assert bounded.endswith("[回答本文は再接続用保持上限のため省略されました]")


def test_oversized_replay_fails_with_bounded_terminal_state():
    entry = web_services.JobEntry(
        web_services.CancellationToken(300), fingerprint=("s", "q", "off", 300, "deep")
    )
    entry.broadcast({"type": "chunk", "text": "a" * 999900})
    entry.broadcast({"type": "chunk", "text": "a" * 1000})
    assert entry.state == "failed"
    assert entry.cancel_token.is_cancelled
    assert len(json.dumps(entry._event_buffer).encode("utf-8")) <= 1_000_000
    assert entry._event_buffer[-1]["code"] == "result_too_large"


@pytest.mark.parametrize("phase", ["headers", "stream"])
def test_cancel_unresponsive_http_reaps_worker(monkeypatch, phase):
    received = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if phase == "stream":
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
            received.set()
            release.wait(4)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    processes = []
    original = deep_transport.subprocess.Popen

    def spawn(*a, **k):
        process = original(*a, **k)
        processes.append(process)
        return process

    monkeypatch.setattr(deep_transport.subprocess, "Popen", spawn)

    def cancel():
        if received.is_set():
            raise web_services.CancelledError("cancel", code="cancelled")

    started = time.monotonic()
    try:
        with pytest.raises(web_services.CancelledError):
            deep_transport.run_worker(
                {
                    "operation": "chat",
                    "host": f"http://127.0.0.1:{server.server_port}",
                    "timeout": 20,
                    "body": {"model": "fake", "messages": [], "stream": True},
                },
                timeout=20,
                cancel_check=cancel,
            )
        assert time.monotonic() - started < 3
        assert all(p.poll() is not None for p in processes)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(2)


# --- 回答状態(answer_state)の分離: 2026-09-23調査の指摘4 ---------------------
# web_services側の固定confidence(1.0/0.0)は根拠の有無しか表さず回答品質の
# 指標ではなかった。generated/verification_failed/not_generatedを分離する。


def test_answer_state_is_generated_when_integration_and_verification_succeed(tmp_path, monkeypatch):
    backend(monkeypatch)
    result = research(tmp_path)
    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert result.answer.startswith("対象である")


def test_answer_state_is_verification_failed_when_final_verdict_rejects(tmp_path, monkeypatch):
    def verdict(user):
        # セクション単位の検証(claimsはdict)は通し、最終統合の検証(claimsは
        # 文字列の回答案)だけ不支持にして、統合後の照合失敗経路を狙う。
        parsed = json.loads(user)
        if isinstance(parsed.get("claims"), str):
            return {**VERDICT, "supported": False}
        return dict(VERDICT)

    backend(monkeypatch, verdict=verdict)
    result = research(tmp_path)
    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert "原文抜粋" in result.answer
    # 根拠IDの検査は通過しており、不合格理由は支持照合として区別して残る。
    assert any("根拠に支持されないと照合" in item for item in result.unconfirmed)
    assert not any("根拠IDが不足または不正" in item for item in result.unconfirmed)


def test_answer_state_is_not_generated_when_final_model_call_fails(tmp_path, monkeypatch):
    """最終回答のモデル呼び出し自体が失敗した場合は、検証失敗ではなく未生成とする。"""
    backend(monkeypatch)

    def failing_final(*args, **kwargs):
        raise OSError("model unavailable")

    monkeypatch.setattr(deep, "_call_ollama_text", failing_final)
    result = research(tmp_path)
    assert result.answer_state == deep.ANSWER_STATE_NOT_GENERATED
    assert result.evidence
    assert "原文抜粋" in result.answer
    assert any("最終回答を生成できず" in item for item in result.unconfirmed)


def _two_section_source(tmp_path):
    (tmp_path / "source.md").write_text("# A\n対象である。\n# B\n対象でもある。\n", encoding="utf-8")
    return deep.run_deep_research("対象", model="fake", source_root=tmp_path, timeout_seconds=300)


@pytest.mark.parametrize(
    "call_error",
    [TimeoutError("deep worker deadline"), deep_transport.DeepWorkerError(code="timeout"), deep_transport.DeepWorkerError(code="stall")],
    ids=["worker-deadline", "worker-timeout", "worker-stall"],
)
def test_call_timeout_during_exploration_still_integrates_verified_evidence(tmp_path, monkeypatch, call_error):
    """2026-09-23の実資料受入(300秒中175秒で回答未生成): 探索中の呼出には
    「残り時間−統合予約」がtimeoutとして渡るため、その呼出のtimeoutは探索の
    予定終了である。照合済み根拠があれば統合へ進むこと。"""
    backend(monkeypatch)
    base_json = deep._call_ollama_json
    calls = {"json": 0, "text": 0}

    def json_call(*args, **kwargs):
        calls["json"] += 1
        if calls["json"] == 3:  # 1節目の抽出・照合は成功、2節目の抽出だけ呼出timeout
            raise call_error
        return base_json(*args, **kwargs)

    def text_call(*args, **kwargs):
        calls["text"] += 1
        return "対象である。[E1]"

    monkeypatch.setattr(deep, "_call_ollama_json", json_call)
    monkeypatch.setattr(deep, "_call_ollama_text", text_call)

    result = _two_section_source(tmp_path)

    assert calls["text"] == 1
    assert result.answer_state == deep.ANSWER_STATE_GENERATED
    assert result.answer.startswith("対象である")
    assert result.status == "partial"
    assert result.stop_reason == "time_budget"
    assert "time_budget:  — モデル呼出が時間内に終わらず探索を終了" in result.unconfirmed


def test_cancel_during_exploration_is_not_treated_as_call_timeout(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = {"text": 0}

    def cancelled(*args, **kwargs):
        raise deep_transport.DeepWorkerError(code="cancelled")

    def text_call(*args, **kwargs):
        calls["text"] += 1
        return "対象である。[E1]"

    monkeypatch.setattr(deep, "_call_ollama_json", cancelled)
    monkeypatch.setattr(deep, "_call_ollama_text", text_call)

    result = _two_section_source(tmp_path)

    assert result.status == "cancelled"
    assert calls["text"] == 0


def test_run_deadline_during_exploration_is_not_treated_as_call_timeout(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = {"text": 0}

    def expired(*args, **kwargs):
        raise deep.DeepBudgetExpired("deep research time budget expired")

    def text_call(*args, **kwargs):
        calls["text"] += 1
        return "対象である。[E1]"

    monkeypatch.setattr(deep, "_call_ollama_json", expired)
    monkeypatch.setattr(deep, "_call_ollama_text", text_call)

    result = _two_section_source(tmp_path)

    assert result.stop_reason == "time_budget"
    assert result.answer_state == deep.ANSWER_STATE_NOT_GENERATED
    assert calls["text"] == 0


def test_answer_state_is_verification_failed_when_final_citations_are_invalid(tmp_path, monkeypatch):
    """回答は返ったが根拠ID検査に通らない場合は検証失敗とする。"""
    backend(monkeypatch, answer="根拠のない断定。")
    result = research(tmp_path)
    assert result.answer_state == deep.ANSWER_STATE_VERIFICATION_FAILED
    assert any("根拠IDが不足または不正" in item for item in result.unconfirmed)
    assert not any("支持されないと照合" in item for item in result.unconfirmed)


def test_answer_state_is_not_generated_when_no_evidence_is_verified(tmp_path, monkeypatch):
    backend(
        monkeypatch,
        extract=lambda user: {
            "subject": "無関係",
            "scope": "無関係",
            "relevance": "irrelevant",
            "conditions": [],
            "exceptions": [],
            "references": [],
        },
    )
    result = research(tmp_path)
    assert result.answer_state == deep.ANSWER_STATE_NOT_GENERATED
    assert result.evidence == []


def test_build_deep_evidence_event_reports_answer_state_instead_of_fixed_confidence():
    result = deep.DeepResearchResult(
        status="completed",
        stop_reason="scope_processed",
        answer="対象である。[E1]",
        answer_state=deep.ANSWER_STATE_GENERATED,
        evidence=[{"path": "a.md", "evidence_id": "E1"}],
    )
    event = web_services.build_deep_evidence_event(result)
    # 固定confidence(1.0 if items else 0.0)は回答品質の指標ではなかったため廃止する。
    assert event["confidence"] is None
    assert event["answerState"] == deep.ANSWER_STATE_GENERATED
    assert "検証を通過" in event["answerStateMessage"] or "生成済み" in event["answerStateMessage"]


def test_build_deep_evidence_event_keeps_important_warnings_over_bulk_ones():
    """2026-09-23調査(指摘3): 表示上限で切り詰めると、後発のmodel_errorが
    大量のcandidate_limitに埋もれて脱落し得た。重要な単発理由を残す。"""
    unconfirmed = [f"candidate_limit: doc-{i}.md L1-2" for i in range(300)]
    unconfirmed.append("model_error: important.md — 追加観点の確認に失敗")
    result = deep.DeepResearchResult(
        status="partial",
        stop_reason="time_budget",
        unconfirmed=unconfirmed,
    )
    event = web_services.build_deep_evidence_event(result)
    assert any("model_error" in w for w in event["warnings"])
