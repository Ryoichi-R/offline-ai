"""P1: 案A（LLM根拠検証）の製品実装の単体テスト。

計画（``plans/offline-ai-evidence-sufficiency-verification-plan.md``案Aの設計、
134〜161行目）の契約を固定する。P1では既定OFFで実装し、2026-09-20のP3で
利用者判断により既定ONへ切り替えた（``OFFLINE_AI_EVIDENCE_VERIFY=false`` で無効化）。
"""

from __future__ import annotations

import json

import pytest

import search


# ---------------------------------------------------------------------------
# JSON検証（矛盾する組み合わせの拒否）
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed_json():
    result = search._validate_evidence_verification_json(
        {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [{"condition": "一般社員", "supported": True}],
        }
    )
    assert result["support"] == "fully_supported"


@pytest.mark.parametrize(
    "payload",
    [
        {"support": "invalid", "reason_code": "answer_found"},
        {"support": "fully_supported", "reason_code": "invalid"},
        {"support": "fully_supported", "reason_code": "answer_found", "conditions": "x"},
        {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [{"condition": "x", "supported": "true"}],
        },
        "not-a-dict",
    ],
)
def test_validate_rejects_malformed_json(payload):
    with pytest.raises(search.EvidenceVerificationError):
        search._validate_evidence_verification_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"support": "fully_supported", "reason_code": "different_subject", "conditions": []},
        {"support": "unsupported", "reason_code": "answer_found", "conditions": []},
        {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [{"condition": "x", "supported": False}],
        },
    ],
)
def test_validate_rejects_contradictory_combinations(payload):
    with pytest.raises(search.EvidenceVerificationError):
        search._validate_evidence_verification_json(payload)


# ---------------------------------------------------------------------------
# verify_evidence_support: transport / cancel
# ---------------------------------------------------------------------------


def _ollama_response(payload: dict):
    from unittest.mock import MagicMock

    body = json.dumps({"message": {"content": json.dumps(payload)}}).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_verify_evidence_support_returns_validated_json(monkeypatch):
    monkeypatch.setattr(
        search.urllib.request,
        "urlopen",
        lambda *a, **k: _ollama_response(
            {"support": "unsupported", "reason_code": "missing_detail", "conditions": []}
        ),
    )
    result = search.verify_evidence_support(
        "質問", [{"path": "a.md", "snippet": "本文"}], "m", timeout=5
    )
    assert result["support"] == "unsupported"


def test_verify_evidence_support_uses_search_plan_call_settings(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        body = json.loads(request.data.decode("utf-8"))
        captured["body"] = body
        return _ollama_response(
            {"support": "fully_supported", "reason_code": "answer_found", "conditions": []}
        )

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)
    search.verify_evidence_support("質問", [], "m", timeout=7)

    assert captured["timeout"] == 7
    assert captured["body"]["think"] is False
    assert captured["body"]["keep_alive"] == search.OLLAMA_KEEP_ALIVE
    assert captured["body"]["options"]["num_ctx"] == search.NUM_CTX
    assert captured["body"]["options"]["num_batch"] == search.NUM_BATCH
    assert captured["body"]["options"]["num_gpu"] == search.NUM_GPU


def test_verify_evidence_support_raises_evidence_error_on_transport_failure(monkeypatch):
    def fail(*_a, **_k):
        raise TimeoutError

    monkeypatch.setattr(search.urllib.request, "urlopen", fail)
    with pytest.raises(search.EvidenceVerificationError):
        search.verify_evidence_support("質問", [], "m", timeout=5)


def test_verify_evidence_support_raises_evidence_error_on_malformed_json(monkeypatch):
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.read.return_value = b'{"message": {"content": "not json"}}'
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(search.urllib.request, "urlopen", lambda *a, **k: resp)
    with pytest.raises(search.EvidenceVerificationError):
        search.verify_evidence_support("質問", [], "m", timeout=5)


class _StopCancel(Exception):
    """web_services.CancelledError相当。search.pyはこの型を知らない。"""


def test_verify_evidence_support_propagates_cancel_check_exception_without_wrapping(monkeypatch):
    """利用者キャンセル・全体タイムアウトはEvidenceVerificationErrorへ変換せず、
    そのまま上位へ伝播させる契約（検証の失敗として扱わない）。
    """

    def cancel_check():
        raise _StopCancel("cancelled")

    def fail_if_called(*_a, **_k):
        raise AssertionError("cancel_check が例外を投げたら通信してはならない")

    monkeypatch.setattr(search.urllib.request, "urlopen", fail_if_called)
    with pytest.raises(_StopCancel):
        search.verify_evidence_support(
            "質問", [], "m", cancel_check=cancel_check, timeout=5
        )


# ---------------------------------------------------------------------------
# 予算計算
# ---------------------------------------------------------------------------


def test_evidence_verify_budget_none_when_no_callback():
    assert search._evidence_verify_budget(None) is None


def test_evidence_verify_budget_subtracts_generation_reserve(monkeypatch):
    monkeypatch.setattr(search, "EVIDENCE_VERIFY_GENERATION_RESERVE", 30)
    assert search._evidence_verify_budget(lambda: 100) == 70


def test_evidence_verify_budget_floors_at_zero(monkeypatch):
    monkeypatch.setattr(search, "EVIDENCE_VERIFY_GENERATION_RESERVE", 30)
    assert search._evidence_verify_budget(lambda: 10) == 0.0


def test_evidence_verify_budget_none_when_callback_raises():
    def broken():
        raise RuntimeError

    assert search._evidence_verify_budget(broken) is None


# ---------------------------------------------------------------------------
# run_retrieval_pipeline 統合（既定OFF/ON、格上げ禁止、予算不足、trace）
# ---------------------------------------------------------------------------


def _prepare_pipeline(monkeypatch, tmp_path, text: str, *, final_status: str = "sufficient"):
    """検索本体をモックし、最終根拠(matches)と状態(final_status)を確定させる。

    ``_calculate_confidence`` だけをモックすると、実際の ``matches`` は空の
    ままになり ``build_user_prompt`` が根拠なし早期パスへ入ってしまうため、
    ``select_final_evidence`` 自体をモックして両方を一致させる。
    """
    (tmp_path / "doc.md").write_text(text, encoding="utf-8")
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
    fixed_matches = [
        {"path": "doc.md", "heading": "", "start_line": 1, "end_line": 2, "snippet": text}
    ]

    def fake_select_final_evidence(*_a, **_k):
        return list(fixed_matches), 0.9, final_status

    monkeypatch.setattr(search, "select_final_evidence", fake_select_final_evidence)
    return fixed_matches


def test_evidence_verify_is_enabled_by_default(monkeypatch):
    """2026-09-20のP3で利用者判断により既定ONとした。"""
    monkeypatch.delenv("OFFLINE_AI_EVIDENCE_VERIFY", raising=False)
    assert search.EVIDENCE_VERIFY_ENABLED_DEFAULT is True
    assert search._evidence_verify_enabled() is True


def test_verification_skipped_when_disabled_by_env(monkeypatch, tmp_path):
    """OFFLINE_AI_EVIDENCE_VERIFY=false では verify_evidence_support を一切呼ばず、既存挙動のままになる。"""
    monkeypatch.setenv("OFFLINE_AI_EVIDENCE_VERIFY", "false")
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("無効時は呼ばれてはならない")),
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.verification_status == "skipped_disabled"
    assert result.answer_support is None
    assert "根拠検証" not in result.user_prompt


def test_verification_downgrades_sufficient_when_not_fully_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(
        monkeypatch, tmp_path, "# 規程\n日当は3,000円である。", final_status="sufficient"
    )
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: {
            "support": "unsupported",
            "reason_code": "different_subject",
            "conditions": [{"condition": "対象部署", "supported": False}],
        },
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.retrieval_status == "sufficient"
    assert result.evidence_status == "partial"
    assert result.verification_status == "verified"
    assert "根拠検証" in result.user_prompt
    assert "対象部署" in result.user_prompt


def test_verification_does_not_upgrade_partial_to_sufficient(monkeypatch, tmp_path):
    """partial かつ fully_supported でも格上げしない（非目標の固定）。"""
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。", final_status="partial")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [],
        },
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.retrieval_status == "partial"
    assert result.evidence_status == "partial"
    assert result.verification_status == "verified"
    # fully_supported なので不足理由は追記しない。
    assert "根拠検証" not in result.user_prompt


def test_verification_runs_for_partial_retrieval_status(monkeypatch, tmp_path):
    """検索時点でpartialの場合も検証・不足理由の受け渡しを実行する（Q16・F05対応）。"""
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。", final_status="partial")
    calls = []
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: calls.append(1)
        or {
            "support": "unsupported",
            "reason_code": "different_subject",
            "conditions": [],
        },
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert calls == [1]
    assert result.evidence_status == "partial"
    assert "根拠検証" in result.user_prompt


def test_verification_failed_keeps_retrieval_status_and_continues(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")

    def raise_failed(*_a, **_k):
        raise search.EvidenceVerificationError("json_parse_failed")

    monkeypatch.setattr(search, "verify_evidence_support", raise_failed)

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.verification_status == "failed"
    assert result.evidence_status == "sufficient"  # failed では格下げしない
    assert result.user_prompt  # 回答生成は継続する


def test_verification_skipped_budget_when_remaining_seconds_too_low(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("予算不足時は呼ばれてはならない")),
    )

    result = search.run_retrieval_pipeline(
        "日当はいくらか", model="m", mode="answer", remaining_seconds=lambda: 1.0
    )

    assert result.verification_status == "skipped_budget"
    assert result.evidence_status == "sufficient"


def test_verification_cancel_propagates_and_stops_before_answer_prompt(monkeypatch, tmp_path):
    """利用者キャンセル・全体タイムアウトは検証の失敗として扱わず、上位へ伝播し、
    回答生成（build_user_prompt）を開始しない。
    """
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")

    def raise_cancel(*_a, **_k):
        raise _StopCancel("cancelled")

    monkeypatch.setattr(search, "verify_evidence_support", raise_cancel)
    monkeypatch.setattr(
        search,
        "build_user_prompt",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("キャンセル伝播後は回答生成を開始してはならない")
        ),
    )

    with pytest.raises(_StopCancel):
        search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")


def test_verification_not_applicable_when_insufficient(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 無関係\n無関係な内容。", final_status="insufficient")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("insufficientでは検証してはならない")
        ),
    )

    result = search.run_retrieval_pipeline("存在しない話題は何か", model="m", mode="answer")

    assert result.retrieval_status == "insufficient"
    assert result.verification_status == "skipped_not_applicable"


def test_search_only_mode_never_verifies(monkeypatch, tmp_path):
    """検索専用（chat呼び出し無し契約）は案Aを実行しない（非目標）。"""
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("検索専用では呼ばれてはならない")),
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="unused", mode="search")

    assert result.verification_status == "skipped_not_applicable"
    assert result.user_prompt == ""


def test_verification_trace_records_status_and_latency_without_answer_body(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *a, **k: {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [],
        },
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    verification_trace = result.trace["final"]["verification"]
    assert verification_trace["verification_status"] == "verified"
    assert verification_trace["answer_support"]["support"] == "fully_supported"
    assert isinstance(verification_trace["latency_ms"], float)


# ---------------------------------------------------------------------------
# 失敗理由の分類（2026-09-18、P2でfailedの原因を区別できなかったため追加）
# ---------------------------------------------------------------------------


def test_contradiction_is_classified_separately_from_schema_errors():
    with pytest.raises(search.EvidenceVerificationError) as contradiction:
        search._validate_evidence_verification_json(
            {
                "support": "fully_supported",
                "reason_code": "answer_found",
                "conditions": [{"condition": "x", "supported": False}],
            }
        )
    assert contradiction.value.reason == search.EVIDENCE_VERIFY_FAILURE_CONTRADICTION

    with pytest.raises(search.EvidenceVerificationError) as schema:
        search._validate_evidence_verification_json({"support": "invalid"})
    assert schema.value.reason == search.EVIDENCE_VERIFY_FAILURE_SCHEMA


def test_transport_timeout_and_json_parse_are_classified(monkeypatch):
    def timeout(*_a, **_k):
        raise TimeoutError

    monkeypatch.setattr(search.urllib.request, "urlopen", timeout)
    with pytest.raises(search.EvidenceVerificationError) as timed_out:
        search.verify_evidence_support("質問", [], "m", timeout=5)
    assert timed_out.value.reason == search.EVIDENCE_VERIFY_FAILURE_TIMEOUT

    def refused(*_a, **_k):
        raise search.urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(search.urllib.request, "urlopen", refused)
    with pytest.raises(search.EvidenceVerificationError) as transport:
        search.verify_evidence_support("質問", [], "m", timeout=5)
    assert transport.value.reason == search.EVIDENCE_VERIFY_FAILURE_TRANSPORT

    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.read.return_value = b'{"message": {"content": "not json"}}'
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(search.urllib.request, "urlopen", lambda *a, **k: resp)
    with pytest.raises(search.EvidenceVerificationError) as parse:
        search.verify_evidence_support("質問", [], "m", timeout=5)
    assert parse.value.reason == search.EVIDENCE_VERIFY_FAILURE_JSON_PARSE


def test_failed_verification_records_failure_reason_in_result_and_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")

    def raise_contradiction(*_a, **_k):
        raise search.EvidenceVerificationError(
            "矛盾", reason=search.EVIDENCE_VERIFY_FAILURE_CONTRADICTION
        )

    monkeypatch.setattr(search, "verify_evidence_support", raise_contradiction)

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.verification_status == "failed"
    assert result.verification_failure_reason == "contradiction"
    assert result.trace["final"]["verification"]["failure_reason"] == "contradiction"


def test_verified_result_has_no_failure_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_evidence_verify_enabled", lambda: True)
    _prepare_pipeline(monkeypatch, tmp_path, "# 規程\n日当は3,000円である。")
    monkeypatch.setattr(
        search,
        "verify_evidence_support",
        lambda *_a, **_k: {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [],
        },
    )

    result = search.run_retrieval_pipeline("日当はいくらか", model="m", mode="answer")

    assert result.verification_status == "verified"
    assert result.verification_failure_reason == ""


def test_verification_prompt_defines_supported_as_answered_not_applicable():
    """否定表現（〜は対象外）の質問で、supported を「対象に当てはまるか」と
    取り違えて fully_supported と矛盾させない定義を持つ（P2のT12で決定的に失敗）。"""
    system = search.EVIDENCE_VERIFICATION_SYSTEM

    assert "明示的に答えているか" in system
    assert "対象外" in system



# ---------------------------------------------------------------------------
# 格下げの注記（2026-09-20、P3の利用者判断で CLI/Web に表示）
# ---------------------------------------------------------------------------


def _retrieval(**overrides):
    from types import SimpleNamespace

    base = dict(
        verification_status="verified",
        retrieval_status="sufficient",
        evidence_status="partial",
        matches=[],
        confidence=0.6,
        route="hybrid",
        index_state="ready",
        route_reason="",
        warnings=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, True),
        ({"evidence_status": "sufficient"}, False),
        ({"retrieval_status": "partial"}, False),
        ({"verification_status": "failed", "evidence_status": "sufficient"}, False),
        ({"verification_status": "skipped_disabled", "evidence_status": "sufficient"}, False),
    ],
)
def test_verification_note_only_when_downgraded(overrides, expected):
    note = search.verification_note(_retrieval(**overrides))
    assert (note == search.VERIFICATION_DOWNGRADE_NOTE) is expected
    assert bool(note) is expected


def test_cli_evidence_summary_shows_downgrade_note_under_header():
    text = search.build_evidence_summary(
        [], evidence_status="partial", confidence=0.6, note=search.VERIFICATION_DOWNGRADE_NOTE
    )
    lines = text.splitlines()
    assert lines[0].startswith("根拠一覧（状態: partial")
    assert search.VERIFICATION_DOWNGRADE_NOTE in lines[1]


def test_web_evidence_event_carries_verification_note_without_changing_status():
    import web_services

    downgraded = web_services.build_evidence_event(_retrieval())
    kept = web_services.build_evidence_event(_retrieval(evidence_status="sufficient"))

    assert downgraded["evidenceStatus"] == "partial"
    assert downgraded["verificationNote"] == search.VERIFICATION_DOWNGRADE_NOTE
    assert kept["evidenceStatus"] == "sufficient"
    assert kept["verificationNote"] == ""
