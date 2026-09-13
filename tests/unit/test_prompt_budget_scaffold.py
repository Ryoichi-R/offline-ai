"""prompt 予算を num_ctx から逆算する scaffold のテスト。

対象:
- compute_evidence_char_limit が reserve 0 のとき現行挙動（逆算なし）を保つ
- reserve > 0 のとき num_ctx から生成予備と prompt shell の overhead を差し引く
- num_ctx を大きくすると PROMPT_EVIDENCE_CHAR_LIMIT で頭打ちになる
- 残量が 0 以下なら根拠を強制挿入せず PromptBudgetError で fail closed する
- 換算係数・安全余裕の環境変数が効く
- PromptBudgetError が SSE の prompt_budget エラーへ写像される

既定値は think レベル・num_ctx との組で実測してから確定する（決定ゲート D-1）。
本ファイルは実効上限の計算契約だけを固定する。

検証対象: _internal/search.py のprompt予算とscaffold
"""

import math
import os

os.environ.pop("OLLAMA_HOST", None)

import pytest  # noqa: E402

import search  # noqa: E402
import web_services  # noqa: E402


def _limit(shell_chars, **kwargs):
    return search.compute_evidence_char_limit(shell_chars, **kwargs)


# --- 既定（逆算無効）--------------------------------------------------------


def test_reserve_zero_keeps_current_behaviour():
    """既定 reserve=0 では num_ctx を参照せず char_limit をそのまま返す。"""
    assert _limit(100_000, reserve_tokens=0, num_ctx=8192) == search.PROMPT_EVIDENCE_CHAR_LIMIT


def test_default_parameters_do_not_change_effective_limit():
    assert search.GENERATION_RESERVE_TOKENS == 0
    assert _limit(6000) == search.PROMPT_EVIDENCE_CHAR_LIMIT


# --- 逆算 -------------------------------------------------------------------


def test_reserve_reduces_limit_below_char_limit():
    limit = _limit(
        1400,  # overhead = ceil(1400 / 1.4) + 0 = 1000 tokens
        num_ctx=8192,
        reserve_tokens=4096,
        chars_per_token=1.4,
        safety_tokens=0,
        char_limit=9000,
    )
    # evidence budget = 8192 - 4096 - overhead(約 1000) tokens
    # ceil/floor の丸めがあるため厳密一致ではなく近傍で検証する。
    expected = math.floor((8192 - 4096 - 1000) * 1.4)
    assert abs(limit - expected) <= 2
    assert limit < 9000


def test_reserve_increase_reduces_limit_monotonically():
    common = dict(num_ctx=8192, chars_per_token=1.4, safety_tokens=0, char_limit=9000)
    assert _limit(1400, reserve_tokens=5120, **common) < _limit(1400, reserve_tokens=4096, **common)


def test_larger_num_ctx_is_capped_by_char_limit():
    limit = _limit(
        1400,
        num_ctx=32768,
        reserve_tokens=4096,
        chars_per_token=1.4,
        safety_tokens=0,
        char_limit=9000,
    )
    assert limit == 9000


def test_safety_tokens_are_subtracted():
    common = dict(num_ctx=8192, reserve_tokens=4096, chars_per_token=1.4, char_limit=9000)
    without = _limit(1400, safety_tokens=0, **common)
    with_safety = _limit(1400, safety_tokens=256, **common)
    assert with_safety < without
    # 差は safety_tokens 分の token 予算（floor 誤差 1 文字以内）
    assert abs((without - with_safety) - 256 * 1.4) <= 1


def test_longer_prompt_shell_reduces_evidence_budget():
    common = dict(
        num_ctx=8192,
        reserve_tokens=4096,
        chars_per_token=1.4,
        safety_tokens=0,
        char_limit=9000,
    )
    assert _limit(2800, **common) < _limit(1400, **common)


def test_chars_per_token_is_applied():
    conservative = _limit(
        1400,
        num_ctx=8192,
        reserve_tokens=4096,
        chars_per_token=1.0,
        safety_tokens=0,
        char_limit=9000,
    )
    loose = _limit(
        1400,
        num_ctx=8192,
        reserve_tokens=4096,
        chars_per_token=2.0,
        safety_tokens=0,
        char_limit=9000,
    )
    assert conservative != loose


# --- fail closed ------------------------------------------------------------


def test_fails_closed_when_reserve_exceeds_context():
    with pytest.raises(search.PromptBudgetError):
        _limit(
            1400,
            num_ctx=4096,
            reserve_tokens=4096,
            chars_per_token=1.4,
            safety_tokens=0,
            char_limit=9000,
        )


def test_fails_closed_when_shell_overhead_exhausts_context():
    with pytest.raises(search.PromptBudgetError):
        _limit(
            100_000,
            num_ctx=8192,
            reserve_tokens=2048,
            chars_per_token=1.4,
            safety_tokens=0,
            char_limit=9000,
        )


def test_error_message_names_actionable_settings():
    with pytest.raises(search.PromptBudgetError) as excinfo:
        _limit(
            1400,
            num_ctx=4096,
            reserve_tokens=4096,
            chars_per_token=1.4,
            safety_tokens=0,
            char_limit=9000,
        )
    message = str(excinfo.value)
    assert "OLLAMA_NUM_CTX" in message
    assert "OFFLINE_AI_GENERATION_RESERVE_TOKENS" in message


# --- _fit_prompt_budget との結合 --------------------------------------------


def test_fit_prompt_budget_respects_computed_limit():
    matches = [{"snippet": "あ" * 1000}, {"snippet": "い" * 1000}]
    fitted = search._fit_prompt_budget(matches, 1500)
    assert sum(len(m["snippet"]) for m in fitted) == 1500


def test_fit_prompt_budget_does_not_force_insert_when_limit_is_zero():
    """最低 1 件の強制挿入をしない（reserve を侵食させない）。"""
    assert search._fit_prompt_budget([{"snippet": "あ" * 100}], 0) == []


# --- SSE への写像 -----------------------------------------------------------


def test_prompt_budget_error_maps_to_dedicated_sse_code(monkeypatch):
    def pipeline(query, *, model, reasoning=None, emit_status=None, cancel_check=None):
        raise search.PromptBudgetError("prompt 予算が不足しています")

    monkeypatch.setattr(web_services, "_search_available", True)
    monkeypatch.setattr(web_services, "detect_model", lambda: "m")
    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)

    token = web_services.CancellationToken(timeout=5.0, stall_timeout=5.0)
    entry = web_services.JobEntry(token)
    web_services.run_search("q", "medium", web_services._BroadcastQueue(entry), token, "rid")

    replay = entry.add_subscriber()
    events = []
    while not replay.empty():
        events.append(replay.get_nowait())
    errors = [e for e in events if e["type"] == "error"]
    assert errors and errors[0]["code"] == "prompt_budget"


# --- snippet 上限の定数化 ---------------------------------------------------


def test_snippet_char_limit_is_configurable_constant():
    assert search.SNIPPET_CHAR_LIMIT == 1200
