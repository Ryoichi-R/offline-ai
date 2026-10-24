"""P0 検証prompt単体試作（`p0_evidence_verification_prototype.py`）の単体テスト。

案Aの設計（計画136〜147行目）が求める厳格なJSON検証（矛盾する組み合わせの拒否）
を固定する。P1で製品実装する際もこの検証規則を土台にする想定のため、先に
振る舞いを固定しておく。
"""

from __future__ import annotations

import pytest

import p0_evidence_verification_prototype as p0


def test_valid_fully_supported_passes():
    result = p0.validate_verification_json(
        {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [{"condition": "一般社員", "supported": True}],
        }
    )
    assert result["support"] == "fully_supported"
    assert result["conditions"] == [{"condition": "一般社員", "supported": True}]


def test_missing_conditions_key_defaults_to_empty_list():
    result = p0.validate_verification_json({"support": "unsupported", "reason_code": "missing_detail"})
    assert result["conditions"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"support": "not_a_value", "reason_code": "answer_found"},
        {"support": "fully_supported", "reason_code": "not_a_value"},
        {"support": "fully_supported", "reason_code": "answer_found", "conditions": "not-a-list"},
        {
            "support": "fully_supported",
            "reason_code": "answer_found",
            "conditions": [{"condition": "x", "supported": "true"}],
        },
        {"support": "fully_supported", "reason_code": "answer_found", "conditions": [{"condition": "x"}]},
        "not-a-dict",
        [],
    ],
)
def test_malformed_json_is_rejected(payload):
    with pytest.raises(p0.VerificationJsonError):
        p0.validate_verification_json(payload)


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
def test_contradictory_combinations_are_rejected(payload):
    with pytest.raises(p0.VerificationJsonError):
        p0.validate_verification_json(payload)


def test_parse_json_object_strips_code_fence():
    text = '```json\n{"support": "unknown", "reason_code": "unclear"}\n```'
    parsed = p0._parse_json_object(text)
    assert parsed == {"support": "unknown", "reason_code": "unclear"}


def test_parse_json_object_returns_none_for_non_json():
    assert p0._parse_json_object("これはJSONではありません") is None


def test_build_verification_prompt_includes_query_and_matches():
    prompt = p0.build_verification_prompt(
        "質問文",
        [{"path": "a/b.md", "heading": "見出し", "start_line": 1, "end_line": 2, "snippet": "本文"}],
    )
    assert "質問文" in prompt
    assert "skill-source/a/b.md" in prompt
    assert "本文" in prompt


def test_build_verification_prompt_handles_no_matches():
    prompt = p0.build_verification_prompt("質問文", [])
    assert "採用根拠: なし" in prompt
