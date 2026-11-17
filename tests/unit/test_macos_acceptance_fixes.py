"""Regression coverage for measured large-model Web/deep failures; no live LLM."""

import json

import pytest
import deep_research as deep
import web_services as web
from test_deep_remediation import backend, research
from test_answer_empty_contract import (
    _FakeRetrieval,
    _prepare_worker,
    _run,
    _stream_line,
)


def _extraction(user):
    relevant = "対象である。" in user
    return {
        "relevance": "relevant" if relevant else "irrelevant",
        "subject": "対象" if relevant else "資料",
        "scope": "",
        "conditions": ["対象である。"] if relevant else [],
        "exceptions": [],
        "references": [],
    }


def _independent_backend(monkeypatch, verdict):
    backend(monkeypatch, extract=_extraction)
    original = deep._call_ollama_json
    observed = []

    def call(model, system, user, **kwargs):
        if system == deep.DEEP_RELEVANCE_SYSTEM:
            data = json.loads(user)
            assert set(data) == {"question", "source", "context"}
            assert "claims" not in data
            observed.append(data["source"])
            return {"relevance": verdict}
        return original(model, system, user, **kwargs)

    monkeypatch.setattr(deep, "_call_ollama_json", call)
    return observed


def test_irrelevant_parent_body_requires_independent_confirmation_before_exclusion(
    tmp_path, monkeypatch
):
    observed = _independent_backend(monkeypatch, "irrelevant")
    result = research(tmp_path, "# 資料\n改訂履歴の説明。\n## 本文\n対象である。\n")
    assert result.status == "completed"
    assert result.answer_state == "generated"
    assert observed and all("対象である。" not in source for source in observed)
    assert result.diagnostics["irrelevant_ranges"]
    assert not any(entry["reason"] == "model_error" for entry in result.ledger)
    assert any("対象である。" in e["excerpt"] for e in result.evidence)


@pytest.mark.parametrize(
    "verdict,reason",
    [("relevant", "relevance_disputed"), ("uncertain", "relevance_uncertain")],
)
def test_parent_conditions_are_not_discarded_when_relevance_is_disputed(
    tmp_path, monkeypatch, verdict, reason
):
    _independent_backend(monkeypatch, verdict)
    result = research(tmp_path, "# 制度\n特定の対象を除く。\n## 本文\n対象である。\n")
    assert result.status == "partial"
    assert result.stop_reason == reason
    assert any(reason in text for text in result.unconfirmed)
    assert any(
        "特定の対象を除く。" in e["excerpt"] and e["verification_status"] == "raw_unverified"
        for e in result.evidence
    )
    assert result.diagnostics["irrelevant_ranges"] == []


def test_malformed_relevance_cannot_silently_exclude_source(tmp_path, monkeypatch):
    _independent_backend(monkeypatch, "not-a-verdict")
    result = research(tmp_path, "# 資料\n改訂履歴の説明。\n## 本文\n対象である。\n")
    assert result.status == "partial"
    assert result.stop_reason == "model_error"
    assert result.diagnostics["irrelevant_ranges"] == []


def test_negative_extraction_support_has_semantic_reason_code(tmp_path, monkeypatch):
    backend(
        monkeypatch,
        verdict=lambda _user: {
            "supported": False,
            "contradictions": [],
            "missing_conditions": ["synthetic missing condition"],
            "unsupported_claims": [],
        },
    )
    result = research(tmp_path)
    assert any(row["reason"] == "extraction_unsupported" for row in result.ledger)
    assert not any(e["verification_status"] == "semantic_checked" for e in result.evidence)
    assert result.status != "completed"


@pytest.mark.parametrize("model", ["qwen3.5:122b", "gpt-oss:120b"])
def test_large_model_planning_can_take_134_seconds_inside_bounded_retrieval(monkeypatch, model):
    clock = [0.0]
    monkeypatch.setattr(web.time, "monotonic", lambda: clock[0])
    observed = {}

    def pipeline(
        query,
        *,
        model,
        reasoning=None,
        emit_status=None,
        cancel_check=None,
        mode="answer",
        remaining_seconds=None,
    ):
        observed["budget"] = remaining_seconds()
        assert observed["budget"] == 270
        clock[0] += 134
        cancel_check()
        observed["remaining"] = remaining_seconds()
        return _FakeRetrieval()

    _prepare_worker(monkeypatch, pipeline)
    monkeypatch.setattr(web, "detect_model", lambda: model)
    events = _run(
        monkeypatch,
        [_stream_line(content="answer"), _stream_line(done=True)],
        timeout=300,
    )
    assert observed["remaining"] == 136
    assert any(e["type"] == "done" for e in events)
    assert not any(e["type"] == "error" for e in events)


@pytest.mark.parametrize("model,budget", [("qwen3.5:122b", 270), ("legacy", 120)])
def test_call_and_retrieval_share_deadline_without_extending_request(monkeypatch, model, budget):
    clock = [0.0]
    monkeypatch.setattr(web.time, "monotonic", lambda: clock[0])

    def pipeline(
        query,
        *,
        model,
        reasoning=None,
        emit_status=None,
        cancel_check=None,
        remaining_seconds=None,
    ):
        assert remaining_seconds() == budget
        clock[0] += budget
        assert remaining_seconds() == 0
        cancel_check()
        pytest.fail("deadline must be enforced")

    _prepare_worker(monkeypatch, pipeline)
    monkeypatch.setattr(web, "detect_model", lambda: model)
    events = _run(monkeypatch, [], timeout=300)
    assert [e.get("code") for e in events if e["type"] == "error"] == ["timeout"]
    assert not any(e["type"] == "done" for e in events)


def test_bodyless_container_expands_children_and_preserves_scope_heading(tmp_path, monkeypatch):
    backend(monkeypatch)
    original = deep._call_ollama_json
    section_prompts = []

    def model(name, system, user, **kwargs):
        if system == deep.DEEP_RELEVANCE_SYSTEM:
            pytest.fail("container heading must be context, not independently discarded")
        if system == deep.DEEP_SECTION_SYSTEM:
            section_prompts.append(user)
        return original(name, system, user, **kwargs)

    monkeypatch.setattr(deep, "_call_ollama_json", model)
    text = "# 未成年者を除く\n\n## 条件\n対象である。\n## 例外\n特別な場合は対象外である。\n"
    result = research(tmp_path, text)
    assert section_prompts and all("未成年者を除く" in prompt for prompt in section_prompts)
    assert {e["start_line"] for e in result.evidence} == {3, 5}
    assert all(
        e["context"]
        == [
            {
                "path": "source.md",
                "start_line": 1,
                "end_line": 1,
                "text": "# 未成年者を除く",
            }
        ]
        for e in result.evidence
    )
    assert not result.diagnostics["irrelevant_ranges"]
    assert not any(e["start_line"] == 1 for e in result.evidence)


def test_nested_bodyless_headings_keep_all_ancestor_locations():
    text = "# 対象を限定\n\n## 適用範囲を限定\n\n### 条件\n条件本文。\n"
    assert deep._ranges_with_context(text, 1, 2) == [(5, 6, "条件")]
    assert deep._ranges_with_context(text, 6, 6) == [(5, 6, "条件")]
    assert deep._parent_heading_context(text, 5, "source.md") == [
        {"path": "source.md", "start_line": 1, "end_line": 1, "text": "# 対象を限定"},
        {
            "path": "source.md",
            "start_line": 3,
            "end_line": 3,
            "text": "## 適用範囲を限定",
        },
    ]


def test_leaf_heading_and_parent_body_are_still_independent_ranges():
    text = "# 制度\n本文の例外。\n## 期限7日\n"
    assert deep._ranges_with_context(text, 3, 3) == [(3, 3, "期限7日"), (1, 2, "制度")]
