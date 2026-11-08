"""Gap reassessment never equates an exhausted search with a resolved question."""

import json

import pytest
import deep_research as deep
from test_deep_remediation import backend, research


def install_assessments(monkeypatch, second):
    backend(monkeypatch)
    original = deep._call_ollama_json
    calls = []

    def model(name, system, user, **kwargs):
        if system == deep.DEEP_GAPS_SYSTEM:
            calls.append(json.loads(user))
            if len(calls) == 1:
                return {"queries": ["追加条件"], "unresolved": ["対象の条件を確認"]}
            if isinstance(second, Exception):
                raise second
            return second
        return original(name, system, user, **kwargs)

    monkeypatch.setattr(deep, "_call_ollama_json", model)
    return calls


def test_resolved_gap_is_reassessed_even_without_new_evidence(tmp_path, monkeypatch):
    calls = install_assessments(monkeypatch, {"queries": [], "unresolved": []})
    result = research(tmp_path)
    assert result.status == "completed"
    assert len(calls) == 2
    assert calls[1]["new_evidence_count"] == 0
    assert calls[1]["previous_gaps"]["unresolved"] == ["対象の条件を確認"]
    assert "追加条件" in calls[1]["searched_queries"]
    assert not result.unconfirmed
    assert result.diagnostics["gap_checks"][-1]["unresolved_count"] == 0


@pytest.mark.parametrize(
    "second",
    [
        {"queries": [], "unresolved": ["対象の条件を確認"]},
        {"queries": ["追加条件"], "unresolved": []},
        {},
        RuntimeError("synthetic model error"),
    ],
)
def test_remaining_or_failed_assessment_never_completes(tmp_path, monkeypatch, second):
    install_assessments(monkeypatch, second)
    result = research(tmp_path)
    assert result.status == "partial"
    assert result.unconfirmed
    if isinstance(second, Exception) or second == {}:
        assert any("対象の条件を確認" in issue for issue in result.unconfirmed)


def test_round_limit_retains_unresolved_gaps(tmp_path, monkeypatch):
    install_assessments(monkeypatch, {"queries": [], "unresolved": []})
    result = research(tmp_path, max_rounds=1)
    assert result.status == "partial"
    assert any("unresolved" in issue for issue in result.unconfirmed)
    assert any("round_limit" in issue for issue in result.unconfirmed)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"queries": []},
        {"queries": [], "unresolved": None},
        {"queries": [5], "unresolved": []},
    ],
)
def test_missing_or_malformed_assessment_is_not_resolution(data):
    with pytest.raises(ValueError):
        deep._gaps(data)
