"""Local extraction support and whole-question coverage are separate checks."""

import json

import pytest
import deep_research as deep
from test_deep_remediation import backend, research, VERDICT


@pytest.mark.parametrize("supported", [True, False])
def test_extraction_scope_does_not_receive_unanswerable_question_parts(
    tmp_path, monkeypatch, supported
):
    backend(monkeypatch)
    original = deep._call_ollama_json
    scopes = []
    gaps = []

    def model(name, system, user, **kwargs):
        if system == deep.DEEP_VERIFY_SYSTEM:
            data = json.loads(user)
            scopes.append(data)
            if data["verification_scope"] == "source_extraction":
                assert "question" not in data
                assert data["source"] and data["claims"]
                return {**VERDICT, "supported": supported}
            assert data["verification_scope"] == "final_answer"
            assert data["question"] == "対象"
        if system == deep.DEEP_GAPS_SYSTEM:
            gaps.append(json.loads(user))
        return original(name, system, user, **kwargs)

    monkeypatch.setattr(deep, "_call_ollama_json", model)
    result = research(tmp_path)
    assert any(row["verification_scope"] == "source_extraction" for row in scopes)
    if supported:
        assert result.status == "completed"
        assert any(row["verification_scope"] == "final_answer" for row in scopes)
        assert gaps and gaps[0]["question"] == "対象"
    else:
        assert result.status != "completed"
        assert any(row["reason"] == "extraction_unsupported" for row in result.ledger)
        assert all(
            row["verification_status"] != "semantic_checked" for row in result.evidence
        )
