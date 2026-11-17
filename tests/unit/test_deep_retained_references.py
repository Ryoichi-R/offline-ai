"""Only retained evidence may schedule reference follow-up searches."""

import pytest

import deep_research as deep
from test_deep_remediation import backend, research


@pytest.mark.parametrize("retained", [False, True])
def test_reference_followup_requires_retained_evidence(tmp_path, monkeypatch, retained):
    reference = "synthetic-reference-document"
    calls = backend(
        monkeypatch,
        extract=lambda user: {
            "subject": "subject",
            "scope": "scope",
            "conditions": [],
            "exceptions": [],
            "references": [reference],
        },
    )
    monkeypatch.setattr(deep, "DEEP_MAX_EVIDENCE_CHARS", 12000 if retained else 1)
    result = research(tmp_path)
    assert (reference in calls) is retained
    assert any(entry["reason"] == "unresolved_reference" for entry in result.ledger) is retained
    assert bool(result.evidence) is retained
    if not retained:
        assert result.status == "failed"
        assert result.stop_reason == "evidence_budget"
        assert any(entry["reason"] == "evidence_budget" for entry in result.ledger)
