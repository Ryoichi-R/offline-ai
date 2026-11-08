"""Unaligned claims must not silently pass checks on limited evidence."""
import deep_research as deep
from test_deep_remediation import backend, research


def evidence():
    return [{"evidence_id": "E1", "path": "test.md", "excerpt": "| 条件 | 記録を五年間保存する。 |", "context": [{"path": "test.md", "text": "## 試験設備専用"}]}]


def test_unmatched_claim_is_reported_and_rewrite_requests_source():
    candidate, issues = deep._check_final_candidate("検査を行う。[E1]", evidence(), {"attempts": []})
    assert issues["unmatched_limited_lines"] == [1]
    assert "原文" in deep._final_retry_prompt("prompt", candidate, issues)


def test_exact_claim_is_left_to_scope_checker():
    claim = "記録を五年間保存する。[E1]"
    assert deep._unmatched_limited_lines(claim, evidence()) == []
    assert deep._limitation_issues(claim, evidence())


def test_unrelated_citation_and_heading_are_not_unmatched_limited_claims():
    assert deep._unmatched_limited_lines("## 見出し[E1]\n別資料の説明。[E2]", evidence()) == []


def test_repeated_unmatched_answer_cannot_pass_model_only_verification(tmp_path, monkeypatch):
    backend(monkeypatch, answer="検査を行う。[E1]")
    result = research(tmp_path, "## 試験設備専用\n| 条件 | 記録を五年間保存する。 |")
    assert result.answer_state == "verification_failed"
    assert result.diagnostics["final_answer"]["result"] == "rejected_unmatched_limits"
    assert result.diagnostics["final_answer"]["unmatched_limited_lines"] == 1


def test_rewrite_to_scoped_source_is_accepted(tmp_path, monkeypatch):
    backend(monkeypatch)
    calls = []
    def answer(*args, **kwargs):
        calls.append(args)
        return "検査を行う。[E1]" if len(calls) == 1 else "試験設備専用：記録を五年間保存する。[E1]"
    monkeypatch.setattr(deep, "_call_ollama_text", answer)
    result = research(tmp_path, "## 試験設備専用\n| 条件 | 記録を五年間保存する。 |")
    assert len(calls) == 2
    assert result.answer_state == "generated"
    assert result.diagnostics["final_answer"]["unmatched_limited_lines"] == 0


def test_unmatched_line_removed_without_losing_verified_partial_answer(tmp_path, monkeypatch):
    backend(monkeypatch, answer="試験設備専用：記録を五年間保存する。[E1]\n検査を行う。[E1]")
    result = research(tmp_path, "## 試験設備専用\n| 条件 | 記録を五年間保存する。 |")
    assert result.answer_state == "generated"
    assert result.status == "partial"
    assert "検査を行う" not in result.answer
    assert "記録を五年間保存する" in result.answer
    assert result.diagnostics["final_answer"]["unmatched_limited_lines_removed"] == 1
    assert any(entry["reason"] == "model_error" for entry in result.ledger)
