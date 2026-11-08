"""Synthetic smoke isolation, wire contracts, deadlines and numeric-only receipts."""

import json
import subprocess
import sys

import pytest
import macos_product_smoke as smoke
from model_config import MAC_MODEL_DIGESTS, resolve_model_reasoning

MODEL = "qwen3.5:122b"
ANSWER = "上限は7日、返却先は研修支援窓口です。[L1] skill-source/handbook.md"


def test_preparation_is_hash_bound_without_holdout_or_model_calls(monkeypatch):
    monkeypatch.setattr(
        smoke.compare.search, "get_model_identity", lambda *_: pytest.fail("live call")
    )
    receipt = smoke.build_plan("test-smoke", ["cli", "web"], list(MAC_MODEL_DIGESTS))
    assert len(receipt["cases"]) == 8
    assert receipt["maxGenerationSeconds"] == 2400
    assert receipt["status"] == "PREPARED_NOT_RUN"
    assert not receipt["holdoutExecuted"]
    assert not receipt["answerTextStored"]
    assert smoke.CORPUS in receipt["product"]["files"]
    assert "tests/eval/macos_product_smoke.py" in receipt["product"]["files"]
    assert not any(
        "macos-large-model-holdout/" in name or name.startswith("skill-source/")
        for name in receipt["product"]["files"]
    )
    assert all(len(digest) == 64 for digest in receipt["product"]["files"].values())


@pytest.mark.parametrize(
    "interfaces,models",
    [
        ([], [MODEL]),
        (["cli", "cli"], [MODEL]),
        (["bad"], [MODEL]),
        (["cli"], []),
        (["cli"], ["unknown"]),
        (["cli"], [MODEL, MODEL]),
    ],
)
def test_invalid_scope_rejected(interfaces, models):
    with pytest.raises(ValueError):
        smoke.build_plan("test-smoke", interfaces, models)


def test_snapshot_reexecutes_without_original_tree(tmp_path):
    receipt = smoke.build_plan("test-copy", ["cli"], [MODEL])
    smoke.compare.copy_snapshot(receipt, tmp_path)
    assert not (tmp_path / "skill-source").exists()
    assert not (tmp_path / "tests/eval/macos-large-model-holdout").exists()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(tmp_path / "tests/eval/macos_product_smoke.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    copied = tmp_path / smoke.CORPUS
    copied.write_text("changed")
    with pytest.raises(ValueError):
        smoke.compare.verify_snapshot(receipt, tmp_path)


def test_numeric_observer_does_not_store_answer():
    result = smoke.summarize_answer(ANSWER)
    assert all(result["requiredFacts"].values())
    assert result["citationCount"] == 1
    assert ANSWER not in json.dumps(result, ensure_ascii=False)
    assert len(result["answerSha256"]) == 64
    assert smoke.summarize_answer("[E1] [1] [L2]")["citationCount"] == 3


def test_resource_counters_include_descendants_but_not_other_processes():
    table = "100 1 10\n101 100 20\n102 101 30\n200 1 999\n"
    assert smoke.process_rss_bytes(table, 100) == 60 * 1024
    assert smoke.process_rss_bytes(table, 999) is None
    assert (
        smoke.parse_swap_bytes("total = 4096.00M used = 2.50M free = 4093.50M")
        == 2621440
    )
    assert smoke.parse_swap_bytes("unavailable") is None


@pytest.mark.parametrize("model", MAC_MODEL_DIGESTS)
def test_acceptance_checks_metadata_and_success_not_only_content(model):
    case = {"model": model, "mode": "deep", "interface": "web"}
    result = {
        **smoke.summarize_answer(ANSWER),
        "answerState": "generated",
        "deepStatus": "completed",
        "terminalCount": 1,
        "errorCount": 0,
        "modelMatches": True,
        "digestMatches": True,
        "jobsReleased": True,
        "apiThinkValue": resolve_model_reasoning(model, "off")["think"],
    }
    assert smoke.case_passed(case, result)
    for key, value in [
        ("answerState", "empty"),
        ("deepStatus", "partial"),
        ("terminalCount", 2),
        ("errorCount", 1),
        ("modelMatches", False),
        ("digestMatches", False),
        ("jobsReleased", False),
        ("apiThinkValue", "invalid"),
        ("failureType", "CaseDeadline"),
        ("citationCount", 0),
        ("requiredFacts", {}),
    ]:
        assert not smoke.case_passed(case, {**result, key: value}), key


def test_web_handler_real_bootstrap_cookie_and_sse(monkeypatch):
    import web_server

    captured = {}

    def fake_run(
        query,
        reasoning,
        output,
        token,
        request_id,
        *,
        mode,
        model_name,
        model_digest,
        **kwargs,
    ):
        captured.update(
            query=query,
            reasoning=reasoning,
            mode=mode,
            model_name=model_name,
            model_digest=model_digest,
            **kwargs,
        )
        for event in [
            {
                "type": "result_meta",
                "chatModel": MODEL,
                "modelDigest": MAC_MODEL_DIGESTS[MODEL],
                "apiThinkValue": False,
            },
            {"type": "evidence", "answerState": "generated"},
            {"type": "chunk", "text": ANSWER},
            {"type": "done"},
        ]:
            output.put(event)

    monkeypatch.setattr(web_server, "run_search", fake_run)
    case = {"model": MODEL, "mode": "answer", "interface": "web"}
    result = smoke.web_case(case)
    assert smoke.case_passed(case, result), result
    assert captured["query"] == smoke.QUERY
    assert captured["reasoning"] == "off"
    assert captured["model_name"] == MODEL
    assert captured["model_digest"] == MAC_MODEL_DIGESTS[MODEL]
    assert captured["mode"] == "answer"
    assert result["httpStatus"] == 200
    assert result["firstContentSeconds"] is not None


def test_cli_observer_calls_entrypoint_and_restores_wrappers(monkeypatch):
    import search

    previous = search.stream_ollama_chat
    monkeypatch.setattr(search, "stream_ollama_chat", lambda *_a, **_k: ANSWER)
    wrapped = search.stream_ollama_chat

    def main():
        assert "--timeout-seconds" in sys.argv
        assert search.stream_ollama_chat() == ANSWER

    monkeypatch.setattr(search, "main", main)
    result = smoke.cli_case({"model": MODEL, "mode": "answer"})
    assert result["requiredFacts"] == {"duration": True, "returnOffice": True}
    assert search.stream_ollama_chat is wrapped
    assert previous is not wrapped


def test_case_deadline_stops_created_group_and_reports_failure(monkeypatch, tmp_path):
    class Process:
        pid = 12345

        def communicate(self, timeout):
            assert timeout == smoke.CASE_SECONDS + 10
            raise subprocess.TimeoutExpired("smoke", timeout)

    class Thread:
        def start(self):
            pass

    class Sampler:
        thread = Thread()

        def __init__(self, pid, ollama_pid):
            assert (pid, ollama_pid) == (12345, 54321)

        def finish(self):
            return {"sampleCount": 0}

    stopped = []
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *_a, **_kw: Process())
    monkeypatch.setattr(smoke, "ResourceSampler", Sampler)
    monkeypatch.setattr(
        smoke, "stop_process_group", lambda process: stopped.append(process.pid)
    )
    case = {"model": MODEL, "mode": "answer", "interface": "cli"}
    result = smoke.run_case(case, {}, tmp_path, 54321)
    assert result["failureType"] == "CaseDeadline"
    assert not result["passed"]
    assert stopped and set(stopped) == {12345}


def test_child_failure_receipt_redacts_exception_text(monkeypatch, tmp_path):
    corpus = tmp_path / "skill-source/handbook.md"
    corpus.parent.mkdir()
    corpus.write_text("synthetic")
    receipt = {
        "product": {"files": {}},
        "corpusSha256": smoke.compare.sha256_file(corpus),
    }
    case = {"model": MODEL, "mode": "answer", "interface": "cli"}
    payload = tmp_path / "case.json"
    payload.write_text(json.dumps({"case": case, "receipt": receipt}))
    monkeypatch.setattr(smoke, "ROOT", tmp_path)

    def fail(_case):
        raise RuntimeError("raw response must not be recorded")

    monkeypatch.setattr(smoke, "cli_case", fail)
    result = smoke.child_case(payload)
    assert result["failureType"] == "RuntimeError"
    assert "raw response" not in json.dumps(result)
    assert not result["passed"]


def test_selected_cases_preserve_order_and_budget():
    result = smoke.build_plan(
        "targeted-diagnosis", ["cli", "web"], [MODEL], ["cli:deep", "web:answer"]
    )
    assert [(c["interface"], c["mode"]) for c in result["cases"]] == [
        ("cli", "deep"),
        ("web", "answer"),
    ]
    assert result["maxGenerationSeconds"] == 600
    for bad in ([], ["web:unknown"], ["cli:deep", "cli:deep"]):
        with pytest.raises(ValueError):
            smoke.build_plan("bad-selection", ["cli", "web"], [MODEL], bad)


def test_diagnostic_structure_never_retains_raw_keys_or_text():
    import macos_smoke_diagnostics as diag

    result = diag.structure(
        {
            "subject": "private input",
            "conditions": ["private value"],
            "supported": False,
            "relevance": "irrelevant",
            "private key": "private content",
        }
    )
    encoded = json.dumps(result)
    assert "private" not in encoded
    assert result["fields"]["subject"] == {"type": "str", "length": 13}
    assert result["fields"]["conditions"] == {"type": "list", "length": 1}
    assert result["fields"]["scope"] == {"type": "missing"}
    assert result["fields"]["supported"]["value"] is False
    assert diag.safe_code("private code") == "other"
    assert "private" not in json.dumps(diag.safe_failure(ValueError("private error")))


def test_diagnostic_wrappers_preserve_signature_results_errors_and_restore(monkeypatch):
    import inspect
    import deep_research
    import search
    import web_services
    import macos_smoke_diagnostics as diag

    def pipeline(query, *, model, mode="answer"):
        assert (query, model, mode) == ("private", "fake", "answer")
        raise ValueError("private message")

    def stage(model, system, user, budget, validator, *, reserve):
        return validator(
            {
                "supported": True,
                "contradictions": [],
                "missing_conditions": [],
                "unsupported_claims": [],
            }
        )

    monkeypatch.setattr(web_services, "run_retrieval_pipeline", pipeline)
    monkeypatch.setattr(deep_research, "_json_stage", stage)
    old_search = search.create_search_plan
    records = []
    with diag.observe(records):
        assert inspect.signature(
            web_services.run_retrieval_pipeline
        ) == inspect.signature(pipeline)
        assert (
            deep_research._json_stage(
                "m", "s", "u", None, deep_research._verification, reserve=1
            )
            is True
        )
        with pytest.raises(ValueError, match="private message"):
            web_services.run_retrieval_pipeline("private", model="fake")
    assert web_services.run_retrieval_pipeline is pipeline
    assert deep_research._json_stage is stage
    assert search.create_search_plan is old_search
    assert "private" not in json.dumps(records)
    assert records[0]["attempts"][0]["verdict"] is True
    assert records[1]["exceptionType"] == "ValueError"


def test_web_error_codes_are_observed_without_messages(monkeypatch):
    import web_server

    def fail(query, reasoning, output, token, request_id, **kwargs):
        output.put({"type": "error", "code": "timeout", "message": "private detail"})

    monkeypatch.setattr(web_server, "run_search", fail)
    result = smoke.web_case({"model": MODEL, "mode": "answer", "interface": "web"})
    assert result["errorCodes"] == ["timeout"]
    assert result["errorCount"] == 1
    assert "private" not in json.dumps(result)
