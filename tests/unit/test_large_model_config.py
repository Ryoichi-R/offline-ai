import json

import pytest
import search

from model_config import (
    MAC_MODEL_DIGESTS,
    ModelConfigurationError,
    resolve_model_reasoning,
    resolve_model_selection,
    validate_model_digest,
)


def test_model_selection_precedence_and_read_only_file(tmp_path):
    model_file = tmp_path / ".model"
    model_file.write_text("gpt-oss:120b\n", encoding="utf-8")
    env = {"OFFLINE_AI_CHAT_MODEL": "qwen3.5:122b"}

    assert resolve_model_selection("gpt-oss:120b", model_file, environ=env).source == "command-line"
    assert resolve_model_selection(None, model_file, environ=env).model == "qwen3.5:122b"
    assert resolve_model_selection(None, model_file, environ={}).model == "gpt-oss:120b"
    assert model_file.read_text(encoding="utf-8") == "gpt-oss:120b\n"


def test_model_selection_rejects_blank_explicit_environment_value():
    with pytest.raises(ModelConfigurationError):
        resolve_model_selection(None, environ={"OFFLINE_AI_CHAT_MODEL": " "})


def test_gpt_oss_off_uses_supported_low_effort_without_ui_visibility():
    contract = resolve_model_reasoning("gpt-oss:120b", "off")
    assert contract == {
        "requested": "off",
        "think": "low",
        "effective": "low",
        "thinking_visible": False,
    }


def test_qwen_off_is_boolean_and_requested_effort_is_not_claimed_as_api_level():
    off = resolve_model_reasoning("qwen3.5:122b", "off")
    low = resolve_model_reasoning("qwen3.5:122b", "low")
    assert off["think"] is False
    assert off["thinking_visible"] is False
    assert low["think"] is True
    assert low["effective"] == "enabled"


def test_fixed_model_digest_is_checked_against_api_tags():
    digest = MAC_MODEL_DIGESTS["qwen3.5:122b"]
    assert (
        validate_model_digest(
            "qwen3.5:122b", [{"name": "qwen3.5:122b", "digest": f"sha256:{digest}"}]
        )
        == digest
    )
    with pytest.raises(ModelConfigurationError, match="digest"):
        validate_model_digest("qwen3.5:122b", [{"name": "qwen3.5:122b", "digest": "sha256:wrong"}])


def test_fixed_model_digest_rejects_missing_model():
    with pytest.raises(ModelConfigurationError, match="導入されていません"):
        validate_model_digest("gpt-oss:120b", [])


@pytest.mark.parametrize(
    ("model", "reasoning", "expected_think"),
    [
        ("qwen3.5:122b", "off", False),
        ("qwen3.5:122b", "low", True),
        ("gpt-oss:120b", "off", "low"),
        ("gpt-oss:120b", "high", "high"),
    ],
)
def test_search_plan_uses_the_requested_model_reasoning(
    monkeypatch, model, reasoning, expected_think
):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"message": {"content": '{"keywords":["policy"],"search_queries":["query"]}'}}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)
    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)

    search.create_search_plan("query", model, reasoning=reasoning, timeout=30)

    assert captured["think"] == expected_think
    assert captured["timeout"] == 30


def test_evidence_verification_uses_the_requested_model_reasoning(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "support": "fully_supported",
                                "reason_code": "answer_found",
                                "conditions": [],
                            }
                        )
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(search.urllib.request, "urlopen", fake_urlopen)

    search.verify_evidence_support("query", [], "qwen3.5:122b", timeout=30, reasoning="high")

    assert captured["think"] is True
    assert captured["timeout"] == 30


@pytest.mark.parametrize("model", MAC_MODEL_DIGESTS)
def test_pinned_cli_never_retries_without_requested_think(monkeypatch, model):
    import urllib.error

    calls = []

    def reject(request, timeout):
        calls.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "rejected", {}, None)

    monkeypatch.setattr(search.urllib.request, "urlopen", reject)
    with pytest.raises(urllib.error.HTTPError):
        search.stream_ollama_chat(model, "system", "question", reasoning="off")
    assert len(calls) == 1
    assert calls[0]["think"] == resolve_model_reasoning(model, "off")["think"]


@pytest.mark.parametrize("model", MAC_MODEL_DIGESTS)
def test_pinned_web_never_retries_without_requested_think(monkeypatch, model):
    import urllib.error
    import web_services
    from test_answer_empty_contract import _prepare_worker, _run

    calls = []

    def reject(request, timeout):
        calls.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "rejected", {}, None)

    _prepare_worker(monkeypatch)
    monkeypatch.setattr(web_services, "detect_model", lambda: model)
    events = _run(monkeypatch, urlopen=reject, reasoning="off")
    assert len(calls) == 1
    assert calls[0]["think"] == resolve_model_reasoning(model, "off")["think"]
    assert any(event["type"] == "error" for event in events)
    assert not any(event["type"] == "done" for event in events)
