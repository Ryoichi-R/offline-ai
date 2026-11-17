"""A real loopback socket stalls before headers or between stream records."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

import deep_transport
import search
import web_services


class Cancelled(RuntimeError):
    code = "cancelled"


@pytest.mark.parametrize("wait_stage", ["headers", "body"])
@pytest.mark.parametrize("stop_kind", ["cancel", "deadline", "stall"])
def test_json_call_stops_and_reaps_its_worker_without_waiting_for_model_output(
    monkeypatch, wait_stage, stop_kind
):
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    processes = []
    errors = []
    activity = []
    token = web_services.CancellationToken(30, stall_timeout=0.3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if wait_stage == "body":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"message": {"content": "{"}}).encode() + b"\n")
                self.wfile.flush()
            entered.set()
            release.wait(10)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serve = threading.Thread(target=server.serve_forever, daemon=True)
    serve.start()
    monkeypatch.setattr(search, "OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    original = deep_transport.subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(deep_transport.subprocess, "Popen", spawn)

    def check():
        if stop_kind == "stall":
            token.check()
        if cancelled.is_set():
            raise Cancelled()

    def touch():
        activity.append(time.monotonic())
        token.touch()

    def call():
        try:
            search.chat_json_stream(
                "gpt-oss:120b",
                "Return JSON.",
                "A synthetic question.",
                timeout=0.8 if stop_kind == "deadline" else 30,
                reasoning="off",
                cancel_check=check,
                touch=touch,
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=call, daemon=True)
    try:
        worker.start()
        assert entered.wait(3), "request did not reach the test server"
        started = time.monotonic()
        if stop_kind == "cancel":
            cancelled.set()
        worker.join(2)
        assert not worker.is_alive()
        assert time.monotonic() - started < 2
        assert len(errors) == 1
        expected = (
            Cancelled
            if stop_kind == "cancel"
            else (web_services.CancelledError if stop_kind == "stall" else TimeoutError)
        )
        assert isinstance(errors[0], expected)
        if stop_kind == "stall":
            assert errors[0].code == "stall"
        if wait_stage == "headers":
            assert activity == [], "silent HTTP wait manufactured model activity"
        assert len(processes) == 1 and processes[0].poll() is not None
    finally:
        release.set()
        worker.join(3)
        server.shutdown()
        serve.join(3)
        server.server_close()


def test_json_call_checks_cancellation_before_starting_a_process(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("cancelled request started a process")

    monkeypatch.setattr(deep_transport.subprocess, "Popen", forbid)

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled):
        search.chat_json_stream("gpt-oss:120b", "JSON", "test", timeout=3, cancel_check=check)


def test_legacy_keyword_extraction_propagates_cancellation_and_preserves_reasoning(monkeypatch):
    entered = []

    def run(payload, **kwargs):
        entered.append(payload)
        raise Cancelled()

    monkeypatch.setattr(deep_transport, "run_worker", run)

    def check():
        if entered:
            raise Cancelled()

    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: False)
    with pytest.raises(Cancelled):
        search.create_search_plan(
            "共用機材の貸付条件", "gpt-oss:120b", reasoning="off", cancel_check=check
        )
    assert (
        entered[0]["body"]["think"]
        == search.resolve_model_reasoning("gpt-oss:120b", "off")["think"]
    )
    assert entered[0]["body"]["stream"] is False
    assert entered[0]["timeout"] > 0


def test_cancellable_search_plan_keeps_the_original_model_request_options(monkeypatch):
    observed = []

    def run(payload, **kwargs):
        observed.append(payload)
        return '{"keywords":["研修"],"search_queries":["研修申請"],"must_find_terms":[]}'

    monkeypatch.setattr(deep_transport, "run_worker", run)
    monkeypatch.setattr(search, "_agentic_lite_enabled", lambda: True)
    result = search.create_search_plan(
        "研修申請", "gpt-oss:120b", reasoning="off", timeout=12, cancel_check=lambda: None
    )
    body = observed[0]["body"]
    assert body["stream"] is False
    assert body["options"] == search.build_chat_options(temperature=0)
    assert body["think"] == search.resolve_model_reasoning("gpt-oss:120b", "off")["think"]
    assert observed[0]["timeout"] == 12
    assert result["keywords"] == ["研修"]
