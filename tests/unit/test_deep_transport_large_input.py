"""Actual pipe transfer regression, independent of real documents or models."""

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import deep_transport as transport


def test_worker_keeps_the_output_limit_reason_without_partial_answer():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"message":{"content":"unfinished"},"done":true,"done_reason":"length"}\n'
            )

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(transport.DeepWorkerError) as caught:
                transport.run_worker(
                    {
                        "operation": "chat",
                        "host": f"http://127.0.0.1:{server.server_port}",
                        "body": {},
                        "timeout": 3,
                    },
                    timeout=3,
                )
            assert caught.value.code == transport.MODEL_OUTPUT_INCOMPLETE_CODE
            assert "unfinished" not in str(caught.value)
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_slow_reader_receives_payload_larger_than_pipe_buffer(monkeypatch):
    original_popen = transport.subprocess.Popen
    child = (
        "import json,sys,time; time.sleep(0.2); "
        "data=json.load(sys.stdin); "
        "print(json.dumps({'ok':True,'value':len(data['padding'])}))"
    )

    def slow_reader(_args, **kwargs):
        return original_popen([sys.executable, "-I", "-c", child], **kwargs)

    monkeypatch.setattr(transport.subprocess, "Popen", slow_reader)
    assert transport.run_worker({"padding": "x" * 4_000_000}, timeout=3) == 4_000_000


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_blocked_input_is_cancelled_and_reaped(monkeypatch, mode):
    original_popen = transport.subprocess.Popen
    processes = []

    def blocked_reader(_args, **kwargs):
        process = original_popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(transport.subprocess, "Popen", blocked_reader)
    started = time.monotonic()

    class Cancelled(Exception):
        pass

    def cancel():
        if time.monotonic() - started > 0.15:
            raise Cancelled()

    with pytest.raises(Cancelled if mode == "cancel" else TimeoutError):
        transport.run_worker(
            {"padding": "x" * 4_000_000},
            timeout=3 if mode == "cancel" else 0.2,
            cancel_check=cancel if mode == "cancel" else None,
        )
    assert time.monotonic() - started < 2
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert not any(t.name == "deep-worker-io" for t in threading.enumerate())
