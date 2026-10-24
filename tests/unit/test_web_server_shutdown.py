import inspect
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

INTERNAL_DIR = Path(__file__).resolve().parents[2] / "_internal"

import web_server


def test_connection_semaphore_released_by_request_thread():
    source = inspect.getsource(web_server.LimitedThreadingServer.process_request_thread)

    assert "finally:" in source
    assert "self._conn_sem.release()" in source


def test_shutdown_jobs_once_is_idempotent():
    source = inspect.getsource(web_server.LimitedThreadingServer.shutdown_jobs_once)

    assert "_shutdown_once.is_set()" in source
    assert "self.job_table.shutdown()" in source
    assert "self.cancel_all()" in source


def test_browser_bootstrap_uses_fragment_instead_of_query_token():
    source = inspect.getsource(web_server.main)

    assert "/bootstrap#token=" in source
    assert "/?token=" not in source


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console control event E2E")
def test_web_server_exits_after_ctrl_break_event():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    bootstrap = f"""
import sys
sys.path.insert(0, {str(INTERNAL_DIR)!r})
import web_server
web_server.webbrowser.open = lambda *_args, **_kwargs: False
sys.argv = ["web_server.py", "--port", "{port}"]
web_server.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        env=os.environ.copy(),
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                pytest.fail(
                    f"server exited before Ctrl+Break: {process.returncode}\n{stdout}\n{stderr}"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server did not start within 10 seconds")

        process.send_signal(signal.CTRL_BREAK_EVENT)
        return_code = process.wait(timeout=5)
        assert return_code == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
