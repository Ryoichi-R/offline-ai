"""Console-free launch preserves arguments and handles errors visibly."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).parents[2] / "_internal" / "web_launcher.py"
    spec = importlib.util.spec_from_file_location("web_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ensure_ollama", lambda: None)
    return module


def test_streams_and_arguments(launcher, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["web_launcher.py", "--port", "8089"])
    original = sys.stdout, sys.stderr
    calls = []

    def run(path, run_name):
        assert sys.argv[1:] == ["--port", "8089", "--exit-when-page-closed"]
        assert sys.stdout is sys.stderr
        print("test output")
        calls.append((Path(path).name, run_name))

    monkeypatch.setattr(launcher.runpy, "run_path", run)
    launcher.main()
    assert calls == [("web_server.py", "__main__")]
    assert (sys.stdout, sys.stderr) == original


@pytest.mark.parametrize(
    "error, expected", [(SystemExit(0), 0), (SystemExit(2), 1), (RuntimeError(), 1)]
)
def test_failure_notification(launcher, monkeypatch, error, expected):
    monkeypatch.setattr(sys, "argv", ["web_launcher.py"])
    notices = []

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(launcher.runpy, "run_path", run)
    monkeypatch.setattr(launcher, "show_error", lambda: notices.append(True))
    launcher.main()
    assert len(notices) == expected


@pytest.mark.parametrize(
    "running, missing", [(True, False), (False, False), (False, True)]
)
def test_detached_ollama_start(monkeypatch, running, missing):
    from contextlib import nullcontext

    path = Path(__file__).parents[2] / "_internal" / "web_launcher.py"
    spec = importlib.util.spec_from_file_location("launcher_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def probe(*args, **kwargs):
        if running:
            return nullcontext()
        raise OSError("unavailable")

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        if missing:
            raise FileNotFoundError()

    monkeypatch.setattr(module.urllib.request, "urlopen", probe)
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        module.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
    )
    module.ensure_ollama()
    assert len(calls) == (0 if running else 1)
    if calls:
        assert calls[0][0] == (["ollama", "serve"],)
        assert calls[0][1]["creationflags"] == 0x08000000
        for stream in ("stdin", "stdout", "stderr"):
            assert calls[0][1][stream] == module.subprocess.DEVNULL
