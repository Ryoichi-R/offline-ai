"""Console-free launch preserves arguments and handles errors visibly."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).parents[2] / "_internal" / "web_launcher.py"
    spec = importlib.util.spec_from_file_location("web_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streams_and_arguments(launcher, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["web_launcher.py", "--port", "8089"])
    original = sys.stdout, sys.stderr
    calls = []

    def run(path, run_name):
        assert sys.argv[1:] == ["--port", "8089"]
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
