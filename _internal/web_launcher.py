"""Run the Web UI without a persistent console window on Windows."""

import ctypes
import os
from pathlib import Path
import runpy
import sys
import subprocess
import urllib.request


def ensure_ollama():
    """Start Ollama without a console; browser health reports availability."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2):
            return
    except OSError:
        pass
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
    except OSError:
        # Open the browser even if unavailable; keyword search can still work.
        pass


def main():
    server = Path(__file__).with_name("web_server.py")
    sys.argv[0] = str(server)
    # 非表示起動ではCtrl+Cで止められないため、開いているページが無くなったら停止する。
    if "--exit-when-page-closed" not in sys.argv[1:]:
        sys.argv.append("--exit-when-page-closed")
    # pythonw provides no standard streams. Supply valid sinks for existing
    # print/logging calls without persisting queries or authentication tokens.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = sink
        try:
            ensure_ollama()
            runpy.run_path(str(server), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                show_error()
        except Exception:
            show_error()
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def show_error():
    ctypes.windll.user32.MessageBoxW(
        None,
        "Web画面を起動できませんでした。\n"
        "診断するにはコマンドプロンプトで次を実行してください。\n\n"
        "set OFFLINE_AI_WEB_CONSOLE=1\nweb.bat",
        "offline-ai",
        0x10,
    )


if __name__ == "__main__":
    main()
