"""Run the Web UI without a persistent console window on Windows."""

import ctypes
import os
from pathlib import Path
import runpy
import sys


def main():
    server = Path(__file__).with_name("web_server.py")
    sys.argv[0] = str(server)
    # pythonw provides no standard streams. Supply valid sinks for existing
    # print/logging calls without persisting queries or authentication tokens.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = sink
        try:
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
