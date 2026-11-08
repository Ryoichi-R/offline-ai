"""Persistent local reference-folder selection; active processes keep their root."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


class SourceFolderError(ValueError):
    """An invalid reference folder or its saved configuration."""


def settings_path(project_root: Path) -> Path:
    return Path(project_root) / "_internal" / ".source-folder.json"


def load_source_root(project_root: Path) -> Path:
    """Load a path without opening documents or falling back on a missing mount."""
    path = settings_path(project_root)
    if not path.exists():
        return (Path(project_root) / "skill-source").resolve()
    try:
        if path.is_symlink() or path.stat().st_size > 16384:
            raise SourceFolderError("参考資料フォルダーの設定ファイルが不正です。")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or "\x00" in value["path"]
        ):
            raise SourceFolderError("参考資料フォルダーの設定内容が不正です。")
        return Path(value["path"]).resolve()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceFolderError("参考資料フォルダーの設定を読み込めません。") from exc


def validate_source_root(value: object) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 4096:
        raise SourceFolderError("参考資料フォルダーのパスを入力してください。")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise SourceFolderError("パソコン上のフォルダーの絶対パスを指定してください。")
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise OSError("not a directory")
        with os.scandir(path):
            pass
    except (OSError, RuntimeError) as exc:
        raise SourceFolderError(
            "フォルダーを開けません。場所・接続・アクセス権を確認してください。"
        ) from exc
    return path


def save_source_root(project_root: Path, value: object) -> Path:
    """Validate first, then atomically save only our dedicated configuration."""
    root = validate_source_root(value)
    path = settings_path(project_root)
    if path.is_symlink():
        raise SourceFolderError("参考資料フォルダーの設定ファイルが不正です。")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".source-folder-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "path": str(root)}, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return root


def describe_source_folder(project_root: Path, active_root: Path) -> dict:
    saved = load_source_root(project_root)
    active = Path(active_root).resolve()
    return {
        "activePath": str(active),
        "savedPath": str(saved),
        "restartRequired": active != saved,
        "available": active.is_dir(),
        "pickerAvailable": sys.platform == "darwin" or sys.platform == "win32",
    }


def pick_source_folder() -> str | None:
    """Open a native chooser only after an authenticated user clicks its button."""
    if sys.platform == "darwin":
        command = [
            "/usr/bin/osascript",
            "-e",
            'try\nPOSIX path of (choose folder with prompt "参考資料フォルダーを選択")'
            '\non error number -128\nreturn ""\nend try',
        ]
    elif sys.platform == "win32":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-STA",
            "-Command",
            "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = '参考資料フォルダーを選択'; "
            "try { if ($dialog.ShowDialog() -eq 'OK') { "
            "[Console]::Write($dialog.SelectedPath) } } finally { $dialog.Dispose() }",
        ]
    else:
        raise SourceFolderError("この環境ではフォルダーのパスを入力してください。")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
        )
        if result.returncode:
            raise SourceFolderError("フォルダー選択を開けません。パスを入力してください。")
        chosen = result.stdout.strip()
        return str(validate_source_root(chosen)) if chosen else None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceFolderError(
            "フォルダー選択を完了できませんでした。パスを入力できます。"
        ) from exc
