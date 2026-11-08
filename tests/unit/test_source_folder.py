"""Persistence, mount failures and native chooser behavior without real dialogs."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import source_folder as sf


def test_persisted_path_changes_next_start_but_not_active_process(tmp_path):
    project = tmp_path / "app"
    old = project / "skill-source"
    new = tmp_path / "Google Drive 同期資料"
    old.mkdir(parents=True)
    new.mkdir()
    assert sf.load_source_root(project) == old
    sf.save_source_root(project, str(new))
    assert sf.load_source_root(project) == new
    assert sf.describe_source_folder(project, old) == {
        "activePath": str(old),
        "savedPath": str(new),
        "restartRequired": True,
        "available": True,
        "pickerAvailable": sf.sys.platform in ("darwin", "win32"),
    }
    assert not sf.describe_source_folder(project, new)["restartRequired"]
    assert list(new.iterdir()) == []


@pytest.mark.parametrize(
    "value", [None, 2, True, "", "  ", "relative", "https://drive.google.com/x", "\x00", "x" * 4097]
)
def test_invalid_path_does_not_create_configuration(tmp_path, value):
    with pytest.raises(sf.SourceFolderError):
        sf.save_source_root(tmp_path, value)
    assert not sf.settings_path(tmp_path).exists()


def test_missing_or_file_path_and_permission_failure_are_rejected(tmp_path, monkeypatch):
    file = tmp_path / "a.txt"
    file.touch()
    for path in (file, tmp_path / "missing"):
        with pytest.raises(sf.SourceFolderError):
            sf.validate_source_root(str(path))
    monkeypatch.setattr(sf.os, "scandir", lambda _: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(sf.SourceFolderError):
        sf.validate_source_root(str(tmp_path))


def test_unmounted_saved_folder_is_not_replaced_by_an_unrelated_default(tmp_path):
    saved = tmp_path / "disconnected"
    config = sf.settings_path(tmp_path)
    config.parent.mkdir()
    config.write_text(json.dumps({"version": 1, "path": str(saved)}))
    assert sf.load_source_root(tmp_path) == saved
    assert not sf.describe_source_folder(tmp_path, saved)["available"]


@pytest.mark.parametrize(
    "value",
    ["{", "[]", '{"version":2,"path":"/x"}', '{"version":1,"path":"relative"}', "x" * 16385],
)
def test_invalid_configuration_fails_instead_of_selecting_other_documents(tmp_path, value):
    config = sf.settings_path(tmp_path)
    config.parent.mkdir()
    config.write_text(value)
    with pytest.raises(sf.SourceFolderError):
        sf.load_source_root(tmp_path)


def test_failed_atomic_write_retains_previous_choice(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sf.save_source_root(tmp_path, str(a))
    monkeypatch.setattr(sf.os, "replace", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        sf.save_source_root(tmp_path, str(b))
    assert sf.load_source_root(tmp_path) == a
    assert list(sf.settings_path(tmp_path).parent.glob(".source-folder-*")) == []


def test_source_symlink_is_allowed_but_configuration_symlink_is_not(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    link = tmp_path / "link"
    link.symlink_to(folder, target_is_directory=True)
    assert sf.save_source_root(tmp_path, str(link)) == folder
    config = sf.settings_path(tmp_path)
    target = config.with_name("other.json")
    config.rename(target)
    config.symlink_to(target)
    with pytest.raises(sf.SourceFolderError):
        sf.save_source_root(tmp_path, str(folder))
    with pytest.raises(sf.SourceFolderError):
        sf.load_source_root(tmp_path)


@pytest.mark.parametrize(
    "platform,output",
    [("darwin", "selected"), ("darwin", ""), ("win32", "selected"), ("win32", "")],
)
def test_native_dialog_success_and_cancel_do_not_save_settings(
    tmp_path, monkeypatch, platform, output
):
    monkeypatch.setattr(sf.sys, "platform", platform)
    monkeypatch.setattr(sf.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    captured = []

    def run(command, **options):
        captured.append((command, options))
        return SimpleNamespace(returncode=0, stdout=str(tmp_path) if output else "")

    monkeypatch.setattr(sf.subprocess, "run", run)
    assert sf.pick_source_folder() == (str(tmp_path) if output else None)
    assert not sf.settings_path(tmp_path).exists()
    command, options = captured[0]
    assert command[0] == ("/usr/bin/osascript" if platform == "darwin" else "powershell.exe")
    assert not options.get("shell") and options["timeout"] == 120


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing", "unsupported"])
def test_dialog_errors_leave_manual_path_entry_available(monkeypatch, failure):
    monkeypatch.setattr(sf.sys, "platform", "linux" if failure == "unsupported" else "darwin")

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("picker", 120)
        if failure == "missing":
            raise FileNotFoundError()
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(sf.subprocess, "run", run)
    with pytest.raises(sf.SourceFolderError):
        sf.pick_source_folder()
